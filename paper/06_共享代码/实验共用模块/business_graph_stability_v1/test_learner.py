"""LR isolation and deterministic instrumentation; zero simulator usage."""
import ast
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from business_graph_summary_v1 import learner as original
from business_graph_summary_v1.test_learner import synthetic_batch, assert_tree_equal
from business_graph_summary_v1.models import digest
from . import learner
from .admission import frozen_actor


@pytest.fixture(autouse=True)
def single_thread():
    count = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(count)


def make_config(module):
    config = module.default_config(source_kind='synthetic', seed=373,
                                    execution_profile='synthetic_test_v1')
    config['ppo']['actor_lr'] = 1e-4
    config['ppo']['minibatch_size'] = 2
    return config


def method_ast(module, name, strip_telemetry=False):
    tree = ast.parse(Path(module.__file__).read_text())
    cls = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='JointCostPPO')
    node = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
    if strip_telemetry:
        class Strip(ast.NodeTransformer):
            def visit_Assign(self, node):
                if any(isinstance(t,ast.Name) and t.id in ('epoch_diagnostics','epoch_loss_start') for t in node.targets):
                    return None
                return self.generic_visit(node)
            def visit_Expr(self, node):
                if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and isinstance(node.value.func.value,ast.Name) and node.value.func.value.id=='epoch_diagnostics':
                    return None
                return self.generic_visit(node)
            def visit_Call(self, node):
                node.keywords = [k for k in node.keywords if k.arg!='epoch_diagnostics']
                return self.generic_visit(node)
        node = Strip().visit(node)
    return ast.dump(node, include_attributes=False)


@pytest.mark.parametrize('name', ['_actor_grad','_critic_grad','_freeze_targets','_policy_rows','_values','_metrics','update'])
def test_original_numerical_path_unchanged_except_observer(name):
    assert method_ast(original,name) == method_ast(learner,name,strip_telemetry=True)


def test_real_configuration_changes_exactly_actor_lr_and_namespace():
    old, new = original.default_config(), learner.default_config()
    assert new.pop('schema') != old.pop('schema')
    assert old['ppo']['actor_lr']==3e-4 and new['ppo']['actor_lr']==1e-4
    new['ppo']['actor_lr']=3e-4
    assert new==old
    learner.validate_config(learner.default_config())
    for key in ('actor_lr','critic_lr','epochs','entropy_coef'):
        cfg=learner.default_config();cfg['ppo'][key]*=2
        with pytest.raises(ValueError,match='frozen PPO'):
            learner.validate_config(cfg)
    with pytest.raises(ValueError,match='schema'):
        learner.validate_config(original.default_config())


def test_epoch_observer_bitwise_matches_original_step_and_rng():
    old=original.JointCostPPO(make_config(original))
    new=learner.JointCostPPO(make_config(learner))
    assert digest(old.actor)==digest(new.actor) and digest(old.critic)==digest(new.critic)
    batch=synthetic_batch(new,recording='synthetic-stability-observer')
    before_rng=torch.get_rng_state().clone()
    old_result=old.update(deepcopy(batch),batch_id='observer')
    old_rng=torch.get_rng_state().clone()
    torch.set_rng_state(before_rng)
    new_result=new.update(batch,batch_id='observer')
    assert torch.equal(old_rng,torch.get_rng_state())
    for name in ('actor','critic','actor_optimizer','critic_optimizer'):
        assert_tree_equal(getattr(old,name).state_dict(),getattr(new,name).state_dict())
    assert_tree_equal(old.generator.get_state(),new.generator.get_state())
    assert_tree_equal(old.sampling_generator.get_state(),new.sampling_generator.get_state())
    assert_tree_equal(old.last_targets,new.last_targets)
    assert old_result['metrics']==new_result['metrics']
    assert old_result['mean']==new_result['mean']
    assert len(new_result['epoch_diagnostics'])==4
    assert new_result['epoch_diagnostics'][-1]['replay']==new_result['metrics']
    for row in new_result['epoch_diagnostics']:
        for role in ('actor','critic'):
            d=row['preclip_gradient_norms'][role]
            assert d['count']==3 and 0<=d['clip_hits']<=d['count']
            assert d['min']<=d['p50']<=d['p95']<=d['max']
        assert row['optimizer_lrs']==dict(actor=[1e-4],critic=[1e-3])


def test_stability_actor_checkpoint_and_next_update_exact_restore(tmp_path):
    model=learner.JointCostPPO(make_config(learner))
    batch=synthetic_batch(model,recording='synthetic-stability-restore1')
    model.update(batch,batch_id='restore1')
    actor_ref=model.save_actor(tmp_path/'actor.pt')
    actor,fingerprint=frozen_actor(actor_ref['path'],allow_small_synthetic=True)
    assert fingerprint==model.policy_version
    model.bind_actor_snapshot(actor_ref['path'])
    ref=model.save_checkpoint(tmp_path/'checkpoint.pt')
    restored=learner.JointCostPPO.from_checkpoint(ref['path'],expected_config=model.config,expected_sha256=ref['sha256'])
    batch2=synthetic_batch(model,recording='synthetic-stability-restore2')
    a=model.update(batch2,batch_id='restore2')
    b=restored.update(deepcopy(batch2),batch_id='restore2')
    assert_tree_equal(model.actor.state_dict(),restored.actor.state_dict())
    assert_tree_equal(model.critic_optimizer.state_dict(),restored.critic_optimizer.state_dict())
    assert a['epoch_diagnostics']==b['epoch_diagnostics']
    assert actor._artifact_config['ppo']['actor_lr']==1e-4


def test_invalid_material_cost_rejected_before_step():
    from dataclasses import replace
    model=learner.JointCostPPO(make_config(learner))
    batch=synthetic_batch(model)
    old=batch[0]['targets']
    batch[0]['targets']=replace(old,step_costs=(old.step_costs[0]+1,)+old.step_costs[1:])
    before=digest(model.actor)
    with pytest.raises(ValueError,match='targets'):
        model.update(batch,batch_id='invalid')
    assert digest(model.actor)==before and not model.actor_optimizer.state


def test_import_original_common_checkpoint_preserves_tensors_rng_and_empty_adam(tmp_path):
    cfg=make_config(original);cfg['ppo']['actor_lr']=3e-4
    old=original.JointCostPPO(cfg)
    ref=old.save_checkpoint(tmp_path/'common_u0.pt')
    imported=learner.JointCostPPO.from_common_initial_checkpoint(ref['path'],
        expected_sha256=ref['sha256'],config=make_config(learner),restore_global_rng=False)
    assert_tree_equal(old.actor.state_dict(),imported.actor.state_dict())
    assert_tree_equal(old.critic.state_dict(),imported.critic.state_dict())
    assert_tree_equal(old.generator.get_state(),imported.generator.get_state())
    assert_tree_equal(old.sampling_generator.get_state(),imported.sampling_generator.get_state())
    assert imported.update_index==0 and not imported.actor_optimizer.state and not imported.critic_optimizer.state
    assert imported.actor_optimizer.param_groups[0]['lr']==1e-4
    assert imported.critic_optimizer.param_groups[0]['lr']==old.critic_optimizer.param_groups[0]['lr']
    old.update(synthetic_batch(old),batch_id='trained')
    trained=old.save_checkpoint(tmp_path/'trained.pt')
    with pytest.raises(ValueError,match='untouched u0'):
        learner.JointCostPPO.from_common_initial_checkpoint(trained['path'],
            expected_sha256=trained['sha256'],config=make_config(learner),restore_global_rng=False)
