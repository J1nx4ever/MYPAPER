"""BG009 integration invariants on synthetic complete shifts; zero simulation use.

The numerical learner is compared with the frozen TQ008 implementation, while
model/graph identities are deliberately different. Synthetic road references
are adapted locally to the production map; historical fixtures are untouched.
"""
import ast
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from single_critic_task_query_v1 import learner as previous_learner
from single_critic_task_query_v1 import targets as previous_targets
from single_critic_task_query_v1.test_evaluation import complete_synthetic_episode
from . import learner, targets
from .admission import frozen_actor
from .c_features import feature_contract
from .evaluation import evaluate_episodes
from .graph import ROAD_ENDPOINTS
from .models import digest
from .policy import replay_policy


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def episode_on_production_map():
    episode = complete_synthetic_episode()
    # Keep the physical/material tape, lengths, limits and closures. The older
    # generic fixture omits the production map and Dump0's ore role metadata.
    roads = episode['states'][:, 1439:1519].reshape(-1, 16, 5)
    roads[..., :2] = (np.asarray(ROAD_ENDPOINTS, dtype=np.float32) + 1) / 9
    dumps = episode['states'][:, 1295:1391].reshape(-1, 2, 48)
    dumps[:, 0, 1] = 1
    dumps[:, 1, 1] = 0
    return episode


def synthetic_batch(model, *, recording='synthetic-bg009-integration'):
    result = []
    features = feature_contract(model.config['feature_config'])
    for i in range(model.config['ppo']['episodes_per_batch']):
        episode = episode_on_production_map()
        episode['identity'].update(
            recording_id=recording, run_index=i + 1, environment_seed=100 + i,
            actor_sampling_seed=900 + i, split='synthetic',
            observation_schema=learner.OBSERVATION_SCHEMA,
            tokenization_version=learner.TOKENIZATION,
            policy_version=model.policy_version, policy_sha256='e' * 64,
            decision_feature_version=features['version'],
            decision_feature_config_sha256=features['config_sha256'],
            graph_contract_sha256=learner._hash(model.config['graph_contract']))
        with torch.no_grad():
            replay = replay_policy(model.actor, episode['states'][:-1],
                                   actions=episode['actions'])
        episode['old_log_probs'] = replay['log_probs'].numpy().astype(np.float64)
        result.append(episode)
    return result


def make_model():
    return learner.JointCostPPO(learner.default_config(
        source_kind='synthetic', seed=373, execution_profile='synthetic_test_v1'))


def assert_tree_equal(a, b):
    if torch.is_tensor(a):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_tree_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b)
        for left, right in zip(a, b):
            assert_tree_equal(left, right)
    else:
        assert a == b


def _algorithm_ast(module, method):
    tree = ast.parse(Path(module.__file__).read_text())
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == 'JointCostPPO')
    result = next(node for node in cls.body
                  if isinstance(node, ast.FunctionDef) and node.name == method)
    if (result.body and isinstance(result.body[0], ast.Expr)
            and isinstance(result.body[0].value, ast.Constant)
            and isinstance(result.body[0].value.value, str)):
        result.body = result.body[1:]
    return ast.dump(result, include_attributes=False)


@pytest.mark.parametrize('method', [
    '_policy_rows', '_values', 'predict_values', '_freeze_targets',
    '_critic_grad', '_actor_grad', 'update', '_metrics',
])
def test_numerical_algorithm_ast_matches_frozen_learner(method):
    assert _algorithm_ast(learner, method) == _algorithm_ast(previous_learner, method)


@pytest.mark.parametrize('profile', ['formal_32ep_v1', 'smoke_5ep_v1'])
def test_real_profiles_keep_exact_ppo_and_input_objective_contract(profile):
    current = learner.default_config(execution_profile=profile)
    old = previous_learner.default_config(candidate_id='s01_TQ128', execution_profile=profile)
    assert current['ppo'] == old['ppo']
    assert current['actor_advantage'] == old['actor_advantage']
    assert current['reward_contract'] == old['reward_contract']
    assert current['feature_config'] == old['feature_config']
    assert current['resources'] == old['resources']
    assert current['schema'] != old['schema']
    assert current['graph_contract'] != old['graph_contract']
    learner.validate_config(current)
    altered = deepcopy(current)
    altered['ppo']['actor_lr'] *= 2
    with pytest.raises(ValueError, match='frozen PPO'):
        learner.validate_config(altered)


def test_cost_mc_gae_are_same_function_objects_not_reimplemented_formulas():
    for name in ('grade_cost', 'composite_contract', 'validate_composite_contract',
                 'build_composite_targets', 'verify_episode_targets', 'cost_gae'):
        assert getattr(targets, name) is getattr(previous_targets, name)
    assert targets.REWARD_VERSION == previous_targets.REWARD_VERSION
    episode = episode_on_production_map()
    assert targets.verify_episode_targets(episode) == previous_targets.verify_episode_targets(episode)


def test_complete_update_freezes_old_values_once_and_exact_checkpoint_continuation(tmp_path):
    model = make_model()
    episodes = synthetic_batch(model)
    old_values = [model.predict_values(episode).copy() for episode in episodes]
    expected = [targets.cost_gae(
        episode['targets'].step_costs, values,
        terminated=[False, False, True], truncated=[False] * 3,
        gae_lambda=model.config['ppo']['gae_lambda'])
        for episode, values in zip(episodes, old_values)]
    initial = (digest(model.actor), digest(model.critic))
    with patch.object(model, '_freeze_targets', wraps=model._freeze_targets) as freeze:
        report = model.update(episodes, batch_id='synthetic-bg009-first')
    assert freeze.call_count == 1
    assert report['episodes'] == 2 and report['transitions'] == 6
    assert report['actor_optimizer_steps'] == report['critic_optimizer_steps'] == 4
    assert model.update_index == 1
    assert digest(model.actor) != initial[0] and digest(model.critic) != initial[1]
    frozen = model.last_targets
    np.testing.assert_array_equal(frozen['old_values'], np.concatenate([v[:-1] for v in old_values]))
    np.testing.assert_array_equal(frozen['advantages'], np.concatenate([v.advantages for v in expected]))
    np.testing.assert_array_equal(frozen['mc_returns'], np.concatenate([
        episode['targets'].mc_returns for episode in episodes]))
    raw = np.concatenate([value.advantages for value in expected])
    centered = raw - raw.mean()
    np.testing.assert_array_equal(frozen['actor_advantages'], centered / (centered.std() + 1e-8))
    np.testing.assert_array_equal(frozen['next_values'][[2, 5]], np.zeros(2))
    assert all(not value.flags.writeable for value in frozen.values())

    ref = model.save_checkpoint(tmp_path / 'bg009.pt')
    restored = learner.JointCostPPO.from_checkpoint(ref['path'], expected_config=model.config,
                                                  expected_sha256=ref['sha256'])
    assert_tree_equal(model.actor.state_dict(), restored.actor.state_dict())
    assert_tree_equal(model.critic.state_dict(), restored.critic.state_dict())
    assert_tree_equal(model.actor_optimizer.state_dict(), restored.actor_optimizer.state_dict())
    assert_tree_equal(model.critic_optimizer.state_dict(), restored.critic_optimizer.state_dict())
    assert_tree_equal(model.generator.get_state(), restored.generator.get_state())
    assert_tree_equal(model.sampling_generator.get_state(), restored.sampling_generator.get_state())
    assert model.last_target_hash == restored.last_target_hash
    for episode in episodes:
        np.testing.assert_array_equal(model.predict_values(episode), restored.predict_values(episode))
    next_batch = synthetic_batch(model, recording='synthetic-bg009-second')
    model.update(next_batch, batch_id='synthetic-bg009-second')
    restored.update(deepcopy(next_batch), batch_id='synthetic-bg009-second')
    assert digest(model.actor) == digest(restored.actor)
    assert digest(model.critic) == digest(restored.critic)
    assert model.last_target_hash == restored.last_target_hash
    assert_tree_equal(model.actor_optimizer.state_dict(), restored.actor_optimizer.state_dict())
    assert_tree_equal(model.critic_optimizer.state_dict(), restored.critic_optimizer.state_dict())


def test_invalid_cost_certificate_is_rejected_before_optimizer_steps():
    model = make_model()
    episodes = synthetic_batch(model)
    old = episodes[0]['targets']
    episodes[0]['targets'] = replace(old, step_costs=(old.step_costs[0] + 1,) + old.step_costs[1:])
    before = (digest(model.actor), digest(model.critic))
    with (patch.object(model.actor_optimizer, 'step', side_effect=AssertionError('Actor updated')),
          patch.object(model.critic_optimizer, 'step', side_effect=AssertionError('Critic updated'))):
        with pytest.raises(ValueError, match='targets'):
            model.update(episodes, batch_id='invalid-bg009-cost')
    assert (digest(model.actor), digest(model.critic)) == before
    assert model.update_index == 0
    assert not model.actor_optimizer.state and not model.critic_optimizer.state


def test_old_configuration_and_sealed_checkpoint_are_rejected(tmp_path):
    config = previous_learner.default_config(source_kind='synthetic', seed=373,
        candidate_id='s01_TQ128', execution_profile='synthetic_test_v1')
    with pytest.raises(ValueError, match='schema'):
        learner.validate_config(config)
    old = previous_learner.JointCostPPO(config)
    ref = old.save_checkpoint(tmp_path / 'historical-tq128.pt')
    with pytest.raises(ValueError, match='schema'):
        learner.JointCostPPO.from_checkpoint(ref['path'], expected_sha256=ref['sha256'])


def test_new_actor_seal_load_and_bind_are_exact_and_readonly(tmp_path):
    model = make_model()
    episode = synthetic_batch(model)[0]
    raw = episode['states'][:-1]
    before_rng = torch.get_rng_state().clone()
    ref = model.save_actor(tmp_path / 'bg009-actor.pt')
    actor, fingerprint = frozen_actor(ref['path'], allow_small_synthetic=True)
    assert fingerprint == model.policy_version
    assert not actor.training and all(not p.requires_grad for p in actor.parameters())
    assert torch.equal(before_rng, torch.get_rng_state())
    with torch.no_grad():
        left = replay_policy(model.actor, raw)['probabilities']
        right = replay_policy(actor, raw)['probabilities']
    assert torch.equal(left, right)
    bound = model.bind_actor_snapshot(ref['path'])
    assert bound['sha256'] == ref['sha256'] and bound['actor_digest'] == fingerprint
    with pytest.raises(ValueError, match='synthetic'):
        frozen_actor(ref['path'])
    # The byte/manifest seal is mandatory even for synthetic artifacts.
    Path(ref['path'] + '.seal.json').unlink()
    with pytest.raises(ValueError, match='incomplete'):
        frozen_actor(ref['path'], allow_small_synthetic=True)


def test_actor_artifact_and_manifest_update_identity_must_match(tmp_path):
    model = make_model()
    ref = model.save_actor(tmp_path / 'actor-update-identity.pt')
    saved, manifest = learner._load_checkpoint(ref['path'])
    saved['update_index'] = manifest['update_index'] + 1
    with patch.object(learner, '_load_checkpoint', return_value=(saved, manifest)):
        with pytest.raises(ValueError, match='update identity'):
            frozen_actor(ref['path'], allow_small_synthetic=True)


def test_evaluation_cannot_relabel_an_old_model_as_business_graph():
    old = previous_learner.JointCostPPO(previous_learner.default_config(
        source_kind='synthetic', candidate_id='s01_TQ128', execution_profile='synthetic_test_v1'))
    with pytest.raises(ValueError, match='business-graph architecture'):
        evaluate_episodes([], old)


def test_independent_evaluation_preserves_training_and_all_eight_metric_contracts():
    model = make_model()
    model.update(synthetic_batch(model), batch_id='synthetic-before-readonly-evaluation')
    episodes = synthetic_batch(model, recording='synthetic-bg009-independent-metric-check')
    before = dict(actor=deepcopy(model.actor.state_dict()), critic=deepcopy(model.critic.state_dict()),
        actor_optimizer=deepcopy(model.actor_optimizer.state_dict()),
        critic_optimizer=deepcopy(model.critic_optimizer.state_dict()),
        updates=model.update_index, consumed=deepcopy(model.consumed_batches),
        rng=torch.get_rng_state().clone(), shuffle=model.generator.get_state().clone(),
        sampling=model.sampling_generator.get_state().clone(), targets=model.last_target_hash,
        actor_grad=[None if p.grad is None else p.grad.clone() for p in model.actor.parameters()],
        critic_grad=[None if p.grad is None else p.grad.clone() for p in model.critic.parameters()])
    with (patch.object(model.actor_optimizer, 'step', side_effect=AssertionError('evaluation updated Actor')),
          patch.object(model.critic_optimizer, 'step', side_effect=AssertionError('evaluation updated Critic'))):
        result = evaluate_episodes(episodes, model, expected_complete_shifts=2,
            checkpoint_provenance=dict(architecture=model.config['graph_contract']['version']))
    head = result['headline_metrics']
    for key in ('grade_rmse', 'joint_four_window_compliance_rate', 'ore_target_met_rate',
                'waste_target_met_rate', 'complete_task_rate', 'mean_total_wait_seconds',
                'critic_mc_mse_pooled', 'critic_mc_explained_variance_pooled'):
        assert key in head and head[key] is not None and np.isfinite(head[key])
    assert head['grade_rmse_valid_windows'] == 4 and head['grade_rmse_total_windows'] == 8
    assert head['empty_windows'] == 4 and head['joint_four_window_compliance_rate'] == 0.
    assert head['mean_total_wait_seconds'] == 10. and head['total_wait_seconds'] == 20.
    assert result['summary']['evaluation_thresholds']['waste_target_tons'] == 4000.
    assert result['summary']['evaluation_thresholds']['waste_dispatch_stop_tons'] == 6000.
    assert result['critic_mc']['pooled']['n_states'] == 6
    assert result['training_state_unchanged'] and result['optimizer_updates'] == 0
    assert result['actor_updates'] == result['critic_updates'] == 0
    assert_tree_equal(before['actor'], model.actor.state_dict())
    assert_tree_equal(before['critic'], model.critic.state_dict())
    assert_tree_equal(before['actor_optimizer'], model.actor_optimizer.state_dict())
    assert_tree_equal(before['critic_optimizer'], model.critic_optimizer.state_dict())
    assert_tree_equal(before['consumed'], model.consumed_batches)
    assert_tree_equal(before['rng'], torch.get_rng_state())
    assert_tree_equal(before['shuffle'], model.generator.get_state())
    assert_tree_equal(before['sampling'], model.sampling_generator.get_state())
    assert_tree_equal(before['actor_grad'], [p.grad for p in model.actor.parameters()])
    assert_tree_equal(before['critic_grad'], [p.grad for p in model.critic.parameters()])
    assert before['updates'] == model.update_index and before['targets'] == model.last_target_hash
    json.dumps(result, allow_nan=False)

