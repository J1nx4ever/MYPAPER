"""BG128 campaign boundary checks; no socket, simulator or optimizer step.

Historical raw rows are used only to exercise frozen inference and private-RNG
replay. They are never admitted as a new BG128 training batch.
"""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from .contract import CANDIDATES, ablation_contract, global_update, schedule
from .learner import JointCostPPO, default_config
from .admission import _check_actor_phase, frozen_actor
from .collection import OnlineActor
from .plan_io import ROOT, reference, write


@pytest.fixture(autouse=True)
def one_thread():
    previous=torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def config():
    return default_config(seed=75597143230906)


def suite_fixture(tmp_path, config):
    arm=dict(candidate_id='s01_BG128',learner_config=config,
             python_output=str(tmp_path/'output'/'arms'/'s01_BG128'))
    suite=dict(campaign_id='isolated-bg128-boundary',pair_id='isolated-single',
        stage='formal',arms=[arm],arm_order=list(CANDIDATES),
        python_output=str(tmp_path/'output'),cpu_threads=1,phase_configs=[],jobs=[])
    path=tmp_path/'suite.json';write(path,suite)
    return suite,path


def test_single_arm_formal_schedule_has_no_evaluation_and_local_global_updates_match(tmp_path,config):
    from . import runtime
    suite,_=suite_fixture(tmp_path,config)
    assert len(schedule(False))==16 and sum(row[3] for row in schedule(False))==512
    for name,role,local,n in schedule(False):
        assert role=='train' and n==32 and global_update('s01_BG128',local)==local
        phase='s01_BG128_'+name
        jobs=[dict(candidate_id='s01_BG128',phase_id=phase,role=role,arm_update=local,
            target_training_update=local,checkpoint_update=local-1) for _ in range(n)]
        cfg=dict(candidate_id='s01_BG128',phase_id=phase,arm_phase_id=name,
            campaign_id=suite['campaign_id'],arm_update=local,jobs=jobs,
            python_output=str(Path(suite['arms'][0]['python_output'])/name))
        assert runtime._phase_contract(cfg,suite)[1:]==('train',local,local)
        cfg['jobs'][0]['target_training_update']=2*local-1
        if local>1:
            with pytest.raises(ValueError,match='checkpoint differs'):
                runtime._phase_contract(cfg,suite)


def test_fresh_checkpoint_actor_and_phase_bind_bg128_exactly(tmp_path,config):
    from . import runtime
    from .c_features import feature_contract
    from .targets import composite_contract
    suite,path=suite_fixture(tmp_path,config)
    model,initial=runtime._initial_arm(suite,path,suite['arms'][0])
    proof=runtime._compare_initializations(suite,model,initial)
    observed=json.loads(Path(proof['path']).read_bytes())
    assert observed['checks'] and all(observed['checks'].values())
    assert not observed['old_weights_loaded'] and observed['optimizer_updates_in_comparison']==0
    assert observed['input_contract']==ablation_contract()
    assert not model.actor_optimizer.state and not model.critic_optimizer.state
    actor_path=tmp_path/'phase_actor.pt';model.save_actor(actor_path)
    actor,policy=frozen_actor(actor_path)
    assert policy==model.policy_version
    phase=dict(candidate_id='s01_BG128',execution_profile='formal_32ep_v1',
        jobs=[dict(actor_seed=config['seed'])],feature_contract=feature_contract(config['feature_config']),
        graph_contract=ablation_contract(),training_reward='negative_combined_grade_production_wait_cost_v1',
        composite_contract=composite_contract())
    _check_actor_phase(actor,phase)
    bad=copy.deepcopy(phase);bad['graph_contract']['graph_layers']=3
    with pytest.raises(ValueError,match='differs from phase'):
        _check_actor_phase(actor,bad)
    restored=runtime._load_model(initial,config,restore=False)
    assert restored.update_index==0 and restored.policy_version==model.policy_version
    assert not restored.actor_optimizer.state and not restored.critic_optimizer.state


def test_old_task_query_actor_and_checkpoint_are_rejected(tmp_path):
    from single_critic_task_query_v1.learner import JointCostPPO as OldPPO
    from single_critic_task_query_v1.learner import default_config as old_config
    old=OldPPO(old_config(seed=75597143230906,candidate_id='s01_TQ128'))
    actor=tmp_path/'old_actor.pt';checkpoint=tmp_path/'old_checkpoint.pt'
    old.save_actor(actor);old.save_checkpoint(checkpoint)
    with pytest.raises(ValueError,match='business-graph Actor artifact required'):
        frozen_actor(actor)
    with pytest.raises(ValueError,match='checkpoint schema differs'):
        JointCostPPO.from_checkpoint(checkpoint)


def test_online_sampling_raw_snapshot_and_exact_private_rng_replay(config):
    from python.constrained_batch import load_dataset
    from .behavior import exact_replay
    from .admission import _verify_sampling
    snapshots=json.loads((ROOT/'reports/token_attention_001/business_graph_summary_009/historical_snapshot_001/snapshots.json').read_bytes())
    snap=snapshots['T1'];raw=np.asarray(snap['raw'],dtype=np.float32)
    rows,_,_,_,_=load_dataset(Path(snapshots['selection']['source_hdf5']).parent,require_real=True)
    try:
        info=copy.deepcopy(rows[snap['state_index']-1]['settlement_info'])
    finally:
        if hasattr(rows,'close'):rows.close()
    info.update(experiment_run_index=9910001,experiment_seed=7211,
                decision_sequence=0,episode_id='isolated-bg128-inference-only')
    jobs=[dict(run_index=9910001,experiment_seed=7211,actor_sampling_seed=7212)]
    model=JointCostPPO(config);before=model.policy_version
    online=OnlineActor(model.actor,jobs)
    action,log_probability=online.sample_with_info(raw,raw[19:23].copy(),info)
    record=dict(action=action,old_log_prob=log_probability,**online.last_metadata)
    replay=exact_replay(online.actor,raw[None,:],np.asarray([action]))
    assert float(replay['log_probs'][0])==log_probability
    _verify_sampling([record],replay['probabilities'],jobs[0])
    assert record['critic_only'] is False and record['behavior_kind']=='frozen_actor'
    assert raw[19+action]==1 and model.policy_version==before
    assert online.summary()[0]['actions']==1
    bad=copy.deepcopy(record);bad['actor_rng_after_sha256']='0'*64
    with pytest.raises(ValueError,match='sampled action/RNG differs'):
        _verify_sampling([bad],replay['probabilities'],jobs[0])
    wrong=copy.deepcopy(info);wrong['decision_sequence']=2
    with pytest.raises(ValueError,match='sequence differs'):
        online.sample_with_info(raw,raw[19:23].copy(),wrong)
    assert not model.actor_optimizer.state and not model.critic_optimizer.state


def test_ready_publication_is_single_claimed_job_and_bg_graph_identity(tmp_path,config,monkeypatch):
    from . import runtime
    from baseline1_v1.java_plan import COLUMNS
    suite,path=suite_fixture(tmp_path,config)
    suite.update(control_directory=str(tmp_path/'control'),source_pins={},master_plan={'sha256':'a'*64})
    model,initial=runtime._initial_arm(suite,path,suite['arms'][0])
    runtime._compare_initializations(suite,model,initial)
    cfg=tmp_path/'phase.json';write(cfg,{'isolated_test':True})
    suite['phase_configs']=[reference(cfg)]
    job={key:'isolated' for key in COLUMNS}
    job.update(candidate_id='s01_BG128',group='BG128',run_index=9910001,role='train')
    ledger=SimpleNamespace(job=lambda observed:dict(status='CLAIMED',identity=observed))
    monkeypatch.setattr(runtime,'check_sources',lambda pins:None)
    runtime._publisher(suite,ledger)(job,cfg)
    ready=(tmp_path/'control'/'ready'/'run_9910001.properties').read_text()
    assert 'group=BG128\n' in ready
    assert 'config_sha256='+reference(cfg)['sha256'] in ready
    assert 'all_comparison_checkpoints_frozen=false\n' in ready
    wrong=SimpleNamespace(job=lambda observed:dict(status='PLANNED',identity=observed))
    with pytest.raises(ValueError,match='claimed'):
        runtime._publisher(suite,wrong)(job,cfg)


def test_authorization_binds_only_single_seed_training_scope(tmp_path,config):
    from .launch import authorization
    suite,path=suite_fixture(tmp_path,config)
    suite.update(execution_profile='formal_32ep_v1',new_shift_cap=512,legacy_reserved_shifts=0,
                 maximum_total_if_approved=21000)
    proof=authorization(suite,reference(path),'先进行一个随机种子的训练')
    assert proof['ack_frozen_bg128_single_seed'] is True and proof['max_new_shifts']==512
    assert 'No extra smoke or independent evaluation' in proof['scope_interpretation']
