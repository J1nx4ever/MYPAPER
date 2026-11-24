"""Synthetic denominator/value checks only; no RPC or physical shifts."""
from copy import deepcopy
import json
import math

import numpy as np
import pytest
import torch

from . import evaluation
from .evaluation import _aggregate_critic_metrics, _grade_rmse
from baseline1_v1.evaluation import _score_stats


def physical(grades, *, ore=16000., waste=4000., wait=100., shift_grade=.27):
    windows=[]
    for i, grade in enumerate(grades):
        compliant=evaluation._grade_compliant(grade)
        windows.append(dict(window=i,grade=grade,empty=grade is None,
            compliant=compliant,violation=not compliant,
            strict_band_compliant=grade is not None and .26 <= grade <= .28,
            center_deviation=None if grade is None else abs(grade-.27),
            exceedance=None if grade is None else max(.26-grade,grade-.28,0.),
            continuous_cost=2. if grade is None else min(max(abs(grade-.27)-.004,0.)/.006,2.),
            tons=4000. if grade is not None else 0.,metal_tons=0.))
    ore_met,waste_met=ore>=16000.,waste>=4000.
    joint=all(w['compliant'] for w in windows)
    queues=evaluation.decision_sampled_queue_statistics(np.zeros((1,1519)))
    return dict(windows=windows,shift_grade=shift_grade,shift_metal_tons=0.,
        shift_grade_compliant=evaluation._grade_compliant(shift_grade),
        ore_tons=ore,waste_tons=waste,total_wait_seconds=wait,
        fleet_wait_hours=wait/3600.,mean_vehicle_wait_seconds=wait/9.,
        ore_target_met=ore_met,waste_target_met=waste_met,both_targets_met=ore_met and waste_met,
        epsilon_ore_target_met=ore>=16000.-evaluation.TONNAGE_EPSILON,
        epsilon_waste_target_met=waste>=4000.-evaluation.TONNAGE_EPSILON,
        all_four_windows_compliant=joint,complete_task=joint and ore_met and waste_met,
        grade_cost=sum(w['continuous_cost'] for w in windows),production_cost=0.,
        waiting_cost=.2*wait/28800.,combined_cost=1.,training_reward=-1.,
        decision_sampled_queues=queues)


def test_grade_rmse_uses_all_valid_windows_and_distinguishes_complete_shift_rmse():
    rows=[physical([.25,.27,None,.31],shift_grade=.265),
          physical([.29,None,.26,.27],shift_grade=.285)]
    before=deepcopy(rows)
    result=_grade_rmse(rows)
    overall=result['overall_valid_windows']
    assert (overall['valid_count'],overall['total_count'],overall['empty_count'])==(6,8,2)
    assert overall['rmse']==pytest.approx(math.sqrt(.0025/6))
    assert [r['valid_count'] for r in result['per_window']]==[2,1,1,2]
    assert result['processed_tonnage_shift_grade']['rmse']==pytest.approx(math.sqrt((.005**2+.015**2)/2))
    assert result['processed_tonnage_shift_grade']['rmse']!=overall['rmse']
    assert rows==before


def test_empty_windows_remain_failures_and_are_never_filled_with_target_grade():
    result=_grade_rmse([physical([None]*4,shift_grade=None)])
    assert result['overall_valid_windows']==dict(rmse=None,sum_squared_error=0.,
        valid_count=0,total_count=4,empty_count=4)
    assert result['processed_tonnage_shift_grade']['rmse'] is None
    false_empty=physical([None]*4)
    false_empty['windows'][0]['grade']=.27
    with pytest.raises(ValueError,match='empty'):_grade_rmse([false_empty])
    json.dumps(result,allow_nan=False)


def test_pooled_value_mse_and_ev_differ_from_mean_episode_metrics():
    targets=[np.array([1.,3.]),np.array([0.,4.,8.,12.])]
    predictions=[np.array([0.,2.]),np.zeros(4)]
    result=_aggregate_critic_metrics([_score_stats(t,p) for t,p in zip(targets,predictions)])
    all_targets=np.concatenate(targets);residual=all_targets-np.concatenate(predictions)
    assert result['pooled']['mse']==pytest.approx(float(np.mean(residual**2)))
    assert result['pooled']['explained_variance']==pytest.approx(1.-np.var(residual)/np.var(all_targets))
    assert result['episode_mean']['mse']==pytest.approx((1.+56.)/2)
    assert result['episode_mean']['explained_variance']==pytest.approx(.5)
    assert result['complete_shift_equal']['explained_variance']!=pytest.approx(.5)
    assert result['state_equal']==result['pooled']


def test_near_constant_mc_targets_have_null_ev_and_explicit_episode_denominator():
    constant=_score_stats(np.array([2.,2.+1e-10]),np.array([1.,2.]))
    regular=_score_stats(np.array([1.,3.]),np.array([0.,2.]))
    mixed=_aggregate_critic_metrics([constant,regular])
    assert mixed['episode_mean']['explained_variance']==pytest.approx(1.)
    assert mixed['episode_mean']['explained_variance_valid_shifts']==1
    assert mixed['episode_mean']['explained_variance_undefined_shifts']==1
    result=_aggregate_critic_metrics([constant])
    assert result['pooled']['explained_variance'] is None
    assert result['episode_mean']['explained_variance'] is None
    assert result['pooled']['mse'] is not None
    json.dumps(result,allow_nan=False)


def episodes_for_summary():
    rows=[physical([.27]*4),physical([.25,None,.27,.27],ore=15999.,waste=6000.,wait=900.)]
    return [dict(identity=dict(run_index=i+1,environment_seed=100+i,policy_version='a'*64,
        source_kind='synthetic',reward_version=evaluation.REWARD_VERSION),
        composite_contract=evaluation.composite_contract(),actions=[0],
        states=np.zeros((2,1519)),fake_physical=row) for i,row in enumerate(rows)]


def test_eight_metric_summary_keeps_failed_shifts_for_wait_and_all_denominators(monkeypatch):
    episodes=episodes_for_summary()
    monkeypatch.setattr(evaluation,'_physical',lambda episode:deepcopy(episode['fake_physical']))
    report=evaluation.evaluate_episodes(episodes,expected_complete_shifts=2)
    summary=report['summary'];head=report['headline_metrics']
    assert report['metrics'] is summary and report['episodes']==report['per_episode']
    assert summary['joint_four_window_compliance_rate']==.5
    assert summary['overall_window_compliance_rate']==.75
    assert [r['compliance_rate'] for r in head['per_window_compliance']]==[.5,.5,1.,1.]
    assert (head['grade_rmse_valid_windows'],head['grade_rmse_total_windows'],head['empty_windows'])==(7,8,1)
    assert head['grade_rmse']==pytest.approx(math.sqrt(.02**2/7))
    assert head['ore_target_met_rate']==.5 and head['waste_target_met_rate']==1.
    assert head['complete_task_rate']==.5
    assert head['mean_total_wait_seconds']==500. and head['total_wait_seconds']==1000.
    assert summary['evaluation_thresholds']['waste_target_tons']==4000.
    assert summary['evaluation_thresholds']['waste_dispatch_stop_tons']==6000.
    assert report['critic_mc'] is None and head['critic_mc_mse_pooled'] is None
    assert report['optimizer_updates']==report['actor_updates']==report['critic_updates']==0
    json.dumps(report,allow_nan=False)


@pytest.mark.parametrize('mutation',[
    lambda es:es[1]['identity'].update(run_index=1),
    lambda es:es[1]['identity'].update(environment_seed=100),
    lambda es:es[1]['identity'].update(policy_version='b'*64),
    lambda es:es[1]['identity'].update(reward_version='old-target'),
    lambda es:es[1]['composite_contract'].update(wait_weight=.1),
])
def test_identity_and_objective_mismatch_fail_before_physical_scoring(monkeypatch,mutation):
    episodes=episodes_for_summary();mutation(episodes)
    def forbidden(_):raise AssertionError('invalid evaluation reached physical scoring')
    monkeypatch.setattr(evaluation,'_physical',forbidden)
    with pytest.raises(ValueError):evaluation.evaluate_episodes(episodes)


def complete_synthetic_episode():
    from grade_event_memory_v1.events import build_memory
    from grade_event_memory_v1.test_events import synthetic_events
    from relation_gat_ego_v1.test_admission_support import _state
    from .targets import build_composite_targets
    memory=build_memory(synthetic_events())
    terminal=dict(ore_tons=40.,waste_tons=0.,total_wait_seconds=10.)
    episode=dict(identity=dict(run_index=1,environment_seed=101,policy_version='a'*64,
        source_kind='synthetic',reward_version=evaluation.REWARD_VERSION),memory=memory,
        states=np.stack([_state(0),_state(20,q=20.,fifo=20.,shift=20.),
            _state(7200,fifo=10.,shift=30.),_state(28800,terminal=True,shift=40.)]),
        actions=np.array([0,0,0]),wait_deltas=np.array([3.,0.,7.]),evaluation=terminal,
        composite_contract=evaluation.composite_contract())
    episode['targets']=build_composite_targets(memory,wait_deltas=episode['wait_deltas'],**terminal)
    return episode


def test_real_material_and_mc_checks_are_retained_without_a_model():
    episode=complete_synthetic_episode()
    report=evaluation.evaluate_episodes([episode],expected_complete_shifts=1)
    assert report['headline_metrics']['empty_windows']==2
    assert report['headline_metrics']['ore_target_met_rate']==0.
    assert report['summary']['grade_cost']['mean']==4.
    bad=deepcopy(episode);bad['wait_deltas'][0]+=1.
    with pytest.raises(ValueError,match='conserve|targets'):
        evaluation.evaluate_episodes([bad])


@pytest.mark.parametrize('grade,compliant',[
    (.26-.5e-12,True),(.26-2e-12,False),(.28+.5e-12,True),(.28+2e-12,False),
])
def test_window_and_shift_grade_use_the_audited_rpc_tolerance(grade,compliant):
    from dataclasses import replace
    from grade_event_memory_v1.events import build_memory,grade_reward
    from grade_event_memory_v1.test_events import synthetic_events
    from .targets import build_composite_targets
    events=[]
    for event in synthetic_events():
        if event.kind in ('UNLOAD','PROCESS'):
            event=replace(event,grade=grade,metal=event.tons*grade)
        elif event.kind=='WINDOW_CLOSE' and event.window_q>0:
            _,_,raw=grade_reward(event.window_q,event.window_q*grade)
            event=replace(event,window_m=event.window_q*grade,window_grade=grade,grade_reward_raw=raw)
        events.append(event)
    episode=complete_synthetic_episode();episode['memory']=build_memory(events)
    episode['targets']=build_composite_targets(episode['memory'],wait_deltas=episode['wait_deltas'],
        **episode['evaluation'])
    result=evaluation.evaluate_episodes([episode])
    actual=result['per_episode'][0]['physical']
    assert actual['windows'][0]['compliant'] is compliant
    assert actual['shift_grade_compliant'] is compliant
    assert result['summary']['per_window_compliance'][0]['compliance_rate']==float(compliant)
    assert result['summary']['evaluation_thresholds']['grade_epsilon']==1e-12
    assert result['summary']['grade_epsilon_boundary_disagreement_windows']==(2 if compliant else 0)


def test_tonnage_epsilon_is_a_compatibility_diagnostic_and_cannot_relax_primary_targets(monkeypatch):
    rows=[physical([.27]*4,ore=16000.-.5e-6,waste=4000.-.5e-6)]
    episode=episodes_for_summary()[0];episode['fake_physical']=rows[0]
    monkeypatch.setattr(evaluation,'_physical',lambda episode:deepcopy(episode['fake_physical']))
    result=evaluation.evaluate_episodes([episode]);summary=result['summary']
    assert summary['ore_target_met_rate']==summary['waste_target_met_rate']==summary['complete_task_rate']==0.
    assert summary['epsilon_ore_target_met_rate']==summary['epsilon_waste_target_met_rate']==1.
    assert summary['epsilon_compatibility_disagreement_shifts']==1
    assert summary['evaluation_thresholds']['tonnage_epsilon_used_for_primary_production'] is False


@pytest.mark.parametrize('candidate',['s01_TQ0','s01_TQ128'])
def test_scalar_model_evaluation_and_offline_restore_preserve_training_state(tmp_path,candidate):
    from .learner import JointCostPPO, default_config
    from .models import digest
    before_threads=torch.get_num_threads();torch.set_num_threads(1)
    try:
        episode=complete_synthetic_episode()
        model=JointCostPPO(default_config(source_kind='synthetic',candidate_id=candidate,
            seed=17,execution_profile='synthetic_test_v1'))
        episode['identity']['policy_version']=digest(model.actor)
        before=(digest(model.actor),digest(model.critic),model.update_index,torch.get_rng_state().clone())
        report=evaluation.evaluate_episodes([episode],model,expected_complete_shifts=1)
        assert report['critic_mc']['pooled']['n_states']==3
        assert report['per_episode'][0]['critic_mc']['mse']>=0.
        assert report['headline_metrics']['critic_mc_mse_pooled']==report['critic_mc']['pooled']['mse']
        assert (digest(model.actor),digest(model.critic),model.update_index)==before[:3]
        assert torch.equal(torch.get_rng_state(),before[3])
        assert not model.actor_optimizer.state and not model.critic_optimizer.state
        assert all(p.grad is None for p in model.actor.parameters())
        assert all(p.grad is None for p in model.critic.parameters())
        ref=model.save_checkpoint(tmp_path/'frozen.pt')
        restored=JointCostPPO.from_checkpoint(ref['path'],expected_config=model.config,
            expected_sha256=ref['sha256'])
        consistency=evaluation.offline_observation_check(model,restored,[episode])
        assert consistency['additional_physical_shifts']==0 and not consistency['independent_evaluation']
        assert consistency['episodes'][0]['policy_probabilities_bitwise_equal']
        assert consistency['episodes'][0]['value_predictions_bitwise_equal']
        assert consistency['episodes'][0]['true_terminal_value']==0.
        json.dumps(report,allow_nan=False)
    finally:
        torch.set_num_threads(before_threads)
