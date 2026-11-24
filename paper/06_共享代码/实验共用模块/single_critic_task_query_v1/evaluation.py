"""Read-only task-query evaluation with explicit physical/value denominators.

All admitted complete shifts, including failed production, remain in the
evaluation set.  Window grade RMSE excludes empty windows and reports their
count; compliance includes those windows as failures.  Critic fit is against
the same frozen joint-cost MC targets used by the learner, never raw RPC reward.
"""
from __future__ import annotations

import math
import hashlib
import numpy as np
import torch

from baseline1_v1.evaluation import ReadOnlyTrainingGuard, _distribution, _regression, _score_stats
from relation_gat_ego_formal_v1.evaluation import _grade_rmse
from grade_event_memory_v1.events import build_memory
from .targets import REWARD_VERSION, verify_episode_targets, composite_contract, grade_cost
from .models import digest
from .policy import replay_policy
from python.observation_contract import FIELD_NAMES


FORMAT = 'single-critic-task-query-joint-evaluation-v1'
# Frozen RPC numeric tolerances, already audited by reward_contract and the
# prepared common config. Production's primary metric stays strictly >=target;
# tonnage epsilon is only an explicitly named compatibility diagnostic.
GRADE_EPSILON = 1e-12
TONNAGE_EPSILON = 1e-6


def _grade_compliant(grade):
    return grade is not None and .26-GRADE_EPSILON <= grade <= .28+GRADE_EPSILON


def _aggregate_critic_metrics(records):
    """Separate a pooled EV from the arithmetic mean of per-shift EVs.

    The legacy complete_shift_equal metric is retained, but its EV describes
    an equal-shift mixture.  It is generally different from mean episode EV.
    Undefined per-shift EVs (near-zero target variance) are counted, not zeroed.
    """
    _require(bool(records), 'Critic metrics require nonempty complete shifts')
    pooled = _regression(records, shift_equal=False)
    mixture = _regression(records, shift_equal=True)
    per_episode = [_regression([row], shift_equal=False) for row in records]
    valid = [row['explained_variance'] for row in per_episode
             if row['explained_variance'] is not None]
    episode_mean = dict(n_shifts=len(records), n_states=pooled['n_states'],
        weighting='arithmetic_mean_of_complete_shift_metrics',
        mse=math.fsum(row['mse'] for row in per_episode)/len(records),
        bias=math.fsum(row['bias'] for row in per_episode)/len(records),
        explained_variance=math.fsum(valid)/len(valid) if valid else None,
        explained_variance_valid_shifts=len(valid),
        explained_variance_undefined_shifts=len(records)-len(valid),
        explained_variance_reason='constant_target_in_all_shifts' if not valid else None,
        explained_variance_denominator='complete shifts with non-negligible MC-target variance',
        target_variance=math.fsum(row['target_variance'] for row in per_episode)/len(records),
        residual_variance=math.fsum(row['residual_variance'] for row in per_episode)/len(records))
    return dict(state_equal=pooled, complete_shift_equal=mixture,
                pooled=pooled, episode_mean=episode_mean)


def _require(condition, message):
    if not condition: raise ValueError(message)


def _queue_samples(observations):
    """Use occupied existing slots, excluding the post-action terminal state."""
    x=np.asarray(observations)
    _require(x.ndim==2 and x.shape[1]==1519 and len(x)>0 and np.isfinite(x).all(),
             'queue samples require finite decision states[N,1519]')
    result={}
    for site in [f'load.{index}' for index in range(4)]+[f'dump.{index}' for index in range(2)]:
        indices=[index for index,name in enumerate(FIELD_NAMES) if name.startswith(site+'.queue.')]
        _require(len(indices)==9, 'queue schema slot inventory changed')
        result[site]=np.count_nonzero(x[:,indices]>0,axis=1)
    return result


def decision_sampled_queue_statistics(observations):
    """Action-decision occupancy; neither interpolation nor time weighting."""
    samples=_queue_samples(observations)
    return dict(weighting='action_decision_samples; not time weighted',time_weighted=False,
                sites={site:_distribution(values) for site,values in samples.items()})


def _combined_queue_statistics(episodes, physical):
    samples=[_queue_samples(episode['states'][:-1]) for episode in episodes]
    sites=list(samples[0])
    return dict(time_weighted=False,sampling='existing queue occupancy at action decisions only',
        state_equal=dict(weighting='all_action_decisions; not time weighted',
            sites={site:_distribution(np.concatenate([sample[site] for sample in samples])) for site in sites}),
        complete_shift_equal=dict(weighting='equal complete shifts; distribution of per-shift decision means',
            sites={site:_distribution([row['decision_sampled_queues']['sites'][site]['mean']
                                      for row in physical]) for site in sites}))


def _physical(episode):
    memory = episode['memory']
    _require(build_memory(memory.events) == memory, 'evaluation material tape differs')
    _require(episode['targets'] == verify_episode_targets(episode), 'evaluation joint labels differ')
    windows = sorted(memory.settlements,key=lambda item:item.window)
    _require([window.window for window in windows] == [0,1,2,3], 'complete four-window evaluation required')
    records = []
    for window in windows:
        valid = bool(window.grade_valid)
        grade = float(window.grade) if valid else None
        compliant = _grade_compliant(grade)
        cost = grade_cost(grade,empty=not valid)
        records.append(dict(window=window.window,tons=float(window.tons),metal_tons=float(window.metal),
            grade=grade,empty=not valid,compliant=compliant,violation=not compliant,
            strict_band_compliant=valid and .26 <= grade <= .28,
            center_deviation=abs(grade-.27) if valid else None,
            exceedance=max(.26-grade,grade-.28,0.) if valid else None,
            continuous_cost=cost))
    physical = episode['evaluation']
    ore,waste,wait = (float(physical[key]) for key in ('ore_tons','waste_tons','total_wait_seconds'))
    _require(all(math.isfinite(value) and value >= 0 for value in (ore,waste,wait)),
             'nonfinite/negative evaluation production or fleet waiting')
    ore_met,waste_met = ore >= 16000., waste >= 4000.
    joint = all(window['compliant'] for window in records)
    _require(len(episode['states'])==len(episode['actions'])+1,
             'decision queue samples require N+1 complete-shift states')
    processed_q=math.fsum(w['tons'] for w in records)
    processed_m=math.fsum(w['metal_tons'] for w in records)
    shift_grade=processed_m/processed_q if processed_q>0 else None
    return dict(windows=records,ore_tons=ore,waste_tons=waste,total_wait_seconds=wait,
                shift_grade=shift_grade,shift_metal_tons=processed_m,
                shift_grade_compliant=_grade_compliant(shift_grade),
                decision_sampled_queues=decision_sampled_queue_statistics(episode['states'][:-1]),
                fleet_wait_hours=wait/3600.,mean_vehicle_wait_seconds=wait/9.,
                ore_target_met=ore_met,waste_target_met=waste_met,
                epsilon_ore_target_met=ore>=16000.-TONNAGE_EPSILON,
                epsilon_waste_target_met=waste>=4000.-TONNAGE_EPSILON,
                both_targets_met=ore_met and waste_met,all_four_windows_compliant=joint,
                complete_task=ore_met and waste_met and joint,
                grade_cost=math.fsum(episode['targets'].grade_costs),
                production_cost=math.fsum(episode['targets'].production_costs),
                waiting_cost=math.fsum(episode['targets'].wait_costs),
                combined_cost=math.fsum(episode['targets'].step_costs),
                training_reward=-math.fsum(episode['targets'].step_costs))


def evaluate_episodes(episodes, model=None, *, expected_complete_shifts=None,
                      checkpoint_provenance=None, training_environment=None):
    """Keep failed-production shifts, score MC cost values, never update a model.

    Episode inputs must already pass the source-bound collection loader. When a
    model is supplied it provides predict_values(episode) in *cost* convention.
    """
    episodes = tuple(episodes)
    _require(bool(episodes), 'empty independent evaluation')
    if expected_complete_shifts is not None:
        _require(type(expected_complete_shifts) is int and expected_complete_shifts > 0 and
                 len(episodes) == expected_complete_shifts, 'independent evaluation shift count differs')
    _require(len({episode['identity']['run_index'] for episode in episodes}) == len(episodes),
             'duplicate evaluated run identity')
    _require(len({episode['identity']['environment_seed'] for episode in episodes}) == len(episodes),
             'duplicate evaluated environment seed')
    _require(len({episode['identity']['policy_version'] for episode in episodes}) == 1,
             'mixed evaluated Actors')
    _require(len({episode['identity']['source_kind'] for episode in episodes}) == 1,
             'mixed evaluation source scope')
    _require(all(episode['identity']['reward_version'] == REWARD_VERSION for episode in episodes),
             'evaluation reward contract differs')
    _require(all(episode.get('composite_contract') == composite_contract() for episode in episodes),
             'evaluation frozen joint-cost coefficients differ')
    _require(all(episode['identity']['source_kind']=='synthetic' or
                 episode['identity']['split']=='validation' for episode in episodes),
             'independent evaluation requires validation partition, never training shifts')
    records, value_stats, entropies = [], [], []
    with ReadOnlyTrainingGuard(model,training_environment):
        if model is not None:
            _require(not hasattr(model,'critics') and not hasattr(model,'lambdas'),
                     'joint-cost evaluation requires a single scalar Critic')
            _require(digest(model.actor) == episodes[0]['identity']['policy_version'],
                     'evaluated trajectories belong to a different Actor')
            _require(model.config.get('reward_contract') == composite_contract(),
                     'evaluated Critic objective differs from frozen MC targets')
        for episode in episodes:
            if model is not None and episode['identity']['source_kind'] != 'synthetic':
                from .c_features import feature_contract
                contract = feature_contract(model.actor.feature_config)
                _require(episode['identity'].get('decision_feature_version') == contract['version'] and
                         episode['identity'].get('decision_feature_config_sha256') == contract['config_sha256'],
                         'evaluated decision-feature identity differs')
            physical = _physical(episode)
            record = dict(identity=dict(episode['identity']),physical=physical,
                          transitions=len(episode['actions']))
            if model is not None:
                with torch.no_grad():
                    predictions = model.predict_values(episode)
                    if isinstance(predictions,torch.Tensor): predictions=predictions.detach().cpu().numpy()
                    predictions=np.asarray(predictions,dtype=np.float64)
                    n=len(episode['actions'])
                    _require(predictions.shape in ((n,),(n+1,)) and np.isfinite(predictions).all(),
                             'Critic must predict one finite cost scalar per decision')
                    if len(predictions)==n+1:
                        _require(predictions[-1] == 0., 'true terminal cost value must be exactly zero')
                    predictions=predictions[:n]
                    targets=np.asarray(episode['targets'].mc_returns,dtype=np.float64)
                    stat=_score_stats(targets,predictions)
                    value_stats.append(stat)
                    entropy=replay_policy(model.actor, episode['states'][:-1])['entropies'].cpu().numpy()
                    _require(entropy.shape==(n,) and np.isfinite(entropy).all(), 'nonfinite policy entropy')
                    entropies.append(entropy.astype(float))
                    record['value_statistics']=stat
                    record['critic_mc']=_regression([stat],shift_equal=False)
                    record['policy_entropy_mean']=float(entropy.mean())
            records.append(record)
    physical=[record['physical'] for record in records]
    per_window=[]
    for index in range(4):
        rows=[row['windows'][index] for row in physical]
        grades=[row['grade'] for row in rows if row['grade'] is not None]
        per_window.append(dict(window=index,shifts=len(rows),valid_grade_count=len(grades),
            empty_windows=sum(row['empty'] for row in rows),
            actual_grades=[row['grade'] for row in rows],
            grade=_distribution(grades) if grades else None,
            violation_count=sum(row['violation'] for row in rows),
            violation_rate=float(np.mean([row['violation'] for row in rows])),
            mean_center_deviation=float(np.mean([row['center_deviation'] for row in rows
                                                 if row['center_deviation'] is not None])) if grades else None,
            mean_exceedance=float(np.mean([row['exceedance'] for row in rows
                                           if row['exceedance'] is not None])) if grades else None,
            cost=_distribution([row['continuous_cost'] for row in rows])))
    valid_windows=[window for row in physical for window in row['windows'] if not window['empty']]
    summary=dict(n_shifts=len(records),windows=per_window,
        joint_four_window_compliance_rate=float(np.mean([row['all_four_windows_compliant'] for row in physical])),
        ore_target_met_rate=float(np.mean([row['ore_target_met'] for row in physical])),
        waste_target_met_rate=float(np.mean([row['waste_target_met'] for row in physical])),
        both_targets_met_rate=float(np.mean([row['both_targets_met'] for row in physical])),
        complete_task_rate=float(np.mean([row['complete_task'] for row in physical])),
        production_shortfall_shifts=sum(not row['both_targets_met'] for row in physical),
        valid_grade_windows=len(valid_windows),empty_grade_windows=4*len(records)-len(valid_windows),
        mean_center_deviation=float(np.mean([row['center_deviation'] for row in valid_windows])) if valid_windows else None,
        mean_exceedance=float(np.mean([row['exceedance'] for row in valid_windows])) if valid_windows else None,
        vehicle_utilization=None,queue_statistics=None,
        coverage_gaps=['per_vehicle_busy_time_not_recorded','time_weighted_queue_lengths_not_recorded'])
    summary['decision_sampled_queues']=_combined_queue_statistics(episodes,physical)
    for key in ('ore_tons','waste_tons','total_wait_seconds','fleet_wait_hours','grade_cost','production_cost','waiting_cost','combined_cost','training_reward'):
        summary[key]=_distribution([row[key] for row in physical])
    summary['mean_wait_per_vehicle_seconds']=_distribution([row['total_wait_seconds']/9.0 for row in physical])
    worst=[max(w['exceedance'] for w in row['windows'] if w['exceedance'] is not None)
           for row in physical if any(w['exceedance'] is not None for w in row['windows'])]
    summary['worst_window_exceedance']=_distribution(worst) if worst else None
    summary['waiting_denominator']='all_9_vehicles_per_complete_shift_not_per_request'
    grades=[row['shift_grade'] for row in physical if row['shift_grade'] is not None]
    summary['shift_grade']=_distribution(grades) if grades else None
    summary['shift_grade_compliant_rate']=float(np.mean([row['shift_grade_compliant'] for row in physical]))
    total_q=math.fsum(w['tons'] for row in physical for w in row['windows'])
    summary['pooled_tonnage_weighted_shift_grade']=(math.fsum(row['shift_metal_tons'] for row in physical)/total_q
                                                    if total_q>0 else None)
    rmse=_grade_rmse(physical)
    summary['grade_rmse']=rmse
    summary['grade_rmse_all_valid_windows']=rmse['overall_valid_windows']['rmse']
    summary['grade_rmse_complete_shift']=rmse['processed_tonnage_shift_grade']['rmse']
    summary['overall_window_compliance_rate']=float(np.mean([
        window['compliant'] for row in physical for window in row['windows']]))
    summary['per_window_compliance']=[dict(window=i,
        compliant_shifts=sum(row['windows'][i]['compliant'] for row in physical),
        total_shifts=len(physical),
        compliance_rate=float(np.mean([row['windows'][i]['compliant'] for row in physical])),
        empty_windows=sum(row['windows'][i]['empty'] for row in physical)) for i in range(4)]
    summary['evaluation_thresholds']=dict(grade_lower=.26,grade_upper=.28,
        grade_epsilon=GRADE_EPSILON,grade_compliance='frozen_RPC_grade_bounds_with_grade_epsilon',
        ore_target_tons=16000.,waste_target_tons=4000.,
        production_primary='strict_greater_or_equal_target',tonnage_epsilon=TONNAGE_EPSILON,
        tonnage_epsilon_used_for_primary_production=False,
        waste_dispatch_stop_tons=6000.,waste_dispatch_stop_is_production_target=False)
    summary['epsilon_compatibility_disagreement_shifts']=sum(
        row['ore_target_met']!=row['epsilon_ore_target_met'] or
        row['waste_target_met']!=row['epsilon_waste_target_met'] for row in physical)
    summary['epsilon_ore_target_met_rate']=float(np.mean([row['epsilon_ore_target_met'] for row in physical]))
    summary['epsilon_waste_target_met_rate']=float(np.mean([row['epsilon_waste_target_met'] for row in physical]))
    summary['grade_epsilon_boundary_disagreement_windows']=sum(
        w['compliant']!=w['strict_band_compliant'] for row in physical for w in row['windows'])
    summary['waiting_evaluation_scope']='all_complete_evaluation_shifts_including_production_or_grade_failures'
    values = None if not value_stats else _aggregate_critic_metrics(value_stats)
    headline=dict(complete_evaluation_shifts=len(records),decision_count=sum(row['transitions'] for row in records),
        grade_rmse=rmse['overall_valid_windows']['rmse'],
        grade_rmse_valid_windows=rmse['overall_valid_windows']['valid_count'],
        grade_rmse_total_windows=rmse['overall_valid_windows']['total_count'],
        empty_windows=rmse['overall_valid_windows']['empty_count'],
        complete_shift_grade_rmse=rmse['processed_tonnage_shift_grade']['rmse'],
        joint_four_window_compliance_rate=summary['joint_four_window_compliance_rate'],
        all_four_compliant_shifts=sum(row['all_four_windows_compliant'] for row in physical),
        overall_window_compliance_rate=summary['overall_window_compliance_rate'],
        per_window_compliance=summary['per_window_compliance'],
        ore_target_met_rate=summary['ore_target_met_rate'],
        waste_target_met_rate=summary['waste_target_met_rate'],
        complete_task_rate=summary['complete_task_rate'],
        ore_target_met_shifts=sum(row['ore_target_met'] for row in physical),
        waste_target_met_shifts=sum(row['waste_target_met'] for row in physical),
        complete_task_shifts=sum(row['complete_task'] for row in physical),
        production_shortfall_shifts=summary['production_shortfall_shifts'],
        mean_ore_tons=summary['ore_tons']['mean'],mean_waste_tons=summary['waste_tons']['mean'],
        mean_total_wait_seconds=summary['total_wait_seconds']['mean'],
        total_wait_seconds=math.fsum(row['total_wait_seconds'] for row in physical),
        mean_fleet_wait_hours=summary['fleet_wait_hours']['mean'],
        mean_wait_per_vehicle_seconds=summary['mean_wait_per_vehicle_seconds']['mean'],
        critic_mc_mse_pooled=None if values is None else values['pooled']['mse'],
        critic_mc_mse_episode_mean=None if values is None else values['episode_mean']['mse'],
        critic_mc_explained_variance_pooled=None if values is None else values['pooled']['explained_variance'],
        critic_mc_explained_variance_episode_mean=None if values is None else values['episode_mean']['explained_variance'])
    entropy = None if not entropies else dict(state_mean=float(np.concatenate(entropies).mean()),
                                            complete_shift_mean=float(np.mean([row.mean() for row in entropies])))
    return dict(schema=FORMAT,source_kind=episodes[0]['identity']['source_kind'],
        role='independent_evaluation',training_state_unchanged=True,optimizer_updates=0,gradient_updates=0,
        fingerprints=None if model is None else dict(actor=digest(model.actor),critic=digest(model.critic)),
        n_shifts=len(records),n_states=sum(record['transitions'] for record in records),
        per_episode=records,metrics=summary,
        headline_metrics=headline,grade_rmse=rmse,
        reward_version=REWARD_VERSION,checkpoint_provenance=checkpoint_provenance,
        summary=summary,critic_mc=values,policy_entropy=entropy,episodes=records,
        evaluation_optimizer_updates=0,actor_updates=0,critic_updates=0,
        waiting_in_training_targets=True,production_in_training_targets=True,
        composite_contract=composite_contract(),
        ppo_diagnostics=dict(kl=None,clip_fraction=None,reason='evaluation_has_no_policy_update'),
        production_guarantee='pending')


def offline_observation_check(model, restored, episodes):
    """Compare one frozen post-update model and its restore on already captured raw.

    These are training observations, not an independent policy evaluation. No
    simulator callback, action sample, optimizer step or extra shift is created.
    """
    episodes=tuple(episodes)
    _require(bool(episodes),'offline restore check needs captured complete shifts')
    rows=[]
    with ReadOnlyTrainingGuard(model), ReadOnlyTrainingGuard(restored):
        for episode in episodes:
            raw=np.asarray(episode['states'])
            with torch.no_grad():
                before=replay_policy(model.actor,raw[:-1])['probabilities']
                after=replay_policy(restored.actor,raw[:-1])['probabilities']
                values=model.predict_values(episode)
                restored_values=restored.predict_values(episode)
            _require(torch.equal(before,after) and np.array_equal(values,restored_values),
                     'checkpoint restored predictions differ on captured observations')
            rows.append(dict(run_index=episode['identity']['run_index'],
                raw_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
                decisions=len(raw)-1,policy_probabilities_bitwise_equal=True,
                value_predictions_bitwise_equal=True,true_terminal_value=float(values[-1]),
                physical=_physical(episode)))
    return dict(schema='single-critic-task-query-captured-observation-restore-v1',
        role='engineering_offline_training_observation_consistency',
        independent_evaluation=False,additional_physical_shifts=0,
        action_samples=0,optimizer_updates=0,gradient_updates=0,
        training_state_unchanged=True,checkpoint_update=model.update_index,
        fingerprints=dict(actor=digest(model.actor),critic=digest(model.critic)),episodes=rows)
