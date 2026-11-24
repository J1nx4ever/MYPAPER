"""Bounded aggregation: numerical tail copied from the frozen official evaluator.

Input records already came from official single-episode evaluation. Only the
queue aggregation input and source-kind lookup change to avoid retaining tapes.
"""
import math
import numpy as np
from baseline1_v1.evaluation import _distribution
from relation_gat_ego_formal_v1.evaluation import _grade_rmse
from single_critic_task_query_v1.evaluation import (_aggregate_critic_metrics, GRADE_EPSILON, TONNAGE_EPSILON, FORMAT)
from single_critic_task_query_v1.targets import REWARD_VERSION,composite_contract
from single_critic_task_query_v1.models import digest

SOURCE_SHA256 = 'bff247b7007984c6b2b8cb43a4a2568f6908e9f9256dc1ee0e81118ec537ded6'

def combined_queues(samples, physical):
    sites=list(samples[0])
    return dict(time_weighted=False,sampling='existing queue occupancy at action decisions only',
        state_equal=dict(weighting='all_action_decisions; not time weighted',
            sites={site:_distribution(np.concatenate([sample[site] for sample in samples])) for site in sites}),
        complete_shift_equal=dict(weighting='equal complete shifts; distribution of per-shift decision means',
            sites={site:_distribution([row['decision_sampled_queues']['sites'][site]['mean']
                                      for row in physical]) for site in sites}))

def aggregate(records, queue_samples, entropies, model, checkpoint_provenance):
    value_stats=[row['value_statistics'] for row in records if 'value_statistics' in row]
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
    summary['decision_sampled_queues']=combined_queues(queue_samples,physical)
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
    return dict(schema=FORMAT,source_kind=records[0]['identity']['source_kind'],
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

