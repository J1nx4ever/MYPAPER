"""Formal final evaluation metrics over source-admitted complete shifts.

The frozen evaluator remains responsible for material/target/Actor validation.
This module adds explicit grade RMSE denominators and compact inference timing;
it neither selects checkpoints nor collects shifts or updates a network.
"""
from __future__ import annotations

import math
import re

from baseline1_v1.evaluation import _distribution
from relation_gat_ego_v1.contract import CANDIDATES
from relation_gat_ego_v1.evaluation import evaluate_episodes as _frozen_evaluate

FORMAT='relation-gat-ego-formal-joint-evaluation-v1'
TIMING_SCHEMA='relation-gat-ego-formal-inference-timing-v1'
TIMING_SCOPE='actor_decision_path_excluding_visualization_rpc_and_simulation'
TARGET_GRADE=.27
_TIMING_FIELDS={'schema','scope','run_index','candidate_id','policy_version','decision_count',
                'total_seconds','sum_squared_seconds','min_seconds','max_seconds'}


def _require(condition,message):
    if not condition:raise ValueError(message)


def _number(value):
    return type(value) in (int,float) and math.isfinite(value) and value>=0


def _valid_identity(run_index,candidate_id,policy_version,decision_count):
    _require(type(run_index) is int and run_index>=0 and candidate_id in CANDIDATES and
             isinstance(policy_version,str) and re.fullmatch('[0-9a-f]{64}',policy_version) is not None and
             type(decision_count) is int and decision_count>0,'invalid inference timing identity/count')


def validate_inference_timing(record,*,run_index,candidate_id,policy_version,decision_count):
    """Bind one compact measured shift record to its admitted policy decisions."""
    _valid_identity(run_index,candidate_id,policy_version,decision_count)
    _require(isinstance(record,dict) and set(record)==_TIMING_FIELDS,'inference timing fields differ')
    _require(record['schema']==TIMING_SCHEMA and record['scope']==TIMING_SCOPE,'inference timing scope differs')
    _valid_identity(record['run_index'],record['candidate_id'],record['policy_version'],record['decision_count'])
    _require(all(record[key]==expected for key,expected in dict(run_index=run_index,candidate_id=candidate_id,
        policy_version=policy_version,decision_count=decision_count).items()),'inference timing identity/count differs')
    total,square,lo,hi=(record[key] for key in ('total_seconds','sum_squared_seconds','min_seconds','max_seconds'))
    _require(all(_number(value) for value in (total,square,lo,hi)),'nonfinite/negative inference timing')
    n=decision_count
    _require(lo<=hi,'inference timing extrema reversed')
    # Check moment feasibility without manufacturing per-request durations.
    lower_total,upper_total=n*lo,n*hi
    lower_square=total*(total/n);upper_square=hi*total
    _require(all(math.isfinite(value) for value in (lower_total,upper_total,lower_square,upper_square)),
             'inference timing moment overflow')
    tol=1e-12*max(abs(total),abs(lower_total),abs(upper_total),1e-12)
    tol_square=1e-12*max(abs(square),abs(lower_square),abs(upper_square),1e-24)
    _require(lower_total-tol<=total<=upper_total+tol and lower_square-tol_square<=square<=upper_square+tol_square,
             'inference timing moments inconsistent')
    if n==1:
        _require(math.isclose(lo,hi,rel_tol=1e-12,abs_tol=1e-24) and
                 math.isclose(total,lo,rel_tol=1e-12,abs_tol=1e-24),
                 'single-decision timing extrema differ')
    return dict(record)


class InferenceTimingAccumulator:
    """O(1) memory. Call observe after sampling, before optional visualization."""
    def __init__(self):
        self.decision_count=0
        self.total_seconds=0.
        self.sum_squared_seconds=0.
        self.min_seconds=None
        self.max_seconds=None

    def observe(self,elapsed_seconds):
        _require(_number(elapsed_seconds),'inference elapsed time must be finite nonnegative seconds')
        value=float(elapsed_seconds)
        total=self.total_seconds+value;square=self.sum_squared_seconds+value*value
        _require(math.isfinite(total) and math.isfinite(square),'inference timing accumulator overflow')
        self.decision_count+=1;self.total_seconds=total;self.sum_squared_seconds=square
        self.min_seconds=value if self.min_seconds is None else min(self.min_seconds,value)
        self.max_seconds=value if self.max_seconds is None else max(self.max_seconds,value)

    def record(self,*,run_index,candidate_id,policy_version):
        record=dict(schema=TIMING_SCHEMA,scope=TIMING_SCOPE,run_index=run_index,candidate_id=candidate_id,
                    policy_version=policy_version,decision_count=self.decision_count,
                    total_seconds=self.total_seconds,sum_squared_seconds=self.sum_squared_seconds,
                    min_seconds=self.min_seconds,max_seconds=self.max_seconds)
        return validate_inference_timing(record,run_index=run_index,candidate_id=candidate_id,
                                         policy_version=policy_version,decision_count=self.decision_count)


def aggregate_inference_timing(episodes):
    """Measured policy-call timing; no RPC, simulation, or training throughput claim."""
    episodes=tuple(episodes)
    _require(bool(episodes),'empty timing episode inventory')
    present=['policy_inference_timing' in episode for episode in episodes]
    if not any(present):
        _require(all(e['identity']['source_kind']=='synthetic' for e in episodes),
                 'real final evaluation requires measured inference timing for every shift')
        return dict(status='NOT_RECORDED',scope=TIMING_SCOPE,complete_shifts=len(episodes),
                    decision_count=sum(len(e['actions']) for e in episodes),
                    reason='synthetic numerical fixture has no measured collection latency')
    _require(all(present),'partial inference timing inventory')
    records=[]
    for episode in episodes:
        identity=episode['identity']
        records.append(validate_inference_timing(episode['policy_inference_timing'],
            run_index=identity['run_index'],candidate_id=identity['candidate_id'],
            policy_version=identity['policy_version'],decision_count=len(episode['actions'])))
    _require(len({r['run_index'] for r in records})==len(records),'duplicate inference timing run')
    _require(len({(r['candidate_id'],r['policy_version']) for r in records})==1,'mixed inference timing policy')
    records.sort(key=lambda r:r['run_index'])
    n=sum(r['decision_count'] for r in records)
    total=math.fsum(r['total_seconds'] for r in records)
    squares=math.fsum(r['sum_squared_seconds'] for r in records)
    _require(math.isfinite(total) and math.isfinite(squares),'aggregate inference timing overflow')
    mean=total/n;centered=squares-total*mean
    tolerance=1e-12*max(abs(squares),abs(total*mean),1e-24)
    _require(math.isfinite(centered) and centered>=-tolerance,'aggregate inference timing variance invalid')
    sample_sd=math.sqrt(max(0.,centered)/(n-1)) if n>1 else None
    return dict(status='RECORDED',schema=TIMING_SCHEMA,scope=TIMING_SCOPE,complete_shifts=len(records),
        decision_count=n,total_seconds=total,
        decision_weighted=dict(n=n,mean_seconds=mean,sample_sd_seconds=sample_sd,
            min_seconds=min(r['min_seconds'] for r in records),max_seconds=max(r['max_seconds'] for r in records),
            p50_seconds=None,p95_seconds=None,
            quantile_reason='per-request durations are not retained; exact request quantiles are not recoverable'),
        complete_shift_equal_mean_seconds=_distribution([r['total_seconds']/r['decision_count'] for r in records]),
        per_complete_shift_total_seconds=_distribution([r['total_seconds'] for r in records]),
        per_complete_shift=records,
        measurement_notes='Includes validation, cache/C preparation, neural Actor inference and stochastic action sampling; excludes RPC, simulator time, visualization and optimizer work.')


def _rmse(grades,total_count):
    _require(type(total_count) is int and total_count>=0 and len(grades)<=total_count,'invalid RMSE denominator')
    _require(all(type(value) in (int,float) and math.isfinite(value) and 0<=value<=1 for value in grades),
             'invalid grade in RMSE')
    squared=math.fsum((grade-TARGET_GRADE)**2 for grade in grades)
    return dict(rmse=math.sqrt(squared/len(grades)) if grades else None,
                sum_squared_error=squared,valid_count=len(grades),total_count=total_count,
                empty_count=total_count-len(grades))


def _grade_rmse(physical):
    """Equal-window RMSE is distinct from RMSE of processed-tonnage shift grades."""
    _require(bool(physical),'grade RMSE requires complete shifts')
    for row in physical:
        _require([w['window'] for w in row['windows']]==[0,1,2,3],'RMSE requires four ordered windows')
        for w in row['windows']:
            _require(type(w['empty']) is bool and w['empty']==(w['grade'] is None),'empty grade must be None')
            _require(not w['empty'] or (w['compliant'] is False and w['violation'] is True),
                     'empty window must remain noncompliant')
    per_window=[]
    for i in range(4):
        grades=[row['windows'][i]['grade'] for row in physical if row['windows'][i]['grade'] is not None]
        per_window.append(dict(window=i,**_rmse(grades,len(physical))))
    grades=[w['grade'] for row in physical for w in row['windows'] if w['grade'] is not None]
    shift_grades=[row['shift_grade'] for row in physical if row['shift_grade'] is not None]
    return dict(target_grade=TARGET_GRADE,unit='grade_fraction',window_weighting='equal_valid_windows_not_tonnage_weighted',
        empty_window_policy='excluded_from_RMSE_valid_denominator_but_counted_as_noncompliant',
        per_window=per_window,overall_valid_windows=_rmse(grades,4*len(physical)),
        processed_tonnage_shift_grade=_rmse(shift_grades,len(physical)),
        shift_grade_definition='Each shift grade is processed metal / processed tons; the shift RMSE weights valid complete shifts equally.')


def evaluate_episodes(episodes,model=None,*,expected_complete_shifts=None,
                      checkpoint_provenance=None,training_environment=None):
    """Extend the frozen strict evaluator without weakening source/label checks."""
    episodes=tuple(episodes)
    report=_frozen_evaluate(episodes,model,expected_complete_shifts=expected_complete_shifts,
        checkpoint_provenance=checkpoint_provenance,training_environment=training_environment)
    physical=[record['physical'] for record in report['per_episode']]
    rmse=_grade_rmse(physical);timing=aggregate_inference_timing(episodes)
    report['inherited_schema']=report['schema'];report['schema']=FORMAT
    report['grade_rmse']=report['summary']['grade_rmse']=rmse
    report['policy_inference_timing']=report['summary']['policy_inference_timing']=timing
    # These counts include failed-production shifts and empty/noncompliant
    # windows. RMSE's valid-only denominator never replaces compliance's one.
    report['headline_metrics']=dict(complete_evaluation_shifts=len(episodes),decision_count=report['n_states'],
        compliant_windows=sum(w['compliant'] for row in physical for w in row['windows']),
        window_compliance_denominator=4*len(episodes),empty_windows=rmse['overall_valid_windows']['empty_count'],
        overall_window_compliance_rate=sum(w['compliant'] for row in physical for w in row['windows'])/(4*len(episodes)),
        per_window_compliance=[dict(window=i,compliant_shifts=sum(row['windows'][i]['compliant'] for row in physical),
            total_shifts=len(episodes),compliance_rate=sum(row['windows'][i]['compliant'] for row in physical)/len(episodes),
            empty_windows=sum(row['windows'][i]['empty'] for row in physical)) for i in range(4)],
        all_four_compliant_shifts=sum(row['all_four_windows_compliant'] for row in physical),
        ore_target_met_shifts=sum(row['ore_target_met'] for row in physical),
        waste_target_met_shifts=sum(row['waste_target_met'] for row in physical),
        both_production_targets_met_shifts=sum(row['both_targets_met'] for row in physical),
        complete_task_shifts=sum(row['complete_task'] for row in physical),
        production_shortfall_shifts=report['summary']['production_shortfall_shifts'],
        mean_ore_tons=report['summary']['ore_tons']['mean'],mean_waste_tons=report['summary']['waste_tons']['mean'],
        mean_fleet_wait_hours=report['summary']['fleet_wait_hours']['mean'],
        mean_wait_per_vehicle_seconds=report['summary']['mean_wait_per_vehicle_seconds']['mean'],
        grade_rmse=rmse['overall_valid_windows']['rmse'],grade_rmse_valid_windows=rmse['overall_valid_windows']['valid_count'])
    report['training_seed_uncertainty']=dict(training_seed_count=1,evaluation_complete_shifts=len(episodes),
        variance_across_training_seeds=None,estimable=False,
        reason='One paired training initialization per arm. Evaluation environment/sampling seeds measure within-policy variability, not variance across independently trained policies.')
    return report
