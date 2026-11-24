"""Compare original and grade-aware dispatch on the same sealed offline frames."""
import argparse
from collections import Counter
import json
from math import fsum
from pathlib import Path
from statistics import mean
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
FLOW = HERE.parent
if str(FLOW) not in sys.path:
    sys.path.insert(0, str(FLOW))

from flow_matching_campaign import METHOD, OUTPUT, REPORT, load_reference, reference
from change_plans.token_attention_001.business_graph_summary_009.baseline_comparison_001.dispatch_flow_matching_001.review_offline import (
    _choose, _plan_record, _policy, sealed_frames,
)
from traditional_dispatch_v1.runtime import _publish_json, _require


MIN_GRADE = .26
TARGET_GRADE = .27
MAX_GRADE = .28
GRADE_TOLERANCE = 1e-9
POLICY_OPTIONS = ('planning_ms', 'replan_seconds', 'match_trucks', 'soft_replan_seconds')


def _historical_reference():
    """Summarize the earlier five-shift campaign through its completed receipt seal."""
    completion = reference(OUTPUT / 'suite_complete.json')
    concluded = load_reference(completion)
    evaluation = concluded['evaluation']
    _require(evaluation == reference(REPORT), 'historical five-shift report reference differs')
    report = load_reference(evaluation)
    historical_suite = load_reference(concluded['suite'])
    _require(concluded['status'] == report['status'] == 'COMPLETE'
             and report['campaign_id'] == concluded['campaign_id']
             and report['physical_shifts'] == len(report['shifts']) == 5,
             'five historical physical shifts are not fully reconciled')
    shift_rows, branches = [], Counter()
    for shift, receipt in zip(report['shifts'], report['receipts'], strict=True):
        _require(load_reference(receipt) == shift, 'reported shift differs from its sealed receipt')
        physical = shift['physical']
        matching = shift['engineering']['route_matching']
        branches.update(matching['branch_counts'])
        shift_rows.append(dict(
            run_index=shift['run_index'],
            complete_task=physical['complete_task'],
            ore_target_met=physical['ore_target_met'],
            waste_target_met=physical['waste_target_met'],
            windows=[dict(window=row['window'], grade=row['grade'], ore_tons=row['tons'],
                          metal_tons=row['metal_tons'], compliant=row['compliant'])
                     for row in physical['windows']],
            total_effective_planned_dispatch_tons=fsum(row['planned_tons'] for row in matching['route_periods']),
            total_confirmed_dispatch_tons=fsum(row['confirmed_tons'] for row in matching['route_periods']),
            absolute_flow_error_tons=matching['absolute_flow_error_tons'],
            first_forecast_grade_by_window={
                window: row['metal_tons'] / row['ore_tons'] if row['ore_tons'] else None
                for window, row in matching['first_forecast_by_window'].items()
            },
        ))
    windows = [window for row in shift_rows for window in row['windows']]
    by_window = []
    for index in range(4):
        results = [window for window in windows if window['window'] == index]
        observed_grades = [window['grade'] for window in results if window['grade'] is not None]
        by_window.append(dict(window=index, failed=sum(not window['compliant'] for window in results),
                              mean_grade=mean(observed_grades) if observed_grades else None))
    return dict(
        completion=completion,
        historical_suite=concluded['suite'],
        frozen_policy_sources=[
            source for source in historical_suite['source_pins']
            if Path(source['path']).name in ('flow_matching_policy.py', 'flow_matching_plan.py')
        ],
        evaluation=evaluation,
        physical_shifts=len(shift_rows),
        complete_task_shifts=sum(row['complete_task'] for row in shift_rows),
        engineering_passed_shifts=report['engineering_passed_shifts'],
        ore_target_met_shifts=sum(row['ore_target_met'] for row in shift_rows),
        waste_target_met_shifts=sum(row['waste_target_met'] for row in shift_rows),
        windows=len(windows),
        failed_windows=sum(not window['compliant'] for window in windows),
        over_max_windows=sum(window['grade'] is not None and window['grade'] > MAX_GRADE for window in windows),
        under_min_windows=sum(window['grade'] is not None and window['grade'] < MIN_GRADE for window in windows),
        mean_combined_cost=report['mean_combined_cost'],
        branch_counts=dict(branches),
        mean_effective_planned_dispatch_tons=mean(row['total_effective_planned_dispatch_tons'] for row in shift_rows),
        mean_confirmed_dispatch_tons=mean(row['total_confirmed_dispatch_tons'] for row in shift_rows),
        by_window=by_window,
        shifts=shift_rows,
    )


def _on_frame(factory, observation, run, frame):
    policy = factory()
    choice, detail = _choose(policy, observation, run, frame)
    plan = _plan_record(policy.plan)
    # Confirm exactly the recommendation on this isolated frame to verify the
    # dispatch ledger, without advancing an archive on a counterfactual action.
    policy.commit(choice)
    ledger = policy.summary()
    _require(sum(ledger['branch_counts'].values()) == 1,
             f'expected one confirmed offline action at {run}/{frame}')
    _require(abs(fsum(row['confirmed_tons'] for row in ledger['route_periods']) - choice.assigned_tons) < 1e-6,
             f'confirmed offline dispatch mass differs at {run}/{frame}')
    if plan is not None:
        detail['plan'] = {key: value for key, value in plan.items() if key != 'routes'}
        detail['plan']['positive_path_count'] = len(plan['routes'])
        detail['plan']['total_planned_dispatch_tons'] = fsum(row['planned_tons'] for row in plan['routes'])
        planning_grade_slack = getattr(policy.plan, 'planning_grade_slack_tons', None)
        if planning_grade_slack is not None:
            detail['plan']['planning_grade_slack_tons'] = planning_grade_slack
    else:
        detail['plan'] = None
    detail['grade_guard'] = ledger.get('grade_control', {})
    return detail


def _grade_statistics(rows):
    plans = [row['plan'] for row in rows if row['plan'] is not None]
    times = [row['elapsed_ms'] for row in rows]
    grades = [window['grade'] for plan in plans for window in plan['windows'] if window['grade'] is not None]
    guard_outcomes, guard_assessments = Counter(), Counter()
    for row in rows:
        guard_outcomes.update(row['grade_guard'].get('corrections_by_original_branch', {}))
        guard_assessments.update(row['grade_guard'].get('assessments_by_reason', {}))
    numeric_compliance = sum(
        bool(plan['windows']) and all(window['grade'] is not None
                                     and MIN_GRADE - GRADE_TOLERANCE <= window['grade'] <= MAX_GRADE + GRADE_TOLERANCE
                                     for window in plan['windows'])
        for plan in plans
    )
    return dict(
        frames=len(rows),
        fallback_frames=sum(row['kind'] == 'fallback' for row in rows),
        fallback_reasons=dict(Counter(row['fallback_reason'] for row in rows if row['fallback_reason'])),
        planning_timeouts=sum(row['timed_out'] for row in rows),
        planning_errors=dict(Counter(row['plan_error'] for row in rows if row['plan_error'])),
        cold_planner_latency_p95_ms=float(np.quantile(times, .95)),
        cold_planner_latency_max_ms=max(times),
        over_planning_budget_frames=sum(t > METHOD['planning_ms'] for t in times),
        over_rpc_target_frames_offline_only=sum(t > METHOD['response_target_ms'] for t in times),
        matching_outcomes=dict(Counter(row['matching_kind'] for row in rows if row['matching_kind'])),
        grade_guard_corrections_by_original_branch=dict(guard_outcomes),
        grade_guard_assessments_by_reason=dict(guard_assessments),
        grade_guard_corrected_frames=sum(row['matching_kind'] == 'grade_correction' for row in rows),
        plans=len(plans),
        plans_with_positive_path_flow=sum(plan['positive_path_count'] > 0 for plan in plans),
        plans_with_ore_shortfall=sum(plan['ore_shortfall_tons'] > 1e-6 for plan in plans),
        plans_with_waste_shortfall=sum(plan['waste_shortfall_tons'] > 1e-6 for plan in plans),
        nominally_compliant_remaining_window_plans=numeric_compliance,
        modeled_windows_with_grade=len(grades),
        modeled_windows_touching_or_over_upper_limit=sum(g >= MAX_GRADE - GRADE_TOLERANCE for g in grades),
        modeled_windows_outside_physical_limits=sum(
            not MIN_GRADE - GRADE_TOLERANCE <= g <= MAX_GRADE + GRADE_TOLERANCE for g in grades
        ),
        mean_modeled_absolute_grade_error=mean(abs(g - TARGET_GRADE) for g in grades) if grades else None,
    )


def review():
    if __package__:
        from .grade_policy import GradeControlPolicy
    else:
        from grade_policy import GradeControlPolicy

    def revised_factory():
        return GradeControlPolicy(**{key: METHOD[key] for key in POLICY_OPTIONS})

    revised_paths = [HERE / name for name in ('grade_policy.py', 'grade_flow_plan.py', 'grade_guard.py')]
    revised_sources = [reference(path) for path in revised_paths]
    samples = []
    names = ('original', 'grade_control')
    factories = dict(zip(names, (_policy, revised_factory), strict=True))
    for ordinal, (run, frame, seal, observation) in enumerate(sealed_frames()):
        order = names if ordinal % 2 == 0 else names[::-1]
        decisions = {name: _on_frame(factories[name], observation, run, frame) for name in order}
        matching_sha = {value['observation_sha256'] for value in decisions.values()}
        _require(len(matching_sha) == 1, 'policies did not receive identical observation bytes')
        samples.append(dict(run_index=run, frame=frame, archive_sha256=seal,
                            observation_sha256=next(iter(matching_sha)), decisions=decisions))
    _require(len(samples) == 24 and len({row['run_index'] for row in samples}) == 3,
             'expected 24 sealed decisions across three archived nominal shifts')
    openings = [row for row in samples if row['frame'] == 0]
    _require(len(openings) == 3, 'all archived initial plans must be represented')
    physical_baseline = _historical_reference()
    original_sources = [reference(FLOW / name) for name in ('flow_matching_policy.py', 'flow_matching_plan.py')]
    _require(len(physical_baseline['frozen_policy_sources']) == len(original_sources)
             and all(source in physical_baseline['frozen_policy_sources'] for source in original_sources),
             'offline baseline strategy differs from frozen physical baseline bytes')
    _require(revised_sources == [reference(path) for path in revised_paths],
             'grade-control source bytes changed during offline replay')
    return dict(
        schema='bg128-dispatch-flow-grade-control-offline-v1',
        scope='same archived observations with isolated decisions and local ACKs',
        physical_shifts_started=0,
        grade_unit='mass_fraction',
        mass_unit='metric_tons',
        archived_frames=len(samples),
        distinct_observations=len({row['observation_sha256'] for row in samples}),
        planning_budget_ms=METHOD['planning_ms'],
        source_references=dict(review=reference(__file__), original=original_sources, grade_control=revised_sources),
        physical_baseline=physical_baseline,
        arms={name: _grade_statistics([row['decisions'][name] for row in samples]) for name in names},
        differing_actions=sum(row['decisions']['original']['action'] != row['decisions']['grade_control']['action']
                              for row in samples),
        initial_planned_windows=[
            dict(run_index=row['run_index'],
                 plans={name: (decision['plan'] or {}).get('windows', [])
                        for name, decision in row['decisions'].items()})
            for row in openings
        ],
        rows=samples,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Exclusive JSON output path')
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error('the output path already exists')
    result = review()
    if args.output is None:
        print(json.dumps(result, allow_nan=False, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(_publish_json(args.output, result), ensure_ascii=False))
