"""Summarize completed MILP shifts by their matched environment-seed slots."""

from collections import Counter
from statistics import median

import numpy as np

from change_plans.token_attention_001.business_graph_summary_009.baseline_comparison_001.paired_v2_100_001.formal_report import (
    paired_report,
)
from traditional_dispatch_v1.runtime import _require
from traditional_dispatch_v2.runtime import _verify_anchor


SLOTS = 10
POLICIES = ('rolling_quota', 'trip_fifo')


def _merged_counts(rows, key):
    counts = Counter()
    for row in rows:
        counts.update(row[key])
    return dict(counts)


def _arm(records):
    result = paired_report._summarize_arm(records)
    engineering = [row['engineering'] for row in records]
    gaps = [row['max_mip_gap'] for row in engineering if row['max_mip_gap'] is not None]
    time_limits = sum(row['solver_time_limit_requests'] for row in engineering)
    return dict(
        result,
        all_four_windows_compliant_shifts=sum(row['physical']['all_four_windows_compliant'] for row in records),
        ore_target_met_shifts=sum(row['physical']['ore_target_met'] for row in records),
        waste_target_met_shifts=sum(row['physical']['waste_target_met'] for row in records),
        fallback_reasons=_merged_counts(engineering, 'fallback_reasons'),
        solver_status_counts=_merged_counts(engineering, 'solver_status_counts'),
        solver_time_limit_requests=time_limits,
        planning_deadline_fraction=time_limits / result['decisions'],
        max_mip_gap=max(gaps, default=None),
        accepted_milp_requests=sum(row['solver']['mip_accepted'] for row in engineering),
        accepted_joint_milp_requests=sum(row['solver'].get('joint_plan_commits', 0) for row in engineering),
        policy_shift_p95_median_ms=median(row['policy_latency_ms']['p95'] for row in engineering),
        policy_max_ms=max(row['policy_latency_ms']['max'] for row in engineering),
    )


def _pair(ordinal, quota, trip):
    quota_physical, trip_physical = (row['physical'] for row in (quota, trip))
    return dict(
        pair_ordinal=ordinal,
        environment_seed=quota['experiment_seed'],
        runs={row['policy_kind']: row['run_index'] for row in (quota, trip)},
        complete_task={row['policy_kind']: row['physical']['complete_task'] for row in (quota, trip)},
        difference={key: float(trip_physical[key]) - float(quota_physical[key]) for key in paired_report.METRICS},
        windows=[
            dict(window=window, rolling_quota_grade=q['grade'], trip_fifo_grade=t['grade'],
                 rolling_quota_compliant=q['compliant'], trip_fifo_compliant=t['compliant'])
            for window, (q, t) in enumerate(zip(quota_physical['windows'], trip_physical['windows'], strict=True))
        ],
    )


def aggregate(shifts):
    """Use all twenty complete shifts and resample at the matched-slot level."""
    _require(len(shifts) == SLOTS * len(POLICIES), 'all twenty paired MILP shifts are required')
    for field in ('run_index', 'recording_id'):
        _require(len({row[field] for row in shifts}) == len(shifts), f'MILP physical identity repeats: {field}')
    groups = {name: [] for name in POLICIES}
    seeds, pairs = set(), []
    for ordinal, (quota, trip) in enumerate(zip(shifts[::2], shifts[1::2], strict=True)):
        for row, name in zip((quota, trip), POLICIES, strict=True):
            _require((row['pair_ordinal'], row['policy_kind'], row['scene_id']) == (ordinal, name, 'nominal'),
                     'invalid MILP slot, strategy order or scenario')
            _require(row['status'] == 'COMPLETE' and row['normal_completion'] is True
                     and row['completed_shifts'] == 1 and row['optimizer_updates'] == 0
                     and row['actor_sampling_seed'] == 0, 'paired MILP analysis requires a complete deterministic shift')
            groups[name].append(row)
        _verify_anchor(trip, quota)
        _require(quota['experiment_seed'] not in seeds, 'MILP environment seed repeated between slots')
        seeds.add(quota['experiment_seed'])
        pairs.append(_pair(ordinal, quota, trip))

    draws = np.random.default_rng(paired_report.ANALYSIS_SEED).integers(
        0, SLOTS, size=(paired_report.BOOTSTRAP_REPLICATES, SLOTS))
    contrast = paired_report._matched_comparison(groups, 'trip_fifo', 'rolling_quota', draws)
    contrast['primary'] = True
    return dict(
        evaluation_slots=SLOTS,
        matched_initial_state_and_rpc=True,
        groups={name: _arm(records) for name, records in groups.items()},
        engineering_acceptance={
            name: dict(passed=sum(row['engineering_acceptance']['passed'] for row in records), shifts=len(records))
            for name, records in groups.items()
        },
        primary_outcome='mean_paired_combined_cost_trip_fifo_minus_rolling_quota',
        comparison=contrast,
        pairs=pairs,
        intervals=dict(method='percentile_bootstrap_over_environment_seed_slots',
                       replicates=paired_report.BOOTSTRAP_REPLICATES, seed=paired_report.ANALYSIS_SEED, level=.95),
    )
