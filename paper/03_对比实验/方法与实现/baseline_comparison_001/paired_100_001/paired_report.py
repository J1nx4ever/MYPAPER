"""Summarize the frozen comparison by complete paired nominal shifts."""
from collections import Counter
import math

import numpy as np
from scipy.stats import binomtest

from paired_plan import STRATEGIES, require


BOOTSTRAP_REPLICATES = 10_000
ANALYSIS_SEED = 90221
METRICS = (
    'combined_cost', 'grade_cost', 'waiting_cost', 'production_cost',
    'total_wait_seconds', 'ore_tons', 'waste_tons',
    'complete_task', 'all_four_windows_compliant', 'ore_target_met', 'waste_target_met',
)
CONTRASTS = (
    ('bg128', 'rolling'),
    ('rolling', 'fifo_greedy'),
    ('fifo_greedy', 'unload_greedy'),
)


def _sample(records, key):
    return np.array([row['physical'][key] for row in records], dtype=np.float64)


def _window_summary(records, window):
    rows = [row['physical']['windows'][window] for row in records]
    require(all(row['window'] == window for row in rows), 'settled windows out of order')
    grades = [row['grade'] for row in rows if not row['empty']]
    return dict(window=window, compliant=sum(row['compliant'] for row in rows),
                empty=len(rows)-len(grades), valid_windows=len(grades),
                grade_rmse_pp=(100.0 * math.sqrt(sum((grade - .27)**2 for grade in grades) / len(grades))
                               if grades else None),
                mean_grade=(sum(grades) / len(grades) if grades else None),
                mean_tons=float(np.mean([row['tons'] for row in rows])))


def _summarize_arm(records):
    means = {key: float(_sample(records, key).mean()) for key in METRICS}
    engineering = [row['engineering'] for row in records]
    counts = Counter()
    for row in engineering:
        counts.update(row['action_counts'])
    decisions = sum(row['decisions'] for row in engineering)
    require(decisions > 0, 'empty physical decision inventory')
    return dict(
        shifts=len(records), mean=means, decisions=decisions, action_counts=dict(counts),
        complete_task_shifts=sum(row['physical']['complete_task'] for row in records),
        waiting_by_stage_mean_seconds={
            key: float(np.mean([row['physical']['wait_breakdown'][key] for row in records]))
            for key in ('load_seconds', 'dump_resource_queue_seconds', 'dump_capacity_admission_seconds')
        },
        windows=[_window_summary(records, window) for window in range(4)],
        rpc_shift_p95_median_ms=float(np.median([row['rpc_receive_to_response_ms']['p95'] for row in engineering])),
        rpc_max_ms=max(row['rpc_receive_to_response_ms']['max'] for row in engineering),
        rpc_over_target=sum(row['rpc_responses_over_target'] for row in engineering),
        fallback_rate=counts['fallback']/decisions,
        planning_deadline_fraction=counts['timed_out']/decisions,
    )


def _difference(first, second, draws):
    delta = first - second
    lo, hi = np.quantile(delta[draws].mean(axis=1), [.025, .975])
    return dict(mean=float(delta.mean()), paired_bootstrap_ci95=[float(lo), float(hi)],
                positive=int(np.count_nonzero(delta > 0)),
                negative=int(np.count_nonzero(delta < 0)), ties=int(np.count_nonzero(delta == 0)))


def _matched_comparison(groups, first, second, draws):
    lhs, rhs = groups[first], groups[second]
    metrics = {key: _difference(_sample(lhs, key), _sample(rhs, key), draws) for key in METRICS}
    complete = metrics['complete_task']
    better, worse = complete['positive'], complete['negative']
    complete['both_complete'] = sum(a['physical']['complete_task'] and b['physical']['complete_task']
                                    for a, b in zip(lhs, rhs, strict=True))
    complete['both_failed'] = complete['ties'] - complete['both_complete']
    complete['mcnemar_two_sided_p'] = float(binomtest(better, better+worse, p=.5).pvalue) if better+worse else 1.0
    windows = []
    for window in range(4):
        flags = [[float(row['physical']['windows'][window]['compliant']) for row in records]
                 for records in (lhs, rhs)]
        windows.append(dict(window=window, **_difference(np.array(flags[0]), np.array(flags[1]), draws)))
    return dict(first=first, second=second, effect_direction='first_minus_second',
                primary=first == 'bg128' and second == 'rolling',
                metrics=metrics, per_window_compliance=windows)


def summarize(shifts, *, suite):
    """The only analysis unit is the frozen environment seed shared within a slot."""
    expected = {(job['pair_ordinal'], job['policy_kind']): job for job in suite['jobs']}
    slots = suite['evaluation_slots']
    require(len(expected) == len(shifts) == slots * len(STRATEGIES), 'all four-arm pairs are required')
    matched = {}
    for row in shifts:
        key = (row['pair_ordinal'], row['policy_kind'])
        require(key in expected and key not in matched, 'unplanned or repeated comparison slot')
        job = expected[key]
        require(all(row[field] == job[field] for field in ('run_index', 'experiment_seed', 'recording_id')),
                'comparison receipt does not match the frozen run')
        matched[key] = row
    groups = {name: [matched[i, name] for i in range(slots)] for name in STRATEGIES}
    same_seed = [{matched[i, name]['experiment_seed'] for name in STRATEGIES} for i in range(slots)]
    require(all(len(values) == 1 for values in same_seed) and
            len(set.union(*same_seed)) == slots, 'comparison seeds are not paired only within each slot')
    draws = np.random.default_rng(ANALYSIS_SEED).integers(0, slots, size=(BOOTSTRAP_REPLICATES, slots))
    return dict(
        schema='bg128-traditional-paired100-comparison-v1', campaign_id=suite['campaign_id'],
        evaluation_slots=slots, physical_shifts=len(shifts), optimizer_updates=0,
        primary_outcome='mean_paired_combined_cost_bg128_minus_rolling',
        intervals=dict(method='percentile_bootstrap_over_environment_seed_slots',
                       replicates=BOOTSTRAP_REPLICATES, seed=ANALYSIS_SEED, level=.95,
                       secondary_contrasts='descriptive_intervals_without_multiplicity_adjustment'),
        groups={name: _summarize_arm(records) for name, records in groups.items()},
        comparisons=[_matched_comparison(groups, first, second, draws) for first, second in CONTRASTS],
    )
