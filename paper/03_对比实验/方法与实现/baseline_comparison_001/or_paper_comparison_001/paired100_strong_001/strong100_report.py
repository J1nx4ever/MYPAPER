"""Join the two new baselines to the 100 sealed historical environment slots."""

from collections import Counter
from math import fsum
from statistics import median

import numpy as np

from change_plans.token_attention_001.business_graph_summary_009.baseline_comparison_001.paired_100_001 import (
    paired_report,
)
from traditional_dispatch_v1.runtime import _require, load_reference
from traditional_dispatch_v2.runtime import _verify_anchor


OLD = ('unload_greedy', 'fifo_greedy', 'rolling', 'dispatch_inspired', 'bg128')
NEW = ('dispatch_grade', 'rolling_quota')
FIFO_V3 = 'dispatch_fifo_v3'


def _require_receipt(row, kind, slot):
    _require((row['policy_kind'], row['pair_ordinal'], row['scene_id']) == (kind, slot, 'nominal')
             and row['status'] == 'COMPLETE' and row['normal_completion'] is True
             and row['completed_shifts'] == 1 and row['optimizer_updates'] == 0,
             'unexpected or incomplete receipt in the nominal paired result')


def _with_historical(shifts, suite):
    prior = load_reference(suite['historical_v2']['report'])
    prior_v3 = load_reference(suite['historical_v3']['report'])
    slots = suite['evaluation_slots']
    _require(slots == prior['evaluation_slots'] == prior_v3['evaluation_slots'] == 100
             and len(prior['shifts']) == len(prior['receipts']) == slots * len(OLD)
             and len(prior_v3['shifts']) == slots
             and len(shifts) == len(suite['jobs']) == slots * len(NEW),
             'paired report requires all shifts in the frozen nominal roster')
    groups = {kind: [] for kind in (*OLD, FIFO_V3, *NEW)}
    pairs = []
    for slot in range(slots):
        old_offset = slot * len(OLD)
        anchor_idx = old_offset + OLD.index('bg128')
        anchor = prior['shifts'][anchor_idx]
        runs = {}
        for index, kind in enumerate(OLD, old_offset):
            row = prior['shifts'][index]
            _require_receipt(row, kind, slot)
            _verify_anchor(row, anchor)
            groups[kind].append(row)
            runs[kind] = row['run_index']
        v3 = prior_v3['shifts'][slot]
        _require_receipt(v3, FIFO_V3, slot)
        _verify_anchor(v3, anchor)
        groups[FIFO_V3].append(v3)
        runs[FIFO_V3] = v3['run_index']
        for index, kind in enumerate(NEW, slot * len(NEW)):
            row, job = shifts[index], suite['jobs'][index]
            _require_receipt(row, kind, slot)
            fields = ('run_index', 'recording_id', 'experiment_seed', 'phase_config')
            _require(all(row[key] == job[key] for key in fields)
                     and job['anchor_receipt'] == prior['receipts'][anchor_idx]
                     and job['anchor_run_index'] == anchor['run_index']
                     and row['actor_sampling_seed'] == 0,
                     'new shift or anchor differs from its frozen roster')
            _verify_anchor(row, anchor)
            groups[kind].append(row)
            runs[kind] = row['run_index']
        pairs.append(dict(pair_ordinal=slot, experiment_seed=anchor['experiment_seed'], runs=runs))
    _require(len({pair['experiment_seed'] for pair in pairs}) == slots
             and len({row['run_index'] for rows in groups.values() for row in rows}) == slots * len(groups),
             'pair seeds or physical run identities overlap')
    return groups, pairs


def _compute(rows, *, historical):
    engineering = [row['engineering'] for row in rows]
    summary = dict(
        sample='archived_live_receipts' if historical else 'new_live_receipts',
        decisions=sum(row['decisions'] for row in engineering),
        policy_shift_p95_median_ms=median(row['policy_latency_ms']['p95'] for row in engineering),
        policy_max_ms=max(row['policy_latency_ms']['max'] for row in engineering),
        rpc_shift_p95_median_ms=median(row['rpc_receive_to_response_ms']['p95'] for row in engineering),
        rpc_max_ms=max(row['rpc_receive_to_response_ms']['max'] for row in engineering),
        rpc_responses_over_target=sum(row['rpc_responses_over_target'] for row in engineering),
        policy_cpu_seconds=None,
        policy_wall_seconds=None,
    )
    if historical:
        return summary
    measured = [row['compute'] for row in engineering]
    for field in ('policy_wall_seconds', 'policy_cpu_seconds', 'policy_commit_wall_seconds',
                  'policy_commit_cpu_seconds'):
        summary[field] = fsum(row[field] for row in measured)
    summary['policy_cpu_ms_per_decision'] = 1000 * summary['policy_cpu_seconds'] / summary['decisions']
    summary['policy_wall_ms_per_decision'] = 1000 * summary['policy_wall_seconds'] / summary['decisions']
    summary['process_context'] = measured[0]['process_context']
    summary['same_process_context_within_group'] = all(
        row['process_context'] == summary['process_context'] for row in measured)
    return summary


def _summarize(rows, *, historical):
    result = paired_report._summarize_arm(rows)
    engineering = [row['engineering'] for row in rows]
    for field in ('max_scored_depth', 'max_matched_trucks'):
        if field in result['action_counts']:
            result['action_counts'][field] = max(row['action_counts'].get(field, 0) for row in engineering)
    result['passed_engineering_shifts'] = sum(row['engineering_acceptance']['passed'] for row in rows)
    result['all_four_windows_compliant_shifts'] = sum(row['physical']['all_four_windows_compliant'] for row in rows)
    result['compute'] = _compute(rows, historical=historical)
    if not historical:
        statuses = Counter()
        for row in engineering:
            statuses.update(row['solver_status_counts'])
        gaps = [row['max_mip_gap'] for row in engineering if row['max_mip_gap'] is not None]
        result['solver_status_counts'] = dict(statuses)
        result['solver_time_limit_requests'] = sum(row['solver_time_limit_requests'] for row in engineering)
        result['accepted_solver_requests'] = sum(row['accepted_solver_requests'] for row in engineering)
        result['max_mip_gap'] = max(gaps, default=None)
    return result


def aggregate(shifts, *, suite):
    groups, pairs = _with_historical(shifts, suite)
    slots = suite['evaluation_slots']
    rng = np.random.default_rng(paired_report.ANALYSIS_SEED)
    draws = rng.integers(0, slots, size=(paired_report.BOOTSTRAP_REPLICATES, slots))
    main = [('bg128', kind) for kind in NEW]
    secondary = [(a, b) for a in NEW for b in (*OLD, FIFO_V3) if b != 'bg128']
    secondary.append((NEW[0], NEW[1]))
    comparisons = []
    for first, second in main + secondary:
        result = paired_report._matched_comparison(groups, first, second, draws)
        result['primary'] = (first, second) in main
        comparisons.append(result)
    return dict(
        evaluation_slots=slots,
        total_matched_receipts=slots * len(groups),
        matched_initial_state_and_rpc=True,
        historical_reports={key: suite[key]['report'] for key in ('historical_v2', 'historical_v3')},
        same_host_compute_probe=suite.get('compute_probe'),
        compute_scope=dict(
            historical='archived wall-clock policy and RPC latencies; historical CPU consumption unavailable',
            new='controller process CPU and wall time around choose; action acknowledgement timed separately',
        ),
        pairs=pairs,
        groups={kind: _summarize(rows, historical=kind not in NEW) for kind, rows in groups.items()},
        primary_outcome='mean_paired_combined_cost_bg128_minus_each_new_baseline',
        comparisons=comparisons,
        intervals=dict(method='percentile_bootstrap_over_environment_seed_slots',
                       replicates=paired_report.BOOTSTRAP_REPLICATES, seed=paired_report.ANALYSIS_SEED,
                       level=.95, multiplicity_adjustment='none'),
    )
