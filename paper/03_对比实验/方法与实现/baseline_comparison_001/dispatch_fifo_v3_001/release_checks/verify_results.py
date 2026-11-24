"""Read-only physical, identity and paired statistical audit of a sealed FIFO campaign."""
import argparse
from collections import Counter
from importlib import import_module
import json
from math import comb, isfinite
from pathlib import Path
import sqlite3
import sys

import numpy as np


DEFAULT_PLAN = Path(__file__).resolve().parents[1]
PROJECT = DEFAULT_PLAN.parents[4]
for directory in (PROJECT, PROJECT.parents[1]):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from change_plans.token_attention_001.business_graph_summary_009.baseline_comparison_001.paired_v2_100_001 import (
    verify_results as v2,
)


def _path_ref(ref, path):
    v2.require(Path(ref['path']) == path, f'unexpected artifact path for {path.name}')
    return v2.checked(ref)


def _check_fifo(row):
    """Validate V3 counters/witness and use the frozen physical V2 settlement equations."""
    e = row['engineering']
    counts = e['action_counts']
    n = e['decisions']
    for key, value in counts.items():
        v2.require(type(value) is int and value >= 0, f'invalid engineering counter {key}')
    trials = counts['fifo_trials_completed']
    eligible = counts.get('fifo_requests_with_completed_trial', 0)
    matched = counts.get('live_multitruck_matches', 0)
    compared = counts['fifo_trials_compared']
    decisions_compared = counts['fifo_requests_with_compared_trial']
    nominated = counts['fifo_trials_nominated']
    v2.require(matched == decisions_compared <= eligible <= trials and matched <= compared <= trials
               and matched <= counts.get('planned', 0)
               and counts['fifo_selected_from_cohort_requests'] <= matched,
               'executable FIFO comparisons disagree with acknowledged match count')
    v2.require(counts['fifo_roots_completed'] <= counts['fifo_candidates'] <= 4 * n
               and trials + counts['fifo_trials_rejected'] <= nominated <= 12 * n
               and counts.get('fifo_guard_restored_root', 0) <= counts.get('timed_out', 0) <= n,
               'FIFO trial budget, root coverage or deadline counters differ')
    witness = e['first_completed_fifo_trial']
    compared_witness = e['first_compared_fifo_trial']
    v2.require((witness is not None) == (eligible > 0)
               and (compared_witness is not None) == (decisions_compared > 0),
               'FIFO trial witness differs from acknowledged requests')
    for record in (witness, compared_witness):
        if record is None:
            continue
        actions, trucks = record['selected_actions'], record['selected_truck_ids']
        v2.require(0 <= record['request_ordinal'] < n and 1 <= record['roots_completed'] <= 4
                   and 1 <= record['trials_completed'] <= record['trials_nominated'] <= 12
                   and 0 <= record['trials_compared'] <= record['trials_completed']
                   and 1 <= len(actions) == len(trucks) <= 3
                   and all(type(action) is int and 0 <= action < 4 for action in actions)
                   and all(type(truck) is int and 0 <= truck < 9 for truck in trucks)
                   and record['action'] == actions[0] and type(record['selected_feasible']) is bool
                   and type(record['selected_from_cohort']) is bool
                   and (not record['selected_from_cohort'] or record['trials_compared'] > 0)
                   and isfinite(record['selected_cost']), 'invalid acknowledged FIFO witness')
    if compared_witness is not None:
        live = e['first_live_multitruck_match']
        v2.require(live is not None and live['decision_sequence'] == compared_witness['request_ordinal']
                   and live['action'] == compared_witness['action']
                   and live['truck_id'] == compared_witness['selected_truck_ids'][0]
                   and 2 <= live['matched_trucks'] <= 3 and live['mask'][live['action']] is True
                   and compared_witness['trials_compared'] > 0
                   and witness['request_ordinal'] <= compared_witness['request_ordinal'],
                   'first executable comparison differs from the live match')
    errors = Counter()
    for error in e['fifo_planning_errors']:
        v2.require(error['stage'] in ('guidance', 'model') and bool(error['reason'])
                   and type(error['requests']) is int and error['requests'] > 0, 'invalid FIFO model error')
        errors[error['stage']] += error['requests']
    v2.require(all(errors[key] == counts[f'fifo_{key}_error_requests'] for key in ('guidance', 'model')),
               'FIFO model error counts differ')
    v2.require(0 <= e['rpc_responses_over_target'] <= n and row['physical']['waste_tons'] >= 0
               and all(row['physical']['wait_breakdown'][key] >= 0 for key in (
                   'load_seconds', 'dump_resource_queue_seconds', 'dump_capacity_admission_seconds')),
               'negative physical quantity or invalid RPC count')
    expected = dict(rpc_and_physical_completion=True, met_response_target=e['rpc_responses_over_target'] == 0,
                    flow_plan_observed=counts.get('flow_plan_successes', 0) > 0,
                    live_multitruck_match_observed=matched > 0, fifo_trial_observed=witness is not None,
                    fifo_executable_comparison_observed=compared_witness is not None, fifo_models_valid=not errors)
    expected['passed'] = all(expected.values())
    v2.require(row['engineering_acceptance'] == expected, 'V3 engineering acceptance differs from evidence')
    # Only the engineering-gate adapter differs. All mass, wait, cost, timing and
    # callback evidence is passed unchanged to the independent V2 equations.
    common = {key: expected[key] for key in ('rpc_and_physical_completion', 'met_response_target',
                                              'flow_plan_observed', 'live_multitruck_match_observed')}
    common['passed'] = all(common.values())
    v2.verify_shift(dict(row, policy_kind='dispatch_inspired', engineering_acceptance=common))


def _verify_receipt(suite, job, row, ref, slot, old, old_ref, bg, runtime):
    from python.runtime_scenario import PREFIX, fingerprint

    v2.require(slot['identity'] == job and slot['status'] == 'COMPLETE' and slot['receipt'] == ref
               and row == json.loads(_path_ref(ref, Path(job['python_output'])/'evaluation_receipt.json')),
               'report and ledger must reference the same physical receipt')
    v2.require(all(row[key] == job[key] for key in v2.SHIFT_FIELDS)
               and row['schema'] == runtime.RECEIPT_SCHEMA and row['completed_shifts'] == 1
               and row['normal_completion'] is True and row['status'] == 'COMPLETE'
               and job['actor_seed'] == row['actor_sampling_seed'] == 0 and job['actor_sampling_seed_consumed'] is False
               and not any(key in row for key in ('actor', 'checkpoint', 'actor_digest'))
               and 'actor_sampling_rng' not in row['engineering'], 'physical identity or deterministic policy differs')
    v2.require(job['v2_reference_run_index'] == old['run_index'] and job['v2_recording_id'] == old['recording_id']
               and job['v2_phase_config'] == old['phase_config'] and row['v2_anchor'] == old_ref
               and row['pair_ordinal'] == old['pair_ordinal'] == bg['pair_ordinal']
               and all(row[key] == old[key] == bg[key] for key in v2.PAIR_FIELDS),
               'same-slot V2/BG128 seed, configuration, first frame or receipt differs')
    claim_path = Path(suite['control_directory']) / 'java_claims' / f"run_{job['run_index']}.once.claim"
    lines = [line.split('=', 1) for line in _path_ref(row['java_start_claim'], claim_path).decode().splitlines()]
    claim = dict(lines)
    digest = job['phase_config']['sha256']
    v2.require(len(lines) == len(claim) and claim['config_sha256'] == claim['snapshot_sha256'] == digest
               and claim['protocol'] == 'baseline1-one-click-java-start-v1'
               and claim['master_plan_sha256'] == suite['master_plan']['sha256']
               and claim['campaign_id'] == suite['campaign_id'] and claim['ordinal'] == str(job['ordinal'])
               and claim['run_index'] == str(job['run_index']), 'Java START claim differs')
    cfg = row['actual_rpc_configuration']
    v2.require(all(cfg.get(key) == val for key, val in job['expected_rpc_config'].items())
               and fingerprint({key: val for key, val in cfg.items() if key.startswith(PREFIX)})
               == job['runtime_scene_sha256'], 'RPC differs from the frozen scenario')
    eng = row['engineering']
    native = eng['native_fallback_telemetry']
    v2.require(0 <= native['callbacks_with_both_flags'] <= native['total_callbacks'] == eng['decisions'] + 1
               and eng['rpc_response_target_ms'] == suite['method']['response_target_ms']
               and row['episode_id'] == Path(job['event_bundle']).name
               and row['terminal_request_id'] == f"{row['episode_id']}-{eng['decisions']}",
               'callback, END or telemetry inventory differs')
    _check_fifo(row)


def _verify_statistics(report, groups, policy):
    count = len(groups[policy])
    expected = dict(method='percentile_bootstrap_over_environment_seed_slots', replicates=10000,
                    seed=90221, level=.95, secondary_contrasts='descriptive_intervals_without_multiplicity_adjustment')
    v2.require(set(report['groups']) == set(groups) and set(report['engineering_acceptance']) == {policy}
               and report['intervals'] == expected
               and report['primary_outcome'] == f'mean_paired_combined_cost_{policy}_minus_dispatch_inspired',
               'frozen primary outcome, groups or analysis settings differ')
    for arm, rows in groups.items():
        passed = sum(row['engineering_acceptance']['passed'] for row in rows)
        status = report['engineering_acceptance'][policy] if arm == policy else dict(passed=passed, failed=count-passed)
        v2._verify_arm(report['groups'][arm], status, rows)
    draws = np.random.default_rng(90221).integers(0, count, size=(10000, count))
    for idx, (other, contrast) in enumerate(zip(('dispatch_inspired', 'bg128'), report['comparisons'], strict=True)):
        v2.require((contrast['first'], contrast['second']) == (policy, other) and contrast['primary'] is (idx == 0)
                   and contrast['effect_direction'] == 'first_minus_second'
                   and set(contrast['metrics']) == set(v2.METRICS), 'paired contrast inventory differs')
        lhs, rhs = groups[policy], groups[other]
        for key in v2.METRICS:
            v2._verify_effect(contrast['metrics'][key], v2._values(lhs, key) - v2._values(rhs, key), draws)
        task = contrast['metrics']['complete_task']
        discordant = task['positive'] + task['negative']
        tail = min(task['positive'], task['negative'])
        p = min(1., 2 * sum(comb(discordant, k) for k in range(tail + 1)) / 2 ** discordant)
        both = sum(a['physical']['complete_task'] and b['physical']['complete_task'] for a, b in zip(lhs, rhs, strict=True))
        v2.require(task['both_complete'] == both and task['both_failed'] == task['ties'] - both, 'task pair table differs')
        v2._same_number(task['mcnemar_two_sided_p'], p)
        for window, result in zip(range(4), contrast['per_window_compliance'], strict=True):
            v2.require(result['window'] == window, 'window contrast order differs')
            delta = np.array([int(a['physical']['windows'][window]['compliant']) -
                              int(b['physical']['windows'][window]['compliant']) for a, b in zip(lhs, rhs, strict=True)])
            v2._verify_effect(result, delta, draws)


def verify(plan_dir=DEFAULT_PLAN):
    plan_dir = Path(plan_dir).resolve()
    sys.path.insert(0, str(plan_dir))
    plan = import_module('v3_plan')
    v2.require(plan.PLAN == plan_dir, 'conflicting plan module already imported')
    seal_path = plan.OUTPUT / 'suite_complete.json'
    v2.require(seal_path.is_file(), f'physical campaign has no COMPLETE seal: {seal_path}')
    ledger = import_module('v3_ledger')
    runtime = import_module('v3_runtime')
    suite = plan.load_suite()
    closed = json.loads(seal_path.read_bytes())
    report = json.loads(_path_ref(closed['evaluation'], plan.REPORT / plan.REPORT_FILENAME))
    status = json.loads((plan.OUTPUT/'run_status.json').read_bytes())
    v2.require(closed['suite'] == plan.reference(plan.SUITE) and closed['protected_core'] == suite['protected_core']
               and closed['campaign_id'] == report['campaign_id'] == status['campaign_id'] == plan.CAMPAIGN
               and closed['status'] == report['status'] == status['status'] == 'COMPLETE'
               and report['schema'] == runtime.REPORT_SCHEMA and report['method'] == suite['method']
               and closed['training_shifts'] == closed['optimizer_updates'] == report['optimizer_updates'] == 0
               and not (plan.CONTROL/'STOP.properties').exists(), 'V3 campaign seal or completion state differs')
    v2.require(closed['physical_shifts'] == closed['evaluation_slots'] == status['completed_slots']
               == status['completed_shifts'] == report['evaluation_slots'] == report['completed_shifts']
               == report['physical_shifts'] == len(report['shifts']) == len(report['receipts']) == plan.COUNT == 100,
               'all 100 physical completions are required')
    previous = plan.previous
    groups, old_report = v2._verify_campaign(previous.OUTPUT, 100, False)
    v2._verify_cross_slot(groups)
    v2._verify_report_statistics(old_report, groups)
    v2.require(report['old_completion'] == plan.reference(previous.OUTPUT/'suite_complete.json')
               and report['old_report'] == plan.reference(previous.REPORT / previous.REPORT_FILENAME),
               'sealed historical reference differs')
    state = ledger.Ledger(suite).snapshot()
    entry, slots = state['entry'], state['slots']
    v2._verify_ledger_scope(plan.OUTPUT, closed, entry, slots, suite['jobs'])
    v2.require(entry['v2_completion'] == report['old_completion'] and entry['v2_report'] == report['old_report'],
               'historical report and V3 ledger differ')
    rows_by_arm = {plan.POLICY: report['shifts'], 'dispatch_inspired': [], 'bg128': []}
    for i, (job, row, ref, group) in enumerate(zip(suite['jobs'], report['shifts'], report['receipts'], groups, strict=True)):
        source = {old_job['policy_kind']: (old_job, record) for old_job, record, _ in group}
        old_job, old = source['dispatch_inspired']
        bg = source['bg128'][1]
        v2.require(job['pair_ordinal'] == i, 'V3 slots out of order')
        _verify_receipt(suite, job, row, ref, slots[str(job['run_index'])], old, old_report['receipts'][old_job['ordinal']],
                        bg, runtime)
        rows_by_arm['dispatch_inspired'].append(old)
        rows_by_arm['bg128'].append(bg)
    _verify_statistics(report, rows_by_arm, plan.POLICY)
    return dict(status='VERIFIED_COMPLETE', campaign_id=plan.CAMPAIGN, paired_slots=100,
                verified_v3_physical_shifts=100, verified_reference_physical_shifts=500,
                engineering_acceptance=report['engineering_acceptance'][plan.POLICY],
                primary_combined_cost_difference=report['comparisons'][0]['metrics']['combined_cost'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-dir', type=Path, default=DEFAULT_PLAN)
    args = parser.parse_args()
    try:
        result = verify(args.plan_dir)
    except (KeyError, OSError, IndexError, sqlite3.Error, TypeError, ValueError, RuntimeError) as error:
        parser.exit(1, f'Verification failed: {error}\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
