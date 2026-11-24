"""Read-only verification of the sealed nominal five-policy paired evaluation."""

import argparse
from collections import Counter
from contextlib import closing
from hashlib import sha256
import json
from math import comb, fsum, isclose, sqrt
from pathlib import Path
import sqlite3
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from all_recheck_001.verify_results import checked, require, verify_shift
from python.offline_split import JAVA_MASK, java_random_states
from python.runtime_scenario import PREFIX, fingerprint


POLICIES = ('unload_greedy', 'fifo_greedy', 'rolling', 'dispatch_inspired', 'bg128')
CONTRASTS = (('bg128', 'rolling'), ('bg128', 'dispatch_inspired'), ('bg128', 'fifo_greedy'),
             ('bg128', 'unload_greedy'), ('rolling', 'fifo_greedy'), ('fifo_greedy', 'unload_greedy'))
METRICS = ('combined_cost', 'grade_cost', 'waiting_cost', 'production_cost', 'total_wait_seconds',
           'ore_tons', 'waste_tons', 'complete_task', 'all_four_windows_compliant',
           'ore_target_met', 'waste_target_met')
PAIR_FIELDS = ('experiment_seed', 'initial_state', 'actual_rpc_configuration')
SHIFT_FIELDS = ('run_index', 'recording_id', 'experiment_seed', 'actor_sampling_seed', 'policy_kind',
                'phase_config', 'pair_ordinal', 'scene_id')
FORMAL_SUITE = 'bg128-traditional-paired100-v2'
FORMAL_REPORT = 'bg128-traditional-paired100-comparison-v2'
SMOKE_REPORT = 'bg128-traditional-audit-report-v2'


def _canonical(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                       separators=(',', ':')) + '\n').encode()


def _digest(value):
    return sha256(_canonical(value)).hexdigest()


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _payload(raw, checksum):
    value = json.loads(raw)
    require(_digest(value) == checksum, 'SQLite ledger checksum differs')
    return value


def _ledger(run, jobs, smoke):
    if smoke:
        entry = json.loads((run / 'budget.json').read_bytes())
        return entry, entry['slots']

    path = (run / 'budget.sqlite').as_uri()
    with closing(sqlite3.connect(path + '?mode=ro', uri=True)) as con:
        (campaign, raw, checksum), = con.execute('SELECT id,payload,checksum FROM campaign')
        entry = _payload(raw, checksum)
        require(entry['campaign_id'] == campaign and entry['schema'] == 'bg128-traditional-paired100-v2-budget-v1',
                'SQLite campaign identity or schema differs')
        slots = {}
        for run_index, status, raw, checksum in con.execute('SELECT run,status,payload,checksum FROM slots'):
            slot = _payload(raw, checksum)
            require(run_index in jobs and slot['status'] == status
                    and slot['identity_sha256'] == _digest(jobs[run_index]), 'SQLite slot identity differs')
            slots[str(run_index)] = dict(slot, identity=jobs[run_index])
    return entry, slots


def _verify_artifacts(suite):
    actor_manifest = json.loads(checked(suite['actor_manifest']))
    checkpoint_manifest = json.loads(checked(suite['checkpoint_manifest']))
    seal = json.loads(checked(suite['checkpoint_seal']))
    for artifact, manifest in (('actor', actor_manifest), ('checkpoint', checkpoint_manifest)):
        raw = checked(suite[artifact])
        require(manifest['sha256'] == suite[artifact]['sha256'] and manifest['bytes'] == len(raw)
                and manifest['update_index'] == 32, f'{artifact} manifest differs')
    digest = actor_manifest['actor_digest']
    require(_sha(digest) and digest == checkpoint_manifest['actor_digest']
            and seal['sha256'] == suite['checkpoint']['sha256']
            and seal['manifest_sha256'] == suite['checkpoint_manifest']['sha256'],
            'frozen u32 Actor and checkpoint disagree')
    return digest


def _verify_phase(suite, job):
    phase = json.loads(checked(job['phase_config']))
    require(phase['schema'] == 'bg128-traditional-audit-phase-v2'
            and phase['job'] == {key: value for key, value in job.items() if key != 'phase_config'}
            and phase['campaign_id'] == suite['campaign_id'] and phase['policy_kind'] == job['policy_kind']
            and phase['method'] == suite['method'] and phase['protected_core'] == suite['protected_core']
            and phase['execution'] == suite['execution']
            and phase['optimizer_updates'] == phase['training_shifts'] == 0
            and phase['capture_material_events'] is phase['capture_physical_events'] is False
            and phase['rpc_configuration_authoritative'] is True, 'phase configuration differs from the frozen job')
    refs = ('actor', 'checkpoint', 'checkpoint_manifest', 'checkpoint_seal')
    if job['policy_kind'] == 'bg128':
        require(all(phase[key] == suite[key] for key in refs), 'BG128 phase uses different frozen artifacts')
    else:
        require(not any(key in phase for key in refs), 'traditional phase contains Actor artifacts')


def _verify_receipt(suite, job, row, ref, slot, actor_digest):
    require(row == json.loads(checked(ref)) and slot['receipt'] == ref and slot['status'] == 'COMPLETE'
            and slot['identity'] == job, 'report, receipt, or ledger slot differ')
    require(all(row[key] == job[key] for key in SHIFT_FIELDS)
            and row['schema'] == 'bg128-traditional-audit-receipt-v2' and row['completed_shifts'] == 1,
            'receipt identity or evaluated shift count differs')
    require(Path(ref['path']) == Path(job['python_output']) / 'evaluation_receipt.json'
            and row['episode_id'] == Path(job['event_bundle']).name
            and row['terminal_request_id'] == f"{row['episode_id']}-{row['engineering']['decisions']}"
            and row['engineering']['rpc_response_target_ms'] == suite['method']['response_target_ms'],
            'receipt path, terminal request, or response target differs')

    claim_ref = row['java_start_claim']
    claim_path = Path(suite['control_directory']) / 'java_claims' / f"run_{job['run_index']}.once.claim"
    require(Path(claim_ref['path']) == claim_path, 'Java START claim path differs')
    lines = [line.split('=', 1) for line in checked(claim_ref).decode().splitlines()]
    claim = dict(lines)
    require(len(lines) == len(claim) and claim['config_sha256'] == claim['snapshot_sha256'] == job['phase_config']['sha256']
            and claim['master_plan_sha256'] == suite['master_plan']['sha256']
            and claim['ordinal'] == str(job['ordinal']) and claim['run_index'] == str(job['run_index'])
            and claim['campaign_id'] == suite['campaign_id'], 'Java START claim identity differs')

    configuration = row['actual_rpc_configuration']
    scene = {key: value for key, value in configuration.items() if key.startswith(PREFIX)}
    require(all(configuration.get(key) == value for key, value in job['expected_rpc_config'].items())
            and fingerprint(scene) == job['runtime_scene_sha256'],
            'actual RPC scene differs from frozen configuration')
    start = row['initial_state']
    require(start['sim_time_seconds'] == 0 and _sha(start['observation_sha256'])
            and type(start['truck_id']) is int and 0 <= start['truck_id'] < 9
            and len(start['mask']) == 4 and all(type(value) is bool for value in start['mask'])
            and any(start['mask']), 'invalid first observation witness')
    telemetry = row['engineering']['native_fallback_telemetry']
    require(0 <= telemetry['callbacks_with_both_flags'] <= telemetry['total_callbacks']
            == row['engineering']['decisions'] + 1, 'RPC callback telemetry count differs')
    if job['policy_kind'] == 'bg128':
        rng = row['engineering']['actor_sampling_rng']
        require(row['actor'] == suite['actor'] and row['checkpoint'] == suite['checkpoint']
                and row['actor_digest'] == actor_digest and _sha(rng['initial_sha256'])
                and _sha(rng['final_sha256']) and rng['initial_sha256'] != rng['final_sha256'],
                'BG128 receipt uses different frozen artifacts or invalid RNG evidence')
    else:
        require(not any(key in row for key in ('actor', 'actor_digest', 'checkpoint'))
                and 'actor_sampling_rng' not in row['engineering'], 'traditional receipt contains Actor evidence')
    return verify_shift(row)


def _verify_roster(suite, jobs, count):
    pairing = suite['pairing']
    require(pairing['policies'] == list(POLICIES) and pairing['slots'] == count
            and pairing['scope'] == 'this_campaign_only' and pairing['share'] == 'experiment_seed',
            'frozen pairing contract differs')
    require(len(jobs) == count * len(POLICIES), 'incomplete five-policy roster')
    for ordinal, job in enumerate(jobs):
        slot, arm = divmod(ordinal, len(POLICIES))
        actor = POLICIES[arm] == 'bg128'
        require(job['ordinal'] == ordinal and job['pair_ordinal'] == slot
                and job['policy_kind'] == POLICIES[arm] and job['scene_id'] == 'nominal'
                and job['actor_sampling_seed_consumed'] is actor, 'job order, scene, or sampling contract differs')
        require(type(job['run_index']) is int and job['run_index'] > 0
                and type(job['experiment_seed']) is int and 0 < job['experiment_seed'] <= JAVA_MASK
                and type(job['actor_sampling_seed']) is int
                and (0 < job['actor_sampling_seed'] <= JAVA_MASK if actor else job['actor_sampling_seed'] == 0),
                'invalid run or RNG identity')
        require(job['actor_seed'] == (suite['learner_config']['seed'] if actor else 0)
                and job['checkpoint_update'] == (32 if actor else 0), 'frozen training lineage differs')


def _verify_campaign(run, count, smoke):
    closed = json.loads((run / 'suite_complete.json').read_bytes())
    suite = json.loads(checked(closed['suite']))
    report = json.loads(checked(closed['evaluation']))
    jobs = suite['jobs']
    expected_physical = count * len(POLICIES)
    require(run.name == closed['campaign_id'] == suite['campaign_id'] == report['campaign_id']
            and Path(suite['python_output']) == run and closed['status'] == report['status'] == 'COMPLETE'
            and closed['physical_shifts'] == suite['physical_shifts'] == report['completed_shifts'] == expected_physical
            and suite['evaluation_slots'] == count and closed['training_shifts'] == suite['training_shifts'] == 0
            and closed['optimizer_updates'] == suite['optimizer_updates'] == report['optimizer_updates'] == 0
            and closed['protected_core'] == suite['protected_core'] and report['method'] == suite['method'],
            'campaign seal, scope or frozen method differs')
    if smoke:
        require(count == 1 and report['schema'] == SMOKE_REPORT, 'expected a sealed one-slot V2 smoke campaign')
        report_name = 'single_shift_validation.json'
    else:
        require(suite['schema'] == FORMAL_SUITE and report['schema'] == FORMAL_REPORT
                and closed['evaluation_slots'] == report['evaluation_slots'] == count
                and report['physical_shifts'] == expected_physical
                and report['pairing'] == suite['pairing'], 'formal five-policy report or suite schema differs')
        report_name = 'paired_comparison.json'
    require(Path(closed['evaluation']['path']) == Path(suite['report_directory']) / report_name,
            'sealed report is outside its frozen location')

    _verify_roster(suite, jobs, count)
    for ref in (suite['protected_core'], suite['master_plan'], *suite['source_pins']):
        checked(ref)
    actor_digest = _verify_artifacts(suite)
    entry, slots = _ledger(run, {job['run_index']: job for job in jobs}, smoke)
    _verify_ledger_scope(run, closed, entry, slots, jobs)

    groups = []
    for job, row, ref in zip(jobs, report['shifts'], report['receipts'], strict=True):
        try:
            _verify_phase(suite, job)
            status = _verify_receipt(suite, job, row, ref, slots[str(job['run_index'])], actor_digest)
            if job['policy_kind'] == POLICIES[0]:
                groups.append([])
            if groups[-1]:
                anchor = groups[-1][0][1]
                require(all(row[key] == anchor[key] for key in PAIR_FIELDS), 'same-slot first frame, seed, or RPC differs')
            groups[-1].append((job, row, status))
        except (KeyError, OSError, TypeError, ValueError) as error:
            raise ValueError(f"run {job['run_index']}: {error}") from error
    return groups, report


def _verify_ledger_scope(run, closed, entry, slots, jobs):
    require(entry['suite'] == closed['suite'] and set(slots) == {str(job['run_index']) for job in jobs}
            and len(jobs) == len(slots) and len(jobs) == len({job['recording_id'] for job in jobs})
            and entry.get('cap', len(jobs)) == len(jobs), 'ledger spans different physical identities')
    authority = json.loads(checked(entry['authority']))
    first = jobs[0]
    require(Path(entry['authority']['path']) == run / 'launch' / 'run_authority.json'
            and authority['suite'] == closed['suite'] and authority['campaign_id'] == closed['campaign_id']
            and authority['maximum_new_shifts'] == len(jobs) and authority['authorized_by_matching_anylogic_run'] is True
            and authority['first_run'] == {key: first[key] for key in ('run_index', 'experiment_seed', 'recording_id')},
            'ledger authority does not match the sealed roster')


def _verify_cross_slot(groups):
    jobs = [job for group in groups for job, _, _ in group]
    for field in ('run_index', 'recording_id', 'python_output', 'event_directory', 'event_bundle'):
        require(len({job[field] for job in jobs}) == len(jobs), f'physical identities reuse {field}')
    for field in ('path', 'sha256'):
        require(len({job['phase_config'][field] for job in jobs}) == len(jobs), f'phase configs reuse {field}')
    environment = [group[0][0]['experiment_seed'] for group in groups]
    sampling = [group[-1][0]['actor_sampling_seed'] for group in groups]
    lineage = {group[-1][0]['actor_seed'] & JAVA_MASK for group in groups}
    states = [state for seed in environment for state in java_random_states(seed).values()]
    require(len(set(environment + sampling)) == 2 * len(groups)
            and len(set(states)) == 5 * len(groups)
            and len(lineage) == 1 and not lineage.intersection(environment + sampling),
            'environment, Actor sampling, training lineage, or effective Java RNG states overlap')
    rng = [group[-1][1]['engineering']['actor_sampling_rng'] for group in groups]
    for field in ('initial_sha256', 'final_sha256'):
        require(len({row[field] for row in rng}) == len(groups), f'Actor RNG evidence repeats {field}')
    _verify_actor_rng(sampling, rng)
    rows = [group[0][1] for group in groups]
    require(all(row['actual_rpc_configuration'] == rows[0]['actual_rpc_configuration'] for row in rows)
            and all(group[-1][1]['actor'] == groups[0][-1][1]['actor']
                    and group[-1][1]['checkpoint'] == groups[0][-1][1]['checkpoint'] for group in groups),
            'nominal RPC configuration or frozen artifacts changed across slots')
    repeated_physical = {}
    for arm, policy in enumerate(POLICIES):
        outcome = Counter(_digest(group[arm][1]['physical']) for group in groups)
        repeated_physical[policy] = sum(replicas - 1 for replicas in outcome.values())
    return dict(environment_seeds=len(environment), actor_sampling_seeds=len(sampling), effective_java_states=len(states),
                distinct_initial_observations=len({row['initial_state']['observation_sha256'] for row in rows}),
                duplicate_physical_rows_by_policy=repeated_physical)


def _verify_actor_rng(seeds, evidence):
    import torch

    weights = torch.ones((1, 4), dtype=torch.float32)
    for seed, rng in zip(seeds, evidence, strict=True):
        generator = torch.Generator(device='cpu').manual_seed(seed)
        start = sha256(generator.get_state().numpy().tobytes()).hexdigest()
        # The frozen CPU sampler draws one category from four weights on each callback.
        # The weights affect the sampled action and consume the same RNG state transition.
        for _ in range(rng['draws']):
            torch.multinomial(weights, 1, replacement=True, generator=generator)
        end = sha256(generator.get_state().numpy().tobytes()).hexdigest()
        require((start, end) == (rng['initial_sha256'], rng['final_sha256']),
                f'Actor draw count or RNG state differs for sampling seed {seed}')


def _values(rows, key):
    return np.asarray([row['physical'][key] for row in rows], dtype=np.float64)


def _same_number(actual, expected):
    require(isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12),
            f'reported statistic differs: {actual} vs {expected}')


def _verify_effect(reported, delta, draws):
    _same_number(reported['mean'], float(delta.mean()))
    observed = (int(np.count_nonzero(delta > 0)), int(np.count_nonzero(delta < 0)),
                int(np.count_nonzero(delta == 0)))
    require(observed == tuple(reported[key] for key in ('positive', 'negative', 'ties')),
            'paired effect direction counts differ')
    limits = np.quantile(delta[draws].mean(axis=1), [.025, .975])
    for actual, expected in zip(reported['paired_bootstrap_ci95'], limits, strict=True):
        _same_number(actual, float(expected))


def _verify_arm(summary, acceptance, rows):
    count = len(rows)
    passed = sum(row['engineering_acceptance']['passed'] for row in rows)
    engineering = [row['engineering'] for row in rows]
    counts = Counter()
    for row in engineering:
        counts.update(row['action_counts'])
    for key in ('max_scored_depth', 'max_matched_trucks'):
        if key in counts:
            counts[key] = max(row['action_counts'].get(key, 0) for row in engineering)
    decisions = sum(row['decisions'] for row in engineering)
    require(acceptance == dict(passed=passed, failed=count - passed) and summary['shifts'] == count
            and summary['complete_task_shifts'] == sum(row['physical']['complete_task'] for row in rows)
            and summary['decisions'] == decisions and summary['action_counts'] == dict(counts)
            and summary['rpc_over_target'] == sum(row['rpc_responses_over_target'] for row in engineering),
            'per-policy report inventory differs')
    for key in METRICS:
        _same_number(summary['mean'][key], fsum(row['physical'][key] for row in rows) / count)
    for key in ('callbacks_with_both_flags', 'total_callbacks'):
        require(summary['native_fallback_telemetry'][key]
                == sum(row['native_fallback_telemetry'][key] for row in engineering), 'native telemetry total differs')
    for key in ('load_seconds', 'dump_resource_queue_seconds', 'dump_capacity_admission_seconds'):
        _same_number(summary['waiting_by_stage_mean_seconds'][key],
                     fsum(row['physical']['wait_breakdown'][key] for row in rows) / count)
    _same_number(summary['rpc_shift_p95_median_ms'],
                 float(np.median([row['rpc_receive_to_response_ms']['p95'] for row in engineering])))
    _same_number(summary['rpc_max_ms'], max(row['rpc_receive_to_response_ms']['max'] for row in engineering))
    _same_number(summary['fallback_rate'], counts['fallback'] / decisions)
    _same_number(summary['planning_deadline_fraction'], counts['timed_out'] / decisions)
    for window, reported in zip(range(4), summary['windows'], strict=True):
        records = [row['physical']['windows'][window] for row in rows]
        grades = [row['grade'] for row in records if not row['empty']]
        require(reported['window'] == window and reported['compliant'] == sum(row['compliant'] for row in records)
                and reported['empty'] == count - len(grades) and reported['valid_windows'] == len(grades),
                'per-policy window inventory differs')
        _same_number(reported['mean_tons'], fsum(row['tons'] for row in records) / count)
        if grades:
            _same_number(reported['mean_grade'], fsum(grades) / len(grades))
            _same_number(reported['grade_rmse_pp'], 100 * sqrt(fsum((grade - .27)**2 for grade in grades) / len(grades)))
        else:
            require(reported['mean_grade'] is None and reported['grade_rmse_pp'] is None,
                    'report assigns a grade to an empty window')


def _verify_report_statistics(report, groups):
    rows_by_policy = {policy: [group[arm][1] for group in groups] for arm, policy in enumerate(POLICIES)}
    count = len(groups)
    expected_intervals = dict(method='percentile_bootstrap_over_environment_seed_slots',
                              replicates=10000, seed=90221, level=.95,
                              secondary_contrasts='descriptive_intervals_without_multiplicity_adjustment')
    require(report['intervals'] == expected_intervals
            and report['primary_outcome'] == 'mean_paired_combined_cost_bg128_minus_rolling'
            and set(report['groups']) == set(POLICIES)
            and set(report['engineering_acceptance']) == set(POLICIES), 'frozen report statistics contract differs')
    for policy, rows in rows_by_policy.items():
        _verify_arm(report['groups'][policy], report['engineering_acceptance'][policy], rows)

    draws = np.random.default_rng(90221).integers(0, count, size=(10000, count))
    for index, ((first, second), comparison) in enumerate(zip(CONTRASTS, report['comparisons'], strict=True)):
        require((comparison['first'], comparison['second']) == (first, second)
                and comparison['primary'] is (index == 0) and comparison['effect_direction'] == 'first_minus_second',
                'frozen paired contrast order or direction differs')
        lhs, rhs = rows_by_policy[first], rows_by_policy[second]
        for key in METRICS:
            _verify_effect(comparison['metrics'][key], _values(lhs, key) - _values(rhs, key), draws)
        complete = comparison['metrics']['complete_task']
        both = int(np.count_nonzero(_values(lhs, 'complete_task') * _values(rhs, 'complete_task')))
        discordant = complete['positive'] + complete['negative']
        tail = min(complete['positive'], complete['negative'])
        p = min(1., 2 * sum(comb(discordant, k) for k in range(tail + 1)) / (2 ** discordant))
        require(complete['both_complete'] == both and complete['both_failed'] == complete['ties'] - both,
                'joint complete-task counts differ')
        _same_number(complete['mcnemar_two_sided_p'], p)
        for window, reported in zip(range(4), comparison['per_window_compliance'], strict=True):
            require(reported['window'] == window, 'paired window order differs')
            delta = np.asarray([int(left['physical']['windows'][window]['compliant'])
                                - int(right['physical']['windows'][window]['compliant'])
                                for left, right in zip(lhs, rhs, strict=True)])
            _verify_effect(reported, delta, draws)


def verify(runs, *, expected_slots=100, historical_smoke=False):
    """Verify complete evidence. Historical smoke mode pools sealed single-slot V2 campaigns only."""
    paths = [Path(run).resolve() for run in runs]
    require(type(expected_slots) is int and expected_slots > 0
            and len(paths) == (expected_slots if historical_smoke else 1), 'wrong number of campaign directories')
    groups = []
    methods = []
    for path in paths:
        campaign_groups, report = _verify_campaign(path, 1 if historical_smoke else expected_slots, historical_smoke)
        groups.extend(campaign_groups)
        methods.append(report['method'])
    require(all(method == methods[0] for method in methods), 'campaigns use different method settings')
    diagnostics = _verify_cross_slot(groups)
    if not historical_smoke:
        _verify_report_statistics(report, groups)
    summary = {}
    for arm, policy in enumerate(POLICIES):
        statuses = [group[arm][2] for group in groups]
        summary[policy] = dict(mean_combined_cost=fsum(item['cost'] for item in statuses) / expected_slots,
                               complete_tasks=sum(item['complete_task'] for item in statuses),
                               engineering_failures=sum(not item['engineering_passed'] for item in statuses),
                               python_fallbacks=sum(item['python_fallbacks'] for item in statuses),
                               rpc_responses_over_target=sum(item['over_response_target'] for item in statuses))
    return dict(campaigns=[path.name for path in paths], historical_smoke=historical_smoke,
                verified_slots=expected_slots, verified_physical_shifts=expected_slots * len(POLICIES),
                diagnostics=diagnostics, policies=summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--expected-slots', type=int, default=100)
    parser.add_argument('--historical-smoke', action='store_true')
    args = parser.parse_args()
    try:
        result = verify(args.runs, expected_slots=args.expected_slots, historical_smoke=args.historical_smoke)
    except (KeyError, OSError, sqlite3.Error, TypeError, ValueError) as error:
        parser.exit(1, f'Verification failed: {error}\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
