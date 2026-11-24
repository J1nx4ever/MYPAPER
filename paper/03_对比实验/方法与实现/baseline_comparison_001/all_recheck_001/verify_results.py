"""Independently verify sealed five-arm results against physical totals and the frozen roster."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def require(value, message):
    if not value:
        raise ValueError(message)


def checked(ref):
    raw = Path(ref['path']).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == ref['sha256'], 'SHA256 mismatch: ' + ref['path'])
    return raw


def close(actual, expected):
    require(math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=2e-5),
            f'inconsistent settlement: {actual} vs {expected}')


def verify_shift(row):
    p, e = row['physical'], row['engineering']
    cfg = row['actual_rpc_configuration']
    require(row['normal_completion'] and row['status'] == 'COMPLETE'
            and row['terminal_time_seconds'] == 28800 and row['optimizer_updates'] == 0, 'invalid END')
    require([w['window'] for w in p['windows']] == [0, 1, 2, 3], 'window order')
    penalties, compliance = [], []
    for w in p['windows']:
        q, m = w['tons'], w['metal_tons']
        require(0 <= m <= q and w['empty'] == (q == 0), 'invalid window mass')
        grade = m/q if q else None
        if q:
            close(w['grade'], grade)
        else:
            require(w['grade'] is None, 'empty grade')
        good = bool(q and cfg['config.grade_lower'] - cfg['config.grade_epsilon'] <= grade
                    <= cfg['config.grade_upper'] + cfg['config.grade_epsilon'])
        penalty = min(2., max(0., abs(grade-.27)-.004)/.006) if q else 2.
        require(w['compliant'] == good, 'window compliance')
        close(w['continuous_cost'], penalty)
        penalties.append(penalty)
        compliance.append(good)
    close(sum(w['tons'] for w in p['windows']), p['ore_tons'])
    wait = p['wait_breakdown']
    require(wait['stage_errors'] == wait['load_fallback_hits'] == 0, 'invalid wait stage counters')
    close(wait['load_seconds'], wait['counted_load_seconds'])
    close(wait['dump_total_seconds'], wait['dump_resource_queue_seconds'] + wait['dump_capacity_admission_seconds'])
    close(wait['load_seconds'] + wait['dump_resource_queue_seconds'] + wait['dump_capacity_admission_seconds'],
          p['total_wait_seconds'])
    ore, waste = p['ore_tons'], p['waste_tons']
    production = 8*max(0., 1-ore/16000) + 8*max(0., 1-waste/4000)
    waiting = .2*p['total_wait_seconds']/28800
    close(p['grade_cost'], sum(penalties))
    close(p['production_cost'], production)
    close(p['waiting_cost'], waiting)
    close(p['combined_cost'], sum(penalties)+production+waiting)
    require(p['all_four_windows_compliant'] == all(compliance) and p['ore_target_met'] == (ore >= 16000)
            and p['waste_target_met'] == (waste >= 4000)
            and p['complete_task'] == (all(compliance) and ore >= 16000 and waste >= 4000), 'task outcome')
    n = e['decisions']
    require(n == sum(e['dispatches_by_site']) and n == sum(
        e['action_counts'].get(k, 0) for k in ('planned', 'single_legal', 'fallback', 'sampled')), 'action count')
    counts, policy = e['action_counts'], row['policy_kind']
    require(n > 0 and len(e['dispatches_by_site']) == 4 and all(
        isinstance(k, int) and k >= 0 for k in e['dispatches_by_site']), 'invalid site counts')
    require(sum(e['fallback_reasons'].values()) == counts.get('fallback', 0), 'fallback reason count')
    close(e['fallback_rate'], counts.get('fallback', 0)/n)
    good_latency = e['rpc_responses_over_target'] == 0
    require(good_latency == (e['rpc_receive_to_response_ms']['max'] <= e['rpc_response_target_ms']),
            'response-limit count and maximum disagree')
    expected = dict(rpc_and_physical_completion=True, met_response_target=good_latency)
    if policy == 'dispatch_inspired':
        expected.update(flow_plan_observed=counts.get('flow_plan_successes', 0) > 0,
                        live_multitruck_match_observed=counts.get('live_multitruck_matches', 0) > 0)
        require(expected['live_multitruck_match_observed'] == (e['first_live_multitruck_match'] is not None),
                'multi-truck match witness mismatch')
    elif policy == 'rolling':
        expected['multi_step_trial_observed'] = counts.get('retained_multi_step_requests', 0) > 0
    elif policy == 'bg128':
        require(e['actor_sampling_rng']['draws'] == counts['sampled'] == n, 'Actor draws')
    else:
        expected['candidate_comparison_observed'] = counts.get('planned', 0) > 0
    expected['passed'] = all(expected.values())
    require(expected == row['engineering_acceptance'], 'engineering gate mismatch')
    return dict(policy=row['policy_kind'], decisions=n, compliant_windows=sum(compliance),
                cost=p['combined_cost'], complete_task=p['complete_task'], python_fallbacks=counts.get('fallback', 0),
                over_response_target=e['rpc_responses_over_target'], engineering_passed=row['engineering_acceptance']['passed'])


def verify(run):
    closed = json.loads((run/'suite_complete.json').read_bytes())
    suite = json.loads(checked(closed['suite']))
    report = json.loads(checked(closed['evaluation']))
    ledger = json.loads((run/'budget.json').read_bytes())
    checked(closed['protected_core'])
    require(closed['campaign_id'] == suite['campaign_id'] == report['campaign_id'] == run.resolve().name
            and closed['physical_shifts'] == 5 and closed['training_shifts'] == closed['optimizer_updates'] == 0
            and report['method'] == suite['method'] and report['optimizer_updates'] == 0
            and closed['protected_core'] == suite['protected_core'], 'campaign seal differs')
    require(closed['status'] == report['status'] == 'COMPLETE' and report['completed_shifts'] == 5,
            'incomplete five-arm suite')
    require(tuple(j['policy_kind'] for j in suite['jobs']) ==
            ('unload_greedy', 'fifo_greedy', 'rolling', 'dispatch_inspired', 'bg128'), 'five distinct policies')
    for source in suite['source_pins']:
        checked(source)
    anchor = report['shifts'][0]
    rows = []
    for job, row, ref in zip(suite['jobs'], report['shifts'], report['receipts'], strict=True):
        slot = ledger['slots'][str(job['run_index'])]
        require(row == json.loads(checked(ref)) and slot['receipt'] == ref and slot['status'] == 'COMPLETE'
                and slot['identity'] == job, 'physical receipt or ledger mismatch')
        require(row['engineering_acceptance'] == report['engineering_acceptance'][row['policy_kind']],
                'suite/shift engineering result mismatch')
        fields = ('policy_kind', 'run_index', 'recording_id', 'experiment_seed', 'actor_sampling_seed', 'phase_config',
                  'scene_id', 'pair_ordinal')
        require(all(row[k] == job[k] for k in fields), 'shift identity')
        checked(job['phase_config'])
        claim = dict(line.split('=', 1) for line in checked(row['java_start_claim']).decode().splitlines())
        require(claim['config_sha256'] == claim['snapshot_sha256'] == job['phase_config']['sha256']
                and claim['master_plan_sha256'] == suite['master_plan']['sha256']
                and claim['ordinal'] == str(job['ordinal']) and claim['run_index'] == str(job['run_index'])
                and claim['campaign_id'] == suite['campaign_id'], 'Java START identity')
        require(Path(ref['path']) == Path(job['python_output'])/'evaluation_receipt.json'
                and row['episode_id'] == Path(job['event_bundle']).name
                and row['engineering']['rpc_response_target_ms'] == suite['method']['response_target_ms'],
                'receipt path, episode or response target differs')
        require(all(row['actual_rpc_configuration'].get(k) == v for k, v in job['expected_rpc_config'].items()),
                'actual RPC configuration differs from frozen job')
        require(all(row[k] == anchor[k] for k in ('experiment_seed', 'initial_state', 'actual_rpc_configuration')),
                'unmatched starting observations or scene')
        rows.append(verify_shift(row))
    return dict(campaign=suite['campaign_id'], verified_physical_shifts=len(rows), shifts=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    print(json.dumps(verify(parser.parse_args().run), indent=2))
