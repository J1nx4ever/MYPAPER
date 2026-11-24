"""Verify the retained first DISPATCH witness in each of the first 60 formal slots."""
from collections import Counter
from hashlib import sha256
from itertools import product
import json
from math import fsum, isclose
from pathlib import Path
import sys

from reproduce import ROOT, checked_archive


FORMAL = Path(__file__).resolve().parents[1] / 'paired_v2_100_001'
RECHECK = FORMAL.parent / 'all_recheck_001'
sys.path[:0] = [str(FORMAL), str(RECHECK)]

from algorithm_review import independent_loss
import verify_results as audit
from formal_plan import OUTPUT, SUITE, load_suite
from traditional_dispatch_v1.decision import _cost
from traditional_dispatch_v2 import decision, dispatch_policy as dispatch


def recorded_witnesses(suite, digest):
    jobs = suite['jobs']
    selected = [job for job in jobs if job['pair_ordinal'] < 60 and job['policy_kind'] == 'dispatch_inspired']
    audit.require(len(selected) == 60, 'the formal roster must supply 60 DISPATCH receipts')
    entry, ledger = audit._ledger(OUTPUT, {job['run_index']: job for job in jobs}, False)
    suite_ref = dict(path=str(SUITE), sha256=sha256(SUITE.read_bytes()).hexdigest())
    audit._verify_ledger_scope(OUTPUT, dict(suite=suite_ref, campaign_id=suite['campaign_id']), entry, ledger, jobs)
    witnesses, refs, counters = [], [], Counter()
    for job in selected:
        slot = ledger[str(job['run_index'])]
        audit.require(slot['status'] == 'COMPLETE', f"incomplete run {job['run_index']}")
        ref = slot['receipt']
        row = json.loads(audit.checked(ref))
        audit._verify_phase(suite, job)
        audit._verify_receipt(suite, job, row, ref, slot, actor_digest=None)
        initial = row['initial_state']
        witness = row['engineering']['first_live_multitruck_match']
        audit.require(initial['observation_sha256'] == digest and witness['decision_sequence'] == 0
                      and all(initial[k] == witness[k] for k in ('truck_id', 'sim_time_seconds', 'mask')),
                      'the first retained witness must use the byte-identical initial observation')
        witnesses.append(witness)
        refs.append(ref)
        counters.update(row['engineering']['action_counts'])
    audit.require(all(item == witnesses[0] for item in witnesses), 'initial live witnesses differ')
    return witnesses[0], refs, counters


def check_feasible(state, options):
    available = state._available_tons.copy()
    waste = state._waste_committed()
    for option in options:
        available[option.site] -= option.tons
        assert available[option.site] >= 0. and option.tons > 0.
        if not state._loads[option.site].ore:
            assert waste < dispatch.WASTE_STOP_TONS
            waste += option.tons


def project_fifo(state, sites, truck_ids):
    branch = state.clone()
    for site, expected in zip(sites, truck_ids, strict=True):
        assert branch.current_truck == expected
        branch.apply_action(site)
        branch.advance()
    prefix_only = branch.clone()
    prefix_only._move(check_deadline=None, dispatch=False)
    committed = prefix_only.score_inputs()
    # The candidate prefixes differ. All candidates then use the same continuation rule.
    while not branch.finished:
        branch.apply_action(decision._tail_action(branch, lambda: None))
        branch.advance()
    forecast = branch.score_inputs()
    return dict(prefix_only_window1_ore_tons=committed.fifo_q[0],
                prefix_only_window1_grade=(committed.fifo_m[0] / committed.fifo_q[0] if committed.fifo_q[0] else None),
                combined_cost=_cost(forecast, unload=False),
                window_tons=list(forecast.fifo_q),
                window_grades=[m / q if q else None for q, m in zip(forecast.fifo_q, forecast.fifo_m, strict=True)],
                ore_tons=forecast.ore_tons, waste_tons=forecast.waste_tons, wait_vehicle_seconds=forecast.wait_seconds)


def replay(obs, witness, settings):
    policy = dispatch.DispatchPolicy(**{k: settings[k] for k in ('planning_ms', 'replan_seconds', 'match_trucks')})
    selection = policy.choose(obs, obs[19:23])
    state = dispatch.NominalState.from_observation(obs)
    trucks = dispatch._ready_trucks(state, settings['match_trucks'], lambda: None)
    ids = [truck.truck_id for truck in trucks]
    targets = dispatch._targets(policy.plan)
    servers, bookings = dispatch._shovel_calendars(state)
    assert state.current_truck == witness['truck_id'] and state.now == witness['sim_time_seconds']
    assert [bool(value) for value in obs[19:23]] == witness['mask']
    allocations = []
    scored = 0
    max_score_error = 0.
    for group in (trucks[:1], trucks):
        for options in product(*(truck.choices for truck in group)):
            check_feasible(state, options)
            retained = dispatch._reserve(options, state._available_tons, state._loads, state._waste_committed())
            assert retained == options
            score = independent_loss(options, group, targets, servers, bookings)
            implementation = dispatch._loss(options, group, targets, [0.] * 4, servers, bookings)
            max_score_error = max(max_score_error, abs(score - implementation))
            assert isclose(score, implementation, rel_tol=0, abs_tol=1e-8)
            scored += 1
            if group is trucks:
                sites = tuple(option.site for option in retained)
                allocations.append(dict(sites=sites, matching_loss=score, implementation_loss=implementation,
                                        fifo=project_fifo(state, sites, ids)))
    expected = min(allocations, key=lambda x: (x['implementation_loss'], x['sites']))
    expected_fifo = min(allocations, key=lambda x: (x['fifo']['combined_cost'], x['sites']))
    equal_tonnage_prefix = [x for x in allocations if x['fifo']['prefix_only_window1_grade'] is not None
                            and .26 <= x['fifo']['prefix_only_window1_grade'] <= .28
                            and isclose(x['fifo']['prefix_only_window1_ore_tons'],
                                        expected['fifo']['prefix_only_window1_ore_tons'], abs_tol=1e-7)]
    same_mass_compliant = min(equal_tonnage_prefix, key=lambda x: (x['matching_loss'], x['sites']))
    assert all(item['sites'][0] == expected['sites'][0] for item in allocations
               if isclose(item['matching_loss'], expected['matching_loss'], rel_tol=0, abs_tol=1e-8))
    assert selection.kind == 'planned' and selection.lp_status == 'optimal' and not selection.timed_out
    assert scored == witness['assignments_scored'] == selection.assignments_scored
    assert len(ids) == witness['matched_trucks'] == selection.matched_trucks
    assert expected['sites'][0] == witness['action'] == selection.action
    roots = []
    for root in range(4):
        options = [alloc for alloc in allocations if alloc['sites'][0] == root]
        roots.append(dict(site=root, best_matching_loss=min(x['matching_loss'] for x in options),
                          best_fifo_nominal_cost=min(x['fifo']['combined_cost'] for x in options)))
    return dict(truck_order=ids, scores_recomputed=scored, three_truck_combinations=len(allocations),
                maximum_score_difference=max_score_error, best_under_v2=expected,
                best_under_nominal_fifo=expected_fifo, same_mass_compliant_prefix=same_mass_compliant, best_by_root=roots,
                online_witness=witness, all_combinations=allocations)


if __name__ == '__main__':
    if not __debug__:
        raise RuntimeError('run with assertions enabled')
    suite = load_suite()
    observations, archive_digest, _ = checked_archive()
    obs = observations[0]
    observation_digest = sha256(obs.tobytes()).hexdigest()
    witness, receipts, counts = recorded_witnesses(suite, observation_digest)
    result = replay(obs, witness, suite['method'])
    result.update(campaign_id=suite['campaign_id'], slots=60, matching_witnesses_verified=60,
                  distinct_online_observation_frames_replayed=1, observation_sha256=observation_digest,
                  archived_input_sha256=archive_digest, receipts=receipts,
                  live_multitruck_matches=int(counts['live_multitruck_matches']),
                  other_multitruck_matches_without_replay_inputs=int(counts['live_multitruck_matches']) - 60,
                  planned_requests=int(counts['planned']),
                  later_action_status='only aggregate counts retained',
                  fifo_analysis='nominal event-model counterfactual with shared continuation')
    print(json.dumps(result, indent=2, allow_nan=False))
