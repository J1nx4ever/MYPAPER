"""Read sealed action/RPC/material records and independently reconstruct two pairs."""
from collections import deque
from contextlib import ExitStack
import math
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CORE, CORE_SHA, REPORT, RUN, checked, read, reference, require, save, sha, state_sha
from adaptation_v1_1.collect import _rng_sha
from business_graph_summary_v1.admission import frozen_actor
from grade_event_memory_v1.events import build_memory
from grade_event_memory_v1.journal import read_journal
from physical_realization_v1.engine import estimate
from physical_realization_v1.observation import adapt_observation
from python.constrained_batch import load_dataset
from relation_gat_ego_v1.adapter import validate_states
from runtime import canonical_events, physical_info, through_cut
from single_critic_task_query_v1.features_cache import SummaryFeatureCache

OUT = REPORT / 'followup_001'
DEADLINE = 7200.0
RATE = 2500.0 / 3600.0


def load_rows(ref, stack):
    rows, episodes, _, _, _ = load_dataset(checked(ref).parent, require_real=True)
    stack.callback(rows.close)
    require(len(episodes) == 1, 'expected one shift')
    return rows, episodes[0]


def replay_actor(actor, rows, seed, cut, override=None):
    """Run the saved Actor on each received raw row using its original CPU sampler."""
    generator = torch.Generator(device='cpu')
    generator.manual_seed(seed)
    features = SummaryFeatureCache(feature_config=actor.feature_config)
    audit = None
    with torch.no_grad():
        for index, row in enumerate(rows):
            raw = row['observation']
            mask = np.asarray(row['mask'])
            require(np.array_equal(raw[19:23], mask) and mask[row['action']] == 1, 'mask differs')
            features.append(raw)
            distribution = actor.distribution(torch.from_numpy(raw.copy())[None,:], features.prefix([index]))
            require(_rng_sha(generator) == row['actor_rng_before_sha256'] and
                    row['actor_sampling_draw_index'] == index and row['actor_sampling_seed'] == seed,
                    'Actor RNG before draw or draw index differs')
            draw = torch.multinomial(distribution.probs, 1, replacement=True, generator=generator).reshape(1)
            sampled = int(draw.item())
            log_prob = float(distribution.log_prob(draw).item())
            forced = override is not None and cut == index
            expected_action = override if forced else sampled
            expected_log = 0.0 if forced else log_prob
            require(_rng_sha(generator) == row['actor_rng_after_sha256'] and
                    row['action'] == expected_action and row['old_log_prob'] == expected_log and
                    row.get('sampled_actor_action', row['action']) == sampled and
                    row.get('sampled_actor_log_prob', row['old_log_prob']) == log_prob and
                    row.get('forced_action', False) is forced,
                    'sampled or executed action does not replay')
            if index == cut:
                probabilities = distribution.probs[0].tolist()
                audit = dict(probabilities=probabilities, argmax_actor_action=int(np.argmax(probabilities)),
                    sampled_actor_action=sampled, sampled_actor_log_prob=log_prob,
                    sampled_actor_probability=probabilities[sampled], executed_action=expected_action,
                    executed_behavior_log_prob=expected_log, forced_action=forced,
                    rng_before= row['actor_rng_before_sha256'], rng_after=row['actor_rng_after_sha256'],
                    draw_index=index, state_sha256=state_sha(raw))
    require(audit is not None, 'missing intervention row')
    return audit


def mass(events):
    return dict(q=math.fsum(e.tons for e in events), m=math.fsum(e.metal for e in events))


def fifo_at_unload(events, unload):
    """Reconstruct material ahead immediately before the trip is enqueued."""
    fifo = deque()
    for event in events:
        if event.seq >= unload.seq:
            break
        if event.kind == 'UNLOAD':
            fifo.append([event.batch_id, event.tons, event.grade])
        elif event.kind == 'PROCESS':
            require(fifo[0][0] == event.batch_id, 'FIFO head changed')
            fifo[0][1] -= event.tons
            if fifo[0][1] <= 1e-8:
                fifo.popleft()
    return dict(ahead_batches=len(fifo), fifo_position=len(fifo)+1, ahead_tons=math.fsum(b[1] for b in fifo),
                ahead_metal=math.fsum(b[1]*b[2] for b in fifo), batches_ahead=list(fifo))


def observed_timing(rows, markers, cut, unload, truck):
    base = 27+120*truck
    cuts = []
    for marker in markers[cut:]:
        if marker.seq >= unload.seq: break
        raw = rows[marker.decision_sequence]['observation']
        phase = int(np.argmax(raw[base+4:base+16]))
        loaded = bool(raw[base+17])
        cuts.append((marker.time, phase, loaded, float(raw[base+84]), float(raw[base+87])))
    first_loaded = next((i for i, values in enumerate(cuts) if values[2]), None)
    last_unloaded = cuts[-1][0] if first_loaded is None else cuts[first_loaded-1][0]
    first_at_dump = next((i for i, values in enumerate(cuts) if values[1] in (6,7,8,9)), None)
    last_pre_dump = cuts[-1][0] if first_at_dump is None else cuts[first_at_dump-1][0]
    load_begins = [t-age*600 for t,p,_,age,_ in cuts if p==4]
    dump_begins = [t-age*600 for t,p,_,_,age in cuts if p==9]
    return dict(load_end_interval_seconds=[last_unloaded, unload.time if first_loaded is None else cuts[first_loaded][0]],
        dump_arrival_interval_seconds=[last_pre_dump, unload.time if first_at_dump is None else cuts[first_at_dump][0]],
        approx_load_start_seconds=float(np.median(load_begins)) if load_begins else None,
        approx_dump_service_start_seconds=float(np.median(dump_begins)) if dump_begins else None,
        snapshots_between_dispatch_and_unload=len(cuts))


def trip_trace(rows, journal, branch, cut):
    memory = build_memory(journal.events)
    events = memory.events
    markers = [e for e in events if e.kind == 'DECISION']
    truck = rows[cut]['truck_id']
    action = rows[cut]['action']
    unload, = branch['chosen_trip']['unload']
    unload = next(e for e in events if e.seq==unload['seq'])
    require(unload.kind=='UNLOAD' and unload.truck_id==truck and unload.source_id==action and
            markers[cut].seq<unload.seq, 'selected trip journal association differs')
    observed = observed_timing(rows, markers, cut, unload, truck)
    adaptation = adapt_observation(rows[cut]['observation'])
    require(adaptation.snapshot is not None, 'expected valid case projection')
    projection = estimate(adaptation.snapshot, adaptation.candidates).comparisons[action]
    require(projection.valid, 'expected selected action projection')
    job = adaptation.candidates[action].job
    packet = next(p for p in projection.projection.packets if p.key==job.key)
    proc = [e for e in events if e.kind=='PROCESS' and e.window==0]
    chosen = [e for e in proc if e.batch_id==unload.batch_id]
    remainder = [e for e in proc if e.batch_id!=unload.batch_id]
    require(chosen and all(e.interval_end<=DEADLINE+1e-8 for e in chosen), 'trip did not reach first-window PROCESS')
    actual, others = mass(chosen), mass(remainder)
    window = memory.settlements[0]
    require(abs(actual['q']+others['q']-window.tons)<1e-6 and
            abs(actual['m']+others['m']-window.metal)<1e-6, 'Q/M trip attribution does not close')
    require(abs(actual['q']-branch['chosen_trip']['first_window_processed_tons'])<1e-6,
            'processed trip amount differs from original report')
    first = min(e.interval_start for e in chosen)
    last = max(e.interval_end for e in chosen)
    fifo = fifo_at_unload(events, unload)
    return dict(run_index=branch['run_index'], branch=branch['branch'], action=action,
        dispatch_seconds=markers[cut].time,
        predicted_dump_arrival_seconds=job.arrival_time,
        predicted_unload_seconds=packet.unload_complete,
        predicted_trip_q=packet.processed[0].q,
        **observed, actual_unload_seconds=unload.time, unload_grade=unload.grade,
        batch_id=unload.batch_id, **fifo, first_process_seconds=first, last_process_seconds=last,
        unloading_to_first_process_seconds=first-unload.time, deadline_slack_seconds=DEADLINE-last,
        window0_trip=actual, window0_other_batches=others, window0_total=dict(q=window.tons,m=window.metal,grade=window.grade))


def report():
    torch.set_num_threads(2)
    done_ref = reference(REPORT/'completion_validation.json')
    completion = read(checked(done_ref))
    result_ref = completion['result']
    result = read(checked(result_ref))
    selection_ref = reference(REPORT/'candidate_selection.json')
    selection = read(checked(selection_ref))
    actor, digest = frozen_actor(checked(selection['actor']))
    assert sha(CORE)==CORE_SHA
    by_run = {r['historical_run']:r for r in selection['candidates']}
    pairs = []
    for case in result['cases']:
        decision = by_run[case['historical_failed_run']]
        cut = decision['decision']
        with ExitStack() as stack:
            history, historical_ep = load_rows(decision['historical_failed']['raw'], stack)
            old_tape = read_journal(checked(decision['historical_failed']['journal']))
            seed = history[cut]['actor_sampling_seed']
            history_policy = replay_actor(actor, history, seed, cut)
            branch_rows, policy_rows, traces = [], [], []
            for item in ('original','alternative'):
                branch = case[item]
                rows, ep = load_rows(branch['rpc_dataset'], stack)
                tape = read_journal(checked(branch['material_journal']))
                current_cut = replay_actor(actor, rows, seed, cut, decision['alternative'] if item=='alternative' else None)
                require(current_cut['state_sha256']==decision['state_sha256'] and
                        current_cut['probabilities']==decision['actor_probabilities']==history_policy['probabilities'] and
                        current_cut['sampled_actor_action']==branch['policy_suggestion'],
                        'sealed decision probability or field mapping differs')
                ref_info, live_info = historical_ep['start_info'], ep['start_info']
                for i in range(cut+1):
                    require(physical_info(ref_info)==physical_info(live_info) and
                            history[i]['observation'].tobytes()==rows[i]['observation'].tobytes() and
                            (i==cut or history[i]['action']==rows[i]['action']),
                            'recorded common prefix differs')
                    ref_info, live_info = history[i]['settlement_info'], rows[i]['settlement_info']
                require(through_cut(tape.events,cut)==through_cut(old_tape.events,cut), 'common event prefix differs')
                if item=='original':
                    require(canonical_events(tape.events)==canonical_events(old_tape.events), 'original entire journal differs')
                branch_rows.append(rows)
                policy_rows.append(current_cut)
                traces.append(trip_trace(rows, tape, branch, cut))
            original_rows, alternative_rows = branch_rows
            common_draw_count = min(len(original_rows),len(alternative_rows))
            require(all(original_rows[i]['actor_rng_before_sha256']==alternative_rows[i]['actor_rng_before_sha256'] and
                        original_rows[i]['actor_rng_after_sha256']==alternative_rows[i]['actor_rng_after_sha256']
                        for i in range(common_draw_count)), 'Actor stream differs for common draw indices')
            orig, alt = traces
            deltas = {part:{k:alt[part][k]-orig[part][k] for k in ('q','m')}
                      for part in ('window0_trip','window0_other_batches','window0_total')}
            for k in ('q','m'):
                require(abs(deltas['window0_trip'][k]+deltas['window0_other_batches'][k]-deltas['window0_total'][k])<1e-6,
                        'branch delta Q/M does not close')
            pairs.append(dict(historical_failed_run=decision['historical_run'], decision=cut,
                truck=decision['truck'], actor_seed=seed, replayed_historical_decisions=len(history),
                replayed_original_decisions=len(original_rows), replayed_alternative_decisions=len(alternative_rows),
                equal_actor_rng_draw_indices=common_draw_count, environment_seed=orig_tape_seed(old_tape),
                common_event_prefix_count=len(through_cut(old_tape.events,cut)),
                policy_suggestion_legacy_meaning='sampled_actor_action before action intervention',
                original_policy=policy_rows[0], alternative_policy=policy_rows[1],
                original_trip=orig, alternative_trip=alt, window0_attribution_delta=deltas,
                same_seed_success=dict(run_index=decision['historical_success']['run_index'],
                    first_divergence=decision['historical_pair_first_divergence'],
                    window0=decision['historical_success']['windows'][0])))
    assert sha(CORE)==CORE_SHA
    return save(OUT/'action_and_material_audit.json', dict(status='PASS_SEALED_TRACE_AUDIT', pairs=pairs,
        source_result=result_ref, candidate_selection=selection_ref, completion=done_ref, actor=selection['actor'],
        actor_digest=digest, policy_probabilities='reconstructed_from_byte_identical_observations_with_original_sampling_path',
        random_coupling=dict(environment='same initial seed and observed physical/event prefix',
            after_branch='service events may consume common engine stream in a different order',
            actor='identical before/after RNG state at shared draw indices, one draw also consumed on forced row'),
        timing=dict(nominal_arrival='entrance to dump before unloading service, based on dispatch state',
            load_completion='bracketed by last sampled unloaded and first sampled loaded or unload', 
            dump_arrival='bracketed by last pre-dump snapshot and first observed dump stage or unload',
            fifo='counts and remaining tonnes before target UNLOAD by journal event order'),
        physical_shifts_added=0, optimizer_updates=0, protected_ppo=reference(CORE)))


def orig_tape_seed(tape):
    return tape.header['seed']


if __name__=='__main__':
    print(report())
