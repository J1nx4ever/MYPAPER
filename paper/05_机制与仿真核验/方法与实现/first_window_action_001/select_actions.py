"""Rank prospectively visible candidate projections; pair facts only for interpretation."""
from contextlib import ExitStack
import csv
import math

import numpy as np
import torch

from common import *
from business_graph_summary_campaign_v1.collection import OnlineActor
from business_graph_summary_v1.admission import frozen_actor
from grade_event_memory_v1.journal import read_journal
from physical_realization_v1.engine import estimate
from physical_realization_v1.observation import adapt_observation
from python.constrained_batch import load_dataset


def load_source(job, audit):
    journal_path = checked(audit['material_journal'])
    require(journal_path == Path(job['material_journal']), 'journal path differs')
    raw_path = checked(audit['raw_manifest'])
    rows, episodes, *_ = load_dataset(raw_path.parent, require_real=True)
    require(len(episodes) == 1 and episodes[0]['experiment_run_index'] == job['run_index'],
            'historic trajectory identity differs')
    return rows, episodes[0], read_journal(journal_path)


def summary(journal):
    windows = []
    for close in (e for e in journal.events if e.kind == 'WINDOW_CLOSE'):
        sources = {}
        for source in (0, 1, 2):
            events = [e for e in journal.events if e.kind == 'PROCESS' and
                      e.window == close.window and e.source_id == source]
            sources[str(source)] = dict(q=math.fsum(e.tons for e in events),
                                        m=math.fsum(e.metal for e in events))
        windows.append(dict(window=close.window, q=close.window_q, m=close.window_m,
                            grade=close.window_grade, sources=sources))
    return windows


def select():
    torch.set_num_threads(2)
    (old, old_audit), (low, low_audit) = source_records()
    actor_ref = read(OLD_REPORT/'completion_independent_audit.json')['actor_artifact']
    actor, _ = frozen_actor(checked(actor_ref))
    locks_path = REPORT.parent/'next_stage_mechanism_001/replay.json'
    locks = {r['run_index']: r['earliest_observed_exclusion']
             for r in read(locks_path)['trajectories'] if r['run_index'] in CASES}
    selections, inventory = [], []
    for run in CASES:
        job = next(j for j in old['jobs'] if j['run_index'] == run)
        pair = next(j for j in low['jobs'] if j['ordinal'] == job['ordinal'])
        keys = ('experiment_seed', 'actor_sampling_seed', 'expected_rpc_config')
        require(all(job[k] == pair[k] for k in keys), 'historical counterpart differs in paired settings')
        with ExitStack() as stack:
            rows, episode, journal = load_source(job, old_audit[run])
            stack.callback(rows.close)
            other, other_episode, other_journal = load_source(pair, low_audit[pair['run_index']])
            stack.callback(other.close)
            actual, comparison = summary(journal), summary(other_journal)
            require(actual[0]['grade'] < .26 and .26 <= comparison[0]['grade'] <= .28, 'failed/success role differs')
            lock = locks[run]
            options = []
            for index in range(lock['decision_sequence']):
                row = rows[index]
                adaptation = adapt_observation(row['observation'])
                if adaptation.snapshot is None:
                    continue
                effects = estimate(adaptation.snapshot, adaptation.candidates).comparisons
                factual = effects[row['action']]
                if not factual.valid or factual.projection.windows[0].grade is None:
                    continue
                baseline = factual.projection.windows[0].grade
                for candidate in effects:
                    action = candidate.action
                    if action == row['action'] or action not in (0, 1) or not candidate.valid:
                        continue
                    window = candidate.projection.windows[0]
                    own = candidate.candidate_self[0].q
                    if window.grade is None:
                        continue
                    # Reserve 0.002 margin to each edge of the official band.
                    eligible = bool(baseline < .26 and .262 <= window.grade <= .278 and own >= 30)
                    entry = dict(historical_run=run, decision=index, time=row['action_time_seconds'],
                        truck=row['truck_id'], factual=row['action'], alternative=action,
                        candidate_tons=float(row['observation'][23+action])*193,
                        nominal_factual_grade=baseline, nominal_alternative_grade=window.grade,
                        nominal_alternative_processed_tons=own, eligible=eligible)
                    inventory.append(entry)
                    if eligible:
                        options.append(entry)
            require(bool(options), 'no eligible single-action candidate')
            chosen = max(options, key=lambda x: (x['time'], x['decision'], -abs(x['nominal_alternative_grade']-.27), -x['alternative']))
            index = chosen['decision']
            state = rows[index]['observation']
            require(state[19+chosen['alternative']] == 1 and state[23+chosen['alternative']] > 0,
                    'frozen candidate is not legal')
            behavior = OnlineActor(actor, [job])
            info = episode['start_info']
            for i in range(index+1):
                row = rows[i]
                predicted, log_prob = behavior.sample_with_info(row['observation'], row['mask'], info)
                require(predicted == row['action'] and log_prob == row['old_log_prob'],
                        'historical action replay differs')
                require(behavior.last_metadata['actor_rng_after_sha256'] == row['actor_rng_after_sha256'],
                        'historical private Actor RNG differs')
                info = row['settlement_info']
            tensor = torch.from_numpy(state.copy())[None, :]
            with torch.no_grad():
                probabilities = behavior.actor.distribution(tensor, behavior._summary_cache.prefix([index])).probs[0].tolist()
            divergence = next((i for i in range(min(len(rows), len(other)))
                if not np.array_equal(rows[i]['observation'], other[i]['observation']) or rows[i]['action'] != other[i]['action']), None)
            selections.append(dict(**chosen, state_sha256=state_sha(state), legal_mask=state[19:23].tolist(),
                actor_probabilities=probabilities, lock_decision=lock['decision_sequence'], lock_time=lock['time'],
                historical_failed=dict(run_index=run, raw=old_audit[run]['raw_manifest'],
                    journal=old_audit[run]['material_journal'], windows=actual),
                historical_success=dict(run_index=pair['run_index'], raw=low_audit[pair['run_index']]['raw_manifest'],
                    journal=low_audit[pair['run_index']]['material_journal'], windows=comparison),
                historical_pair_first_divergence=divergence,
                historical_pair_length=[len(rows), len(other)]))
    REPORT.mkdir(parents=True, exist_ok=True)
    with (REPORT/'candidate_screen.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(inventory[0]))
        writer.writeheader()
        writer.writerows(inventory)
    return save(REPORT/'candidate_selection.json', dict(
        status='HISTORICAL_CANDIDATES_FROZEN', selection_rule='latest eligible decision, then nearest nominal grade to 0.27',
        selection_basis='original observation plus original action; no future material input',
        minimum_nominal_own_processed_tons=30, nominal_grade_band=[.262,.278],
        projection_scope='fixed observed load workload and nominal route speed, modeled dump/buffer/FIFO',
        actor=actor_ref, source_lock_report=reference(locks_path),
        candidates=selections, inventory=reference(REPORT/'candidate_screen.csv')))


if __name__ == '__main__':
    print(json.dumps(select(), ensure_ascii=False))
