"""Authenticate the completed discovery batch and screen its failures offline."""
from collections import deque
import math
from pathlib import Path
import sys

import torch

from discovery_common import (CAMPAIGN, CORE, CORE_SHA, COUNT, OUTPUT, PLAN, REPORT, SUITE, check_package,
    checked, read, reference, require, sha, write)
from discovery_budget import LEDGER, budget_snapshot
from baseline1_v1.collection import verify_durable_reference
from business_graph_summary_v1.admission import frozen_actor
from grade_event_memory_v1.events import build_memory
from grade_event_memory_v1.journal import read_journal
from physical_realization_v1.risk import fifo_risk
from python.constrained_batch import load_dataset


def broad_bounds(fifo, q0, m0, capacity):
    """Existing optimistic FIFO envelope with the earlier 0.1t allowance."""
    q, m, left = q0, m0, capacity
    for _,remaining,grade in fifo:
        forced = min(left,remaining)
        q += forced
        m += forced*grade
        left -= forced
    if q <= 1e-9:
        return 0.0, 1.0
    lower = min(m/q, m/(q+left))
    upper = max(m/q, (m+left)/(q+left))
    return max(0.,(lower*q-.1)/(q+.1)), min(1.,(upper*q+.1)/max(q-.1,1e-9))


def first_exclusion(rows, tape, grade):
    fifo = deque()
    q = m = 0.0
    for event in tape:
        if event.kind == 'WINDOW_CLOSE' or event.window!=0:
            break
        if event.kind == 'UNLOAD':
            fifo.append([event.batch_id,event.tons,event.grade])
        elif event.kind == 'PROCESS':
            require(fifo and fifo[0][0]==event.batch_id,'physical FIFO order differs')
            fifo[0][1] -= event.tons
            if fifo[0][1]<=1e-8:
                fifo.popleft()
            q += event.tons
            m += event.metal
        elif event.kind == 'DECISION':
            index = event.decision_sequence
            state = rows[index]['observation']
            p = math.fsum(b[1] for b in fifo)
            require(max(abs(float(state[4])*5000-q),abs(float(state[5])*5000-m),
                        abs(float(state[1300])*500-p)) < .02, 'raw observation/journal mass mismatch')
            require(float(state[1304])==1.0, 'expected original 2500tph crusher')
            capacity = (7200.-event.time)*2500/3600
            lower,upper = broad_bounds(fifo,q,m,capacity)
            comparison = fifo_risk(state)
            if grade < .26:
                if upper >= .26-1e-12 and not comparison.recovery_excluded:
                    continue
                kind = 'broad_fifo' if upper < .26-1e-12 else 'visible_support_outward_float32'
            elif lower > .28+1e-12:
                kind = 'broad_fifo'
            else:
                continue
            return dict(decision=index,time=event.time,kind=kind,
                bound=[lower,upper],visible_support_upper=comparison.grade_upper)
    return None


def audit():
    suite,cfg = check_package()
    done = read(OUTPUT/'suite_complete.json')
    require(done['campaign']==reference(SUITE) and done['physical_shifts']==COUNT and
            done['optimizer_updates']==done['training_updates']==done['gradient_updates']==0 and
            done['parent_artifacts_before']==done['parent_artifacts_after']==suite['parent_artifacts'],
            'completion scope or frozen checkpoint differs')
    result_path = REPORT/'discovery_result.json'
    result = read(result_path)
    require(result['status']=='DISCOVERY_COMPLETE' and result['sealed_evaluation']==done['reports']['evaluation'],
            'catalog does not bind the completed evaluation')
    for ref in done['reports'].values():
        checked(ref)
    eval_receipt = read(checked(done['reports']['receipt']))
    require(eval_receipt['model_state_before']==eval_receipt['model_state_after'], 'frozen runtime changed')
    state = budget_snapshot()
    require(state['maximum_total_shifts']==suite['maximum_total_if_approved'], 'merged ledger capacity differs')
    catalog = {row['run_index']:row for row in result['all_shifts']}
    observed_failed = set()
    total_transitions = 0
    for job in cfg['jobs']:
        run = job['run_index']
        slot = state['jobs'][str(run)]
        require(slot['status']=='COMPLETE' and slot['identity']==job, 'incomplete discovery slot')
        receipt = slot['receipt']
        refs = receipt['references']
        manifest = verify_durable_reference(refs['raw_manifest'])
        require(manifest['provenance']['configuration']==suite['phase_config'], 'raw RPC phase binding differs')
        total_transitions += receipt['transitions']
        for ref in refs.values():
            checked(ref)
        j = read_journal(checked(refs['journal']))
        require(j.header['run_index']==run and j.header['seed']==job['experiment_seed'] and
                j.header['config_sha256']==suite['phase_config']['sha256'] and
                j.header['recording_id']==cfg['recording_id'], 'material journal identity differs')
        memory = build_memory(j.events)
        rank = catalog[run]
        windows = rank['windows']
        for actual,record in zip(memory.settlements,windows):
            require(actual.window==record['window'] and math.isclose(actual.tons,record['tons'],abs_tol=1e-8) and
                    math.isclose(actual.metal,record['metal_tons'],abs_tol=1e-8), 'sealed window Q/M differs')
        if not windows[0]['compliant']:
            observed_failed.add(run)
    require(observed_failed=={row['run_index'] for row in result['first_window_failed_shifts']},
            'catalog filtered one or more failed first windows')
    torch.set_num_threads(2)
    actor,digest = frozen_actor(checked(eval_receipt['actor']))
    require(digest==done['fingerprints']['actor']==read(checked(suite['selection']))['selected_actor_policy_digest'],
            'executed Actor differs from original u13')
    sys.path.insert(0,str(PLAN.parent/'followup_001'))
    from screen_other_failures import screen
    from audit_actions import replay_actor

    screened = []
    for row in result['first_window_failed_shifts']:
        run = row['run_index']
        job = state['jobs'][str(run)]['identity']
        receipt = state['jobs'][str(run)]['receipt']
        refs = receipt['references']
        grade = row['windows'][0]['grade']
        rows, *_ = load_dataset(checked(refs['raw_manifest']).parent,require_real=True)
        try:
            tape = read_journal(checked(refs['journal'])).events
            lock = first_exclusion(rows,tape,grade) if grade is not None else None
            boundary = lock['decision'] if lock else sum(e.kind=='DECISION' and e.window==0 for e in tape)
            candidates, skipped = screen(rows,boundary,grade<.26) if grade is not None else ([],boundary)
            selected = candidates[0] if candidates else None
            if selected:
                actor_cut = replay_actor(actor,rows,job['actor_sampling_seed'],selected['decision'])
                require(actor_cut['executed_action']==selected['original_action'],'historical actor action differs')
                selected = dict(selected,actor=actor_cut,substitute_is_argmax=selected['substitute_action']==actor_cut['argmax_actor_action'])
            screened.append(dict(run_index=run, grade=grade, kind=row['kind'],
                first_observed_exclusion=lock, screening_stop_decision=boundary,
                insufficient_factual_projection=skipped, qualifying_candidates=len(candidates),
                selected=selected,candidates=candidates))
        finally:
            rows.close()
    require(sha(CORE)==CORE_SHA,'protected PPO changed')
    write(REPORT/'completion_audit.json',dict(status='PASS_DISCOVERY_100_SEALED',result=reference(result_path),
        completion=reference(OUTPUT/'suite_complete.json'),ledger=reference(LEDGER),physical_shifts=COUNT,
        global_merged_shifts=state['maximum_total_shifts'],first_window_failures=len(observed_failed),
        optimizer_updates=0, transitions_verified=total_transitions, source_actor=eval_receipt['actor'],
        screened_failures=screened, screening_status='offline_nominal_only', actual_substitutions=0,
        protected_core=reference(CORE)))
    print(f'PASS {COUNT} complete shifts; first-window failures={len(observed_failed)};'
          f' cases with nominal candidates={sum(x["selected"] is not None for x in screened)}')


if __name__=='__main__':
    audit()
