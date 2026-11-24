"""Screen other real first-window failures under their historical 1e-4 Actor."""
from contextlib import ExitStack

import torch

from audit_actions import OUT, replay_actor
from common import CORE, CORE_SHA, LOW_REPORT, REPORT, checked, read, reference, require, save, sha, source_records
from business_graph_stability_v1.admission import frozen_actor
from physical_realization_v1.engine import estimate
from physical_realization_v1.observation import adapt_observation
from select_actions import load_source, summary


def cutoff(failure):
    certificates = [
        (failure['earliest_observed_exclusion'], 'fifo_wide_accounting_envelope'),
        (failure['earliest_visible_support_low_exclusion'], 'outward_float32_low_grade_envelope'),
    ]
    valid = [(record, kind) for record, kind in certificates if record is not None]
    require(bool(valid), 'missing conditional pre-lock delimiter')
    return min(valid, key=lambda pair: (pair[0]['time'], pair[0]['decision_sequence']))


def screen(rows, limit, low_failure):
    candidates = []
    no_model = 0
    for index in range(limit):
        row = rows[index]
        state = row['observation']
        adaptation = adapt_observation(state)
        if adaptation.snapshot is None:
            no_model += 1
            continue
        effects = estimate(adaptation.snapshot, adaptation.candidates).comparisons
        factual = effects[row['action']]
        if not factual.valid or factual.projection.windows[0].grade is None:
            no_model += 1
            continue
        factual_grade = factual.projection.windows[0].grade
        factual_violates = factual_grade < .26 if low_failure else factual_grade > .28
        if not factual_violates:
            continue
        for effect in effects:
            action = effect.action
            if not effect.valid or action == row['action']:
                continue
            job = adaptation.candidates[action].job
            if job is None or job.commodity != 'ore' or not bool(state[19+action]):
                continue
            projected = effect.projection.windows[0]
            self_q = effect.candidate_self[0].q
            if projected.grade is None or self_q < 30.0 or not .262 <= projected.grade <= .278:
                continue
            candidates.append(dict(decision=index, time=float(row['action_time_seconds']), truck=row['truck_id'],
                original_action=row['action'], substitute_action=action,
                candidate_tons=job.tons, predicted_substitute_processed_tons=self_q,
                predicted_factual_window_grade=factual_grade, predicted_substitute_window_grade=projected.grade,
                predicted_dump_arrival_seconds=job.arrival_time))
    candidates.sort(key=lambda c:(-c['time'], -c['decision'], abs(c['predicted_substitute_window_grade']-.27), c['substitute_action']))
    return candidates, no_model


def main():
    torch.set_num_threads(2)
    (baseline, base_audit), (low, low_audit) = source_records()
    actor_ref = read(LOW_REPORT/'completion_independent_audit.json')['actor_artifact']
    actor, digest = frozen_actor(checked(actor_ref))
    chain_path = REPORT.parent/'next_stage_mechanism_001/replay.json'
    chain_ref = reference(chain_path)
    traces = read(checked(chain_ref))['trajectories']
    failures = sorted((r for r in traces if not r['compliant'] and r['window']==0),
                      key=lambda r:r['run_index'])
    main_failures = [r['run_index'] for r in failures if r['lr']=='3e-4']
    require(main_failures == [9950012,9950029], 'baseline case scope changed')
    output = []
    for failure in failures:
        if failure['lr'] != '1e-4':
            continue
        lock, lock_kind = cutoff(failure)
        run = failure['run_index']
        job = next(j for j in low['jobs'] if j['run_index']==run)
        control = next(j for j in baseline['jobs'] if j['ordinal']==job['ordinal'])
        require(all(job[k]==control[k] for k in ('experiment_seed','actor_sampling_seed','expected_rpc_config')),
                'historical comparison is not paired')
        with ExitStack() as stack:
            rows, _, journal = load_source(job,low_audit[run])
            other, _, other_journal = load_source(control,base_audit[control['run_index']])
            stack.callback(rows.close)
            stack.callback(other.close)
            failed_grade = summary(journal)[0]['grade']
            control_grade = summary(other_journal)[0]['grade']
            require((failed_grade < .26 or failed_grade > .28) and .26<=control_grade<=.28,
                    'source case/comparator status differs')
            count = lock['decision_sequence']
            candidates, skipped = screen(rows, count, failed_grade < .26)
            selected = candidates[0] if candidates else None
            if selected:
                trace = replay_actor(actor,rows,job['actor_sampling_seed'],selected['decision'])
                require(trace['sampled_actor_action']==selected['original_action'],
                        'screened historical action differs')
                selected = dict(selected, historical_actor_probabilities=trace['probabilities'],
                    historical_actor_argmax=trace['argmax_actor_action'],
                    historical_actor_sample=trace['sampled_actor_action'], state_sha256=trace['state_sha256'])
            output.append(dict(run_index=run, same_seed_success_run=control['run_index'],
                observed_first_window_grade=failed_grade, same_seed_success_first_window_grade=control_grade,
                lock_seconds=lock['time'], lock_kind=lock_kind, screened_pre_lock_decisions=count,
                insufficient_factual_projection=skipped, qualifying_candidates_count=len(candidates),
                selected=selected, qualifying_candidates=candidates,
                raw=low_audit[run]['raw_manifest'], journal=low_audit[run]['material_journal'],
                comparator_raw=base_audit[control['run_index']]['raw_manifest'],
                comparator_journal=base_audit[control['run_index']]['material_journal']))
    require(len(output)==6 and sha(CORE)==CORE_SHA, 'screening set or protected hash differs')
    print(save(OUT/'other_failures_screen_002.json', dict(status='PASS_NOMINAL_SCREEN_ONLY',
        primary_lr=3e-4, primary_first_window_failures_with_real_interventions=main_failures,
        screening_source_lr=1e-4, historical_actor=actor_ref, actor_digest=digest,
        candidates_are_measured_counterfactuals=False, physical_shifts_added=0, optimizer_updates=0,
        rule='pre-lock legal ore action, same-direction nominal baseline violation, substitute grade .262-.278, predicted own processed ore >=30t; latest, then closest to .27',
        cutoff_rule='earliest applicable observed exclusion across the two pre-existing FIFO certificates',
        supersedes=reference(OUT/'other_failures_screen.json'),
        conditional_projection='current inputs only; fixed visible loading workload and nominal route speeds with approximate unloading and FIFO; future dispatch feedback omitted',
        source_lock_analysis=chain_ref, failures=output)))


if __name__=='__main__':
    main()
