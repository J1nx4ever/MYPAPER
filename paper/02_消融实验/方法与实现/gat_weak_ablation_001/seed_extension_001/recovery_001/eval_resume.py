"""Finish the original GAT1 s04 evaluation from its committed u32 checkpoint."""

from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import extension_run as extension
import torch
from baseline1_v1.collection import verify_durable_reference
from business_graph_summary_campaign_v1.plan_io import load_reference, reference
from business_graph_summary_campaign_v1.progress import write_progress
from efficient_validation import validation_scope
from recovery_002.storage_scope import bg128_storage
from replica_budget import _complete_receipt
from transport import per_job_transport


require = extension.parent.require
INTERRUPTED_RUN = 10049254
WINDOW_COLUMNS = ('window', 'tons', 'metal_tons', 'grade', 'empty', 'compliant', 'violation',
                  'strict_band_compliant', 'center_deviation', 'exceedance', 'continuous_cost')


def _check_receipts(cfg, phase_ref, rows):
    """Bind each completed ledger receipt to the original physical files and seeds."""
    verified = {}
    for job in cfg['jobs']:
        row = rows[str(job['run_index'])]
        if row['status'] != 'COMPLETE':
            continue
        receipt = row['receipt']
        _complete_receipt(job, phase_ref, row['dataset_ref'], receipt)
        refs = receipt['references']
        identity = receipt['identity']
        require(refs['config'] == phase_ref and identity['run_index'] == job['run_index']
                and identity['environment_seed'] == job['experiment_seed']
                and identity['actor_sampling_seed'] == job['actor_sampling_seed']
                and identity['recording_id'] == job['recording_id']
                and identity['config_sha256'] == phase_ref['sha256']
                and identity['raw_manifest_sha256'] == refs['raw_manifest']['sha256']
                and identity['policy_sha256'] == refs['actor']['sha256'],
                'ledger receipt identity differs from frozen slot')
        for ref in refs.values():
            if ref['path'] not in verified:
                verified[ref['path']] = reference(ref['path'])
            require(verified[ref['path']] == ref, 'receipt source changed: ' + ref['path'])
        verify_durable_reference(refs['raw_manifest'])
        path = Path(refs['raw_manifest']['path']).parent / 'receipt.json'
        require(extension.parent.read(path) == receipt['raw_receipt'], 'physical receipt bytes differ')


def _sealed_evaluation(runtime, cfg, phase_ref, checkpoint, rows, fingerprints):
    output = Path(cfg['python_output'])
    report_ref = reference(output / 'evaluation_complete.json')
    report = load_reference(report_ref)
    jobs = cfg['jobs']
    require(report['scenario_id'] == cfg['scenario_id'] and report['phase_config'] == phase_ref
            and report['checkpoint_provenance'] == checkpoint and report['fingerprints'] == fingerprints
            and report['n_shifts'] == len(jobs) == extension.parent.EVAL_COUNT
            and report['training_state_unchanged'] and report['evaluation_optimizer_updates'] == 0
            and report['policy_arm'] == cfg['policy_arm'] == 'new', 'sealed evaluation binding differs')
    for job, item, source in zip(jobs, report['per_episode'], report['source_refs'], strict=True):
        row = rows[str(job['run_index'])]
        require(row['status'] == 'COMPLETE' and row['dataset_ref'] == source
                and row['receipt']['identity'] == item['identity']
                and row['receipt']['transitions'] == item['transitions']
                and row['receipt']['references']['actor'] == report['actor_artifact'],
                'sealed evaluation row differs from its physical receipt')
        # The published JSON sorts keys. Restore the evaluator's original CSV column order.
        windows = item['physical']['windows']
        require(all(set(window) == set(WINDOW_COLUMNS) for window in windows), 'window columns differ')
        item['physical']['windows'] = [{key: window[key] for key in WINDOW_COLUMNS} for window in windows]
    proof = extension.parent.read(output / 'validation_summary.json')
    require(proof['exact_replays'] == len(jobs), 'sealed phase exact replay count differs')
    summary = runtime.build_report(report, expected_complete_shifts=len(jobs))
    summary.update(policy_arm=cfg['policy_arm'], scenario_id=cfg['scenario_id'],
                   scenario_evidence=report['scenario_evidence'])
    folder = output / 'reports'
    require(all((folder / name).is_file() for name in ('summary.json', 'per_shift.csv', 'per_window.csv', 'README.md')),
            'incomplete sealed report publication')
    # The frozen writer byte-compares existing files, including both paired-slot CSVs.
    published = runtime.write_reports(summary, report, folder)
    return dict(checkpoint=checkpoint, reports=published, report=report_ref)


def _audit(suite, phases, ledger, runtime):
    require(suite['unit'] == 'gat1-s04' and len(suite['jobs']) == extension.CAP
            and len(phases) == extension.parent.UPDATES + len(extension.parent.EVAL_CASES),
            'recovery expects the frozen GAT1 s04 campaign')
    runtime._check_sources(suite)
    state = ledger.snapshot()
    rows = state['jobs']
    jobs = suite['jobs']
    states = [rows[str(job['run_index'])]['status'] for job in jobs]
    completed = states.count('COMPLETE')
    unresolved = states.count('CLAIMED')
    require(completed > 0 and states[:completed] == ['COMPLETE'] * completed
            and states[completed:] == ['CLAIMED'] * unresolved + ['PLANNED'] * states.count('PLANNED')
            and unresolved <= 1 and (not unresolved or jobs[completed]['run_index'] == INTERRUPTED_RUN)
            and jobs[completed - 1]['run_index'] >= INTERRUPTED_RUN - 1,
            'physical ledger boundary differs from this recovery')
    commits = state['campaigns'][suite['campaign_id']]['commits']
    updates = extension.parent.UPDATES
    require(set(commits) == {str(i) for i in range(1, updates + 1)}, 'incomplete training commit chain')
    arm = Path(suite['arms'][0]['python_output'])
    initial = load_reference(reference(arm / 'initial_checkpoint.json'))
    checkpoint = runtime._require_reference(initial)
    for cfg, ref in zip(phases, suite['phase_configs'], strict=True):
        _check_receipts(cfg, ref, rows)
        if cfg['policy_arm'] == 'train':
            update = cfg['arm_update']
            directory = arm / 'updates' / f'{update:04d}'
            commit = commits[str(update)]
            prepared = extension.parent.read(directory / 'prepared.json')
            intent = prepared['intent']
            require(intent == extension.parent.read(directory / 'intent.json')
                    and intent['parent_checkpoint'] == checkpoint and intent['phase_config'] == ref
                    and intent['campaign'] == reference(suite['suite_path']) and intent['update'] == update
                    and intent['run_indices'] == [job['run_index'] for job in cfg['jobs']]
                    and commit['job_ids'] == [str(index) for index in intent['run_indices']]
                    and intent['sources'] == [rows[index]['dataset_ref'] for index in commit['job_ids']]
                    and intent['batch_id'] == commit['batch_id'] and commit['checkpoint_ref'] == prepared['checkpoint']
                    and extension.parent.read(Path(cfg['python_output']) / 'validation_summary.json')['exact_replays']
                    == len(cfg['jobs']), 'sealed training phase or checkpoint lineage differs')
            checkpoint = runtime._check_saved_update(suite, prepared, directory, intent, update)

    model = runtime._load_new(suite, checkpoint, updates)
    fingerprints = dict(actor=runtime.digest(model.actor), critic=runtime.digest(model.critic))
    sealed = []
    for cfg, ref in zip(phases[updates:], suite['phase_configs'][updates:], strict=True):
        if not (Path(cfg['python_output']) / 'validation_summary.json').is_file():
            break
        sealed.append(_sealed_evaluation(runtime, cfg, ref, checkpoint, rows, fingerprints))
    require(len(sealed) >= 5, 'missing one of the five pre-interruption evaluation phases')
    return initial, checkpoint, sealed, dict(Counter(states))


def inspect(suite, phases):
    """Read-only recovery preflight including every historical completion receipt."""
    torch.set_num_threads(suite['cpu_threads'])
    ledger = extension.ExtensionLedger(suite, phases)
    with bg128_storage(), extension.runtime_scope(suite) as runtime:
        initial, checkpoint, sealed, counts = _audit(suite, phases, ledger, runtime)
    return dict(unit=suite['unit'], initial_checkpoint=initial, checkpoint=checkpoint,
                sealed_evaluation_phases=len(sealed), status_counts=counts)


def resume(suite, phases):
    """Run synchronously with the campaign and RPC lane locks held by the caller."""
    torch.set_num_threads(suite['cpu_threads'])
    ledger = extension.ExtensionLedger(suite, phases)
    require(not (Path(suite['control_directory']) / 'STOP.properties').exists(), 'campaign has a STOP marker')
    with bg128_storage(), per_job_transport(), extension.runtime_scope(suite) as runtime:
        initial, checkpoint, results, counts = _audit(suite, phases, ledger, runtime)
        require('CLAIMED' not in counts and ledger.job(ledger.jobs[INTERRUPTED_RUN])['status'] == 'COMPLETE',
                'interrupted physical slot must be independently reconciled before continuing')
        updates = extension.parent.UPDATES
        offset = updates + len(results)
        for cfg, ref in zip(phases[offset:], suite['phase_configs'][offset:], strict=True):
            runtime._check_sources(suite)
            with validation_scope(cfg['jobs'], runtime) as proof:
                results.append(runtime._evaluate(suite, cfg, ref, checkpoint, ledger))
            require(proof.count['exact_replays'] == len(cfg['jobs']), 'incomplete resumed Actor admission proofs')
            runtime._same_or_save(Path(cfg['python_output']) / 'validation_summary.json', proof.count)
        state = ledger.resume()
        require(Counter(row['status'] for row in state['jobs'].values()) == {'COMPLETE': extension.CAP}
                and len(state['campaigns'][suite['campaign_id']]['commits']) == updates
                and len(results) == len(extension.parent.EVAL_CASES)
                and reference(extension.parent.CORE) == suite['protected_core'], 'final training/evaluation inventory differs')
        path = Path(suite['report_directory']) / 'paired_evaluation.json'
        runtime._same_or_save(path, extension._evaluation_report(suite, phases[updates:], results, checkpoint))
        complete = dict(status='COMPLETE', suite=reference(suite['suite_path']), initial_checkpoint=initial,
                        new_checkpoint=checkpoint, training_updates=updates, training_shifts=extension.TRAIN,
                        evaluation_shifts=extension.EVAL, evaluation_optimizer_updates=0,
                        phase_results=results, comparison=reference(path))
        runtime._same_or_save(Path(suite['python_output']) / 'suite_complete.json', complete)
    write_progress(suite, status='COMPLETE', training_updates=updates, completed_shifts=extension.CAP)
    extension.runner._status(suite, 'COMPLETE', completed_shifts=extension.CAP, training_updates=updates)
    return complete
