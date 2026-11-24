"""Admit the already-finished 9991101 shift and reconcile its one ledger row."""
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from recovery_common import (
    APPROVAL, CAMPAIGN, CONTROL, FIRST_REMAINING, LEDGER, OFFSET, OUTPUT, OUTPUT_DIR, RECONCILIATION,
    REMAINING, SUITE, SUSPENDED_RUN, TOTAL, anchor_files, original_campaign, read, reference, require,
)
from gen_budget import GenLedger, prior
from baseline1_v1.collection import _java_start_properties, verify_durable_reference
from business_graph_summary_campaign_v1.budget import _payload, _state, connection, digest
from business_graph_summary_campaign_v1.plan_io import encoded, load_phase
from business_graph_summary_recovery_v1 import admission, collection
from exact_admission import exact_online_admission
from grade_only_ppo_v2.runtime import _controller_lock
from online_v1_1.operations import _publish


SCHEMA = 'bg128-u32-generalization-reconciliation-v1'
ORIGINAL_ERROR = "ValueError('batch/online probability difference exceeds roundoff tolerance')"
TRANSITIONS = 153
SAVED_DATASET = OUTPUT_DIR / 'material' / f'run_{SUSPENDED_RUN}.h5'


def _seal(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        require(path.read_bytes() == data, 'existing recovery artifact differs: ' + str(path))
    else:
        _publish(path, data)
        path.chmod(0o444)
    return reference(path)


def _unfinished_slices(suite):
    require(not any((CONTROL / 'java_claims' / f"run_{job['run_index']}.once.claim").exists()
                    for job in suite['jobs'][OFFSET:]),
            'the frozen remaining identity range must not have started')
    require(not (CONTROL / 'ready' / f'run_{FIRST_REMAINING}.properties').exists(),
            'the next frozen job must not have received an original ready token')


def _admit(suite, output_path):
    phase = next(phase for phase in suite['phases'] if phase['scenario_id'] == 'grade_swap')
    cfg_ref = phase['phase_config']
    cfg = load_phase(cfg_ref['path'])
    job = cfg['jobs'][0]
    require(job == suite['jobs'][OFFSET-1] and job['run_index'] == SUSPENDED_RUN,
            'the reconciliation applies only to the first existing grade-swap run')
    _unfinished_slices(suite)
    source = anchor_files()
    stop_path = Path(source['original_stop']['path'])
    require(stop_path.read_text().strip() == 'reason=' + ORIGINAL_ERROR, 'original STOP changed')
    require(_java_start_properties(source['first_failed_java_claim']['path']) ==
            dict(protocol='baseline1-one-click-java-start-v1', campaign_id=CAMPAIGN,
                 master_plan_sha256=suite['master_plan']['sha256'], ordinal=str(job['ordinal']),
                 run_index=str(SUSPENDED_RUN), snapshot_sha256=cfg_ref['sha256'], config_sha256=cfg_ref['sha256']),
            'original exclusive Java start claim differs')
    folder = Path(phase['python_output'])
    raw_dir = folder / 'episodes' / f'run_{SUSPENDED_RUN}'
    manifest_path = raw_dir / 'manifest.json'
    raw_receipt_path = raw_dir / 'receipt.json'
    raw_receipt = read(raw_receipt_path)
    manifest_ref = reference(manifest_path)
    manifest = verify_durable_reference(manifest_ref)
    actor = folder / 'actor.pt'
    actor_manifest = read(Path(str(actor) + '.manifest.json'))
    actor_digest = read(Path(suite['parent_artifacts']['manifest']['path']))['actor_digest']
    source.update(raw_receipt=reference(raw_receipt_path), raw_manifest=manifest_ref,
                  raw_payloads={name: reference(raw_dir/name) for name in manifest['files_sha256']},
                  actor=reference(actor), phase_config=cfg_ref, journal=reference(job['material_journal']))
    require(manifest.get('source_kind') == 'real_anylogic_rpc'
            and manifest['episodes'] == 1 and manifest['transitions'] == TRANSITIONS
            and manifest['provenance'] == dict(recording_id=job['recording_id'], configuration=cfg_ref,
                 run_index=SUSPENDED_RUN, seed=job['experiment_seed'])
            and raw_receipt['normal_completion'] is True and raw_receipt['four_windows'] == 4
            and raw_receipt['transitions'] == TRANSITIONS and raw_receipt['run_index'] == SUSPENDED_RUN
            and raw_receipt['experiment_seed'] == job['experiment_seed']
            and raw_receipt['phase_id'] == job['phase_id'] and raw_receipt['role'] == job['role']
            and raw_receipt['phase_config'] == cfg_ref and raw_receipt['dataset_files'] == manifest['files_sha256']
            and raw_receipt['policy_version'] == manifest['policy_version'] == actor_digest
            and actor_manifest['actor_digest'] == actor_digest and actor_manifest['update_index'] == 32
            and actor_manifest['sha256'] == source['actor']['sha256'],
            'raw receipt, manifest, frozen job, or u32 policy does not bind this complete shift')
    evidence = []
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(suite['cpu_threads'])
        with exact_online_admission(evidence):
            output_manifest = Path(str(output_path) + '.manifest.json')
            if not output_path.exists():
                admitted = admission.admit_episode(raw_dir, job['material_journal'], config_path=cfg_ref['path'],
                    actor_path=actor, output_path=output_path, split=job['partition'])
                require(admitted['status'] == 'PASS_DATA_PREPARATION_ONLY'
                        and admitted['actor_or_critic_fitted'] is False and admitted['collection_started'] is False
                        and admitted['transitions'] == TRANSITIONS, 'original admission failed')
            identity = read(output_manifest)['identity']
            receipt = dict(schema=collection.RECEIPT_SCHEMA, normal_completion=True, identity=identity,
                phase_id=job['phase_id'], role=job['role'], raw_receipt=raw_receipt, run_index=SUSPENDED_RUN,
                experiment_seed=job['experiment_seed'], phase_config=cfg_ref,
                references=collection._references(cfg_ref['path'], actor, job['material_journal'],
                    manifest_path, output_path), four_windows=4, transitions=TRANSITIONS,
                training_objective=admission.TRAINING_REWARD, composite_contract=cfg['composite_contract'],
                material_evidence=True, production_guarantee='pending', production_failures_retained=True)
            reread = collection.load_admitted_episode(receipt)
    finally:
        torch.set_num_threads(threads)
    require(reread['identity'] == identity and len(reread['actions']) == TRANSITIONS
            and reread['states'].shape == (TRANSITIONS + 1, 1519)
            and len(reread['memory'].settlements) == 4
            and evidence and all(item == evidence[0] for item in evidence)
            and evidence[0]['exact_log_probs_bitwise_matches'] == evidence[0]['exact_action_and_rng_verified'] == TRANSITIONS,
            'recovered data failed a full independent read or online replay')
    dataset_ref = receipt['references']['episode_manifest']
    prior._complete_receipt(job, cfg_ref, dataset_ref, receipt)
    return job, receipt, dataset_ref, evidence[0], source


def _inventory(con, ledger, job, stop_status):
    rows = con.execute('SELECT run,campaign,status,payload,checksum FROM slots ORDER BY run').fetchall()
    reg = con.execute('SELECT id,payload,checksum FROM campaigns').fetchall()
    require(len(reg) == 1 and reg[0][0] == CAMPAIGN and _payload(reg[0][1:]) == ledger.entry
            and [row[0] for row in rows] == list(ledger.jobs)
            and len(rows) == TOTAL and con.execute('SELECT COUNT(*) FROM commits').fetchone()[0] == 0,
            'original approved evaluation inventory differs')
    require(all(row[2] == ('COMPLETE' if index < OFFSET-1 else
                          stop_status if index == OFFSET-1 else 'PLANNED')
                for index, row in enumerate(rows)),
            'only the original first grade-swap slot may be reconciled')
    state = _state(con, ledger.suite, ledger.entry, ledger.hashes)
    require(state['jobs'][str(SUSPENDED_RUN)]['identity'] == job
            and state['campaigns'][CAMPAIGN]['commits'] == {}, 'canonical slot or optimization state changed')
    return rows, reg, _payload(rows[OFFSET-1][3:])


def reconcile(*, suite, ledger_path, evidence_dir, output_path, apply=False):
    ledger_path, evidence_dir, output_path = (Path(p).resolve() for p in (ledger_path, evidence_dir, output_path))
    require(type(apply) is bool, 'apply must be explicit')
    production = ledger_path == LEDGER
    if production:
        require(output_path == SAVED_DATASET and evidence_dir == RECONCILIATION.parent,
                'production recovery must use the dedicated frozen paths')
    else:
        temp = Path('/tmp')
        require(all(path.is_relative_to(temp) for path in (ledger_path, evidence_dir, output_path)),
                'test reconciliation must be isolated under /tmp')
    original, _ = original_campaign()
    require(suite == original, 'only the original frozen suite can be reconciled')
    ledger = GenLedger(suite, suite_path=SUITE, approval_path=APPROVAL, path=None if production else ledger_path)
    before_ref = reference(ledger_path)
    lock = _controller_lock(OUTPUT) if production and apply else nullcontext()
    with lock, TemporaryDirectory(prefix='bg128-reconcile-') as scratch:
        job, receipt, dataset_ref, evidence, source = _admit(suite,
            output_path if apply else Path(scratch) / f'run_{SUSPENDED_RUN}.h5')
        result = dict(schema=SCHEMA, status='VERIFIED_NOT_APPLIED', campaign_id=CAMPAIGN,
            run_index=SUSPENDED_RUN, original_status='UNKNOWN', new_status='COMPLETE',
            preserved_stop=source['original_stop'], completed_after=OFFSET, planned_after=REMAINING,
            new_reserved_shifts=0, new_physical_shifts=0, publication_scope='production' if production else 'test')
        with connection(ledger_path, readonly=not apply) as con:
            require(reference(ledger_path) == before_ref, 'ledger changed before obtaining the lock')
            require(not any(Path(str(ledger_path) + suffix).exists() for suffix in ('-wal','-journal')),
                    'require a quiescent original ledger with no pending journal')
            row_status = con.execute('SELECT status FROM slots WHERE run=?', (SUSPENDED_RUN,)).fetchone()
            if row_status == ('COMPLETE',):
                require(apply, 'historical slot is already reconciled')
                _, _, current = _inventory(con, ledger, job, 'COMPLETE')
                result = _payload(con.execute(
                    'SELECT payload,checksum FROM generalization_reconciliations WHERE run=?',
                    (SUSPENDED_RUN,)).fetchone())
                require(result['material_receipt'] == reference(evidence_dir/'material_receipt.json')
                        and result['dataset_ref'] == dataset_ref and current['receipt'] == receipt
                        and current['reconciled_from'] == 'UNKNOWN',
                        'existing reconciliation does not match the same verified facts')
            else:
                rows, registration, old = _inventory(con, ledger, job, 'UNKNOWN')
                require(old['error'] == ORIGINAL_ERROR and old['identity_sha256'] == digest(job),
                        'original UNKNOWN reason or frozen slot changed')
                _unfinished_slices(suite)
                require(anchor_files()['original_stop'] == source['original_stop'], 'original STOP changed')
                if not apply:
                    return dict(result, exact_online_evidence=evidence, before_database=before_ref)
                before_database = _seal(evidence_dir/'ledger_before.sqlite', ledger_path.read_bytes())
                require(before_database['sha256'] == before_ref['sha256'], 'SQLite before-image hashes differ')
                row = rows[OFFSET-1]
                before_unknown_row = _seal(evidence_dir/'unknown_slot_before.json',
                    encoded(dict(run_index=row[0], campaign_id=row[1], sql_status=row[2], payload=old,
                                 original_payload_text=row[3], original_checksum=row[4])))
                proof = _seal(evidence_dir/'admission_proof.json', encoded(dict(
                    original_sources=source, exact_online_replay=evidence, transitions=TRANSITIONS, four_windows=4)))
                receipt_ref = _seal(evidence_dir/'material_receipt.json', encoded(receipt))
                result.update(status='APPLIED', before_database=before_database, before_unknown_row=before_unknown_row,
                    exact_online_evidence=proof, material_receipt=receipt_ref, dataset_ref=dataset_ref,
                    unchanged_commits=0, original_error=ORIGINAL_ERROR)
                repaired = dict(old, status='COMPLETE', reconciled_from='UNKNOWN', receipt=receipt,
                    dataset_ref=dataset_ref, completion_sha256=digest(dict(dataset_ref=dataset_ref, receipt=receipt)),
                    reconciliation_evidence=proof, unknown_preimage=before_unknown_row)
                con.execute('CREATE TABLE IF NOT EXISTS generalization_reconciliations '
                            '(run INTEGER PRIMARY KEY, payload TEXT NOT NULL, checksum TEXT NOT NULL)')
                con.execute('INSERT INTO generalization_reconciliations VALUES (?,?,?)',
                    (SUSPENDED_RUN, encoded(result).decode(), digest(result)))
                changed = con.execute('UPDATE slots SET status=?,payload=?,checksum=? '
                    'WHERE run=? AND campaign=? AND status=? AND checksum=?',
                    ('COMPLETE', encoded(repaired).decode(), digest(repaired), SUSPENDED_RUN,
                     CAMPAIGN, 'UNKNOWN', row[4]))
                require(changed.rowcount == 1, 'no exact UNKNOWN slot was updated')
                after, after_registration, _ = _inventory(con, ledger, job, 'COMPLETE')
                require([other for other in after if other[0] != SUSPENDED_RUN] ==
                        [other for other in rows if other[0] != SUSPENDED_RUN]
                        and registration == after_registration, 'another slot or registered scope changed')
        resumed = ledger.resume()
        require(Counter(row['status'] for row in resumed['jobs'].values()) ==
                {'COMPLETE': OFFSET, 'PLANNED': REMAINING}, 'original budget cannot read the reconciled state')
        _seal(evidence_dir/'reconciliation.json', encoded(result))
        return result
