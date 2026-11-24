"""Explicit, evidence-bound reconciliation of the one completed BG128 shift.

Default operation is read-only verification.  --apply performs exactly one
UNKNOWN -> COMPLETE migration after archiving the entire suspended database and
the original slot payload.  It never deletes STOP or claims, registers capacity,
collects a shift, fits a model or changes another slot.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys

PROJECT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT))

from business_graph_summary_campaign_v1 import budget
from business_graph_summary_campaign_v1.gate import check_approval, check_sources
from business_graph_summary_campaign_v1.plan_io import encoded, load_suite, reference

CAMPAIGN = 'business-graph-summary-formal-seed1-20261118-001'
RUN = 9900001
ENVIRONMENT_SEED = 255314794623997
TRANSITIONS = 184
EVIDENCE_SCHEMA = 'bg128-first-episode-reconciliation-evidence-v1'
AUDIT_SCHEMA = 'bg128-evidence-bound-budget-reconciliation-v1'
AUDIT_TABLE = 'bg128_reconciliations'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(ref):
    require(isinstance(ref, dict) and set(ref) == {'path', 'sha256'}, 'exact file reference required')
    require(reference(ref['path']) == ref, 'referenced bytes changed: ' + ref['path'])
    return json.loads(Path(ref['path']).read_bytes())


def seal_file(path, data):
    """Publish one archive once, fsync, then remove write permission."""
    path = Path(path)
    with path.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o444)
    return reference(path)


def validate_evidence(suite_path, approval_path, evidence_path):
    suite_path, approval_path, evidence_path = map(lambda p: Path(p).resolve(),
        (suite_path, approval_path, evidence_path))
    suite, phases = load_suite(suite_path)
    suite_ref, approval_ref, evidence_ref = map(reference, (suite_path, approval_path, evidence_path))
    require(suite['campaign_id'] == CAMPAIGN and suite['new_shift_cap'] == suite['training_shifts'] == 512
        and suite['evaluation_shifts'] == 0 and len(suite['jobs']) == 512,
        'only the original approved 512-shift campaign can be reconciled')
    check_sources(suite['source_pins'])
    approval = load(approval_ref)
    check_approval(approval, suite, suite_ref['sha256'])
    job = suite['jobs'][0]
    require(job['run_index'] == RUN and job['experiment_seed'] == ENVIRONMENT_SEED
        and job['target_training_update'] == 1 and job['ordinal'] == 0 and job['role'] == 'train',
        'only the canonical first original slot can be reconciled')
    evidence = load(evidence_ref)
    require(evidence.get('schema') == EVIDENCE_SCHEMA and evidence.get('status') == 'PASS'
        and evidence.get('original_campaign') == suite_ref
        and evidence.get('original_approval') == approval_ref,
        'complete admission evidence must bind original suite and approval bytes')
    checks = evidence.get('checks')
    require(isinstance(checks, dict) and bool(checks) and all(v is True for v in checks.values()),
        'all explicit full-chain evidence checks must pass')
    pins = evidence.get('source_pins')
    require(isinstance(pins, dict) and bool(pins), 'explicit new storage/admission source pins required')
    for path, checksum in pins.items():
        require(reference(path)['sha256'] == checksum, 'recovery source changed: ' + path)
    # These exact modules provide the new graph-contract writer and strong
    # policy/material reader. A Boolean PASS without their byte pins is invalid.
    mandatory = [PROJECT / 'business_graph_summary_storage_v1/episode.py',
        PROJECT / 'business_graph_summary_recovery_v1/collection.py',
        PROJECT / 'business_graph_summary_recovery_v1/admission.py']
    require(all(str(p.resolve()) in pins for p in mandatory), 'required recovery implementation is not pinned')
    raw_receipt = load(evidence['raw_receipt'])
    material_receipt = load(evidence['material_receipt'])
    admission = load(evidence['admission_report'])
    require(raw_receipt.get('normal_completion') is True
        and raw_receipt.get('run_index') == RUN and raw_receipt.get('experiment_seed') == ENVIRONMENT_SEED
        and raw_receipt.get('transitions') == TRANSITIONS and raw_receipt.get('four_windows') == 4
        and raw_receipt.get('phase_id') == job['phase_id'] and raw_receipt.get('role') == 'train',
        'raw terminal receipt does not prove this exact complete 184-transition shift')
    require(material_receipt.get('normal_completion') is True
        and material_receipt.get('run_index') == RUN and material_receipt.get('experiment_seed') == ENVIRONMENT_SEED
        and material_receipt.get('transitions') == TRANSITIONS and material_receipt.get('four_windows') == 4
        and material_receipt.get('material_evidence') is True and material_receipt.get('role') == 'train'
        and material_receipt.get('raw_receipt') == raw_receipt,
        'canonical admitted receipt must retain the original complete raw receipt')
    require(admission.get('status') == 'PASS_DATA_PREPARATION_ONLY'
        and admission.get('transitions') == TRANSITIONS
        and admission.get('actor_or_critic_fitted') is False
        and admission.get('collection_started') is False,
        'fresh complete BG admission report required; no fitting or collection')
    refs = material_receipt.get('references')
    require(isinstance(refs, dict) and set(refs) == {'config','actor','journal','raw_manifest',
        'episode','episode_manifest','episode_seal'}, 'all seven admitted source bindings required')
    for ref in refs.values():
        require(reference(ref['path']) == ref, 'admitted reference bytes changed')
    phase_ref = suite['phase_configs'][0]
    actor_path = Path(phases[0]['python_output']) / 'actor.pt'
    raw_path = Path(phases[0]['python_output']) / 'episodes' / ('run_' + str(RUN))
    require(refs['config'] == material_receipt.get('phase_config') == raw_receipt.get('phase_config') == phase_ref
        and refs['actor'] == reference(actor_path)
        and refs['journal'] == reference(job['material_journal'])
        and refs['raw_manifest'] == reference(raw_path / 'manifest.json')
        and evidence['raw_receipt'] == reference(raw_path / 'receipt.json'),
        'original Actor/config/raw/journal paths and bytes cannot be replaced')
    raw_manifest = load(refs['raw_manifest'])
    require(raw_manifest.get('status') == 'complete' and raw_manifest.get('source_kind') == 'real_anylogic_rpc'
        and raw_manifest.get('episodes') == 1 and raw_manifest.get('transitions') == TRANSITIONS
        and raw_manifest['provenance']['run_index'] == RUN
        and raw_manifest['provenance']['seed'] == ENVIRONMENT_SEED
        and raw_manifest['provenance']['recording_id'] == job['recording_id']
        and raw_manifest['provenance']['configuration'] == phase_ref,
        'sealed real raw source does not bind this canonical original job')
    require(raw_receipt['dataset_files'] == raw_manifest['files_sha256'], 'raw file hash inventories disagree')
    raw_files = []
    for name, checksum in raw_manifest['files_sha256'].items():
        path = raw_path / name
        require(path.resolve().parent == raw_path.resolve() and reference(path)['sha256'] == checksum,
            'raw payload changed or escaped the canonical source directory')
        raw_files.append(reference(path))
    require(admission.get('output') == refs['episode'], 'admission report output differs from sealed episode')
    identity = material_receipt.get('identity')
    require(identity == admission.get('identity') and identity.get('run_index') == RUN
        and identity.get('recording_id') == job['recording_id']
        and identity.get('environment_seed') == ENVIRONMENT_SEED
        and identity.get('actor_sampling_seed') == job['actor_sampling_seed']
        and identity.get('candidate_id') == job['candidate_id']
        and identity.get('source_kind') == 'fixed_actor' and identity.get('split') == 'train'
        and identity.get('policy_sha256') == refs['actor']['sha256']
        and identity.get('config_sha256') == phase_ref['sha256']
        and identity.get('policy_version') == raw_receipt['policy_version'] == raw_manifest['policy_version'],
        'derived policy/run/environment/action-seed/graph identity changed')
    from business_graph_summary_recovery_v1.collection import load_admitted_episode
    episode = load_admitted_episode(material_receipt)
    require(episode['identity'] == identity and len(episode['actions']) == TRANSITIONS
        and episode['states'].shape == (TRANSITIONS + 1, 1519)
        and len(episode['memory'].settlements) == 4,
        'independent strong read of the complete episode failed')
    stop = Path(suite['control_directory']) / 'STOP.properties'
    require(stop.is_file() and "unknown task-query candidate" in stop.read_text(),
        'the original failure STOP must be preserved')
    bound_refs = [suite_ref, approval_ref, evidence_ref, evidence['raw_receipt'],
        evidence['material_receipt'], evidence['admission_report'], *refs.values(), *raw_files, reference(stop)]
    return dict(suite=suite, suite_ref=suite_ref, approval=approval, approval_ref=approval_ref,
        evidence=evidence, evidence_ref=evidence_ref, job=job, dataset_ref=refs['episode_manifest'],
        receipt=material_receipt, bound_refs=bound_refs, stop=reference(stop))


def inspect_database(con, verified):
    suite = verified['suite']
    campaigns = con.execute('SELECT id,payload,checksum FROM campaigns').fetchall()
    require(len(campaigns) == 1 and campaigns[0][0] == CAMPAIGN, 'exactly the original single campaign required')
    entry = budget._payload(campaigns[0][1:])
    require(entry['cap'] == 512 and entry['suite_reference'] == verified['suite_ref']
        and entry['approval'] == verified['approval']
        and entry['approval_sha256'] == verified['approval_ref']['sha256'],
        'registered capacity/suite/approval changed')
    rows = con.execute('SELECT run,campaign,status,payload,checksum FROM slots ORDER BY run').fetchall()
    require(len(rows) == 512 and [r[0] for r in rows] == [j['run_index'] for j in suite['jobs']],
        'original 512-slot inventory changed')
    require(Counter(r[2] for r in rows) == {'UNKNOWN': 1, 'PLANNED': 511}
        and rows[0][0] == RUN and rows[0][2] == 'UNKNOWN', 'only the suspended first UNKNOWN may be reconciled')
    require(con.execute('SELECT COUNT(*) FROM commits').fetchone()[0] == 0, 'zero-update suspended campaign required')
    state = budget._state(con, suite, entry)
    require(state['jobs'][str(RUN)]['identity'] == verified['job'], 'original canonical slot identity differs')
    preimage = budget._payload(rows[0][3:])
    require(preimage['status'] == 'UNKNOWN' and preimage.get('error') == "ValueError('unknown task-query candidate')"
        and preimage['identity_sha256'] == budget.digest(verified['job']), 'exact original UNKNOWN failure required')
    return entry, rows, preimage


def reconcile(*, suite_path, approval_path, evidence_path, expected_ledger_sha256,
              ledger_path=None, apply=False, backup_directory=None, receipt_path=None, test_only=False):
    ledger = Path(ledger_path or budget.GLOBAL_PATH).resolve()
    require(ledger.is_file(), 'existing suspended ledger required; this tool never registers a campaign')
    require(ledger != budget.GLOBAL_PATH.resolve() if test_only else ledger == budget.GLOBAL_PATH.resolve(),
        'production/test database scope differs')
    require(type(test_only) is bool and type(apply) is bool, 'explicit Boolean modes required')
    before_ref = reference(ledger)
    require(before_ref['sha256'] == expected_ledger_sha256, 'ledger bytes differ from explicitly reviewed suspended state')
    verified = validate_evidence(suite_path, approval_path, evidence_path)
    con = sqlite3.connect('file:' + str(ledger) + '?mode=' + ('rw' if apply else 'ro'), uri=True, timeout=30)
    try:
        if apply:
            con.execute('PRAGMA synchronous=FULL')
            con.execute('BEGIN IMMEDIATE')
        require(reference(ledger) == before_ref, 'ledger changed before reconciliation lock')
        require(not any(Path(str(ledger) + suffix).exists() for suffix in ('-wal','-journal')),
            'a journal/WAL exists; require a clean stopped single-file database')
        entry, rows, preimage = inspect_database(con, verified)
        for ref in verified['bound_refs']:
            require(reference(ref['path']) == ref, 'source changed while verification was running')
        result = dict(schema=AUDIT_SCHEMA, status='VERIFIED_NOT_APPLIED', campaign_id=CAMPAIGN,
            run_index=RUN, before_ledger=before_ref, evidence=verified['evidence_ref'],
            previous_status='UNKNOWN', resulting_status='COMPLETE', original_error=preimage['error'],
            cap_before=512, cap_after=512, other_planned_slots=511, commit_count=0,
            new_physical_shifts=0, new_reserved_shifts=0, old_stop=verified['stop'],
            test_only=test_only, strong_read_passed=True, transitions=TRANSITIONS, four_windows=4)
        if not apply:
            return result
        require(backup_directory is not None and receipt_path is not None,
            'explicit fresh immutable backup directory and receipt path required for --apply')
        backup_dir, receipt_file = Path(backup_directory).resolve(), Path(receipt_path).resolve()
        require(not backup_dir.exists() and not receipt_file.exists(), 'backup and applied receipt must never be overwritten')
        require(backup_dir != ledger.parent and ledger not in backup_dir.parents,
            'archive must use a distinct explicit recovery evidence directory')
        backup_dir.mkdir(parents=True)
        archive_db = seal_file(backup_dir / 'ledger_before.sqlite', ledger.read_bytes())
        require(archive_db['sha256'] == before_ref['sha256'], 'whole-database immutable archive differs')
        slot_archive = seal_file(backup_dir / 'unknown_slot_before.json', encoded(dict(
            run_index=RUN, campaign_id=CAMPAIGN, sql_status=rows[0][2],
            payload_text=rows[0][3], payload=preimage, checksum=rows[0][4])))
        audit = dict(result, status='APPLIED', applied_at_utc=datetime.now(timezone.utc).isoformat(),
            original_suite=verified['suite_ref'], original_approval=verified['approval_ref'],
            backup_database=archive_db, backup_unknown_slot=slot_archive,
            original_payload_text=rows[0][3], original_checksum=rows[0][4],
            dataset_ref=verified['dataset_ref'], completion_receipt=verified['receipt'])
        audit_ref = seal_file(backup_dir / 'reconciliation_audit.json', encoded(audit))
        updated = dict(preimage, status='COMPLETE', dataset_ref=verified['dataset_ref'],
            receipt=verified['receipt'], completion_sha256=budget.digest(dict(
                dataset_ref=verified['dataset_ref'], receipt=verified['receipt'])),
            reconciled_from='UNKNOWN', reconciliation_evidence=verified['evidence_ref'],
            reconciliation_audit=audit_ref, unknown_preimage=slot_archive)
        con.execute('CREATE TABLE IF NOT EXISTS ' + AUDIT_TABLE +
            ' (run INTEGER PRIMARY KEY, payload TEXT NOT NULL, checksum TEXT NOT NULL)')
        require(con.execute('SELECT COUNT(*) FROM ' + AUDIT_TABLE).fetchone()[0] == 0,
            'an existing reconciliation event must not be overwritten')
        con.execute('INSERT INTO ' + AUDIT_TABLE + ' VALUES (?,?,?)',
            (RUN, encoded(audit).decode(), budget.digest(audit)))
        con.execute('UPDATE slots SET status=?,payload=?,checksum=? WHERE run=? AND campaign=? AND status=?',
            ('COMPLETE', encoded(updated).decode(), budget.digest(updated), RUN, CAMPAIGN, 'UNKNOWN'))
        require(con.execute('SELECT changes()').fetchone()[0] == 1, 'exactly one UNKNOWN row must migrate')
        after_rows = con.execute('SELECT run,campaign,status,payload,checksum FROM slots ORDER BY run').fetchall()
        require(after_rows[1:] == rows[1:], 'one of the other 511 planned slots changed')
        require(con.execute('SELECT id,payload,checksum FROM campaigns').fetchall() ==
            [(CAMPAIGN, encoded(entry).decode(), budget.digest(entry))],
            'registered campaign/approval bytes changed')
        require(con.execute('SELECT COUNT(*) FROM commits').fetchone()[0] == 0, 'reconciliation cannot create an update')
        state = budget._state(con, verified['suite'], entry)
        require(Counter(r['status'] for r in state['jobs'].values()) == {'COMPLETE': 1, 'PLANNED': 511},
            'old budget reader rejects reconciled state')
        require(reference(Path(verified['stop']['path'])) == verified['stop'], 'old STOP changed')
        con.commit()
        result.update(schema='bg128-budget-reconciliation-applied-v1', status='APPLIED',
            old_suite=verified['suite_ref'], plan=audit_ref, ledger_before=archive_db,
            ledger_after_sha256=reference(ledger)['sha256'], old_row=slot_archive,
            admission_proof=verified['evidence_ref'], derived_manifest=verified['dataset_ref'],
            new_status='COMPLETE', original_status='UNKNOWN', new_shift_cap=0,
            unchanged_planned_slots=511, unchanged_commits=0, stop=verified['stop'],
            after_ledger=reference(ledger), backup_database=archive_db,
            backup_unknown_slot=slot_archive, reconciliation_audit=audit_ref,
            old_reader_validation=True, other_511_slots_unchanged=True, campaign_entry_unchanged=True)
        receipt_file.parent.mkdir(parents=True, exist_ok=True)
        seal_file(receipt_file, encoded(result))
        return result
    except BaseException:
        if apply:
            con.rollback()
        raise
    finally:
        con.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', required=True)
    parser.add_argument('--approval', required=True)
    parser.add_argument('--evidence', required=True)
    parser.add_argument('--expected-ledger-sha256', required=True)
    parser.add_argument('--ledger')
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--test-only', action='store_true')
    parser.add_argument('--backup-directory')
    parser.add_argument('--receipt')
    args = parser.parse_args()
    result = reconcile(suite_path=args.suite, approval_path=args.approval,
        evidence_path=args.evidence, expected_ledger_sha256=args.expected_ledger_sha256,
        ledger_path=args.ledger, apply=args.apply, backup_directory=args.backup_directory,
        receipt_path=args.receipt, test_only=args.test_only)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
