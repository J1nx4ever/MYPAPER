"""Temporary-database tests using the authentic completed first-shift evidence."""
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sqlite3
import tempfile

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('bg128_reconcile_budget', HERE / 'reconcile_budget.py')
reconcile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reconcile)
PROJECT = reconcile.PROJECT


def run():
    suite = HERE.parent / 'prepared_001/suite.json'
    approval = HERE.parent / 'prepared_001/approval.json'
    evidence = PROJECT / 'reports/token_attention_001/business_graph_summary_009/formal_training_001/recovery_001/first_episode_evidence.json'
    output = evidence.parent / 'reconciliation_tool_test.json'
    original = reconcile.budget.GLOBAL_PATH
    original_ref = reconcile.reference(original)
    stop = PROJECT / 'runs/business-graph-summary-formal-seed1-20261118-001/control/STOP.properties'
    stop_ref = reconcile.reference(stop)
    checks = []
    def check(label, value):
        if not value:
            raise AssertionError(label)
        checks.append(label)
    def rejects(label, fn):
        try:
            fn()
        except (ValueError, AssertionError):
            checks.append(label)
            return
        raise AssertionError('did not reject: ' + label)
    def rows(path):
        with sqlite3.connect('file:' + str(path.resolve()) + '?mode=ro', uri=True) as con:
            return dict(slots=con.execute('SELECT run,campaign,status,payload,checksum FROM slots ORDER BY run').fetchall(),
                campaigns=con.execute('SELECT * FROM campaigns').fetchall(),
                commits=con.execute('SELECT * FROM commits').fetchall())
    with tempfile.TemporaryDirectory(prefix='bg128-reconciliation-test-') as tmp:
        root = Path(tmp)
        ledger = root / 'ledger.sqlite'
        ledger.write_bytes(original.read_bytes())
        before = rows(ledger)
        options = dict(suite_path=suite, approval_path=approval, evidence_path=evidence,
            expected_ledger_sha256=reconcile.reference(ledger)['sha256'], ledger_path=ledger, test_only=True)
        dry = reconcile.reconcile(**options)
        check('authentic full-chain dry-run succeeds', dry['status'] == 'VERIFIED_NOT_APPLIED')
        check('dry-run leaves database bytes unchanged', reconcile.reference(ledger)['sha256'] == original_ref['sha256'])
        rejects('test mode cannot target production database', lambda: reconcile.reconcile(
            **dict(options, ledger_path=original)))
        rejects('reviewed ledger SHA mismatch rejected', lambda: reconcile.reconcile(
            **dict(options, expected_ledger_sha256='0' * 64)))
        bad_evidence = json.loads(evidence.read_bytes())
        bad_evidence['checks'][next(iter(bad_evidence['checks']))] = False
        invalid = root / 'invalid_evidence.json'
        invalid.write_bytes(reconcile.encoded(bad_evidence))
        rejects('failed full-chain check rejects migration', lambda: reconcile.reconcile(
            **dict(options, evidence_path=invalid), apply=True,
            backup_directory=root/'invalid_backup', receipt_path=root/'invalid_receipt.json'))
        check('invalid evidence creates no archive or receipt',
            not (root/'invalid_backup').exists() and not (root/'invalid_receipt.json').exists())
        check('invalid evidence leaves every database row unchanged', rows(ledger) == before)

        backup, receipt = root / 'archive', root / 'applied_receipt.json'
        applied = reconcile.reconcile(**options, apply=True, backup_directory=backup, receipt_path=receipt)
        after = rows(ledger)
        check('authentic temporary migration applies', applied['status'] == 'APPLIED'
            and applied['schema'] == 'bg128-budget-reconciliation-applied-v1')
        check('only first row migrates UNKNOWN to COMPLETE', after['slots'][0][2] == 'COMPLETE'
            and before['slots'][0][2] == 'UNKNOWN' and after['slots'][1:] == before['slots'][1:])
        check('campaign approval and cap bytes unchanged', after['campaigns'] == before['campaigns'])
        check('zero commits unchanged', after['commits'] == before['commits'] == [])
        record = json.loads(after['slots'][0][3])
        original_record = json.loads(before['slots'][0][3])
        check('original failure and identity checksum retained', record['error'] == original_record['error']
            and record['identity_sha256'] == original_record['identity_sha256']
            and record['reconciled_from'] == 'UNKNOWN')
        check('canonical completion checksum valid', record['completion_sha256'] == reconcile.budget.digest(dict(
            dataset_ref=record['dataset_ref'], receipt=record['receipt'])))
        check('whole database archive exactly matches original bytes',
            reconcile.reference(backup/'ledger_before.sqlite')['sha256'] == original_ref['sha256'])
        preimage = json.loads((backup/'unknown_slot_before.json').read_bytes())
        check('unknown preimage preserves exact payload text and checksum',
            preimage['payload_text'] == before['slots'][0][3] and preimage['checksum'] == before['slots'][0][4])
        check('archives and receipt are read-only', all(not (p.stat().st_mode & 0o222)
            for p in [*backup.iterdir(), receipt]))
        with sqlite3.connect('file:' + str(ledger.resolve()) + '?mode=ro', uri=True) as con:
            event = con.execute('SELECT run,payload,checksum FROM bg128_reconciliations').fetchall()
        check('one immutable audit event retained in transaction', len(event) == 1
            and event[0][0] == 9900001 and reconcile.budget.digest(json.loads(event[0][1])) == event[0][2])
        check('applied receipt matches archived plan and original STOP',
            applied['plan'] == reconcile.reference(backup/'reconciliation_audit.json')
            and applied['stop'] == stop_ref)
        check('old budget reader accepts reconciled state', applied['old_reader_validation'] is True)
        rejects('duplicate migration rejects already COMPLETE slot', lambda: reconcile.reconcile(
            **dict(options, expected_ledger_sha256=reconcile.reference(ledger)['sha256']), apply=True,
            backup_directory=root/'double_archive', receipt_path=root/'double_receipt.json'))
        check('duplicate migration changes no rows', rows(ledger) == after)

        # Force a failure only at the final old-reader validation, after SQL
        # UPDATE/INSERT, to check rollback rather than just preflight rejection.
        rollback_db = root / 'rollback.sqlite'
        rollback_db.write_bytes(original.read_bytes())
        real_state = reconcile.budget._state
        calls = [0]
        def fail_final_state(*args, **kwargs):
            calls[0] += 1
            if calls[0] == 2:
                raise ValueError('injected final old-reader rejection')
            return real_state(*args, **kwargs)
        reconcile.budget._state = fail_final_state
        try:
            rejects('post-update validation failure rolls transaction back', lambda: reconcile.reconcile(
                **dict(options, ledger_path=rollback_db), apply=True,
                backup_directory=root/'rollback_archive', receipt_path=root/'rollback_receipt.json'))
        finally:
            reconcile.budget._state = real_state
        check('rollback preserves all original slot/campaign/commit rows', rows(rollback_db) == before)
        check('rollback emits no APPLIED receipt', not (root/'rollback_receipt.json').exists())
        with sqlite3.connect('file:' + str(rollback_db.resolve()) + '?mode=ro', uri=True) as con:
            check('audit table creation is rolled back atomically', not con.execute(
                "SELECT name FROM sqlite_master WHERE name='bg128_reconciliations'").fetchall())
    check('production database bytes unchanged', reconcile.reference(original) == original_ref)
    check('original STOP bytes unchanged', reconcile.reference(stop) == stop_ref)
    result = dict(schema='bg128-reconciliation-tool-temporary-test-v1', status='PASS',
        check_count=len(checks), checks=checks, production_database_unchanged=True,
        production_migrations=0, temporary_migrations=1, optimizer_updates=0,
        rpc_started=False, anylogic_started=False)
    with output.open('xb') as stream:
        stream.write(reconcile.encoded(result))
    print(json.dumps({k:v for k,v in result.items() if k!='checks'}))


if __name__ == '__main__':
    run()
