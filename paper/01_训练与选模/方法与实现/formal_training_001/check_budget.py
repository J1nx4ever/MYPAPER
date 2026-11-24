"""Exercise BG128 budget registration on a disposable database only.

This checks the real frozen 512-job suite and approval gate.  Completion files
are small synthetic accounting fixtures; no observation, model, RPC, optimizer
or AnyLogic execution is involved.  Production ledgers are hashed before/after.
"""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import sys
import tempfile

PROJECT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT))

from business_graph_summary_campaign_v1 import budget
from business_graph_summary_campaign_v1.launch import authorization
from business_graph_summary_campaign_v1.plan_io import load_suite, reference, write


def run(suite_path, report_dir):
    suite_path = Path(suite_path).resolve()
    suite, _ = load_suite(suite_path)
    suite_ref = reference(suite_path)
    approval = authorization(suite, suite_ref, 'Temporary budget-only integrity test; no execution authorization.')
    ledger_paths = [Path(ref['path']) for ref in budget.preceding_reference().values()
                    if isinstance(ref, dict) and 'path' in ref]
    ledger_paths.append(budget.GLOBAL_PATH)
    before = {str(p): reference(p) if p.exists() else None for p in ledger_paths}
    checks = []

    def check(label, condition):
        if not condition:
            raise AssertionError(label)
        checks.append(label)

    def rejects(label, fn):
        try:
            fn()
        except (ValueError, AssertionError):
            checks.append(label)
            return
        raise AssertionError('did not reject: ' + label)

    with tempfile.TemporaryDirectory(prefix='bg128-budget-test-') as folder:
        root = Path(folder)
        def register(path, spec=suite, decision=approval):
            return budget.BudgetLedger(path, spec, campaign_sha256=suite_ref['sha256'],
                approval=decision, approval_sha256=budget.digest(decision), test_only=True)

        bad_approval = dict(approval, max_new_shifts=suite['new_shift_cap'] + 1)
        rejects('approval cap mismatch rejected before write',
            lambda: register(root / 'rejected' / 'ledger.sqlite', decision=bad_approval))
        check('invalid approval created no database directory', not (root / 'rejected').exists())
        path = root / 'ledger.sqlite'
        ledger = register(path)
        original = ledger.snapshot()
        check('all 512 physical slots registered', len(original['jobs']) == len(suite['jobs']) == 512)
        check('registration leaves every slot planned',
              Counter(r['status'] for r in original['jobs'].values()) == {'PLANNED': 512})
        check('empty resume passes', ledger.resume() == original)
        check('same registration is idempotent', register(path).snapshot() == original)
        merged = budget.budget_snapshot(path)
        check('merged capacity includes preceding histories plus 512',
              merged['maximum_total_shifts'] == suite['maximum_total_if_approved'])
        check('historical unknown preserved',
              sum(r['status'] == 'UNKNOWN' for r in merged['jobs'].values()) == 1)
        check('historical planned preserved',
              sum(r['status'] == 'PLANNED' for r in merged['jobs'].values()) == 521)

        historical = budget.preceding_snapshot()
        historical_rows = [r['identity'] for r in historical['jobs'].values()]
        for field in ('run_index', 'recording_id', 'source_recording_id'):
            altered = copy.deepcopy(suite)
            altered['jobs'][0][field] = next(r[field] for r in historical_rows if r.get(field) is not None)
            rejects('historical collision rejected: ' + field,
                    lambda altered=altered: budget.check_collisions(altered, historical))
        historical_seed = next(r['identity'][field] for r in historical['jobs'].values()
            for field in ('experiment_seed', 'actor_sampling_seed', 'actor_seed')
            if r['identity'].get(field) is not None)
        for field in ('experiment_seed', 'actor_sampling_seed', 'actor_seed'):
            altered = copy.deepcopy(suite)
            altered['jobs'][0][field] = historical_seed
            rejects('historical cross-type seed collision rejected: ' + field,
                    lambda altered=altered: budget.check_collisions(altered, historical))

        first_batch = [j for j in suite['jobs'] if j['target_training_update'] == 1]
        check('first update is exactly 32 canonical train jobs', len(first_batch) == 32)
        check('canonical job read is planned', ledger.job(first_batch[0])['status'] == 'PLANNED')
        first_dataset = root / ('accounting-fixture-' + str(first_batch[0]['run_index']) + '.json')
        write(first_dataset, dict(synthetic_accounting_fixture=True, run_index=first_batch[0]['run_index']))
        rejects('complete before claim rejected', lambda: ledger.complete(first_batch[0],
            reference(first_dataset), dict(run_index=first_batch[0]['run_index'],
                experiment_seed=first_batch[0]['experiment_seed'], normal_completion=True)))
        for job in first_batch:
            ledger.claim(job)
            if job is first_batch[0]:
                rejects('duplicate claim rejected', lambda: ledger.claim(job))
                rejects('claimed slot prevents resume', ledger.resume)
            dataset = root / ('accounting-fixture-' + str(job['run_index']) + '.json')
            if not dataset.exists():
                write(dataset, dict(synthetic_accounting_fixture=True, run_index=job['run_index']))
            receipt = dict(run_index=job['run_index'], experiment_seed=job['experiment_seed'],
                normal_completion=True)
            ledger.complete(job, reference(dataset), receipt)
        checkpoint = root / 'synthetic-checkpoint-accounting-fixture.json'
        write(checkpoint, dict(synthetic_accounting_fixture=True, model_forward=False, optimizer_updates=0))
        checkpoint_ref = reference(checkpoint)
        rejects('reordered canonical batch rejected', lambda: ledger.commit(1,
            list(reversed(first_batch)), checkpoint_ref, 'synthetic-budget-batch-1'))
        rejects('noncontiguous update rejected', lambda: ledger.commit(2,
            [j for j in suite['jobs'] if j['target_training_update'] == 2],
            checkpoint_ref, 'synthetic-budget-batch-2'))
        ledger.commit(1, first_batch, checkpoint_ref, 'synthetic-budget-batch-1')
        record = ledger.commit_record(1)
        check('committed checkpoint binding retrievable', record['checkpoint_ref'] == checkpoint_ref)
        check('committed update includes 32 completed slots', all(
            ledger.job(j).get('committed_update') == 1 for j in first_batch))
        rejects('double commit rejected', lambda: ledger.commit(1,
            first_batch, checkpoint_ref, 'synthetic-budget-batch-1'))
        next_job = next(j for j in suite['jobs'] if j['target_training_update'] == 2)
        ledger.claim(next_job)
        ledger.unknown(next_job, 'synthetic interruption accounting fixture')
        rejects('unknown slot prevents resume', ledger.resume)
        rejects('unknown slot cannot be claimed again', lambda: ledger.claim(next_job))
        state = ledger.snapshot()
        check('test-only statuses reflect preserved interruption',
            Counter(r['status'] for r in state['jobs'].values()) ==
            {'COMPLETE': 32, 'UNKNOWN': 1, 'PLANNED': 479})

    after = {str(p): reference(p) if p.exists() else None for p in ledger_paths}
    check('all production ledgers unchanged or absent', before == after)
    result = dict(schema='bg128-budget-test-only-integrity-v1', status='PASS',
        suite=suite_ref, check_count=len(checks), checks=checks,
        production_ledgers_unchanged=True, real_slots_registered=0,
        synthetic_slots_registered=512, model_forward=False, optimizer_updates=0,
        rpc_started=False, anylogic_started=False, temporary_database_removed=True)
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    write(report_dir / 'budget_integrity.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', required=True)
    parser.add_argument('--report-dir', required=True)
    args = parser.parse_args()
    result = run(args.suite, args.report_dir)
    print(json.dumps({k: v for k, v in result.items() if k != 'checks'}, ensure_ascii=False))
