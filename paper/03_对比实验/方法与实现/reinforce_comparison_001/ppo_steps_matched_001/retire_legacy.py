"""Delete legacy S02 payloads after validating the retained budget chain."""
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
from tempfile import mkdtemp
from zoneinfo import ZoneInfo
import shutil

from run import plan
from change_plans.token_attention_001.business_graph_summary_009.reinforce_comparison_001 import prepare as legacy
from change_plans.token_attention_001.business_graph_summary_009.reinforce_comparison_001.paired_inference_001 import prepare as old_eval
from business_graph_summary_campaign_v1.plan_io import reference, write


def counts(path):
    with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True) as db:
        return dict(db.execute('SELECT status, count(*) FROM slots GROUP BY status'))


def deletion_paths():
    for phase in legacy.ARM.glob('train_u[0-9][0-9][0-9][0-9]'):
        for name in ('episodes', 'material', 'actor.pt', 'actor.pt.manifest.json', 'actor.pt.seal.json'):
            yield phase / name
    yield legacy.MATERIAL
    yield old_eval.OUTPUT / 'evaluation'
    yield old_eval.MATERIAL
    for name in ('offline_replay_s02.json', 'training_comparison.csv', 'launch_confirmed_s02.json',
                 'prepared_verification_s02.json'):
        yield legacy.REPORT / name
    for name in ('launch_confirmed.json', 'prepared_verification.json'):
        yield old_eval.REPORT / name


def main():
    old_progress = json.loads((old_eval.OUTPUT / 'training_progress.json').read_bytes())
    require = plan.original.require
    require(old_progress['campaign_id'] == old_eval.CAMPAIGN and old_progress['status'] == 'STOP'
            and (old_eval.CONTROL / 'STOP.properties').is_file(), 'previous controller must first reach STOP')
    history = {legacy.CAMPAIGN: counts(legacy.LEDGER), old_eval.CAMPAIGN: counts(old_eval.LEDGER)}
    require(history[legacy.CAMPAIGN] == {'COMPLETE': 1024}
            and sum(history[old_eval.CAMPAIGN].values()) == 700
            and 'CLAIMED' not in history[old_eval.CAMPAIGN], 'previous physical ledger differs or is still active')
    report = plan.REPORT / 'legacy_cleanup.json'
    require(not report.exists(), 'cleanup receipt is already published')
    targets = [p for p in deletion_paths() if p.exists()]
    require(bool(targets) and all(p.resolve() == p and p.is_relative_to(plan.ROOT) for p in targets),
            'cleanup targets must be nonempty and contained in the workspace')
    for p in targets:
        require(not p.is_dir() or all(not f.is_symlink() for f in p.rglob('*')), 'symlink in old payload')
    protected = [legacy.LEDGER, old_eval.LEDGER, legacy.OUTPUT / 'suite_complete.json',
                 legacy.REPORT / 'training_complete.json', legacy.ARM / 'final_actor.pt',
                 legacy.ARM / 'updates/0032/checkpoint.pt',
                 legacy.OUTPUT / 'launch/run_authority.json', old_eval.OUTPUT / 'launch/run_authority.json',
                 old_eval.CONTROL / 'STOP.properties']
    seals = [reference(p) for p in protected]
    stage = Path(mkdtemp(dir=plan.ROOT / 'runs', prefix='.legacy-reinforce-cleanup-'))
    staged = {}
    try:
        inventory = Counter()
        removed = []
        for source in targets:
            payloads = [f for f in source.rglob('*') if f.is_file()] if source.is_dir() else [source]
            size = sum(f.stat().st_size for f in payloads)
            inventory.update(files=len(payloads), bytes=size)
            removed.append(dict(path=str(source.relative_to(plan.ROOT)), files=len(payloads), bytes=size))
            target = stage / source.relative_to(plan.ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            source.rename(target)
            staged[source] = target
        cli = Path(__file__).with_name('run.py')
        check = subprocess.run([sys.executable, '-B', str(cli), 'verify'], cwd=plan.ROOT,
                               capture_output=True, text=True, check=True)
        verification = json.loads(check.stdout)
        require(verification['planned_actor_optimizer_steps'] == 23612
                and [reference(p) for p in protected] == seals, 'retained artifacts or new schedule changed')
    except BaseException:
        for source, target in reversed(staged.items()):
            target.rename(source)
        shutil.rmtree(stage)
        raise
    shutil.rmtree(stage)
    require(all(not p.exists() for p in targets), 'legacy payload still exists')
    data = dict(status='DELETED', at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
                inventory=dict(inventory), deleted=removed, retained=seals,
                old_ledger_counts=history, new_preflight_after_detaching_old_payloads=verification)
    write(report, data)
    print(json.dumps(dict(report=str(report), inventory=dict(inventory), old_ledger_counts=history)))


if __name__ == '__main__':
    main()
