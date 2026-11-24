"""Baseline2 Java-start bridge. --check/--status are read-only."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from resilience_runtime import atomic, record, sha
from repair_runtime import serve


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', type=Path, required=True)
    p.add_argument('--run-index', type=int)
    p.add_argument('--environment-seed', type=int)
    p.add_argument('--recording-id')
    group = p.add_mutually_exclusive_group(required=True)
    for option in ('--from-anylogic-run', '--controller', '--check', '--status'):
        group.add_argument(option, action='store_true')
    args = p.parse_args(argv)
    from repair_campaign import load_suite
    suite_file = args.suite.resolve(strict=True)
    suite = load_suite(suite_file)
    root = Path(suite['output'])
    control = Path(suite['control_directory'])
    first = suite['jobs'][0]
    intent = root/'launch_intent.json'
    status = root/'run_status.json'
    entry = dict(suite_sha256=sha(suite_file), suite=str(suite_file), campaign_id=suite['campaign_id'],
                 first_run=first['run_index'], first_seed=first['experiment_seed'], recording_id=first['recording_id'])
    if args.check:
        print(json.dumps(dict(state='PREFLIGHT', launch_identity=entry, physical_shifts=len(suite['jobs']),
                              output_exists=root.exists()), ensure_ascii=False))
        return
    if args.status:
        print(status.read_text() if status.is_file() else '{"state":"NOT_STARTED"}')
        return
    stop = control/'STOP.properties'
    if stop.exists():
        raise RuntimeError(f'campaign stopped; inspect {stop}')
    if args.from_anylogic_run:
        if (args.run_index,args.environment_seed,args.recording_id) != (
                first['run_index'],first['experiment_seed'],first['recording_id']):
            raise ValueError('AnyLogic start does not match first frozen run')
        # Existing exclusive intent means the physical campaign was already launched.
        # A failed/partial slot needs ledger reconciliation before replacement.
        record(intent, entry)
        with (root/'controller.log').open('xb', buffering=0) as log:
            child = subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'--suite',
                                      str(suite_file),'--controller'], start_new_session=True,
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=log)
        ready = control/'ready'/f"run_{first['run_index']}.properties"
        deadline = time.monotonic() + 840
        while not ready.exists():
            if child.poll() is not None or stop.exists():
                raise RuntimeError(f'controller failed; inspect {root/"controller.log"}')
            if time.monotonic() >= deadline:
                atomic(stop, b'reason=controller-startup-timeout\n')
                raise TimeoutError('controller startup timed out')
            time.sleep(.1)
        return

    if json.loads(intent.read_text()) != entry:
        raise ValueError('controller intent does not match frozen suite')
    with (root/'controller.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            serve(suite)
        except BaseException as exc:
            if not stop.exists():
                atomic(stop,b'reason=controller-failure\n')
            record(status,dict(state='STOP', failure=repr(exc)),replace=True)
            raise


if __name__ == '__main__':
    main()
