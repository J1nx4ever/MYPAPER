"""One-click bridge called by the prepared AnyLogic helper."""
import argparse
import os
import socket
import subprocess
import sys

from common import *
from runtime import verify

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', required=True, type=lambda p: Path(p).resolve())
    parser.add_argument('--run-index', type=int, required=True)
    parser.add_argument('--environment-seed', type=int, required=True)
    parser.add_argument('--recording-id', required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--from-anylogic-run', action='store_true')
    args = parser.parse_args()
    suite, cfg = verify(args.suite)
    first = cfg['jobs'][0]
    require((args.run_index, args.environment_seed, args.recording_id) ==
            (first['run_index'], first['experiment_seed'], first['recording_id']), 'wrong Java starter identity')
    if args.check:
        print(json.dumps(dict(status='OFFLINE_PREFLIGHT_PASS', config=suite['config'], physical_shifts=4)))
        return
    from budget import budget_snapshot, check_new_slots
    before = budget_snapshot()
    check_new_slots(cfg, before)
    require(before['maximum_total_shifts'] == cfg['occupied_before'] and not MATERIALS.exists(),
            'existing experiment evidence or changed complete budget')
    with socket.socket() as probe:
        probe.bind((cfg['execution']['host'], cfg['execution']['port']))
    RUN.mkdir()
    control = RUN/'control'
    control.mkdir()
    with (control/'controller.log').open('xb') as log:
        process = subprocess.Popen([sys.executable,'-B',str(PLAN/'runtime.py'),'--suite',str(args.suite)],
            cwd=ROOT, start_new_session=True, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT)
        os.fsync(log.fileno())
    save(control/'launch_receipt.json', dict(pid=process.pid, suite=reference(args.suite),
         first_run=first['run_index']))
    print('Starting first-window action diagnostic, PID', process.pid)


if __name__ == '__main__':
    main()
