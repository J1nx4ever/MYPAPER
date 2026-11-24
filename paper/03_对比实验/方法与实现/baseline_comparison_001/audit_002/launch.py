"""Verify the frozen V2 package or accept the first matching AnyLogic startup."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess

from audit_plan import (CAMPAIGN, CONTROL, CORE, COUNT, EXECUTION, OUTPUT, PREPARED, PYTHON, REPORT_FILENAME, ROOT,
                        SUITE, encoded, load_reference, load_suite, reference, require)
from audit_budget import AUTHORITY, LEDGER, LOCKS, Ledger, authority_for, check_fresh, preceding_snapshot
from business_graph_summary_campaign_v1.launch import _exclusive_json, _launch_lock
from online_v1_1.operations import _publish


ENTRY = Path(__file__).resolve()
LAUNCH_DIR = OUTPUT/'launch'
LANE = LOCKS/f"port-{EXECUTION['port']}.lock"
INTENT = LAUNCH_DIR/'launch_intent.json'
STATUS = OUTPUT/'run_status.json'
STOP = CONTROL/'STOP.properties'


def _intent(suite, fd):
    return dict(suite=reference(SUITE), campaign_id=suite['campaign_id'], authority=reference(AUTHORITY),
                command=[PYTHON, '-B', str(ENTRY), '--suite', str(SUITE), '--controller', '--lane-fd', str(fd)])


def _status(status, **details):
    value = dict(status=status, campaign_id=CAMPAIGN, scheduled_shifts=COUNT,
                 updated_at_utc=datetime.now(timezone.utc).isoformat(), **details)
    _publish(STATUS, encoded(value), replace=STATUS.exists())


def _preflight(suite):
    card = load_reference(reference(PREPARED/'launch_identity.json'))
    require(card['suite'] == reference(SUITE) and card['java_helper'] == suite['java_helper'],
            'launch card differs from the sealed revision suite')
    require(not OUTPUT.exists(), 'existing campaign requires reconciliation before launch')
    return dict(status='PREFLIGHT_PASS_NO_LISTENER', suite=reference(SUITE), campaign_id=CAMPAIGN,
                java_helper=suite['java_helper'], physical_shifts=COUNT,
                budget=check_fresh(suite, preceding_snapshot()))


def _start(suite):
    preflight = _preflight(suite)
    LANE.parent.mkdir(parents=True, exist_ok=True)
    with LANE.open('a+b') as lease:
        try:
            fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('the AnyLogic RPC port is leased by another campaign') from error
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind((EXECUTION['host'], EXECUTION['port']))
        OUTPUT.mkdir()
        with _launch_lock(LAUNCH_DIR):
            _exclusive_json(AUTHORITY, authority_for(suite))
            intent = _intent(suite, lease.fileno())
            _exclusive_json(INTENT, intent)
            log = LAUNCH_DIR/'controller.log'
            with log.open('xb', buffering=0) as stream:
                process = subprocess.Popen(intent['command'], cwd=ROOT, stdin=subprocess.DEVNULL,
                    stdout=stream, stderr=subprocess.STDOUT, pass_fds=(lease.fileno(),), start_new_session=True,
                    env=dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1'))
            _exclusive_json(LAUNCH_DIR/'launch_receipt.json', dict(campaign_id=CAMPAIGN, suite=preflight['suite'],
                            pid=process.pid, log=str(log)))
    return dict(status='CONTROLLER_STARTED', campaign_id=CAMPAIGN, pid=process.pid, rpc_port=EXECUTION['port'])


def _controller(suite, lane_fd):
    require(type(lane_fd) is int and os.path.samestat(os.fstat(lane_fd), LANE.stat()),
            'inherited RPC port lease differs')
    fcntl.flock(lane_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    require(not STOP.exists() and not LEDGER.exists() and load_reference(reference(INTENT)) == _intent(suite, lane_fd),
            'launch intent differs or existing campaign requires reconciliation')
    try:
        from traditional_dispatch_v2.runtime import run

        _status('STARTING', completed_shifts=0)
        ledger = Ledger(suite)
        ledger.register()

        def progress(count, job):
            _status('RUNNING', completed_shifts=count, last_run_index=job['run_index'], last_policy_kind=job['policy_kind'])

        report = run(suite, ledger, progress=progress)
        counts = Counter(slot['status'] for slot in ledger.snapshot()['slots'].values())
        require(counts == {'COMPLETE': COUNT} and reference(CORE) == suite['protected_core']
                and report == reference(Path(suite['report_directory'])/REPORT_FILENAME),
                'revised validation completion, report or protected PPO core differs')
        result = dict(status='COMPLETE', campaign_id=CAMPAIGN, suite=reference(SUITE), physical_shifts=COUNT,
                      training_shifts=0, optimizer_updates=0, protected_core=suite['protected_core'], evaluation=report)
        _exclusive_json(OUTPUT/'suite_complete.json', result)
        _status('COMPLETE', completed_shifts=COUNT, optimizer_updates=0)
        return result
    except BaseException as error:
        CONTROL.mkdir(parents=True, exist_ok=True)
        if not STOP.exists():
            message = repr(error).replace('\n', ' ').replace('\r', ' ')
            _publish(STOP, ('reason=' + message + '\n').encode('ascii', 'backslashreplace'))
        _status('STOP', reason=repr(error))
        raise


def _view_status():
    if STATUS.is_file():
        return json.loads(STATUS.read_bytes())
    if OUTPUT.is_dir():
        status = 'AWAITING_CONTROLLER_STATUS'
    else:
        status = 'PREPARED_NOT_STARTED' if (PREPARED/'launch_identity.json').is_file() else 'NOT_PREPARED'
    return dict(status=status, campaign_id=CAMPAIGN, scheduled_shifts=COUNT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, default=SUITE)
    parser.add_argument('--run-index', type=int)
    parser.add_argument('--environment-seed', type=int)
    parser.add_argument('--recording-id')
    parser.add_argument('--lane-fd', type=int, help=argparse.SUPPRESS)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight', '--check', action='store_true')
    mode.add_argument('--status', action='store_true')
    mode.add_argument('--from-anylogic-run', action='store_true')
    mode.add_argument('--controller', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.status:
        result = _view_status()
    else:
        suite = load_suite(args.suite)
        if args.preflight:
            result = _preflight(suite)
        elif args.from_anylogic_run:
            first = suite['jobs'][0]
            require((args.run_index, args.environment_seed, args.recording_id)
                    == (first['run_index'], first['experiment_seed'], first['recording_id']),
                    'first matching AnyLogic physical identity required')
            result = _start(suite)
        else:
            result = _controller(suite, args.lane_fd)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
