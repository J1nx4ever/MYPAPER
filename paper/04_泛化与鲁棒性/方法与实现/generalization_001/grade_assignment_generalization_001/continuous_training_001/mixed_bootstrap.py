"""Read-only preflight/status and the frozen operator-triggered controller."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

from mixed_plan import (
    APPROVAL, CAMPAIGN, CONTROL, CORE, EVAL_SHIFTS, LAUNCH, LEDGER, MATERIALS,
    OUTPUT, PLAN, ROOT, SUITE, TOTAL, TRAINING_SHIFTS, UPDATES, authorization_for,
    check_collisions, load_package, read, reference, require,
)
from mixed_budget import MixedLedger
from business_graph_summary_campaign_v1.health import _process_identity
from business_graph_summary_campaign_v1.launch import _exclusive_json, _launch_lock
from business_graph_summary_campaign_v1.plan_io import encoded
from grade_only_ppo_v2.runtime import _controller_lock
from online_v1_1.operations import _publish

PYTHON = '/home/jinx/miniconda3/envs/anylogic-env/bin/python'
INTENT = LAUNCH / 'launch_intent.json'
RECEIPT = LAUNCH / 'launch_receipt.json'
STOP = CONTROL / 'STOP.properties'


def _intent():
    return dict(
        suite=reference(SUITE), authority=reference(APPROVAL),
        command=[PYTHON, '-B', str(PLAN / 'mixed_bootstrap.py'), '--controller'],
        train_shifts=TRAINING_SHIFTS, eval_shifts=EVAL_SHIFTS, total_shifts=TOTAL,
    )


def _status(state, **fields):
    _publish(OUTPUT / 'run_status.json', encoded(dict(
        campaign_id=CAMPAIGN, status=state, total_shifts=TOTAL,
        updated_at_utc=datetime.now(timezone.utc).isoformat(), **fields,
    )), replace=True)


def start(suite, first_config):
    with _launch_lock(LAUNCH):
        require(not any(p.exists() for p in (INTENT, RECEIPT, APPROVAL, MATERIALS, CONTROL, LEDGER))
                and all(p == LAUNCH for p in OUTPUT.iterdir()), 'existing output requires reconciliation')
        check_collisions(suite)
        execution = first_config['execution']
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind((execution['host'], execution['port']))
        _exclusive_json(APPROVAL, authorization_for(suite))
        intent = _intent()
        _exclusive_json(INTENT, intent)
        log_path = LAUNCH / 'controller.log'
        with log_path.open('xb', buffering=0) as log:
            child = subprocess.Popen(
                intent['command'], cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True,
                env=dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1'),
            )
        process = _process_identity(child.pid)
        _exclusive_json(RECEIPT, dict(suite=intent['suite'], pid=child.pid,
            process_start_ticks=process.get('start_ticks'), log=str(log_path)))
    return dict(status='CONTINUOUS_MIXED_CONTROLLER_STARTED', pid=child.pid)


def controller():
    from mixed_runtime import run

    suite, configs = load_package()
    require(read(INTENT) == _intent() and read(APPROVAL) == authorization_for(suite),
            'operator launch intent does not bind the frozen configuration')
    ledger = MixedLedger(suite, configs)
    with _controller_lock(OUTPUT):
        require(not STOP.exists() and not LEDGER.exists(), 'previous execution requires reconciliation')
        _status('STARTING')
        try:
            run(suite, configs, ledger)
            state = ledger.snapshot()
            counts = Counter(row['status'] for row in state['jobs'].values())
            require(counts == {'COMPLETE': TOTAL}
                    and len(state['campaigns'][CAMPAIGN]['commits']) == UPDATES
                    and reference(CORE) == suite['protected_core'],
                    'campaign completion, training updates or protected core differs')
            _status('COMPLETE', completed_shifts=TOTAL, training_updates=UPDATES)
        except BaseException as error:
            CONTROL.mkdir(parents=True, exist_ok=True)
            if not STOP.exists():
                _publish(STOP, ('reason=' + repr(error).replace('\n', ' ') + '\n').encode())
            _status('STOP', reason=repr(error))
            raise


def read_status(suite, configs):
    result = dict(campaign_id=CAMPAIGN, total_shifts=TOTAL, status='PREPARED_OFFLINE_ONLY')
    progress = OUTPUT / 'run_status.json'
    if progress.is_file():
        result.update(read(progress))
    if LEDGER.is_file():
        state = MixedLedger(suite, configs).snapshot()
        result['registered_status_counts'] = dict(Counter(row['status'] for row in state['jobs'].values()))
        result['committed_training_updates'] = len(state['campaigns'][CAMPAIGN]['commits'])
    training = OUTPUT / 'training_progress.json'
    if training.is_file():
        result['training_progress'] = read(training)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=lambda path: Path(path).resolve(), default=SUITE)
    parser.add_argument('--run-index', type=int)
    parser.add_argument('--environment-seed', type=int)
    parser.add_argument('--recording-id')
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--check', action='store_true')
    modes.add_argument('--status', action='store_true')
    modes.add_argument('--from-anylogic-run', action='store_true')
    modes.add_argument('--controller', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    require(args.suite == SUITE, 'unexpected suite path')
    if args.controller:
        controller()
        return
    suite, configs = load_package()
    if args.from_anylogic_run:
        first = suite['jobs'][0]
        require((args.run_index, args.environment_seed, args.recording_id)
                == (first['run_index'], first['experiment_seed'], first['recording_id']),
                'first frozen AnyLogic identity differs')
        result = start(suite, configs[0])
    elif args.status:
        result = read_status(suite, configs)
    else:
        check_collisions(suite)
        result = dict(status='PASS_OFFLINE_NO_LISTENER', suite=reference(SUITE),
                      train_shifts=TRAINING_SHIFTS, eval_shifts=EVAL_SHIFTS, total_shifts=TOTAL)
    json.dump(result, sys.stdout, ensure_ascii=False, sort_keys=True)
    print()


if __name__ == '__main__':
    main()
