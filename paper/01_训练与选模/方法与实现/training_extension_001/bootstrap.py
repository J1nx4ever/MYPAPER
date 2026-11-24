"""Start the frozen Python controller from the first AnyLogic Main startup."""
import argparse
import json
from pathlib import Path
import os
import socket
import subprocess

from extension_common import (
    CONTROL, MATERIALS, OUTPUT, PLAN, PREPARED, ROOT, SUITE, TOTAL,
    load_package, reference, require,
)
from business_graph_summary_campaign_v1.health import _process_identity
from business_graph_summary_campaign_v1.launch import _exclusive_json, _launch_lock
from extension_budget import check_collisions

PYTHON = '/home/jinx/miniconda3/envs/anylogic-env/bin/python'


def start(suite, first_cfg):
    launch = PREPARED / 'launch'
    command = [PYTHON, '-B', str(PLAN / 'run_extension.py')]
    receipt_path = launch / 'launch_receipt.json'
    intent_path = launch / 'launch_intent.json'
    with _launch_lock(launch):
        require(not any(path.exists() for path in
                        (receipt_path, intent_path, OUTPUT, MATERIALS, CONTROL / 'STOP.properties')),
                'existing extension launch or output requires inspection')
        check_collisions(suite)
        execution = first_cfg['execution']
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind((execution['host'], execution['port']))
        _exclusive_json(intent_path, dict(suite=reference(SUITE), command=command, shifts=TOTAL))
        log_path = launch / 'controller.log'
        with log_path.open('xb', buffering=0) as log:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True,
                env=dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1'))
        process = _process_identity(child.pid)
        _exclusive_json(receipt_path, dict(suite=reference(SUITE), command=command, pid=child.pid,
            process_start_ticks=process.get('start_ticks'), log=str(log_path)))
    return dict(status='EXTENSION_CONTROLLER_STARTED', pid=child.pid)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=lambda path: Path(path).resolve(), required=True)
    parser.add_argument('--run-index', type=int, required=True)
    parser.add_argument('--environment-seed', type=int, required=True)
    parser.add_argument('--recording-id', required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--from-anylogic-run', action='store_true')
    args = parser.parse_args()
    require(args.suite == SUITE, 'unexpected extension suite')
    suite, phases = load_package()
    first = phases[0]['jobs'][0]
    require((args.run_index, args.environment_seed, args.recording_id)
            == (first['run_index'], first['experiment_seed'], first['recording_id']),
            'first frozen run identity differs')
    if args.check:
        check_collisions(suite)
        print(json.dumps(dict(status='PASS_OFFLINE_NO_LISTENER', suite=reference(SUITE), shifts=TOTAL)))
        return
    print(json.dumps(start(suite, phases[0])))


if __name__ == '__main__':
    main()
