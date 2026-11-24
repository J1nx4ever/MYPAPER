"""Bind the first AnyLogic startup to one frozen u32 generalization run."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess

from gen_common import (
    APPROVAL, CONTROL, LAUNCH, MATERIALS, OUTPUT, PLAN, ROOT, SUITE, TOTAL,
    authorization_for, load_package, reference, require,
)
from business_graph_summary_campaign_v1.health import _process_identity
from business_graph_summary_campaign_v1.launch import _exclusive_json, _launch_lock
from gen_budget import check_collisions

PYTHON = '/home/jinx/miniconda3/envs/anylogic-env/bin/python'


def start(suite, first_cfg):
    command = [PYTHON, '-B', str(PLAN / 'run_generalization.py')]
    intent_path = LAUNCH / 'launch_intent.json'
    receipt_path = LAUNCH / 'launch_receipt.json'
    with _launch_lock(LAUNCH):
        require(not any(p.exists() for p in (intent_path, receipt_path, APPROVAL, MATERIALS, CONTROL))
                and all(p == LAUNCH for p in OUTPUT.iterdir()),
                'existing generalization start or output requires review')
        check_collisions(suite)
        execution = first_cfg['execution']
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind((execution['host'], execution['port']))
        _exclusive_json(APPROVAL, authorization_for(suite))
        _exclusive_json(intent_path, dict(suite=reference(SUITE), authorization=reference(APPROVAL),
                                          command=command, shifts=TOTAL))
        log_path = LAUNCH / 'controller.log'
        with log_path.open('xb', buffering=0) as log:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                env=dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1'))
        process = _process_identity(child.pid)
        _exclusive_json(receipt_path, dict(suite=reference(SUITE), pid=child.pid, command=command,
                                          process_start_ticks=process.get('start_ticks'), log=str(log_path)))
    return dict(status='GENERALIZATION_CONTROLLER_STARTED', pid=child.pid)


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
    require(args.suite == SUITE, 'unexpected generalization suite')
    suite, phases = load_package()
    first = phases[0]['jobs'][0]
    require((args.run_index, args.environment_seed, args.recording_id)
            == (first['run_index'], first['experiment_seed'], first['recording_id']),
            'first frozen run identity differs')
    if args.check:
        check_collisions(suite)
        print(json.dumps(dict(status='PASS_OFFLINE_NO_LISTENER', suite=reference(SUITE), shifts=TOTAL)))
    else:
        print(json.dumps(start(suite, phases[0])))


if __name__ == '__main__':
    main()
