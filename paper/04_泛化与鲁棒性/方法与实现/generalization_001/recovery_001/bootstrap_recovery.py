"""Start the existing budget from original run 9991102 on a new ready channel."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess

from recovery_common import (
    CONTROL_DIR, DESCRIPTOR, LAUNCH_DIR, OFFSET, PLAN, REMAINING, ROOT, preflight,
    reference, require,
)
from business_graph_summary_campaign_v1.health import _process_identity
from business_graph_summary_campaign_v1.launch import _exclusive_json, _launch_lock

PYTHON = '/home/jinx/miniconda3/envs/anylogic-env/bin/python'


def start(recovery):
    command = [PYTHON, '-B', str(PLAN / 'resume_generalization.py')]
    intent_path = LAUNCH_DIR / 'launch_intent.json'
    receipt_path = LAUNCH_DIR / 'launch_receipt.json'
    with _launch_lock(LAUNCH_DIR):
        require(not any(p.exists() for p in (intent_path, receipt_path, CONTROL_DIR)),
                'an existing recovery attempt must be reviewed')
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(('127.0.0.1', 50051))
        intent = dict(recovery=reference(DESCRIPTOR), command=command, remaining_shifts=REMAINING,
                      additional_shifts=0, original_suite=recovery['original_suite'])
        _exclusive_json(intent_path, intent)
        with (LAUNCH_DIR / 'controller.log').open('xb', buffering=0) as log:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True,
                env=dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1'))
        process = _process_identity(child.pid)
        _exclusive_json(receipt_path, dict(intent, pid=child.pid, process_start_ticks=process.get('start_ticks')))
    return dict(status='RECOVERY_STARTED', pid=child.pid, remaining_shifts=REMAINING)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recovery', type=Path, required=True)
    parser.add_argument('--run-index', type=int, required=True)
    parser.add_argument('--environment-seed', type=int, required=True)
    parser.add_argument('--recording-id', required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--from-anylogic-run', action='store_true')
    args = parser.parse_args()
    recovery, suite, _, _ = preflight(args.recovery)
    row = suite['jobs'][OFFSET]
    require((args.run_index, args.environment_seed, args.recording_id)
            == (row['run_index'], row['experiment_seed'], row['recording_id']),
            'recovery must start on the first unused original run')
    if args.check:
        result = dict(status='PASS_OFFLINE_RECOVERY_CHECK', remaining_shifts=REMAINING, additional_shifts=0)
    else:
        result = start(recovery)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
