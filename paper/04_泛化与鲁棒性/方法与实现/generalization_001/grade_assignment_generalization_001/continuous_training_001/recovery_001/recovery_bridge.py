"""Resume the continuous mixed campaign before its first physical shift."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ORIGINAL_PLAN = HERE.parent
sys.path.insert(0, str(ORIGINAL_PLAN))

from recovery_plan import (
    RECOVERY, ROOT, RECOVERY_OUTPUT, RECOVERY_CONTROL, load_recovery, check_previous_stop,
)
from mixed_plan import CAMPAIGN, CORE, OUTPUT, SUITE, TOTAL, UPDATES, read, reference, require
from mixed_budget import MixedLedger
from business_graph_summary_campaign_v1.budget import connection
from business_graph_summary_campaign_v1.health import _process_identity
from business_graph_summary_campaign_v1.launch import _exclusive_json, _launch_lock
from business_graph_summary_campaign_v1.plan_io import encoded
from grade_only_ppo_v2.runtime import _controller_lock
from online_v1_1.operations import _publish

PYTHON = '/home/jinx/miniconda3/envs/anylogic-env/bin/python'
RECOVERY_LAUNCH = RECOVERY_OUTPUT / 'launch'
AUTHORITY = RECOVERY_LAUNCH / 'run_authority.json'
INTENT = RECOVERY_LAUNCH / 'launch_intent.json'
RECEIPT = RECOVERY_LAUNCH / 'launch_receipt.json'
LOG = RECOVERY_LAUNCH / 'controller.log'
STOP = RECOVERY_CONTROL / 'STOP.properties'
PROGRESS = RECOVERY_OUTPUT / 'run_status.json'


class RecoveryLedger(MixedLedger):
    """Keep the original registry identity and make a fresh update query optional."""

    def register(self):
        state = self.snapshot()
        require(Counter(row['status'] for row in state['jobs'].values()) == {'PLANNED': TOTAL}
                and not state['campaigns'][CAMPAIGN]['commits'],
                'original registry is not at the first untouched slot')

    def commit_record(self, update):
        require(type(update) is int and update in self.update_jobs, 'invalid mixed training update')
        with connection(self.path, readonly=True) as database:
            present = database.execute('SELECT 1 FROM commits WHERE campaign=? AND update_id=?',
                                       (self.campaign_id, update)).fetchone()
        if present is None:
            return None
        return super().commit_record(update)


def _authority(suite):
    first = suite['jobs'][0]
    return dict(schema='bg128-mixed-preclaim-recovery-authority-v1', campaign_id=CAMPAIGN,
                recovery=reference(RECOVERY), original_suite=reference(SUITE),
                authorized_by_anylogic_run=True, first_run_index=first['run_index'],
                first_environment_seed=first['experiment_seed'], first_recording_id=first['recording_id'],
                existing_registered_shifts=TOTAL, additional_registered_shifts=0)


def _intent():
    return dict(recovery=reference(RECOVERY), authority=reference(AUTHORITY),
                command=[PYTHON, '-B', str(Path(__file__).resolve()), '--suite', str(RECOVERY), '--controller'],
                existing_registered_shifts=TOTAL, additional_registered_shifts=0)


def _status(status, **fields):
    value = dict(schema='bg128-mixed-preclaim-recovery-progress-v1', campaign_id=CAMPAIGN,
                 status=status, total_shifts=TOTAL, additional_registered_shifts=0,
                 updated_at_utc=datetime.now(timezone.utc).isoformat(), **fields)
    _publish(PROGRESS, encoded(value), replace=True)


def start(manifest, suite, configs):
    """Detach the fixed controller only for the matching operator-triggered run."""
    with _launch_lock(RECOVERY_LAUNCH):
        require(not any(path.exists() for path in (AUTHORITY, INTENT, RECEIPT, LOG, RECOVERY_CONTROL, PROGRESS)),
                'existing recovery attempt requires review')
        check_previous_stop(manifest, suite, configs)
        execution = configs[0]['execution']
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind((execution['host'], execution['port']))
        _exclusive_json(AUTHORITY, _authority(suite))
        intent = _intent()
        _exclusive_json(INTENT, intent)
        with LOG.open('xb', buffering=0) as logfile:
            process = subprocess.Popen(intent['command'], cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=logfile, stderr=subprocess.STDOUT, start_new_session=True,
                env=dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1'))
        identity = _process_identity(process.pid)
        _exclusive_json(RECEIPT, dict(recovery=intent['recovery'], pid=process.pid,
            process_start_ticks=identity.get('start_ticks'), log=str(LOG)))
    return dict(status='CONTINUOUS_MIXED_RECOVERY_CONTROLLER_STARTED', pid=process.pid,
                existing_registered_shifts=TOTAL, additional_registered_shifts=0)


def controller():
    from mixed_runtime import run

    manifest, suite, configs = load_recovery()
    require(read(AUTHORITY) == _authority(suite) and read(INTENT) == _intent(),
            'operator recovery authority or launch intent differs')
    with _controller_lock(OUTPUT):
        require(not STOP.exists() and not PROGRESS.exists(), 'previous recovery controller requires review')
        check_previous_stop(manifest, suite, configs)
        active_suite = dict(suite, control_directory=str(RECOVERY_CONTROL))
        ledger = RecoveryLedger(suite, configs)
        _status('STARTING', first_run_index=suite['jobs'][0]['run_index'])
        try:
            run(active_suite, configs, ledger)
            state = ledger.snapshot()
            counts = Counter(row['status'] for row in state['jobs'].values())
            require(counts == {'COMPLETE': TOTAL}
                    and set(state['campaigns'][CAMPAIGN]['commits']) == {str(update) for update in range(1, UPDATES + 1)}
                    and reference(CORE) == suite['protected_core'], 'recovered completion or protected core differs')
            _status('COMPLETE', completed_shifts=TOTAL, training_updates=UPDATES)
        except BaseException as error:
            RECOVERY_CONTROL.mkdir(parents=True, exist_ok=True)
            if not STOP.exists():
                _publish(STOP, ('reason=' + repr(error).replace('\n', ' ') + '\n').encode())
            _status('STOP', reason=repr(error))
            raise


def read_status(suite, configs):
    state = RecoveryLedger(suite, configs).snapshot()
    result = dict(campaign_id=CAMPAIGN, status='RECOVERY_PREPARED_OFFLINE_ONLY',
                  total_shifts=TOTAL, additional_registered_shifts=0)
    if PROGRESS.is_file():
        result.update(read(PROGRESS))
    result['registered_status_counts'] = dict(Counter(row['status'] for row in state['jobs'].values()))
    result['committed_training_updates'] = len(state['campaigns'][CAMPAIGN]['commits'])
    progress = OUTPUT / 'training_progress.json'
    if progress.is_file():
        result['training_progress'] = read(progress)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=lambda value: Path(value).resolve(), default=RECOVERY)
    parser.add_argument('--run-index', type=int)
    parser.add_argument('--environment-seed', type=int)
    parser.add_argument('--recording-id')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--status', action='store_true')
    mode.add_argument('--from-anylogic-run', action='store_true')
    mode.add_argument('--controller', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    require(args.suite == RECOVERY, 'unexpected recovery descriptor path')
    if args.controller:
        controller()
        return
    manifest, suite, configs = load_recovery()
    if args.from_anylogic_run:
        first = suite['jobs'][0]
        require((args.run_index, args.environment_seed, args.recording_id)
                == (first['run_index'], first['experiment_seed'], first['recording_id']),
                'operator recovery must begin at the original untouched first run')
        result = start(manifest, suite, configs)
    elif args.status:
        result = read_status(suite, configs)
    else:
        check_previous_stop(manifest, suite, configs)
        result = dict(status='PASS_OFFLINE_NO_LISTENER', recovery=reference(RECOVERY),
                      original_suite=reference(SUITE), planned_shifts=TOTAL, additional_registered_shifts=0)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()
