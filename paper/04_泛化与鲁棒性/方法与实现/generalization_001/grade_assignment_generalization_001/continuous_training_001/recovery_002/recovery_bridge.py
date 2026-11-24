"""Resume the frozen continuous BG128 campaign after its first completed shift."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess

from recovery_plan import (
    COMPLETED_RUN, FIRST_REMAINING, OFFSET, RECOVERY, RECOVERY_CONTROL,
    RECOVERY_OUTPUT, REMAINING, ROOT, check_start, load_recovery,
)
from mixed_plan import CAMPAIGN, CORE, OUTPUT, SUITE, TOTAL, UPDATES, read, reference, require
from mixed_budget import MixedLedger
from business_graph_summary_campaign_v1.budget import connection
from business_graph_summary_campaign_v1.health import _process_identity
from business_graph_summary_campaign_v1.launch import _exclusive_json, _launch_lock
from business_graph_summary_campaign_v1.plan_io import encoded
from grade_only_ppo_v2.runtime import _controller_lock
from online_v1_1.operations import _publish
from storage_scope import bg128_storage

PYTHON = '/home/jinx/miniconda3/envs/anylogic-env/bin/python'
LAUNCH = RECOVERY_OUTPUT / 'launch'
AUTHORITY = LAUNCH / 'run_authority.json'
INTENT = LAUNCH / 'launch_intent.json'
RECEIPT = LAUNCH / 'launch_receipt.json'
LOG = LAUNCH / 'controller.log'
STOP = RECOVERY_CONTROL / 'STOP.properties'
STATUS = RECOVERY_OUTPUT / 'run_status.json'


class RecoveryLedger(MixedLedger):
    """Reuse the registered inventory and distinguish a pending update from corruption."""

    def register(self):
        state = self.snapshot()
        counts = Counter(row['status'] for row in state['jobs'].values())
        require(counts == {'COMPLETE': OFFSET, 'PLANNED': REMAINING}
                and state['jobs'][str(COMPLETED_RUN)]['status'] == 'COMPLETE'
                and not state['campaigns'][CAMPAIGN]['commits'], 'incorrect recovery start ledger')

    def commit_record(self, update):
        require(type(update) is int and update in self.update_jobs, 'invalid mixed training update')
        with connection(self.path, readonly=True) as database:
            present = database.execute('SELECT 1 FROM commits WHERE campaign=? AND update_id=?',
                                       (self.campaign_id, update)).fetchone()
        return None if present is None else super().commit_record(update)


def _authority(suite):
    job = suite['jobs'][OFFSET]
    return dict(schema='bg128-mixed-post-first-shift-recovery-authority-v1', campaign_id=CAMPAIGN,
                recovery=reference(RECOVERY), original_suite=reference(SUITE),
                authorized_by_anylogic_run=True, prior_completed_run=COMPLETED_RUN,
                first_remaining_run=FIRST_REMAINING, environment_seed=job['experiment_seed'],
                recording_id=job['recording_id'], remaining_shifts=REMAINING, additional_shifts=0)


def _intent():
    return dict(recovery=reference(RECOVERY), authority=reference(AUTHORITY),
                command=[PYTHON, '-B', str(Path(__file__).resolve()), '--controller'],
                completed_before=OFFSET, remaining_shifts=REMAINING, additional_shifts=0)


def _write_status(status, **fields):
    data = dict(schema='bg128-mixed-post-first-shift-recovery-progress-v1', campaign_id=CAMPAIGN,
                status=status, total_shifts=TOTAL, additional_shifts=0,
                updated_at_utc=datetime.now(timezone.utc).isoformat(), **fields)
    _publish(STATUS, encoded(data), replace=True)


def _recheck_first(state, suite):
    import torch
    from business_graph_summary_campaign_v1.collection import load_admitted_episode
    from mixed_runtime import _exact_verification, _verify_observation_sources

    job = suite['jobs'][0]
    original_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(suite['cpu_threads'])
        with bg128_storage(), _exact_verification([job]):
            episode = load_admitted_episode(state['jobs'][str(COMPLETED_RUN)]['receipt'])
            _verify_observation_sources(episode, job)
        require(len(episode['actions']) == 184 and len(episode['memory'].settlements) == 4,
                'reconciled shift content differs')
    finally:
        torch.set_num_threads(original_threads)


def start(manifest, suite, configs):
    with _launch_lock(LAUNCH):
        require(not any(path.exists() for path in (AUTHORITY, INTENT, RECEIPT, LOG, RECOVERY_CONTROL, STATUS)),
                'existing recovery attempt requires review')
        state = check_start(manifest, suite, configs)
        _recheck_first(state, suite)
        execution = configs[0]['execution']
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind((execution['host'], execution['port']))
        _exclusive_json(AUTHORITY, _authority(suite))
        intent = _intent()
        _exclusive_json(INTENT, intent)
        with LOG.open('xb', buffering=0) as log:
            child = subprocess.Popen(intent['command'], cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True,
                env=dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1'))
        process = _process_identity(child.pid)
        _exclusive_json(RECEIPT, dict(recovery=intent['recovery'], pid=child.pid,
            process_start_ticks=process.get('start_ticks'), log=str(LOG)))
    return dict(status='CONTINUOUS_MIXED_RECOVERY_CONTROLLER_STARTED', pid=child.pid,
                first_remaining_run=FIRST_REMAINING, remaining_shifts=REMAINING, additional_shifts=0)


def controller():
    from mixed_runtime import run

    manifest, suite, configs = load_recovery()
    require(read(AUTHORITY) == _authority(suite) and read(INTENT) == _intent(),
            'operator recovery or launch intent differs')
    with _controller_lock(OUTPUT):
        require(not STOP.exists() and not STATUS.exists(), 'previous continuation requires review')
        _write_status('STARTING', first_remaining_run=FIRST_REMAINING)
        try:
            state = check_start(manifest, suite, configs)
            _recheck_first(state, suite)
            active_suite = dict(suite, control_directory=str(RECOVERY_CONTROL))
            ledger = RecoveryLedger(suite, configs)
            with bg128_storage():
                run(active_suite, configs, ledger)
            state = ledger.snapshot()
            require(Counter(row['status'] for row in state['jobs'].values()) == {'COMPLETE': TOTAL}
                    and set(state['campaigns'][CAMPAIGN]['commits']) == {str(i) for i in range(1, UPDATES + 1)}
                    and reference(CORE) == suite['protected_core'], 'completion or protected core differs')
            _write_status('COMPLETE', completed_shifts=TOTAL, training_updates=UPDATES)
        except BaseException as error:
            RECOVERY_CONTROL.mkdir(parents=True, exist_ok=True)
            if not STOP.exists():
                _publish(STOP, ('reason=' + repr(error).replace('\n', ' ') + '\n').encode())
            _write_status('STOP', reason=repr(error))
            raise


def read_status(suite, configs):
    state = RecoveryLedger(suite, configs).snapshot()
    result = dict(campaign_id=CAMPAIGN, status='RECOVERY_PREPARED_OFFLINE_ONLY',
                  total_shifts=TOTAL, first_remaining_run=FIRST_REMAINING, additional_shifts=0)
    if STATUS.is_file():
        result.update(read(STATUS))
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
        first = suite['jobs'][OFFSET]
        require((args.run_index, args.environment_seed, args.recording_id)
                == (FIRST_REMAINING, first['experiment_seed'], first['recording_id']),
                'recovery must start at the first unused identity')
        result = start(manifest, suite, configs)
    elif args.status:
        result = read_status(suite, configs)
    else:
        state = check_start(manifest, suite, configs)
        _recheck_first(state, suite)
        result = dict(status='PASS_OFFLINE_NO_LISTENER', recovery=reference(RECOVERY),
                      verified_completed_run=COMPLETED_RUN, first_remaining_run=FIRST_REMAINING,
                      remaining_shifts=REMAINING, additional_shifts=0)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()
