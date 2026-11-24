"""Prepare, verify or execute the two s01 ablations through AnyLogic."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from ablation_plan import ALP_RUNS, CAP, EVAL, PREPARED, TRAIN, UNITS, load_package, locations, parent, prepare
from ablation_budget import AblationLedger, LOCKS, assert_available, authority_path, authorization, budget_snapshot
from ablation_report import complete_report, paired_report
from ablation_runtime import runtime_scope
from business_graph_summary_campaign_v1.health import _process_identity
from business_graph_summary_campaign_v1.launch import _exclusive_json, _launch_lock
from business_graph_summary_campaign_v1.plan_io import encoded, load_reference, reference
from business_graph_summary_campaign_v1.progress import write_progress
from efficient_validation import validation_scope
from grade_only_ppo_v2.runtime import _controller_lock
from online_v1_1.operations import _publish
from recovery_002.storage_scope import bg128_storage
from replica_run import _previous_complete, HANDOFF_SECONDS, PYTHON
from transport import per_job_transport

ENTRY = Path(__file__).resolve()


def _lane(suite):
    return LOCKS / f"port-{suite['rpc_port']}.lock"


def _intent(suite, lane_fd):
    return dict(suite=reference(suite['suite_path']), authority=reference(authority_path(suite)),
                command=[PYTHON, '-B', str(ENTRY), '--suite', suite['suite_path'], '--controller', f'--lane-fd={lane_fd}'])


def _status(suite, status, **extra):
    path = Path(suite['python_output']) / 'run_status.json'
    state = dict(status=status, campaign_id=suite['campaign_id'], scheduled_shifts=CAP,
                 updated_at_utc=datetime.now(timezone.utc).isoformat(), **extra)
    _publish(path, encoded(state), replace=True)


def _start(suite):
    index = UNITS.index(suite['unit'])
    deadline = time.monotonic() + HANDOFF_SECONDS
    if index:
        previous = locations(UNITS[index - 1])
        parent.require((Path(previous['python_output']) / 'launch/launch_receipt.json').is_file(),
                       'preceding ablation has not been started')
        while not _previous_complete(previous):
            parent.require(time.monotonic() < deadline, 'previous ablation has not finished')
            time.sleep(.25)
    launch = Path(suite['python_output']) / 'launch'
    lane = _lane(suite)
    lane.parent.mkdir(parents=True, exist_ok=True)
    with lane.open('a+b') as lease:
        while True:
            try:
                fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                parent.require(index and time.monotonic() < deadline, 'port is reserved by another campaign')
                time.sleep(.1)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(('127.0.0.1', suite['rpc_port']))
        assert_available([suite], budget_snapshot())
        with _launch_lock(launch):
            log = launch / 'controller.log'
            parent.require(not any(launch.glob('*.json')) and not log.exists()
                           and all(path == launch for path in launch.parent.iterdir())
                           and not Path(suite['material_root']).exists(), 'prior run evidence requires reconciliation')
            _exclusive_json(authority_path(suite), authorization(suite))
            intent = _intent(suite, lease.fileno())
            _exclusive_json(launch / 'launch_intent.json', intent)
            with log.open('xb', buffering=0) as stream:
                child = subprocess.Popen(intent['command'], cwd=parent.ROOT, stdin=subprocess.DEVNULL,
                    stdout=stream, stderr=subprocess.STDOUT, pass_fds=(lease.fileno(),), start_new_session=True,
                    env=dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1'))
            _exclusive_json(launch / 'launch_receipt.json', dict(suite=intent['suite'], pid=child.pid,
                process_start_ticks=_process_identity(child.pid).get('start_ticks'), log=str(log)))
    return dict(status='ABLATION_CONTROLLER_STARTED', pid=child.pid, rpc_port=suite['rpc_port'])


def controller(suite, phases, lane_fd):
    import torch

    output, control = (Path(suite[key]) for key in ('python_output', 'control_directory'))
    parent.require(os.path.samestat(os.fstat(lane_fd), _lane(suite).stat()), 'inherited RPC lane lease differs')
    fcntl.flock(lane_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    stop = control / 'STOP.properties'
    with _controller_lock(output):
        try:
            parent.require(parent.read(output / 'launch/launch_intent.json') == _intent(suite, lane_fd)
                           and not stop.exists() and not Path(suite['budget_path']).exists(),
                           'launch intent differs or prior execution needs reconciliation')
            _status(suite, 'STARTING')
            torch.set_num_threads(suite['cpu_threads'])
            ledger = AblationLedger(suite, phases)
            updates = parent.UPDATES
            checkpoints, results = [], []
            with bg128_storage(), per_job_transport(), runtime_scope(suite) as runtime:
                runtime._check_sources(suite)
                ledger.register()
                initial = checkpoint = runtime._initial(suite)
                for cfg, ref in zip(phases, suite['phase_configs'], strict=True):
                    runtime._check_sources(suite)
                    with validation_scope(cfg['jobs'], runtime) as proof:
                        if cfg['policy_arm'] == 'train':
                            checkpoint = runtime._train(suite, cfg, ref, checkpoint, ledger)
                            checkpoints.append(checkpoint)
                        else:
                            parent.require(len(checkpoints) == updates, 'evaluation requires committed u32')
                            results.append(runtime._evaluate(suite, cfg, ref, checkpoint, ledger))
                    parent.require(proof.count['exact_replays'] == len(cfg['jobs']),
                                   'phase has missing exact Actor admission proofs')
                    runtime._same_or_save(Path(cfg['python_output']) / 'validation_summary.json', proof.count)
                state = ledger.resume()
                parent.require(len(checkpoints) == updates and len(results) == len(parent.EVAL_CASES)
                               and len(state['campaigns'][suite['campaign_id']]['commits']) == updates
                               and Counter(row['status'] for row in state['jobs'].values()) == {'COMPLETE': CAP}
                               and reference(parent.CORE) == suite['protected_core'], 'final ledger or PPO core differs')
                path = Path(suite['report_directory']) / 'paired_evaluation.json'
                runtime._same_or_save(path, paired_report(suite, phases[updates:], results, checkpoint))
                complete = dict(status='COMPLETE', suite=reference(suite['suite_path']), initial_checkpoint=initial,
                                new_checkpoint=checkpoint, training_updates=updates, training_shifts=TRAIN,
                                evaluation_shifts=EVAL, evaluation_optimizer_updates=0, phase_results=results,
                                comparison=reference(path))
                runtime._same_or_save(output / 'suite_complete.json', complete)
                if suite['unit'] == UNITS[-1]:
                    complete_report(runtime._same_or_save)
            write_progress(suite, status='COMPLETE', training_updates=updates, completed_shifts=CAP)
            _status(suite, 'COMPLETE', completed_shifts=CAP, training_updates=updates)
        except BaseException as error:
            control.mkdir(parents=True, exist_ok=True)
            if not stop.exists():
                reason = repr(error).replace('\n', ' ').replace('\r', ' ')
                _publish(stop, ('reason=' + reason + '\n').encode('ascii', 'backslashreplace'))
            _status(suite, 'STOP', reason=repr(error))
            raise


def _check():
    start = load_reference(reference(PREPARED / 'launch_identity.json'))
    parent.require(start['order'] == list(UNITS) and start['physical_shifts'] == ALP_RUNS
                   and start['java_helper'] == reference(PREPARED / 'Baseline1CampaignPlan.java'),
                   'combined Java launch identity differs')
    suites, pending = [], []
    for unit, identity in zip(UNITS, start['launch_identities'], strict=True):
        suite, _ = load_package(unit)
        frozen = load_reference(identity)
        parent.require(frozen['unit'] == unit and frozen['suite'] == reference(suite['suite_path'])
                       and frozen['java_helper'] == suite['java_helper'], 'launch unit identity differs')
        with runtime_scope(suite) as runtime:
            runtime._check_sources(suite)
        suites.append(unit)
        if not Path(suite['budget_path']).is_file():
            pending.append(suite)
    return dict(status='OFFLINE_CHECKED', units=suites, physical_shifts=ALP_RUNS,
                budget=assert_available(pending, budget_snapshot()), combined_java=start['java_helper'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=lambda value: str(Path(value).resolve()))
    parser.add_argument('--run-index', type=int)
    parser.add_argument('--environment-seed', type=int)
    parser.add_argument('--recording-id')
    parser.add_argument('--lane-fd', type=int, help=argparse.SUPPRESS)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ('prepare', 'check', 'status', 'from-anylogic-run', 'controller'):
        modes.add_argument(f'--{mode}', action='store_true')
    args = parser.parse_args()
    if args.prepare or args.check:
        parent.require(args.suite is None, 'prepare/check applies to the combined package')
        result = prepare() if args.prepare else _check()
    else:
        matches = [unit for unit in UNITS if locations(unit)['suite_path'] == args.suite]
        parent.require(len(matches) == 1, 'suite path must identify a prepared ablation')
        suite, phases = load_package(matches[0])
        if args.from_anylogic_run:
            first = suite['jobs'][0]
            parent.require((args.run_index, args.environment_seed, args.recording_id)
                           == (first['run_index'], first['experiment_seed'], first['recording_id']),
                           'first physical AnyLogic identity differs')
            result = _start(suite)
        elif args.controller:
            parent.require(type(args.lane_fd) is int and args.lane_fd > 2, 'controller needs inherited lane lease')
            controller(suite, phases, args.lane_fd)
            return
        else:
            output = Path(suite['python_output'])
            result = dict(status='PREPARED_OFFLINE_ONLY', unit=suite['unit'])
            if (output / 'run_status.json').is_file():
                result = parent.read(output / 'run_status.json')
            if (output / 'training_progress.json').is_file():
                result['progress'] = parent.read(output / 'training_progress.json')
            if Path(suite['budget_path']).is_file():
                rows = AblationLedger(suite, phases).snapshot()['jobs']
                result['status_counts'] = dict(Counter(row['status'] for row in rows.values()))
    sys.stdout.write(encoded(result).decode())


if __name__ == '__main__':
    main()
