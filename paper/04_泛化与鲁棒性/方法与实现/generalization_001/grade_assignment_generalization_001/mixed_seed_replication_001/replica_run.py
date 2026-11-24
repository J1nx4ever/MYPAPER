"""Prepare, validate or run one frozen BG128 replication via AnyLogic."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import socket
from statistics import fmean
import subprocess
import sys
import time

from replica_plan import CAP, EVAL, REPLICAS, TRAIN, anchored_source, load_package, locations, parent, prepare
from replica_budget import LOCKS, ReplicaLedger, assert_available, authority_path, authorization, budget_snapshot
from business_graph_summary_campaign_v1.health import _process_identity
from business_graph_summary_campaign_v1.launch import _exclusive_json, _launch_lock
from business_graph_summary_campaign_v1.plan_io import encoded, load_reference, reference
from online_v1_1.operations import _publish
from efficient_validation import validation_scope

PYTHON = '/home/jinx/miniconda3/envs/anylogic-env/bin/python'
ENTRY = Path(__file__).resolve()
HANDOFF_SECONDS = 600


def _lane_path(suite):
    return LOCKS / f"port-{suite['rpc_port']}.lock"


def _intent(suite, lane_fd):
    return dict(suite=reference(suite['suite_path']), authority=reference(authority_path(suite)),
                command=[PYTHON, '-B', str(ENTRY), '--suite', suite['suite_path'], '--controller', f'--lane-fd={lane_fd}'])


def _status(suite, status, **extra):
    path = Path(suite['python_output']) / 'run_status.json'
    _publish(path, encoded(dict(status=status, campaign_id=suite['campaign_id'], scheduled_shifts=CAP,
                                updated_at_utc=datetime.now(timezone.utc).isoformat(), **extra)), replace=True)


def _previous_complete(previous):
    output = Path(previous['python_output'])
    status_file = output / 'run_status.json'
    stop = Path(previous['control_directory']) / 'STOP.properties'
    parent.require(not stop.exists(), 'preceding seed stopped; preserve evidence before continuing')
    if not status_file.is_file():
        return False
    status = parent.read(status_file)
    parent.require(status['campaign_id'] == previous['campaign_id'] and status['status'] != 'STOP',
                   'preceding seed status or identity differs')
    if status['status'] != 'COMPLETE':
        return False
    complete = parent.read(output / 'suite_complete.json')
    parent.require(status.get('completed_shifts') == CAP and status.get('training_updates') == 32
                   and complete['status'] == 'COMPLETE'
                   and complete['suite'] == reference(previous['suite_path'])
                   and complete['training_shifts'] == TRAIN and complete['training_updates'] == 32
                   and complete['evaluation_shifts'] == EVAL and complete['evaluation_optimizer_updates'] == 0,
                   'preceding seed has no matching completion record')
    return True


def _await_previous(suite, deadline):
    position = REPLICAS.index(suite['replica'])
    if position == 0:
        return
    previous = locations(REPLICAS[position - 1])
    parent.require((Path(previous['python_output']) / 'launch/launch_receipt.json').is_file(),
                   'preceding seed has not been started')
    while not _previous_complete(previous):
        parent.require(time.monotonic() < deadline, 'waiting for preceding seed completion timed out')
        time.sleep(.25)


def start(suite):
    output = Path(suite['python_output'])
    launch = output / 'launch'
    lane = _lane_path(suite)
    lane.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + HANDOFF_SECONDS
    _await_previous(suite, deadline)
    with lane.open('a+b') as lease:
        while True:
            try:
                fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError as error:
                if suite['replica'] == REPLICAS[0] or time.monotonic() >= deadline:
                    raise ValueError(f"RPC port {suite['rpc_port']} is still reserved by another replica") from error
                time.sleep(.1)
        with _launch_lock(launch):
            parent.require(not any(launch.glob('*.json')) and not (launch / 'controller.log').exists()
                           and all(path == launch for path in output.iterdir())
                           and not Path(suite['material_root']).exists(), 'prior run artifacts require reconciliation')
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.bind(('127.0.0.1', suite['rpc_port']))
            _exclusive_json(authority_path(suite), authorization(suite))
            intent = _intent(suite, lease.fileno())
            _exclusive_json(launch / 'launch_intent.json', intent)
            log = launch / 'controller.log'
            with log.open('xb', buffering=0) as stream:
                child = subprocess.Popen(intent['command'], cwd=parent.ROOT, stdin=subprocess.DEVNULL,
                    stdout=stream, stderr=subprocess.STDOUT, pass_fds=(lease.fileno(),), start_new_session=True,
                    env=dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1'))
            _exclusive_json(launch / 'launch_receipt.json', dict(suite=intent['suite'], pid=child.pid,
                process_start_ticks=_process_identity(child.pid).get('start_ticks'), log=str(log)))
    return dict(status='REPLICA_CONTROLLER_STARTED', pid=child.pid, rpc_port=suite['rpc_port'])


def _paired_report(suite, phases, results):
    from mixed_runtime import _window_outcomes

    origin = load_reference(suite['source_completion'])
    anchor = load_reference(origin['comparison']['json'])
    source = {(row['scenario_id'], row['pair_ordinal']): row for row in anchor['rows']}
    binary = ('complete_task', 'all_four_windows_compliant', 'ore_target_met', 'waste_target_met')
    continuous = ('ore_tons', 'waste_tons', 'total_wait_seconds', 'combined_cost')
    rows, cases = [], {}
    for cfg, result in zip(phases, results, strict=True):
        scenario = cfg['scenario_id']
        report = load_reference(result['report'])
        parent.require(report['evaluation_optimizer_updates'] == 0 and report['training_state_unchanged'] is True,
                       'evaluation changed model state')
        block = []
        for job, episode in zip(cfg['jobs'], report['per_episode'], strict=True):
            old = source[scenario, job['pair_ordinal']]
            identity = episode['identity']
            parent.require(old['environment_seed'] == job['experiment_seed'] == identity['environment_seed']
                           and old['actor_sampling_seed'] == job['actor_sampling_seed'] == identity['actor_sampling_seed']
                           and identity['run_index'] == job['run_index'], 'paired evaluation identity differs')
            physical = episode['physical']
            block.append(dict(old, seed01=old['new'], new=dict(run_index=job['run_index'],
                window_outcomes=_window_outcomes(physical['windows']),
                **{key: physical[key] for key in (*binary, *continuous)})))
        parent.require(len(block) == 100, 'paired scenario count differs')
        rows.extend(block)
        cases[scenario] = dict(
            counts={key: {arm: sum(row[arm][key] for row in block) for arm in ('old', 'seed01', 'new')}
                    for key in binary},
            mean_delta={arm: {key: fmean(row['new'][key] - row[arm][key] for row in block) for key in continuous}
                        for arm in ('old', 'seed01')})
    return dict(replica=suite['replica'], paired_slots=len(rows), new_physical_evaluation_shifts=EVAL,
                source_report=origin['comparison']['json'], comparisons=cases, rows=rows)


def controller(suite, phases, lane_fd):
    import torch
    import mixed_runtime as runtime
    from business_graph_summary_campaign_v1.progress import write_progress
    from grade_only_ppo_v2.runtime import _controller_lock
    from recovery_002.storage_scope import bg128_storage
    from transport import per_job_transport

    output, control = (Path(suite[key]) for key in ('python_output', 'control_directory'))
    lease = os.fstat(lane_fd)
    parent.require(os.path.samestat(lease, _lane_path(suite).stat()), 'inherited RPC lane lease differs')
    fcntl.flock(lane_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    stop = control / 'STOP.properties'
    with _controller_lock(output):
        try:
            parent.require(parent.read(output / 'launch/launch_intent.json') == _intent(suite, lane_fd)
                           and not stop.exists() and not Path(suite['budget_path']).exists(),
                           'launch intent differs or prior execution needs reconciliation')
            _status(suite, 'STARTING')
            torch.set_num_threads(suite['cpu_threads'])
            ledger = ReplicaLedger(suite, phases)
            ledger.register()
            with bg128_storage(), per_job_transport():
                initial = cursor = runtime._initial(suite)
                checkpoints, results = [], []
                for cfg, ref in zip(phases, suite['phase_configs'], strict=True):
                    runtime._check_sources(suite)
                    with validation_scope(cfg['jobs'], runtime) as proof:
                        if cfg['policy_arm'] == 'train':
                            cursor = runtime._train(suite, cfg, ref, cursor, ledger)
                            checkpoints.append(cursor)
                        else:
                            parent.require(len(checkpoints) == 32, 'evaluation requires committed u32')
                            results.append(runtime._evaluate(suite, cfg, ref, cursor, ledger))
                    parent.require(proof.count['exact_replays'] == len(cfg['jobs']),
                                   'phase is missing an exact admission proof')
                    runtime._same_or_save(Path(cfg['python_output']) / 'validation_summary.json', proof.count)
            state = ledger.resume()
            parent.require(len(checkpoints) == 32 and len(results) == 7
                           and len(state['campaigns'][suite['campaign_id']]['commits']) == 32
                           and Counter(row['status'] for row in state['jobs'].values()) == {'COMPLETE': CAP}
                           and reference(parent.CORE) == suite['protected_core'], 'replica completion counts or PPO core differ')
            path = Path(suite['report_directory']) / 'paired_evaluation.json'
            runtime._same_or_save(path, _paired_report(suite, phases[32:], results))
            runtime._same_or_save(output / 'suite_complete.json', dict(status='COMPLETE', suite=reference(suite['suite_path']),
                initial_checkpoint=initial, new_checkpoint=cursor, training_checkpoints=checkpoints,
                training_updates=32, training_shifts=TRAIN, evaluation_shifts=EVAL, evaluation_optimizer_updates=0,
                phase_results=results, comparison=reference(path)))
            write_progress(suite, status='COMPLETE', training_updates=32, completed_shifts=CAP)
            _status(suite, 'COMPLETE', completed_shifts=CAP, training_updates=32)
        except BaseException as error:
            control.mkdir(parents=True, exist_ok=True)
            if not stop.exists():
                reason = repr(error).replace('\n', ' ').replace('\r', ' ')
                _publish(stop, ('reason=' + reason + '\n').encode('ascii', 'backslashreplace'))
            _status(suite, 'STOP', reason=repr(error))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=lambda path: str(Path(path).resolve()))
    parser.add_argument('--run-index', type=int)
    parser.add_argument('--environment-seed', type=int)
    parser.add_argument('--recording-id')
    parser.add_argument('--lane-fd', type=int, help=argparse.SUPPRESS)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ('prepare', 'check', 'status', 'from-anylogic-run', 'controller'):
        modes.add_argument(f'--{mode}', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        parent.require(args.suite is None, 'preparation creates all four suites together')
        prepare()
        return
    replica, = [name for name in REPLICAS if locations(name)['suite_path'] == args.suite]
    source = anchored_source()
    suite, phases = load_package(replica, source)
    if args.from_anylogic_run:
        first = suite['jobs'][0]
        parent.require((args.run_index, args.environment_seed, args.recording_id)
                       == (first['run_index'], first['experiment_seed'], first['recording_id']),
                       'first physical AnyLogic identity differs')
        result = start(suite)
    elif args.controller:
        parent.require(type(args.lane_fd) is int and args.lane_fd > 2, 'controller needs an inherited lane lease')
        controller(suite, phases, args.lane_fd)
        return
    else:
        output = Path(suite['python_output'])
        result = dict(status='PREPARED_OFFLINE_ONLY', rpc_port=suite['rpc_port'], replica=replica)
        if (output / 'run_status.json').is_file():
            result.update(parent.read(output / 'run_status.json'))
        if Path(suite['budget_path']).is_file():
            state = ReplicaLedger(suite, phases).snapshot()
            result['status_counts'] = dict(Counter(row['status'] for row in state['jobs'].values()))
        elif args.check:
            result['available'] = assert_available([suite], budget_snapshot())
        if args.check:
            result['suite'] = reference(suite['suite_path'])
            result['java'] = suite['java_validation']
    sys.stdout.write(encoded(result).decode())


if __name__ == '__main__':
    main()
