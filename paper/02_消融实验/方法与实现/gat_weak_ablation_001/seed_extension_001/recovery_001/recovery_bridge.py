"""Continue the registered s04 GAT1 evaluation after the disk-full interruption."""

import argparse
from collections import Counter
import csv
import fcntl
import os
from pathlib import Path
import socket
import subprocess
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import extension_run as original
from extension_plan import CAP, GOLDEN_EVAL_SHA, PREPARED, locations, parent
from extension_budget import ExtensionLedger, authority_path
from business_graph_summary_campaign_v1.plan_io import encoded, load_reference, reference, write
from business_graph_summary_campaign_v1.health import _process_identity
from business_graph_summary_campaign_v1.launch import _exclusive_json, _launch_lock
from grade_only_ppo_v2.runtime import _controller_lock
from online_v1_1.operations import _publish
from replica_run import PYTHON


UNIT = 'gat1-s04'
FIRST = 10049255
IDENTITY = HERE / 'prepared_001' / 'restart_identity.json'
UNITS = (UNIT, 'full-s05', 'gat1-s05')


def manifest():
    original_launch = load_reference(reference(PREPARED / 'launch_identity.json'))
    units = []
    for unit, low, high in zip(UNITS, [FIRST, 10049449, 10051173], [10049448, 10051172, 10052896], strict=True):
        idx = original_launch['order'].index(unit)
        launch_ref = original_launch['launch_identities'][idx]
        launch = load_reference(launch_ref)
        roster = launch['identities']
        parent.require(reference(roster['path']) == roster, 'original roster changed')
        with Path(roster['path']).open(newline='', encoding='utf-8') as stream:
            rows = [r for r in csv.DictReader(stream) if int(r['run_index']) >= low]
        parent.require([int(r['run_index']) for r in rows] == list(range(low, high + 1)),
                       'physical run inventory differs')
        first = rows[0]
        cfg = next(p for p in launch['phases'] if p['recording_id'] == first['recording_id'])
        config = cfg['phase_config']
        parent.require(first['config_sha256'] == config['sha256'] == reference(config['path'])['sha256'],
                       'first physical configuration changed')
        units.append(dict(unit=unit, first=first, first_config=config, first_run=low, last_run=high,
                          physical_shifts=len(rows), identities=roster, previous_launch_identity=launch_ref,
                          python_output=launch['python_output'], anylogic_event_root=launch['anylogic_event_root']))
    sources = [HERE / name for name in ('recovery_bridge.py', 'eval_resume.py', 'recovery_ledger.py')]
    return dict(status='FROZEN_RECOVERY', original_launch=reference(PREPARED / 'launch_identity.json'),
                fixed_evaluation_slot_sha256=GOLDEN_EVAL_SHA, units=units, physical_shifts=3642,
                protected_core=reference(parent.CORE), recovery_sources=[reference(path) for path in sources])


def verify_start(suite, phases):
    stop = Path(suite['control_directory']) / 'STOP.properties'
    parent.require(not stop.exists() and not (Path(suite['python_output']) / 'suite_complete.json').exists(),
                   'campaign already stopped or sealed')
    parent.require(parent.read(IDENTITY) == manifest(), 'sealed recovery metadata changed')
    parent.require(reference(parent.CORE) == suite['protected_core'] and Path(authority_path(suite)).is_file(),
                   'original PPO or execution authority is missing')
    state = ExtensionLedger(suite, phases).resume()
    rows = state['jobs']
    parent.require(len(rows) == CAP and all(row['status'] == ('COMPLETE' if int(run) < FIRST else 'PLANNED')
                                           for run, row in rows.items()), 'recovery frontier differs')
    parent.require(len(state['campaigns'][suite['campaign_id']]['commits']) == parent.UPDATES,
                   'original training update chain is incomplete')
    return Counter(row['status'] for row in rows.values())


def start(suite, phases):
    verify_start(suite, phases)
    launch = Path(suite['python_output']) / 'launch' / 'recovery_001'
    with original._lane(suite).open('a+b') as lease, _launch_lock(launch):
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', suite['rpc_port']))
        parent.require(not (launch / 'launch_intent.json').exists(), 'existing recovery launch requires inspection')
        fd = lease.fileno()
        command = [PYTHON, '-B', str(Path(__file__).resolve()), '--suite', suite['suite_path'],
                   '--controller', f'--lane-fd={fd}']
        intent = dict(recovery=reference(IDENTITY), suite=reference(suite['suite_path']), command=command)
        with (launch / 'controller.log').open('xb', buffering=0) as log:
            _exclusive_json(launch / 'launch_intent.json', intent)
            child = subprocess.Popen(command, cwd=parent.ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, pass_fds=(fd,), start_new_session=True,
                                     env=dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1'))
        _exclusive_json(launch / 'launch_receipt.json', dict(pid=child.pid, first_run=FIRST,
                        process_start_ticks=_process_identity(child.pid).get('start_ticks'), **intent))
    return dict(status='RECOVERY_CONTROLLER_STARTED', pid=child.pid, first_run=FIRST)


def controller(suite, phases, fd):
    from eval_resume import resume

    output = Path(suite['python_output'])
    parent.require(fd > 2 and os.path.samestat(os.fstat(fd), original._lane(suite).stat()), 'wrong RPC lease')
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with _controller_lock(output):
        intent = parent.read(output / 'launch/recovery_001/launch_intent.json')
        parent.require(intent['recovery'] == reference(IDENTITY) and intent['suite'] == reference(suite['suite_path']),
                       'launch intent differs')
        try:
            verify_start(suite, phases)
            return resume(suite, phases)
        except BaseException as error:
            stop = Path(suite['control_directory']) / 'STOP.properties'
            if not stop.exists():
                reason = repr(error).replace('\n', ' ').replace('\r', ' ')
                _publish(stop, ('reason=' + reason + '\n').encode('ascii', 'backslashreplace'))
            original.runner._status(suite, 'STOP', reason=repr(error))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare', 'check', 'from-anylogic-run', 'controller'):
        modes.add_argument('--' + name, action='store_true')
    parser.add_argument('--suite')
    parser.add_argument('--run-index', type=int)
    parser.add_argument('--environment-seed', type=int)
    parser.add_argument('--recording-id')
    parser.add_argument('--lane-fd', type=int)
    args = parser.parse_args()
    if args.prepare:
        doc = manifest()
        if IDENTITY.exists():
            parent.require(parent.read(IDENTITY) == doc, 'an incompatible recovery manifest exists')
        else:
            write(IDENTITY, doc)
        return reference(IDENTITY)
    suite, phases = original.load_package(UNIT)
    if args.check:
        return dict(first_run=FIRST, counts=verify_start(suite, phases), identity=reference(IDENTITY))
    parent.require(args.suite == locations(UNIT)['suite_path'], 'unexpected suite')
    if args.from_anylogic_run:
        job = next(job for job in suite['jobs'] if job['run_index'] == FIRST)
        parent.require((args.run_index, args.environment_seed, args.recording_id)
                       == (FIRST, job['experiment_seed'], job['recording_id']), 'first Java identity differs')
        return start(suite, phases)
    controller(suite, phases, args.lane_fd)


if __name__ == '__main__':
    result = main()
    if result is not None:
        sys.stdout.write(encoded(result).decode())
