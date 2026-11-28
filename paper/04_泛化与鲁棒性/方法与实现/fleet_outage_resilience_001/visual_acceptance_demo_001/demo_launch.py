"""Start the dedicated frozen-policy RPC controller for one AnyLogic Simulation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

PLAN = Path(__file__).resolve().parent.parent
if str(PLAN) not in sys.path:
    sys.path.insert(0, str(PLAN))

import grpc
import torch

from fleet_policy import FleetLearner
from physical import verify_terminal
from resilience_campaign import reference
from resilience_rpc import FIELD_NAMES, Shift, wire
from resilience_runtime import Handshake, LiveRPC, atomic, dispatch_pb2_grpc, record, require, sha


IDLE_TIMEOUT = 3600.0


def single_job(suite):
    job, = suite['jobs']
    require(job['ordinal'] == 0 and job['role'] == 'eval' and job['split'] == 'visual_demo'
            and job['arm'] == 'dynamic'
            and job['condition'] == 'outage' and job['fault_truck_id'] >= 0,
            'expected one frozen dynamic outage shift')
    return job


class DemoShift(Shift):
    """Keep only the first validated RPC view of each target-vehicle stage."""

    def __init__(self, job, learner, status_file):
        super().__init__(job, learner, verify_terminal)
        self.status_file = status_file
        self.milestones = {}
        self.last_callback = time.monotonic()

    def accept(self, request):
        previous = request.request_id in self.processed
        response = super().accept(request)
        if previous:
            return response
        self.last_callback = time.monotonic()
        self.capture_stage(request)
        if request.kind == wire.EPISODE_END:
            self.receipt['visual_acceptance'] = {
                'fault_truck_id': self.job['fault_truck_id'],
                'scheduled_trigger_seconds': self.job['fault_trigger_seconds'],
                'first_validated_rpc_by_stage': {
                    stage: self.milestones.get(stage) for stage in ('active','retiring','retired')
                },
                'first_observation_is_rpc_sample': True,
                'no_dispatch_to_fault_truck_after_trigger': True,
                'retired_truck_inactive_references_check_passed': 'retired' in self.milestones,
                'terminal_mass_balance_check_passed': True,
            }
        return response

    def capture_stage(self, request):
        truck_id = self.job['fault_truck_id']
        bit = 1 << truck_id
        dispatchable, in_system = self.dispatchable[-1], self.in_system[-1]
        stage = 'active' if dispatchable & bit else 'retiring' if in_system & bit else 'retired'
        if stage in self.milestones:
            return
        state = self.states[-1]
        prefix = f'truck.{truck_id}.'
        start = FIELD_NAMES.index(prefix + 'phase.0')
        phase = next((i for i,value in enumerate(state[start:start+12]) if value), None)
        fields = dict(
            sim_time_seconds=float(request.sim_time_seconds),
            decision_sequence=int(request.decision_sequence),
            terminal=request.kind == wire.EPISODE_END,
            dispatchable_mask=dispatchable,
            in_system_mask=in_system,
            dispatchable_trucks=dispatchable.bit_count(),
            in_system_trucks=in_system.bit_count(),
            target_truck_phase=phase,
            target_truck_loaded=bool(state[FIELD_NAMES.index(prefix+'loaded')]),
            actual_trigger_seconds=float(request.reward_components['fleet.trigger_seconds']),
            actual_exit_seconds=float(request.reward_components['fleet.exit_seconds']),
        )
        self.milestones[stage] = fields
        record(self.status_file, dict(state='RUNNING', run_index=self.job['run_index'],
               truck_stage=stage, first_observation=fields), replace=True)
        print(f"[ACCEPTANCE DEMO] truck {truck_id} {stage} as observed at "
              f"{fields['sim_time_seconds']:.3f} s, dispatchable={fields['dispatchable_trucks']}, "
              f"in_system={fields['in_system_trucks']}", flush=True)


def serve(suite, *, budget=None, address=None):
    if budget is None:
        import demo_budget as budget
    job = single_job(suite)
    out = Path(suite['output'])
    pin = suite['checkpoint']
    require(sha(pin['path']) == pin['sha256'], 'frozen checkpoint bytes differ')
    torch.set_num_threads(2)
    model = FleetLearner.load(Path(pin['path']))
    require(model.update_index == job['checkpoint_update'] == 9, 'expected the sealed dynamic u9')
    require(model.policy_version == suite['actor_digest'], 'frozen Actor digest differs')
    model.actor.eval()
    model.critic.eval()
    master = suite['master_plan']
    require(sha(master['path']) == master['sha256'], 'master plan bytes differ')
    with open(master['path'], encoding='utf-8', newline='') as stream:
        row, = csv.DictReader(stream)
    handshake = Handshake(suite, job, row, budget.claim)
    service = LiveRPC()
    status_file = out/'run_status.json'
    shift = DemoShift(job, model, status_file)
    service.arm(shift, handshake)
    with ThreadPoolExecutor(max_workers=2) as executor:
        server = grpc.server(executor, options=[('grpc.so_reuseport', 0)])
        dispatch_pb2_grpc.add_DispatchServiceServicer_to_server(service, server)
        require(server.add_insecure_port(address or f"127.0.0.1:{suite.get('rpc_port', 50051)}") != 0,
                'unable to bind demonstration RPC port')
        server.start()
        receipt = Path(job['python_output'])/'receipt.json'
        try:
            budget.register(suite)
            record(status_file, dict(state='WAITING_FOR_SHIFT', run_index=job['run_index']), replace=True)
            handshake.publish()
            shift.last_callback = time.monotonic()
            while not service.finished.wait(.5):
                handshake.claim()
                require(not (handshake.control/'STOP.properties').exists(), 'demonstration STOP')
                require(time.monotonic()-shift.last_callback < IDLE_TIMEOUT,
                        'demonstration timed out waiting for a validated RPC')
            with service.lock:
                if service.failure is not None:
                    raise service.failure
                require(handshake.claimed and shift.receipt is not None, 'demonstration END incomplete')
                record(receipt, shift.receipt)
                budget.complete(suite, job, receipt)
                service.shift = None
            result = dict(state='COMPLETE', run_index=job['run_index'], physical_shifts=1,
                          receipt=reference(receipt), terminal=shift.receipt['outage'],
                          visual_acceptance=shift.receipt['visual_acceptance'])
            record(out/'suite_complete.json', result)
            record(status_file, result, replace=True)
        except BaseException as exc:
            if handshake.claimed and not receipt.is_file():
                budget.unknown(suite, job, str(exc))
            raise
        finally:
            server.stop(2).wait()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--run-index', type=int)
    parser.add_argument('--environment-seed', type=int)
    parser.add_argument('--recording-id')
    mode = parser.add_mutually_exclusive_group(required=True)
    for flag in ('--from-anylogic-run', '--controller', '--check', '--status'):
        mode.add_argument(flag, action='store_true')
    args = parser.parse_args(argv)
    from demo_campaign import load_suite

    path = args.suite.resolve(strict=True)
    suite = load_suite(path)
    job = single_job(suite)
    out = Path(suite['output'])
    control = Path(suite['control_directory'])
    status = out/'run_status.json'
    intent = out/'launch_intent.json'
    entry = dict(suite_sha256=sha(path), suite=str(path), campaign_id=suite['campaign_id'],
                 first_run=job['run_index'], first_seed=job['experiment_seed'],
                 recording_id=job['recording_id'], config_sha256=job['config_sha256'])
    if args.check:
        print(json.dumps(dict(state='PREFLIGHT', launch_identity=entry, physical_shifts=1,
                              output_exists=out.exists(), rpc_port=suite.get('rpc_port', 50051)),
                         ensure_ascii=False))
        return
    if args.status:
        print(status.read_text() if status.is_file() else '{"state":"NOT_STARTED"}')
        return

    stop = control/'STOP.properties'
    require(not stop.exists(), f'demonstration stopped, inspect {stop}')
    if args.from_anylogic_run:
        require((args.run_index, args.environment_seed, args.recording_id) ==
                (job['run_index'], job['experiment_seed'], job['recording_id']),
                'AnyLogic startup does not match the frozen demonstration')
        record(intent, entry)
        with (out/'controller.log').open('xb', buffering=0) as log:
            child = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()),
                                      '--suite', str(path), '--controller'], start_new_session=True,
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=log)
        ready = control/'ready'/f"run_{job['run_index']}.properties"
        deadline = time.monotonic()+840
        while not ready.exists():
            require(child.poll() is None and not stop.exists(), f'demonstration failed, inspect {out/"controller.log"}')
            if time.monotonic() >= deadline:
                atomic(stop, b'reason=controller-startup-timeout\n')
                raise TimeoutError('demonstration RPC startup timed out')
            time.sleep(.1)
        return

    require(json.loads(intent.read_bytes()) == entry, 'controller intent differs from frozen demonstration')
    with (out/'controller.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            serve(suite)
        except BaseException as exc:
            if not stop.exists():
                atomic(stop, b'reason=controller-failure\n')
            record(status, dict(state='STOP', run_index=job['run_index'], failure=repr(exc)), replace=True)
            raise


if __name__ == '__main__':
    main()
