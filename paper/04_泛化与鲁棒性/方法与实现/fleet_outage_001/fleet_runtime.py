"""Evaluate the frozen BG128 actor with six active trucks in the nine-slot wire schema."""

from collections import Counter
from math import fsum
from pathlib import Path
from threading import Lock
from types import FunctionType

import numpy as np
import torch

from python.dispatch_bridge import DispatchCallbackBridge, ProtocolError
from python.generated import dispatch_pb2
from traditional_dispatch_v1 import runtime as shared
from traditional_dispatch_v2 import runtime as v2


SUITE_SCHEMA = 'bg128-fleet-outage-v1'
REPORT_SCHEMA = 'bg128-fleet-outage-evaluation-v1'
REPORT_FILENAME = 'fleet_outage_evaluation.json'
EVALUATION_SLOTS = 5
DISABLED = (2, 5, 8)
ACTIVE = (0, 1, 3, 4, 6, 7)
OUTAGE_FIELDS = {'config.fleet.outage_version': 1, 'config.fleet.disabled_mask': 292,
                 'config.fleet.active_count': len(ACTIVE)}
_DISABLED_REF = np.array(DISABLED) + 1
_TYPES = np.eye(3, dtype=np.float32)[np.arange(9) // 3]
_CAPACITY = np.repeat(np.array([91.7, 147., 193.]) / 193., 3).astype(np.float32)


def validate_outage(observation, components, *, truck_id, terminal):
    """Check parked slots and their absence from the active fleet's live references."""
    if {key: components.get(key) for key in OUTAGE_FIELDS} != OUTAGE_FIELDS:
        raise ProtocolError('fleet outage configuration differs from the frozen six-active contract')
    if (observation.shape != (1519,) or observation.dtype != np.float32
            or not np.isfinite(observation).all() or bool(observation[17]) != terminal):
        raise ProtocolError('fleet outage requires a valid float32 1519D lifecycle observation')
    trucks = observation[27:1107].reshape(9, 120)
    parked = trucks[list(DISABLED)]
    if (not np.array_equal(trucks[:, 1:4], _TYPES)
            or not np.allclose(trucks[:, 16], _CAPACITY, rtol=0, atol=1e-7)):
        raise ProtocolError('fleet outage changed the frozen truck types or capacities')
    if (parked[:, 0].any() or parked[:, 4:15].any() or not (parked[:, 15] == 1).all()
            or parked[:, 17:41].any() or parked[:, 43:].any()):
        raise ProtocolError('disabled truck has an ego, active phase, material, node, route, or motion state')
    if not terminal and (truck_id not in ACTIVE or trucks[truck_id, 0] != 1):
        raise ProtocolError('fleet outage dispatch must originate from an active truck')
    loads = observation[1107:1295].reshape(4, 47)
    dumps = observation[1295:1391].reshape(2, 48)
    references = np.concatenate((trucks[:, 81], loads[:, 20:].ravel(), dumps[:, 11:].ravel()))
    if np.isin(np.rint(references * 9), _DISABLED_REF).any():
        raise ProtocolError('disabled truck appears as a predecessor or service, queue, or admission reference')
    return tuple(tuple(float(value) for value in row) for row in parked[:, 41:43])


class FleetOutageBridge(DispatchCallbackBridge):
    """Add the outage contract to the ordinary START, DECISION, and END callbacks."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fleet_lock = Lock()
        self._fleet_requests = set()
        self._parked_coordinates = None
        self._active_dispatches = Counter()
        self._fleet_end = False

    def _validate_request(self, request, *, grade_bounds=True, config=None):
        super()._validate_request(request, grade_bounds=grade_bounds, config=config)
        terminal = request.kind == dispatch_pb2.EPISODE_END
        observation = np.asarray(request.obs.values, dtype=np.float32)
        coords = validate_outage(observation, request.reward_components, truck_id=request.truck_id, terminal=terminal)
        with self._fleet_lock:
            if self._parked_coordinates is None:
                self._parked_coordinates = coords
            if coords != self._parked_coordinates:
                raise ProtocolError('a disabled truck changed its parking coordinates')
            if request.request_id not in self._fleet_requests:
                self._fleet_requests.add(request.request_id)
                if terminal:
                    self._fleet_end = True
                else:
                    self._active_dispatches[request.truck_id] += 1

    def outage_receipt(self, decisions):
        with self._fleet_lock:
            shared._require(self._fleet_end and len(self._fleet_requests) == decisions + 1
                            and sum(self._active_dispatches.values()) == decisions,
                            'outage validation does not cover every accepted callback')
            return dict(disabled_ids=list(DISABLED), active_ids=list(ACTIVE),
                        validated_callbacks=len(self._fleet_requests),
                        active_dispatches={str(truck): self._active_dispatches[truck] for truck in ACTIVE},
                        parked_positions_normalized={str(truck): list(position) for truck, position in
                                                     zip(DISABLED, self._parked_coordinates, strict=True)})


def _collect_with_outage(env, job, method, anchor, actor):
    result = v2._collect(env, job, method, anchor, actor)
    engineering = result['engineering']
    outage = env.bridge.outage_receipt(engineering['decisions'])
    engineering['fleet_outage'] = outage
    acceptance = result['engineering_acceptance']
    acceptance['fleet_outage_validated'] = True
    acceptance['all_six_active_trucks_dispatched'] = all(outage['active_dispatches'].values())
    acceptance['passed'] = all(value for key, value in acceptance.items() if key != 'passed')
    return result


def _run_shift(suite, startup, ledger, job, actor, actor_digest):
    # Scope the two V2 dependencies to this invocation. The rest of its listener,
    # claim, receipt publication, and accounting code runs from the sealed source.
    bindings = dict(vars(v2), DispatchCallbackBridge=FleetOutageBridge, _collect=_collect_with_outage)
    execute = FunctionType(v2._run_shift.__code__, bindings)
    return execute(suite, startup, ledger, job, None, actor, actor_digest)


def _validate_suite(suite):
    jobs = suite['jobs']
    shared._require(suite['schema'] == SUITE_SCHEMA and suite['evaluation_slots'] == EVALUATION_SLOTS
                    and suite['physical_shifts'] == len(jobs) == EVALUATION_SLOTS
                    and suite['training_shifts'] == suite['optimizer_updates'] == 0,
                    'five frozen outage evaluation slots required')
    for field in ('run_index', 'recording_id', 'python_output', 'event_bundle'):
        shared._require(len({job[field] for job in jobs}) == len(jobs), f'physical identity reuses {field}')
    seeds = []
    for slot, job in enumerate(jobs):
        shared._require(job['ordinal'] == job['pair_ordinal'] == slot and job['policy_kind'] == 'bg128'
                        and job['actor_sampling_seed_consumed'] is True and job['checkpoint_update'] == 32
                        and job['actor_seed'] == suite['learner_config']['seed']
                        and all(job['expected_rpc_config'].get(key) == value for key, value in OUTAGE_FIELDS.items()),
                        'outage job does not match the frozen BG128 roster')
        seeds.extend((job['experiment_seed'], job['actor_sampling_seed']))
    shared._require(all(type(seed) is int and seed > 0 for seed in seeds)
                    and len(set(seeds)) == 2 * len(jobs) and suite['learner_config']['seed'] not in seeds,
                    'outage evaluation requires distinct environment and Actor sampling seeds')


def _summary(shifts):
    physical = [row['physical'] for row in shifts]
    engineering = [row['engineering'] for row in shifts]
    return dict(
        complete_tasks=sum(row['complete_task'] for row in physical),
        ore_target_met=sum(row['ore_target_met'] for row in physical),
        waste_target_met=sum(row['waste_target_met'] for row in physical),
        mean={key: fsum(row[key] for row in physical) / len(physical) for key in
              ('combined_cost', 'grade_cost', 'production_cost', 'waiting_cost',
               'ore_tons', 'waste_tons', 'total_wait_seconds')},
        compliant_windows=[sum(row['windows'][window]['compliant'] for row in physical) for window in range(4)],
        engineering_passed=sum(row['engineering_acceptance']['passed'] for row in shifts),
        validated_callbacks=sum(row['fleet_outage']['validated_callbacks'] for row in engineering),
        rpc_responses_over_target=sum(row['rpc_responses_over_target'] for row in engineering),
        dispatches_by_truck={str(truck): sum(row['fleet_outage']['active_dispatches'][str(truck)]
                                           for row in engineering) for truck in ACTIVE},
        max_rpc_response_ms=max(row['rpc_receive_to_response_ms']['max'] for row in engineering),
    )


def run(suite, ledger, *, progress=None):
    """Run the five independent six-active BG128 evaluations through the V2 runtime."""
    _validate_suite(suite)
    control = Path(suite['control_directory'])
    report_path = Path(suite['report_directory']) / REPORT_FILENAME
    slots = ledger.snapshot()['slots']
    jobs = suite['jobs']
    shared._require(not report_path.exists() and not (control / 'STOP.properties').exists()
                    and set(slots) == {str(job['run_index']) for job in jobs}
                    and all(slots[str(job['run_index'])]['status'] == 'PLANNED'
                            and slots[str(job['run_index'])]['identity'] == job for job in jobs),
                    'outage evaluation requires the exact fresh ledger roster')
    shared._check_startup(suite)
    torch.set_num_threads(suite.get('cpu_threads', 2))
    actor, actor_digest = shared._load_actor(suite)
    startup = dict(control_directory=str(control), campaign_id=suite['campaign_id'],
                   master_plan_sha256=suite['master_plan']['sha256'])
    shifts, refs = [], []
    for job in jobs:
        shift, ref = _run_shift(suite, startup, ledger, job, actor, actor_digest)
        shifts.append(shift)
        refs.append(ref)
        print(f"COMPLETE run_index={job['run_index']} slot={job['pair_ordinal']} policy=bg128", flush=True)
        if progress is not None:
            progress(len(shifts), job)
    slots = ledger.snapshot()['slots']
    shared._require(all(slots[str(job['run_index'])]['status'] == 'COMPLETE'
                        and slots[str(job['run_index'])]['receipt'] == ref
                        for job, ref in zip(jobs, refs, strict=True)), 'outage ledger lacks complete physical receipts')
    report = dict(schema=REPORT_SCHEMA, status='COMPLETE', campaign_id=suite['campaign_id'],
                  evaluation_slots=len(shifts), completed_shifts=len(shifts), physical_shifts=len(shifts),
                  optimizer_updates=0, method=suite['method'], outage_configuration=OUTAGE_FIELDS,
                  shifts=shifts, receipts=refs, summary=_summary(shifts))
    return shared._publish_json(report_path, report)
