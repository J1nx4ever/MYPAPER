"""Replay a six-truck FIFO witness under fixed favorable trip and service factors."""

from dataclasses import astuple, dataclass, replace
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from traditional_dispatch_v1.nominal import _DUMP_OFFSET, _FINISH
from traditional_dispatch_v2.nominal import NominalState, SHIFT_END, WINDOW_SECONDS, _ObservedLoad


CAMPAIGN = 'bg128-full-s02-u32-fleet6-zero-shot10-20261124-001'
ARCHIVE = ROOT/'runs/bg128-continuous-mixed-20261120-s02/evaluation/nominal/new/material/run_10004025.h5'
ACTIVE = (0, 1, 3, 4, 6, 7)
PARKED = (2, 5, 8)
SPEED_FACTOR = 1.08
SERVICE_FACTOR = .92
ITINERARIES = {
    0: '3333333333333333333333322222222200',
    1: '3333333333333333333333020020002010',
    3: '02021202221002211002220',
    4: '220021202022202010220010',
    6: '120120201102011000210',
    7: '0022000020120021220100',
}
SCENE_PREFIX = 'config.runtime_scene.'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _read_ref(ref):
    data = Path(ref['path']).read_bytes()
    require(sha256(data).hexdigest() == ref['sha256'], 'sealed physical receipt changed')
    return json.loads(data)


def _inputs():
    seal = json.loads((ROOT/'runs'/CAMPAIGN/'suite_complete.json').read_bytes())
    report = _read_ref(seal['evaluation'])
    receipt_ref = report['receipts'][0]
    receipt = _read_ref(receipt_ref)
    require(report['status'] == seal['status'] == 'COMPLETE' and receipt['run_index'] == 10042001,
            'expected the completed six-truck reference receipt')
    cfg = receipt['actual_rpc_configuration']
    require(cfg['config.fleet.disabled_mask'] == sum(1 << truck for truck in PARKED)
            and cfg['config.fleet.active_count'] == len(ACTIVE)
            and cfg[SCENE_PREFIX+'env.freeSpeedFactorMin'] <= SPEED_FACTOR <= cfg[SCENE_PREFIX+'env.freeSpeedFactorMax'],
            'parked fleet or travel factor differs from the reference contract')
    source_seal = json.loads(ARCHIVE.with_suffix('.h5.seal.json').read_bytes())
    digest = sha256(ARCHIVE.read_bytes()).hexdigest()
    require(source_seal['hdf5_sha256'] == digest, 'sealed nominal observation archive changed')
    with h5py.File(ARCHIVE) as archive:
        observation = archive['states'][0]
    trucks = observation[27:1107].reshape(9, 120)
    require(observation[0] == 0 and int(np.flatnonzero(trucks[:, 0])[0]) == receipt['initial_state']['truck_id'],
            'nominal archive does not have the expected first dispatch')
    for truck in PARKED:
        static = (1, 2, 3, 16, 41, 42)
        values = trucks[truck, static].copy()
        trucks[truck] = 0
        trucks[truck, static] = values
        trucks[truck, 15] = 1
    trucks[6, 81] = 5 / 9
    return observation, cfg, dict(archive_path=str(ARCHIVE), archive_sha256=digest,
                                  projected_first_observation_sha256=sha256(observation.tobytes()).hexdigest(),
                                  actual_reference_first_observation_sha256=receipt['initial_state']['observation_sha256'],
                                  actual_rpc_receipt=receipt_ref)


def _road_times(observation, cfg, *, loaded, truck_type):
    matrix = np.full((9, 9), np.inf)
    np.fill_diagonal(matrix, 0.0)
    direction = 'loaded' if loaded else 'empty'
    speeds = (20, 18, 16) if loaded else (26, 24, 22)
    for index, row in enumerate(observation[1439:1519].reshape(16, 5)):
        source, target = (int(round(value * 9)) - 1 for value in row[:2])
        prefix = f'{SCENE_PREFIX}road.{index}.'
        require(source == cfg[prefix+'from_node_index'] and target == cfg[prefix+'to_node_index']
                and row[2] == np.float32(cfg[prefix+'length_scaled_f32'])
                and round(float(row[3]) * 10, 5) == round(cfg[prefix+'speedLimit_mps'], 5)
                and not row[4], 'reference road, closure or speed limit changed')
        speed = min(cfg[prefix+'speedLimit_mps'],
                    cfg[prefix+direction+'Base_mps'] * speeds[truck_type] / speeds[0] * SPEED_FACTOR)
        matrix[source, target] = float(row[2]) * 10000 / speed
    for node in range(9):
        matrix = np.minimum(matrix, matrix[:, node, None] + matrix[None, node, :])
    return matrix.tolist()


@dataclass(frozen=True, slots=True)
class _ScaledLoad(_ObservedLoad):
    def service(self, tons):
        return _ObservedLoad.service(self, tons) * SERVICE_FACTOR


class _WitnessState(NominalState):
    def __init__(self, observation, cfg):
        super().__init__(observation)
        trucks = observation[27:1107].reshape(9, 120)
        types = [int(np.argmax(row[1:4])) for row in trucks]
        require(all(cfg[f'{SCENE_PREFIX}truck.{truck}.type'] == truck_type
                    and trucks[truck, 16] == np.float32(cfg[f'{SCENE_PREFIX}truck.{truck}.capacity_t'] / 193.0)
                    for truck, truck_type in enumerate(types)), 'reference truck type or capacity changed')
        require(all(load.ore == bool(cfg[f'{SCENE_PREFIX}load.{site}.isOre'])
                    and load.grade == np.float32(cfg[f'{SCENE_PREFIX}load.{site}.grade'])
                    for site, load in enumerate(self._loads)), 'reference source shape or material grade changed')
        require(self._crusher_capacity == cfg[f'{SCENE_PREFIX}dump.0.crusherBufferCapacityTons']
                and self._crusher_rate * 3600 == cfg[f'{SCENE_PREFIX}dump.0.crusherRateTph']
                and all(self._dump_capacity[dump] == cfg[f'{SCENE_PREFIX}dump.{dump}.serviceCapacity']
                        for dump in range(2)), 'reference warehouse or unloading capacity changed')
        self.empty_times = [_road_times(observation, cfg, loaded=False, truck_type=truck_type) for truck_type in types]
        self.loaded_times = [_road_times(observation, cfg, loaded=True, truck_type=truck_type) for truck_type in types]
        self._loads = tuple(_ScaledLoad(*astuple(load)) for load in self._loads)
        self._trucks = tuple(replace(truck, ore_service=truck.ore_service * SERVICE_FACTOR,
                                     waste_service=truck.waste_service * SERVICE_FACTOR) for truck in self._trucks)
        self.ore_arrivals = []
        self.maximum_warehouse_claimed = 0.0
        require(not self._events and not self._stored and not self._reserved, 'expected empty shift-start state')

    def _future_candidates(self, truck):
        self._empty_times = self.empty_times[truck]
        self._loaded_times = self.loaded_times[truck]
        return super()._future_candidates(truck)

    def apply_action(self, action):
        self._empty_times = self.empty_times[self.current_truck]
        self._loaded_times = self.loaded_times[self.current_truck]
        super().apply_action(action)

    def _process_event(self, kind, resource, truck):
        self._loaded_times = self.loaded_times[truck]
        if kind == _FINISH and resource == _DUMP_OFFSET:
            trip = self._trips[truck]
            require(trip.ore, 'waste cannot enter the FIFO crusher')
            self.ore_arrivals.append((self.now, trip.tons, trip.grade))
        super()._process_event(kind, resource, truck)
        self._check_warehouse()

    def _start_services(self):
        super()._start_services()
        self._check_warehouse()

    def _check_warehouse(self):
        volume = self._stored + self._reserved
        require(0 <= volume <= self._crusher_capacity + 2e-5, 'warehouse capacity exceeded')
        self.maximum_warehouse_claimed = max(self.maximum_warehouse_claimed, volume)


def _check_fifo_independently(arrivals, rate, result):
    ore = [0.0] * 4
    metal = [0.0] * 4
    free_at = 0.0
    for arrival, payload, grade in arrivals:
        start = max(arrival, free_at)
        free_at = start + payload / rate
        for window in range(4):
            seconds = max(0.0, min(free_at, (window + 1) * WINDOW_SECONDS) - max(start, window * WINDOW_SECONDS))
            tons = seconds * rate
            ore[window] += tons
            metal[window] += tons * grade
    for actual, replayed in zip((*result.fifo_q, *result.fifo_m), (*ore, *metal), strict=True):
        require(math.isclose(actual, replayed, rel_tol=0.0, abs_tol=1e-7), 'independent FIFO interval integral differs')


def replay():
    observation, cfg, provenance = _inputs()
    state = _WitnessState(observation, cfg)
    positions = {truck: 0 for truck in ACTIVE}
    timeline = []
    while not state.finished:
        truck = state.current_truck
        require(truck in ACTIVE and positions[truck] < len(ITINERARIES[truck]), 'unexpected request in fixed itinerary')
        site = int(ITINERARIES[truck][positions[truck]])
        require(site in {row.action for row in state.candidates()}, 'fixed itinerary tried an unavailable action')
        timeline.append((state.now, truck, site))
        state.apply_action(site)
        state.advance()
        positions[truck] += 1
    require(all(positions[truck] == len(ITINERARIES[truck]) for truck in ACTIVE), 'itinerary was not fully replayed')
    result = state.score_inputs()
    _check_fifo_independently(state.ore_arrivals, state._crusher_rate, result)
    grades = [mass / tons for tons, mass in zip(result.fifo_q, result.fifo_m, strict=True)]
    require(result.ore_tons >= 16000 and result.waste_tons >= 4000 and all(.26 <= grade <= .28 for grade in grades),
            'fixed favorable witness no longer achieves all task targets')
    half_range = max(cfg[f'{SCENE_PREFIX}load.{site}.gradeHalfRange'] for site in range(4))
    grade_bands = [[grade - half_range, grade + half_range] for grade in grades]
    require(all(.26 <= low <= high <= .28 for low, high in grade_bands),
            'witness grade bands no longer cover the reference geology support')
    return dict(scope='offline_discrete_event_witness', physical_shifts=0,
                assumptions=dict(speed_factor=SPEED_FACTOR, service_factor=SERVICE_FACTOR,
                                 projected_start_from_nine_truck_snapshot=True, nominal_source_grades=True,
                                 acceleration_and_initial_spacing_delay=False, following_and_junction_delay=False,
                                 all_routes_selected_at_simulated_release=True, warehouse_capacity_tons=state._crusher_capacity,
                                 crusher_rate_tons_per_hour=state._crusher_rate * 3600),
                **provenance, shift_seconds=SHIFT_END, active_truck_ids=ACTIVE, parked_truck_ids=PARKED,
                dispatch_counts=positions, timeline_sha256=sha256(json.dumps(timeline).encode()).hexdigest(),
                total_ore_tons=result.ore_tons, total_waste_tons=result.waste_tons, total_wait_seconds=result.wait_seconds,
                maximum_warehouse_stock_and_reserved_tons=state.maximum_warehouse_claimed,
                fifo_integral_independently_checked=True, window_ore_tons=result.fifo_q,
                window_metal_tons=result.fifo_m, window_grade=grades, window_grade_bands_at_fixed_timing=grade_bands)


if __name__ == '__main__':
    print(json.dumps(replay(), indent=2, allow_nan=False))
