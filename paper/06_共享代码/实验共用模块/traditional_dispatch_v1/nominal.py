"""Deterministic shift projection from a single decision-time observation."""
from __future__ import annotations

from collections import deque
from copy import copy
from dataclasses import dataclass
from heapq import heappop, heappush
import math

import numpy as np

from grade_commitment_ppo_v1 import features as legacy
from physical_realization_v1.observation import _remaining_service
from single_critic_task_query_v1.c_features import private_c_state


WINDOW_SECONDS = 7200.0
SHIFT_END = 4 * WINDOW_SECONDS
WASTE_STOP_TONS = 6000.0
MASS_EPS = 1e-6
MASS_TOL = .002  # Observation is float32 scaled in physical units.
_FINISH, _ARRIVE = range(2)
_DUMP_OFFSET = 4
_CFG = legacy._config(None)


class NominalModelError(ValueError):
    """Key inputs cannot support a nominal decision."""


@dataclass(frozen=True, slots=True)
class DispatchCandidate:
    action: int
    tons: float
    ore: bool
    grade: float


@dataclass(frozen=True, slots=True)
class CandidateTiming:
    wait_seconds: float
    unload_complete: float


@dataclass(frozen=True, slots=True)
class ScoreInputs:
    fifo_q: tuple[float, ...]
    fifo_m: tuple[float, ...]
    unload_q: tuple[float, ...]
    unload_m: tuple[float, ...]
    ore_tons: float
    waste_tons: float
    wait_seconds: float
    first_window: int
    root_cross_window: bool = False


@dataclass(frozen=True, slots=True)
class _Load:
    node: int
    available: bool
    ore: bool
    grade: float
    capacity: int
    bucket_tons: float
    bucket_minutes: float

    def service(self, tons):
        bucket_count = math.ceil((tons - 1e-9) / self.bucket_tons)
        return 60.0 * (_CFG['load_setup_minutes'] + self.bucket_minutes * bucket_count)


@dataclass(frozen=True, slots=True)
class _Truck:
    capacity: float
    ore_service: float
    waste_service: float


@dataclass(frozen=True, slots=True)
class _Trip:
    tons: float
    ore: bool
    grade: float
    root: bool = False


@dataclass(frozen=True, slots=True)
class _Packet:
    tons: float
    grade: float
    root: bool = False


def _require(condition, reason):
    if not condition:
        raise NominalModelError(reason)


def _near(left, right):
    return math.isclose(left, right, abs_tol=MASS_TOL, rel_tol=2e-6)


def _count(raw, name, *, positive=False):
    value = round(raw)
    _require(abs(raw - value) <= 2e-6 and value >= int(positive), name)
    return value


def _service_residual(mean, age, deterministic):
    if deterministic and mean > 0 and abs(mean - age) <= .001:
        return 0.0
    duration, reason = _remaining_service(mean, age, deterministic=deterministic)
    _require(duration is not None, reason)
    return float(duration)


def _owners(raw, start, stop):
    ids = legacy._ids(raw[start:stop])
    _require(len(ids) == len(set(ids)), 'duplicate_resource_owner')
    return ids


class NominalState:
    """Mutable branch of the causal forecast. Clone before applying a different action."""

    @classmethod
    def from_observation(cls, observation):
        """Construct the current decision and all visible dispatched commitments."""
        try:
            snapshot = legacy._state(private_c_state(observation))
            return cls(snapshot)
        except ValueError as error:
            if isinstance(error, NominalModelError):
                raise
            raise NominalModelError(str(error)) from error

    def __init__(self, snapshot):
        self._first_window = int(round(snapshot[2] * 4))
        _require(not snapshot[17] and 0 <= self._first_window < 4,
                 'terminal_observation_is_not_actionable')
        stop = (self._first_window + 1) * WINDOW_SECONDS
        time_from_window = stop - float(snapshot[3]) * WINDOW_SECONDS
        _require(abs(time_from_window - float(snapshot[0]) * SHIFT_END) <= .005,
                 'snapshot_clocks_disagree')
        self.now = min(time_from_window, math.nextafter(stop, -math.inf))
        self._ore_at_observation = float(snapshot[7]) * 20000.0
        self._waste_delivered = float(snapshot[9]) * 4000.0
        self._future_ore = 0.0
        self._wait_seconds = 0.0
        self._fifo_q = [0.0] * 4
        self._fifo_m = [0.0] * 4
        self._fifo_q[self._first_window] = float(snapshot[4]) * 5000.0
        self._fifo_m[self._first_window] = float(snapshot[5]) * 5000.0
        _require(self._fifo_m[self._first_window] <= self._fifo_q[self._first_window] + MASS_TOL
                 and self._fifo_q[self._first_window] <= self._ore_at_observation + MASS_TOL,
                 'snapshot_processed_mass_inconsistent')
        packets = legacy._fifo(snapshot, _CFG)
        self._fifo = deque(_Packet(q, grade) for q, grade, *_ in packets)
        self._stored = sum(packet.tons for packet in self._fifo)
        self._unload_q = self._fifo_q.copy()
        self._unload_m = self._fifo_m.copy()
        self._unload_q[self._first_window] += self._stored
        self._unload_m[self._first_window] += sum(packet.tons * packet.grade for packet in self._fifo)
        self._root_unload_window = None
        self._root_cross_window = False
        self._timing_truck = None
        self._timing_wait = 0.0
        self._timing_complete = math.inf

        trucks = snapshot[27:1107].reshape(9, 120)
        loads = snapshot[1107:1295].reshape(4, 47)
        dumps = snapshot[1295:1391].reshape(2, 48)
        self._empty_times = legacy._road_times(snapshot, _CFG, loaded=False).tolist()
        self._loaded_times = legacy._road_times(snapshot, _CFG, loaded=True).tolist()
        self._loads = tuple(
            _Load(node, bool(row[0]), bool(row[1]), float(row[2]),
                  _count(row[14], 'invalid_shovel_capacity', positive=True),
                  float(row[18]) * 30.0, float(row[19]))
            for node, row in zip(_CFG['load_node_indices'], loads, strict=True)
        )
        _require(all(spec.bucket_tons > 0 and spec.bucket_minutes >= 0 and 0 <= spec.grade <= 1
                     for spec in self._loads), 'invalid_load_service_or_grade')
        self._available_tons = [float(row[7]) * 30000.0 for row in loads]
        _require(all(tons >= 0 for tons in self._available_tons), 'negative_available_supply')
        self._trucks = tuple(
            _Truck(float(row[16]) * 193.0, legacy._dump_mean(row, _CFG, ore=True),
                   legacy._dump_mean(row, _CFG, ore=False)) for row in trucks
        )
        _require(all(truck.capacity > 0 and math.isfinite(truck.ore_service)
                     and math.isfinite(truck.waste_service)
                     for truck in self._trucks), 'invalid_truck_type_or_capacity')
        self._nodes = [legacy._onehot_index(row[32:41]) for row in trucks]
        _require(bool(dumps[0, 1]) and not bool(dumps[1, 1]), 'dump_commodity_mapping_invalid')
        self._dump_available = tuple(bool(row[0]) for row in dumps)
        self._dump_capacity = tuple(_count(row[2] * 3.0, 'invalid_dump_capacity', positive=True)
                                    for row in dumps)
        self._crusher_capacity = float(dumps[0, 8]) * 500.0
        self._crusher_rate = float(dumps[0, 9]) * (2500.0 / 3600.0)
        self._reserved = float(dumps[0, 6]) * 500.0
        _require(self._crusher_capacity > 0 and self._stored + self._reserved <=
                 self._crusher_capacity + MASS_TOL
                 and _near(self._stored + self._reserved + float(dumps[0, 7]) * 500.0,
                           self._crusher_capacity), 'crusher_capacity_inconsistent')

        self._load_wait = [deque() for _ in loads]
        self._dump_wait = [deque() for _ in dumps]
        self._capacity_wait = deque()
        self._load_active = [0] * len(loads)
        self._dump_active = [0] * len(dumps)
        self._events = []
        self._ready = []
        self._first_dispatch_wait = {}
        self._initial_dispatched = {tid for tid, row in enumerate(trucks) if row[80]}
        self._trips = [None] * len(trucks)
        self.current_truck = int(np.flatnonzero(trucks[:, 0])[0])
        self._is_root_decision = True
        self._candidates = tuple(
            self._candidate(action, float(snapshot[23 + action]) * 193.0)
            for action in range(len(loads)) if snapshot[19 + action]
        )
        _require(all(c.tons > MASS_EPS for c in self._candidates), 'legal_candidate_without_load')
        _require(self._nodes[self.current_truck] is not None, 'current_truck_node_unknown')
        self._seed_commitments(trucks, loads, dumps, bool(snapshot[16]))

    @property
    def finished(self):
        return self.now >= SHIFT_END

    @property
    def window_index(self):
        return min(3, int(self.now // WINDOW_SECONDS))

    def candidates(self):
        """Eligible actions at the current dispatch. Root actions use the actual mask and tons."""
        return self._candidates

    def clone(self):
        other = copy(self)
        for name in ('_fifo', '_fifo_q', '_fifo_m', '_unload_q', '_unload_m', '_available_tons',
                     '_nodes', '_capacity_wait', '_load_active', '_dump_active', '_events',
                     '_ready', '_first_dispatch_wait', '_initial_dispatched', '_trips'):
            setattr(other, name, getattr(self, name).copy())
        other._load_wait = [queue.copy() for queue in self._load_wait]
        other._dump_wait = [queue.copy() for queue in self._dump_wait]
        return other

    def _push(self, timestamp, kind, resource, truck):
        _require(math.isfinite(timestamp) and timestamp >= self.now, 'invalid_nominal_event_time')
        if timestamp < SHIFT_END:
            heappush(self._events, (float(timestamp), kind, resource, truck))

    def _seed_commitments(self, trucks, loads, dumps, deterministic):
        load_owner = {}
        dump_owner = {}
        dump_reservations = 0.0
        for aid, row in enumerate(loads):
            queued = tuple(dict.fromkeys(_owners(row, 20, 29) + _owners(row, 29, 38)))
            busy = _owners(row, 38, 47)
            _require(len(busy) == _count(row[15], 'invalid_shovel_occupancy')
                     and len(busy) <= self._loads[aid].capacity, 'shovel_occupancy_mismatch')
            self._load_wait[aid].extend(queued)
            self._load_active[aid] = len(busy)
            for phase, members in ((3, queued), (4, busy)):
                for tid in members:
                    _require(tid not in load_owner, 'truck_holds_multiple_load_stages')
                    load_owner[tid] = (phase, aid)

        for did, row in enumerate(dumps):
            _require(not row[10] and not row[11], 'dump_admission_reservation_owner_ambiguous')
            queued = tuple(dict.fromkeys(_owners(row, 12, 21) + _owners(row, 21, 30)))
            waiting = _owners(row, 30, 39)
            busy = _owners(row, 39, 48)
            _require(not did or not waiting, 'waste_dump_has_capacity_wait')
            _require(len(busy) + len(waiting) == _count(row[3] * 3.0, 'invalid_dump_occupancy')
                     and len(busy) + len(waiting) <= self._dump_capacity[did],
                     'dump_occupancy_mismatch')
            self._dump_wait[did].extend(queued)
            self._capacity_wait.extend(waiting)
            self._dump_active[did] = len(busy)
            for phase, members in ((6, queued), (7, waiting), (9, busy)):
                for tid in members:
                    _require(tid not in dump_owner, 'truck_holds_multiple_dump_stages')
                    dump_owner[tid] = (phase, did)

        for tid, row in enumerate(trucks):
            phase = legacy._onehot_index(row[4:16])
            loaded = bool(row[17])
            ore = bool(row[18])
            payload, reserved = (float(row[index]) * 193.0 for index in (19, 20))
            _require(phase is not None and min(payload, reserved) >= 0,
                     'invalid_truck_phase_or_mass')
            if tid in load_owner:
                _require(load_owner[tid][0] == phase, 'shovel_owner_phase_mismatch')
            if tid in dump_owner:
                _require(dump_owner[tid][0] == phase, 'dump_owner_phase_mismatch')
            if phase in (0, 1, 10, 11):
                _require(not loaded and max(payload, reserved) <= MASS_EPS,
                         'committed_truck_phase_unmodeled')
                if tid != self.current_truck and phase == 0:
                    heappush(self._ready, tid)
                elif phase == 1:
                    self._first_dispatch_wait[tid] = round(row[81] * 9.0) - 1
                continue
            _require(tid != self.current_truck, 'current_truck_already_dispatched')
            if phase in (2, 3, 4):
                aid = legacy._onehot_index(row[26:30])
                _require(not loaded and reserved > MASS_EPS and payload <= MASS_EPS
                         and aid is not None and ore == self._loads[aid].ore,
                         'load_commitment_inconsistent')
                grade = float(row[23]) if phase == 4 else self._loads[aid].grade
                _require(not ore or 0 <= grade <= 1, 'invalid_known_loading_grade')
                self._trips[tid] = _Trip(reserved, ore, grade if ore else 0.0)
                if phase == 2:
                    duration = legacy._moving_seconds(row, _CFG, loaded=False)
                    self._push(self.now + duration, _ARRIVE, aid, tid)
                else:
                    _require(load_owner.get(tid) == (phase, aid), 'load_commitment_owner_missing')
                    if phase == 4:
                        duration = _service_residual(row[85] * 600.0, row[84] * 600.0, deterministic)
                        self._push(self.now + duration, _FINISH, aid, tid)
                continue
            did = int(not ore)
            grade = float(row[22]) if ore else 0.0
            _require(loaded and payload > MASS_EPS and reserved <= MASS_EPS
                     and (not ore or (row[21] and 0 <= grade <= 1)), 'payload_commitment_inconsistent')
            self._trips[tid] = _Trip(payload, ore, grade)
            if phase == 5:
                duration = legacy._moving_seconds(row, _CFG, loaded=True)
                self._push(self.now + duration, _ARRIVE, _DUMP_OFFSET + did, tid)
            else:
                _require(dump_owner.get(tid) == (phase, did), 'dump_commitment_owner_missing')
                if phase == 9:
                    duration = _service_residual(row[88] * 600.0, row[87] * 600.0, deterministic)
                    self._push(self.now + duration, _FINISH, _DUMP_OFFSET + did, tid)
                    if ore:
                        dump_reservations += payload
        _require(_near(self._reserved, dump_reservations), 'buffer_reservation_owner_mismatch')
        self._reserved = dump_reservations

    def _candidate(self, action, tons):
        load = self._loads[action]
        return DispatchCandidate(action, tons, load.ore, load.grade if load.ore else 0.0)

    def _future_candidates(self, tid):
        node = self._nodes[tid]
        _require(node is not None, 'dispatch_truck_node_unknown')
        waste_open = self._waste_committed() < WASTE_STOP_TONS - 1e-6
        candidates = []
        for action, load in enumerate(self._loads):
            did = int(not load.ore)
            tons = min(self._trucks[tid].capacity, self._available_tons[action])
            if (tons > MASS_EPS and load.available and self._dump_available[did]
                    and (load.ore or waste_open)
                    and math.isfinite(self._empty_times[node][load.node])
                    and math.isfinite(self._loaded_times[load.node][did])):
                candidates.append(self._candidate(action, tons))
        return tuple(candidates)

    def _release_initial(self):
        # The predecessor relation is visible. The virtual release assumes no
        # additional headway; real startup order is controlled by the live RPCs.
        released = [tid for tid, predecessor in self._first_dispatch_wait.items()
                    if predecessor == -1 or predecessor in self._initial_dispatched]
        for tid in released:
            del self._first_dispatch_wait[tid]
            heappush(self._ready, tid)

    def apply_action(self, action):
        """Commit the current truck's action. Call advance() for the next decision."""
        candidate = next((c for c in self._candidates if c.action == action), None)
        _require(self.current_truck is not None and candidate is not None,
                 'action_outside_nominal_candidate_set')
        tid = self.current_truck
        load = self._loads[candidate.action]
        node = self._nodes[tid]
        did = int(not candidate.ore)
        arrival = self.now + self._empty_times[node][load.node]
        _require(math.isfinite(arrival) and math.isfinite(self._loaded_times[load.node][did])
                 and self._dump_available[did], 'candidate_route_or_dump_unavailable')
        _require(self._trips[tid] is None
                 and candidate.tons <= self._available_tons[candidate.action] + MASS_TOL,
                 'candidate_reservation_inconsistent')
        remaining = self._available_tons[candidate.action] - candidate.tons
        # Current candidate tons and the stock are independently normalized in
        # the observation. Do not turn their rounding difference into a trip.
        self._available_tons[candidate.action] = remaining if remaining > MASS_TOL else 0.0
        self._trips[tid] = _Trip(candidate.tons, candidate.ore, candidate.grade, self._is_root_decision)
        self._push(arrival, _ARRIVE, candidate.action, tid)
        self._initial_dispatched.add(tid)
        self._is_root_decision = False
        self.current_truck = None
        self._candidates = ()

    def _start_services(self):
        for aid, queue in enumerate(self._load_wait):
            load = self._loads[aid]
            while queue and self._load_active[aid] < load.capacity:
                tid = queue.popleft()
                self._load_active[aid] += 1
                duration = load.service(self._trips[tid].tons)
                self._push(self.now + duration, _FINISH, aid, tid)
        while (self._dump_wait[0]
               and self._dump_active[0] + len(self._capacity_wait) < self._dump_capacity[0]):
            self._capacity_wait.append(self._dump_wait[0].popleft())
        while self._capacity_wait:
            tid = self._capacity_wait[0]
            tons = self._trips[tid].tons
            if self._stored + self._reserved + tons > self._crusher_capacity + MASS_EPS:
                break
            self._capacity_wait.popleft()
            self._dump_active[0] += 1
            self._reserved += tons
            self._push(self.now + self._trucks[tid].ore_service, _FINISH, _DUMP_OFFSET, tid)
        while self._dump_wait[1] and self._dump_active[1] < self._dump_capacity[1]:
            tid = self._dump_wait[1].popleft()
            self._dump_active[1] += 1
            self._push(self.now + self._trucks[tid].waste_service, _FINISH, _DUMP_OFFSET + 1, tid)

    def _capacity_retry(self):
        if not self._capacity_wait or self._crusher_rate <= 0:
            return math.inf
        tons = self._trips[self._capacity_wait[0]].tons
        shortage = self._stored + self._reserved + tons - self._crusher_capacity
        if shortage > self._stored + MASS_EPS:
            return math.inf  # Reserved material must first finish unloading.
        return self.now + max(0.0, shortage) / self._crusher_rate

    def _advance_time(self, until):
        waiting = (*self._load_wait, *self._dump_wait, self._capacity_wait)
        elapsed = until - self.now
        if self._timing_truck is not None:
            if any(self._timing_truck in queue for queue in waiting):
                self._timing_wait += elapsed
            # The timing clone needs the total stock for capacity admission.
            # Packet grades and window splits cannot affect service timing.
            self._stored = max(0.0, self._stored - elapsed * self._crusher_rate)
            self.now = until
            return
        self._wait_seconds += elapsed * sum(map(len, waiting))
        while self.now < until:
            window = self.window_index
            stop = min(until, (window + 1) * WINDOW_SECONDS)
            budget = (stop - self.now) * self._crusher_rate
            while budget > 0.0 and self._fifo:
                packet = self._fifo[0]
                tons = min(budget, packet.tons)
                self._fifo_q[window] += tons
                self._fifo_m[window] += tons * packet.grade
                self._future_ore += tons
                self._stored -= tons
                budget -= tons
                if packet.root and window > self._root_unload_window:
                    self._root_cross_window = True
                if tons == packet.tons:
                    self._fifo.popleft()
                else:
                    self._fifo[0] = _Packet(packet.tons - tons, packet.grade, packet.root)
            if not self._fifo:
                self._stored = 0.0
            self.now = stop

    def _process_event(self, kind, resource, tid):
        if kind == _ARRIVE:
            if resource < _DUMP_OFFSET:
                self._load_wait[resource].append(tid)
            else:
                self._dump_wait[resource - _DUMP_OFFSET].append(tid)
            return
        trip = self._trips[tid]
        if resource < _DUMP_OFFSET:
            self._load_active[resource] -= 1
            did = int(not trip.ore)
            node = self._loads[resource].node
            self._push(self.now + self._loaded_times[node][did], _ARRIVE, _DUMP_OFFSET + did, tid)
            return
        did = resource - _DUMP_OFFSET
        self._dump_active[did] -= 1
        if trip.ore:
            self._reserved = max(0.0, self._reserved - trip.tons)
            self._stored += trip.tons
            if self._timing_truck is None:
                self._fifo.append(_Packet(trip.tons, trip.grade, trip.root))
                self._unload_q[self.window_index] += trip.tons
                self._unload_m[self.window_index] += trip.tons * trip.grade
                if trip.root:
                    self._root_unload_window = self.window_index
        else:
            self._waste_delivered += trip.tons
        if tid == self._timing_truck:
            self._timing_complete = self.now
        self._trips[tid] = None
        self._nodes[tid] = did
        heappush(self._ready, tid)

    def _move(self, *, check_deadline, dispatch):
        while not self.finished:
            if check_deadline is not None:
                check_deadline()
            while self._events and self._events[0][0] == self.now:
                _, kind, resource, tid = heappop(self._events)
                self._process_event(kind, resource, tid)
            if not dispatch and math.isfinite(self._timing_complete):
                return
            self._start_services()
            if dispatch:
                self._release_initial()
                while self._ready:
                    tid = heappop(self._ready)
                    options = self._future_candidates(tid)
                    if options:
                        self.current_truck, self._candidates = tid, options
                        return
                    # No feasible site means this virtual truck parks.
                    self._initial_dispatched.add(tid)
                    self._release_initial()
            next_event = self._events[0][0] if self._events else math.inf
            next_time = min(next_event, self._capacity_retry(), SHIFT_END)
            _require(next_time > self.now, 'nominal_event_clock_stalled')
            self._advance_time(next_time)

    def advance(self, *, check_deadline=None):
        """Advance resource work and FIFO to the next legal dispatch or the end of shift."""
        if self.current_truck is None:
            self._move(check_deadline=check_deadline, dispatch=True)

    def candidate_timing(self, action, *, check_deadline=None):
        """Insert one candidate and forecast its queue wait and unload completion."""
        branch = self.clone()
        branch._timing_truck = self.current_truck
        branch.apply_action(action)
        branch._move(check_deadline=check_deadline, dispatch=False)
        return CandidateTiming(branch._timing_wait, branch._timing_complete)

    def score_inputs(self):
        return ScoreInputs(tuple(self._fifo_q), tuple(self._fifo_m), tuple(self._unload_q),
                           tuple(self._unload_m), self._ore_at_observation + self._future_ore,
                           self._waste_delivered, self._wait_seconds, self._first_window,
                           self._root_cross_window)

    def _waste_committed(self):
        return self._waste_delivered + sum(trip.tons for trip in self._trips if trip and not trip.ore)

    def tail_context(self):
        """Processed plus FIFO Q/M, and shift ore/waste totals with visible commitments."""
        window = self.window_index
        return (self._fifo_q[window] + self._stored,
                self._fifo_m[window] + sum(packet.tons * packet.grade for packet in self._fifo),
                self._ore_at_observation + self._future_ore + self._stored
                + sum(trip.tons for trip in self._trips if trip and trip.ore),
                self._waste_committed())
