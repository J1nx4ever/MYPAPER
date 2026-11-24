"""Causal route-flow LP for the load-site-only BG128 dispatch interface."""
from __future__ import annotations

from dataclasses import dataclass
import math
from time import perf_counter

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix, vstack

from single_critic_task_query_v1.targets import ORE_TARGET_TONS, WASTE_TARGET_TONS
from traditional_dispatch_v2.flow_plan import _CommitmentReplay
from traditional_dispatch_v2.nominal import (
    MASS_EPS, MASS_TOL, SHIFT_END, WASTE_STOP_TONS, WINDOW_SECONDS, NominalModelError,
)


NEAR_SECONDS = 1800
NEAR_SLOT_SECONDS = 300
FAR_SLOT_SECONDS = 900
CONSTRAINT_CHECK_SECONDS = 150
SUBSEGMENTS = 3
GRADE_MIN = .26
GRADE_MAX = .28
RouteKey = tuple[int, int, int, int]


class FlowPlanError(RuntimeError):
    """A new flow plan could not be published within the planner contract."""


@dataclass(frozen=True, slots=True)
class FlowSlot:
    start: float
    stop: float
    tons: float


@dataclass(frozen=True, slots=True)
class WindowProjection:
    window: int
    ore_tons: float
    metal_tons: float

    @property
    def grade(self):
        return self.metal_tons / self.ore_tons if self.ore_tons > MASS_EPS else None


@dataclass(frozen=True, slots=True)
class FlowPlan:
    now: float
    routes: dict[RouteKey, tuple[FlowSlot, ...]]
    committed_windows: tuple[WindowProjection, ...]
    grade_windows: tuple[WindowProjection, ...]
    ore_tons: float
    waste_tons: float
    ore_shortfall_tons: float
    waste_shortfall_tons: float
    grade_slack_tons: tuple[tuple[float, float], ...]
    feed_slack_tons: tuple[float, ...]
    status: str
    residual_max: float
    _clock: np.ndarray
    _ore_curve: np.ndarray
    _start_window_ore: float
    _guard_columns: dict[RouteKey, tuple[tuple[float, float, int], ...]]
    _guard_matrix: csr_matrix
    _guard_residual: np.ndarray

    def expected_window_ore_at(self, t: float) -> float:
        """Cumulative nominal crushed ore for the window containing t."""
        t = min(max(t, self.now), SHIFT_END)
        window_start = min(3, int(t // WINDOW_SECONDS)) * WINDOW_SECONDS
        start_ore = self._start_window_ore if window_start <= self.now else 0.
        return float(start_ore + np.interp(t, self._clock, self._ore_curve)
                     - np.interp(max(window_start, self.now), self._clock, self._ore_curve))

    def accepts_extra(self, route, t, tons, prior=()):
        """Test excess assignments on the fixed physical/quality coefficients."""
        added = np.zeros(self._guard_matrix.shape[1])
        for key, time, mass in (*prior, (route, t, tons)):
            if mass <= MASS_EPS:
                continue
            slots = self._guard_columns.get(key, ())
            column = next((i for begin, end, i in slots if begin <= time < end), None)
            if column is None:
                return False
            added[column] += mass
        return bool(np.all(self._guard_matrix @ added <= self._guard_residual + MASS_TOL))


class _Replay(_CommitmentReplay):
    def __init__(self, snapshot):
        self.timeline = []
        super().__init__(snapshot)

    def _advance_time(self, until):
        self.timeline.append((self.now, self._stored, self._reserved, self._future_ore))
        super()._advance_time(until)


class _Warehouse:
    def __init__(self, replay):
        self.times, self.stock, self.reserved, self.processed = np.asarray(replay.timeline).T
        self.capacity = replay._crusher_capacity
        self.rate = replay._crusher_rate

    def balances(self, clock):
        i = np.maximum(0, np.searchsorted(self.times, clock, side='right') - 1)
        crushed = np.minimum(self.stock[i], (clock - self.times[i]) * self.rate)
        return self.stock[i] - crushed, self.reserved[i], self.processed[i] + crushed

    def available_at(self, t, tons):
        """First committed-baseline time when a full ore payload fits."""
        i = max(0, int(np.searchsorted(self.times, t, side='right')) - 1)
        for index in range(i, len(self.times)):
            available = self.capacity - self.reserved[index] - tons
            if available < -MASS_TOL:
                continue
            start = self.times[index]
            next_time = self.times[index + 1] if index + 1 < len(self.times) else math.inf
            if self.rate <= 0 and self.stock[index] > available + MASS_TOL:
                continue
            delay = max(0., self.stock[index] - available) / self.rate if self.rate > 0 else 0.
            at = max(start + delay, t)
            if at < next_time:
                return at
        return math.inf


def _blocked(intervals, capacity):
    """Intervals in which all positions are already committed."""
    deltas = {}
    for start, stop in intervals:
        if stop > start:
            deltas[start] = deltas.get(start, 0) + 1
            deltas[stop] = deltas.get(stop, 0) - 1
    occupied, blocked, start = 0, [], None
    for t, change in sorted(deltas.items()):
        updated = occupied + change
        if occupied < capacity <= updated:
            start = t
        elif updated < capacity <= occupied:
            blocked.append((start, t))
        occupied = updated
    return tuple(blocked)


def _start_after(blocked, arrival, duration):
    for start, stop in blocked:
        if arrival + duration <= start:
            break
        arrival = max(arrival, stop)
    return arrival


@dataclass(frozen=True, slots=True)
class _Column:
    key: RouteKey
    depart: tuple[float, float]
    q: float
    grade: float
    offsets: tuple[tuple[float, ...], ...]


def _offsets(depart, empty, loaded, load_time, unload_time, load_queue,
             dump_queue, dump, warehouse, q):
    load_start = _start_after(load_queue, depart + empty, load_time)
    load_end = load_start + load_time
    arrival = load_end + loaded
    berth = _start_after(dump_queue, arrival, unload_time)
    if dump == 0:
        reserve = warehouse.available_at(berth, q)
        if not math.isfinite(reserve):
            return None
        # A truck holds its berth during the warehouse wait as well as unloading.
        while True:
            delayed = _start_after(dump_queue, berth, reserve - berth + unload_time)
            if delayed == berth:
                break
            berth = delayed
            reserve = warehouse.available_at(berth, q)
            if not math.isfinite(reserve):
                return None
        unload = reserve + unload_time
        if warehouse.rate > 0:
            stock, _, _ = warehouse.balances(unload)
            crush_start = unload + stock / warehouse.rate
            crush_end = crush_start + q / warehouse.rate
        else:
            crush_start, crush_end = SHIFT_END + 1, SHIFT_END + 2
    else:
        reserve = berth
        unload = berth + unload_time
        crush_start, crush_end = SHIFT_END + 1, SHIFT_END + 2
    return tuple(float(event - depart) for event in
                 (load_start, load_end, berth, reserve, unload, crush_start, crush_end))


def _require_time(deadline):
    if perf_counter() >= deadline:
        raise FlowPlanError('flow_plan_deadline')


def _slots(now):
    near = min(now + NEAR_SECONDS, SHIFT_END)
    edges = {*np.arange(now, near, NEAR_SLOT_SECONDS), *np.arange(near, SHIFT_END, FAR_SLOT_SECONDS),
             *[w * WINDOW_SECONDS for w in range(1, 4) if w * WINDOW_SECONDS > now], SHIFT_END}
    return tuple(zip(sorted(edges)[:-1], sorted(edges)[1:]))


def _columns(replay, truck_types, warehouse, ready, slots, deadline):
    calendars = [_blocked(intervals, cap) for intervals, cap in zip(
        replay.resource_intervals, [load.capacity for load in replay._loads] + list(replay._dump_capacity),
        strict=True)]
    waste_open = replay.waste_committed_at_origin < WASTE_STOP_TONS - MASS_EPS
    for truck_type in sorted(set(truck_types)):
        truck = replay._trucks[truck_types.index(truck_type)]
        origins = {node for tid, node, _ in ready if truck_types[tid] == truck_type}
        if not origins:
            continue
        origins.update(range(2))
        for origin in sorted(origins):
            for site, load in enumerate(replay._loads):
                _require_time(deadline)
                dump = int(not load.ore)
                supply = replay._available_tons[site]
                empty = replay._empty_times[origin][load.node]
                loaded = replay._loaded_times[load.node][dump]
                if (not load.available or not replay._dump_available[dump] or supply <= MASS_EPS
                        or not math.isfinite(empty + loaded) or (dump == 1 and not waste_open)):
                    continue
                q = min(truck.capacity, supply)
                load_time = load.service(q)
                unload_time = truck.ore_service if dump == 0 else truck.waste_service
                for start, stop in slots:
                    if start + empty + load_time + loaded + unload_time >= SHIFT_END:
                        break
                    offsets = tuple(_offsets(
                        start + (stop - start) * (i + .5) / SUBSEGMENTS, empty, loaded,
                        load_time, unload_time, calendars[site], calendars[4 + dump], dump, warehouse, q,
                    ) for i in range(SUBSEGMENTS))
                    if all(offset is not None for offset in offsets):
                        yield _Column((truck_type, origin, site, dump), (start, stop), q,
                                      load.grade if dump == 0 else 0., offsets)


def _curves(columns, clock, warehouse):
    """Evaluate the fixed kernels and occupied seconds on the LP time grid."""
    count = len(columns)
    periods = np.asarray([column.depart for column in columns], dtype=float).reshape(count, 2)
    offsets = np.asarray([column.offsets for column in columns], dtype=float).reshape(
        count, SUBSEGMENTS, 7)
    width = (periods[:, 1] - periods[:, 0])[None, :, None] / SUBSEGMENTS
    starts = periods[None, :, :1] + np.arange(SUBSEGMENTS)[None, None, :] * width
    relative = clock[:, None, None] - starts

    def cdf(offset):
        return np.clip((relative - offset) / width, 0., 1.).mean(axis=2)

    def primitive(offset):
        delay = relative - offset
        span = np.clip(delay, 0., width)
        return span * (delay - .5 * span) / width

    zero = np.zeros((count, SUBSEGMENTS))
    load_start, load_end, berth, reserve, unload, crush_start, crush_end = offsets.transpose(2, 0, 1)
    dispatched, completed = cdf(zero), cdf(unload)
    reserved = cdf(reserve)
    crushed = (np.clip((primitive(crush_start) - primitive(crush_end))
                       / (crush_end - crush_start), 0., 1.).mean(axis=2))
    crushed[:, [column.key[3] == 1 for column in columns]] = 0.
    elapsed = np.maximum(0., np.stack([
        np.diff((primitive(first) - primitive(last)).mean(axis=2), axis=0)
        for first, last in ((zero, unload), (load_start, load_end), (berth, unload))
    ], axis=0))
    return dispatched, completed, reserved, crushed, elapsed, unload.mean(axis=1)


class _Rows:
    def __init__(self, width):
        self.width = width
        self.data, self.indices, self.rows, self.rhs = [], [], [], []

    def add(self, indices, data, rhs):
        self.rhs.append(float(rhs))
        indices = np.asarray(indices, dtype=np.int32)
        data = np.asarray(data, dtype=float)
        nz = np.abs(data) > 1e-11
        self.indices.append(indices[nz])
        self.data.append(data[nz])
        self.rows.append(np.full(np.count_nonzero(nz), len(self.rhs) - 1, dtype=np.int32))

    def add_block(self, indices, data, rhs):
        row, column = np.where(np.abs(data) > 1e-11)
        self.data.append(data[row, column])
        self.indices.append(indices[column])
        self.rows.append(row.astype(np.int32) + len(self.rhs))
        self.rhs.extend(rhs)

    def matrix(self):
        matrix = coo_matrix((np.concatenate(self.data),
                             (np.concatenate(self.rows), np.concatenate(self.indices))),
                            shape=(len(self.rhs), self.width)).tocsr()
        return matrix, np.asarray(self.rhs, dtype=float)


def _ready_origins(replay):
    """Known post-commitment origins plus the virtual initial release dependencies."""
    possible = []
    for tid, when in enumerate(replay.dispatch_release_times()):
        if when < SHIFT_END and replay._nodes[tid] is not None:
            possible.append((tid, replay._nodes[tid], when))
    return tuple(possible)


def _time_grid(replay, slots, ready):
    # Source constraints reserve every interval's launches from its opening
    # fleet balance. Include commitment releases to avoid delaying known returns
    # to a later regular grid tick.
    now = replay.origin
    points = {now, SHIFT_END}
    points.update(np.arange(now, SHIFT_END, CONSTRAINT_CHECK_SECONDS))
    points.update(t for segment in slots for t in segment)
    points.update(t for _, _, t in ready)
    points.update(t for t, *_ in replay.timeline)
    return np.asarray(sorted(points), dtype=float)


def _received(ready, truck_types, typ, origin, clock):
    estimates = [time for tid, node, time in ready if node == origin and truck_types[tid] == typ]
    return np.searchsorted(np.sort(estimates), clock, side='right')


def _add_stock_rows(ub, eq, columns, truck_types, ready, clock, dispatched, completed, first_stock):
    types, origins, _, dumps = np.asarray([column.key for column in columns], dtype=int).reshape(-1, 4).T
    q = np.asarray([column.q for column in columns])
    keys = [(t, o) for t in sorted(set(truck_types))
            for o in sorted(set(origins[types == t]) | set(dumps[types == t]))]
    for group, (typ, origin) in enumerate(keys):
        local = np.flatnonzero((types == typ) & ((origins == origin) | (dumps == origin)))
        present = local[origins[local] == origin]
        leave = np.diff(dispatched[:, present], axis=0) / q[present]
        in_group = dumps[local] == origin
        outbound = np.diff(dispatched[:, local], axis=0) / q[local] * (origins[local] == origin)
        inbound = np.diff(completed[:, local], axis=0) / q[local] * in_group
        supply = _received(ready, truck_types, typ, origin, clock)
        states = first_stock + group * len(clock) + np.arange(len(clock))
        eq.add([states[0]], [1.], supply[0])
        for j, release in enumerate(np.diff(supply)):
            eq.add(np.r_[local, states[j:j + 2]], np.r_[outbound[j] - inbound[j], -1., 1.], release)
            ub.add(np.r_[present, states[j]], np.r_[leave[j], -1.], 0.)


def _problem(replay, columns, truck_types, ready, warehouse, clock, committed):
    count = len(columns)
    groups = {(col.key[0], origin) for col in columns for origin in (col.key[1], col.key[3])}
    windows = tuple(range(committed.first_window, 4))
    first_stock = count + 2 + 3 * len(windows)
    variables = first_stock + len(clock) * len(groups)
    ub, eq = _Rows(variables), _Rows(variables)
    keys = np.asarray([column.key for column in columns], dtype=int).reshape(count, 4)
    types, _, sites, dumps = keys.T
    ore_cols = np.flatnonzero(dumps == 0)
    waste_cols = np.flatnonzero(dumps == 1)
    coeff = np.arange(count)
    q = np.asarray([column.q for column in columns])
    grade = np.asarray([column.grade for column in columns])
    durations = np.diff(clock)
    dispatched, completed, reserved, crushed, work, cycle = _curves(columns, clock, warehouse)
    # A resource row uses occupied seconds. Slots awaiting warehouse admission
    # are included in the committed and prospective ore-dump intervals.
    for site, supply in enumerate(replay._available_tons):
        indices = np.flatnonzero(sites == site)
        ub.add(indices, np.ones(indices.size), supply)
    ub.add(waste_cols, np.ones(waste_cols.size),
           max(0., WASTE_STOP_TONS - replay.waste_committed_at_origin))
    for resource, intervals in enumerate(replay.resource_intervals):
        is_loader = resource < 4
        indices = np.flatnonzero(sites == resource if is_loader else dumps == resource - 4)
        cap = replay._loads[resource].capacity if is_loader else replay._dump_capacity[resource - 4]
        remaining = durations * cap
        for start, stop in intervals:
            remaining -= np.maximum(0., np.minimum(stop, clock[1:]) - np.maximum(start, clock[:-1]))
        ub.add_block(indices, work[1 if is_loader else 2][:, indices] / q[indices],
                     np.maximum(0., remaining))
    stock, initial_reserved, old_crushed = warehouse.balances(clock)
    ub.add_block(ore_cols, reserved[:, ore_cols] - crushed[:, ore_cols],
                 np.maximum(0., warehouse.capacity - stock - initial_reserved))
    ub.add_block(ore_cols, np.diff(crushed[:, ore_cols], axis=0),
                 np.maximum(0., warehouse.rate * durations - np.diff(old_crushed)))
    for typ in sorted(set(types)):
        indices = np.flatnonzero(types == typ)
        availability = np.zeros(len(durations))
        for tid, _, release in ready:
            if truck_types[tid] == typ:
                availability += np.maximum(0., clock[1:] - np.maximum(clock[:-1], release))
        ub.add_block(indices, work[0][:, indices] / q[indices], availability)
    # Upper bound rows also serve as the extra-dispatch check. Availability rows
    # are added after that view because the current truck is observed as ready.
    added_q = np.asarray([
        crushed[np.searchsorted(clock, (w + 1) * WINDOW_SECONDS)] -
        crushed[np.searchsorted(clock, max(w * WINDOW_SECONDS, replay.origin))] for w in windows
    ])
    added_m = added_q * grade
    delivered_waste = completed[-1] * (dumps == 1)
    x_ore, x_waste = count, count + 1
    ub.add(np.r_[coeff, x_ore], np.r_[-added_q.sum(axis=0), -1.],
           committed.ore_tons - ORE_TARGET_TONS)
    ub.add(np.r_[coeff, x_waste], np.r_[-delivered_waste, -1.],
           committed.waste_tons - WASTE_TARGET_TONS)
    lack, fleet = np.zeros(variables), np.zeros(variables)
    lack[x_ore:x_waste + 1] = (1 / ORE_TARGET_TONS, 1 / WASTE_TARGET_TONS)
    fleet[:count] = cycle / q
    minimum_supply = min(truck.capacity for truck in replay._trucks)
    for offset, w in enumerate(windows):
        low, high, empty = count + 2 + 3 * offset + np.arange(3)
        q0, m0 = committed.fifo_q[w], committed.fifo_m[w]
        ub.add(np.r_[coeff, low], np.r_[GRADE_MIN * added_q[offset] - added_m[offset], -1.],
               m0 - GRADE_MIN * q0)
        ub.add(np.r_[coeff, high], np.r_[added_m[offset] - GRADE_MAX * added_q[offset], -1.],
               GRADE_MAX * q0 - m0)
        ub.add(np.r_[coeff, empty], np.r_[-added_q[offset], -1.], q0 - minimum_supply)
        lack[[low, high, empty]] = (1 / (.01 * ORE_TARGET_TONS / 4),
                                    1 / (.01 * ORE_TARGET_TONS / 4), 1 / minimum_supply)
    guard_rows = len(ub.rhs)
    _add_stock_rows(ub, eq, columns, truck_types, ready, clock, dispatched, completed, first_stock)
    return (ub.matrix(), eq.matrix() if groups else (None, None), lack, fleet, guard_rows,
            added_q, added_m, delivered_waste, old_crushed, crushed)


def _verified(result, ub, rhs, eq, equals):
    if result.status not in (0, 1) or result.x is None or not np.isfinite(result.x).all():
        return None
    if np.any(result.x < -MASS_TOL):
        return None
    x = np.maximum(0., result.x)
    if np.any(ub @ x - rhs > MASS_TOL) or (
            eq is not None and np.any(np.abs(eq @ x - equals) > MASS_TOL)):
        return None
    return x


def _solve(objective, ub, rhs, eq, equals, deadline):
    _require_time(deadline)
    return linprog(objective, A_ub=ub, b_ub=rhs, A_eq=eq, b_eq=equals,
                   bounds=(0., None), method='highs',
                   options={'time_limit': max(.001, deadline - perf_counter())})


def solve_plan(observation, state, *, deadline):
    """Publish a verified, sparse, two-stage flow plan for an actionable RPC state."""
    _require_time(deadline)
    try:
        replay = _Replay.from_observation(observation)
        if (replay.now != state.now or replay.current_truck != state.current_truck
                or replay.candidates() != state.candidates()):
            raise FlowPlanError('flow_plan_observation_mismatch')
        replay._move(check_deadline=lambda: _require_time(deadline), dispatch=False)
    except NominalModelError as error:
        raise FlowPlanError(str(error)) from error
    committed = replay.score_inputs()
    truck_types = np.argmax(np.asarray(observation)[27:1107].reshape(9, 120)[:, 1:4], axis=1).tolist()
    ready = _ready_origins(replay)
    warehouse = _Warehouse(replay)
    slots = _slots(replay.origin)
    columns = tuple(_columns(replay, truck_types, warehouse, ready, slots, deadline))
    clock = _time_grid(replay, slots, ready)
    _require_time(deadline)
    (ub, rhs), (eq, equals), lack, fleet, guard_rows, added_q, added_m, waste, old_crushed, crushed = (
        _problem(replay, columns, truck_types, ready, warehouse, clock, committed)
    )
    first = _solve(lack, ub, rhs, eq, equals, deadline)
    x = _verified(first, ub, rhs, eq, equals)
    if x is None:
        raise FlowPlanError(f'flow_lp_stage_one_status_{first.status}')
    status = 'feasible_limit'
    if first.status == 0:
        status = 'feasible_stage_one'
        if deadline - perf_counter() >= .012:
            tightened = vstack([ub, csr_matrix(lack.reshape(1, -1))], format='csr')
            bound = float(lack @ x + 1e-7)
            tightened_rhs = np.r_[rhs, bound]
            fit = _solve(fleet, tightened, tightened_rhs, eq, equals, deadline)
            candidate = _verified(fit, tightened, tightened_rhs, eq, equals)
            if candidate is not None and lack @ candidate <= bound + 1e-7:
                x = candidate
                status = 'optimal' if fit.status == 0 else 'feasible_limit'
    count = len(columns)
    routes, guard = {}, {}
    for index, (column, quantity) in enumerate(zip(columns, x[:count], strict=True)):
        guard.setdefault(column.key, []).append((*column.depart, index))
        if quantity > MASS_EPS:
            routes.setdefault(column.key, []).append(FlowSlot(*column.depart, float(quantity)))
    visible = tuple(WindowProjection(w, committed.fifo_q[w], committed.fifo_m[w])
                    for w in range(committed.first_window, 4))
    plan_windows = tuple(
        WindowProjection(window.window, float(window.ore_tons + q), float(window.metal_tons + m))
        for window, q, m in zip(visible, added_q @ x[:count], added_m @ x[:count], strict=True)
    )
    slack = x[count + 2:count + 2 + 3 * len(visible)].reshape(-1, 3)
    residual = np.maximum(0., rhs[:guard_rows] - ub[:guard_rows] @ x)
    residual_max = max(0., float(np.max(ub @ x - rhs)),
                       float(np.max(np.abs(eq @ x - equals))) if eq is not None else 0.)
    return FlowPlan(
        replay.origin, {key: tuple(value) for key, value in routes.items()},
        visible, plan_windows, float(committed.ore_tons + (added_q @ x[:count]).sum()),
        float(committed.waste_tons + waste @ x[:count]), float(x[count]), float(x[count + 1]),
        tuple((float(low), float(high)) for low, high in slack[:, :2]),
        tuple(map(float, slack[:, 2])), status, residual_max, clock,
        old_crushed + crushed @ x[:count], replay.initial_processed[committed.first_window],
        {key: tuple(value) for key, value in guard.items()}, ub[:guard_rows, :count], residual,
    )
