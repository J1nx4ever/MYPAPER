"""Plan nominal haulage target tonnes for the remaining part of a shift."""
from __future__ import annotations

from dataclasses import dataclass
import math
from time import perf_counter

import numpy as np
from scipy.optimize import linprog

from single_critic_task_query_v1.targets import (
    GRADE_CENTER, GRADE_DEADBAND, GRADE_SCALE, ORE_TARGET_TONS, ORE_WEIGHT,
    WASTE_TARGET_TONS, WASTE_WEIGHT,
)
from .nominal import (
    MASS_EPS, MASS_TOL, SHIFT_END, WASTE_STOP_TONS, WINDOW_SECONDS, NominalModelError,
    NominalState, _DUMP_OFFSET, _FINISH,
)


FLOW_PLAN_CONTRACT = {
    'schema': 'dispatch-inspired-remaining-shift-flow-v2',
    'period_seconds': WINDOW_SECONDS,
    'classes': 3,
    'load_sites': 4,
    'flow_units': 'tonnes_dispatched_per_period',
    'dispatch_distribution': 'uniform_after_first_visible_class_release_in_each_quality_window',
    'reported_rates': 'tonnes_divided_by_full_remaining_window_duration',
    'cycle_origin': 'same_commodity_dump_steady_cycle',
    'partial_load_tons': 'minimum_truck_capacity_and_observed_available_site_supply',
    'new_ore_projection': 'unload_plus_visible_backlog_at_earliest_arrival_then_rate_limited_batch',
    'crusher_constraint': 'window_capacity_after_earliest_nominal_new_processing_less_existing_work',
    'grade_lower': GRADE_CENTER - GRADE_DEADBAND,
    'grade_upper': GRADE_CENTER + GRADE_DEADBAND,
    'grade_reference_tons': ORE_TARGET_TONS / 4,
    'grade_excess_scale': GRADE_SCALE,
    'minimum_grade_window_supply': 'one_smallest_observed_truck_payload',
    'grade_window_supply_shortfall_weight': 2.0,
    'production_targets_tons': [ORE_TARGET_TONS, WASTE_TARGET_TONS],
    'production_shortfall_weights': [ORE_WEIGHT, WASTE_WEIGHT],
    'waste_dispatch_stop_tons': WASTE_STOP_TONS,
    'fleet_time_weight': .02,
    'solver': 'scipy.linprog.highs',
    'default_time_limit_seconds': .25,
}


class FlowPlanError(ValueError):
    """The observation or solver could not provide a usable plan."""


@dataclass(frozen=True, slots=True)
class FlowPeriod:
    window: int
    start: float
    end: float
    tons_by_type_site: tuple[tuple[float, ...], ...]
    rate_tph_by_type_site: tuple[tuple[float, ...], ...]
    predicted_ore_tons: float
    predicted_metal_tons: float
    grade_window_supply_shortfall_tons: float
    dispatch_start_by_type_site: tuple[tuple[float | None, ...], ...]


@dataclass(frozen=True, slots=True)
class FlowPlan:
    now: float
    periods: tuple[FlowPeriod, ...]
    objective: float
    ore_shortfall_tons: float
    waste_shortfall_tons: float
    grade_slack_tons: tuple[tuple[float, float], ...]
    status: str
    committed_ore_tons: float
    committed_waste_tons: float
    predicted_ore_tons: float
    predicted_waste_tons: float
    solve_seconds: float


class _CommitmentReplay(NominalState):
    """The nominal model's existing trips only, with resource intervals observed."""

    def __init__(self, snapshot):
        self.resource_intervals = [[] for _ in range(6)]
        self.released_at = [SHIFT_END] * 9
        self.ore_arrivals = []
        super().__init__(snapshot)
        self.origin = self.now
        self.initial_stock = self._stored
        self.initial_processed = self.score_inputs().fifo_q
        self.initial_commitments = tuple(trip is not None for trip in self._trips)
        self.waste_committed_at_origin = self._waste_committed()
        self.initial_ready = set(self._ready) | {self.current_truck}
        self.initial_wait = self._first_dispatch_wait.copy()
        self.initial_dispatched = self._initial_dispatched.copy()

    def _push(self, timestamp, kind, resource, truck):
        super()._push(timestamp, kind, resource, truck)
        if kind == _FINISH:
            self.resource_intervals[resource].append((self.now, min(float(timestamp), SHIFT_END)))

    def _process_event(self, kind, resource, tid):
        if kind == _FINISH and resource >= _DUMP_OFFSET:
            trip = self._trips[tid]
            self.released_at[tid] = self.now
            if trip.ore:
                self.ore_arrivals.append((self.now, trip.tons))
        super()._process_event(kind, resource, tid)

    def _advance_time(self, until):
        # Waiting for space holds an ore dump position after the truck has
        # acquired a berth. This does not overlap its unload service interval.
        self.resource_intervals[_DUMP_OFFSET].extend(
            (self.now, until) for _ in self._capacity_wait
        )
        super()._advance_time(until)

    def ore_balance_at(self, arrival):
        stock, last, processed = self.initial_stock, self.origin, 0.0
        for event_time, tons in self.ore_arrivals:
            if event_time > arrival:
                break
            crushed = min(stock, (event_time - last) * self._crusher_rate)
            processed += crushed
            stock += tons - crushed
            last = event_time
        crushed = min(stock, (arrival - last) * self._crusher_rate)
        return stock - crushed, processed + crushed

    def dispatch_release_times(self):
        """First virtual availability after existing trips and initial dispatch gates."""
        release = [self.released_at[tid] if committed else SHIFT_END
                   for tid, committed in enumerate(self.initial_commitments)]
        for tid in self.initial_ready:
            release[tid] = self.origin
        pending = self.initial_wait.copy()
        while pending:
            ready = {}
            for truck, previous in pending.items():
                if previous == -1 or previous in self.initial_dispatched:
                    ready[truck] = self.origin
                elif release[previous] < SHIFT_END:
                    ready[truck] = release[previous]
            if not ready:
                break
            for truck, when in ready.items():
                release[truck] = when
                del pending[truck]
        return release


@dataclass(frozen=True, slots=True)
class _Route:
    truck_type: int
    site: int
    q: float
    dump: int
    grade: float
    load_start: float
    load_end: float
    dump_start: float
    dump_end: float

    @property
    def ore(self):
        return self.dump == 0


@dataclass(frozen=True, slots=True)
class _Launch:
    period_idx: int
    route: _Route
    departure: tuple[float, float]
    ore_processing_offsets: tuple[float, float] | None


def _overlap(start, end, lower, upper):
    return max(0.0, min(end, upper) - max(start, lower))


def _mean_overlap(departure, resource, offset_start, offset_end):
    """Mean occupied seconds in a target period for uniform launch time."""
    start, end = departure
    lower, upper = resource
    duration = end - start

    def primitive(cut):
        delta = cut - start
        span = min(duration, max(0.0, delta))
        return span * (delta - .5 * span)

    area = (primitive(upper - offset_start) - primitive(upper - offset_end)
            - primitive(lower - offset_start) + primitive(lower - offset_end))
    return min(max(0.0, area / duration), upper - lower, offset_end - offset_start)


def _routes(replay, truck_types):
    for typ in sorted(set(truck_types)):
        truck = replay._trucks[truck_types.index(typ)]
        for site, load in enumerate(replay._loads):
            dump = int(not load.ore)
            empty_s = replay._empty_times[dump][load.node]
            loaded_s = replay._loaded_times[load.node][dump]
            if (not load.available or not replay._dump_available[dump]
                    or replay._available_tons[site] <= MASS_EPS
                    or not math.isfinite(empty_s + loaded_s)):
                continue
            q = min(truck.capacity, replay._available_tons[site])
            load_end = empty_s + load.service(q)
            dump_start = load_end + loaded_s
            dump_duration = truck.ore_service if load.ore else truck.waste_service
            yield _Route(typ, site, q, dump, load.grade if load.ore else 0.0, empty_s, load_end,
                         dump_start, dump_start + dump_duration)


def _ore_offsets(route, departure_start, replay):
    rate = replay._crusher_rate
    if not route.ore or rate <= 0:
        return None
    # Reuse the visible committed queue at the earliest nominal arrival in
    # this period. New trips share the separate LP crusher capacity; their
    # mutual FIFO reordering is evaluated only by the physical simulation.
    delay = replay.ore_balance_at(departure_start + route.dump_end)[0] / rate
    start = route.dump_end + delay
    processing = route.q / rate
    return (start, start + processing)


def _launches(replay, truck_types, windows, release_times):
    routes = tuple(_routes(replay, truck_types))
    first_release = {typ: min(release for truck_type, release in zip(truck_types, release_times)
                              if truck_type == typ) for typ in set(truck_types)}
    for index, window in enumerate(windows):
        for route in routes:
            start = max(window[0], first_release[route.truck_type])
            if start < window[1]:
                yield _Launch(index, route, (start, window[1]), _ore_offsets(route, start, replay))


def _free_work(replay, truck_types, windows, release_times):
    capacity = np.zeros((len(windows), 9), dtype=float)
    for index, window in enumerate(windows):
        duration = window[1] - window[0]
        for tid, typ in enumerate(truck_types):
            capacity[index, typ] += max(0.0, window[1] - max(window[0], release_times[tid]))
        limits = [load.capacity for load in replay._loads] + list(replay._dump_capacity)
        for resource, limit in enumerate(limits):
            existing = math.fsum(_overlap(start, end, *window)
                                 for start, end in replay.resource_intervals[resource])
            capacity[index, 3 + resource] = max(0.0, limit * duration - existing)
    return capacity


def _lp(replay, truck_types, windows, committed):
    release_times = replay.dispatch_release_times()
    launches = tuple(_launches(replay, truck_types, windows, release_times))
    earliest_ore = min((launch.departure[0] + launch.ore_processing_offsets[0]
                        for launch in launches if launch.ore_processing_offsets), default=SHIFT_END)
    n_flows = len(launches)
    ore_gap, waste_gap = n_flows, n_flows + 1
    n = n_flows + 2 + 3 * len(windows)
    cost = np.zeros(n, dtype=float)
    cost[ore_gap], cost[waste_gap] = ORE_WEIGHT / ORE_TARGET_TONS, WASTE_WEIGHT / WASTE_TARGET_TONS
    supply = np.zeros((4, n), dtype=float)
    work = np.zeros((len(windows), 9, n), dtype=float)
    ore = np.zeros((len(windows), n), dtype=float)
    metal = np.zeros_like(ore)
    waste = np.zeros(n, dtype=float)
    waste_launch = np.zeros(n, dtype=float)
    for cursor, launch in enumerate(launches):
        route, departure = launch.route, launch.departure
        if launch.ore_processing_offsets:
            start, end = launch.ore_processing_offsets
            ore[:, cursor] = [_mean_overlap(departure, target, start, end) / (end - start)
                              for target in windows]
            metal[:, cursor] = route.grade * ore[:, cursor]
        if not route.ore:
            dispatch_time = _overlap(*departure, replay.origin, SHIFT_END - route.dump_end)
            waste[cursor] = dispatch_time / (departure[1] - departure[0])
            waste_launch[cursor] = 1.0
        supply[route.site, cursor] = 1.0
        cost[cursor] = (FLOW_PLAN_CONTRACT['fleet_time_weight'] * route.dump_end
                        / (route.q * len(truck_types) * SHIFT_END))
        for target_index, target in enumerate(windows):
            work[target_index, route.truck_type, cursor] = (
                _mean_overlap(departure, target, 0.0, route.dump_end) / route.q
            )
            work[target_index, 3 + route.site, cursor] = (
                _mean_overlap(departure, target, route.load_start, route.load_end) / route.q
            )
            work[target_index, 7 + route.dump, cursor] = (
                _mean_overlap(departure, target, route.dump_start, route.dump_end) / route.q
            )

    rows = [*supply, *work.reshape(-1, n), waste_launch]
    capacity = _free_work(replay, truck_types, windows, release_times)
    limits = [*replay._available_tons, *capacity.flatten(),
              max(0.0, WASTE_STOP_TONS - replay.waste_committed_at_origin)]
    ore_required = -ore.sum(axis=0)
    ore_required[ore_gap] = -1.0
    waste_required = -waste
    waste_required[waste_gap] = -1.0
    rows.extend((ore_required, waste_required))
    limits.extend((committed.ore_tons - ORE_TARGET_TONS,
                   committed.waste_tons - WASTE_TARGET_TONS))

    lo, hi = FLOW_PLAN_CONTRACT['grade_lower'], FLOW_PLAN_CONTRACT['grade_upper']
    minimum_supply = min(truck.capacity for truck in replay._trucks)
    for index, (start, end) in enumerate(windows):
        window = committed.first_window + index
        q0, m0 = committed.fifo_q[window], committed.fifo_m[window]
        previous = replay.initial_processed[window]
        rows.append(ore[index])
        # The aggregate crusher window cannot lend time before any new ore
        # can reach it. Deduct the committed work in the eligible interval.
        eligible_start = min(end, max(start, earliest_ore))
        before = replay.ore_balance_at(eligible_start)[1]
        after = replay.ore_balance_at(end)[1]
        remaining_capacity = (end - eligible_start) * replay._crusher_rate - (after - before)
        limits.append(max(0.0, remaining_capacity))

        lower = lo * ore[index] - metal[index]
        upper = metal[index] - hi * ore[index]
        lower_idx = n_flows + 2 + 3 * index
        upper_idx, feed_idx = lower_idx + 1, lower_idx + 2
        lower[lower_idx] = upper[upper_idx] = -1.0
        rows.extend((lower, upper))
        limits.extend((m0 - lo * q0, hi * q0 - m0))
        reference_tons = max(1.0, q0, min(FLOW_PLAN_CONTRACT['grade_reference_tons'],
                             previous + (end - start) * replay._crusher_rate))
        cost[lower_idx] = cost[upper_idx] = 1.0 / (GRADE_SCALE * reference_tons)
        # A Q=M=0 window satisfies the linear grade inequalities trivially.
        # Preserve a soft positive feed target even after shift tonnes are met.
        feed = -ore[index]
        feed[feed_idx] = -1.0
        rows.append(feed)
        limits.append(q0 - minimum_supply)
        cost[feed_idx] = FLOW_PLAN_CONTRACT['grade_window_supply_shortfall_weight'] / minimum_supply

    return launches, cost, np.asarray(rows), np.asarray(limits), ore, metal, waste


def solve_flow_plan(observation, state: NominalState | None = None, *, time_limit_seconds=None):
    """Solve the continuous plan; only the current raw decision state is read."""
    time_limit = (FLOW_PLAN_CONTRACT['default_time_limit_seconds']
                  if time_limit_seconds is None else time_limit_seconds)
    if not math.isfinite(time_limit) or time_limit <= 0:
        raise ValueError('flow LP requires a positive finite solver time limit')
    try:
        replay = _CommitmentReplay.from_observation(observation)
        if state is not None and (state.now != replay.now or state.current_truck != replay.current_truck
                                  or state.candidates() != replay.candidates()):
            raise FlowPlanError('nominal decision does not correspond to flow plan observation')
        first = replay.window_index
        windows = tuple((max(replay.now, w * WINDOW_SECONDS), (w + 1) * WINDOW_SECONDS)
                        for w in range(first, 4))
        replay._move(check_deadline=None, dispatch=False)
        committed = replay.score_inputs()
    except NominalModelError as error:
        raise FlowPlanError(str(error)) from error
    # NominalState has admitted the type one-hots and vehicle count already.
    trucks = np.asarray(observation)[27:1107].reshape(9, 120)
    truck_types = list(map(int, np.argmax(trucks[:, 1:4], axis=1)))
    launches, cost, matrix, limits, ore, metal, waste = _lp(replay, truck_types, windows, committed)
    started = perf_counter()
    fit = linprog(cost, A_ub=matrix, b_ub=limits, bounds=(0.0, None), method='highs',
                  options={'time_limit': time_limit})
    solve_seconds = perf_counter() - started
    if (fit.status not in (0, 1) or fit.x is None or not np.isfinite(fit.x).all()
            or np.min(fit.x) < -MASS_TOL or np.any(matrix @ fit.x - limits > MASS_TOL)):
        raise FlowPlanError(f'flow_lp_no_feasible_solution_status_{fit.status}')

    x = np.maximum(0.0, fit.x)
    shape = (len(windows), FLOW_PLAN_CONTRACT['classes'], FLOW_PLAN_CONTRACT['load_sites'])
    by_period = np.zeros(shape)
    dispatch_starts = [[[None] * shape[2] for _ in range(shape[1])] for _ in range(shape[0])]
    for index, launch in enumerate(launches):
        route = launch.route
        by_period[launch.period_idx, route.truck_type, route.site] = x[index]
        dispatch_starts[launch.period_idx][route.truck_type][route.site] = launch.departure[0]
    added_ore, added_metal = ore @ x, metal @ x
    n_flows = len(launches)
    periods = []
    for index, (start, end) in enumerate(windows):
        w = first + index
        tons = tuple(tuple(map(float, row)) for row in by_period[index])
        rates = tuple(tuple(q * 3600.0 / (end - start) for q in row) for row in tons)
        periods.append(FlowPeriod(w, start, end, tons, rates,
                                  float(committed.fifo_q[w] + added_ore[index]),
                                  float(committed.fifo_m[w] + added_metal[index]),
                                  float(x[n_flows + 4 + 3 * index]),
                                  tuple(tuple(row) for row in dispatch_starts[index])))
    slack = tuple((float(x[n_flows + 2 + 3 * index]), float(x[n_flows + 3 + 3 * index]))
                  for index in range(len(windows)))
    return FlowPlan(replay.origin, tuple(periods), float(cost @ x), float(x[n_flows]), float(x[n_flows + 1]),
                    slack, 'optimal' if fit.status == 0 else 'feasible_limit', float(committed.ore_tons),
                    float(committed.waste_tons), float(committed.ore_tons + added_ore.sum()),
                    float(committed.waste_tons + waste @ x), solve_seconds)
