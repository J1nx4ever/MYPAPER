"""Rolling trip appointments with a FIFO MILP and a common-event replay check."""

from collections import Counter
from dataclasses import dataclass
from math import inf, isfinite
from time import perf_counter

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, vstack

from change_plans.token_attention_001.business_graph_summary_009.baseline_comparison_001.dispatch_flow_matching_001.flow_matching_plan import (
    _Replay, _Warehouse, _blocked, _start_after,
)
from single_critic_task_query_v1.targets import ORE_TARGET_TONS, WASTE_TARGET_TONS
from traditional_dispatch_v1.decision import _cost, _fallback_action
from traditional_dispatch_v2.decision import _tail_action
from traditional_dispatch_v2.nominal import (
    MASS_EPS, MASS_TOL, SHIFT_END, WASTE_STOP_TONS, WINDOW_SECONDS,
    NominalModelError, NominalState,
)


class _Deadline(Exception):
    pass


@dataclass(frozen=True, slots=True)
class TripChoice:
    action: int
    kind: str
    fallback_reason: str | None = None
    timed_out: bool = False
    replanned: bool = False
    lp_status: str | None = None
    matched_trucks: int = 0
    assignments_scored: int = 0
    flow_shortfall_ore_tons: float | None = None
    flow_shortfall_waste_tons: float | None = None
    flow_grade_slack_tons: float | None = None
    mip_attempted: bool = False
    mip_feasible: bool = False
    mip_gap: float | None = None
    solve_stages: int = 0
    solver_calls: int = 0
    replay_rejections: int = 0
    estimated_cost: float | None = None
    incumbent_cost: float | None = None
    candidate_cost: float | None = None
    fifo_cross_window: bool = False


@dataclass(frozen=True, slots=True)
class _Request:
    truck: int
    time: float
    origin: int
    options: tuple


@dataclass(frozen=True, slots=True)
class _Appointment:
    truck: int
    action: int
    dump: int
    tons: float
    grade: float
    load_arrival: float
    load_start: float
    load_end: float
    dump_arrival: float
    berth: float
    reserve: float
    unload: float
    crush_start: float
    crush_end: float

    @property
    def wait(self):
        return (min(self.load_start, SHIFT_END) - min(self.load_arrival, SHIFT_END)
                + min(self.reserve, SHIFT_END) - min(self.dump_arrival, SHIFT_END))


class _Rows:
    def __init__(self, width):
        self.width = width
        self.row = []
        self.col = []
        self.data = []
        self.low = []
        self.high = []

    def add(self, indices, values, *, high, low=-inf):
        self.row.extend([len(self.high)] * len(indices))
        self.col.extend(indices)
        self.data.extend(values)
        self.low.append(low)
        self.high.append(high)

    def matrix(self):
        return coo_matrix((self.data, (self.row, self.col)), shape=(len(self.high), self.width)).tocsr()


def _check(deadline):
    if perf_counter() >= deadline:
        raise _Deadline


def _requests_and_tail(initial, horizon, max_trips, check):
    """One tentative rollout identifies distinct requests and estimates the far contribution."""
    state = initial.clone()
    requests = {}
    prefix = None
    while not state.finished:
        check()
        action = _tail_action(state, check)
        truck = state.current_truck
        if prefix is None:
            if truck in requests or state.now >= horizon or len(requests) >= max_trips:
                prefix = state.clone()
            else:
                requests[truck] = _Request(truck, state.now, state._nodes[truck], state.candidates())
        if not requests:
            raise NominalModelError('trip_horizon_has_no_current_request')
        state.apply_action(action)
        state.advance(check_deadline=check)
    near = prefix if prefix is not None else state
    near._move(check_deadline=check, dispatch=False)
    return tuple(requests.values()), state.score_inputs(), near.score_inputs()


def _old_processing(replay):
    """Reserve the visible commitments' nominal crusher intervals."""
    rate = replay._crusher_rate
    if rate <= 0:
        return ()
    time = replay.origin
    jobs = []
    packets = [(time, replay.initial_stock), *replay.ore_arrivals]
    for arrival, tons in packets:
        if tons > MASS_EPS:
            start = max(time, arrival)
            time = start + tons / rate
            jobs.append((arrival, start, time))
    return tuple(jobs)


def _crusher_slots(unload, tons, rate, old, step):
    if rate <= 0 or unload >= SHIFT_END:
        yield inf, inf
        return
    start = unload
    for arrival, old_start, end in old:
        if arrival <= unload:
            start = max(start, end)
        else:
            stop = old_start
            break
    else:
        stop = inf
    duration = tons / rate
    for delay in (0., step, 2 * step, 3 * step):
        left = start + delay
        right = left + duration
        if min(right, SHIFT_END) < min(stop, SHIFT_END) + 1e-7:
            yield left, right
        if left >= SHIFT_END:
            break


def _appointments(request, replay, warehouse, old, blocked, step, check):
    truck = replay._trucks[request.truck]
    latest_old_arrival = old[-1][0] if old else replay.origin
    for candidate in request.options:
        check()
        site, q = candidate.action, candidate.tons
        load = replay._loads[site]
        dump = int(not candidate.ore)
        load_time = load.service(q)
        dump_time = truck.waste_service if dump else truck.ore_service
        load_arrival = request.time + replay._empty_times[request.origin][load.node]
        for load_delay in (0., step):
            load_start = _start_after(blocked[site], load_arrival + load_delay, load_time)
            load_end = load_start + load_time
            dump_arrival = load_end + replay._loaded_times[load.node][dump]
            berth_options = (dump_arrival, dump_arrival + step)
            if not dump and dump_arrival < latest_old_arrival:
                berth_options += (max(dump_arrival, latest_old_arrival - dump_time + 1e-5),)
            for berth_option in berth_options:
                berth = _start_after(blocked[4 + dump], berth_option, dump_time)
                reserve = berth
                if not dump and berth < SHIFT_END:
                    while berth < SHIFT_END:
                        check()
                        reserve = warehouse.available_at(berth, q)
                        if not isfinite(reserve):
                            break
                        next_berth = _start_after(blocked[4], berth, reserve - berth + dump_time)
                        if next_berth <= berth:
                            break
                        berth = next_berth
                    reserve = max(berth, reserve)
                unload = reserve + dump_time
                slots = _crusher_slots(unload, q, warehouse.rate, old, step) if not dump else ((inf, inf),)
                for start, end in slots:
                    yield _Appointment(request.truck, site, dump, q, candidate.grade, load_arrival,
                                       load_start, load_end, dump_arrival, berth, reserve, unload, start, end)


def _resource_rows(rows, intervals, cap, clock):
    """Capacity of a fixed committed calendar and the selected appointments."""
    starts, ends = intervals
    deltas = Counter()
    for start, end in clock:
        if start < end and start < SHIFT_END:
            deltas[start] += 1
            deltas[end] -= 1
    times = sorted(set(starts[np.isfinite(starts) & (starts < SHIFT_END)]) | set(deltas))
    committed = 0
    for t in times:
        if t >= SHIFT_END:
            break
        committed += deltas[t]
        ids = np.flatnonzero((starts <= t) & (t < ends))
        if len(ids):
            rows.add(ids, np.ones(len(ids)), high=max(0, cap - committed))


def _fifo_conflicts(rows, options, check):
    ids = np.array([option.truck for option in options])
    dump = np.array([option.dump for option in options])
    site = np.array([option.action for option in options])
    la = np.array([option.load_arrival for option in options])
    ls = np.array([option.load_start for option in options])
    da = np.array([option.dump_arrival for option in options])
    berth = np.array([option.berth for option in options])
    unload = np.array([option.unload for option in options])
    start = np.array([option.crush_start for option in options])
    for i in range(len(options)):
        check()
        earlier_load = (la[i] < la) | ((la[i] == la) & (ids[i] < ids))
        earlier_dump = (da[i] < da) | ((da[i] == da) & (ids[i] < ids))
        earlier_packet = (unload[i] < unload) | ((unload[i] == unload) & (ids[i] < ids))
        conflict = ((earlier_load & (site[i] == site) & (ls[i] > ls))
                    | (earlier_dump & (dump[i] == dump) & (berth[i] > berth))
                    | (earlier_packet & (dump[i] == 0) & (dump == 0) & (start[i] > start)))
        for j in np.flatnonzero(conflict & (ids != ids[i])):
            rows.add((i, j), (1., 1.), high=1.)


def _projection(options, initial, baseline, near, rate):
    q = np.array([item.tons for item in options])
    grades = np.array([item.grade for item in options])
    starts = np.array([item.crush_start for item in options])
    ends = np.array([item.crush_end for item in options])
    windows = range(initial.first_window, 4)
    if rate > 0:
        processed = np.array([np.maximum(0., np.minimum(ends, (w + 1) * WINDOW_SECONDS)
                                        - np.maximum(starts, w * WINDOW_SECONDS)) * rate for w in windows])
    else:
        processed = np.zeros((len(windows), len(options)))
    processed = np.minimum(processed, q)
    metals = processed * grades
    initial_q = np.array(initial.fifo_q)[initial.first_window:]
    initial_m = np.array(initial.fifo_m)[initial.first_window:]
    far_q = np.maximum(0., np.array(baseline.fifo_q)[initial.first_window:]
                       - np.array(near.fifo_q)[initial.first_window:])
    far_m = np.clip(np.array(baseline.fifo_m)[initial.first_window:]
                    - np.array(near.fifo_m)[initial.first_window:], 0., far_q)
    delivered = np.array([item.tons if item.dump and item.unload < SHIFT_END else 0. for item in options])
    ore_floor = initial.ore_tons + far_q.sum()
    waste_floor = initial.waste_tons + max(0., baseline.waste_tons - near.waste_tons)
    return processed, metals, initial_q + far_q, initial_m + far_m, delivered, ore_floor, waste_floor


def _model(initial, baseline, near, requests, options, replay, warehouse, check):
    n = len(options)
    ids = np.array([item.truck for item in options])
    sites = np.array([item.action for item in options])
    dumps = np.array([item.dump for item in options])
    tons = np.array([item.tons for item in options])
    delivered_time = np.array([item.unload for item in options])
    reserve = np.array([item.reserve for item in options])
    crush_start = np.array([item.crush_start for item in options])
    crush_end = np.array([item.crush_end for item in options])
    q, m, base_q, base_m, waste, base_ore, base_waste = _projection(
        options, initial, baseline, near, warehouse.rate)
    window_count = len(base_q)
    primary_end = n + 2 + 3 * window_count
    width = primary_end + 2 * window_count
    rows = _Rows(width)
    for request in requests:
        found = np.flatnonzero(ids == request.truck)
        rows.add(found, np.ones(len(found)), low=1., high=1.)
    for site, load in enumerate(replay._loads):
        found = np.flatnonzero(sites == site)
        rows.add(found, tons[found], high=replay._available_tons[site] + MASS_TOL)
        starts = np.array([item.load_start if item.action == site else inf for item in options])
        ends = np.array([item.load_end for item in options])
        _resource_rows(rows, (starts, ends), load.capacity, replay.resource_intervals[site])
    for dump, cap in enumerate(replay._dump_capacity):
        starts = np.array([item.berth if item.dump == dump else inf for item in options])
        _resource_rows(rows, (starts, delivered_time), cap, replay.resource_intervals[4 + dump])
    _resource_rows(rows, (crush_start, crush_end), 1, ())
    for time in sorted(set(reserve[(dumps == 0) & (reserve < SHIFT_END)]) | set(warehouse.times)):
        check()
        stock, reserved, _ = warehouse.balances(time)
        found = np.flatnonzero((dumps == 0) & (reserve <= time) & (time < crush_end))
        if len(found):
            used = tons[found] - np.maximum(0., time - crush_start[found]) * warehouse.rate
            rows.add(found, np.maximum(0., used), high=max(0., warehouse.capacity - stock - reserved) + MASS_TOL)
    # Dispatching the final waste vehicle may cross the stop threshold once.
    waste_indices = np.flatnonzero(dumps == 1)
    maximum_overrun = max(truck.capacity for truck in replay._trucks)
    remaining = max(0., WASTE_STOP_TONS + maximum_overrun - replay.waste_committed_at_origin)
    rows.add(waste_indices, tons[waste_indices], high=remaining + MASS_TOL)
    dispatch_slack = max(0., WASTE_STOP_TONS - replay.waste_committed_at_origin - MASS_EPS)
    for prefix_size, request in enumerate(requests):
        earlier = waste_indices[np.isin(ids[waste_indices], [r.truck for r in requests[:prefix_size]])]
        current = waste_indices[ids[waste_indices] == request.truck]
        if len(current):
            rows.add(np.r_[earlier, current], np.r_[tons[earlier], np.full(len(current), remaining)],
                     high=dispatch_slack + remaining)
    _fifo_conflicts(rows, options, check)

    features = np.arange(n)

    def shortage(offset, coefficients, limit, scale):
        rows.add(np.r_[features, offset], np.r_[coefficients / scale, -1.], high=limit / scale)

    shortage(n, -q.sum(axis=0), base_ore - ORE_TARGET_TONS, ORE_TARGET_TONS)
    shortage(n + 1, -waste, base_waste - WASTE_TARGET_TONS, WASTE_TARGET_TONS)
    for i in range(window_count):
        outer = n + 2 + 3 * i
        inner = primary_end + 2 * i
        shortage(outer, .26 * q[i] - m[i], base_m[i] - .26 * base_q[i], 40.)
        shortage(outer + 1, m[i] - .28 * q[i], .28 * base_q[i] - base_m[i], 40.)
        shortage(outer + 2, -q[i], base_q[i] - 1., 91.7)
        shortage(inner, .266 * q[i] - m[i], base_m[i] - .266 * base_q[i], 24.)
        shortage(inner + 1, m[i] - .274 * q[i], .274 * base_q[i] - base_m[i], 24.)
    primary = np.zeros(width)
    primary[n:primary_end] = 1.
    secondary = np.zeros(width)
    secondary[:n] = np.array([item.wait for item in options]) * (.2 / SHIFT_END)
    secondary[primary_end:] = 1.
    return rows, primary, secondary


def _solve(options, rows, primary, secondary, deadline):
    n, width = len(options), rows.width
    matrix, low, high = rows.matrix(), np.array(rows.low), np.array(rows.high)
    bounds = Bounds(np.zeros(width), np.r_[np.ones(n), np.full(width - n, inf)])
    integral = np.r_[np.ones(n), np.zeros(width - n)]
    accepted, gap, stages, timed_out = None, None, 0, False
    for objective in (primary, secondary):
        remaining = deadline - perf_counter()
        if remaining < .015:
            timed_out = True
            break
        fit = milp(objective, integrality=integral, bounds=bounds, constraints=LinearConstraint(matrix, low, high),
                   options={'time_limit': min(.190, remaining * .80), 'mip_rel_gap': .001})
        timed_out = fit.status == 1
        x = fit.x
        if x is None or not np.all(np.isfinite(x)):
            break
        rounded = np.round(x[:n])
        if np.max(np.abs(x[:n] - rounded)) > 1e-5 or np.min(x) < -1e-5 or np.max(rounded) > 1.:
            break
        x = np.r_[rounded, x[n:]]
        fitted = matrix @ x
        if np.any(fitted < low - 1e-5) or np.any(fitted > high + 1e-5):
            break
        accepted = tuple(item for item, pick in zip(options, rounded, strict=True) if pick)
        ratio = getattr(fit, 'mip_gap', None)
        gap = float(ratio) if ratio is not None and isfinite(ratio) else None
        if fit.status != 0:
            break
        stages += 1
        matrix = vstack((matrix, objective.reshape(1, -1)), format='csr')
        low = np.r_[low, -inf]
        high = np.r_[high, objective @ x + 1e-6]
    return accepted, gap, stages, timed_out


def _replay_choice(initial, chosen, cutoff, check):
    remaining = {item.truck: item for item in chosen}
    branch = initial.clone()
    while not branch.finished:
        check()
        item = remaining.pop(branch.current_truck, None)
        if item is None:
            action = _tail_action(branch, check)
        else:
            if branch.now >= cutoff:
                raise NominalModelError('mip_trip_released_after_lookahead')
            candidate = next((option for option in branch.candidates() if option.action == item.action), None)
            if candidate is None or abs(candidate.tons - item.tons) > MASS_TOL:
                raise NominalModelError('mip_discrete_trip_not_feasible_in_fifo_replay')
            action = item.action
        if remaining and branch.now >= cutoff:
            raise NominalModelError('mip_truck_missing_from_near_dispatches')
        branch.apply_action(action)
        branch.advance(check_deadline=check)
    if remaining:
        raise NominalModelError('mip_truck_missing_from_nominal_dispatches')
    return branch.score_inputs()


class TripFifoPolicy:
    """Optimize individual near-term truck choices, then commit only the real action."""

    def __init__(self, *, planning_ms=800, horizon_seconds=1800, max_trips=4, slot_seconds=90):
        if (not isfinite(planning_ms) or planning_ms <= 0 or not isfinite(horizon_seconds)
                or horizon_seconds <= 0 or type(max_trips) is not int or not 2 <= max_trips <= 9
                or not isfinite(slot_seconds) or slot_seconds <= 0):
            raise ValueError('invalid trip FIFO planning limits')
        self.planning_ms = planning_ms
        self.horizon_seconds = horizon_seconds
        self.max_trips = max_trips
        self.slot_seconds = slot_seconds
        self._pending = None
        self._counts = Counter()
        self._statuses = Counter()
        self._fallbacks = Counter()
        self._actions = [0] * 4
        self._max_gap = None

    def choose(self, observation, mask):
        if self._pending is not None:
            raise ValueError('previous trip FIFO action was not confirmed')
        start = perf_counter()
        obs, legal = np.asarray(observation), np.asarray(mask)
        if (obs.dtype != np.float32 or obs.shape != (1519,) or not np.isfinite(obs).all()
                or obs[17] or legal.shape != (4,) or not np.isin(legal, (0, 1)).all()
                or not np.array_equal(obs[19:23], legal) or not legal.any()):
            raise ValueError('actionable 1519D observation and matching nonempty legal mask required')
        actions = tuple(map(int, np.flatnonzero(legal)))
        if len(actions) == 1:
            self._pending = TripChoice(actions[0], 'single_legal')
        else:
            self._pending = self._plan(obs, actions, start)
        return self._pending

    def _plan(self, obs, actions, started):
        deadline = started + self.planning_ms / 1000.
        incumbent_action = _fallback_action(obs, actions)
        diagnostics = {}
        try:
            initial = NominalState.from_observation(obs)
            if tuple(item.action for item in initial.candidates()) != actions:
                raise NominalModelError('nominal_root_candidates_differ_from_live_mask')
            cutoff = min(initial.now + self.horizon_seconds, SHIFT_END)
            incumbent_action = _tail_action(initial, lambda: _check(deadline))
            requests, baseline, near = _requests_and_tail(
                initial, cutoff, self.max_trips, lambda: _check(deadline))
            incumbent_cost = _cost(baseline, unload=False)
            diagnostics['incumbent_cost'] = incumbent_cost
            replay_budget = max(.095, (perf_counter() - started) * 1.5)
            solve_cutoff = deadline - replay_budget
            check = lambda: _check(solve_cutoff)
            replay = _Replay.from_observation(obs)
            replay._move(check_deadline=check, dispatch=False)
            warehouse = _Warehouse(replay)
            old = _old_processing(replay)
            capacities = [load.capacity for load in replay._loads] + list(replay._dump_capacity)
            calendars = tuple(_blocked(intervals, cap)
                              for intervals, cap in zip(replay.resource_intervals, capacities, strict=True))
            options = tuple(dict.fromkeys(
                option for request in requests
                for option in _appointments(request, replay, warehouse, old, calendars, self.slot_seconds, check)
            ))
            diagnostics['assignments_scored'] = len(options)
            if {item.truck for item in options} != {request.truck for request in requests}:
                raise NominalModelError('mip_truck_has_no_discrete_appointment')
            rows, primary, secondary = _model(replay.score_inputs(), baseline, near, requests, options,
                                              replay, warehouse, check)
            diagnostics.update(mip_attempted=True, lp_status='failed', replay_rejections=0)
            rejection = 'mip_no_verified_incumbent'
            for attempt in range(1, 4):
                check()
                diagnostics['solver_calls'] = attempt
                solution, gap, stages, timed_out = _solve(options, rows, primary, secondary, solve_cutoff)
                diagnostics['timed_out'] = diagnostics.get('timed_out', False) or timed_out
                if solution is None:
                    break
                diagnostics.update(mip_feasible=True, lp_status='optimal' if stages == 2 else 'feasible_limit',
                                   mip_gap=gap, solve_stages=stages)
                try:
                    predicted = _replay_choice(initial, solution, cutoff, lambda: _check(deadline))
                    cost = _cost(predicted, unload=False)
                    diagnostics['candidate_cost'] = cost
                    if cost > incumbent_cost + 1e-8:
                        raise NominalModelError('mip_fifo_replay_worse_than_nominal_incumbent')
                except NominalModelError as error:
                    rejection = str(error)
                    diagnostics['replay_rejections'] += 1
                    # Replay depends only on assignments, not on the surrogate
                    # appointment times. Exclude that entire joint assignment.
                    assigned = {(item.truck, item.action) for item in solution}
                    excluded = [index for index, item in enumerate(options)
                                if (item.truck, item.action) in assigned]
                    rows.add(excluded, [1.] * len(excluded), high=len(solution) - 1.)
                    continue
                grade_slack = sum(max(0., .26 * q - m, m - .28 * q)
                                  for q, m in zip(predicted.fifo_q, predicted.fifo_m, strict=True))
                action = next(item.action for item in solution if item.truck == initial.current_truck)
                return TripChoice(action, 'planned', replanned=True, matched_trucks=len(solution),
                                  flow_shortfall_ore_tons=max(0., ORE_TARGET_TONS - predicted.ore_tons),
                                  flow_shortfall_waste_tons=max(0., WASTE_TARGET_TONS - predicted.waste_tons),
                                  flow_grade_slack_tons=grade_slack, estimated_cost=cost,
                                  fifo_cross_window=bool(predicted.root_cross_window), **diagnostics)
            raise NominalModelError(rejection)
        except _Deadline:
            diagnostics['timed_out'] = True
            return TripChoice(incumbent_action, 'fallback', 'trip_planning_deadline', **diagnostics)
        except NominalModelError as error:
            return TripChoice(incumbent_action, 'fallback', str(error), **diagnostics)

    def commit(self, choice):
        if choice is not self._pending or choice is None:
            raise ValueError('trip FIFO commit differs from the unconfirmed choice')
        self._pending = None
        self._actions[choice.action] += 1
        self._counts[choice.kind] += 1
        self._counts['mip_attempts'] += int(choice.mip_attempted)
        self._counts['mip_feasible'] += int(choice.mip_feasible)
        self._counts['mip_accepted'] += int(choice.replanned)
        self._counts['joint_plan_commits'] += int(choice.matched_trucks > 1)
        self._counts['solver_calls'] += choice.solver_calls
        self._counts['replay_rejections'] += choice.replay_rejections
        self._counts['timed_out'] += int(choice.timed_out)
        self._counts['fifo_cross_window'] += int(choice.fifo_cross_window)
        if choice.fallback_reason:
            self._fallbacks[choice.fallback_reason] += 1
        if choice.lp_status:
            self._statuses[choice.lp_status] += 1
        if choice.mip_gap is not None:
            self._max_gap = max(choice.mip_gap, self._max_gap or 0.)

    def summary(self):
        return dict(self._counts, committed_actions=list(self._actions),
                    solver_status_counts=dict(self._statuses), fallback_reasons=dict(self._fallbacks),
                    max_mip_gap=self._max_gap)
