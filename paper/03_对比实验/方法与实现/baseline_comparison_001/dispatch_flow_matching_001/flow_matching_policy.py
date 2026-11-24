"""Deterministic path-first DISPATCH matching on the live four-action interface."""

from collections import Counter, defaultdict
from dataclasses import dataclass
from heapq import heappop
from math import floor, inf, isfinite
from time import perf_counter

import numpy as np

from traditional_dispatch_v1.nominal import _Trip
from traditional_dispatch_v2 import dispatch_policy as shared
from traditional_dispatch_v2.nominal import (
    MASS_EPS, MASS_TOL, SHIFT_END, WASTE_STOP_TONS, WINDOW_SECONDS,
    NominalModelError, NominalState, _ARRIVE,
)

if __package__:
    from .flow_matching_plan import FlowPlanError, solve_plan
else:
    from flow_matching_plan import FlowPlanError, solve_plan


MATCHING_CONTRACT = {
    "schema": "dispatch-path-flow-matching-v1",
    "route_key": ("truck_type", "empty_origin_node", "load_site", "fixed_dump"),
    "matching": "earliest_eligible_need_then_marginal_lost_tons",
    "local_calendar_seconds": 1800,
    "report_period_seconds": 900,
    "planner_guard_ms": 40,
    "flow_clock": "confirmed_dispatch",
    "max_waste_committed_before_dispatch_tons": WASTE_STOP_TONS,
}

_HORIZON = MATCHING_CONTRACT["local_calendar_seconds"]
_REPORT = MATCHING_CONTRACT["report_period_seconds"]


@dataclass(frozen=True)
class MatchChoice(shared.Choice):
    route: tuple[int, int, int, int] | None = None
    assigned_tons: float = 0.0
    need_time: float | None = None
    loss_tons: float | None = None
    oversupply_tons: float = 0.0
    deadhead_seconds: float | None = None
    plan_version: int | None = None
    plan_error: str | None = None
    forecast_quality: tuple = ()
    replan_reason: str | None = None
    matching_kind: str | None = None
    provisional_trucks: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class _Offer:
    truck: int
    route: tuple[int, int, int, int]
    ready: float
    tons: float
    empty: float
    extra_empty: float

    @property
    def site(self):
        return self.route[2]


def _need(slots, assigned):
    """Invert a piecewise cumulative dispatch curve on its original clock."""
    remaining = max(0.0, assigned)
    for slot in slots:
        if slot.tons <= MASS_EPS:
            continue
        if remaining < slot.tons - MASS_EPS:
            fraction = remaining / slot.tons
            return slot.start + fraction * (slot.stop - slot.start), slot
        remaining = max(0.0, remaining - slot.tons)
    return inf, None


def _overlap(start, stop, left, right):
    return max(0.0, min(stop, right) - max(start, left))


def _period_tons(slots, start, stop):
    return sum(slot.tons * _overlap(start, stop, slot.start, slot.stop) / (slot.stop - slot.start)
               for slot in slots if slot.start < stop and slot.stop > start)


def _is_due(offer, demand):
    due, slot = demand
    return (slot is not None and offer.ready >= slot.start
            and int(offer.ready // WINDOW_SECONDS) == int(due // WINDOW_SECONDS))


def _remaining_window_tons(slots, assigned, ready):
    """A load covering the next quota can spill beyond the dispatch window."""
    stop = (int(ready // WINDOW_SECONDS) + 1) * WINDOW_SECONDS
    return max(0.0, _period_tons(slots, 0.0, stop) - assigned)


def _site_rates(targets, assigned):
    """Only effective, unfilled flow contributes a shovel-idleness cost."""
    schedules = [[] for _ in range(4)]
    for route, slots in targets.items():
        due, slot = _need(slots, assigned.get(route, 0.0))
        if slot is None:
            continue
        for part in slots:
            if part.stop > due and part.tons > MASS_EPS:
                schedules[route[2]].append((max(due, part.start), part.stop,
                                            part.tons / (part.stop - part.start)))
    return schedules


class _Calendar(NominalState):
    """Local insertion comparison for wait, shovel availability and unload service."""

    def __init__(self, snapshot):
        self.rates = None
        self.shovel_rates = ()
        self.loss_tons = 0.0
        super().__init__(snapshot)

    def _advance_time(self, until):
        if self.rates is not None:
            waiting = (*self._load_wait, *self._dump_wait, self._capacity_wait)
            self.loss_tons += (until - self.now) * sum(self.rates[tid] for line in waiting for tid in line)
            for site, rates in enumerate(self.shovel_rates):
                load = self._loads[site]
                idle = (load.capacity - self._load_active[site]) / load.capacity
                self.loss_tons += idle * sum(rate * _overlap(self.now, until, first, last)
                                             for first, last, rate in rates)
        super()._advance_time(until)

    def replay(self, offers, rates, shovel_rates, stop, check):
        branch = self.clone()
        branch.rates, branch.shovel_rates = rates, shovel_rates
        pending = {offer.truck: offer for offer in offers}
        while branch.now < stop:
            check()
            while branch._events and branch._events[0][0] == branch.now:
                _, kind, resource, tid = heappop(branch._events)
                branch._process_event(kind, resource, tid)

            for tid, offer in tuple(pending.items()):
                if offer.ready > branch.now or branch._trips[tid] is not None:
                    continue
                if branch._nodes[tid] != offer.route[1]:
                    return inf
                load = branch._loads[offer.site]
                branch._trips[tid] = _Trip(offer.tons, load.ore, load.grade if load.ore else 0.0)
                branch._push(branch.now + offer.empty, _ARRIVE, offer.site, tid)
                del pending[tid]

            if branch._events and branch._events[0][0] == branch.now:
                continue
            branch._start_services()
            next_event = branch._events[0][0] if branch._events else inf
            next_truck = min((offer.ready for tid, offer in pending.items()
                              if branch._trips[tid] is None and offer.ready > branch.now), default=inf)
            next_time = min(next_event, next_truck, branch._capacity_retry(), stop)
            if next_time <= branch.now:
                raise NominalModelError("matching_calendar_time_stalled")
            branch._advance_time(next_time)
        if pending:
            return inf
        return branch.loss_tons


def _truck_offers(state, truck_types):
    options = shared._options(state)
    tid = options.truck_id
    return tuple(_Offer(tid, (truck_types[tid], state._nodes[tid], option.site,
                              int(not state._loads[option.site].ore)),
                        state.now, option.tons, option.empty_seconds, option.empty_seconds - options.shortest_empty)
                 for option in options.choices)


def _cohort(state, truck_types, size, end, check):
    forecast = state.clone()
    result = {}
    while not forecast.finished and forecast.now < end:
        check()
        tid = forecast.current_truck
        offers = _truck_offers(forecast, truck_types)
        if offers:
            result[tid] = offers
        if len(result) == size:
            break
        # Preserve visible initial release dependencies as in the shared forecast.
        forecast._initial_dispatched.add(tid)
        forecast.current_truck = None
        forecast._candidates = ()
        forecast.advance(check_deadline=check)
    return result


def _reference_rates(state, truck_types):
    rates = {}
    for kind in set(truck_types):
        truck = state._trucks[truck_types.index(kind)]
        cycles = [state._empty_times[node][load.node] + load.service(truck.capacity)
                  + state._loaded_times[load.node][int(not load.ore)]
                  + (truck.ore_service if load.ore else truck.waste_service)
                  for node in {0, 1, *(item for item in state._nodes if item is not None)}
                  for load in state._loads if load.available and state._dump_available[int(not load.ore)]]
        shortest = min((cycle for cycle in cycles if cycle > 0.0), default=inf)
        rates[kind] = truck.capacity / shortest
    return tuple(rates[kind] for kind in truck_types)


def _reserve(offer, state, tentative):
    available = state._available_tons[offer.site] - sum(
        other.tons for other in tentative if other.site == offer.site
    )
    if offer.tons > available + MASS_TOL:
        return False
    if offer.route[3]:
        waste = state._waste_committed()
        waste_offers = (other for other in (*tentative, offer) if other.route[3])
        for other in sorted(waste_offers, key=lambda row: (row.ready, row.truck != state.current_truck, row.truck)):
            if waste >= WASTE_STOP_TONS - MASS_EPS:
                return False
            waste += other.tons
    return True


class FlowMatchingPolicy:
    """One shift of rolling path-flow LP and deterministic route matching."""

    def __init__(self, *, planning_ms=800, replan_seconds=900, match_trucks=3,
                 soft_replan_seconds=120, solve_plan=solve_plan):
        limits = (planning_ms, replan_seconds, soft_replan_seconds)
        if (any(not isfinite(value) or value <= 0 for value in limits)
                or type(match_trucks) is not int or not 1 <= match_trucks <= 3):
            raise ValueError("invalid path-flow planning limits")
        self.planning_ms = planning_ms
        self.replan_seconds = replan_seconds
        self.match_trucks = match_trucks
        self.soft_replan_seconds = soft_replan_seconds
        self._solve_plan = solve_plan
        self.plan = None
        self._signature = None
        self._version = 0
        self._assigned = defaultdict(float)
        self._excess = []
        self._last_time = 0.0
        self._last_attempt = -inf
        self._pending = None
        self._last_committed = None
        self._soft_replan = False
        self._planned = defaultdict(float)
        self._actual = defaultdict(float)
        self._counts = Counter()
        self._solver_outcomes = Counter()
        self._forecast = {}

    def _record_until(self, now):
        start = self._last_time
        if self.plan is not None:
            while start < now:
                period = floor(start / _REPORT)
                stop = min(now, (period + 1) * _REPORT)
                for route, slots in self.plan.routes.items():
                    planned = _period_tons(slots, start, stop)
                    if planned > 0.0:
                        self._planned[period, route] += planned
                start = stop
        self._last_time = now

    def _replan_reason(self, state, signature):
        if self.plan is not None and signature != self._signature:
            return "window_or_resource_change"
        if state.now - self._last_attempt < self.soft_replan_seconds:
            return None
        if self.plan is None:
            return "initial_plan"
        if state.now - self.plan.now >= self.replan_seconds:
            return "periodic"
        if self._soft_replan:
            return "forced_oversupply"
        expected = getattr(self.plan, "expected_window_ore_at", None)
        if expected is not None:
            discrepancy = abs(expected(state.now) - state._fifo_q[state.window_index])
            if discrepancy >= min(truck.capacity for truck in state._trucks):
                return "crusher_flow_deviation"
        return None

    def _install(self, plan, signature):
        self.plan = plan
        self._signature = signature
        self._version += 1
        self._assigned.clear()
        self._excess.clear()
        self._soft_replan = False
        for row in plan.grade_windows:
            self._forecast.setdefault(row.window, (plan.now, row.ore_tons, row.metal_tons))

    def _choice(self, offer, status, metadata, *, kind=None, due=None, loss=None, provisional=()):
        slots = self.plan.routes.get(offer.route, ()) if self.plan is not None else ()
        outstanding = _remaining_window_tons(slots, self._assigned[offer.route], offer.ready)
        overflow = max(0.0, offer.tons - outstanding) if self.plan is not None else 0.0
        return MatchChoice(offer.site, status, **metadata, route=offer.route, assigned_tons=offer.tons,
                           need_time=due, loss_tons=loss, oversupply_tons=overflow,
                           deadhead_seconds=offer.empty if isfinite(offer.empty) else None,
                           plan_version=self._version if self.plan is not None else None,
                           matching_kind=kind, provisional_trucks=tuple(item.truck for item in provisional))

    def _select(self, state, truck_types, root, backup, deadline, metadata):
        def check():
            if perf_counter() >= deadline:
                raise shared._Deadline

        targets = self.plan.routes
        stop = min(SHIFT_END, state.now + _HORIZON)
        cohort = _cohort(state, truck_types, self.match_trucks, stop, check)
        rates = _reference_rates(state, truck_types)
        virtual = dict(self._assigned)
        tentative = []
        scored = 0

        def compare(options):
            nonlocal scored
            shovel = _site_rates(targets, virtual)
            baseline = state.replay(tentative, rates, shovel, stop, check)
            return _minimum_loss(options, tentative, state, rates, shovel, stop, baseline, check)

        while cohort:
            demands = {route: _need(slots, virtual.get(route, 0.0)) for route, slots in targets.items()}
            eligible = defaultdict(list)
            for offers in cohort.values():
                for offer in offers:
                    if _is_due(offer, demands.get(offer.route, (inf, None))) and _reserve(offer, state, tentative):
                        eligible[offer.route].append(offer)
            for route in sorted(eligible, key=lambda key: (demands[key][0], key)):
                scored += len(eligible[route])
                loss, choice = compare(eligible[route])
                if isfinite(loss):
                    break
            else:
                break
            tentative.append(choice)
            virtual[route] = virtual.get(route, 0.0) + choice.tons
            del cohort[choice.truck]
            if choice.truck == state.current_truck:
                metadata.update(matched_trucks=len(tentative), assignments_scored=scored)
                return self._choice(choice, "planned", metadata, kind="need_time", due=demands[route][0],
                                    loss=loss, provisional=tentative)

        supported = []
        for offer in root:
            check()
            if not _reserve(offer, state, tentative):
                continue
            slots = targets.get(offer.route, ())
            outstanding = _remaining_window_tons(slots, virtual.get(offer.route, 0.0), offer.ready)
            excess = max(0.0, offer.tons - outstanding)
            if excess <= MASS_EPS or self.plan.accepts_extra(offer.route, offer.ready, excess, prior=self._excess):
                supported.append(offer)
        # The current live action takes priority over hypothetical allocations.
        if not supported:
            tentative.clear()
            virtual = dict(self._assigned)
        options = supported or root
        scored += len(options)
        loss, choice = compare(options)
        metadata.update(matched_trucks=len(tentative) + 1, assignments_scored=scored)
        return self._choice(choice if isfinite(loss) else backup, "planned", metadata,
                            kind="overflow" if supported else "forced_oversupply",
                            loss=loss if isfinite(loss) else None, provisional=tentative)

    def choose(self, observation, mask):
        obs, legal = np.asarray(observation), np.asarray(mask)
        if (self._pending is not None or obs.shape != (1519,) or obs.dtype != np.float32
                or not np.isfinite(obs).all() or obs[17] or legal.shape != (4,) or not legal.any()
                or not np.array_equal(obs[19:23], legal) or not np.isin(legal, [0, 1]).all()):
            raise ValueError("fresh 1519D decision and a matching nonempty mask required")
        started = perf_counter()
        deadline = started + self.planning_ms / 1000.0
        actions = tuple(map(int, np.flatnonzero(legal)))
        fallback = shared._fallback_action(obs.astype(np.float64), actions)
        trucks = obs[27:1107].reshape(9, 120)
        truck_types = tuple(map(int, np.argmax(trucks[:, 1:4], axis=1)))
        tid = int(np.argmax(trucks[:, 0]))
        decoded_origin = int(np.argmax(trucks[tid, 32:41]))
        dump = int(not obs[1107 + 47 * fallback + 1])
        current_time = (int(round(obs[2] * 4)) + 1 - float(obs[3])) * WINDOW_SECONDS
        now = max(0.0, current_time)
        if now + .005 < self._last_time:
            raise ValueError("physical dispatch time regressed")
        self._record_until(max(now, self._last_time))
        backup = _Offer(tid, (truck_types[tid], decoded_origin, fallback, dump), now,
                        float(obs[23 + fallback]) * 193.0, inf, 0.0)
        metadata = {}
        if len(actions) == 1:
            choice = self._choice(backup, "single_legal", metadata, kind="single_legal")
        else:
            choice = self._plan_and_match(obs, actions, truck_types, backup, started, deadline, metadata)
        self._pending = (choice, now)
        return choice

    def _plan_and_match(self, obs, actions, truck_types, backup, started, deadline, metadata):
        try:
            state = _Calendar.from_observation(obs)
            root = _truck_offers(state, truck_types)
            if tuple(offer.site for offer in root) != actions:
                raise NominalModelError("nominal root routes differ from the live mask")
            backup = next(offer for offer in root if offer.site == backup.site)
            signature = shared._signature(state, obs)
            reason = self._replan_reason(state, signature)
            if reason is not None:
                metadata["replan_reason"] = reason
                self._last_attempt = state.now
                reserve = min(MATCHING_CONTRACT["planner_guard_ms"] / 1000.0, (deadline - started) / 5)
                try:
                    plan = self._solve_plan(obs, state, deadline=deadline - reserve)
                except FlowPlanError as error:
                    metadata.update(plan_error=str(error), lp_status="failed",
                                    timed_out=perf_counter() >= deadline - reserve)
                    if self._signature != signature:
                        self.plan = None
                else:
                    self._install(plan, signature)
                    self._solver_outcomes[plan.status] += 1
                    metadata.update(
                        replanned=True,
                        lp_status="feasible_limit" if plan.status == "feasible_stage_one" else plan.status,
                        flow_shortfall_ore_tons=plan.ore_shortfall_tons,
                        flow_shortfall_waste_tons=plan.waste_shortfall_tons,
                        flow_grade_slack_tons=sum(lo + hi for lo, hi in plan.grade_slack_tons),
                        forecast_quality=tuple((row.window, row.ore_tons, row.metal_tons)
                                               for row in plan.grade_windows),
                    )
            if self.plan is None:
                metadata["fallback_reason"] = "flow_plan_unavailable"
                return self._choice(backup, "fallback", metadata, kind="no_plan")
            return self._select(state, truck_types, root, backup, deadline, metadata)
        except shared._Deadline:
            metadata.update(timed_out=True, fallback_reason="matching_deadline")
            return self._choice(backup, "fallback", metadata, kind="deadline")
        except NominalModelError as error:
            metadata.update(fallback_reason=str(error))
            return self._choice(backup, "fallback", metadata, kind="nominal_error")

    def commit(self, choice):
        """Post-echo allocation ledger. A duplicate echo cannot increase tonnes."""
        if choice is not None and choice is self._last_committed:
            return
        if self._pending is None or choice is not self._pending[0]:
            raise ValueError("cannot commit an unissued path allocation")
        _, now = self._pending
        self._pending = None
        self._last_committed = choice
        self._counts[choice.matching_kind or choice.kind] += 1
        self._actual[floor(now / _REPORT), choice.route] += choice.assigned_tons
        if self.plan is not None:
            self._assigned[choice.route] += choice.assigned_tons
            if choice.oversupply_tons > MASS_TOL:
                self._excess.append((choice.route, now, choice.oversupply_tons))
                self._soft_replan = True

    def summary(self):
        """Aggregate the actually active versions and confirmed route flow."""
        if self._pending is not None:
            raise ValueError("cannot settle a pending path allocation")
        self._record_until(SHIFT_END)
        rows = []
        for period, route in sorted(self._planned.keys() | self._actual.keys()):
            planned, actual = self._planned[period, route], self._actual[period, route]
            rows.append(dict(period=period, route=route, planned_tons=planned,
                             confirmed_tons=actual, signed_error_tons=actual - planned))
        return dict(plan_versions=self._version, branch_counts=dict(self._counts),
                    accepted_solver_outcomes=dict(self._solver_outcomes),
                    route_periods=rows, absolute_flow_error_tons=sum(abs(row["signed_error_tons"]) for row in rows),
                    first_forecast_by_window={window: dict(at_seconds=values[0], ore_tons=values[1],
                                                          metal_tons=values[2])
                                              for window, values in self._forecast.items()})


def _minimum_loss(options, tentative, state, rates, shovel, stop, baseline, check):
    ranked = []
    for offer in options:
        check()
        inserted = state.replay((*tentative, offer), rates, shovel, stop, check)
        increment = inserted - baseline if isfinite(baseline) else inf
        loss = rates[offer.truck] * offer.extra_empty + increment
        ranked.append((loss, offer.ready, offer.truck, offer.route, offer))
    best = min(ranked, key=lambda row: row[:4])
    return best[0], best[-1]
