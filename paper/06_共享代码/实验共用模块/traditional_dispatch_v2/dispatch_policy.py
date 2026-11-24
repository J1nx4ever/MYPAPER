"""Online target-flow ledger and finite DISPATCH-style truck/shovel matching."""
from collections import defaultdict
from dataclasses import dataclass, replace
from heapq import heapify, heapreplace
from itertools import pairwise, product
from math import inf, isfinite
from time import perf_counter

import numpy as np

from traditional_dispatch_v1.decision import _fallback_action
from .flow_plan import FlowPlanError, solve_flow_plan
from .nominal import (
    MASS_EPS, MASS_TOL, NominalModelError, NominalState, SHIFT_END, WASTE_STOP_TONS, _ARRIVE, _FINISH,
)


class _Deadline(Exception):
    """Return the completed incumbent before the RPC response deadline."""


@dataclass(frozen=True)
class Choice:
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


@dataclass(frozen=True)
class _Option:
    site: int
    tons: float
    arrival: float
    empty_seconds: float
    cycle_seconds: float
    loading_seconds: float


@dataclass(frozen=True)
class _TruckOptions:
    truck_id: int
    choices: tuple[_Option, ...]
    shortest_empty: float


def _options(state):
    truck_id = state.current_truck
    truck, node = state._trucks[truck_id], state._nodes[truck_id]
    choices = []
    for offer in state.candidates():
        station = state._loads[offer.action]
        dump = int(not offer.ore)
        empty = state._empty_times[node][station.node]
        loaded = state._loaded_times[station.node][dump]
        loading = station.service(offer.tons)
        unloading = truck.ore_service if offer.ore else truck.waste_service
        if isfinite(empty + loaded) and state._dump_available[dump] and offer.tons > 0:
            choices.append(_Option(offer.action, offer.tons, state.now + empty,
                                   empty, empty+loading+loaded+unloading, loading))
    return _TruckOptions(truck_id, tuple(choices), min((o.empty_seconds for o in choices), default=inf))


def _ready_trucks(state, size, check):
    """Project outstanding commitments, holding the forecast trucks for matching."""
    # NominalState currently exposes the next dispatch rather than a fleet
    # release list. These fields belong to the byte-pinned nominal model.
    forecast = state.clone()
    trucks = []
    while not forecast.finished and len(trucks) < size:
        check()
        truck = _options(forecast)
        if truck.choices:
            trucks.append(truck)
        forecast._initial_dispatched.add(forecast.current_truck)
        forecast.current_truck = None
        forecast._candidates = ()
        if len(trucks) < size:
            forecast.advance(check_deadline=check)
    return trucks


def _shovel_calendars(state):
    """Combine presently occupied resources and visible pre-existing arrivals."""
    servers = [[time for time, kind, site, _ in sorted(state._events) if site == j and kind == _FINISH]
               for j in range(4)]
    bookings = [[] for _ in range(4)]
    for site, shovel in enumerate(state._loads):
        missing_completions = state._load_active[site] - len(servers[site])
        if missing_completions < 0:
            raise NominalModelError('nominal shovel completion accounting differs')
        servers[site] += [SHIFT_END] * missing_completions
        servers[site] += [state.now] * (shovel.capacity - state._load_active[site])
        for rank, truck in enumerate(state._load_wait[site]):
            bookings[site].append((state.now, 0, rank, shovel.service(state._trips[truck].tons), None))
    for time, kind, site, truck in state._events:
        if kind == _ARRIVE and site < 4:
            duration = state._loads[site].service(state._trips[truck].tons)
            bookings[site].append((time, 1, truck, duration, None))
    return servers, bookings


@dataclass(frozen=True)
class _Target:
    start: float
    stop: float
    tons: float


def _targets(plan):
    targets = [[] for _ in range(4)]
    for period in plan.periods:
        for site in range(4):
            added_rate = defaultdict(float)
            for tons_row, start_row in zip(period.tons_by_type_site, period.dispatch_start_by_type_site, strict=True):
                start, tons = start_row[site], tons_row[site]
                if start is not None and tons > MASS_EPS:
                    added_rate[start] += tons / (period.end - start)
            rate = 0.0
            for start, end in pairwise([*sorted(added_rate), period.end]):
                rate += added_rate[start]
                targets[site].append(_Target(start, end, rate * (end - start)))
    return tuple(tuple(site) for site in targets)


def _need_time(targets, assigned):
    """Inverse cumulative target tonnage. A zero-rate segment has no slot."""
    for target in targets:
        if target.tons > 1e-6:
            if assigned < target.tons - 1e-6:
                rate = target.tons / (target.stop - target.start)
                return target.start + assigned/rate, rate
            assigned -= target.tons
    return inf, 0.0


def _loss(options, trucks, targets, committed, servers, bookings):
    queue = [events.copy() for events in bookings]
    assigned = committed.copy()
    needs = []
    for i, option in enumerate(options):
        site = option.site
        needs.append(_need_time(targets[site], assigned[site]))
        assigned[site] += option.tons
        queue[site].append((option.arrival, 2, trucks[i].truck_id, option.loading_seconds, i))
    total = 0.0
    for site, arrivals in enumerate(queue):
        available = servers[site].copy()
        heapify(available)
        for arrival, _, _, duration, choice in sorted(arrivals):
            free = available[0]
            start = max(free, arrival)
            heapreplace(available, start + duration)
            if choice is None:
                continue
            option = options[choice]
            due, feed = needs[choice]
            truck_rate = option.tons / option.cycle_seconds
            total += truck_rate * (start - arrival + option.empty_seconds - trucks[choice].shortest_empty)
            if isfinite(due):
                # The LP defines tonnes by dispatch time. Empty travel and
                # shovel waiting are already charged in the first term.
                dispatch_time = option.arrival - option.empty_seconds
                total += min(option.tons, feed * abs(dispatch_time - due))
            else:
                total += option.tons
    return total


def _reserve(options, available, loads, waste):
    remaining = available.copy()
    reserved = []
    for index, option in enumerate(options):
        site = option.site
        load = loads[site]
        if not load.ore:
            if waste >= WASTE_STOP_TONS - 1e-6:
                return None
        # The actual root keeps its measured load. A later virtual truck may
        # legally take the residual stock left by earlier reservations.
        tons = option.tons if index == 0 else min(option.tons, remaining[site])
        if tons <= MASS_EPS or tons > remaining[site] + MASS_TOL:
            return None
        if not load.ore:
            waste += tons
        remaining[site] = max(0.0, remaining[site] - tons)
        if tons != option.tons:
            duration = load.service(tons)
            option = replace(option, tons=tons, loading_seconds=duration,
                             cycle_seconds=option.cycle_seconds - option.loading_seconds + duration)
        reserved.append(option)
    return tuple(reserved)


def _signature(state, observation):
    # Route and resource changes invalidate the old flow plan. The legal mask
    # itself is vehicle-specific and is verified at every real request.
    return (state.window_index, state._waste_committed() >= WASTE_STOP_TONS - 1e-6,
            tuple((load.available, load.ore, stock > MASS_EPS)
                  for load, stock in zip(state._loads, state._available_tons, strict=True)), state._dump_available,
            observation[1439:1519].tobytes())


class DispatchPolicy:
    """One shift of a deterministic flow planner with an accepted-tonnage ledger."""

    def __init__(self, *, planning_ms=800, replan_seconds=900, match_trucks=3):
        if (not isfinite(planning_ms) or planning_ms <= 0 or not isfinite(replan_seconds)
                or replan_seconds <= 0 or type(match_trucks) is not int or not 1 <= match_trucks <= 3):
            raise ValueError('invalid DISPATCH planning limits')
        self.planning_ms = planning_ms
        self.replan_seconds = replan_seconds
        self.match_trucks = match_trucks
        self.plan = None
        self._plan_signature = None
        self._assigned = [0.0] * 4
        self._pending = None
        self._last_time = -inf

    def _select(self, obs, actions, deadline):
        fallback = _fallback_action(obs.astype(np.float64), actions)
        lp_status = None
        replanned = False
        bounds = {}
        best = None
        scored = 0
        matched = 0

        def check():
            if perf_counter() >= deadline:
                raise _Deadline

        try:
            check()
            state = NominalState.from_observation(obs)
            signature = _signature(state, obs)
            if (self.plan is None or state.now - self.plan.now >= self.replan_seconds
                    or signature != self._plan_signature):
                lp_status = 'failed'
                plan = solve_flow_plan(obs, state, time_limit_seconds=max(0.001, deadline-perf_counter()))
                if plan.status not in ('optimal', 'feasible_limit') or not plan.periods:
                    raise FlowPlanError('no feasible flow plan')
                self.plan = plan
                self._plan_signature = signature
                self._assigned = [0.0] * 4
                replanned, lp_status = True, plan.status
                bounds = dict(flow_shortfall_ore_tons=plan.ore_shortfall_tons,
                              flow_shortfall_waste_tons=plan.waste_shortfall_tons,
                              flow_grade_slack_tons=sum(low+high for low, high in plan.grade_slack_tons))
            check()
            root = _options(state)
            if root.truck_id != state.current_truck or tuple(o.site for o in root.choices) != actions:
                raise NominalModelError('nominal root candidate routes differ from the legal mask')
            targets = _targets(self.plan)
            servers, bookings = _shovel_calendars(state)
            # Evaluate all immediate choices before expanding the lookahead.
            first = min((_loss((option,), (root,), targets, self._assigned, servers, bookings), option.site)
                        for option in root.choices)
            best = (first[0], (first[1],))
            matched, scored = 1, len(root.choices)
            if self.match_trucks > 1:
                trucks = _ready_trucks(state, self.match_trucks, check)
                for size in range(len(trucks), 1, -1):
                    cohort = trucks[:size]
                    incumbent = None
                    waste = state._waste_committed()
                    for candidates in product(*(truck.choices for truck in cohort)):
                        check()
                        allocated = _reserve(candidates, state._available_tons, state._loads, waste)
                        if allocated is None:
                            continue
                        cost = _loss(allocated, cohort, targets, self._assigned, servers, bookings)
                        rank = (cost, tuple(option.site for option in allocated))
                        if incumbent is None or rank < incumbent:
                            incumbent = rank
                        scored += 1
                    if incumbent is not None:
                        best, matched = incumbent, size
                        break
        except _Deadline:
            if best is None:
                return Choice(fallback, 'fallback', 'deadline_without_plan', True, replanned, lp_status, **bounds)
            return Choice(best[1][0], 'planned', timed_out=True, replanned=replanned, lp_status=lp_status,
                          matched_trucks=matched, assignments_scored=scored, **bounds)
        except (FlowPlanError, NominalModelError) as error:
            return Choice(fallback, 'fallback', str(error), replanned=replanned, lp_status=lp_status, **bounds)
        return Choice(best[1][0], 'planned', replanned=replanned, lp_status=lp_status,
                      matched_trucks=matched, assignments_scored=scored, **bounds)

    def choose(self, observation, mask):
        """Select a legal root action and prepare the corresponding allocation."""
        obs, legal = np.asarray(observation), np.asarray(mask)
        if (self._pending is not None or obs.shape != (1519,) or obs.dtype != np.float32
                or not np.isfinite(obs).all() or obs[17] or legal.shape != (4,)
                or not np.array_equal(obs[19:23], legal) or not np.isin(legal, [0, 1]).all() or not legal.any()):
            raise ValueError('fresh 1519D decision and a nonempty matching mask required')
        now = float(obs[0])*SHIFT_END
        if now < self._last_time:
            raise ValueError('physical dispatch time regressed')
        self._last_time = now
        actions = tuple(map(int, np.flatnonzero(legal)))
        if len(actions) == 1:
            choice = Choice(actions[0], 'single_legal')
        else:
            choice = self._select(obs, actions, perf_counter() + self.planning_ms/1000)
        self._pending = (choice, float(obs[23+choice.action])*193.0)
        return choice

    def commit(self, choice):
        """Account for an action only after the actual RPC accepts it."""
        if self._pending is None or choice is not self._pending[0]:
            raise ValueError('cannot commit an unissued or already accepted assignment')
        _, tons = self._pending
        if self.plan is not None:
            self._assigned[choice.action] += tons
        self._pending = None
