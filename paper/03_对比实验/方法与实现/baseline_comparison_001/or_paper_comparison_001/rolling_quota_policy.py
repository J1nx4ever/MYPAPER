"""Rolling integer dispatch quotas with the four-stage Policy4 tracker."""
from collections import Counter
from dataclasses import dataclass
from math import isfinite
from time import perf_counter

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, vstack

from single_critic_task_query_v1.targets import ORE_TARGET_TONS, WASTE_TARGET_TONS
from traditional_dispatch_v1.decision import _fallback_action
from traditional_dispatch_v2.nominal import (
    MASS_EPS, MASS_TOL, SHIFT_END, WASTE_STOP_TONS, WINDOW_SECONDS,
    NominalModelError, NominalState,
)

if __package__:
    from ..dispatch_flow_matching_001 import flow_matching_plan as flow
else:
    import flow_matching_plan as flow


PLAN_LEAD_SECONDS = 900
DISPATCH_PERIOD_SECONDS = 1800
WASTE_OVERRUN_TONS = 193
BUSINESS_GRADE = (.26, .28)
PLANNING_GRADE = (.266, .274)
SOLVER_GAP = .005
SOLVER_RETURN_MARGIN_SECONDS = .015
TARGET_GRADE = .27
POLICY4_INITIAL_QUOTAS = (
    ((5, 0, 0, 8), (8, 0, 3, 2), (1, 2, 7, 0)),
    ((0, 3, 0, 7), (5, 1, 5, 3), (1, 7, 3, 0)),
    ((2, 0, 1, 4), (5, 5, 1, 3), (1, 4, 6, 1)),
    ((0, 3, 0, 5), (6, 1, 4, 4), (1, 6, 4, 0)),
)


def plan_window(now, physical_window):
    return min(3, max(physical_window, int((now + PLAN_LEAD_SECONDS) // WINDOW_SECONDS)))


def _dispatch_periods(now):
    edges = {now, SHIFT_END, *np.arange(now, SHIFT_END, DISPATCH_PERIOD_SECONDS)}
    for window in range(1, 4):
        boundary = window * WINDOW_SECONDS - PLAN_LEAD_SECONDS
        if now < boundary:
            edges.add(boundary)
    edges = sorted(edges)
    return tuple(zip(edges[:-1], edges[1:]))


@dataclass(frozen=True, slots=True)
class QuotaPlan:
    counts: np.ndarray
    status: str
    gap: float | None
    stages: int
    ore_shortfall_tons: float
    waste_shortfall_tons: float
    grade_slack_tons: float
    feed_slack_tons: float
    timed_out: bool
    max_row_residual: float


def _columns(replay, ready, truck_types, periods, warehouse, deadline):
    calendars = [flow._blocked(intervals, cap) for intervals, cap in zip(
        replay.resource_intervals,
        [shovel.capacity for shovel in replay._loads] + list(replay._dump_capacity),
        strict=True,
    )]
    waste_open = replay.waste_committed_at_origin < WASTE_STOP_TONS - MASS_EPS
    for kind in sorted(set(truck_types)):
        releases = sorted((at, tid, node) for tid, node, at in ready if truck_types[tid] == kind)
        if not releases:
            continue
        release, tid, observed_origin = releases[0]
        truck = replay._trucks[tid]
        for site, shovel in enumerate(replay._loads):
            dump = int(not shovel.ore)
            payload = min(truck.capacity, replay._available_tons[site])
            if (not shovel.available or not replay._dump_available[dump] or payload <= MASS_EPS
                    or (dump == 1 and not waste_open)):
                continue
            loaded = replay._loaded_times[shovel.node][dump]
            loading = shovel.service(payload)
            unloading = truck.waste_service if dump else truck.ore_service
            for start, stop in periods:
                flow._require_time(deadline)
                dispatch_start = max(start, release)
                # The first segment uses a visible vehicle origin. Subsequent
                # aggregate trips use the same-commodity steady cycle.
                origin = observed_origin if start == replay.origin else dump
                empty = replay._empty_times[origin][shovel.node]
                if not isfinite(empty + loaded) or dispatch_start + empty + loading + loaded + unloading >= SHIFT_END:
                    continue
                if dispatch_start >= stop:
                    continue
                offsets = tuple(flow._offsets(
                    dispatch_start + (stop - dispatch_start) * (idx + .5) / flow.SUBSEGMENTS,
                    empty, loaded, loading, unloading, calendars[site], calendars[4 + dump], dump,
                    warehouse, payload,
                ) for idx in range(flow.SUBSEGMENTS))
                if any(offset is None for offset in offsets):
                    continue
                wait = sum((offset[0] - empty) + (offset[4] - unloading - offset[1] - loaded)
                           for offset in offsets) / flow.SUBSEGMENTS
                yield flow._Column(
                    (kind, origin, site, dump), (dispatch_start, stop), payload,
                    shovel.grade if shovel.ore else 0., offsets,
                ), max(0., wait)


def _problem(replay, ready, truck_types, periods, warehouse, committed, deadline):
    pairs = tuple(_columns(replay, ready, truck_types, periods, warehouse, deadline))
    columns = tuple(column for column, _ in pairs)
    windows = range(committed.first_window, 4)
    count = len(columns)
    wall = np.asarray(sorted({replay.origin, SHIFT_END,
                              *np.arange(replay.origin, SHIFT_END, PLAN_LEAD_SECONDS),
                              *(w * WINDOW_SECONDS for w in windows if w * WINDOW_SECONDS > replay.origin),
                              *(t for segment in periods for t in segment)}))
    durations = np.diff(wall)
    width = count + 2 + 5 * len(windows)
    ub = flow._Rows(width)
    q = np.asarray([column.q for column in columns])
    grade = np.asarray([column.grade for column in columns])
    types, _, sites, dumps = np.asarray([column.key for column in columns], dtype=int).reshape(-1, 4).T
    if count:
        _, completed, reserved, crushed, work, cycle = flow._curves(columns, wall, warehouse)
    else:
        completed = reserved = crushed = np.zeros((len(wall), 0))
        work, cycle = np.zeros((3, len(durations), 0)), np.zeros(0)
    ore_cols, waste_cols = np.flatnonzero(dumps == 0), np.flatnonzero(dumps == 1)
    for site, supply in enumerate(replay._available_tons):
        indices = np.flatnonzero(sites == site)
        ub.add(indices, q[indices], supply)
    ub.add(waste_cols, q[waste_cols],
           max(0., WASTE_STOP_TONS + WASTE_OVERRUN_TONS - replay.waste_committed_at_origin))
    for resource, intervals in enumerate(replay.resource_intervals):
        shovel = resource < 4
        indices = np.flatnonzero(sites == resource if shovel else dumps == resource - 4)
        cap = replay._loads[resource].capacity if shovel else replay._dump_capacity[resource - 4]
        remaining = durations * cap
        for start, stop in intervals:
            remaining -= np.maximum(0., np.minimum(stop, wall[1:]) - np.maximum(start, wall[:-1]))
        ub.add_block(indices, work[1 if shovel else 2][:, indices], np.maximum(0., remaining))
    for kind in sorted(set(truck_types)):
        indices = np.flatnonzero(types == kind)
        available = np.zeros(len(durations))
        for tid, _, at in ready:
            if truck_types[tid] == kind:
                available += np.maximum(0., wall[1:] - np.maximum(wall[:-1], at))
        ub.add_block(indices, work[0][:, indices], available)

    stock, reserved_existing, crushed_existing = warehouse.balances(wall)
    ub.add_block(ore_cols, (reserved - crushed)[:, ore_cols] * q[ore_cols],
                 np.maximum(0., warehouse.capacity - stock - reserved_existing))
    ub.add_block(ore_cols, np.diff(crushed[:, ore_cols], axis=0) * q[ore_cols],
                 np.maximum(0., warehouse.rate * durations - np.diff(crushed_existing)))
    ore = np.asarray([
        crushed[np.searchsorted(wall, (w + 1) * WINDOW_SECONDS)]
        - crushed[np.searchsorted(wall, max(w * WINDOW_SECONDS, replay.origin))]
        for w in windows
    ]) * q
    metal = ore * grade
    waste = completed[-1] * (dumps == 1) * q
    trips = np.arange(count)
    ub.add(np.r_[trips, count], np.r_[-ore.sum(axis=0), -1.], committed.ore_tons - ORE_TARGET_TONS)
    ub.add(np.r_[trips, count + 1], np.r_[-waste, -1.], committed.waste_tons - WASTE_TARGET_TONS)
    first_cost, second_cost = np.zeros(width), np.zeros(width)
    first_cost[count:count + 2] = (1 / ORE_TARGET_TONS, 1 / WASTE_TARGET_TONS)
    second_cost[:count] = .2 * np.asarray([wait for _, wait in pairs]) / SHIFT_END + 1.e-9 * cycle
    min_feed = min(truck.capacity for truck in replay._trucks)
    for index, window in enumerate(windows):
        q0, m0 = committed.fifo_q[window], committed.fifo_m[window]
        base = count + 2 + 5 * index
        for band, shift in ((BUSINESS_GRADE, 0), (PLANNING_GRADE, 2)):
            low, high = band
            ub.add(np.r_[trips, base + shift], np.r_[low * ore[index] - metal[index], -1.], m0 - low * q0)
            ub.add(np.r_[trips, base + shift + 1], np.r_[metal[index] - high * ore[index], -1.], high * q0 - m0)
        ub.add(np.r_[trips, base + 4], np.r_[-ore[index], -1.], q0 - min_feed)
        first_cost[base:base + 2] = 1. / (.01 * ORE_TARGET_TONS / 4)
        first_cost[base + 4] = 1. / min_feed
        second_cost[base + 2:base + 4] = 1. / (.006 * ORE_TARGET_TONS / 4)
    return columns, *ub.matrix(), first_cost, second_cost


def _milp(objective, matrix, limit, integer, deadline):
    flow._require_time(deadline)
    return milp(
        objective, integrality=integer, bounds=Bounds(0., np.inf),
        constraints=LinearConstraint(matrix, -np.inf, limit),
        options={'time_limit': max(.001, deadline - perf_counter()), 'mip_rel_gap': SOLVER_GAP},
    )


def _verified(result, matrix, limit, count):
    values = flow._verified(result, matrix, limit, None, None)
    if values is None or np.any(np.abs(values[:count] - np.rint(values[:count])) > 1.e-5):
        return None
    values[:count] = np.rint(values[:count])
    return values if np.all(matrix @ values <= limit + MASS_TOL) else None


def solve_quota(observation, state, *, deadline):
    deadline -= SOLVER_RETURN_MARGIN_SECONDS
    replay = flow._Replay.from_observation(observation)
    if (replay.now, replay.current_truck, replay.candidates()) != (state.now, state.current_truck, state.candidates()):
        raise flow.FlowPlanError('quota_state_mismatch')
    replay._move(check_deadline=lambda: flow._require_time(deadline), dispatch=False)
    committed = replay.score_inputs()
    truck_types = tuple(int(t) for t in observation[27:1107].reshape(9, 120)[:, 1:4].argmax(axis=1))
    columns, matrix, limit, first, second = _problem(
        replay, flow._ready_origins(replay), truck_types, _dispatch_periods(replay.origin),
        flow._Warehouse(replay), committed, deadline,
    )
    count = len(columns)
    integer = np.zeros(len(first), dtype=np.uint8)
    integer[:count] = 1
    # Leave time for the second objective without discarding a valid first incumbent.
    now = perf_counter()
    primary = _milp(first, matrix, limit, integer, now + .7 * (deadline - now))
    incumbent = _verified(primary, matrix, limit, count)
    if incumbent is None:
        reason = 'quota_milp_limit_without_incumbent' if primary.status == 1 else 'quota_milp_invalid_incumbent'
        raise flow.FlowPlanError(reason)
    result, stages = primary, 1
    gaps = [getattr(primary, 'mip_gap', None)]
    accepted_matrix, accepted_limit = matrix, limit
    timed_out = primary.status == 1
    if primary.status == 0 and deadline - perf_counter() >= .025:
        second_matrix = vstack((matrix, csr_matrix(first.reshape(1, -1))), format='csr')
        second_limit = np.r_[limit, first @ incumbent + 1.e-7]
        try:
            candidate = _milp(second, second_matrix, second_limit, integer, deadline)
        except flow.FlowPlanError:
            timed_out = True
        else:
            timed_out = candidate.status == 1
            confirmed = _verified(candidate, second_matrix, second_limit, count)
            if confirmed is not None:
                incumbent, result, stages = confirmed, candidate, 2
                gaps.append(getattr(candidate, 'mip_gap', None))
                accepted_matrix, accepted_limit = second_matrix, second_limit

    counts = np.zeros((4, 3, 4), dtype=int)
    for column, trips in zip(columns, incumbent[:count], strict=True):
        kind, _, site, _ = column.key
        now = column.depart[0]
        counts[plan_window(now, int(now // WINDOW_SECONDS)), kind, site] += int(trips)
    slack = incumbent[count + 2:].reshape(-1, 5)
    gap = max((float(gap) for gap in gaps if gap is not None and isfinite(gap)), default=None)
    status = ('optimal' if stages == 2 and result.status == 0 and (gap is None or gap <= SOLVER_GAP)
              else 'feasible_limit')
    return QuotaPlan(counts, status, gap, stages,
                     float(incumbent[count]), float(incumbent[count + 1]),
                     float(slack[:, :2].sum()), float(slack[:, 4].sum()), timed_out,
                     max(0., float(np.max(accepted_matrix @ incumbent - accepted_limit))))


def _fifo_signal(state, observation, window):
    next_window = window > state._first_window
    current_capacity = max(0., (state._first_window + 1) * WINDOW_SECONDS - state.now) * state._crusher_rate
    remaining = WINDOW_SECONDS * state._crusher_rate if next_window else current_capacity
    tons = 0. if next_window else state._fifo_q[state._first_window]
    metal = 0. if next_window else state._fifo_m[state._first_window]
    skip = current_capacity if next_window else 0.
    for batch in state._fifo:
        discarded = min(skip, batch.tons)
        skip -= discarded
        usable = min(remaining, batch.tons - discarded)
        remaining -= usable
        tons += usable
        metal += usable * batch.grade
    weight = .65 if next_window else .35
    for row in observation[27:1107].reshape(9, 120):
        if row[17] and row[18] and row[19] * 193. > MASS_EPS:
            mass = float(row[19]) * 193.
            grade = float(row[22])
        elif row[20] * 193. > MASS_EPS and row[26:30].any():
            site = int(row[26:30].argmax())
            if not state._loads[site].ore:
                continue
            mass = float(row[20]) * 193.
            grade = state._loads[site].grade
        else:
            continue
        tons += weight * mass
        metal += weight * mass * grade
    return tons, metal


def _policy4(state, observation, quota, used, window, actions):
    rows = observation[1107:1295].reshape(4, 47)
    active = (np.count_nonzero(rows[:, 20:38], axis=1)
              + np.rint(rows[:, 16]).astype(int) + np.rint(rows[:, 17] * 9.).astype(int))
    tons, metal = _fifo_signal(state, observation, window)
    grade = metal / tons if tons > MASS_EPS else TARGET_GRADE
    pressure = min(1., tons / 1000.) * float(np.clip((TARGET_GRADE - grade) / .025, -1., 1.))
    offers = {candidate.action: candidate for candidate in state.candidates()}
    truck = observation[27:1107].reshape(9, 120)[state.current_truck]
    kind = int(truck[1:4].argmax())
    for stage, targets, assigned in ((1, quota[kind], used[kind]), (2, quota.sum(axis=0), used.sum(axis=0))):
        candidates = {site: (targets[site] - assigned[site]) / targets[site] - .15 * active[site]
                      + (1.5 * pressure * (offers[site].grade - TARGET_GRADE) / .115 if offers[site].ore else 0.)
                      for site in actions if targets[site] > assigned[site]}
        if candidates:
            return max(candidates, key=candidates.get), stage
    candidates = {site: -abs((metal + offers[site].tons * offers[site].grade) /
                            (tons + offers[site].tons) - TARGET_GRADE) / .01 - .10 * active[site]
                  for site in actions if offers[site].ore}
    if candidates:
        return max(candidates, key=candidates.get), 3
    return min(actions, key=lambda site: active[site]), 4


@dataclass(frozen=True, slots=True)
class QuotaChoice:
    action: int
    kind: str
    plan_window: int
    truck_type: int
    stage: int
    fallback_reason: str | None = None
    timed_out: bool = False
    replanned: bool = False
    lp_status: str | None = None
    matched_trucks: int = 0
    assignments_scored: int = 0
    flow_shortfall_ore_tons: float | None = None
    flow_shortfall_waste_tons: float | None = None
    flow_grade_slack_tons: float | None = None
    mip_gap: float | None = None
    mip_stages: int = 0
    mip_max_row_residual: float | None = None
    replan_reason: str | None = None


class RollingQuotaPolicy:
    """Keep a current quota table and settle each real dispatch after acknowledgement."""

    def __init__(self, *, planning_ms=800, replan_seconds=900, rolling=True, solve_plan=solve_quota,
                 bootstrap_replan=False):
        if not (isfinite(planning_ms) and planning_ms > 0 and isfinite(replan_seconds) and replan_seconds > 0):
            raise ValueError('invalid quota planning limits')
        self.planning_ms = planning_ms
        self.replan_seconds = replan_seconds
        self.rolling = rolling
        self.solve_plan = solve_plan
        # Cold snapshots can test forecast and action plumbing with a zero
        # ledger. Live runs use the historical table for their first dispatch.
        self.bootstrap_replan = bootstrap_replan
        self.quota = np.asarray(POLICY4_INITIAL_QUOTAS, dtype=int)
        self.assigned = np.zeros_like(self.quota)
        self._pending = None
        self._last_attempt = None
        self._last_window = None
        self._last_time = None
        self._stages = Counter()
        self._plans = Counter()

    def choose(self, observation, mask):
        if self._pending is not None:
            raise RuntimeError('previous quota choice is unacknowledged')
        deadline = perf_counter() + self.planning_ms / 1000.
        obs = np.asarray(observation, dtype=np.float32)
        actions = tuple(index for index, legal in enumerate(mask) if legal)
        if obs.shape != (1519,) or len(mask) != 4 or actions != tuple(np.flatnonzero(obs[19:23])) or not actions:
            raise ValueError('quota mask differs from the decision observation')
        truck = obs[27:1107].reshape(9, 120)
        current = np.flatnonzero(truck[:, 0])
        if len(current) != 1:
            raise ValueError('quota requires exactly one observed decision truck')
        kind = int(truck[current[0], 1:4].argmax())
        physical = int(round(float(obs[2]) * 4))
        now = (physical + 1 - float(obs[3])) * WINDOW_SECONDS
        window = plan_window(now, physical)
        if self._last_time is not None and now + .005 < self._last_time:
            raise ValueError('quota decisions must arrive in causal time order')
        self._last_time = now
        try:
            state = NominalState.from_observation(obs)
        except NominalModelError:
            choice = QuotaChoice(_fallback_action(obs.astype(float), actions), 'fallback', window, kind, 0,
                                 fallback_reason='nominal_decision_invalid')
            self._pending = choice
            return choice
        window = plan_window(state.now, state._first_window)
        plan, replan_reason, status, timed_out = None, None, None, False
        first = self._last_attempt is None
        if first:
            self._last_attempt = state.now
        should_replan = (self.bootstrap_replan if first else
                         window != self._last_window or state.now - self._last_attempt >= self.replan_seconds)
        if self.rolling and should_replan:
            self._last_attempt = state.now
            try:
                plan = self.solve_plan(obs, state, deadline=deadline)
            except (flow.FlowPlanError, NominalModelError) as error:
                replan_reason = str(error)
                status = 'failed'
                timed_out = 'deadline' in replan_reason or 'limit' in replan_reason
            else:
                self.quota[window:] = self.assigned[window:] + plan.counts[window:]
                status = plan.status
                timed_out = plan.timed_out
        self._last_window = window
        action, stage = _policy4(state, obs, self.quota[window], self.assigned[window], window, actions)
        label = 'single_legal' if len(actions) == 1 else 'planned'
        choice = QuotaChoice(
            action, label, window, kind, stage,
            timed_out=timed_out, replanned=plan is not None, lp_status=status,
            flow_shortfall_ore_tons=plan.ore_shortfall_tons if plan else None,
            flow_shortfall_waste_tons=plan.waste_shortfall_tons if plan else None,
            flow_grade_slack_tons=plan.grade_slack_tons if plan else None,
            mip_gap=plan.gap if plan else None, mip_stages=plan.stages if plan else 0,
            mip_max_row_residual=plan.max_row_residual if plan else None,
            replan_reason=replan_reason,
        )
        self._pending = choice
        return choice

    def commit(self, choice):
        if choice is not self._pending or choice is None:
            raise RuntimeError('quota acknowledgement differs from the selected decision')
        self.assigned[choice.plan_window, choice.truck_type, choice.action] += 1
        self._stages[str(choice.stage)] += 1
        if choice.lp_status:
            self._plans[choice.lp_status] += 1
        self._pending = None

    def summary(self):
        return dict(assigned_trips=self.assigned.tolist(), quota_trips=self.quota.tolist(),
                    stages=dict(self._stages), solver_status=dict(self._plans),
                    mip_accepted=self._plans['optimal'] + self._plans['feasible_limit'],
                    committed_actions=sum(self._stages.values()))
