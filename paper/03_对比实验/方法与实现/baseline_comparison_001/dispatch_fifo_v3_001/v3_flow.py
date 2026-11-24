"""Remaining-shift fluid targets with exact committed-backlog integration.

The LP proposes target tonnes. Its independent new-truck coefficients do
not enforce shared sub-window discharge rates, warehouse admission or FIFO.
The dispatcher tests discrete candidate actions in its FIFO rollout.
"""
import math
from time import perf_counter

import numpy as np
from scipy.optimize import linprog

from traditional_dispatch_v2 import flow_plan as v2
from traditional_dispatch_v2.flow_plan import FlowPlan, FlowPlanError, FlowPeriod


FLOW_PLAN_CONTRACT = dict(
    v2.FLOW_PLAN_CONTRACT,
    schema='dispatch-inspired-remaining-shift-flow-v3',
    new_ore_projection='exact_uniform_integral_over_visible_nominal_arrival_and_depletion_knots',
    projection_scope='independent_route_fluid_target_then_discrete_fifo_candidate_check',
    feasibility_check='validate_post_nonnegative_clip',
)


def _backlog_segments(replay):
    """Arrival intervals and the times the FIFO stock at their start would clear."""
    rate = replay._crusher_rate
    if not math.isfinite(rate) or rate <= 0:
        raise FlowPlanError('committed FIFO projection requires a positive finite crusher rate')
    start, stock = replay.origin, replay.initial_stock
    segments = []
    for end, incoming in replay.ore_arrivals:
        if end > start:
            segments.append((start, end, start + stock / rate))
            stock = max(0.0, stock - rate * (end - start))
        start, stock = end, stock + incoming
    segments.append((start, v2.SHIFT_END, start + stock / rate))
    return segments


def _ore_fractions(launch, windows, segments, rate):
    """Uniform dispatch-time expectation for one route considered independently."""
    depart_start, depart_stop = launch.departure
    unload_offset = launch.route.dump_end
    processing = launch.route.q / rate
    normalization = (depart_stop - depart_start) * processing
    contributions = [[] for _ in windows]
    for first, last, clear in segments:
        lower = max(depart_start, first - unload_offset)
        upper = min(depart_stop, last - unload_offset)
        if upper <= lower:
            continue
        split = min(upper, max(lower, clear - unload_offset))
        for index, target in enumerate(windows):
            if split > lower:
                duration = v2._overlap(clear, clear + processing, *target)
                contributions[index].append((split - lower) * duration)
            if upper > split:
                duration = v2._mean_overlap((split, upper), target, unload_offset, unload_offset + processing)
                contributions[index].append((upper - split) * duration)
    return [math.fsum(parts) / normalization for parts in contributions]


def _lp(replay, truck_types, windows, committed):
    # Preserve the V2 release, resource and objective conventions. Recompute
    # all rows that use the four ore or metal projection coefficients.
    launches, cost, matrix, limits, ore, metal, waste = v2._lp(replay, truck_types, windows, committed)
    segments = _backlog_segments(replay)
    for column, launch in enumerate(launches):
        if launch.route.ore:
            ore[:, column] = _ore_fractions(launch, windows, segments, replay._crusher_rate)
            metal[:, column] = launch.route.grade * ore[:, column]

    n = len(launches)
    # The shared LP layout ends with production rows, then capacity, lower
    # grade, upper grade and feed rows for each remaining quality window.
    tail = matrix[-(2 + 4 * len(windows)):]
    tail[0, :n] = -ore[:, :n].sum(axis=0)
    lower, upper = FLOW_PLAN_CONTRACT['grade_lower'], FLOW_PLAN_CONTRACT['grade_upper']
    for index, rows in enumerate(tail[2:].reshape(len(windows), 4, -1)):
        q, m = ore[index, :n], metal[index, :n]
        rows[:, :n] = q, lower * q - m, m - upper * q, -q
    return launches, cost, matrix, limits, ore, metal, waste


def _clipped_feasible(fit, matrix, limits):
    if (fit.status not in (0, 1) or fit.x is None or not np.isfinite(fit.x).all()
            or np.min(fit.x) < -v2.MASS_TOL):
        raise FlowPlanError(f'flow_lp_no_feasible_solution_status_{fit.status}')
    x = np.maximum(0.0, fit.x)
    if np.any(matrix @ x - limits > v2.MASS_TOL):
        raise FlowPlanError('flow_lp_infeasible_after_nonnegative_clip')
    return x


def solve_flow_plan(observation, state=None, *, time_limit_seconds=None):
    time_limit = FLOW_PLAN_CONTRACT['default_time_limit_seconds'] if time_limit_seconds is None else time_limit_seconds
    if not math.isfinite(time_limit) or time_limit <= 0:
        raise ValueError('flow LP requires a positive finite solver time limit')
    try:
        replay = v2._CommitmentReplay.from_observation(observation)
        if state is not None and (state.now != replay.now or state.current_truck != replay.current_truck
                                  or state.candidates() != replay.candidates()):
            raise FlowPlanError('nominal decision does not correspond to flow plan observation')
        first = replay.window_index
        windows = tuple((max(replay.now, w * v2.WINDOW_SECONDS), (w + 1) * v2.WINDOW_SECONDS)
                        for w in range(first, 4))
        replay._move(check_deadline=None, dispatch=False)
        committed = replay.score_inputs()
    except v2.NominalModelError as error:
        raise FlowPlanError(str(error)) from error

    truck_types = np.argmax(np.asarray(observation)[27:1107].reshape(9, 120)[:, 1:4], axis=1).tolist()
    launches, cost, matrix, limits, ore, metal, waste = _lp(replay, truck_types, windows, committed)
    started = perf_counter()
    fit = linprog(cost, A_ub=matrix, b_ub=limits, bounds=(0.0, None), method='highs',
                  options={'time_limit': time_limit})
    elapsed = perf_counter() - started
    x = _clipped_feasible(fit, matrix, limits)

    shape = (len(windows), FLOW_PLAN_CONTRACT['classes'], FLOW_PLAN_CONTRACT['load_sites'])
    tonnes = np.zeros(shape)
    starts = [[[None] * shape[2] for _ in range(shape[1])] for _ in range(shape[0])]
    for index, launch in enumerate(launches):
        w, typ, site = launch.period_idx, launch.route.truck_type, launch.route.site
        tonnes[w, typ, site] = x[index]
        starts[w][typ][site] = launch.departure[0]
    added_ore, added_metal = ore @ x, metal @ x
    n = len(launches)
    periods = []
    for i, (start, end) in enumerate(windows):
        w = first + i
        target = tuple(tuple(map(float, row)) for row in tonnes[i])
        rates = tuple(tuple(q * 3600.0 / (end - start) for q in row) for row in target)
        periods.append(FlowPeriod(w, start, end, target, rates,
                                  float(committed.fifo_q[w] + added_ore[i]),
                                  float(committed.fifo_m[w] + added_metal[i]), float(x[n + 4 + 3*i]),
                                  tuple(tuple(row) for row in starts[i])))
    slack = tuple((float(x[n + 2 + 3*i]), float(x[n + 3 + 3*i])) for i in range(len(windows)))
    return FlowPlan(replay.origin, tuple(periods), float(cost @ x), float(x[n]), float(x[n + 1]), slack,
                    'optimal' if fit.status == 0 else 'feasible_limit', float(committed.ore_tons),
                    float(committed.waste_tons), float(committed.ore_tons + added_ore.sum()),
                    float(committed.waste_tons + waste @ x), elapsed)
