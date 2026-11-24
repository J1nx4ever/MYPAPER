"""Prioritize business feasibility, blend margin and truck time in that order."""
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack

if __package__:
    from .. import flow_matching_plan as base
else:
    import flow_matching_plan as base


INNER_LOWER = .266
INNER_UPPER = .274


@dataclass(frozen=True, slots=True)
class GradeFlowPlan(base.FlowPlan):
    planning_grade_slack_tons: tuple[tuple[float, float], ...]
    planning_stages: int


def _margin_rows(base_matrix, rhs, eq, shortfall, truck_time, ore, metal, committed):
    """Add inner grade slacks while leaving all physical and business rows intact."""
    window_count, flow_count = ore.shape
    old_width = base_matrix.shape[1]
    slack_count = 2 * window_count
    rows = base._Rows(old_width + slack_count)
    indices = np.r_[np.arange(flow_count), old_width]
    for offset in range(window_count):
        window = committed.first_window + offset
        q0, m0 = committed.fifo_q[window], committed.fifo_m[window]
        indices[-1] = old_width + 2 * offset
        rows.add(indices, np.r_[INNER_LOWER * ore[offset] - metal[offset], -1.], m0 - INNER_LOWER * q0)
        indices[-1] += 1
        rows.add(indices, np.r_[metal[offset] - INNER_UPPER * ore[offset], -1.], INNER_UPPER * q0 - m0)
    inner, inner_rhs = rows.matrix()
    ub = vstack((hstack((base_matrix, csr_matrix((base_matrix.shape[0], slack_count)))), inner), format='csr')
    if eq is not None:
        eq = hstack((eq, csr_matrix((eq.shape[0], slack_count))), format='csr')
    objectives = (np.pad(shortfall, (0, slack_count)),
                  np.r_[np.zeros(old_width), np.ones(slack_count)],
                  np.pad(truck_time, (0, slack_count)))
    return ub, np.r_[rhs, inner_rhs], eq, objectives


def _lexicographic(ub, rhs, eq, equals, objectives, deadline):
    x = None
    stages = 0
    for objective in objectives:
        if x is not None and deadline - perf_counter() < .020:
            break
        fit = base._solve(objective, ub, rhs, eq, equals, deadline)
        candidate = base._verified(fit, ub, rhs, eq, equals)
        if candidate is None:
            if x is None:
                raise base.FlowPlanError(f'grade_flow_lp_status_{fit.status}')
            break
        x = candidate
        if fit.status != 0:
            break
        stages += 1
        ub = vstack((ub, csr_matrix(objective.reshape(1, -1))), format='csr')
        rhs = np.r_[rhs, float(objective @ x) + 1e-7]
    return x, stages


def solve_grade_plan(observation, state, *, deadline):
    """Use the frozen route and FIFO kernels with an additional nominal margin."""
    base._require_time(deadline)
    try:
        replay = base._Replay.from_observation(observation)
        if (replay.now != state.now or replay.current_truck != state.current_truck
                or replay.candidates() != state.candidates()):
            raise base.FlowPlanError('grade_flow_observation_mismatch')
        replay._move(check_deadline=lambda: base._require_time(deadline), dispatch=False)
    except base.NominalModelError as error:
        raise base.FlowPlanError(str(error)) from error
    committed = replay.score_inputs()
    types = np.asarray(observation)[27:1107].reshape(9, 120)[:, 1:4].argmax(axis=1).tolist()
    ready = base._ready_origins(replay)
    warehouse = base._Warehouse(replay)
    periods = base._slots(replay.origin)
    columns = tuple(base._columns(replay, types, warehouse, ready, periods, deadline))
    clock = base._time_grid(replay, periods, ready)
    base._require_time(deadline)
    (physical, bounds), (eq, equals), lack, fleet, guard_rows, ore, metal, waste, old, crushed = (
        base._problem(replay, columns, types, ready, warehouse, clock, committed)
    )
    ub, rhs, eq, objectives = _margin_rows(physical, bounds, eq, lack, fleet, ore, metal, committed)
    x, stages = _lexicographic(ub, rhs, eq, equals, objectives, deadline)
    count = len(columns)
    dispatched = x[:count]
    routes, guard = {}, {}
    for index, (column, tons) in enumerate(zip(columns, dispatched, strict=True)):
        guard.setdefault(column.key, []).append((*column.depart, index))
        if tons > base.MASS_EPS:
            routes.setdefault(column.key, []).append(base.FlowSlot(*column.depart, float(tons)))
    visible = tuple(base.WindowProjection(w, committed.fifo_q[w], committed.fifo_m[w])
                    for w in range(committed.first_window, 4))
    windows = tuple(base.WindowProjection(w.window, float(w.ore_tons + q), float(w.metal_tons + m))
                    for w, q, m in zip(visible, ore @ dispatched, metal @ dispatched, strict=True))
    outer_slack = x[count + 2:count + 2 + 3 * len(windows)].reshape(-1, 3)
    inner_slack = x[physical.shape[1]:].reshape(-1, 2)
    residual_max = max(0., float(np.max(ub @ x - rhs)),
                       float(np.max(np.abs(eq @ x - equals))) if eq is not None else 0.)
    guard_remaining = np.maximum(0., bounds[:guard_rows] - physical[:guard_rows] @ x[:physical.shape[1]])
    return GradeFlowPlan(
        now=replay.origin,
        routes={key: tuple(slots) for key, slots in routes.items()},
        committed_windows=visible,
        grade_windows=windows,
        ore_tons=float(committed.ore_tons + (ore @ dispatched).sum()),
        waste_tons=float(committed.waste_tons + waste @ dispatched),
        ore_shortfall_tons=float(x[count]),
        waste_shortfall_tons=float(x[count + 1]),
        grade_slack_tons=tuple(tuple(map(float, pairs)) for pairs in outer_slack[:, :2]),
        feed_slack_tons=tuple(map(float, outer_slack[:, 2])),
        status='optimal' if stages == len(objectives) else 'feasible_limit',
        residual_max=residual_max,
        _clock=clock,
        _ore_curve=old + crushed @ dispatched,
        _start_window_ore=replay.initial_processed[committed.first_window],
        _guard_columns={key: tuple(rows) for key, rows in guard.items()},
        _guard_matrix=physical[:guard_rows, :count],
        _guard_residual=guard_remaining,
        planning_grade_slack_tons=tuple(tuple(map(float, pairs)) for pairs in inner_slack),
        planning_stages=stages,
    )
