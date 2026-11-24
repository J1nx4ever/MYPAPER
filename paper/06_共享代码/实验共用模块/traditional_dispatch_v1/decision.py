"""Bounded, per-request dispatch on the common nominal model."""
from dataclasses import dataclass
from math import inf, isfinite
from time import perf_counter

import numpy as np

from grade_commitment_ppo_v1 import features as physical_features
from single_critic_task_query_v1.targets import (
    ORE_TARGET_TONS, ORE_WEIGHT, WASTE_TARGET_TONS, WASTE_WEIGHT,
    WAIT_REFERENCE_SECONDS, WAIT_WEIGHT, grade_cost,
)
from .nominal import NominalModelError, NominalState, SHIFT_END, WINDOW_SECONDS


POLICIES = ('unload_greedy', 'fifo_greedy', 'rolling')
TAIL_LEAD_SECONDS = 900.0


@dataclass
class Decision:
    action: int
    kind: str
    fallback_reason: str | None = None
    timed_out: bool = False
    completed_candidates: int = 0
    cross_window_candidates: int = 0
    search_nodes: int = 0
    completed_plans: int = 0
    multi_step_plans: int = 0
    max_scored_depth: int = 0
    invalid_branches: int = 0
    tree_complete: bool = False
    initial_cost: float | None = None
    selected_cost: float | None = None


class _PlanningDeadline(Exception):
    pass


def _grade(tons, metal):
    return grade_cost(None, empty=True) if tons == 0 else grade_cost(metal / tons)


def _shortfall(ore_tons, waste_tons, progress):
    return (ORE_WEIGHT * max(0.0, progress - ore_tons / ORE_TARGET_TONS)
            + WASTE_WEIGHT * max(0.0, progress - waste_tons / WASTE_TARGET_TONS))


def _cost(inputs, *, unload):
    q, m = (inputs.unload_q, inputs.unload_m) if unload else (inputs.fifo_q, inputs.fifo_m)
    return (sum(_grade(q[window], m[window]) for window in range(inputs.first_window, 4))
            + _shortfall(inputs.ore_tons, inputs.waste_tons, 1.0)
            + WAIT_WEIGHT * inputs.wait_seconds / WAIT_REFERENCE_SECONDS)


def _fallback_action(obs, actions):
    """Compare visible nominal travel and queues, with station index as the last tie-break."""
    ego_ids = np.flatnonzero(obs[27:1107:120])
    if len(ego_ids) != 1:
        return actions[0]
    ego_id = int(ego_ids[0])
    offset = 27 + 120 * ego_id
    node = physical_features._onehot_index(obs[offset + 32:offset + 41])
    if node is None:
        return actions[0]
    cfg = physical_features.FEATURE_CONFIG
    empty = physical_features._road_times(obs, cfg, loaded=False)
    loaded = physical_features._road_times(obs, cfg, loaded=True)

    def score(action):
        load = obs[1107 + 47 * action:1154 + 47 * action]
        load_node = cfg['load_node_indices'][action]
        dump_id = 0 if load[1] else 1
        dump = obs[1295 + 48 * dump_id:1343 + 48 * dump_id]
        seconds = (empty[node, load_node] + loaded[load_node, cfg['dump_node_indices'][dump_id]]
                   + physical_features._queue_seconds(obs, load, load=True, truck_id=ego_id, cfg=cfg)
                   + physical_features._queue_seconds(obs, dump, load=False, truck_id=ego_id, cfg=cfg))
        return (seconds if isfinite(seconds) else inf, action)

    return min(actions, key=score)


def _tail_action(state, check):
    choices = state.candidates()
    if len(choices) == 1:
        return choices[0].action
    q, m, ore, waste = state.tail_context()
    progress = min(1.0, (state.now + TAIL_LEAD_SECONDS) / SHIFT_END)
    best = None
    for candidate in choices:
        check()
        timing = state.candidate_timing(candidate.action, check_deadline=check)
        if candidate.ore:
            tons, metal = candidate.tons, candidate.tons * candidate.grade
            production = _shortfall(ore + tons, waste, progress)
        else:
            tons, metal = 0.0, 0.0
            production = _shortfall(ore, waste + candidate.tons, progress)
        cost = (_grade(q + tons, m + metal) + production
                + WAIT_WEIGHT * timing.wait_seconds / WAIT_REFERENCE_SECONDS)
        rank = (cost, timing.unload_complete, candidate.action)
        if best is None or rank < best:
            best = rank
    if best is None:
        raise NominalModelError("no candidate at a nominal dispatch event")
    return best[2]


def _finish(state, prefix, cache, depth, horizon, check):
    """Complete a trajectory and reuse it for any identical explicit prefix."""
    if prefix in cache:
        return cache[prefix], False
    rollout = state.clone()
    equivalent_prefixes = [prefix]
    while not rollout.finished:
        check()
        action = _tail_action(rollout, check)
        if rollout.now < horizon and len(prefix) < depth:
            prefix += (action,)
            equivalent_prefixes.append(prefix)
        rollout.apply_action(action)
        rollout.advance(check_deadline=check)
    check()
    inputs = rollout.score_inputs()
    for equivalent in equivalent_prefixes:
        cache.setdefault(equivalent, inputs)
    return inputs, True


def _extensions(frontier, horizon):
    eligible = [(sequence, state, state.candidates()) for sequence, state in frontier
                if not state.finished and state.now < horizon]
    # Interleave parents when the wall-clock budget stops a depth early.
    for rank in range(4):
        for sequence, state, choices in eligible:
            if rank < len(choices):
                yield sequence + (choices[rank].action,), state


def _search(result, frontier, cache, depth, horizon, check):
    best_sequence = None
    for _ in range(1, depth):
        next_frontier = []
        for sequence, parent in _extensions(frontier, horizon):
            check()
            result.search_nodes += 1
            try:
                state = parent.clone()
                state.apply_action(sequence[-1])
                state.advance(check_deadline=check)
                inputs, simulated = _finish(state, sequence, cache, depth, horizon, check)
                cost = _cost(inputs, unload=False)
            except NominalModelError:
                result.invalid_branches += 1
                continue
            result.completed_plans += int(simulated)
            result.multi_step_plans += 1
            result.max_scored_depth = max(result.max_scored_depth, len(sequence))
            if (cost < result.selected_cost
                    or (cost == result.selected_cost and best_sequence is not None
                        and sequence < best_sequence)):
                best_sequence = sequence
                result.action = sequence[0]
                result.selected_cost = cost
            next_frontier.append((sequence, state))
        frontier = next_frontier
        if not frontier:
            break
    result.tree_complete = result.invalid_branches == 0


def choose(policy, observation, mask, *, planning_ms=800, depth=3):
    """Return one legal action and request-local diagnostics for aggregation."""
    if policy not in POLICIES:
        raise ValueError(f"unknown traditional dispatch policy: {policy}")
    if type(depth) is not int or not 1 <= depth <= 3 or not isfinite(planning_ms) or planning_ms <= 0:
        raise ValueError("expected a positive planning limit and depth 1 through 3")
    obs = np.asarray(observation)
    legal = np.asarray(mask)
    if (obs.dtype != np.float32 or obs.shape != (1519,) or not np.isfinite(obs).all()
            or obs[17] or legal.shape != (4,) or not np.isin(legal, [0, 1]).all()
            or not np.array_equal(obs[19:23], legal) or not legal.any()):
        raise ValueError("actionable 1519D observation and matching nonempty legal mask required")
    actions = tuple(map(int, np.flatnonzero(legal)))
    if len(actions) == 1:
        return Decision(actions[0], "single_legal")

    deadline = perf_counter() + planning_ms / 1000

    def check():
        if perf_counter() >= deadline:
            raise _PlanningDeadline

    result = Decision(_fallback_action(obs.astype(np.float64), actions), "fallback")
    cache = {}
    frontier = []
    costs = []
    search_depth = depth if policy == "rolling" else 1
    try:
        check()
        initial = NominalState.from_observation(obs)
        if tuple(candidate.action for candidate in initial.candidates()) != actions:
            raise NominalModelError("nominal root candidates differ from the real legal mask")
        horizon = min((initial.window_index + 2) * WINDOW_SECONDS, SHIFT_END)
        for action in actions:
            check()
            branch = initial.clone()
            branch.apply_action(action)
            branch.advance(check_deadline=check)
            inputs, _ = _finish(branch, (action,), cache, search_depth, horizon, check)
            costs.append((_cost(inputs, unload=policy == "unload_greedy"), action))
            result.completed_candidates += 1
            result.completed_plans += 1
            result.max_scored_depth = 1
            result.cross_window_candidates += int(inputs.root_cross_window)
            frontier.append(((action,), branch))
        result.initial_cost, result.action = min(costs)
        result.selected_cost = result.initial_cost
        result.kind = "planned"
        _search(result, frontier, cache, search_depth, horizon, check)
    except NominalModelError as error:
        if result.kind == "fallback":
            result.fallback_reason = str(error)
        else:
            result.invalid_branches += 1
    except _PlanningDeadline:
        result.timed_out = True
        if result.kind == "fallback":
            result.fallback_reason = "incomplete_one_step_at_deadline"
    return result
