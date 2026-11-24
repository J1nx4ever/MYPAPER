"""Classic BG128 baselines with a deterministic continuation work limit."""

from dataclasses import dataclass
from math import isfinite
from time import perf_counter

import numpy as np

from traditional_dispatch_v1.decision import (
    Decision as FrozenDecision, POLICIES, TAIL_LEAD_SECONDS, WAIT_REFERENCE_SECONDS, WAIT_WEIGHT,
    _cost, _fallback_action, _grade, _shortfall,
)
from .nominal import NominalModelError, NominalState, SHIFT_END, WINDOW_SECONDS


ROLLOUT_LIMIT = 20


@dataclass
class Decision(FrozenDecision):
    budget_exhausted: bool = False


class _PlanningDeadline(Exception):
    pass


def _tail_action(state, check):
    choices = state.candidates()
    if len(choices) == 1:
        return choices[0].action
    q, m, ore, waste = state.tail_context()
    progress = min(1.0, (state.now + TAIL_LEAD_SECONDS) / SHIFT_END)
    scores = []
    for candidate in choices:
        tons = candidate.tons if candidate.ore else 0.0
        waste_delta = 0.0 if candidate.ore else candidate.tons
        lower = _grade(q + tons, m + tons * candidate.grade) + _shortfall(
            ore + tons, waste + waste_delta, progress,
        )
        scores.append((lower, candidate.action))
    best = None
    for lower, action in sorted(scores):
        check()
        if best is not None and lower > best[0]:
            break  # The remaining waiting costs are nonnegative.
        timing = state.candidate_timing(action, check_deadline=check)
        rank = (lower + WAIT_WEIGHT * timing.wait_seconds / WAIT_REFERENCE_SECONDS,
                timing.unload_complete, action)
        if best is None or rank < best:
            best = rank
    if best is None:
        raise NominalModelError('no candidate at a nominal dispatch event')
    return best[2]


def _finish(state, prefix, cache, depth, horizon, check):
    if prefix in cache:
        return cache[prefix], False
    rollout = state.clone()
    equivalent = [prefix]
    while not rollout.finished:
        check()
        action = _tail_action(rollout, check)
        if rollout.now < horizon and len(prefix) < depth:
            prefix += (action,)
            equivalent.append(prefix)
        rollout.apply_action(action)
        rollout.advance(check_deadline=check)
    check()
    inputs = rollout.score_inputs()
    for key in equivalent:
        cache.setdefault(key, inputs)
    return inputs, True


def _extensions(frontier, horizon):
    eligible = [(prefix, state, state.candidates()) for prefix, state in frontier
                if not state.finished and state.now < horizon]
    for rank in range(4):
        for sequence, state, choices in eligible:
            if rank < len(choices):
                # When only part of a layer can be searched, start from
                # different station ranks across the root actions.
                offset = sequence[0] if len(sequence) > 1 else 0
                action = choices[(rank + offset) % len(choices)].action
                yield sequence + (action,), state


def _rank_frontier(scored):
    preferred, others = {}, []
    for _, sequence, state in sorted(scored, key=lambda row: (row[0], row[1])):
        if sequence[0] not in preferred:
            preferred[sequence[0]] = (sequence, state)
        else:
            others.append((sequence, state))
    return list(preferred.values()) + others


def _search(result, frontier, cache, depth, horizon, work_limit, check):
    best_sequence = None
    for _ in range(1, depth):
        next_frontier = []
        for sequence, parent in _extensions(frontier, horizon):
            if result.search_nodes == work_limit:
                result.budget_exhausted = True
                return
            check()
            result.search_nodes += 1
            try:
                branch = parent.clone()
                branch.apply_action(sequence[-1])
                branch.advance(check_deadline=check)
                inputs, simulated = _finish(branch, sequence, cache, depth, horizon, check)
                cost = _cost(inputs, unload=False)
            except NominalModelError:
                result.invalid_branches += 1
                continue
            result.completed_plans += int(simulated)
            result.multi_step_plans += 1
            result.max_scored_depth = max(result.max_scored_depth, len(sequence))
            if (cost < result.selected_cost
                    or (cost == result.selected_cost and best_sequence is not None and sequence < best_sequence)):
                best_sequence = sequence
                result.action = sequence[0]
                result.selected_cost = cost
            if not branch.finished and branch.now < horizon:
                next_frontier.append((cost, sequence, branch))
        frontier = _rank_frontier(next_frontier)
        if not frontier:
            break
    result.tree_complete = result.invalid_branches == 0


def choose(policy, observation, mask, *, planning_ms=800, depth=3, rollout_limit=ROLLOUT_LIMIT):
    """Evaluate all root candidates, then a fixed number of continuation nodes."""
    if policy not in POLICIES:
        raise ValueError(f'unknown traditional dispatch policy: {policy}')
    if (type(depth) is not int or not 1 <= depth <= 3 or not isfinite(planning_ms) or planning_ms <= 0
            or type(rollout_limit) is not int or rollout_limit < 1):
        raise ValueError('invalid depth, work limit or wall-clock safety limit')
    obs, legal = np.asarray(observation), np.asarray(mask)
    if (obs.dtype != np.float32 or obs.shape != (1519,) or not np.isfinite(obs).all()
            or obs[17] or legal.shape != (4,) or not np.isin(legal, [0, 1]).all()
            or not np.array_equal(obs[19:23], legal) or not legal.any()):
        raise ValueError('actionable 1519D observation and matching nonempty legal mask required')
    actions = tuple(map(int, np.flatnonzero(legal)))
    if len(actions) == 1:
        return Decision(actions[0], 'single_legal')

    deadline = perf_counter() + planning_ms / 1000.0

    def check():
        if perf_counter() >= deadline:
            raise _PlanningDeadline

    result = Decision(_fallback_action(obs.astype(np.float64), actions), 'fallback')
    incumbent = None
    try:
        check()
        initial = NominalState.from_observation(obs)
        if tuple(candidate.action for candidate in initial.candidates()) != actions:
            raise NominalModelError('nominal root candidates differ from the real legal mask')
        horizon = min((initial.window_index + 2) * WINDOW_SECONDS, SHIFT_END)
        search_depth = depth if policy == 'rolling' else 1
        cache, frontier, costs = {}, [], []
        for action in actions:
            check()
            branch = initial.clone()
            branch.apply_action(action)
            branch.advance(check_deadline=check)
            inputs, _ = _finish(branch, (action,), cache, search_depth, horizon, check)
            costs.append((_cost(inputs, unload=policy == 'unload_greedy'), action))
            result.completed_candidates += 1
            result.completed_plans += 1
            result.max_scored_depth = 1
            result.cross_window_candidates += int(inputs.root_cross_window)
            frontier.append(((action,), branch))
        result.initial_cost, incumbent = min(costs)
        result.selected_cost, result.action, result.kind = result.initial_cost, incumbent, 'planned'
        _search(result, frontier, cache, search_depth, horizon, rollout_limit, check)
    except NominalModelError as error:
        result.fallback_reason = str(error)
    except _PlanningDeadline:
        result.timed_out = True
        if incumbent is None:
            result.fallback_reason = 'incomplete_one_step_at_deadline'
        else:
            # A machine-dependent partial continuation never becomes the live
            # action. The whole root comparison remains available.
            result.action = incumbent
            result.selected_cost = result.initial_cost
    return result
