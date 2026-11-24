"""Grade-only labels and sign checks; no network, optimizer, or execution entry.

The values below are costs, not RPC rewards.  Waiting and production never enter
these helpers.  All boundaries are journal sequence boundaries, not timestamps.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

import numpy as np


REWARD_VERSION = "grade-only-alp-clip-linear-v1"


def _finite(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def grade_cost(grade: float | None, *, empty: bool = False,
               raw_grade_reward: float | None = None) -> float:
    """ALP's current symmetric .26--.28 contract, explicitly handling empty Q.

    An empty window has undefined grade and cost 2.  If a raw ALP reward is
    supplied it must agree with the physical formula, rather than just be finite.
    """
    if not isinstance(empty, (bool, np.bool_)):
        raise ValueError("empty must be boolean")
    if empty:
        if grade is not None:
            raise ValueError("empty window grade must be undefined (None)")
        cost = 2.0
    else:
        if grade is None:
            raise ValueError("nonempty window requires grade")
        physical_grade = _finite(grade, "grade")
        if not 0.0 <= physical_grade <= 1.0:
            raise ValueError("grade must lie in [0,1]")
        cost = min(abs(physical_grade - 0.27) / 0.01, 2.0)
    if raw_grade_reward is not None:
        raw = _finite(raw_grade_reward, "raw_grade_reward")
        if not -1.0 <= raw <= 1.0 or not math.isclose(
                1.0 - raw, cost, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("raw grade reward disagrees with physical grade cost")
    return cost


@dataclass(frozen=True)
class GradeTargets:
    step_costs: tuple[float, ...]
    mc_returns: tuple[float, ...]
    closed_windows_before: tuple[int, ...]
    terminal_return: float = 0.0


def build_grade_targets(memory: Any) -> GradeTargets:
    """Create complete-shift MC labels from validated MemoryTape query markers.

    For query t, c[t] contains closures strictly after its DECISION and before
    its next callback marker.  A closure before a same-time DECISION is already
    paid.  MC includes each not-yet-closed window's *whole final cost*.
    """
    queries = tuple(memory.queries)
    settlements = tuple(memory.settlements)
    if not queries or len(settlements) != 4:
        raise ValueError("complete labels require decisions and exactly four windows")
    if sorted(s.window for s in settlements) != [0, 1, 2, 3]:
        raise ValueError("settlement windows must be unique 0..3")
    for settlement in settlements:
        expected = grade_cost(
            settlement.grade if settlement.grade_valid else None,
            empty=not settlement.grade_valid,
            raw_grade_reward=settlement.grade_reward_raw,
        )
        if not math.isclose(_finite(settlement.cost, "settlement cost"), expected,
                            rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("settlement cost is not the grade-only contract")
    for index, query in enumerate(queries):
        if query.current_seq >= query.next_seq:
            raise ValueError("query sequence boundary is not forward")
        if bool(query.terminal_next) != (index == len(queries) - 1):
            raise ValueError("only the final query may lead to true END")
        if index and queries[index - 1].next_seq != query.current_seq:
            raise ValueError("query sequence boundaries are discontinuous")
    if any(s.seq >= queries[-1].next_seq for s in settlements):
        raise ValueError("settlement must precede terminal marker")
    costs = tuple(float(sum(s.cost for s in settlements
                            if query.current_seq < s.seq < query.next_seq))
                  for query in queries)
    mc = tuple(float(sum(s.cost for s in settlements if s.seq > query.current_seq))
               for query in queries)
    reverse = 0.0
    for index in range(len(costs) - 1, -1, -1):
        reverse += costs[index]
        if not math.isclose(reverse, mc[index], rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("MC labels and callback cost ledger disagree")
    closed = tuple(sum(s.seq < q.current_seq for s in settlements) for q in queries)
    if any(y < -1e-9 or y > 2.0 * (4 - n) + 1e-9 for y, n in zip(mc, closed)):
        raise ValueError("future cost exceeds remaining-window bound")
    return GradeTargets(costs, mc, closed)


def _vector(values: Iterable[float], name: str) -> np.ndarray:
    result = np.asarray(tuple(values), dtype=np.float64)
    if result.ndim != 1 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite vector")
    return result


@dataclass(frozen=True)
class CostGAE:
    old_values: tuple[float, ...]
    advantages: tuple[float, ...]
    lambda_returns: tuple[float, ...]
    mc_returns: tuple[float, ...]


def cost_gae(costs: Iterable[float], old_values: Iterable[float], *,
             terminated: Iterable[bool], truncated: Iterable[bool],
             gae_lambda: float = 0.95) -> CostGAE:
    """Freeze pure cost GAE for one full shift, gamma=1 and true END bootstrap=0.

    Window closes are intentionally absent from reset inputs.  The caller must
    snapshot values before any Critic fitting; returned tuples cannot alias them.
    """
    c, values = _vector(costs, "costs"), _vector(old_values, "old_values")
    term, trunc = np.asarray(tuple(terminated)), np.asarray(tuple(truncated))
    count = len(c)
    if count == 0 or values.shape != (count + 1,):
        raise ValueError("old_values must contain N+1 values for a nonempty shift")
    if term.shape != (count,) or trunc.shape != (count,) or term.dtype.kind != "b" or trunc.dtype.kind != "b":
        raise ValueError("terminal/truncation flags must be N booleans")
    if not term[-1] or term[:-1].any() or trunc.any():
        raise ValueError("only complete, untruncated single shifts are accepted")
    if values[-1] != 0.0:
        raise ValueError("true END bootstrap value must be zero")
    if (c < 0).any():
        raise ValueError("costs must be nonnegative")
    lam = _finite(gae_lambda, "gae_lambda")
    if not 0 <= lam <= 1:
        raise ValueError("gae_lambda must lie in [0,1]")
    advantages = np.empty(count, dtype=np.float64)
    carry = 0.0
    for t in range(count - 1, -1, -1):
        carry = c[t] + values[t + 1] - values[t] + lam * carry
        advantages[t] = carry
    mc = np.cumsum(c[::-1])[::-1]
    return CostGAE(tuple(values), tuple(advantages),
                   tuple(advantages + values[:-1]), tuple(mc))


def cost_surrogate_loss(ratios: Iterable[float], advantages: Iterable[float], *,
                        clip_epsilon: float = 0.2, entropy: float = 0.0,
                        entropy_coefficient: float = 0.0) -> float:
    """Pure numerical cost-PPO sign reference (minimize MAX surrogate).

    This helper does not authorize or perform PPO.  Production assurance is a
    separate unsatisfied gate.  It is deliberately not wired to a learner.
    """
    ratio, advantage = _vector(ratios, "ratios"), _vector(advantages, "advantages")
    if ratio.size == 0 or ratio.shape != advantage.shape or (ratio <= 0).any():
        raise ValueError("ratios and advantages require matching nonempty vectors and positive ratios")
    epsilon = _finite(clip_epsilon, "clip_epsilon")
    ent = _finite(entropy, "entropy")
    eta = _finite(entropy_coefficient, "entropy_coefficient")
    if not 0 <= epsilon < 1 or ent < 0 or eta < 0:
        raise ValueError("invalid clip/entropy parameters")
    clipped = np.clip(ratio, 1 - epsilon, 1 + epsilon)
    return float(np.maximum(ratio * advantage, clipped * advantage).mean() - eta * ent)
