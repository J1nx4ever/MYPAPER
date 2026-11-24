"""Frozen single-Critic targets for simultaneous grade/production/wait learning.

These are nonnegative costs; the Actor minimizes cost advantages and the single
Critic predicts their *same combined* future return.  Raw RPC reward (including
its original 4/4 production term) is evidence, never an additional target term.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Iterable, Mapping

import numpy as np

from grade_event_memory_v1.targets import build_grade_targets, cost_gae


REWARD_VERSION = "grade-production-wait-clip-linear-8-8-v1"
CONTRACT_SCHEMA = "grade-production-wait-cost-contract-v1"
GRADE_WEIGHT = 1.0
ORE_WEIGHT = 8.0
WASTE_WEIGHT = 8.0
WAIT_WEIGHT = 0.1
ORE_TARGET_TONS = 16000.0
WASTE_TARGET_TONS = 4000.0
WAIT_REFERENCE_SECONDS = 28800.0


def _wait_weight(value: Any) -> float:
    weight = _nonnegative(value, "wait_weight")
    if weight not in (0.1, 0.15):
        raise ValueError("wait_weight must be exactly the declared 0.1 or 0.15 branch")
    return weight


def composite_contract(wait_weight: float = WAIT_WEIGHT) -> dict:
    """Return a fresh JSON-sized copy of the exact, fixed training contract."""
    weight = _wait_weight(wait_weight)
    return dict(schema=CONTRACT_SCHEMA, reward_version=REWARD_VERSION,
                convention="nonnegative_cost_minimization", gamma=1.0,
                grade_weight=GRADE_WEIGHT,
                grade_cost="min(abs(actual_window_grade-0.27)/0.01,2)",
                empty_window_cost=2.0, grade_settlement="each_window_once",
                ore_target_tons=ORE_TARGET_TONS, waste_target_tons=WASTE_TARGET_TONS,
                ore_shortfall_weight=ORE_WEIGHT, waste_shortfall_weight=WASTE_WEIGHT,
                production_settlement="true_shift_end_once",
                wait_weight=weight, wait_reference_seconds=WAIT_REFERENCE_SECONDS,
                wait_settlement="raw_fleet_wait_seconds_increment_each_transition",
                rpc_reward_used_as_target=False, rpc_production_weights=[4.0, 4.0],
                terminal_bootstrap=0.0)


def validate_composite_contract(value: Mapping[str, Any]) -> float:
    """Validate all frozen fields and return this file/phase's explicit branch."""
    if not isinstance(value, Mapping) or "wait_weight" not in value:
        raise ValueError("explicit composite_contract with wait_weight is required")
    weight = _wait_weight(value["wait_weight"])
    expected = composite_contract(wait_weight=weight)
    try:
        actual_json = json.dumps(dict(value), sort_keys=True, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise ValueError("composite_contract must be canonical finite JSON") from error
    if actual_json != json.dumps(expected, sort_keys=True, allow_nan=False, separators=(",", ":")):
        raise ValueError("composite_contract differs from the frozen objective branch")
    return weight


def _nonnegative(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite nonnegative number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite nonnegative number") from error
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return number


def _waiting(values: Iterable[float], count: int, total: float) -> tuple[float, ...]:
    try:
        source = tuple(values)
    except TypeError as error:
        raise ValueError("wait_deltas must contain one raw fleet-seconds increment per transition") from error
    if len(source) != count:
        raise ValueError("wait_deltas must contain one raw fleet-seconds increment per transition")
    deltas = tuple(_nonnegative(value, "wait_delta_seconds") for value in source)
    if not math.isclose(math.fsum(deltas), total, rel_tol=1e-9, abs_tol=1e-6):
        raise ValueError("raw wait increments do not conserve terminal cumulative fleet wait seconds")
    return deltas


@dataclass(frozen=True)
class CompositeTargets:
    step_costs: tuple[float, ...]
    mc_returns: tuple[float, ...]
    closed_windows_before: tuple[int, ...]
    terminal_return: float
    grade_costs: tuple[float, ...]
    production_costs: tuple[float, ...]
    wait_costs: tuple[float, ...]


def build_composite_targets(memory: Any, *, wait_deltas: Iterable[float],
                            ore_tons: float, waste_tons: float,
                            total_wait_seconds: float,
                            wait_weight: float = WAIT_WEIGHT) -> CompositeTargets:
    """Freeze complete-shift targets without ingesting the raw RPC reward.

    Existing validated event boundaries determine grade settlements. Production
    is charged only on the final true-END transition. Waiting uses incremental
    *fleet vehicle seconds*, whose sum must equal the terminal cumulative value.
    No reset occurs at a grade-window boundary and no future truth is an input.
    """
    weight = _wait_weight(wait_weight)
    grade = build_grade_targets(memory)
    count = len(grade.step_costs)
    ore = _nonnegative(ore_tons, "ore_tons")
    waste = _nonnegative(waste_tons, "waste_tons")
    total = _nonnegative(total_wait_seconds, "total_wait_seconds")
    deltas = _waiting(wait_deltas, count, total)
    production = (ORE_WEIGHT * max(0.0, 1.0 - ore / ORE_TARGET_TONS) +
                  WASTE_WEIGHT * max(0.0, 1.0 - waste / WASTE_TARGET_TONS))
    production_costs = (0.0,) * (count - 1) + (production,)
    grade_costs = tuple(GRADE_WEIGHT * value for value in grade.step_costs)
    wait_costs = tuple(weight * value / WAIT_REFERENCE_SECONDS for value in deltas)
    costs = tuple(math.fsum(values) for values in zip(grade_costs, production_costs, wait_costs))
    mc = tuple(float(value) for value in np.cumsum(np.asarray(costs, dtype=np.float64)[::-1])[::-1])
    if not all(math.isfinite(value) for value in (*costs, *mc)):
        raise ValueError("combined targets overflow finite cost range")
    return CompositeTargets(costs, mc, grade.closed_windows_before, 0.0,
                            grade_costs, production_costs, wait_costs)


def verify_episode_targets(episode: Mapping[str, Any]) -> CompositeTargets:
    """Rebuild labels and verify the sealed episode's explicit reward identity."""
    weight = validate_composite_contract(episode.get("composite_contract"))
    identity = episode.get("identity")
    if not isinstance(identity, Mapping) or identity.get("reward_version") != REWARD_VERSION:
        raise ValueError("episode reward identity differs from the composite contract")
    evaluation = episode.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise ValueError("episode requires physical terminal production and cumulative waiting")
    try:
        rebuilt = build_composite_targets(episode["memory"], wait_deltas=episode["wait_deltas"],
            ore_tons=evaluation["ore_tons"], waste_tons=evaluation["waste_tons"],
            total_wait_seconds=evaluation["total_wait_seconds"], wait_weight=weight)
    except KeyError as error:
        raise ValueError("episode physical component evidence is incomplete") from error
    if episode.get("targets") != rebuilt:
        raise ValueError("episode composite targets differ from physical component evidence")
    return rebuilt
