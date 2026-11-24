"""V1-A: strict RPC physical accounting; independent of the legacy reward."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

from .reward_config import configuration_from_components
from .reward_contract import REWARD_CONTRACT_ID
from .scenario_contract import from_components
from .observation_contract import OBSERVATION_SCHEMA_ID

ADAPTER_CONTRACT_ID = "dispatch-constrained-v1.1-indicator-costs"
WEIGHT_CONTRACT_ID = "mean-of-eight-undiscounted-episode-sums-v1"
WAIT_SCALE = 25000.0


class SamplingError(RuntimeError):
    """Invalid collection, never a business constraint failure."""


def require(condition, message):
    if not condition:
        raise SamplingError(message)


def number(mapping, key):
    require(key in mapping, f"missing RPC field: {key}")
    value = float(mapping[key])
    require(math.isfinite(value), f"nonfinite RPC field: {key}")
    return value


@dataclass(frozen=True)
class AdapterConfiguration:
    # Preserve *all* config fields, including upstream preprocessing parameters.
    entries: tuple[tuple[str, float], ...]

    @classmethod
    def from_rpc(cls, components):
        configuration_from_components(components)  # original contract still enforced
        scenario = from_components(components)
        require(scenario is not None and scenario.waste_task_limit_enabled,
                "V1.1 requires explicit enabled waste-task bounds from RPC")
        keys = ("grade_lower", "grade_upper", "target_ore_tons_shift",
                "target_waste_tons_shift", "waste_task_upper_tons", "shift_seconds",
                "window_seconds", "grade_epsilon", "tonnage_epsilon", "mass_balance_epsilon")
        values = {k: number(components, "config." + k) for k in keys}
        require(values["shift_seconds"] == 480 * 60 and
                values["window_seconds"] * 4 == values["shift_seconds"],
                "V1.1 requires a 480-minute shift and four windows")
        require(0 <= values["grade_lower"] <= values["grade_upper"] <= 1,
                "invalid grade bounds")
        require(values["target_ore_tons_shift"] > 0 and
                0 <= values["target_waste_tons_shift"] <= values["waste_task_upper_tons"],
                "invalid production targets")
        require(all(values[k] >= 0 for k in ("grade_epsilon", "tonnage_epsilon", "mass_balance_epsilon")),
                "negative physical tolerance")
        return cls(tuple(sorted((k, number(components, k)) for k in components if k.startswith("config."))))

    def get(self, key):
        return dict(self.entries)["config." + key]

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.entries).encode()).hexdigest()


class RewardCostAdapter:
    def __init__(self, config: AdapterConfiguration):
        self.config = config
        self.active = False

    def _check_config(self, info):
        require(info["observation_schema_id"] == OBSERVATION_SCHEMA_ID, "observation schema changed")
        require(info["reward_contract_id"] == REWARD_CONTRACT_ID, "original reward contract changed")
        require(AdapterConfiguration.from_rpc(info["reward_components"]) == self.config,
                "RPC configuration changed within batch")

    def start(self, info):
        require(not self.active, "cannot reset incomplete episode/history")
        self._check_config(info)
        c = info["reward_components"]
        require(info["decision_sequence"] == 0 and info["previous_action"] == -1 and
                info["sim_time_delta_seconds"] == 0 and not info["closed_windows"] and
                number(c, "delta_wait_vehicle_seconds") == 0 and
                info["transition_reward_raw"] == 0, "invalid EPISODE_START")
        # An unaccounted prefix must never disappear from full-shift conservation.
        require(number(c, "wait_vehicle_seconds") == 0, "nonzero waiting before first action")
        self.identity = (info["episode_id"], info["experiment_run_index"], info["experiment_seed"])
        self.sequence = 0
        self.time = float(info["sim_time_seconds"])
        self.grade_failed_so_far = False
        self.windows = []
        self.wait = 0.0
        self.total_reward = 0.0
        self.active = True

    def transition(self, info, *, terminated, truncated):
        require(self.active, "transition without active episode")
        self._check_config(info)
        require((info["episode_id"], info["experiment_run_index"], info["experiment_seed"]) == self.identity,
                "episode identity changed")
        require(info["decision_sequence"] == self.sequence + 1, "noncontiguous transition sequence")
        time = number(info, "sim_time_seconds")
        elapsed = number(info, "sim_time_delta_seconds")
        require(time >= self.time and math.isclose(time - self.time, elapsed, rel_tol=0, abs_tol=1e-6),
                "inconsistent transition time")
        require(not truncated, "truncated episode is a sampling anomaly")
        require((terminated and time == self.config.get("shift_seconds")) or
                (not terminated and time < self.config.get("shift_seconds")),
                "normal terminal must be exactly 480 minutes")
        c = info["reward_components"]
        delta_wait = number(c, "delta_wait_vehicle_seconds")
        cumulative_wait = number(c, "wait_vehicle_seconds")
        require(delta_wait >= 0 and cumulative_wait >= self.wait, "negative/regressing waiting time")
        # Numerical reconciliation only, not a business constraint tolerance.
        require(math.isclose(self.wait + delta_wait, cumulative_wait, rel_tol=1e-10, abs_tol=1e-7),
                "waiting increments do not conserve cumulative vehicle seconds")
        before = self.grade_failed_so_far
        for window in info["closed_windows"]:
            index = len(self.windows)
            require(index < 4 and window["window_index"] == index, "missing/duplicate/out-of-order window")
            start = number(window, "window_start_seconds")
            end = number(window, "window_end_seconds")
            require(start == index * self.config.get("window_seconds") and
                    end == (index + 1) * self.config.get("window_seconds") and end <= time,
                    "window boundary is inconsistent with callback time")
            ore, metal, grade = (number(window, k) for k in ("ore_tons", "metal_tons", "grade"))
            require(ore >= 0 and 0 <= metal <= ore + self.config.get("tonnage_epsilon"),
                    "invalid window material")
            require(abs(metal - grade * ore) <= self.config.get("mass_balance_epsilon"),
                    "window grade does not match metal/ore")
            if ore == 0:
                require(grade == 0, "empty window must have zero grade")
            upper = ore > 0 and grade > self.config.get("grade_upper") + self.config.get("grade_epsilon")
            lower = ore <= 0 or grade < self.config.get("grade_lower") - self.config.get("grade_epsilon")
            require(window["violation_upper"] == upper and window["violation_lower"] == lower,
                    "ALP violation flags disagree with RPC bounds/tolerances")
            self.grade_failed_so_far |= upper or lower
            self.windows.append(dict(window))
        reward = -delta_wait / WAIT_SCALE
        ore_cost = waste_cost = 0.0
        if terminated:
            require(len(self.windows) == 4, "normal terminal requires exactly four windows")
            physical = {k: number(c, k) for k in (
                "ore_tons_shift", "waste_tons_shift", "crusher_received_tons", "crusher_buffer_tons",
                "mined_ore_tons", "mined_waste_tons", "carried_ore_tons", "carried_waste_tons",
                "load_sites_reserved_tons", "assigned_not_loaded_tons")}
            require(all(v >= 0 for v in physical.values()), "negative terminal material")
            ore, waste = physical["ore_tons_shift"], physical["waste_tons_shift"]
            residuals = (
                sum(w["ore_tons"] for w in self.windows) - ore,
                physical["crusher_received_tons"] - ore - physical["crusher_buffer_tons"],
                physical["mined_ore_tons"] - physical["crusher_received_tons"] - physical["carried_ore_tons"],
                physical["mined_waste_tons"] - waste - physical["carried_waste_tons"],
                physical["load_sites_reserved_tons"] - physical["assigned_not_loaded_tons"],
            )
            require(all(abs(r) <= self.config.get("mass_balance_epsilon") for r in residuals),
                    "terminal material conservation failed")
            eps = self.config.get("tonnage_epsilon")
            ore_cost = float(ore < self.config.get("target_ore_tons_shift") - eps)
            waste_cost = float(not (self.config.get("target_waste_tons_shift") - eps <= waste <=
                                   self.config.get("waste_task_upper_tons") + eps))
            self.active = False
        self.sequence += 1
        self.time = time
        self.wait = cumulative_wait
        self.total_reward += reward
        require(math.isclose(self.total_reward, -self.wait / WAIT_SCALE, rel_tol=1e-10, abs_tol=1e-10),
                "shift reward conservation failed")
        return dict(adapter_contract_id=ADAPTER_CONTRACT_ID, reward=reward,
                    grade_cost=float(not before and self.grade_failed_so_far), ore_cost=ore_cost,
                    waste_cost=waste_cost, history_before=before, history_after=self.grade_failed_so_far)
