"""Causal, schema-exact feature paths for the isolated Task-Query CMDP V1.

The original entity encoder validates CPU float32 S1519 observations.  This
module preserves that supported domain; it never regenerates an action mask.
All scaled physical quantities are reconstructed in float64 before conversion
to the float32 neural features.  No journal or future outcome is read here.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from relation_gat_ego_v1.adapter import validate_states

TASK_FEATURE_NAMES = (
    'ore_min_div_20000', 'waste_min_div_4000', 'waste_stop_div_4000',
    'waste_delivered_div_4000', 'waste_shortfall_div_4000',
    'waste_committed_div_4000', 'nonnegative_remaining_stop_div_4000',
    'waste_limit_enabled', 'grade_failed_before', 'ore_crushed_div_20000',
    'ore_shortfall_div_20000', 'ore_buffer_div_500', 'ore_loaded_div_20000',
    'ore_source_reserved_div_20000', 'ore_unload_reserved_div_500',
)
RELATION_FEATURE_NAMES = (
    'is_ego', 'is_candidate_load', 'truck_load_target_matches_candidate',
    'road_on_ego_known_route', 'material_valid', 'is_ore', 'is_waste',
    'grade_valid', 'grade_nominal', 'grade_offset',
)
ACTION_TOKEN_INDICES = (9, 10, 11, 12)


@dataclass(frozen=True)
class SceneConfig:
    """A validated RPC scene, never a replacement for missing RPC fields."""

    ore_min: float = 16000.
    waste_min: float = 4000.
    waste_stop: float = 6000.
    grade_target: float = .27
    grade_lower: float = .26
    grade_upper: float = .28
    waste_limit_enabled: bool = True
    max_truck_capacity: float = 193.
    grade_tolerance: float = 1e-12
    ton_tolerance: float = 1e-6

    def __post_init__(self):
        for key, value in vars(self).items():
            if key != 'waste_limit_enabled' and (isinstance(value, bool) or
                    not isinstance(value, (float, int)) or not math.isfinite(value)):
                raise ValueError('nonfinite/non-numeric scene configuration: ' + key)
        if type(self.waste_limit_enabled) is not bool:
            raise ValueError('waste_limit_enabled must be bool')
        if not (0 <= self.grade_lower < self.grade_target < self.grade_upper <= 1):
            raise ValueError('invalid grade acceptance interval')
        if not (self.ore_min > 0 and self.waste_min > 0 and
                self.waste_stop >= self.waste_min and self.max_truck_capacity > 0):
            raise ValueError('invalid production configuration')
        if self.grade_tolerance < 0 or self.ton_tolerance < 0:
            raise ValueError('negative tolerance')

    @property
    def target_ore_tons_shift(self):
        return self.ore_min

    @property
    def target_waste_tons_shift(self):
        return self.waste_min

    @property
    def production_tolerance(self):
        return self.ton_tolerance


def require_states(states, config: SceneConfig, *, allow_terminal=False):
    if not isinstance(config, SceneConfig):
        raise ValueError('validated SceneConfig required')
    if not isinstance(states, torch.Tensor) or states.ndim != 2 or states.shape[1] != 1519:
        raise ValueError('CPU float32 tensor[B,1519] required')
    if states.device.type != 'cpu' or states.dtype != torch.float32 or states.shape[0] == 0:
        raise ValueError('nonempty CPU float32 tensor[B,1519] required')
    validate_states(states)
    if not allow_terminal and bool(states[:, 17].any()):
        raise ValueError('Actor is defined on DECISION rows, not END rows')
    # Comparing the observed wire encoding avoids an invented absolute tolerance
    # that would conflate RPC double configuration and float32 snapshots.
    for index, value in ((11, config.ore_min / 20000), (12, config.waste_min / 4000),
                         (13, config.grade_lower), (14, config.grade_upper),
                         (15, config.grade_target)):
        expected = torch.tensor(value, dtype=torch.float32)
        if not bool((states[:, index] == expected).all()):
            raise ValueError('observation/RPC scene mismatch at field ' + str(index))
    loads = states[:, 1107:1295].reshape(-1, 4, 47)
    dumps = states[:, 1295:1391].reshape(-1, 2, 48)
    # This first implementation is deliberately scoped to the verified scene.
    if not bool((loads[:, :, 1] == states.new_tensor([1, 1, 1, 0])).all()):
        raise ValueError('action-to-material scene mapping changed')
    if not bool((dumps[:, :, 1] == states.new_tensor([1, 0])).all()):
        raise ValueError('FIFO-to-crusher scene mapping changed')
    return states


def require_failed_before(grade_failed_before, states):
    value = torch.as_tensor(grade_failed_before, device=states.device)
    if value.ndim != 1 or value.shape[0] != states.shape[0]:
        raise ValueError('grade_failed_before[B] required')
    if not bool(torch.isfinite(value).all()) or not bool(((value == 0) | (value == 1)).all()):
        raise ValueError('grade_failed_before must be binary')
    return value.to(dtype=states.dtype)


def physical_ledger(states):
    """Exclusive current mass bins.  Does not decide masks or enforce RPC audits."""
    x = states.double()
    trucks = x[:, 27:1107].reshape(-1, 9, 120)
    loads = x[:, 1107:1295].reshape(-1, 4, 47)
    loaded, ore = trucks[:, :, 17], trucks[:, :, 18]
    return {
        'ore_crushed': x[:, 7] * 20000,
        'ore_buffer': x[:, 1300] * 500,
        'ore_unload_reserved': x[:, 1301] * 500,
        'ore_loaded': (loaded * ore * trucks[:, :, 19]).sum(-1) * 193,
        'ore_source_reserved': (loads[:, :, 1] * loads[:, :, 6]).sum(-1) * 30000,
        'ore_source_produced': (loads[:, :, 1] * loads[:, :, 8]).sum(-1) * 30000,
        'waste_delivered': x[:, 9] * 4000,
        'waste_shortfall': x[:, 10] * 4000,
        'waste_loaded': (loaded * (1 - ore) * trucks[:, :, 19]).sum(-1) * 193,
        'waste_source_reserved': ((1 - loads[:, :, 1]) * loads[:, :, 6]).sum(-1) * 30000,
        'waste_committed': ((1 - loads[:, :, 1]) *
                            (loads[:, :, 6] + loads[:, :, 8])).sum(-1) * 30000,
    }


def task_features(states, grade_failed_before, config: SceneConfig):
    require_states(states, config)
    failed = require_failed_before(grade_failed_before, states).double()
    d = physical_ledger(states)
    const = lambda number: torch.full_like(d['ore_crushed'], number)
    features = torch.stack((
        const(config.ore_min / 20000), const(config.waste_min / 4000),
        const(config.waste_stop / 4000), d['waste_delivered'] / 4000,
        d['waste_shortfall'] / 4000, d['waste_committed'] / 4000,
        (config.waste_stop - d['waste_committed']).clamp_min(0) / 4000,
        const(float(config.waste_limit_enabled)), failed,
        d['ore_crushed'] / 20000,
        (config.ore_min - d['ore_crushed']).clamp_min(0) / 20000,
        d['ore_buffer'] / 500, d['ore_loaded'] / 20000,
        d['ore_source_reserved'] / 20000, d['ore_unload_reserved'] / 500,
    ), -1).float()
    if not bool(torch.isfinite(features).all()):
        raise ValueError('nonfinite task features')
    return features


def relation_features(states, config: SceneConfig, *, actor: bool):
    """Return [B,4,32,10] for Actor or [B,1,32,10] for a Critic.

    Grade is defined for ore.  Load means are nominal; loaded Truck grade is
    current known cargo grade; Dump0 grade is current positive-mass FIFO mean.
    Provenance is retained by the canonical typed entity, never guessed for
    unknown material.  Empty/waste Dump contents are not observable, so zero.
    """
    require_states(states, config)
    batch = states.shape[0]
    trucks = states[:, 27:1107].reshape(batch, 9, 120)
    loads = states[:, 1107:1295].reshape(batch, 4, 47)
    fifo = states[:, 1391:1439].reshape(batch, 16, 3)
    material = states.new_zeros((batch, 32, 6))
    material[:, :9, 0] = trucks[:, :, 17]
    material[:, :9, 1] = trucks[:, :, 17] * trucks[:, :, 18]
    material[:, :9, 2] = trucks[:, :, 17] * (1 - trucks[:, :, 18])
    material[:, :9, 3] = trucks[:, :, 17] * trucks[:, :, 18] * trucks[:, :, 21]
    material[:, :9, 4] = material[:, :9, 3].clone() * trucks[:, :, 22]
    material[:, 9:13, 0] = 1
    material[:, 9:13, 1] = loads[:, :, 1]
    material[:, 9:13, 2] = 1 - loads[:, :, 1]
    material[:, 9:13, 3] = loads[:, :, 1]
    material[:, 9:13, 4] = loads[:, :, 1] * loads[:, :, 2]
    mass = fifo[:, :, 0].double() * fifo[:, :, 1].double()
    total = mass.sum(-1)
    fifo_present = fifo[:, :, 0].bool().any(-1).to(states.dtype)
    valid_grade = (total > 0).to(states.dtype)
    mean = ((mass * fifo[:, :, 2].double()).sum(-1) /
            total.clamp_min(torch.finfo(torch.float64).tiny)).float()
    material[:, 13, 0:2] = fifo_present.unsqueeze(-1)
    material[:, 13, 3] = valid_grade
    material[:, 13, 4] = mean * valid_grade
    material[:, :, 5] = material[:, :, 3].clone() * (
        material[:, :, 4].clone() - config.grade_target) / (config.grade_upper - config.grade_lower)
    queries = 4 if actor else 1
    result = states.new_zeros((batch, queries, 32, 10))
    result[:, :, :, 4:] = material.unsqueeze(1)
    if actor:
        result[:, :, :9, 0] = trucks[:, None, :, 0]
        action = torch.arange(4)
        result[:, action, action + 9, 1] = 1
        result[:, :, :9, 2] = trucks[:, :, 26:30].transpose(1, 2)
        ego = trucks[:, :, 0].argmax(-1)
        ego_route = trucks[torch.arange(batch), ego, 63:79] > 0
        result[:, :, 15:31, 3] = ego_route[:, None, :].to(states.dtype)
    return result


__all__ = ['SceneConfig', 'TASK_FEATURE_NAMES', 'RELATION_FEATURE_NAMES',
           'ACTION_TOKEN_INDICES', 'task_features', 'physical_ledger',
           'relation_features', 'require_states', 'require_failed_before']
