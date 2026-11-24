"""Read-only common input validation for relation-gat-ego-003.

The schema DTO is an independently frozen field oracle, not a neural encoder.
Torch tensors are detached only for validation; callers retain the original
tensor and its continuous autograd paths.
"""
from __future__ import annotations

import math
import numpy as np
import torch

from change_plans.token_attention_001.schema_audit_001 import token_adapter_audit as _dto
from grade_commitment_ppo_v1 import features as _legacy

SIZE = 1519
SCHEMA_SHA256 = _dto.SCHEMA_SHA256
PARTS = dict(_dto.PARTS)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def as_numpy(value):
    """A read-only-use view, never a numerical replacement for neural inputs."""
    if isinstance(value, torch.Tensor):
        _require(value.device.type == 'cpu' and value.dtype == torch.float32,
                 'CPU torch.float32 observation required')
        return value.detach().numpy()
    _require(isinstance(value, np.ndarray) and value.dtype == np.float32,
             'numpy.float32 observation required')
    return value


def validate_original_fifo(state):
    """Validate original slots before any private C zero-mass projection."""
    s = as_numpy(state)
    _require(s.shape == (SIZE,), 'state[1519] required')
    rows = s[1391:1439].reshape(16, 3)
    _require(bool(np.isfinite(rows).all()), 'nonfinite original FIFO')
    _require(bool(np.isin(rows[:, 0], (0, 1)).all()), 'FIFO valid must be binary')
    valid = rows[:, 0].astype(bool)
    _require(not bool((np.diff(valid.astype(np.int8)) > 0).any()),
             'FIFO valid slots must form a prefix')
    _require(not bool(rows[~valid, 1:].any()), 'absent FIFO payload must be zero')
    _require(bool((rows[:, 1] >= 0).all()), 'FIFO tons must be nonnegative')
    _require(bool(((rows[:, 2] >= 0) & (rows[:, 2] <= 1)).all()),
             'original FIFO grade outside [0,1]')
    q = float(rows[valid, 1].astype(np.float64).sum()) * 500
    _require(math.isclose(q, float(s[1300]) * 500, rel_tol=2e-5, abs_tol=.002),
             'FIFO tons disagree with crusher buffer snapshot')
    return state


def _additional_common_rules(s):
    # Keep the legacy C state-domain restrictions without constructing C128.
    # Frozen _state checks bounds, physical quantities and grade target interval.
    _legacy._state(s)
    validate_original_fifo(s)
    trucks = s[27:1107].reshape(9, 120)
    ranks = np.rint(trucks[:, 63:79].astype(np.float64) * 16).astype(np.int64)
    for rank in ranks:
        used = rank[rank > 0]
        _require(np.array_equal(np.sort(used), np.arange(1, len(used) + 1)),
                 'route ranks must be unique contiguous 1..K')
    eps = _legacy.FEATURE_CONFIG['quantity_epsilon_tons']
    for t in trucks:
        q = float(t[19] if t[17] else t[20]) * 193
        _require(q >= 0, 'negative truck payload/reservation')
        if q <= eps or not t[18]:
            continue
        phase = _legacy._onehot_index(t[4:16])
        if t[17]:
            _require(t[21] == 1, 'loaded ore must have observed gradeKnown')
            _require(0 <= t[22] <= 1, 'invalid observed payload grade')
        else:
            target = _legacy._onehot_index(t[26:30])
            _require(target is not None, 'reserved ore requires observed loadTarget')
            load = s[1107 + 47 * target:1154 + 47 * target]
            grade, sd = (t[23], t[24]) if phase == 4 else (load[2], load[3])
            _require(0 <= grade <= 1 and sd >= 0,
                     'invalid causal reservation grade statistics')
        if (t[17] and phase == 5) or (not t[17] and phase == 2):
            _require(t[62] >= 0, 'negative route distance')


def validate_states(states):
    """Return the same input after independent current-row common validation."""
    raw = as_numpy(states)
    _dto.validate_states(raw)
    for row in raw:
        _additional_common_rules(row)
    return states


def validate_state(state, *, truck_id=None):
    raw = as_numpy(state)
    _require(raw.shape == (SIZE,), 'float32 state[1519] required')
    _dto.validate_state(raw, truck_id=truck_id)
    _additional_common_rules(raw)
    return state


def raw_tokens(state):
    """Diagnostic raw copies with exact schema ownership; never learned state."""
    validate_state(state)
    raw = as_numpy(state)
    return {name: raw[a:b].reshape(shape).copy()
            for name, (a, b, shape) in PARTS.items()}


__all__ = ['SIZE', 'SCHEMA_SHA256', 'PARTS', 'as_numpy', 'validate_state',
           'validate_states', 'validate_original_fifo', 'raw_tokens']
