"""Versioned C128 facade; frozen legacy formulas, private zero-mass FIFO view."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import numpy as np

from grade_commitment_ppo_v1 import features as _legacy
from .adapter import as_numpy, validate_state, validate_states

FEATURE_VERSION = 'commitment-summary-v1-zero-mass-adapter-v2'
FEATURE_SIZE = 128
FEATURE_NAMES = tuple(_legacy.FEATURE_NAMES)
LEGACY_SOURCE_SHA256 = '72b3547d5b8b34ff6b5cf62f71f1bda595d625765c3e5bd7ba3c9803359b66da'
if hashlib.sha256(Path(_legacy.__file__).read_bytes()).hexdigest() != LEGACY_SOURCE_SHA256:
    raise RuntimeError('frozen legacy C source SHA256 differs')
_LEGACY_CONTRACT = _legacy.feature_contract()
_FROZEN_CONFIG = {
    'version': FEATURE_VERSION,
    'projection': 'validate_original_then_remove_exact_zero_mass_valid_fifo_privately',
    'legacy_source_sha256': LEGACY_SOURCE_SHA256,
    'legacy_config_sha256': _LEGACY_CONTRACT['config_sha256'],
    'legacy_names_sha256': _LEGACY_CONTRACT['names_sha256'],
    'legacy_config': deepcopy(_LEGACY_CONTRACT['config']),
}
FEATURE_CONFIG = deepcopy(_FROZEN_CONFIG)


def _json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def normalize_feature_config(config=None):
    result = deepcopy(_FROZEN_CONFIG if config is None else config)
    if not isinstance(result, dict) or result != _FROZEN_CONFIG:
        raise ValueError('feature config differs from frozen C-v2 outer contract')
    return result


def feature_config_sha256(config=None):
    return _json_sha(normalize_feature_config(config))


def feature_contract(config=None):
    cfg = normalize_feature_config(config)
    result = {
        'version': FEATURE_VERSION, 'dimension': FEATURE_SIZE,
        'names_sha256': _LEGACY_CONTRACT['names_sha256'],
        'config': cfg, 'config_sha256': _json_sha(cfg),
        'legacy_source_sha256': LEGACY_SOURCE_SHA256,
        'legacy_config_sha256': _LEGACY_CONTRACT['config_sha256'],
        'scope': 'causal_decision_summary_whole_shift',
        'source': 'dispatch.physical.v3',
        'cache_source': 'original_float32_observation_bytes',
    }
    result['contract_sha256'] = _json_sha(result)
    return result


def private_c_state(state):
    """Validate raw first; preserve all original bytes outside the 48 FIFO cols.

    With no zero-mass valid slot, return the original ndarray view unchanged.
    Otherwise return an independent float32 copy, retaining positive packet order.
    """
    validate_state(state)
    raw = as_numpy(state)
    slots = raw[1391:1439].reshape(16, 3)
    zero = (slots[:, 0] == 1) & (slots[:, 1] == 0)
    if not bool(zero.any()):
        return raw
    result = raw.copy()
    target = result[1391:1439].reshape(16, 3)
    positive = slots[(slots[:, 0] == 1) & (slots[:, 1] > 0)].copy()
    target[:] = 0
    target[:len(positive)] = positive
    return result


def build_features(state, *, config=None):
    cfg = normalize_feature_config(config)
    return _legacy.build_features(private_c_state(state), config=cfg['legacy_config'])


def build_features_batch(states, *, config=None):
    cfg = normalize_feature_config(config)
    validate_states(states)
    return np.stack([build_features(row, config=cfg) for row in as_numpy(states)])


build_sequence = build_features_batch

__all__ = ['FEATURE_VERSION', 'FEATURE_SIZE', 'FEATURE_NAMES', 'FEATURE_CONFIG',
           'LEGACY_SOURCE_SHA256', 'normalize_feature_config', 'feature_contract',
           'feature_config_sha256', 'private_c_state', 'build_features',
           'build_features_batch', 'build_sequence']
