"""Explicit v2 RPC admission; this module never opens a simulator or socket.

The original AnyLogic mask is authoritative. Quantization bounds compare two
representations of the same ledger and never change the double stop predicate.
"""
from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np

from python.reward_config import configuration_from_components
from relation_gat_ego_v1.adapter import validate_state
from .features import SceneConfig


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(values, key):
    require(key in values, 'missing RPC field: ' + key)
    value = float(values[key])
    require(math.isfinite(value), 'nonfinite RPC field: ' + key)
    return value


def quantization_error(values, scale):
    """Conservative half-ULP for already-rounded float32 linear operands."""
    x = np.asarray(values, dtype=np.float32)
    hi = np.nextafter(x, np.float32(np.inf)).astype(np.float64)
    lo = np.nextafter(x, np.float32(-np.inf)).astype(np.float64)
    error = np.maximum(hi-x.astype(np.float64), x.astype(np.float64)-lo)
    require(np.isfinite(error).all(), 'nonfinite float32 neighboring spacing')
    return float(error.sum()*abs(scale)/2)


def _near(actual, expected, terms, name):
    bound = 1e-5 + 1e-9*max(abs(actual), abs(expected))
    bound += sum(quantization_error(values, scale) for values, scale in terms)
    require(abs(actual-expected) <= bound, name + ' differs beyond float32 bound')
    return bound


@dataclass(frozen=True)
class RequestIdentity:
    recording_id: str
    run_index: int
    seed: int
    episode_id: str | int
    request_id: str
    decision_sequence: int

    def __post_init__(self):
        for key in ('recording_id', 'request_id'):
            require(isinstance(getattr(self, key), str) and bool(getattr(self, key)),
                    'missing identity: ' + key)
        require((isinstance(self.episode_id, str) and bool(self.episode_id)) or
                (type(self.episode_id) is int and self.episode_id >= 0), 'invalid episode identity')
        for key in ('run_index', 'seed', 'decision_sequence'):
            require(type(getattr(self, key)) is int and getattr(self, key) >= 0,
                    'invalid identity: ' + key)

    @property
    def episode_key(self):
        return self.recording_id, self.run_index, self.seed, self.episode_id


def configuration_from_rpc(components, *, terminal=False):
    """Validate full reward/observation configuration and the new scenario."""
    reward = configuration_from_components(components, terminal=terminal)
    get = lambda key: _number(components, 'config.' + key)
    require(get('waste_task_limit_version') == 2, 'requires explicit waste scenario v2')
    require(get('waste_task_limit_enabled') in (0, 1), 'invalid waste enable flag')
    enabled = bool(get('waste_task_limit_enabled'))
    stop, upper, overrun = [get('waste_task_' + key) for key in
                            ('dispatch_stop_tons', 'upper_tons', 'overrun_tons')]
    # V1 implementation baseline: the user selected an enabled 6000 t stop.
    require(enabled and stop == 6000, 'requires enabled 6000 t V1 baseline')
    require(overrun > 0 and abs(upper-stop-overrun) <= 1e-9,
            'v2 upper must equal stop plus actual fleet maximum capacity')
    require(stop >= get('target_waste_tons_shift'), 'stop below minimum production')
    require(get('grade_lower') < get('grade_upper'), 'empty grade acceptance interval')
    cfg = SceneConfig(ore_min=get('target_ore_tons_shift'),
                      waste_min=get('target_waste_tons_shift'), waste_stop=stop,
                      grade_target=get('target_grade'), grade_lower=get('grade_lower'),
                      grade_upper=get('grade_upper'), waste_limit_enabled=enabled,
                      max_truck_capacity=overrun, grade_tolerance=get('grade_epsilon'),
                      ton_tolerance=get('tonnage_epsilon'))
    return cfg, reward.fingerprint


def validate_snapshot(state, components, *, terminal=False):
    """Same-request physical quantities, independent bins and original mask."""
    validate_state(state)
    cfg, fingerprint = configuration_from_rpc(components, terminal=terminal)
    x = state.astype(np.float64)
    t = x[27:1107].reshape(9, 120)
    s = x[1107:1295].reshape(4, 47)
    f = x[1391:1439].reshape(16, 3)
    require(bool(x[17]) == terminal, 'END flag disagrees with RPC lifecycle')
    require(np.isin(t[:, 17:19], (0, 1)).all() and np.isin(s[:, 1], (0, 1)).all(),
            'nonbinary material type or loaded state')
    loaded, ore = t[:, 17] == 1, t[:, 18] == 1
    os, ws = s[:, 1] == 1, s[:, 1] == 0
    produced_o, reserved_o = 30000*s[os, 8].sum(), 30000*s[os, 6].sum()
    committed = 30000*(s[ws, 6]+s[ws, 8]).sum()
    reserved_d, delivered = 30000*s[ws, 6].sum(), 4000*x[9]
    loaded_o, loaded_d = 193*t[loaded & ore, 19].sum(), 193*t[loaded & ~ore, 19].sum()
    crushed, buffer, unload_reserved = 20000*x[7], 500*x[1300], 500*x[1301]
    raw_c = _number(components, 'audit.waste_task.committed_tons')
    raw_d = _number(components, 'audit.waste_task.delivered_tons')
    closed = _number(components, 'audit.waste_task.assignment_closed')
    require(closed in (0, 1), 'invalid assignment_closed bit')
    require(closed == float(raw_c >= cfg.waste_stop-1e-6), 'raw stop predicate differs')
    require(0 <= raw_d <= raw_c+cfg.ton_tolerance and
            raw_c <= cfg.waste_stop+cfg.max_truck_capacity+cfg.ton_tolerance,
            'invalid double waste ledger bounds')
    bounds = {}
    bounds['commitment'] = _near(raw_c, committed, [(s[ws][:, [6, 8]], 30000)], 'commitment')
    bounds['delivery'] = _near(raw_d, delivered, [(state[9], 4000)], 'delivery')
    bounds['waste_bins'] = _near(committed, delivered+loaded_d+reserved_d,
        [(s[ws][:, [6, 8]], 30000), (state[9], 4000),
         (t[loaded & ~ore, 19], 193), (s[ws, 6], 30000)], 'exclusive waste bins')
    bounds['ore_bins'] = _near(produced_o, crushed+buffer+loaded_o,
        [(s[os, 8], 30000), (state[7], 20000), (state[1300], 500),
         (t[loaded & ore, 19], 193)], 'exclusive ore bins')
    bounds['ore_reservation'] = _near(reserved_o, 193*t[~loaded & ore, 20].sum(),
        [(s[os, 6], 30000), (t[~loaded & ore, 20], 193)], 'ore source reservation')
    bounds['waste_reservation'] = _near(reserved_d, 193*t[~loaded & ~ore, 20].sum(),
        [(s[ws, 6], 30000), (t[~loaded & ~ore, 20], 193)], 'waste source reservation')
    bounds['fifo'] = _near(buffer, 500*f[f[:, 0] == 1, 1].sum(),
        [(state[1300], 500), (f[f[:, 0] == 1, 1], 500)], 'FIFO buffer')
    require(unload_reserved <= loaded_o + 1e-5 + quantization_error(t[loaded & ore, 19], 193)
            + quantization_error(state[1301], 500), 'unload reservation is not loaded subset')
    _near(float(t[:, 16].max()*193), cfg.max_truck_capacity,
          [(t[:, 16], 193)], 'maximum truck capacity')
    for index, expected, scale in ((11, cfg.ore_min, 20000), (12, cfg.waste_min, 4000),
                                   (13, cfg.grade_lower, 1), (14, cfg.grade_upper, 1),
                                   (15, cfg.grade_target, 1)):
        _near(x[index]*scale, expected, [(state[index], scale)], 'observation/RPC target')
    _near(x[10]*4000, max(0, cfg.waste_min-raw_d),
          [(state[10], 4000)], 'waste shortfall')
    mask, tons = state[19:23], state[23:27]
    require(np.array_equal(mask > 0, tons*193 > 1e-6), 'mask and dispatchable tons differ')
    if closed:
        require(not mask[ws].any(), 'closed waste action remains selectable')
    if terminal:
        require(not mask.any(), 'END has selectable actions')
    else:
        require(mask.any(), 'nonterminal has no valid action')
    return dict(config=cfg, configuration_fingerprint=fingerprint,
                committed_tons=raw_c, delivered_tons=raw_d, closed=bool(closed),
                crushed_tons=crushed, consistency_bounds=bounds,
                hard_mask_source='unmodified_AnyLogic_RPC', terminal=terminal)


class RPCAdmission:
    """In-memory per-episode identity and configuration lock; no data capture."""
    def __init__(self):
        self._episodes = {}
        self._request_ids = {}

    def accept(self, identity, state, components, *, terminal=False):
        require(isinstance(identity, RequestIdentity), 'explicit request identity required')
        seen = self._request_ids.get(identity.episode_key, set())
        require(identity.request_id not in seen, 'request id reused within episode')
        result = validate_snapshot(state, components, terminal=terminal)
        previous = self._episodes.get(identity.episode_key)
        if previous is None:
            require(identity.decision_sequence == 0, 'episode must start at decision sequence zero')
        else:
            old_id, old = previous
            require(not old['terminal'], 'request after true END')
            require(identity.decision_sequence == old_id.decision_sequence+1,
                    'duplicate or gapped decision sequence')
            require(identity.request_id != old_id.request_id, 'request id reused')
            require(result['configuration_fingerprint'] == old['configuration_fingerprint'],
                    'configuration drift within episode')
            require(result['committed_tons']+1e-6 >= old['committed_tons'] and
                    result['delivered_tons']+1e-6 >= old['delivered_tons'],
                    'waste ledger regressed')
        self._episodes[identity.episode_key] = identity, result
        self._request_ids.setdefault(identity.episode_key, set()).add(identity.request_id)
        return result


def configuration_bytes_sha256(config_bytes):
    require(isinstance(config_bytes, bytes) and bool(config_bytes), 'exact config bytes required')
    json.loads(config_bytes)
    return hashlib.sha256(config_bytes).hexdigest()
