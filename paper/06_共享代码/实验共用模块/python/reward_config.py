"""RPC-owned production configuration. No guessed historical JSON or defaults."""
from dataclasses import dataclass
import hashlib
import json
import math

REQUIRED = ('production_reward_version', 'wait_kappa', 'shift_seconds', 'window_seconds',
            'target_ore_tons_shift', 'target_waste_tons_shift', 'w_production',
            'waste_shortfall_weight', 'grade_lower', 'grade_upper', 'target_grade',
            'grade_epsilon', 'tonnage_epsilon', 'mass_balance_epsilon', 'num_trucks')
# Exact, versioned exceptions: no wildcard removal of configuration checks.
DIAGNOSTIC_ALIASES = {'config.eval.shovel_idle_seconds': 'diag.shovel_idle_seconds'}
LIFECYCLE_FIELDS = {'config.eval.fallback_used'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(mapping, key):
    require(key in mapping, f'missing RPC field: {key}')
    value = float(mapping[key])
    require(math.isfinite(value), f'nonfinite RPC field: {key}')
    return value


def partition_components(components, *, terminal):
    configuration, lifecycle, diagnostics = {}, {}, {}
    for key in components:
        value = finite(components, key)
        if key in DIAGNOSTIC_ALIASES:
            require(terminal, f'terminal diagnostic appeared before shift end: {key}')
            canonical = DIAGNOSTIC_ALIASES[key]
            if canonical in components:
                require(value == float(components[canonical]), 'diagnostic alias disagreement')
            diagnostics[canonical] = value
        elif key in LIFECYCLE_FIELDS:
            lifecycle[key] = value
            require(value == 0, 'dispatch fallback was used; invalid on-policy transition')
        elif key.startswith('config.'):
            configuration[key] = value
        elif key.startswith('diag.'):
            if key == 'diag.shovel_idle_seconds':
                require(terminal, 'shovel idle summary is terminal-only')
            diagnostics[key] = value
    return configuration, lifecycle, diagnostics


@dataclass(frozen=True)
class RewardConfiguration:
    entries: tuple

    def get(self, name):
        return dict(self.entries)['config.' + name]

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.entries, separators=(',', ':')).encode()).hexdigest()

    def to_dict(self):
        return dict(self.entries)


def configuration_from_components(components, *, terminal=False):
    values, _, _ = partition_components(components, terminal=terminal)
    missing = ['config.' + k for k in REQUIRED if 'config.' + k not in values]
    require(not missing, 'missing RPC configuration: ' + ', '.join(missing))
    cfg = RewardConfiguration(tuple(sorted(values.items())))
    require(cfg.get('production_reward_version') == 2, 'legacy reward is not accepted by production-v2')
    require(cfg.get('shift_seconds') == 28800 and cfg.get('window_seconds') == 7200,
            'requires eight hours / four two-hour windows')
    require(cfg.get('target_ore_tons_shift') == 16000 and cfg.get('target_waste_tons_shift') == 4000,
            'production targets differ from frozen baseline')
    require(cfg.get('num_trucks') == 9, '1519D contract requires nine trucks')
    require(cfg.get('wait_kappa') > 0, 'wait_kappa must be explicitly set and positive')
    require(0 <= cfg.get('grade_lower') <= cfg.get('target_grade') <= cfg.get('grade_upper') <= 1,
            'invalid grade bounds')
    for k in ('w_production', 'waste_shortfall_weight', 'grade_epsilon', 'tonnage_epsilon', 'mass_balance_epsilon'):
        require(cfg.get(k) >= 0, 'negative parameter: ' + k)
    return cfg
