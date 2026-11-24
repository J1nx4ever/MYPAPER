"""AnyLogic is the only reward producer; Python recomputes only for validation."""
import math
from .reward_config import RewardConfiguration, finite, require

REWARD_CONTRACT_ID = 'dispatch-production-wait-shortfall-v2'
SHIFT_SECONDS, WINDOW_SECONDS, WINDOWS_PER_EPISODE = 28800, 7200, 4
ADDITIVE_COMPONENTS = ('wait', 'production', 'waste')


def window_summary_to_dict(s):
    names = ('window_index', 'window_start_seconds', 'window_end_seconds', 'ore_tons',
             'metal_tons', 'grade', 'grade_gap', 'grade_reward_raw', 'grade_reward_weighted',
             'violation_upper', 'violation_lower')
    return {name: getattr(s, name) for name in names}


def validate_window_summary(w, *, config, grade_bounds=True):
    for k in ('window_start_seconds', 'window_end_seconds', 'ore_tons', 'metal_tons', 'grade',
              'grade_gap', 'grade_reward_raw', 'grade_reward_weighted'):
        finite(w, k)
    j = w['window_index']
    require(type(j) is int and 0 <= j < 4, 'invalid window index')
    require(w['window_start_seconds'] == j * WINDOW_SECONDS and
            w['window_end_seconds'] == (j + 1) * WINDOW_SECONDS, 'invalid window boundary')
    q, metal, grade = (w[k] for k in ('ore_tons', 'metal_tons', 'grade'))
    require(q >= 0 and 0 <= metal <= q + config.get('tonnage_epsilon'), 'invalid window material')
    expected = metal / q if q else 0.0
    require(math.isclose(grade, expected, rel_tol=0, abs_tol=1e-12), 'window grade disagrees with material')
    require(abs(metal - grade*q) <= config.get('mass_balance_epsilon'), 'window metal mismatch')
    require(math.isclose(w['grade_gap'], config.get('target_grade') - grade, rel_tol=0, abs_tol=1e-12),
            'window grade gap mismatch')
    upper = q > 0 and grade > config.get('grade_upper') + config.get('grade_epsilon')
    lower = q <= 0 or grade < config.get('grade_lower') - config.get('grade_epsilon')
    require(w['violation_upper'] == upper and w['violation_lower'] == lower, 'window violation mismatch')
    return float(upper or lower)


def validate_reward_accounting(reward, components, windows, *, config, terminal):
    require(math.isfinite(reward), 'nonfinite production reward')
    for key in components:
        finite(components, key)
    delta = finite(components, 'delta_wait_vehicle_seconds')
    require(delta >= 0, 'negative waiting increment')
    require(finite(components, 'wait_vehicle_seconds') >= 0, 'negative cumulative waiting')
    expected_wait = -config.get('wait_kappa') * delta
    require(math.isclose(finite(components, 'wait'), expected_wait, rel_tol=1e-10, abs_tol=1e-9),
            'waiting reward must be linear fleet increment; no clipping/rate conversion')
    for key in ('grade_action', 'grade_2h', 'grade_shift'):
        require(finite(components, key) == 0, 'grade reward must not enter production reward: ' + key)
    for key, tons, target, weight in (
        ('production', 'ore_tons_shift', 16000, config.get('w_production')),
        ('waste', 'waste_tons_shift', 4000, config.get('waste_shortfall_weight'))):
        actual = finite(components, key)
        expected = 0.0
        if terminal:
            q = finite(components, tons)
            require(q >= 0, 'negative actual production')
            expected = -weight * max(0.0, 1.0 - q / target)
        require(math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-9), 'shortfall reward mismatch: ' + key)
    require(math.isclose(reward, sum(components[k] for k in ADDITIVE_COMPONENTS), rel_tol=1e-10, abs_tol=1e-9),
            'production reward component sum mismatch')
    for w in windows:
        validate_window_summary(w, config=config)
