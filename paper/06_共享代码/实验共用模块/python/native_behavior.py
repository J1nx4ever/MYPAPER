"""Record an AnyLogic native decision, never compute or optimize its action.

The proposed action is metadata outside the 1519D observation. No conditional
log probability is asserted for a rule with simulator/plan state unavailable to
the network. Native rows are Critic-only and must never become PPO rollouts.
"""
import copy
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from .constrained_adapter import require

NAME = 'anylogic_native'
PREFIX = 'config.offline_native.'
PROPOSAL = 'offline_native.proposed_action'


def validate_spec(spec, *, check_files=False):
    require(set(spec) == {'name', 'version', 'policy_id', 'implementation_revision',
                         'rng', 'settings', 'artifacts'}, 'explicit native behavior specification required')
    require(spec['name'] == NAME and spec['version'] == '1', 'unsupported native behavior version')
    require(type(spec['policy_id']) is int and spec['policy_id'] in (0, 1, 2, 4), 'unsupported native policy')
    require(isinstance(spec['implementation_revision'], str) and spec['implementation_revision'].strip(),
            'missing native implementation revision')
    expected_rng = 'anylogic_shared_experiment_seed' if spec['policy_id'] == 0 else 'no_dispatch_rng'
    require(spec['rng'] == expected_rng, 'unsupported native dispatch RNG semantics')
    require(isinstance(spec['settings'], dict) and bool(spec['settings']), 'explicit native settings required')
    require(isinstance(spec['artifacts'], list) and bool(spec['artifacts']), 'native source/plan artifacts required')
    seen = set()
    for item in spec['artifacts']:
        require(set(item) == {'role', 'path', 'sha256'} and isinstance(item['role'], str)
                and bool(item['role'].strip()), 'invalid native artifact')
        require(isinstance(item['path'], str) and Path(item['path']).is_absolute(), 'absolute artifact path required')
        require(item['path'] not in seen, 'duplicate native artifact')
        seen.add(item['path'])
        require(isinstance(item['sha256'], str) and re.fullmatch('[0-9a-f]{64}', item['sha256']), 'invalid artifact SHA256')
        if check_files:
            require(hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() == item['sha256'],
                    'native artifact changed: ' + item['path'])
    require(any(a['role'] == 'implementation' for a in spec['artifacts']), 'native implementation artifact required')
    if spec['policy_id'] == 4:
        require(any(a['role'] == 'plan' for a in spec['artifacts']), 'policy4 plan artifact required')
        require(all(k in spec['settings'] for k in ('solver', 'solver_version', 'solver_seed', 'plan_generation', 'tie_break', 'fallback')),
                'explicit policy4 plan generation and fallback settings required')
        plan = spec['settings'].get('plan_trips')
        require(isinstance(plan, list) and len(plan)==4 and all(
            isinstance(w,list) and len(w)==3 and all(isinstance(t,list) and len(t)==4
                and all(type(n) is int and n>=0 for n in t) for t in w) for w in plan),
                'policy4 requires explicit nonnegative integer plan_trips[4][3][4]')
    # Also rejects NaN/Infinity before an identity can be generated.
    json.dumps(spec, allow_nan=False)


def identity(spec):
    return hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def runtime_fields(spec):
    digest = identity(spec)
    fields = {PREFIX+'enabled': 1., PREFIX+'version': 1., PREFIX+'policy_id': float(spec['policy_id']),
            **{PREFIX+f'behavior_word_{i}': float(int(digest[8*i:8*i+8], 16)) for i in range(8)}}
    if spec['policy_id']==4:
        fields.update({PREFIX+f'plan_{w}_{t}_{s}':float(spec['settings']['plan_trips'][w][t][s])
                       for w in range(4) for t in range(3) for s in range(4)})
    return fields


def validate_runtime(info, spec, *, terminal=False):
    c = info['reward_components']
    expected = runtime_fields(spec)
    require({k:v for k,v in c.items() if k.startswith(PREFIX)} == expected,
            'native runtime identity differs from frozen behavior')
    if terminal:
        require(PROPOSAL not in c, 'terminal must not propose an action')
        return None
    value = c.get(PROPOSAL)
    require(type(value) in (int, float) and value in (0, 1, 2, 3), 'missing/invalid native proposed action')
    action = int(value)
    require(info['action_mask'][action] == 1, 'native proposed action is masked')
    return action


class NativeBehavior:
    def __init__(self, spec):
        validate_spec(spec)
        self.spec = copy.deepcopy(spec)
        self.policy_version = identity(spec)

    def assert_frozen(self):
        require(identity(self.spec) == self.policy_version, 'native behavior specification changed')

    def sample_with_info(self, obs, mask, info):
        self.assert_frozen()
        require(np.array_equal(np.asarray(obs)[19:23], mask) and list(mask) == info['action_mask'],
                'native behavior mask differs from observation')
        return validate_runtime(info, self.spec), None

    def row_metadata(self, info):
        return dict(behavior_kind=NAME, behavior_spec=copy.deepcopy(self.spec),
                    action_info=copy.deepcopy(info), log_prob_available=False, critic_only=True)


def validate_row(row, action_info):
    require(row.get('behavior_kind') == NAME and row.get('critic_only') is True
            and row.get('log_prob_available') is False and row['old_log_prob'] is None,
            'native rows must be explicitly Critic-only with unavailable log probability')
    spec = row['behavior_spec']
    validate_spec(spec)
    require(identity(spec) == row['policy_version'], 'native policy version mismatch')
    require(row['action_info'] == action_info, 'native action evidence differs from preceding state')
    require(validate_runtime(action_info, spec) == row['action'], 'native action differs from proposal')
    validate_runtime(row['settlement_info'], spec, terminal=row['terminated'])
