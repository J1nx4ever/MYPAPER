"""Frozen 128D-only pilot, retaining the matched constant raw branch."""
import hashlib
import json
import numpy as np

GROUPS = ('R-128',)
CANDIDATES = ('e01_R-128',)
SLICES = {'G1': (0, 44), 'G2': (44, 88), 'G3': (88, 128)}
VERSION = 'representation-only128-constant-raw-branch-v1'

def require(value, message):
    if not value:
        raise ValueError(message)

def group(candidate):
    require(candidate in CANDIDATES, 'unknown frozen representation candidate')
    return candidate.split('_', 1)[1]

def mask(candidate):
    group(candidate)
    return np.ones(128, dtype=np.float32)

def ablation_contract(candidate):
    return dict(version=VERSION, group=group(candidate), mask=mask(candidate).astype(int).tolist(),
                placement='after_fixed_physical_scaling_before_summary_encoder',
                scope='actor_and_critic_current_and_every_history_token',
                raw_observation_untouched=False,
                raw_learned_input='constant_zero1519_actor_and_critic',
                raw_transport_and_certification_preserved=True,
                legal_action_mask='external_constraint_also_present_in_full_G3',
                learned_state_information='current128_and_full_causal128_history',
                unused_raw_branch_retained_for_parameter_and_initialization_matching=True,
                learned_normalization=False,
                constant_is_information_suppression_not_a_physical_empty_state=True)

def updates(candidate):
    group(candidate)
    return 1

def global_update(candidate, local):
    count = updates(candidate)
    require(type(local) is int and 0 <= local <= count, 'invalid local update')
    return local

def schedule(engineering=False):
    require(engineering is True, 'only the one-seed 36-shift engineering pilot is prepared')
    return [('t0', 't0', 0, 2), ('train_u0001', 'train', 1, 32), ('final_u0001', 'final', 1, 2)]

def seed(namespace, kind, ordinal):
    # Keep all Java/JSON seed integers below 2**53; disjoint named streams.
    return int.from_bytes(hashlib.sha256(f'{namespace}:{kind}:{ordinal}'.encode()).digest()[:6], 'big') + 1

def parameter_digest(module):
    h = hashlib.sha256()
    for name, value in module.named_parameters():
        h.update(name.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()
