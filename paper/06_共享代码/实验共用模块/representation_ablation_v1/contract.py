"""Frozen design, independent identities, and post-scaling ablation masks."""
import hashlib
import json
import numpy as np

GROUPS = ('R0', 'R-full', 'R-G1', 'R-G2', 'R-G3')
CANDIDATES = tuple(f'{s}_{g}' for s in ('e01', 's01') for g in GROUPS)
SLICES = {'G1': (0, 44), 'G2': (44, 88), 'G3': (88, 128)}
VERSION = 'representation-ablation-post-physical-scaling-v1'

def require(value, message):
    if not value:
        raise ValueError(message)

def group(candidate):
    require(candidate in CANDIDATES, 'unknown frozen representation candidate')
    return candidate.split('_', 1)[1]

def mask(candidate):
    g = group(candidate)
    result = np.ones(128, dtype=np.float32)
    if g == 'R0':
        result[:] = 0
    elif g != 'R-full':
        start, end = SLICES[g[2:]]
        result[start:end] = 0
    return result

def ablation_contract(candidate):
    return dict(version=VERSION, group=group(candidate), mask=mask(candidate).astype(int).tolist(),
                placement='after_fixed_physical_scaling_before_summary_encoder',
                scope='actor_and_critic_current_and_every_history_token',
                raw_observation_untouched=True, learned_normalization=False,
                constant_is_information_suppression_not_a_physical_empty_state=True)

def updates(candidate):
    group(candidate)
    return 1 if candidate.startswith('e') else 8

def global_update(candidate, local):
    count = updates(candidate)
    require(type(local) is int and 0 <= local <= count, 'invalid local update')
    seed = int(candidate[1:3]) - 1
    return (seed * 5 + GROUPS.index(group(candidate))) * count + local

def schedule(engineering=False):
    if engineering:
        return [('t0', 't0', 0, 2), ('train_u0001', 'train', 1, 32), ('final_u0001', 'final', 1, 2)]
    result = [('t0', 't0', 0, 16)]
    for u in range(1, 9):
        result += [(f'train_u{u:04d}', 'train', u, 32), (f'monitor_u{u:04d}', 'monitor', u, 16)]
    return result + [('final_u0008', 'final', 8, 32)]

def seed(namespace, kind, ordinal):
    # Keep all Java/JSON seed integers below 2**53; disjoint named streams.
    return int.from_bytes(hashlib.sha256(f'{namespace}:{kind}:{ordinal}'.encode()).digest()[:6], 'big') + 1

def parameter_digest(module):
    h = hashlib.sha256()
    for name, value in module.named_parameters():
        h.update(name.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()
