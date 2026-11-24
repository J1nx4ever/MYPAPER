"""Frozen V1.1 training reward over an unchanged, validated RPC v2 ledger.

RPC rewards, material summaries, four-window labels and masks are retained.
Only the TrainingBatch.rewards view changes, so inherited PPO target and update
code can continue using the original complete-shift validator and GAE.
"""
import copy
import hashlib
import json
import math
from types import MappingProxyType

import numpy as np

from python.constrained_batch import TrainingBatch, validate_rows
from python.reward_config import configuration_from_components

CONTRACT = 'mining-reward-mc-v1.1'
REWARD = MappingProxyType(dict(kappa=4e-5, w_o=4., w_w=4., beta=.25,
                              center=.27, scale=.01, gamma_reward=1, gamma_grade=1))
ANNOTATION = 'training_reward_v11'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _reward_values(rows, episodes):
    """Recompute the training signal only after the entire RPC ledger passes."""
    versions = validate_rows(rows, episodes)
    require(len(versions) == 1, 'online batch mixes policy versions')
    require(all(not r.get('critic_only', False) and
                r.get('behavior_kind') != 'anylogic_native' and
                r.get('old_log_prob') is not None for r in rows),
            'native/offline Critic-only rows cannot enter online PPO')
    values = np.asarray([r['reward'] for r in rows], dtype=np.float64).copy()
    provenance = []
    for episode in episodes:
        cfg = configuration_from_components(episode['start_info']['reward_components'])
        expected = {'wait_kappa': REWARD['kappa'], 'w_production': REWARD['w_o'],
                    'waste_shortfall_weight': REWARD['w_w'], 'target_grade': REWARD['center'],
                    'grade_lower': .26, 'grade_upper': .28}
        require(all(cfg.get(k) == v for k, v in expected.items()),
                'RPC configuration differs from frozen V1.1 reward contract')
        start, stop = episode['start'], episode['stop']
        terminal = rows[stop - 1]
        # validate_rows already proves exactly one true 28800s terminal,
        # independent window order and terminal Q/M material conservation.
        windows = episode['windows']
        ore = math.fsum(float(w['ore_tons']) for w in windows)
        metal = math.fsum(float(w['metal_tons']) for w in windows)
        require(math.isfinite(ore) and math.isfinite(metal) and
                ore >= 0 and 0 <= metal <= ore and (ore > 0 or metal == 0),
                'invalid actual full-shift ore/metal')
        grade = metal / ore if ore > 0 else None
        deviation = abs(grade - REWARD['center']) / REWARD['scale'] if ore > 0 else 0.
        center_penalty = -REWARD['beta'] * deviation
        values[stop - 1] += center_penalty
        require(np.isfinite(values[start:stop]).all(), 'nonfinite V1.1 training reward')
        provenance.append(dict(episode_id=episode['episode_id'], start=start, stop=stop,
                               terminal_time_seconds=terminal['settlement_info']['sim_time_seconds'],
                               Q_ore=ore, M_shift=metal, G_shift=grade, D_shift=deviation,
                               center_penalty=center_penalty,
                               rpc_reward_sum=math.fsum(float(r['reward']) for r in rows[start:stop]),
                               training_reward_sum=math.fsum(values[start:stop])))
    return values, provenance


def annotate_rows(rows, episodes):
    """Return independent copies; retain raw reward and add a checked training view."""
    require(not any(ANNOTATION in r for r in rows), 'V1.1 reward is already annotated')
    values, provenance = _reward_values(rows, episodes)
    copied_rows, copied_episodes = copy.deepcopy(rows), copy.deepcopy(episodes)
    for row, value in zip(copied_rows, values):
        row[ANNOTATION] = float(value)
    for episode, audit in zip(copied_episodes, provenance):
        require('reward_v11_provenance' not in episode, 'V1.1 episode already annotated')
        episode['reward_v11_provenance'] = dict(contract=CONTRACT, parameters=dict(REWARD), **audit)
    return copied_rows, copied_episodes


def _raw_batch_id(episodes, version):
    identity = json.dumps([e['episode_id'] for e in episodes]) + version
    return hashlib.sha256(identity.encode()).hexdigest()


def _training_batch_id(raw_batch_id):
    return hashlib.sha256((raw_batch_id + ':' + CONTRACT).encode()).hexdigest()


class V11TrainingBatch(TrainingBatch):
    """An online batch whose raw ledger remains compatible with validate_rows."""
    reward_contract = CONTRACT

    @classmethod
    def from_rows(cls, rows, episodes, *, policy_version=None, rpc_batch_id=None):
        copied_rows, copied_episodes = annotate_rows(rows, episodes)
        version = next(iter({r['policy_version'] for r in copied_rows}))
        require(policy_version is None or policy_version == version, 'batch policy version mismatch')
        raw_id = rpc_batch_id or _raw_batch_id(copied_episodes, version)
        require(isinstance(raw_id, str) and bool(raw_id), 'explicit raw batch identity required')
        result = cls(copied_rows, copied_episodes, version, _training_batch_id(raw_id))
        result.rpc_batch_id = raw_id
        result.reward_provenance = [copy.deepcopy(e['reward_v11_provenance']) for e in copied_episodes]
        validate_training_batch(result)
        return result

    @classmethod
    def from_buffer(cls, buffer):
        return training_batch(TrainingBatch.from_buffer(buffer))

    @property
    def rewards(self):
        # Revalidate and recompute from raw RPC every time targets are requested.
        # No stored annotation can silently replace the actual training signal.
        return validate_training_batch(self)


def training_batch(batch):
    """Convert one raw complete online batch; reject double transformation."""
    require(isinstance(batch, TrainingBatch), 'TrainingBatch required')
    require(not isinstance(batch, V11TrainingBatch), 'V1.1 batch is already transformed')
    return V11TrainingBatch.from_rows(batch.rows, batch.episodes,
                                     policy_version=batch.policy_version, rpc_batch_id=batch.batch_id)


def validate_training_batch(batch):
    """Check wrapper identity, raw ledger, annotations and reward provenance."""
    require(isinstance(batch, V11TrainingBatch), 'V1.1 training batch required')
    require(batch.reward_contract == CONTRACT, 'training reward contract mismatch')
    require(hasattr(batch, 'rpc_batch_id') and
            batch.batch_id == _training_batch_id(batch.rpc_batch_id), 'V1.1 batch identity mismatch')
    values, provenance = _reward_values(batch.rows, batch.episodes)
    require(all(r['policy_version'] == batch.policy_version for r in batch.rows), 'batch policy version mismatch')
    require(all(ANNOTATION in r for r in batch.rows), 'missing V1.1 training reward annotation')
    stored = np.asarray([r[ANNOTATION] for r in batch.rows], dtype=np.float64)
    require(np.array_equal(stored, values), 'V1.1 training reward annotation mismatch')
    expected = [dict(contract=CONTRACT, parameters=dict(REWARD), **p) for p in provenance]
    require([e.get('reward_v11_provenance') for e in batch.episodes] == expected and
            getattr(batch, 'reward_provenance', None) == expected, 'V1.1 reward provenance mismatch')
    return values
