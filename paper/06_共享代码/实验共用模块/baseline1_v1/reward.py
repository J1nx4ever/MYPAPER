"""Fixed binary-window reward view over the unchanged production-v2 RPC ledger.

Raw rows are retained for the existing strict complete-shift validator. This
module neither creates environment data nor admits offline rows into PPO.
"""
import copy
import hashlib
import json
import math
from types import MappingProxyType

import numpy as np

from python.constrained_batch import TrainingBatch, validate_rows
from python.reward_config import configuration_from_components
from .prepared_batch import PreparedRows, freeze, thaw, readonly_array

CONTRACT = 'baseline1-fixed-binary-grade-v1'
FROZEN_RPC = MappingProxyType(dict(wait_kappa=4e-5, w_production=4.,
    waste_shortfall_weight=4., grade_lower=.26, grade_upper=.28, target_grade=.27,
    target_ore_tons_shift=16000., target_waste_tons_shift=4000.))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _beta(value):
    require(type(value) in (int, float) and math.isfinite(value) and value > 0,
            'beta must be explicit, finite and positive')
    return float(value)


def _raw_batch_id(episodes, version):
    identity = json.dumps([e['episode_id'] for e in episodes]) + version
    return hashlib.sha256(identity.encode()).hexdigest()


def _batch_id(raw_id, beta):
    return hashlib.sha256((raw_id + ':' + CONTRACT + ':' + float(beta).hex()).encode()).hexdigest()


def analyze_rewards(rows, episodes, *, beta):
    """Return (per-transition total rewards, per-shift three-ledger audits).

    Native rows are allowed for calibration/evaluation only. Calling this
    arithmetic helper does not verify file admission, real-source provenance,
    collection authorization or whether an Actor batch is still on-policy.
    """
    beta = _beta(beta)
    require(not any('training_reward_v11' in r for r in rows) and
            not any('reward_v11_provenance' in e for e in episodes),
            'V1.1 center reward annotations cannot enter Baseline 1')
    if isinstance(rows, PreparedRows):
        rows.assert_bound(episodes)
    else:
        validate_rows(rows, episodes)
    costs = np.asarray([r['grade_costs'] for r in rows], dtype=np.float64)
    require(costs.shape == (len(rows), 4) and np.isfinite(costs).all() and
            np.isin(costs, [0., 1.]).all(), 'invalid binary window costs')
    values = np.asarray([r['reward'] for r in rows], dtype=np.float64) - beta * costs.sum(axis=1)
    require(np.isfinite(values).all(), 'nonfinite total reward')
    audits = []
    for e in episodes:
        cfg = configuration_from_components(e['start_info']['reward_components'])
        require(all(cfg.get(k) == v for k, v in FROZEN_RPC.items()),
                'RPC reward configuration differs from frozen Baseline 1')
        start, stop = e['start'], e['stop']
        c = rows[stop - 1]['settlement_info']['reward_components']
        wait = float(c['wait_vehicle_seconds'])
        ore, waste = float(c['ore_tons_shift']), float(c['waste_tons_shift'])
        q_o = max(0., 1. - ore / cfg.get('target_ore_tons_shift'))
        q_w = max(0., 1. - waste / cfg.get('target_waste_tons_shift'))
        window_costs = [float(v) for v in e['window_costs']]
        require(np.array_equal(costs[start:stop].sum(axis=0), window_costs),
                'each final window cost must settle once')
        grade_cost = math.fsum(window_costs)
        wait_reward = math.fsum(float(r['settlement_info']['reward_components']['wait'])
                                for r in rows[start:stop])
        production_reward = math.fsum(
            float(r['settlement_info']['reward_components']['production']) +
            float(r['settlement_info']['reward_components']['waste']) for r in rows[start:stop])
        production_cost = cfg.get('w_production') * q_o + cfg.get('waste_shortfall_weight') * q_w
        grade_penalty = -beta * grade_cost
        raw_sum = math.fsum(float(r['reward']) for r in rows[start:stop])
        total = math.fsum(float(v) for v in values[start:stop])
        expected = -cfg.get('wait_kappa') * wait + grade_penalty - production_cost
        for observed, target, name in (
                (wait_reward, -cfg.get('wait_kappa') * wait, 'waiting'),
                (production_reward, -production_cost, 'production'),
                (raw_sum, wait_reward + production_reward, 'raw RPC'),
                (total, wait_reward + grade_penalty + production_reward, 'three-ledger'),
                (total, expected, 'episode')):
            require(math.isclose(observed, target, rel_tol=1e-10, abs_tol=1e-8),
                    name + ' reward conservation failed')
        audits.append(dict(contract=CONTRACT, beta=beta, episode_id=e['episode_id'],
            experiment_run_index=e['experiment_run_index'], experiment_seed=e['experiment_seed'],
            policy_version=e['policy_version'], start=start, stop=stop,
            wait_vehicle_seconds=wait, grade_cost=grade_cost, window_costs=window_costs,
            ore_tons=ore, waste_tons=waste, q_o=q_o, q_w=q_w, wait_reward=wait_reward,
            grade_penalty=grade_penalty, production_cost=production_cost,
            production_reward=production_reward, rpc_reward_sum=raw_sum, total_reward=total,
            expected_episode_reward=expected, ledger_residual=total - expected,
            kappa=cfg.get('wait_kappa'), w_o=cfg.get('w_production'),
            w_w=cfg.get('waste_shortfall_weight')))
    return values, audits


class SingleCriticTrainingBatch(TrainingBatch):
    """One scalar-reward online view; inherited boundaries/raw fields stay intact."""
    reward_contract = CONTRACT

    @classmethod
    def from_buffer(cls, buffer, *, beta):
        raw = TrainingBatch.from_buffer(buffer)
        raw.source_kind = getattr(buffer, 'source_kind', None)
        return training_batch(raw, beta=beta)

    @property
    def rewards(self):
        return validate_training_batch(self)


class PreparedSingleCriticTrainingBatch(SingleCriticTrainingBatch):
    """A sealed scalar-reward view; cache reuse cannot outlive its immutable input."""
    def __init__(self, raw, beta):
        if type(raw) is not TrainingBatch or not isinstance(raw.rows, PreparedRows):
            raise ValueError('validated immutable raw batch required')
        rows = raw.rows
        rows.assert_bound(raw.episodes)
        require(rows.online, 'native/offline Critic-only rows cannot enter online PPO')
        require(not rows.has_v11, 'V1.1 center reward annotations cannot enter Baseline 1')
        require(rows.versions == {raw.policy_version} and len(rows.versions) == 1,
                'online batch mixes policy versions')
        require(raw.batch_id == _raw_batch_id(raw.episodes, raw.policy_version), 'raw batch identity mismatch')
        self.rows, self.episodes = rows, rows.episodes
        self.policy_version = raw.policy_version
        self.beta = _beta(beta)
        self.rpc_batch_id = raw.batch_id
        self.batch_id = _batch_id(raw.batch_id, self.beta)
        self.source_kind = getattr(raw, 'source_kind', None)
        require(self.source_kind is None or type(self.source_kind) is str,
                'prepared source kind must be an immutable string or None')
        values, audits = analyze_rewards(rows, rows.episodes, beta=self.beta)
        self._values = readonly_array(values)
        self._audits = freeze(audits)
        self._binding = (rows, rows.episodes, self.policy_version, self.beta, self.rpc_batch_id,
                         self.batch_id, self.source_kind, self._values, self._audits)
        self._sealed = True

    def __setattr__(self, name, value):
        if self.__dict__.get('_sealed', False):
            raise TypeError('prepared training batch is immutable')
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        raise TypeError('prepared training batch is immutable')

    def __deepcopy__(self, memo): return self

    @property
    def reward_provenance(self): return thaw(self._audits)
    @property
    def observations(self): return self.rows.column('observation')
    @property
    def next_observations(self): return self.rows.column('next_observation')

    def __getattr__(self, name):
        key = {'actions':'action', 'old_log_probs':'old_log_prob'}.get(name, name)
        if key in ('action','old_log_prob','grade_costs','grade_valid','grade_bootstrap',
                   'grade_trace','reward_bootstrap','reward_trace'):
            return self.rows.column(key)
        raise AttributeError(name)

    def cached_rewards(self, *, beta=None, contract=CONTRACT):
        require(self.__dict__.get('_sealed') is True, 'prepared reward certificate missing')
        actual = (self.rows, self.episodes, self.policy_version, self.beta, self.rpc_batch_id,
                  self.batch_id, self.source_kind, self._values, self._audits)
        require(all(actual[i] is self._binding[i] for i in (0,1,7,8))
                and all(type(a) is type(b) and a == b for a, b in zip(actual[2:7], self._binding[2:7])),
                'prepared reward binding changed')
        self.rows.assert_bound(self.episodes)
        require(contract == CONTRACT and self.reward_contract == CONTRACT, 'training reward contract mismatch')
        require(beta is None or _beta(beta) == self.beta, 'training beta mismatch')
        return self._values


def training_batch(raw, *, beta):
    require(type(raw) is TrainingBatch, 'raw TrainingBatch required; already transformed views are refused')
    beta = _beta(beta)
    if isinstance(raw.rows, PreparedRows):
        return PreparedSingleCriticTrainingBatch(raw, beta)
    rows, episodes = copy.deepcopy(raw.rows), copy.deepcopy(raw.episodes)
    versions = validate_rows(rows, episodes)
    require(len(versions) == 1 and versions == {raw.policy_version}, 'online batch mixes policy versions')
    require(all(not r.get('critic_only', False) and r.get('behavior_kind') != 'anylogic_native'
                and r.get('old_log_prob') is not None for r in rows),
            'native/offline Critic-only rows cannot enter online PPO')
    require(raw.batch_id == _raw_batch_id(episodes, raw.policy_version), 'raw batch identity mismatch')
    _, audits = analyze_rewards(rows, episodes, beta=beta)
    result = SingleCriticTrainingBatch(rows, episodes, raw.policy_version, _batch_id(raw.batch_id, beta))
    result.beta = beta
    result.rpc_batch_id = raw.batch_id
    result.source_kind = getattr(raw, 'source_kind', None)
    result.reward_provenance = audits
    validate_training_batch(result, beta=beta)
    return result


def validate_training_batch(batch, *, beta=None, contract=CONTRACT):
    require(isinstance(batch, SingleCriticTrainingBatch), 'Baseline 1 training batch required')
    if isinstance(batch, PreparedSingleCriticTrainingBatch):
        return batch.cached_rewards(beta=beta, contract=contract)
    actual_beta = _beta(batch.beta)
    require(contract == CONTRACT and batch.reward_contract == contract, 'training reward contract mismatch')
    require(beta is None or actual_beta == _beta(beta), 'training beta mismatch')
    require(batch.rpc_batch_id == _raw_batch_id(batch.episodes, batch.policy_version) and
            batch.batch_id == _batch_id(batch.rpc_batch_id, actual_beta), 'training batch identity mismatch')
    require(all(r['policy_version'] == batch.policy_version and
                not r.get('critic_only', False) and r.get('behavior_kind') != 'anylogic_native'
                and r.get('old_log_prob') is not None for r in batch.rows),
            'online policy/native provenance mismatch')
    values, audits = analyze_rewards(batch.rows, batch.episodes, beta=actual_beta)
    require(batch.reward_provenance == audits, 'three-ledger provenance mismatch')
    return values
