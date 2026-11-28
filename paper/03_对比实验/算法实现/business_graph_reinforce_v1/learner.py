"""Apply Tang et al.'s episode-level policy gradient to the BG128 joint cost."""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path

import numpy as np
import torch

from business_graph_summary_v1 import learner as bg128
from business_graph_summary_v1.features_cache import BatchFeatureCache
from business_graph_summary_v1.models import actor_from_config, digest


CHECKPOINT_SCHEMA = 'business-graph-full-shift-reinforce-checkpoint-v1'
SETTINGS_SCHEMA = 'business-graph-full-shift-reinforce-v1'
REPLAY_CHUNK = 16


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _implementation():
    folder = Path(__file__).resolve().parent
    local = {name: bg128._file_sha(folder / name) for name in ('learner.py', '__init__.py')}
    return dict(reinforce_sources=local, actor_and_admission=bg128.implementation_signature())


class FullShiftReinforce:
    """One Adam step per frozen batch of complete, independent BG128 shifts.

    The Actor descriptor retains the original BG128 artifact format so the RPC
    collector can verify the same schema. Algorithm settings live separately.
    """

    save_actor = bg128.JointCostPPO.save_actor
    bind_actor_snapshot = bg128.JointCostPPO.bind_actor_snapshot
    _validated_actor_snapshot = bg128.JointCostPPO._validated_actor_snapshot
    _prefix = bg128.JointCostPPO._prefix

    def __init__(self, actor=None, *, actor_config=None, actor_lr=3e-4,
                 baseline_decay=.5, episodes_per_batch=None, adam_eps=1e-5):
        self.config = bg128.validate_config(bg128.default_config() if actor_config is None else actor_config)
        self.config_sha256 = bg128._hash(self.config)
        shift_count = self.config['ppo']['episodes_per_batch']
        if episodes_per_batch is None:
            episodes_per_batch = shift_count
        _require(type(episodes_per_batch) is int and episodes_per_batch == shift_count,
                 'shift count must match the BG128 Actor artifact profile')
        _require(all(type(value) in (int, float) and math.isfinite(value)
                     for value in (actor_lr, baseline_decay, adam_eps))
                 and actor_lr > 0 and adam_eps > 0 and 0 <= baseline_decay < 1,
                 'invalid REINFORCE optimizer or EMA coefficients')
        self.settings = dict(schema=SETTINGS_SCHEMA, actor_lr=float(actor_lr), adam_eps=float(adam_eps),
                             baseline_decay=float(baseline_decay), episodes_per_batch=episodes_per_batch)
        self.settings_sha256 = bg128._hash(self.settings)
        with torch.random.fork_rng(devices=[]):
            self.actor = actor_from_config(self.config)
        if actor is not None:
            source = actor.state_dict()
            bg128._validate_model_state(source, self.actor)
            self.actor.load_state_dict(source, strict=True)
        self.actor.eval()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, eps=adam_eps)
        self.baseline = 0.0
        self.update_index = 0
        self.consumed_batches = {}
        self.consumed_episodes = {}
        self.runtime_state = {}
        self._actor_artifact = None
        self._failed_update = False
        self._summary_cache = self._new_summary_cache()

    @property
    def policy_version(self):
        return digest(self.actor)

    def _new_summary_cache(self):
        resources = self.config['resources']
        return BatchFeatureCache(feature_config=self.config['feature_config'],
                                 max_tokens=resources['max_tokens'], max_bytes=resources['max_prefix_bytes'])

    def _check_settings(self):
        _require(bg128._hash(self.config) == self.config_sha256 and
                 bg128._hash(self.settings) == self.settings_sha256,
                 'Actor descriptor or REINFORCE settings changed after initialization')

    def _log_chunks(self, episode):
        states = np.asarray(episode['states'])[:-1]
        actions = np.asarray(episode['actions'])
        for start in range(0, len(states), REPLAY_CHUNK):
            indices = list(range(start, min(start + REPLAY_CHUNK, len(states))))
            distribution = self.actor.distribution(torch.tensor(states[indices].copy()), self._prefix(states, indices))
            yield distribution.log_prob(torch.tensor(actions[indices], dtype=torch.int64))

    def _policy_rows(self, episode):
        # Input expected by the inherited admission/replay checks. There is no
        # entropy term in the REINFORCE estimator.
        return torch.cat(tuple(self._log_chunks(episode))), None

    def update(self, episodes, *, batch_id=None):
        """Validate the complete on-policy batch and consume it at most once."""
        _require(not self._failed_update, 'cannot continue a partially failed optimizer update')
        self._check_settings()
        self._summary_cache = self._new_summary_cache()
        try:
            return self._update(episodes, batch_id=batch_id)
        finally:
            self._summary_cache = self._new_summary_cache()
            self.actor_optimizer.zero_grad(set_to_none=True)

    def _update(self, episodes, *, batch_id):
        # Reuse physical target reconstruction, shift closure, legal masks,
        # frozen Actor byte binding and online log-probability certification.
        episodes = bg128.JointCostPPO._validate_episodes(self, episodes, training=True)
        episode_keys = [bg128._episode_key(episode) for episode in episodes]
        _require(len({episode['identity']['run_index'] for episode in episodes}) == len(episodes),
                 'duplicate physical run indices in training batch')
        if batch_id is None:
            batch_id = bg128._hash(episode_keys)
        _require(type(batch_id) is str and bool(batch_id) and batch_id not in self.consumed_batches,
                 'invalid or already consumed batch ID')

        returns = [-math.fsum(episode['targets'].step_costs) for episode in episodes]
        baseline_before = self.baseline
        mean_return = math.fsum(returns) / len(returns)
        decay = self.settings['baseline_decay']
        baseline_after = decay * baseline_before + (1 - decay) * mean_return
        _require(all(math.isfinite(value) for value in (mean_return, baseline_before, baseline_after)),
                 'nonfinite REINFORCE return or baseline')

        self.actor_optimizer.zero_grad(set_to_none=True)
        losses = []
        for episode, shift_return in zip(episodes, returns, strict=True):
            # Each episode contributes its summed log probability. Chunks
            # only bound the size of the autograd graph. Adam sees their sum.
            coefficient = -(shift_return - baseline_before) / len(episodes)
            chunk_sums = []
            for log_probs in self._log_chunks(episode):
                log_sum = log_probs.sum(dtype=torch.float64)
                loss = coefficient * log_sum
                _require(bool(torch.isfinite(loss)), 'nonfinite REINFORCE actor loss')
                loss.backward()
                chunk_sums.append(float(log_sum.detach()))
            losses.append(coefficient * math.fsum(chunk_sums))

        gradients = [parameter.grad for parameter in self.actor.parameters() if parameter.grad is not None]
        _require(bool(gradients) and bg128._finite_tree(gradients), 'nonfinite or missing Actor gradients')
        grad_norm = math.sqrt(math.fsum(float(gradient.double().square().sum()) for gradient in gradients))
        rollout_actor_digest = self.policy_version
        self._failed_update = True
        self.actor_optimizer.step()
        _require(bg128._finite_tree(self.actor.state_dict()) and
                 bg128._finite_tree(self.actor_optimizer.state_dict()), 'optimizer produced nonfinite Actor or moments')

        receipt = dict(batch_id=batch_id, update_index=self.update_index + 1, episode_keys=episode_keys,
                       episodes=len(episodes), transitions=sum(len(episode['actions']) for episode in episodes),
                       actor_optimizer_steps=1, baseline_before=baseline_before, baseline_after=baseline_after,
                       mean_return=mean_return, mean_combined_cost=-mean_return,
                       policy_loss=math.fsum(losses), actor_grad_norm=grad_norm,
                       rollout_actor_digest=rollout_actor_digest, actor_digest=self.policy_version,
                       settings_sha256=self.settings_sha256)
        self.baseline = baseline_after
        self.update_index = receipt['update_index']
        self.consumed_batches[batch_id] = receipt
        self.consumed_episodes.update(dict.fromkeys(episode_keys, batch_id))
        self._actor_artifact = None
        self._failed_update = False
        return deepcopy(receipt)

    def save_checkpoint(self, path):
        """Publish Actor, Adam, EMA and batch history with byte verification."""
        _require(not self._failed_update, 'cannot seal a partially failed optimizer update')
        self._check_settings()
        bg128._canonical(self.runtime_state)
        _require(bg128._finite_tree(self.actor.state_dict()) and
                 bg128._finite_tree(self.actor_optimizer.state_dict()), 'cannot seal nonfinite optimizer state')
        metadata = dict(schema=CHECKPOINT_SCHEMA, actor_config_sha256=self.config_sha256,
                        settings_sha256=self.settings_sha256, actor_digest=self.policy_version,
                        update_index=self.update_index)
        value = dict(**metadata, actor_config=self.config, settings=self.settings, actor=self.actor.state_dict(),
                     optimizer=self.actor_optimizer.state_dict(), baseline=self.baseline,
                     consumed_batches=self.consumed_batches, consumed_episodes=self.consumed_episodes,
                     runtime_state=self.runtime_state, implementation=_implementation())
        return bg128._atomic_torch(path, value, metadata=metadata)

    @classmethod
    def from_checkpoint(cls, path, *, expected_actor_config=None, expected_sha256=None, expected_settings=None):
        """Restore exact training state from a sealed REINFORCE checkpoint."""
        if expected_sha256 is not None:
            _require(bg128._file_sha(path) == expected_sha256, 'REINFORCE checkpoint expected byte SHA differs')
        saved, manifest = bg128._load_checkpoint(path)
        _require(saved.get('schema') == manifest.get('schema') == CHECKPOINT_SCHEMA,
                 'REINFORCE checkpoint schema differs')
        _require(saved['implementation'] == _implementation(), 'REINFORCE implementation/source binding differs')
        saved_settings = saved['settings']
        learner = cls(actor_config=saved['actor_config'], actor_lr=saved_settings['actor_lr'],
                      adam_eps=saved_settings['adam_eps'], baseline_decay=saved_settings['baseline_decay'],
                      episodes_per_batch=saved_settings['episodes_per_batch'])
        _require(saved_settings == learner.settings and all(saved[key] == manifest.get(key) == expected
                 for key, expected in (('actor_config_sha256', learner.config_sha256),
                                       ('settings_sha256', learner.settings_sha256))),
                 'REINFORCE settings or Actor descriptor binding differs')
        if expected_actor_config is not None:
            _require(learner.config == bg128.validate_config(expected_actor_config),
                     'REINFORCE checkpoint Actor descriptor differs from campaign')
        if expected_settings is not None:
            _require(learner.settings == expected_settings, 'REINFORCE checkpoint settings differ from campaign')

        bg128._validate_model_state(saved['actor'], learner.actor)
        parameters = list(learner.actor.parameters())
        bg128._validate_optimizer(saved['optimizer'], learner.actor_optimizer.state_dict(), parameters)
        learner.actor.load_state_dict(saved['actor'], strict=True)
        learner.actor_optimizer.load_state_dict(saved['optimizer'])
        _require(learner.policy_version == manifest.get('actor_digest') == saved['actor_digest'],
                 'REINFORCE checkpoint Actor fingerprint differs')
        learner._restore_history(saved, manifest)
        return learner

    def _restore_history(self, saved, manifest):
        index, batches = saved['update_index'], saved['consumed_batches']
        _require(type(index) is int and index >= 0 and index == len(batches) == manifest.get('update_index'),
                 'REINFORCE checkpoint update history differs')
        baseline = 0.0
        decay = self.settings['baseline_decay']
        episodes = {}
        previous_actor = None
        for update, (batch_id, receipt) in enumerate(batches.items(), start=1):
            keys = receipt['episode_keys']
            _require(receipt['batch_id'] == batch_id and receipt['update_index'] == update and
                     receipt['actor_optimizer_steps'] == 1 and receipt['baseline_before'] == baseline and
                     receipt['settings_sha256'] == self.settings_sha256 and
                     len(keys) == len(set(keys)) == self.settings['episodes_per_batch'] and
                     episodes.keys().isdisjoint(keys) and bg128._finite_tree(receipt) and
                     (previous_actor is None or receipt['rollout_actor_digest'] == previous_actor),
                     'REINFORCE checkpoint batch history differs')
            baseline = decay * baseline + (1 - decay) * receipt['mean_return']
            _require(receipt['baseline_after'] == baseline, 'REINFORCE checkpoint EMA history differs')
            previous_actor = receipt['actor_digest']
            episodes.update(dict.fromkeys(keys, batch_id))
        steps = {int(state['step'].item()) for state in self.actor_optimizer.state.values()}
        _require((index == 0 and not steps) or (steps and min(steps) > 0 and max(steps) == index),
                 'REINFORCE checkpoint Adam step count differs')
        _require(episodes == saved['consumed_episodes'] and baseline == saved['baseline'] and
                 (previous_actor is None or previous_actor == self.policy_version),
                 'REINFORCE checkpoint episode history, Actor, or baseline differs')
        bg128._canonical(saved['runtime_state'])
        self.baseline = baseline
        self.update_index = index
        self.consumed_batches = batches
        self.consumed_episodes = episodes
        self.runtime_state = saved['runtime_state']
