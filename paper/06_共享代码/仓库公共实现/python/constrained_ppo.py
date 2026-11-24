"""V1-C: four independent critics, per-episode GAE, fixed-weight and Lagrangian PPO.

This module is additive: it never imports or edits `episode_budget_ppo.py`, and it
does not change the collector, the reward adapter or the masked policy. The legacy
`python.train` entry keeps its own rollout, 128-step update and GAE.

Contracts implemented here
--------------------------
* Four separate critics: reward, grade, ore, waste. The grade critic additionally
  consumes `grade_failed_before` from the stored transition.
* GAE is computed per complete episode with gamma=1, lambda=0.95 and a true
  terminal value of zero. Recursion never crosses an episode boundary.
* Every transition keeps the stored weight (1/8 per transition). Losses are
  weighted means, so contributions stay proportional to episode length; nothing is
  divided by an episode's decision count.
* The actor update reuses the stored old log probability, the sampled mask and the
  clipped surrogate, so the first epoch has ratio 1 and approximate KL 0.
* B2 uses fixed cost weights; B3 additionally carries three non-negative
  multipliers that are updated only from complete eight-episode batches. Apart from
  the coefficients both modes share networks, sampling and optimizer settings.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .constrained_adapter import WEIGHT_CONTRACT_ID, require
from .constrained_batch import BATCH_EPISODES, policy_digest
from .masked_policy import DispatchMaskedPolicy
from .observation_contract import OBSERVATION_SIZE

TRAINING_CONTRACT_ID = "dispatch-constrained-ppo-v1"
CRITIC_CONTRACT_ID = "dispatch-constrained-four-critics-v1"
OBJECTIVE = "(1/8) * sum_episode sum_transition; never divide by episode length"
CHECKPOINT_FORMAT = "dispatch-constrained-ppo-checkpoint-v1"
COST_NAMES = ("grade", "ore", "waste")
MODES = ("fixed_weights", "lagrangian")


def _validate_mode(mode):
    require(mode in MODES, f"unknown mode {mode!r}; expected one of {MODES}")


@dataclass(frozen=True)
class ConstrainedPPOConfig:
    """Training configuration; B2 and B3 differ only in multiplier bookkeeping."""

    mode: str = "fixed_weights"
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    n_epochs: int = 4
    minibatch_size: int = 128
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 1e-3
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    cost_weights: tuple = (1.0, 1.0, 1.0)
    cost_limits: tuple = (0.0, 0.0, 0.0)
    lambda_lr: float = 0.05
    lambda_max: float = 10.0
    hidden_sizes: tuple = (128, 128)
    seed: int = 11

    def __post_init__(self):
        _validate_mode(self.mode)
        require(self.gamma == 1.0, "V1-C requires gamma=1 for the four-channel GAE")
        require(0.0 < self.gae_lambda <= 1.0, "gae_lambda must be in (0, 1]")
        require(self.clip_range > 0.0, "clip_range must be positive")
        require(self.n_epochs >= 1 and self.minibatch_size >= 1, "invalid epoch/minibatch")
        require(self.actor_learning_rate > 0.0 and self.critic_learning_rate > 0.0, "invalid learning rate")
        require(self.ent_coef >= 0.0 and self.vf_coef >= 0.0, "invalid loss coefficients")
        require(len(self.cost_weights) == 3 and all(w >= 0.0 for w in self.cost_weights),
                "cost_weights must be three non-negative values")
        require(len(self.cost_limits) == 3 and all(c >= 0.0 for c in self.cost_limits),
                "cost_limits must be three non-negative values")
        require(self.lambda_lr >= 0.0 and self.lambda_max >= 0.0, "invalid multiplier schedule")
        require(len(self.hidden_sizes) >= 1 and all(h > 0 for h in self.hidden_sizes), "invalid critic sizes")

    def to_dict(self):
        payload = dict(self.__dict__)
        for key in ("cost_weights", "cost_limits", "hidden_sizes"):
            payload[key] = list(payload[key])
        payload.update(training_contract_id=TRAINING_CONTRACT_ID, critic_contract_id=CRITIC_CONTRACT_ID,
                       weight_contract_id=WEIGHT_CONTRACT_ID, objective=OBJECTIVE)
        return payload

    def with_updates(self, **kwargs):
        return replace(self, **kwargs)


class CriticNetwork(nn.Module):
    """One scalar value head over the stored observation (plus optional flag)."""

    def __init__(self, input_dim, hidden_sizes, seed):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            layers, last = [], input_dim
            for width in hidden_sizes:
                layers += [nn.Linear(last, width), nn.Tanh()]
                last = width
            layers.append(nn.Linear(last, 1))
            self.mlp = nn.Sequential(*layers)
            for module in self.mlp:
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                    nn.init.constant_(module.bias, 0.0)
            nn.init.orthogonal_(self.mlp[-1].weight, gain=1.0)

    def forward(self, inputs):
        return self.mlp(inputs).squeeze(-1)


class FourCritics(nn.Module):
    """Reward/Ore/Waste critics read 1519D; Grade reads 1519D plus the flag."""

    def __init__(self, hidden_sizes=(128, 128), seed=11):
        super().__init__()
        self.reward = CriticNetwork(OBSERVATION_SIZE, hidden_sizes, seed)
        self.grade = CriticNetwork(OBSERVATION_SIZE + 1, hidden_sizes, seed + 1)
        self.ore = CriticNetwork(OBSERVATION_SIZE, hidden_sizes, seed + 2)
        self.waste = CriticNetwork(OBSERVATION_SIZE, hidden_sizes, seed + 3)

    def forward_reward(self, observations):
        return self.reward(observations)

    def forward_costs(self, observations, history_before):
        require(history_before.shape == observations.shape[:1],
                "grade_failed_before must be one value per transition")
        graded = torch.cat([observations, history_before.unsqueeze(-1).to(observations.dtype)], dim=-1)
        return dict(grade=self.grade(graded), ore=self.ore(observations), waste=self.waste(observations))


@dataclass
class TrainingBatch:
    """Immutable episode-locally ordered tensors taken from one complete batch."""

    observations: torch.Tensor
    next_observations: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    rewards: torch.Tensor
    costs: torch.Tensor
    history_before: torch.Tensor
    history_after: torch.Tensor
    terminals: torch.Tensor
    weights: torch.Tensor
    episode_bounds: tuple
    policy_version: str
    provenance: dict

    @property
    def size(self):
        return int(self.observations.shape[0])

    @property
    def episode_count(self):
        return len(self.episode_bounds)

    @classmethod
    def from_rows(cls, rows, episodes):
        require(len(rows) > 0, "empty batch")
        require(len(episodes) == BATCH_EPISODES, "training requires exactly eight complete episodes")
        versions = {row["policy_version"] for row in rows}
        require(len(versions) == 1, "mixed policy versions in one batch")
        bounds, cursor = [], 0
        for episode in episodes:
            require(episode["start"] == cursor and episode["stop"] > cursor, "episode boundary gap/overlap")
            cursor = episode["stop"]
            bounds.append((int(episode["start"]), int(episode["stop"])))
        require(cursor == len(rows), "unassigned transitions")
        observations = torch.as_tensor(np.stack([np.asarray(r["observation"], dtype=np.float32) for r in rows]))
        next_observations = torch.as_tensor(np.stack([np.asarray(r["next_observation"], dtype=np.float32) for r in rows]))
        require(observations.shape[1:] == (OBSERVATION_SIZE,), "observations must be raw 1519D")
        weights = torch.as_tensor(np.asarray([float(r["sample_weight"]) for r in rows], dtype=np.float32))
        require(torch.allclose(weights, torch.full_like(weights, 1.0 / BATCH_EPISODES), atol=0, rtol=0),
                "transition weights must equal 1/8 for every transition")
        terminals = torch.as_tensor(np.asarray([bool(r["terminated"]) for r in rows]))
        for start, stop in bounds:
            require(bool(terminals[stop - 1]) and not bool(terminals[start:stop - 1].any()),
                    "each episode must end with exactly one terminal transition")
        history_before = torch.as_tensor(np.asarray([bool(r["history_before"]) for r in rows]))
        history_after = torch.as_tensor(np.asarray([bool(r["history_after"]) for r in rows]))
        grade_hit = torch.as_tensor(np.asarray([float(r["grade_cost"]) > 0 for r in rows]))
        require(torch.equal(history_after, history_before | grade_hit),
                "stored history_after disagrees with history_before/grade_cost")
        return cls(
            observations=observations, next_observations=next_observations,
            actions=torch.as_tensor(np.asarray([int(r["action"]) for r in rows], dtype=np.int64)),
            old_log_probs=torch.as_tensor(np.asarray([float(r["old_log_prob"]) for r in rows], dtype=np.float32)),
            rewards=torch.as_tensor(np.asarray([float(r["reward"]) for r in rows], dtype=np.float32)),
            costs=torch.as_tensor(np.stack([[float(r["grade_cost"]), float(r["ore_cost"]), float(r["waste_cost"])]
                                            for r in rows]).astype(np.float32)),
            history_before=history_before, history_after=history_after,
            terminals=terminals, weights=weights, episode_bounds=tuple(bounds),
            policy_version=versions.pop(),
            provenance=dict(episodes=len(episodes), transitions=len(rows),
                            seeds=[int(e["experiment_seed"]) for e in episodes],
                            objective=OBJECTIVE, weight_contract_id=WEIGHT_CONTRACT_ID))

    def episode_cost_totals(self):
        totals = np.zeros((self.episode_count, len(COST_NAMES)), dtype=np.float64)
        costs = self.costs.numpy()
        for index, (start, stop) in enumerate(self.episode_bounds):
            totals[index] = costs[start:stop].sum(axis=0)
        return totals


def gae_channel(values, next_values, signals, terminals, episode_bounds, gamma, lam):
    """Undiscounted-if-gamma-1 GAE with a hard episode boundary and zero terminal value."""
    values = np.asarray(values, dtype=np.float64)
    next_values = np.asarray(next_values, dtype=np.float64)
    signals = np.asarray(signals, dtype=np.float64)
    terminals = np.asarray(terminals, dtype=bool)
    require(values.shape == next_values.shape == signals.shape == terminals.shape, "GAE input shape mismatch")
    advantages = np.zeros_like(signals)
    returns = np.zeros_like(signals)
    for start, stop in episode_bounds:
        carry = 0.0
        for index in range(stop - 1, start - 1, -1):
            next_value = 0.0 if terminals[index] else next_values[index]
            delta = signals[index] + gamma * next_value - values[index]
            carry = delta + gamma * lam * carry
            advantages[index] = carry
            returns[index] = carry + values[index]
    return advantages, returns


def combined_advantages(reward_advantages, cost_advantages, coefficients, weights):
    """A = A_reward - sum_c w_c * A_cost_c, then a single batch-wide normalization."""
    coefficients = np.asarray(coefficients, dtype=np.float64)
    require(coefficients.shape == (len(COST_NAMES),), "one coefficient per cost channel")
    combined = np.asarray(reward_advantages, dtype=np.float64).copy()
    for index, name in enumerate(COST_NAMES):
        combined -= coefficients[index] * np.asarray(cost_advantages[name], dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    mean = float((combined * weights).sum() / weights.sum())
    variance = float((weights * (combined - mean) ** 2).sum() / weights.sum())
    scale = float(np.sqrt(variance))
    return (combined - mean) / (scale + 1e-8), dict(mean=mean, std=scale)


def _digest_payload(payload):
    """Byte-exact digest of a checkpoint payload, including tensors and RNG state."""
    digest = hashlib.sha256()

    def walk(value):
        if isinstance(value, torch.Tensor):
            digest.update(b"tensor" + str((value.dtype, tuple(value.shape))).encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(value, np.ndarray):
            digest.update(b"ndarray" + str((value.dtype, value.shape)).encode())
            digest.update(value.tobytes())
        elif isinstance(value, dict):
            digest.update(b"{")
            for key in sorted(value, key=str):
                digest.update(str(key).encode())
                walk(value[key])
            digest.update(b"}")
        elif isinstance(value, (list, tuple)):
            digest.update(b"[")
            for item in value:
                walk(item)
            digest.update(b"]")
        else:
            digest.update(type(value).__name__.encode())
            digest.update(repr(value).encode())

    walk(payload)
    return digest.hexdigest()


class ConstrainedPPO:
    """Four-critic constrained PPO over complete eight-episode batches."""

    def __init__(self, policy, config=None):
        require(type(policy) is DispatchMaskedPolicy, "only the audited DispatchMaskedPolicy is supported")
        require(policy.observation_space.shape == (OBSERVATION_SIZE,), "V1-C requires raw 1519D observations")
        self.config = config or ConstrainedPPOConfig()
        self.policy = policy
        self.critics = FourCritics(self.config.hidden_sizes, self.config.seed)
        self.lambdas = np.asarray(self.config.cost_weights, dtype=np.float64).copy()
        actor_parameters = [p for name, p in policy.named_parameters() if "value_net" not in name]
        require(len(actor_parameters) > 0, "no actor parameters found")
        self.actor_optimizer = torch.optim.Adam(actor_parameters, lr=self.config.actor_learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critics.parameters(), lr=self.config.critic_learning_rate)
        self.generator = torch.Generator(device="cpu").manual_seed(self.config.seed)
        self.update_index = 0
        policy.set_training_mode(True)
        for parameter in self.critics.parameters():
            require(parameter.requires_grad, "critic parameters must be trainable")

    # ------------------------------------------------------------------ helpers
    @property
    def coefficients(self):
        return np.asarray(self.config.cost_weights if self.config.mode == "fixed_weights" else self.lambdas,
                          dtype=np.float64)

    def actor_digest(self):
        return policy_digest(self.policy)

    def critic_digests(self):
        return {name: hashlib.sha256(
            json.dumps({k: v.detach().cpu().numpy().tolist()
                        for k, v in getattr(self.critics, name).state_dict().items()}).encode()).hexdigest()
            for name in ("reward",) + COST_NAMES}

    def _values(self, batch):
        with torch.no_grad():
            reward_values = self.critics.forward_reward(batch.observations).numpy().astype(np.float64)
            reward_next = self.critics.forward_reward(batch.next_observations).numpy().astype(np.float64)
            cost_values = {k: v.numpy().astype(np.float64)
                           for k, v in self.critics.forward_costs(batch.observations, batch.history_before).items()}
            # The successor state is conditioned on the post-transition history flag.
            cost_next = {k: v.numpy().astype(np.float64) for k, v in
                         self.critics.forward_costs(batch.next_observations, batch.history_after).items()}
        return reward_values, reward_next, cost_values, cost_next

    def compute_advantages(self, batch):
        """Episode-local four-channel GAE plus the unified combined advantage."""
        reward_values, reward_next, cost_values, cost_next = self._values(batch)
        terminals = batch.terminals.numpy()
        reward_adv, reward_returns = gae_channel(
            reward_values, reward_next, batch.rewards.numpy(), terminals, batch.episode_bounds,
            self.config.gamma, self.config.gae_lambda)
        cost_adv, cost_returns = {}, {}
        for name in COST_NAMES:
            cost_adv[name], cost_returns[name] = gae_channel(
                cost_values[name], cost_next[name], batch.costs[:, COST_NAMES.index(name)].numpy(), terminals,
                batch.episode_bounds, self.config.gamma, self.config.gae_lambda)
        coefficients = self.coefficients
        combined, stats = combined_advantages(reward_adv, cost_adv, coefficients, batch.weights.numpy())
        return dict(reward_advantages=reward_adv, reward_returns=reward_returns,
                    cost_advantages=cost_adv, cost_returns=cost_returns,
                    combined_advantages=combined, normalization=stats,
                    coefficients=coefficients.tolist(), values=dict(reward=reward_values, **cost_values))

    # ------------------------------------------------------------------- update
    def update(self, batch):
        """One PPO pass over a complete batch. Never touches multipliers."""
        require(batch.episode_count == BATCH_EPISODES, "updates consume complete eight-episode batches")
        advantages = self.compute_advantages(batch)
        require(np.isfinite(advantages["combined_advantages"]).all(), "non-finite combined advantage")
        combined = torch.as_tensor(advantages["combined_advantages"], dtype=torch.float32)
        returns = dict(reward=torch.as_tensor(advantages["reward_returns"], dtype=torch.float32),
                       **{k: torch.as_tensor(v, dtype=torch.float32)
                          for k, v in advantages["cost_returns"].items()})
        policy_before = self.actor_digest()
        critic_before = self.critic_digests()
        history = batch.history_before.float()

        metrics = []
        for _epoch in range(self.config.n_epochs):
            permutation = torch.randperm(batch.size, generator=self.generator)
            for offset in range(0, batch.size, self.config.minibatch_size):
                index = permutation[offset:offset + self.config.minibatch_size]
                weights = batch.weights[index]
                denominator = weights.sum().clamp_min(torch.finfo(torch.float32).tiny)

                _, log_prob, entropy = self.policy.evaluate_actions(batch.observations[index], batch.actions[index])
                log_ratio = log_prob - batch.old_log_probs[index]
                ratio = log_ratio.exp()
                surrogate = torch.max(-combined[index] * ratio,
                                      -combined[index] * ratio.clamp(1 - self.config.clip_range,
                                                                     1 + self.config.clip_range))
                policy_loss = (surrogate * weights).sum() / denominator - self.config.ent_coef * entropy.mean()
                self.actor_optimizer.zero_grad()
                policy_loss.backward()
                actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in self.policy.parameters() if p.grad is not None], self.config.max_grad_norm)
                self.actor_optimizer.step()

                predicted = dict(reward=self.critics.forward_reward(batch.observations[index]),
                                 **self.critics.forward_costs(batch.observations[index], history[index]))
                value_losses = {}
                total_value_loss = 0.0
                for name in ("reward",) + COST_NAMES:
                    squared = (predicted[name] - returns[name][index]) ** 2
                    value_losses[name] = float(((squared * weights).sum() / denominator).detach())
                    total_value_loss = total_value_loss + squared
                critic_loss = self.config.vf_coef * ((total_value_loss * weights).sum() / denominator)
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_grad_norm = torch.nn.utils.clip_grad_norm_(list(self.critics.parameters()),
                                                                  self.config.max_grad_norm)
                self.critic_optimizer.step()

                metrics.append(dict(
                    policy_loss=float(policy_loss.detach()), critic_loss=float(critic_loss.detach()),
                    value_losses=value_losses,
                    approx_kl=float(((ratio - 1.0) - log_ratio).mean().detach()),
                    clip_fraction=float((torch.abs(ratio - 1.0) > self.config.clip_range).float().mean().detach()),
                    entropy=float(entropy.mean().detach()),
                    actor_grad_norm=float(actor_grad_norm), critic_grad_norm=float(critic_grad_norm),
                    ratio_max_error=float((ratio - 1.0).abs().max().detach())))

        self._assert_finite()
        policy_after = self.actor_digest()
        critic_after = self.critic_digests()
        explained = {}
        with torch.no_grad():
            predictions = dict(reward=self.critics.forward_reward(batch.observations),
                               **self.critics.forward_costs(batch.observations, history))
        for name, target in returns.items():
            residual = (target - predictions[name]).numpy()
            variance = float(np.var(target.numpy()))
            explained[name] = (float(1.0 - np.var(residual) / variance) if variance > 0 else float("nan"))
        self.update_index += 1
        return dict(
            training_contract_id=TRAINING_CONTRACT_ID, mode=self.config.mode,
            transitions=batch.size, episodes=batch.episode_count, update_index=self.update_index,
            epochs=self.config.n_epochs, minibatches=len(metrics),
            policy_version=batch.policy_version, actor_digest_before=policy_before,
            actor_digest_after=policy_after, actor_changed=policy_before != policy_after,
            critic_digests_before=critic_before, critic_digests_after=critic_after,
            critics_changed={name: critic_before[name] != critic_after[name] for name in critic_before},
            coefficients=self.coefficients.tolist(), lambdas=self.lambdas.tolist(),
            normalization=advantages["normalization"],
            mean=dict(policy_loss=float(np.mean([m["policy_loss"] for m in metrics])),
                      critic_loss=float(np.mean([m["critic_loss"] for m in metrics])),
                      approx_kl=float(np.mean([m["approx_kl"] for m in metrics])),
                      clip_fraction=float(np.mean([m["clip_fraction"] for m in metrics])),
                      entropy=float(np.mean([m["entropy"] for m in metrics])),
                      actor_grad_norm=float(np.mean([m["actor_grad_norm"] for m in metrics])),
                      critic_grad_norm=float(np.mean([m["critic_grad_norm"] for m in metrics])),
                      first_minibatch_ratio_error=metrics[0]["ratio_max_error"]),
            value_losses=dict((name, float(np.mean([m["value_losses"][name] for m in metrics])))
                              for name in ("reward",) + COST_NAMES),
            explained_variance=explained,
            objective=OBJECTIVE, sample_weight=float(batch.weights[0]))

    def apply_multiplier_update(self, batch):
        """B3 only: dual ascent from complete-batch episode failure rates."""
        require(self.config.mode == "lagrangian", "fixed-weight (B2) mode keeps its coefficients")
        totals = batch.episode_cost_totals()
        rates = totals.mean(axis=0)
        limits = np.asarray(self.config.cost_limits, dtype=np.float64)
        before = self.lambdas.copy()
        updated = np.maximum(0.0, before + self.config.lambda_lr * (rates - limits))
        updated = np.minimum(updated, self.config.lambda_max)
        self.lambdas = updated
        return dict(mode=self.config.mode, episode_failure_rates=dict(zip(COST_NAMES, rates.tolist())),
                    episode_cost_totals=dict(zip(COST_NAMES, totals.sum(axis=0).tolist())),
                    cost_limits=dict(zip(COST_NAMES, limits.tolist())),
                    lambdas_before=dict(zip(COST_NAMES, before.tolist())),
                    lambdas_after=dict(zip(COST_NAMES, updated.tolist())),
                    lambda_lr=self.config.lambda_lr, lambda_max=self.config.lambda_max)

    def _assert_finite(self):
        for name, parameter in self.policy.named_parameters():
            require(torch.isfinite(parameter).all(), f"non-finite actor parameter after update: {name}")
        for name, parameter in self.critics.named_parameters():
            require(torch.isfinite(parameter).all(), f"non-finite critic parameter after update: {name}")

    # --------------------------------------------------------------- checkpoint
    def _payload(self, extra=None):
        payload = dict(
            format=CHECKPOINT_FORMAT, training_contract_id=TRAINING_CONTRACT_ID,
            critic_contract_id=CRITIC_CONTRACT_ID, config=self.config.to_dict(),
            policy_state=self.policy.state_dict(), critic_state=self.critics.state_dict(),
            actor_optimizer=self.actor_optimizer.state_dict(),
            critic_optimizer=self.critic_optimizer.state_dict(),
            lambdas=self.lambdas.tolist(), update_index=self.update_index,
            actor_digest=self.actor_digest(), critic_digests=self.critic_digests(),
            rng=dict(torch=torch.get_rng_state(), numpy=np.random.get_state(),
                     python=random.getstate(), generator=self.generator.get_state()))
        if extra:
            payload.update(extra)
        return payload

    def save_checkpoint(self, path, extra=None):
        payload = self._payload(extra)
        payload["integrity_sha256"] = _digest_payload(payload)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        return dict(path=str(path), update_index=self.update_index, actor_digest=payload["actor_digest"],
                    lambdas=self.lambdas.tolist())

    @classmethod
    def load_checkpoint(cls, path, policy):
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        require(payload.get("format") == CHECKPOINT_FORMAT, "unknown checkpoint format")
        expected = payload.pop("integrity_sha256", None)
        require(expected is not None and expected == _digest_payload(payload),
                "checkpoint integrity check failed")
        payload["integrity_sha256"] = expected
        config = ConstrainedPPOConfig(**{k: tuple(v) if isinstance(v, list) and k in
                                         ("cost_weights", "cost_limits", "hidden_sizes") else v
                                         for k, v in payload["config"].items()
                                         if k in ConstrainedPPOConfig.__dataclass_fields__})
        trainer = cls(policy, config)
        trainer.policy.load_state_dict(payload["policy_state"])
        trainer.critics.load_state_dict(payload["critic_state"])
        trainer.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        trainer.critic_optimizer.load_state_dict(payload["critic_optimizer"])
        trainer.lambdas = np.asarray(payload["lambdas"], dtype=np.float64)
        trainer.update_index = int(payload["update_index"])
        rng = payload["rng"]
        torch.set_rng_state(rng["torch"])
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])
        trainer.generator.set_state(rng["generator"])
        require(trainer.actor_digest() == payload["actor_digest"], "restored actor digests differ")
        require(trainer.critic_digests() == payload["critic_digests"], "restored critic digests differ")
        return trainer
