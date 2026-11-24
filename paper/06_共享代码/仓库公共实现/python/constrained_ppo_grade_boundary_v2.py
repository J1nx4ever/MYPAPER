"""Independent Grade known-zero revision. Legacy v1 modules stay unchanged.

Only Grade value semantics and the separately identified actor objective change.
Layer structure, signal definitions, actor inputs and dual update are inherited.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .constrained_actor_objective_v2 import (ACTOR_LOSS_CONTRACT_ID, ACTOR_LOSS_FORMULA,
                                           episode_sum_actor_loss)
from .constrained_adapter import require
from .constrained_batch import BATCH_EPISODES
from .constrained_ppo import (COST_NAMES, ConstrainedPPO, ConstrainedPPOConfig, FourCritics,
                              _digest_payload)
from .masked_policy import DispatchMaskedPolicy
from .observation_contract import OBSERVATION_SIZE

IMPLEMENTATION_VERSION = "dispatch-constrained-ppo-grade-zero-v2"
GRADE_VALUE_CONTRACT_ID = "dispatch-grade-value-known-zero-v2"
CHECKPOINT_FORMAT = "dispatch-constrained-ppo-grade-zero-checkpoint-v2"


@dataclass(frozen=True)
class GradeBoundaryConfig(ConstrainedPPOConfig):
    """Same hyperparameters, with explicit independent semantic contract IDs."""
    def to_dict(self):
        return dict(super().to_dict(), implementation_version=IMPLEMENTATION_VERSION,
                    training_contract_id=IMPLEMENTATION_VERSION, grade_value_contract_id=GRADE_VALUE_CONTRACT_ID,
                    actor_loss_contract_id=ACTOR_LOSS_CONTRACT_ID, actor_loss_formula=ACTOR_LOSS_FORMULA)


def grade_effective_value(raw, failed):
    require(raw.shape == failed.shape, "Grade value/history shape mismatch")
    require(bool(torch.isfinite(raw).all()), "nonfinite Grade raw value")
    require(bool(((failed == 0) | (failed == 1)).all()), "Grade history must be binary")
    return (1-failed.to(raw.dtype))*raw


class GradeBoundaryCritics(FourCritics):
    """The same four MLPs/parameters/state_dict; gate is outside the Grade MLP."""
    def forward_raw_costs(self, observations, history_before):
        return super().forward_costs(observations, history_before)

    def forward_costs(self, observations, history_before):
        raw = self.forward_raw_costs(observations, history_before)
        return dict(raw, grade=grade_effective_value(raw["grade"], history_before))


def validate_grade_boundary_batch(batch):
    """Fail on inconsistent histories, rather than zeroing a repeated cost."""
    costs = batch.costs[:, 0]
    require(bool(((costs == 0) | (costs == 1)).all()), "Grade costs must be 0/1")
    require(bool((costs[batch.history_before] == 0).all()), "Grade cost repeated after known failure")
    require(torch.equal(batch.history_after, batch.history_before | (costs == 1)), "Grade history/cost mismatch")
    for start, stop in batch.episode_bounds:
        require(not bool(batch.history_before[start]), "Grade history did not reset between episodes")
        require(torch.equal(batch.history_before[start+1:stop], batch.history_after[start:stop-1]),
                "Grade history is not contiguous")
        require(bool(batch.terminals[stop-1]) and not bool(batch.terminals[start:stop-1].any()),
                "invalid episode terminal boundary")


class ConstrainedPPOGradeBoundary(ConstrainedPPO):
    def __init__(self, policy, config=None):
        require(type(policy) is DispatchMaskedPolicy and policy.observation_space.shape == (OBSERVATION_SIZE,),
                "grade-zero-v2 requires the existing raw1519 masked actor")
        self.config = config or GradeBoundaryConfig()
        require(isinstance(self.config, GradeBoundaryConfig), "explicit GradeBoundaryConfig is required")
        self.policy = policy
        self.critics = GradeBoundaryCritics(self.config.hidden_sizes, self.config.seed)
        self.lambdas = np.asarray(self.config.cost_weights, dtype=np.float64).copy()
        actor_parameters = [p for name, p in policy.named_parameters() if "value_net" not in name]
        self.actor_optimizer = torch.optim.Adam(actor_parameters, lr=self.config.actor_learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critics.parameters(), lr=self.config.critic_learning_rate)
        self.generator = torch.Generator(device="cpu").manual_seed(self.config.seed)
        self.update_index = 0
        policy.set_training_mode(True)

    def compute_advantages(self, batch):
        validate_grade_boundary_batch(batch)
        # _values uses forward_costs(current,before) and forward_costs(next,after).
        # Both GAE deltas and lambda-return targets therefore use effective values.
        result = super().compute_advantages(batch)
        after = batch.history_before.numpy()
        require(np.all(result["cost_advantages"]["grade"][after] == 0), "nonzero Grade advantage after failure")
        require(np.all(result["cost_returns"]["grade"][after] == 0), "nonzero Grade target after failure")
        return result

    def critic_loss(self, batch, index, returns):
        # The same effective forward used in GAE is used in regression, with its
        # gradient gate; do not regress raw Grade values to effective targets.
        predicted = dict(reward=self.critics.forward_reward(batch.observations[index]),
                         **self.critics.forward_costs(batch.observations[index], batch.history_before[index]))
        weights = batch.weights[index]
        denominator = weights.sum()
        losses = {k: ((predicted[k]-returns[k][index])**2*weights).sum()/denominator for k in predicted}
        return self.config.vf_coef*sum(losses.values()), losses

    def update(self, batch):
        """Versioned future training path; never invoked by offline acceptance."""
        require(batch.episode_count == BATCH_EPISODES, "requires eight complete episodes")
        advantages = self.compute_advantages(batch)
        combined = torch.as_tensor(advantages["combined_advantages"], dtype=torch.float32)
        returns = dict(reward=torch.as_tensor(advantages["reward_returns"], dtype=torch.float32),
                       **{k: torch.as_tensor(v, dtype=torch.float32) for k,v in advantages["cost_returns"].items()})
        before, critic_before = self.actor_digest(), self.critic_digests()
        metrics = []
        for _ in range(self.config.n_epochs):
            permutation = torch.randperm(batch.size, generator=self.generator)
            for offset in range(0, batch.size, self.config.minibatch_size):
                index = permutation[offset:offset+self.config.minibatch_size]
                _, logp, entropy = self.policy.evaluate_actions(batch.observations[index], batch.actions[index])
                actor = episode_sum_actor_loss(logp, batch.old_log_probs[index], combined[index], batch.weights[index],
                    entropy, clip_range=self.config.clip_range, ent_coef=self.config.ent_coef)
                self.actor_optimizer.zero_grad()
                actor["loss"].backward()
                actor_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in self.policy.parameters() if p.grad is not None], self.config.max_grad_norm)
                self.actor_optimizer.step()
                critic_loss, value_losses = self.critic_loss(batch, index, returns)
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_norm = torch.nn.utils.clip_grad_norm_(list(self.critics.parameters()), self.config.max_grad_norm)
                self.critic_optimizer.step()
                metrics.append(dict(policy_loss=float(actor["loss"].detach()), critic_loss=float(critic_loss.detach()),
                    value_losses={k: float(v.detach()) for k,v in value_losses.items()},
                    approx_kl=float(actor["approx_kl"].detach()), clip_fraction=float(actor["clip_fraction"].detach()),
                    entropy=float(entropy.mean().detach()), actor_grad_norm=float(actor_norm), critic_grad_norm=float(critic_norm),
                    ratio_max_error=float((actor["ratio"]-1).abs().max().detach())))
        self._assert_finite()
        critic_after = self.critic_digests()
        self.update_index += 1
        return dict(training_contract_id=IMPLEMENTATION_VERSION, implementation_version=IMPLEMENTATION_VERSION,
                    actor_loss_contract_id=ACTOR_LOSS_CONTRACT_ID, grade_value_contract_id=GRADE_VALUE_CONTRACT_ID,
                    transitions=batch.size, episodes=batch.episode_count, update_index=self.update_index,
                    mode=self.config.mode, epochs=self.config.n_epochs, minibatches=len(metrics), policy_version=batch.policy_version,
                    actor_digest_before=before, actor_digest_after=self.actor_digest(), actor_changed=before!=self.actor_digest(),
                    critic_digests_before=critic_before, critic_digests_after=critic_after,
                    critics_changed={k: critic_before[k]!=critic_after[k] for k in critic_before},
                    coefficients=self.coefficients.tolist(), lambdas=self.lambdas.tolist(), normalization=advantages["normalization"],
                    mean={k: float(np.mean([m[k] for m in metrics])) for k in
                          ("policy_loss","critic_loss","approx_kl","clip_fraction","entropy","actor_grad_norm","critic_grad_norm")},
                    value_losses={k: float(np.mean([m["value_losses"][k] for m in metrics])) for k in ("reward",)+COST_NAMES},
                    sample_weight=1/BATCH_EPISODES, objective=self.config.to_dict()["objective"])

    def _payload(self, extra=None):
        payload = super()._payload()
        payload.update(format=CHECKPOINT_FORMAT, training_contract_id=IMPLEMENTATION_VERSION,
                       implementation_version=IMPLEMENTATION_VERSION, actor_loss_contract_id=ACTOR_LOSS_CONTRACT_ID,
                       grade_value_contract_id=GRADE_VALUE_CONTRACT_ID)
        if extra:
            require(not set(extra)&set(payload), "checkpoint metadata cannot overwrite reserved fields")
            payload.update(extra)
        return payload

    @classmethod
    def load_checkpoint(cls, path, policy):
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        require(payload.get("format") == CHECKPOINT_FORMAT, "legacy checkpoint requires explicit version migration; no implicit resume")
        require(payload.get("implementation_version") == IMPLEMENTATION_VERSION and
                payload.get("actor_loss_contract_id") == ACTOR_LOSS_CONTRACT_ID and
                payload.get("grade_value_contract_id") == GRADE_VALUE_CONTRACT_ID, "revision contract differs")
        require(payload["integrity_sha256"] == _digest_payload({k:v for k,v in payload.items() if k!="integrity_sha256"}),
                "checkpoint integrity check failed")
        config = GradeBoundaryConfig(**{k:tuple(v) if k in ("cost_weights","cost_limits","hidden_sizes") else v
            for k,v in payload["config"].items() if k in GradeBoundaryConfig.__dataclass_fields__})
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
        require(trainer.actor_digest() == payload["actor_digest"] and trainer.critic_digests() == payload["critic_digests"],
                "checkpoint model digest differs")
        return trainer
