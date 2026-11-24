"""Literal frozen episode-sum actor objective, independent of Grade semantics.

For a partition S of the batch, L(S)=sum_{i in S}(1/8)*ell_i.
Consequently sum_S L(S)=L(batch), including the short last partition. There is
no minibatch/episode-length denominator or N/m importance rescaling.
"""
from __future__ import annotations

import torch

from .constrained_adapter import require
from .constrained_batch import BATCH_EPISODES

ACTOR_LOSS_CONTRACT_ID = "dispatch-actor-episode-sum-v2"
ACTOR_LOSS_FORMULA = "sum_transition (1/8) * (clipped_negative_surrogate - ent_coef * entropy)"


def episode_sum_actor_loss(log_prob, old_log_prob, advantages, weights, entropy, *, clip_range, ent_coef):
    """Differentiable additive contribution of any nonempty sample partition."""
    require(log_prob.ndim == 1 and log_prob.numel() > 0, "actor loss requires a nonempty vector")
    require(all(x.shape == log_prob.shape for x in (old_log_prob, advantages, weights, entropy)),
            "actor loss vector shapes differ")
    require(all(bool(torch.isfinite(x).all()) for x in (log_prob, old_log_prob, advantages, weights, entropy)),
            "nonfinite actor loss input")
    require(bool(torch.all(weights == 1 / BATCH_EPISODES)), "actor sample weights must be exactly 1/8")
    require(clip_range > 0 and ent_coef >= 0, "invalid actor objective coefficients")
    ratio = (log_prob-old_log_prob).exp()
    require(bool(torch.isfinite(ratio).all()), "nonfinite importance ratio")
    surrogate = torch.maximum(-advantages*ratio, -advantages*ratio.clamp(1-clip_range, 1+clip_range))
    policy_sum = (weights*surrogate).sum()
    entropy_sum = (weights*entropy).sum()
    return dict(loss=policy_sum-ent_coef*entropy_sum, surrogate_sum=policy_sum,
                entropy_sum=entropy_sum, ratio=ratio,
                approx_kl=((ratio-1)-(log_prob-old_log_prob)).mean(),
                clip_fraction=((ratio-1).abs()>clip_range).to(log_prob.dtype).mean())
