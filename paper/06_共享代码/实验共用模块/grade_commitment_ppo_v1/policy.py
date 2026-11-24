"""One causal decision-prefix path for sampling, admission and evaluation."""
from __future__ import annotations

import numpy as np
import torch

from .models import pack_summary_prefixes


def replay_policy(actor, states, *, actions=None, chunk_size=16):
    """Recompute current policy on each original prefix, without hidden caches.

    This helper does not disable gradients; admission/evaluation own their
    no-grad guards. No future state may enter the prefix of an earlier query.
    """
    states = np.asarray(states)
    if states.ndim != 2 or states.shape[1] != 1519 or not len(states):
        raise ValueError('decision-policy replay requires states[N,1519]')
    if states.dtype != np.float32 or not np.isfinite(states).all():
        raise ValueError('decision-policy replay requires finite float32 states')
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError('positive replay chunk size required')
    if actions is not None:
        actions = np.asarray(actions)
        if actions.shape != (len(states),) or actions.dtype.kind not in 'iu':
            raise ValueError('one integer action per original decision required')
    logs, entropies, probabilities = [], [], []
    for start in range(0, len(states), chunk_size):
        indices = list(range(start, min(len(states), start + chunk_size)))
        prefixes = pack_summary_prefixes(states, indices=indices,
                                         feature_config=actor.feature_config)
        distribution = actor.distribution(torch.tensor(states[indices].copy()), prefixes)
        probabilities.append(distribution.probs)
        entropies.append(distribution.entropy())
        if actions is not None:
            logs.append(distribution.log_prob(torch.tensor(actions[indices], dtype=torch.int64)))
    return dict(entropies=torch.cat(entropies), probabilities=torch.cat(probabilities),
                log_probs=None if actions is None else torch.cat(logs))
