"""One certified causal sequence for sampling, admission and evaluation."""
import numpy as np
import torch

from .features_cache import SummaryFeatureCache,require


def replay_policy(actor, states, *, actions=None, chunk_size=16):
    states=np.asarray(states)
    require(states.ndim==2 and states.shape[1]==1519 and len(states)>0 and
            states.dtype==np.float32 and np.isfinite(states).all(),
            'decision-policy replay requires finite float32 states[N,1519]')
    require(type(chunk_size) is int and chunk_size>0,'positive replay chunk size required')
    if actions is not None:
        actions=np.asarray(actions)
        require(actions.shape==(len(states),) and actions.dtype.kind in 'iu',
                'one integer action per original decision required')
    # Full no-grad sequence evaluation avoids independently replaying every prefix.
    # Under gradients each caller's requested loss still reaches the full history.
    cache=SummaryFeatureCache(states,feature_config=actor.feature_config)
    prefix=cache.prefix()
    distribution=actor.distribution(torch.tensor(states.copy()),prefix)
    return dict(entropies=distribution.entropy(),probabilities=distribution.probs,
                log_probs=None if actions is None else
                distribution.log_prob(torch.tensor(actions,dtype=torch.int64)))
