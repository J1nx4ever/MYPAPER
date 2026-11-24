"""Equivalent full-history GRUs with one same-episode sequence per forward.

Shared history computation keeps the exact recurrence and sums all requested
loss gradients through it. It never detaches a prefix or caches a neural state.
"""
import numpy as np
import torch
from torch.distributions import Categorical

from grade_commitment_ppo_v1.models import (Actor as ReferenceActor,
    CommitmentCostCritic as ReferenceCritic, _states, _current_summary, _stamp,
    normalize_feature_config, feature_config_sha256, common_actor_digest, digest,
    CANDIDATES)
from .features_cache import (EXECUTION_VERSION, SummarySequence, SummaryFeatureCache,
    pack_summary_prefixes, validate_sequence, require)


def _history(summary_encoder, gru, prefixes):
    values=torch.tensor(prefixes.features.copy(),dtype=torch.float32).unsqueeze(0)
    # One forward from h0=0 through the latest queried position. GRU is causal:
    # output[t] uses only x[0:t+1], even though later outputs are also computed.
    outputs,_=gru(summary_encoder(values))
    return outputs[0,list(prefixes.query_indices)]


class Actor(ReferenceActor):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.register_buffer('_execution_contract',_stamp(EXECUTION_VERSION))

    def forward(self,states,prefixes=None):
        _states(states); mask=states[:,19:23]
        require(bool(((mask==0)|(mask==1)).all()) and bool(mask.bool().any(dim=1).all()) and
                bool((states[:,17]==0).all()),'invalid action mask or terminal action request')
        current=(_current_summary(states,self.feature_config) if prefixes is None else
                 validate_sequence(prefixes,states,self.feature_config))
        logits=self.base_head(torch.cat((self.state_encoder(states),self.summary_encoder(current)),dim=-1))
        if self.history_gru is not None:
            require(prefixes is not None,'certified full causal summary sequence required')
            logits=logits+self.history_projection(_history(self.summary_encoder,self.history_gru,prefixes))
        require(bool(torch.isfinite(logits).all()),'nonfinite commitment Actor logits')
        return logits.masked_fill(~mask.bool(),-torch.inf)

    def distribution(self,states,prefixes=None):
        return Categorical(logits=self(states,prefixes))


class CommitmentCostCritic(ReferenceCritic):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.register_buffer('_execution_contract',_stamp(EXECUTION_VERSION))

    def forward(self,states,prefixes=None):
        _states(states)
        current=validate_sequence(prefixes,states,self.feature_config)
        return self.fusion(torch.cat((self.state_encoder(states),self.summary_encoder(current),
               _history(self.summary_encoder,self.history_gru,prefixes)),dim=-1)).squeeze(-1)


def actor_from_config(config):
    return Actor(config['network'],candidate_id=config['candidate_id'],feature_config=config['feature_config'])


MaterialCostCritic=CommitmentCostCritic
# Compatibility name for consumers; representation is compact, not padded.
SummaryPrefixes=SummarySequence
