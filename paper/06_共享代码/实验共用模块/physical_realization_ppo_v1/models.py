"""Equivalent full-history GRUs with one same-episode sequence per forward.

Shared history computation keeps the exact recurrence and sums all requested
loss gradients through it. It never detaches a prefix or caches a neural state.
"""
import copy
import json
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from grade_commitment_ppo_v1.models import (Actor as ReferenceActor,
    CommitmentCostCritic as ReferenceCritic, _states, _current_summary, _stamp,
    normalize_feature_config, feature_config_sha256, common_actor_digest, digest,
    CANDIDATES)
from .features_cache import (EXECUTION_VERSION, SummarySequence, SummaryFeatureCache,
    pack_summary_prefixes, validate_sequence, require)
from .features import physical_contract, PHYSICAL_SIZE


def _history(summary_encoder, gru, prefixes):
    values=torch.tensor(prefixes.features.copy(),dtype=torch.float32).unsqueeze(0)
    # One forward from h0=0 through the latest queried position. GRU is causal:
    # output[t] uses only x[0:t+1], even though later outputs are also computed.
    outputs,_=gru(summary_encoder(values))
    return outputs[0,list(prefixes.query_indices)]


PHYSICAL_BRANCH = dict(dimension=46, hidden=64, activation='relu', head_initialization='zeros',
                       history='current_only', actor_critic_sharing=False)


def physical_branch(spec=None):
    value=copy.deepcopy(PHYSICAL_BRANCH if spec is None else spec)
    require(isinstance(value,dict) and json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)==
            json.dumps(PHYSICAL_BRANCH,sort_keys=True,separators=(',',':'),allow_nan=False),
            'frozen independent 46D physical residual architecture required')
    return value


def _add_physical(module, spec, outputs):
    module.physical_branch=physical_branch(spec)
    module.physical_encoder=nn.Sequential(nn.Linear(PHYSICAL_SIZE,64),nn.ReLU())
    module.physical_head=nn.Linear(64,outputs)
    # Only this final layer is zero: nonzero encoder activations let its head
    # receive gradients on the first minibatch while initial functions agree.
    nn.init.zeros_(module.physical_head.weight); nn.init.zeros_(module.physical_head.bias)
    module.register_buffer('_physical_contract',_stamp(physical_contract()))
    module.register_buffer('_physical_architecture_contract',_stamp(module.physical_branch))


def _current_physical(prefix):
    return torch.tensor(prefix.physical_features[list(prefix.query_indices)].copy(),dtype=torch.float32)


class Actor(ReferenceActor):
    def __init__(self,*args,physical_spec=None,**kwargs):
        super().__init__(*args,**kwargs)
        _add_physical(self,physical_spec,4)
        self.register_buffer('_execution_contract',_stamp(EXECUTION_VERSION))

    def forward(self,states,prefixes=None):
        _states(states); mask=states[:,19:23]
        require(bool(((mask==0)|(mask==1)).all()) and bool(mask.bool().any(dim=1).all()) and
                bool((states[:,17]==0).all()),'invalid action mask or terminal action request')
        current=validate_sequence(prefixes,states,self.feature_config)
        logits=self.base_head(torch.cat((self.state_encoder(states),self.summary_encoder(current)),dim=-1))
        if self.history_gru is not None:
            require(prefixes is not None,'certified full causal summary sequence required')
            logits=logits+self.history_projection(_history(self.summary_encoder,self.history_gru,prefixes))
        logits=logits+self.physical_head(self.physical_encoder(_current_physical(prefixes)))
        require(bool(torch.isfinite(logits).all()),'nonfinite physical Actor logits')
        return logits.masked_fill(~mask.bool(),-torch.inf)

    def distribution(self,states,prefixes=None):
        return Categorical(logits=self(states,prefixes))


class CommitmentCostCritic(ReferenceCritic):
    def __init__(self,*args,physical_spec=None,**kwargs):
        super().__init__(*args,**kwargs)
        _add_physical(self,physical_spec,1)
        self.register_buffer('_execution_contract',_stamp(EXECUTION_VERSION))

    def forward(self,states,prefixes=None):
        _states(states)
        current=validate_sequence(prefixes,states,self.feature_config)
        base=self.fusion(torch.cat((self.state_encoder(states),self.summary_encoder(current),
               _history(self.summary_encoder,self.history_gru,prefixes)),dim=-1)).squeeze(-1)
        return base+self.physical_head(self.physical_encoder(_current_physical(prefixes))).squeeze(-1)


def actor_from_config(config):
    return Actor(config['network'],candidate_id=config['candidate_id'],feature_config=config['feature_config'],
                 physical_spec=config['physical_branch'])


MaterialCostCritic=CommitmentCostCritic
# Compatibility name for consumers; representation is compact, not padded.
SummaryPrefixes=SummarySequence
