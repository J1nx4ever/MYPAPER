"""Matched Actor/Critic modules with exact branch-output ablations.

All eight arms instantiate every module in the same order. An inactive branch
returns exact zeros and is never evaluated, so its biases or GRU step count
cannot become a substitute input. Legal-action masks remain external in every
arm. The 46D branch sees only the current certified row, never a history token.
"""
import copy
import json
import torch
from torch import nn
from torch.distributions import Categorical
from grade_commitment_ppo_v1.models import (
    Actor as ReferenceActor, CommitmentCostCritic as ReferenceCritic,
    _states, _stamp, normalize_feature_config, feature_config_sha256, digest)
from .features_cache import (EXECUTION_VERSION, SummarySequence, SummaryFeatureCache,
    pack_summary_prefixes, validate_sequence, require)
from .features import PHYSICAL_SIZE, physical_contract
from .contract import branches, ablation_contract, parameter_digest

PHYSICAL_BRANCH = dict(dimension=46, hidden=64, activation='relu',
    head_initialization='zeros', history='current_only', actor_critic_sharing=False)


def physical_branch(spec=None):
    value = copy.deepcopy(PHYSICAL_BRANCH if spec is None else spec)
    require(isinstance(value,dict) and json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)==
            json.dumps(PHYSICAL_BRANCH,sort_keys=True,separators=(',',':'),allow_nan=False),
            'frozen independent 46D physical residual architecture required')
    return value


def _add_physical(module,spec,outputs):
    module.physical_branch = physical_branch(spec)
    module.physical_encoder = nn.Sequential(nn.Linear(PHYSICAL_SIZE,64),nn.ReLU())
    module.physical_head = nn.Linear(64,outputs)
    nn.init.zeros_(module.physical_head.weight)
    nn.init.zeros_(module.physical_head.bias)
    module.register_buffer('_physical_contract',_stamp(physical_contract()))
    module.register_buffer('_physical_architecture_contract',_stamp(module.physical_branch))


def _history(module,prefix):
    if not module.branch_flags['H']:
        return torch.zeros((len(prefix.query_indices),module.history_gru.hidden_size),dtype=torch.float32)
    values = torch.tensor(prefix.features.copy(),dtype=torch.float32).unsqueeze(0)
    outputs,_ = module.history_gru(module.summary_encoder(values))
    return outputs[0,list(prefix.query_indices)]


def _current(module,states,summary,raw_width):
    raw = module.state_encoder(states) if module.branch_flags['S'] else states.new_zeros((len(states),raw_width))
    current = (module.summary_encoder(summary) if module.branch_flags['C'] else
               states.new_zeros((len(states),module.spec['summary_embedding'])))
    return raw,current


def _physical(module,prefix,outputs):
    if not module.branch_flags['P']:
        return torch.zeros((len(prefix.query_indices),outputs),dtype=torch.float32)
    current = torch.tensor(prefix.physical_features[list(prefix.query_indices)].copy(),dtype=torch.float32)
    return module.physical_head(module.physical_encoder(current))


def _buffers(module,candidate):
    module.branch_flags = branches(candidate)
    module.register_buffer('_ablation_contract',_stamp(ablation_contract(candidate)))
    module.register_buffer('_execution_contract',_stamp(EXECUTION_VERSION))


class Actor(ReferenceActor):
    def __init__(self,spec,*,candidate_id='s01_A0',feature_config=None,gru_hidden=None,physical_spec=None):
        super().__init__(spec,candidate_id='a1_actor_gru',feature_config=feature_config,gru_hidden=gru_hidden)
        self.candidate_id = candidate_id
        _add_physical(self,physical_spec,4)
        _buffers(self,candidate_id)

    def forward(self,states,prefixes=None):
        _states(states)
        legal = states[:,19:23]
        require(bool(((legal==0)|(legal==1)).all()) and bool(legal.bool().any(dim=1).all()) and
                bool((states[:,17]==0).all()),'invalid action mask or terminal request')
        current = validate_sequence(prefixes,states,self.feature_config)
        raw,summary = _current(self,states,current,self.spec['hidden_sizes'][-1])
        logits = self.base_head(torch.cat((raw,summary),dim=-1))
        if self.branch_flags['H']:
            logits = logits+self.history_projection(_history(self,prefixes))
        logits = logits+_physical(self,prefixes,4)
        require(bool(torch.isfinite(logits).all()),'nonfinite state-branch Actor logits')
        return logits.masked_fill(~legal.bool(),-torch.inf)

    def distribution(self,states,prefixes=None):
        return Categorical(logits=self(states,prefixes))


class CommitmentCostCritic(ReferenceCritic):
    def __init__(self,spec,*,feature_config=None,candidate_id='s01_A0',physical_spec=None):
        super().__init__(spec,feature_config=feature_config)
        self.candidate_id = candidate_id
        _add_physical(self,physical_spec,1)
        _buffers(self,candidate_id)

    def forward(self,states,prefixes=None):
        _states(states)
        current = validate_sequence(prefixes,states,self.feature_config)
        raw,summary = _current(self,states,current,self.spec['state_hidden_sizes'][-1])
        base = self.fusion(torch.cat((raw,summary,_history(self,prefixes)),dim=-1)).squeeze(-1)
        result = base+_physical(self,prefixes,1).squeeze(-1)
        require(bool(torch.isfinite(result).all()),'nonfinite state-branch Critic values')
        return result


def actor_from_config(config):
    require(config.get('state_branch_contract')==ablation_contract(config['candidate_id']),
            'Actor state-branch configuration differs')
    return Actor(config['network'],candidate_id=config['candidate_id'],feature_config=config['feature_config'],
                 physical_spec=config['physical_branch'])


def common_actor_digest(actor):
    return parameter_digest(actor)


MaterialCostCritic = CommitmentCostCritic
SummaryPrefixes = SummarySequence
