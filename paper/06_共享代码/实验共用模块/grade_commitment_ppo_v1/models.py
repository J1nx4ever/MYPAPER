"""Causal decision-summary encoders; Actor and one Critic never share weights."""
from dataclasses import dataclass
import copy
import hashlib
import json

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn.utils.rnn import pack_padded_sequence

from python.constrained_ppo import digest
from .features import (FEATURE_NAMES, FEATURE_SIZE, FEATURE_VERSION, FEATURE_CONFIG,
                       build_features, build_sequence, feature_contract)

CANDIDATES = ('a0_summary', 'a1_actor_gru')


def require(condition,message):
    if not condition:raise ValueError(message)


def canonical(value):
    return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()


def normalize_feature_config(config=None):
    config=copy.deepcopy(FEATURE_CONFIG if config is None else config)
    require(isinstance(config,dict) and config==FEATURE_CONFIG,'explicit frozen summary feature configuration differs')
    canonical(config)
    return config


def feature_config_sha256(config=None):
    return feature_contract(normalize_feature_config(config))['config_sha256']


def _stamp(value):return torch.tensor(list(hashlib.sha256(canonical(value)).digest()),dtype=torch.uint8)


def _immutable(array):
    array=np.asarray(array)
    return np.frombuffer(array.tobytes(),dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True)
class SummaryPrefixes:
    features: np.ndarray
    lengths: np.ndarray
    valid: np.ndarray
    query_indices: tuple
    feature_version: str
    feature_config_sha256: str


def pack_summary_prefixes(states,indices=None,*,feature_config=None,max_tokens=4096,
                          max_batch_bytes=256*1024*1024):
    """Reconstruct x0..xt, INCLUDING current xt, and never inspect later states.

    Prefixes reset only at a true shift start. They are numerical observations,
    not cached neural hidden states, so the same immutable inputs can be replayed
    under every current parameter version during PPO.
    """
    config=normalize_feature_config(feature_config);states=np.asarray(states)
    require(states.ndim==2 and states.shape[1]==1519 and len(states)>0,'nonempty original states[N,1519] required')
    ids=tuple(range(len(states))) if indices is None else tuple(indices)
    require(bool(ids) and len(ids)==len(set(ids)) and
            all(type(i) in (int,np.int64,np.int32) and 0<=i<len(states) for i in ids),
            'unique in-range decision indices required')
    require(type(max_tokens) is int and 0<max_tokens<=4096 and
            type(max_batch_bytes) is int and 0<max_batch_bytes<=256*1024*1024,'explicit bounded prefix limits required')
    largest=max(ids)+1
    require(largest<=max_tokens,'summary prefix exceeds resource limit; never truncate history silently')
    require(len(ids)*largest*FEATURE_SIZE*4<=max_batch_bytes,'padded summary prefixes exceed byte limit')
    sequence=np.asarray(build_sequence(states[:largest],config=config),dtype=np.float32)
    require(sequence.shape==(largest,FEATURE_SIZE) and np.isfinite(sequence).all(),'summary sequence schema differs')
    features=np.zeros((len(ids),largest,FEATURE_SIZE),dtype=np.float32)
    lengths=np.asarray([i+1 for i in ids],dtype=np.int64)
    valid=np.arange(largest)[None,:]<lengths[:,None]
    for row,i in enumerate(ids):features[row,:i+1]=sequence[:i+1]
    return SummaryPrefixes(_immutable(features),_immutable(lengths),_immutable(valid),tuple(int(i) for i in ids),
                           FEATURE_VERSION,feature_config_sha256(config))


def _states(states):
    require(torch.is_tensor(states) and states.ndim==2 and states.shape[1]==1519 and len(states)>0 and
            states.dtype==torch.float32 and states.device.type=='cpu' and bool(torch.isfinite(states).all()),
            'finite float32 CPU states[N,1519] required')


def _current_summary(states,config):
    _states(states)
    values=states.detach().cpu().numpy()
    result=np.asarray(build_sequence(values,config=config),dtype=np.float32)
    require(result.shape==(len(states),FEATURE_SIZE) and np.isfinite(result).all(),'current summary schema differs')
    return torch.tensor(result,dtype=torch.float32)


def _check_prefix(prefix,states,config,current):
    require(isinstance(prefix,SummaryPrefixes),'causal decision-summary prefix required; zero history is not implicit')
    features,lengths,valid=prefix.features,prefix.lengths,prefix.valid
    require(prefix.feature_version==FEATURE_VERSION and prefix.feature_config_sha256==feature_config_sha256(config),
            'summary prefix feature contract differs')
    require(isinstance(features,np.ndarray) and features.dtype==np.float32 and features.ndim==3 and
            features.shape[0]==len(states) and features.shape[2]==FEATURE_SIZE and np.isfinite(features).all() and
            isinstance(lengths,np.ndarray) and lengths.dtype==np.int64 and lengths.shape==(len(states),) and
            np.all((lengths>=1)&(lengths<=features.shape[1])) and np.all(lengths<=4096),
            'nonempty full causal prefix required for every decision')
    require(isinstance(valid,np.ndarray) and valid.dtype==np.bool_ and valid.shape==features.shape[:2] and
            np.array_equal(valid,np.arange(features.shape[1])[None,:]<lengths[:,None]) and
            len(prefix.query_indices)==len(states) and tuple(lengths.tolist())==tuple(i+1 for i in prefix.query_indices),
            'prefix padding/decision index certificate differs')
    require(np.array_equal(features[np.arange(len(states)),lengths-1],current.detach().numpy()),
            'prefix current summary differs from sampled state')


def _encoder(input_size,widths,activation='relu'):
    layers=[];last=input_size
    for width in widths:
        layers.extend((nn.Linear(last,width),nn.ReLU() if activation=='relu' else nn.Tanh()));last=width
    return nn.Sequential(*layers)


def _history(summary_encoder,gru,prefix):
    # pack_padded_sequence excludes all padded timesteps; h0 is freshly zero for
    # every independently replayed full-shift prefix under the CURRENT weights.
    features=torch.tensor(np.asarray(prefix.features),dtype=torch.float32)
    encoded=summary_encoder(features)
    packed=pack_padded_sequence(encoded,torch.tensor(prefix.lengths.copy()),batch_first=True,enforce_sorted=False)
    _,hidden=gru(packed)
    return hidden[0]


class Actor(nn.Module):
    def __init__(self,spec,*,candidate_id='a0_summary',feature_config=None,gru_hidden=None):
        super().__init__()
        require(candidate_id in CANDIDATES,'unknown commitment Actor candidate')
        self.candidate_id=candidate_id;self.spec=copy.deepcopy(spec)
        self.feature_config=normalize_feature_config(feature_config)
        require(spec['history_projection_init']=='zeros','Actor history logit projection must start at zero')
        hidden=spec['gru_hidden'] if gru_hidden is None else gru_hidden
        require(hidden==spec['gru_hidden'],'explicit GRU architecture differs')
        self.state_encoder=_encoder(1519,spec['hidden_sizes'],spec['activation'])
        self.summary_encoder=_encoder(FEATURE_SIZE,[spec['summary_embedding']])
        self.base_head=nn.Sequential(nn.Linear(spec['hidden_sizes'][-1]+spec['summary_embedding'],spec['fusion_hidden']),
                                     nn.ReLU(),nn.Linear(spec['fusion_hidden'],4))
        # All common modules are initialized before optional history modules,
        # so equal independent seeds give exactly equal common branch weights.
        self.history_gru=None;self.history_projection=None
        if candidate_id=='a1_actor_gru':
            self.history_gru=nn.GRU(spec['summary_embedding'],hidden,batch_first=True)
            self.history_projection=nn.Linear(hidden,4,bias=False)
            nn.init.zeros_(self.history_projection.weight)
        self.register_buffer('_feature_contract',_stamp(feature_contract(self.feature_config)))
        self.register_buffer('_architecture_contract',_stamp(dict(candidate_id=candidate_id,spec=spec)))

    def forward(self,states,prefixes=None):
        _states(states)
        mask=states[:,19:23]
        require(bool(((mask==0)|(mask==1)).all()) and bool(mask.bool().any(dim=1).all()) and
                bool((states[:,17]==0).all()),'invalid action mask or terminal action request')
        current=_current_summary(states,self.feature_config)
        logits=self.base_head(torch.cat((self.state_encoder(states),self.summary_encoder(current)),dim=-1))
        if prefixes is not None:_check_prefix(prefixes,states,self.feature_config,current)
        if self.history_gru is not None:
            _check_prefix(prefixes,states,self.feature_config,current)
            logits=logits+self.history_projection(_history(self.summary_encoder,self.history_gru,prefixes))
        require(bool(torch.isfinite(logits).all()),'nonfinite commitment Actor logits')
        return logits.masked_fill(~mask.bool(),-torch.inf)

    def distribution(self,states,prefixes=None):return Categorical(logits=self(states,prefixes))


class CommitmentCostCritic(nn.Module):
    def __init__(self,spec,*,feature_config=None):
        super().__init__();self.spec=copy.deepcopy(spec);self.feature_config=normalize_feature_config(feature_config)
        self.state_encoder=_encoder(1519,spec['state_hidden_sizes'])
        self.summary_encoder=_encoder(FEATURE_SIZE,[spec['summary_embedding']])
        self.history_gru=nn.GRU(spec['summary_embedding'],spec['gru_hidden'],batch_first=True)
        self.fusion=nn.Sequential(nn.Linear(spec['state_hidden_sizes'][-1]+spec['summary_embedding']+spec['gru_hidden'],
                                           spec['fusion_hidden']),nn.ReLU(),nn.Linear(spec['fusion_hidden'],1))
        self.register_buffer('_feature_contract',_stamp(feature_contract(self.feature_config)))
        self.register_buffer('_architecture_contract',_stamp(dict(spec=spec)))

    def forward(self,states,prefixes=None):
        current=_current_summary(states,self.feature_config)
        _check_prefix(prefixes,states,self.feature_config,current)
        return self.fusion(torch.cat((self.state_encoder(states),self.summary_encoder(current),
                                     _history(self.summary_encoder,self.history_gru,prefixes)),dim=-1)).squeeze(-1)


def actor_from_config(config):
    return Actor(config['network'],candidate_id=config['candidate_id'],feature_config=config['feature_config'])


def common_actor_digest(actor):
    class View:
        def state_dict(self):
            return {name:value for name,value in actor.state_dict().items()
                    if name.startswith(('state_encoder.','summary_encoder.','base_head.')) or name=='_feature_contract'}
    return digest(View())


# The independent learner keeps a single scalar Critic despite two GRUs overall.
MaterialCostCritic=CommitmentCostCritic
