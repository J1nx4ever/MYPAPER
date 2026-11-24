"""Certified immutable numerical tapes and current-step full-prefix batches.

Only observation-derived arrays and tensors are cached. Learned encodings,
GRU hidden states and autograd graphs are recomputed after every Adam step.
Opaque weak registrations reject replaced/forged views; immutable bytes protect
NumPy inputs and tensor version counters reject accidental in-place mutation.
"""
from dataclasses import dataclass
from collections import OrderedDict
import hashlib
import weakref
import numpy as np
import torch
from .features import (FEATURE_SIZE, FEATURE_VERSION, feature_contract, physical_contract,
                       PHYSICAL_SIZE, build_numerical_inputs)
from grade_commitment_ppo_v1.models import normalize_feature_config

EXECUTION_VERSION = 'physical-wait-batched-full-prefix-numerical-cache-v1'
_REGISTERED = {}


def require(condition, message):
    if not condition:raise ValueError(message)


def immutable(value):
    value=np.asarray(value)
    return np.frombuffer(value.tobytes(),dtype=value.dtype).reshape(value.shape)


def _initial_hash(config_sha256):
    return hashlib.sha256((EXECUTION_VERSION+'|'+config_sha256+'|'+
                           physical_contract()['contract_sha256']).encode()).digest()


def _chain(previous, state, summary, physical):
    h=hashlib.sha256(previous)
    for value in (state,summary,physical):h.update(value.tobytes())
    return h.digest()


def _finish(chain, indices):
    return hashlib.sha256(chain+repr(tuple(indices)).encode()).hexdigest()


def _certificate(states, features, physical, indices, config_sha256):
    """Slow independent byte reconstruction, used in adversarial acceptance."""
    h=_initial_hash(config_sha256)
    for raw,x,p in zip(states,features,physical):h=_chain(h,raw,x,p)
    return _finish(h,indices)


def _register(value, payload):
    key=id(value)
    def remove(ref):
        if key in _REGISTERED and _REGISTERED[key][0] is ref:del _REGISTERED[key]
    _REGISTERED[key]=(weakref.ref(value,remove),payload)
    return value


def _registered(value):
    entry=_REGISTERED.get(id(value))
    require(entry is not None and entry[0]() is value,'unregistered or forged numerical certificate')
    return entry[1]


@dataclass(frozen=True)
class SummarySequence:
    features:np.ndarray
    source_states:np.ndarray
    physical_features:np.ndarray
    physical_contract_sha256:str
    query_indices:tuple
    feature_version:str
    feature_config_sha256:str
    certificate_sha256:str

    @property
    def lengths(self):return immutable(np.asarray(self.query_indices,dtype=np.int64)+1)


@dataclass(frozen=True)
class BatchedSummarySequence:
    sequences:tuple


class _Tensors:
    def __init__(self,features,physical):
        self.features=torch.tensor(features.copy(),dtype=torch.float32)
        self.physical=torch.tensor(physical.copy(),dtype=torch.float32)
        self.versions=(self.features._version,self.physical._version)
        self.sha256=hashlib.sha256(features.tobytes()+physical.tobytes()).digest()

    def check(self,verify_content=False):
        require((self.features._version,self.physical._version)==self.versions and
                not self.features.requires_grad and not self.physical.requires_grad,
                'cached numerical tensor was mutated or attached to autograd')
        if verify_content:
            require(hashlib.sha256(self.features.numpy().tobytes()+self.physical.numpy().tobytes()).digest()==self.sha256,
                    'cached numerical tensor content was mutated')


class SummaryFeatureCache:
    def __init__(self,states=None,*,feature_config=None,max_tokens=4096,max_batch_bytes=256*1024*1024):
        self.feature_config=normalize_feature_config(feature_config)
        self.config_sha256=feature_contract(self.feature_config)['config_sha256']
        require(type(max_tokens) is int and 0<max_tokens<=4096 and type(max_batch_bytes) is int and
                0<max_batch_bytes<=256*1024*1024,'explicit bounded full-sequence limits required')
        self.max_tokens,self.max_batch_bytes=max_tokens,max_batch_bytes
        self._states,self._features,self._physical=[],[],[]
        self._chains=[];self._materialized=None;self._tensors=None;self.feature_builds=0
        if states is not None:
            states=np.asarray(states)
            require(states.ndim==2 and states.shape[1]==1519 and len(states)>0,'nonempty original states[N,1519] required')
            for state in states:self.append(state)

    def append(self,state):
        state=np.asarray(state)
        require(state.shape==(1519,) and state.dtype==np.float32 and np.isfinite(state).all() and
                state[17]==0,'finite float32 nonterminal causal state[1519] required')
        require(len(self)<self.max_tokens and self.bytes+self._row_bytes<=self.max_batch_bytes,
                'numerical sequence exceeds resource limit; never truncate history')
        summary,physical=build_numerical_inputs(state)
        require(summary.shape==(128,) and physical.shape==(46,) and np.isfinite(summary).all() and
                np.isfinite(physical).all(),'causal numerical feature schema differs')
        self._states.append(immutable(state));self._features.append(immutable(summary));self._physical.append(immutable(physical))
        self._chains.append(_chain(self._chains[-1] if self._chains else _initial_hash(self.config_sha256),
                                   self._states[-1],self._features[-1],self._physical[-1]))
        self._materialized=None;self._tensors=None;self.feature_builds+=1

    def __len__(self):return len(self._states)

    @property
    def _row_bytes(self):return (1519+2*(FEATURE_SIZE+PHYSICAL_SIZE))*4

    @property
    def bytes(self):return len(self)*self._row_bytes

    def prefix(self,indices=None,*,max_tokens=None,max_batch_bytes=None):
        ids=tuple(range(len(self))) if indices is None else tuple(indices)
        require(bool(ids) and len(ids)==len(set(ids)) and
                all(type(i) in (int,np.int64,np.int32) and 0<=i<len(self) for i in ids),
                'unique in-range decision indices required')
        ids=tuple(map(int,ids));largest=max(ids)+1
        tokens=self.max_tokens if max_tokens is None else max_tokens
        byte_limit=self.max_batch_bytes if max_batch_bytes is None else max_batch_bytes
        require(type(tokens) is int and 0<tokens<=self.max_tokens and type(byte_limit) is int and
                0<byte_limit<=self.max_batch_bytes and largest<=tokens and largest*self._row_bytes<=byte_limit,
                'numerical prefix exceeds resource limit')
        if self._materialized is None:
            self._materialized=tuple(immutable(np.stack(x)) for x in (self._states,self._features,self._physical))
            self._states,self._features,self._physical=[list(x) for x in self._materialized]
            self._tensors=_Tensors(self._materialized[1],self._materialized[2])
        raw,x,p=(v[:largest] for v in self._materialized)
        sequence=SummarySequence(x,raw,p,physical_contract()['contract_sha256'],ids,FEATURE_VERSION,
                                 self.config_sha256,_finish(self._chains[largest-1],ids))
        fields=(id(x),id(raw),id(p),ids,sequence.physical_contract_sha256,sequence.feature_version,
                sequence.feature_config_sha256,sequence.certificate_sha256)
        return _register(sequence,(fields,self._tensors,largest))


def _checked(prefix,config=None,verify_tensor=False):
    require(isinstance(prefix,SummarySequence),'certified full causal summary sequence required')
    fields,tensors,largest=_registered(prefix)
    require(fields==(id(prefix.features),id(prefix.source_states),id(prefix.physical_features),prefix.query_indices,
                    prefix.physical_contract_sha256,prefix.feature_version,prefix.feature_config_sha256,prefix.certificate_sha256),
            'numerical certificate fields changed')
    require(prefix.feature_version==FEATURE_VERSION and prefix.physical_contract_sha256==physical_contract()['contract_sha256']
            and (config is None or prefix.feature_config_sha256==feature_contract(config)['config_sha256']),
            'numerical feature contract differs')
    require(all(not v.flags.writeable for v in (prefix.features,prefix.source_states,prefix.physical_features)),
            'immutable byte-backed numerical tape required')
    tensors.check(verify_content=verify_tensor)
    return tensors,largest


def _parts(prefix):
    if isinstance(prefix,BatchedSummarySequence):
        require(_registered(prefix)==tuple(map(id,prefix.sequences)),'batched certificate changed')
        return prefix.sequences
    return (prefix,)


def validate_sequence(prefix,states,config):
    parts=_parts(prefix);raw=[];values=[]
    for part in parts:
        tensors,_=_checked(part,config,verify_tensor=True);ids=list(part.query_indices)
        raw.append(part.source_states[ids]);values.append(tensors.features[ids])
    require(np.array_equal(np.concatenate(raw),states.detach().numpy()),'numerical current raw observation differs')
    return torch.cat(values,dim=0)


def torch_physical(prefix):
    return torch.cat([_checked(p)[0].physical[list(p.query_indices)] for p in _parts(prefix)],dim=0)


def history_inputs(prefix):
    return tuple(_checked(p)[0].features[:len(p.features)] for p in _parts(prefix))


def pack_batch_prefixes(sequences):
    sequences=tuple(sequences);require(bool(sequences),'nonempty independent episode prefix batch required')
    for p in sequences:_checked(p)
    result=BatchedSummarySequence(sequences)
    return _register(result,tuple(map(id,sequences)))


def pack_summary_prefixes(states,indices=None,*,feature_config=None,max_tokens=4096,max_batch_bytes=256*1024*1024):
    states=np.asarray(states)
    require(states.ndim==2 and states.shape[1]==1519 and len(states)>0,'nonempty original states[N,1519] required')
    ids=tuple(range(len(states))) if indices is None else tuple(indices)
    require(bool(ids) and len(ids)==len(set(ids)) and all(type(i) in (int,np.int64,np.int32) and 0<=i<len(states) for i in ids),
            'unique in-range decision indices required')
    return SummaryFeatureCache(states[:max(ids)+1],feature_config=feature_config,max_tokens=max_tokens,
                               max_batch_bytes=max_batch_bytes).prefix(ids)


def _immutable_base(value):
    base=value
    while isinstance(base,np.ndarray) and base.base is not None:base=base.base
    return base if isinstance(base,bytes) and value.flags.c_contiguous else None


class BatchFeatureCache:
    def __init__(self,*,feature_config,max_tokens=4096,max_bytes=256*1024*1024):
        self.feature_config=normalize_feature_config(feature_config);self.max_tokens,self.max_bytes=max_tokens,max_bytes
        self.entries=OrderedDict();self.feature_builds=0;self.hits=0;self._identities={};self.byte_hashes=0

    def clear(self):self.entries.clear();self._identities.clear()

    def get(self,states):
        states=np.asarray(states)
        require(states.ndim==2 and states.shape[1]==1519 and states.dtype==np.float32,'float32 original states required')
        base=_immutable_base(states)
        identity=None if base is None else (id(base),states.__array_interface__['data'][0],states.shape)
        if identity in self._identities:
            pinned,key=self._identities[identity]
            require(pinned is base,'immutable input identity changed')
            if key in self.entries:self.hits+=1;self.entries.move_to_end(key);return self.entries[key]
        key=(states.shape,hashlib.sha256(states.tobytes()).digest(),feature_contract(self.feature_config)['config_sha256'],
             physical_contract()['contract_sha256']);self.byte_hashes+=1
        if key in self.entries:self.hits+=1;self.entries.move_to_end(key);cache=self.entries[key]
        else:
            cache=SummaryFeatureCache(states,feature_config=self.feature_config,max_tokens=self.max_tokens,max_batch_bytes=self.max_bytes)
            while self.entries and sum(v.bytes for v in self.entries.values())+cache.bytes>self.max_bytes:
                old,_=self.entries.popitem(last=False)
                self._identities={i:v for i,v in self._identities.items() if v[1]!=old}
            require(cache.bytes<=self.max_bytes,'numerical cache exceeds byte limit')
            self.entries[key]=cache;self.feature_builds+=cache.feature_builds
        if identity is not None:self._identities[identity]=(base,key)
        return cache
