"""Certified immutable raw/C-v2 inputs only; no P, graph or neural-state cache."""
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import numpy as np
from .c_features import (FEATURE_SIZE, FEATURE_VERSION, feature_contract,
                         normalize_feature_config, build_features)

EXECUTION_VERSION = 'relation-gat-ego003-certified-current-C-v2-v1'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def immutable(value):
    value = np.asarray(value)
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


def _certificate(states, features, indices, config_sha256):
    h = hashlib.sha256(EXECUTION_VERSION.encode())
    for value in (states, features):
        h.update(str(value.dtype).encode())
        h.update(str(value.shape).encode())
        h.update(value.tobytes())
    h.update(repr(tuple(indices)).encode())
    h.update(config_sha256.encode())
    return h.hexdigest()


@dataclass(frozen=True)
class SummarySequence:
    features: np.ndarray
    source_states: np.ndarray
    query_indices: tuple
    feature_version: str
    feature_config_sha256: str
    certificate_sha256: str

    @property
    def lengths(self):
        return immutable(np.asarray(self.query_indices, dtype=np.int64) + 1)


class SummaryFeatureCache:
    def __init__(self, states=None, *, feature_config=None, max_tokens=4096,
                 max_batch_bytes=256*1024*1024):
        self.feature_config = normalize_feature_config(feature_config)
        self.config_sha256 = feature_contract(self.feature_config)['config_sha256']
        require(type(max_tokens) is int and 0 < max_tokens <= 4096 and
                type(max_batch_bytes) is int and 0 < max_batch_bytes <= 256*1024*1024,
                'explicit bounded numerical-cache limits required')
        self.max_tokens, self.max_batch_bytes = max_tokens, max_batch_bytes
        self._states, self._features = [], []
        self._materialized = None
        self.feature_builds = 0
        if states is not None:
            states = np.asarray(states)
            require(states.ndim == 2 and states.shape[1] == 1519 and len(states)>0,
                    'nonempty original states[N,1519] required')
            for state in states:
                self.append(state)

    def append(self, state):
        state = np.asarray(state)
        require(state.shape == (1519,) and state.dtype == np.float32 and np.isfinite(state).all(),
                'finite float32 causal state[1519] required')
        require(state[17] == 0, 'terminal is not an action-history token')
        require(len(self._states) < self.max_tokens and
                (len(self._states)+1)*(1519+FEATURE_SIZE)*4 <= self.max_batch_bytes,
                'numerical cache exceeds bounded complete-shift limits')
        # Validate original raw before the C facade projects a private zero-mass copy.
        summary = np.asarray(build_features(state, config=self.feature_config), dtype=np.float32)
        require(summary.shape == (FEATURE_SIZE,) and np.isfinite(summary).all(),
                'C-v2 summary schema differs')
        self._states.append(immutable(state))
        self._features.append(immutable(summary))
        self._materialized = None
        self.feature_builds += 1

    def __len__(self):
        return len(self._states)

    @property
    def bytes(self):
        return len(self)*(1519+FEATURE_SIZE)*4

    def prefix(self, indices=None, *, max_tokens=None, max_batch_bytes=None):
        ids = tuple(range(len(self))) if indices is None else tuple(indices)
        require(bool(ids) and len(ids) == len(set(ids)) and
                all(type(i) in (int,np.int64,np.int32) and 0 <= i < len(self) for i in ids),
                'unique in-range decision indices required')
        ids = tuple(int(i) for i in ids)
        largest = max(ids)+1
        tokens = self.max_tokens if max_tokens is None else max_tokens
        byte_limit = self.max_batch_bytes if max_batch_bytes is None else max_batch_bytes
        require(type(tokens) is int and 0 < tokens <= self.max_tokens and
                type(byte_limit) is int and 0 < byte_limit <= self.max_batch_bytes and
                largest <= tokens and largest*(1519+FEATURE_SIZE)*4 <= byte_limit,
                'numerical prefix exceeds bounded complete-shift limits')
        # H is absent: the model needs only these requested current rows. Keep
        # their original ordinals as certificate metadata, not padded history.
        # Online one-row queries now copy/hash O(1), avoiding O(T²) prefixes.
        states=immutable(np.stack([self._states[i] for i in ids]))
        features=immutable(np.stack([self._features[i] for i in ids]))
        return SummarySequence(features, states, ids, FEATURE_VERSION, self.config_sha256,
                               _certificate(states,features,ids,self.config_sha256))


def validate_sequence(prefix, states, config):
    import torch
    require(isinstance(prefix,SummarySequence), 'certified raw/C-v2 sequence required')
    require(prefix.feature_version == FEATURE_VERSION and
            prefix.feature_config_sha256 == feature_contract(config)['config_sha256'],
            'C-v2 sequence feature contract differs')
    x, raw, ids = prefix.features, prefix.source_states, prefix.query_indices
    require(isinstance(x,np.ndarray) and x.dtype == np.float32 and x.ndim == 2 and
            x.shape[1] == FEATURE_SIZE and 0 < len(x) <= 4096 and np.isfinite(x).all() and
            isinstance(raw,np.ndarray) and raw.dtype == np.float32 and raw.shape == (len(x),1519) and
            np.isfinite(raw).all() and not x.flags.writeable and not raw.flags.writeable,
            'immutable finite raw/C-v2 sequence required')
    require(isinstance(ids,tuple) and len(ids)==len(states)==len(x) and len(ids)==len(set(ids)) and
            all(type(i) is int and 0 <= i < 4096 for i in ids),
            'numerical-cache decision indices differ')
    require(_certificate(raw,x,ids,prefix.feature_config_sha256)==prefix.certificate_sha256,
            'numerical-cache byte certificate differs')
    require(np.array_equal(raw,states.detach().cpu().numpy()),
            'numerical-cache current raw observation differs')
    return torch.tensor(x.copy(),dtype=torch.float32,device=states.device)


def pack_summary_prefixes(states, indices=None, *, feature_config=None, max_tokens=4096,
                         max_batch_bytes=256*1024*1024):
    states = np.asarray(states)
    require(states.ndim == 2 and states.shape[1] == 1519 and len(states)>0,
            'nonempty original states[N,1519] required')
    ids = tuple(range(len(states))) if indices is None else tuple(indices)
    require(bool(ids) and len(ids)==len(set(ids)) and
            all(type(i) in (int,np.int64,np.int32) and 0<=i<len(states) for i in ids),
            'unique in-range decision indices required')
    return SummaryFeatureCache(states[:max(ids)+1],feature_config=feature_config,
        max_tokens=max_tokens,max_batch_bytes=max_batch_bytes).prefix(ids)


class BatchFeatureCache:
    def __init__(self, *, feature_config, max_tokens=4096, max_bytes=256*1024*1024):
        self.feature_config=normalize_feature_config(feature_config)
        self.max_tokens,self.max_bytes=max_tokens,max_bytes
        self.entries=OrderedDict()
        self.feature_builds=0
        self.hits=0

    def clear(self):
        self.entries.clear()

    def get(self, states):
        states=np.asarray(states)
        require(states.ndim==2 and states.shape[1]==1519 and states.dtype==np.float32,
                'cache requires original float32 states[N,1519]')
        key=(states.shape,hashlib.sha256(states.tobytes()).digest(),
             feature_contract(self.feature_config)['config_sha256'])
        if key in self.entries:
            self.hits+=1
            self.entries.move_to_end(key)
            return self.entries[key]
        cache=SummaryFeatureCache(states,feature_config=self.feature_config,
            max_tokens=self.max_tokens,max_batch_bytes=self.max_bytes)
        while self.entries and sum(item.bytes for item in self.entries.values())+cache.bytes>self.max_bytes:
            self.entries.popitem(last=False)
        require(cache.bytes<=self.max_bytes,'numerical input cache exceeds byte limit')
        self.entries[key]=cache
        self.feature_builds+=cache.feature_builds
        return cache
