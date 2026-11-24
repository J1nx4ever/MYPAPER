"""Immutable causal numerical inputs, never neural hidden-state caches.

One pure physical summary is constructed per observed state. A query context
contains only x[0:max(query)+1]; recurrent outputs are recomputed with current
weights on every forward. Byte certificates catch changed/forged cache views.
"""
from dataclasses import dataclass
from collections import OrderedDict
import hashlib

import numpy as np

from grade_commitment_ppo_v1.features import (FEATURE_SIZE, FEATURE_VERSION,
    build_features, feature_contract)
from grade_commitment_ppo_v1.models import normalize_feature_config

EXECUTION_VERSION = 'grade-commitment-full-sequence-fast-v1'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def immutable(value):
    value = np.asarray(value)
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


def _certificate(states, features, indices, config_sha256):
    h = hashlib.sha256(EXECUTION_VERSION.encode())
    for value in (states, features):
        h.update(str(value.dtype).encode()); h.update(str(value.shape).encode())
        h.update(value.tobytes())
    h.update(repr(tuple(indices)).encode()); h.update(config_sha256.encode())
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
    """Append-only same-episode feature cache; call a new instance at shift start."""
    def __init__(self, states=None, *, feature_config=None, max_tokens=4096,
                 max_batch_bytes=256*1024*1024):
        self.feature_config = normalize_feature_config(feature_config)
        self.config_sha256 = feature_contract(self.feature_config)['config_sha256']
        require(type(max_tokens) is int and 0 < max_tokens <= 4096 and
                type(max_batch_bytes) is int and 0 < max_batch_bytes <= 256*1024*1024,
                'explicit bounded full-sequence limits required')
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
        require(len(self._states) < self.max_tokens, 'summary sequence exceeds resource limit; never truncate history silently')
        require((len(self._states)+1)*(1519+FEATURE_SIZE)*4 <= self.max_batch_bytes,
                'immutable observation/summary cache exceeds byte limit')
        # build_features is pure and receives only the state currently observed.
        summary = np.asarray(build_features(state, config=self.feature_config), dtype=np.float32)
        require(summary.shape == (FEATURE_SIZE,) and np.isfinite(summary).all(), 'summary sequence schema differs')
        self._states.append(immutable(state)); self._features.append(immutable(summary))
        self._materialized = None; self.feature_builds += 1

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
        ids = tuple(int(i) for i in ids); largest = max(ids)+1
        tokens = self.max_tokens if max_tokens is None else max_tokens
        byte_limit = self.max_batch_bytes if max_batch_bytes is None else max_batch_bytes
        require(type(tokens) is int and 0 < tokens <= self.max_tokens and
                type(byte_limit) is int and 0 < byte_limit <= self.max_batch_bytes,
                'explicit bounded full-sequence limits required')
        require(largest <= tokens, 'summary sequence exceeds resource limit; never truncate history silently')
        require(largest*(1519+FEATURE_SIZE)*4 <= byte_limit, 'immutable observation/summary prefix exceeds byte limit')
        # Later observations are not included even when this cache has grown.
        if self._materialized is None:
            self._materialized = (immutable(np.stack(self._states)), immutable(np.stack(self._features)))
            # Keep row views into the immutable matrices, releasing the separate
            # append buffers rather than retaining a second full episode copy.
            self._states=list(self._materialized[0])
            self._features=list(self._materialized[1])
        states, features = (value[:largest] for value in self._materialized)
        return SummarySequence(features, states, ids, FEATURE_VERSION, self.config_sha256,
                               _certificate(states,features,ids,self.config_sha256))


def validate_sequence(prefix, states, config):
    require(isinstance(prefix,SummarySequence), 'certified full causal summary sequence required')
    require(prefix.feature_version == FEATURE_VERSION and
            prefix.feature_config_sha256 == feature_contract(config)['config_sha256'],
            'summary sequence feature contract differs')
    x, raw, ids = prefix.features, prefix.source_states, prefix.query_indices
    require(isinstance(x,np.ndarray) and x.dtype == np.float32 and x.ndim == 2 and
            x.shape[1] == FEATURE_SIZE and 0 < len(x) <= 4096 and np.isfinite(x).all() and
            isinstance(raw,np.ndarray) and raw.dtype == np.float32 and raw.shape == (len(x),1519) and
            np.isfinite(raw).all() and not x.flags.writeable and not raw.flags.writeable,
            'immutable nonempty full causal sequence required')
    require(isinstance(ids,tuple) and len(ids)==len(states) and len(ids)==len(set(ids)) and
            all(type(i) is int and 0 <= i < len(x) for i in ids) and max(ids)+1==len(x),
            'summary sequence decision indices differ')
    require(_certificate(raw,x,ids,prefix.feature_config_sha256)==prefix.certificate_sha256,
            'summary sequence byte certificate differs')
    require(np.array_equal(raw[list(ids)],states.detach().numpy()),
            'summary sequence current raw observation differs')
    return torch_current(prefix)


def torch_current(prefix):
    import torch
    return torch.tensor(prefix.features[list(prefix.query_indices)].copy(),dtype=torch.float32)


def pack_summary_prefixes(states, indices=None, *, feature_config=None, max_tokens=4096,
                          max_batch_bytes=256*1024*1024):
    states = np.asarray(states)
    require(states.ndim == 2 and states.shape[1] == 1519 and len(states)>0,
            'nonempty original states[N,1519] required')
    ids = tuple(range(len(states))) if indices is None else tuple(indices)
    require(bool(ids) and len(ids)==len(set(ids)) and
            all(type(i) in (int,np.int64,np.int32) and 0<=i<len(states) for i in ids),
            'unique in-range decision indices required')
    # Deliberately never read/validate future rows after the latest query.
    cache = SummaryFeatureCache(states[:max(ids)+1],feature_config=feature_config,
                                max_tokens=max_tokens,max_batch_bytes=max_batch_bytes)
    return cache.prefix(ids)


class BatchFeatureCache:
    """Bounded input-byte-keyed cache; changed observations cause a fresh build."""
    def __init__(self, *, feature_config, max_tokens=4096, max_bytes=256*1024*1024):
        self.feature_config=normalize_feature_config(feature_config)
        self.max_tokens,self.max_bytes=max_tokens,max_bytes
        self.entries=OrderedDict(); self.feature_builds=0; self.hits=0

    def clear(self):
        self.entries.clear()

    def get(self, states):
        states=np.asarray(states)
        require(states.ndim==2 and states.shape[1]==1519 and states.dtype==np.float32,
                'cache requires original float32 states[N,1519]')
        key=(states.shape,hashlib.sha256(states.tobytes()).digest())
        if key in self.entries:
            self.hits+=1; self.entries.move_to_end(key)
            return self.entries[key]
        cache=SummaryFeatureCache(states,feature_config=self.feature_config,
                                  max_tokens=self.max_tokens,max_batch_bytes=self.max_bytes)
        while self.entries and sum(item.bytes for item in self.entries.values())+cache.bytes>self.max_bytes:
            self.entries.popitem(last=False)
        require(cache.bytes<=self.max_bytes,'causal feature cache exceeds byte limit')
        self.entries[key]=cache; self.feature_builds+=cache.feature_builds
        return cache
