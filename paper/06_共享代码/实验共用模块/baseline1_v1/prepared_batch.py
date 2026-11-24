"""Immutable, validated current-batch data; no file or optimizer side effects.

The certificate is constructed only after the ordinary full row validator ran.
Composition reuses unchanged independent episode proofs and checks translated
boundaries/identities. Mutable TrainingBatch inputs never inherit this cache.
"""
from collections.abc import Mapping, Sequence
import copy
import hashlib
import json
import struct
from types import MappingProxyType

import numpy as np

from python.constrained_batch import TrainingBatch, validate_rows

_TOKEN = object()


def _equal(left, right):
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        try:
            return np.array_equal(left, right)
        except (TypeError, ValueError):
            return False
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (isinstance(left, Mapping) and isinstance(right, Mapping)
                and set(left) == set(right) and all(_equal(left[k], right[k]) for k in left))
    if isinstance(left, (list, tuple, FrozenSequence)) or isinstance(right, (list, tuple, FrozenSequence)):
        return (isinstance(left, (list, tuple, FrozenSequence))
                and isinstance(right, (list, tuple, FrozenSequence))
                and len(left) == len(right) and all(_equal(a, b) for a, b in zip(left, right)))
    return bool(left == right)


class FrozenMapping(Mapping):
    __slots__ = ('_data',)

    def __init__(self, values, token):
        if token is not _TOKEN:
            raise TypeError('use immutable snapshot factory')
        object.__setattr__(self, '_data', MappingProxyType(dict(values)))

    def __getitem__(self, key): return self._data[key]
    def __iter__(self): return iter(self._data)
    def __len__(self): return len(self._data)
    def __setattr__(self, name, value): raise TypeError('prepared metadata is immutable')
    def __delattr__(self, name): raise TypeError('prepared metadata is immutable')
    def __eq__(self, other): return _equal(self, other)
    def __deepcopy__(self, memo): return thaw(self)


class FrozenSequence(Sequence):
    __slots__ = ('_items', '_tuple')

    def __init__(self, values, token, *, tuple_kind=False):
        if token is not _TOKEN:
            raise TypeError('use immutable snapshot factory')
        object.__setattr__(self, '_items', tuple(values))
        object.__setattr__(self, '_tuple', tuple_kind)

    def __getitem__(self, key):
        if isinstance(key, slice): return FrozenSequence(self._items[key], _TOKEN, tuple_kind=self._tuple)
        return self._items[key]
    def __iter__(self): return iter(self._items)
    def __len__(self): return len(self._items)
    def __setattr__(self, name, value): raise TypeError('prepared metadata is immutable')
    def __delattr__(self, name): raise TypeError('prepared metadata is immutable')
    def __eq__(self, other): return _equal(self, other)
    def __deepcopy__(self, memo): return thaw(self)


def readonly_array(value):
    value = np.ascontiguousarray(value)
    if value.dtype.hasobject:
        raise TypeError('object arrays cannot be sealed training columns')
    # Immutable bytes, rather than only setflags(False): writeability cannot be restored.
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


class _Freezer:
    def __init__(self):
        self.pool = {}

    def node(self, value):
        if isinstance(value, (FrozenMapping, FrozenSequence)):
            value = thaw(value)
        if isinstance(value, np.ndarray):
            data = np.ascontiguousarray(value)
            if data.dtype.hasobject: raise TypeError('object arrays cannot be sealed')
            payload = data.tobytes()
            key = ('array', data.dtype.str, data.shape, payload)
            if key not in self.pool:
                self.pool[key] = np.frombuffer(payload, dtype=data.dtype).reshape(data.shape)
            return self.pool[key]
        if isinstance(value, Mapping):
            return FrozenMapping(((self.node(k), self.node(v)) for k, v in value.items()), _TOKEN)
        if isinstance(value, (list, tuple)):
            return FrozenSequence((self.node(v) for v in value), _TOKEN, tuple_kind=isinstance(value, tuple))
        if value is None or type(value) in (str, int, float, bool) or isinstance(value, np.generic):
            # Reuse repeated strings/numbers while holding only this one batch.
            if isinstance(value, np.generic):
                if value.dtype.hasobject:
                    raise TypeError('object scalars cannot be sealed')
                key = (type(value), value.dtype.str, value.tobytes())
            elif isinstance(value, float):
                key = (float, struct.pack('!d', value))
            else:
                key = (type(value), value)
            return self.pool.setdefault(key, value)
        raise TypeError('unsupported mutable training metadata: ' + type(value).__name__)


def freeze(value):
    return _Freezer().node(value)


def thaw(value):
    if isinstance(value, FrozenMapping): return {k: thaw(v) for k, v in value.items()}
    if isinstance(value, FrozenSequence):
        values = [thaw(v) for v in value]
        return tuple(values) if value._tuple else values
    if isinstance(value, np.ndarray): return value.copy()
    return value


def _raw_id(episodes, version):
    text = json.dumps([e['episode_id'] for e in episodes]) + version
    return hashlib.sha256(text.encode()).hexdigest()


class PreparedRows(Sequence):
    __slots__ = ('_rows', '_episodes', '_versions', '_columns', '_binding', '_online', '_v11')

    def __init__(self, rows, episodes, versions, token):
        if token is not _TOKEN: raise TypeError('use prepare_rows')
        object.__setattr__(self, '_rows', tuple(rows))
        object.__setattr__(self, '_episodes', episodes)
        object.__setattr__(self, '_versions', frozenset(versions))
        object.__setattr__(self, '_online', all(not r.get('critic_only', False)
            and r.get('behavior_kind') != 'anylogic_native' and r.get('old_log_prob') is not None for r in rows))
        object.__setattr__(self, '_v11', any('training_reward_v11' in r for r in rows)
            or any('reward_v11_provenance' in e for e in episodes))
        columns = {}
        names = ('observation', 'next_observation', 'action', 'reward', 'grade_costs', 'grade_valid',
                 'grade_bootstrap', 'grade_trace', 'reward_bootstrap', 'reward_trace', 'terminated')
        for name in names:
            columns[name] = readonly_array(np.asarray([r[name] for r in rows]))
        if self._online:
            columns['old_log_prob'] = readonly_array(np.asarray([r['old_log_prob'] for r in rows]))
        object.__setattr__(self, '_columns', MappingProxyType(columns))
        object.__setattr__(self, '_binding', (self._rows, self._episodes, self._versions, self._columns,
                                            self._online, self._v11))

    @property
    def episodes(self): return self._episodes
    @property
    def versions(self): return self._versions
    @property
    def online(self): return self._online
    @property
    def has_v11(self): return self._v11
    def __len__(self): return len(self._rows)
    def __iter__(self): return iter(self._rows)
    def __getitem__(self, key): return self._rows[key]
    def __setattr__(self, name, value): raise TypeError('prepared rows are immutable')
    def __delattr__(self, name): raise TypeError('prepared rows are immutable')
    def __deepcopy__(self, memo): return self

    def assert_bound(self, episodes):
        actual = (self._rows, self._episodes, self._versions, self._columns, self._online, self._v11)
        if any(a is not b for a, b in zip(actual[:4], self._binding[:4])) or actual[4:] != self._binding[4:]:
            raise ValueError('prepared row certificate changed')
        if episodes is not self._episodes:
            raise ValueError('prepared episode certificate differs')

    def column(self, name):
        self.assert_bound(self._episodes)
        return self._columns[name]


def prepare_rows(rows, episodes):
    if isinstance(rows, PreparedRows):
        rows.assert_bound(episodes)
        return rows
    freezer = _Freezer()
    frozen_rows = tuple(freezer.node(row) for row in rows)
    if any(r.get('critic_only', False) or r.get('behavior_kind') == 'anylogic_native' for r in frozen_rows):
        raise ValueError('native/offline rows retain the legacy validation path')
    frozen_episodes = freezer.node(episodes)
    versions = validate_rows(frozen_rows, frozen_episodes)
    return PreparedRows(frozen_rows, frozen_episodes, versions, _TOKEN)


def join_prepared_rows(parts, episodes):
    if not parts or not all(isinstance(p, PreparedRows) for p in parts):
        raise ValueError('prepared parts required')
    expected, rows, versions, offset = [], [], set(), 0
    for part in parts:
        part.assert_bound(part.episodes)
        for episode in part.episodes:
            e = thaw(episode)
            e.update(start=e['start'] + offset, stop=e['stop'] + offset)
            expected.append(e)
        rows.extend(part)
        versions.update(part.versions)
        offset += len(part)
    if not _equal(episodes, expected): raise ValueError('joined episode translation differs')
    ids = [e['episode_id'] for e in expected]
    seeds = [e['experiment_seed'] for e in expected]
    if len(set(ids)) != len(ids) or len(set(seeds)) != len(seeds):
        raise ValueError('duplicate prepared episode/seed')
    frozen_episodes = freeze(expected)
    return PreparedRows(rows, frozen_episodes, versions, _TOKEN)


def make_raw_batch(rows, source_kind):
    if not isinstance(rows, PreparedRows): raise ValueError('prepared rows required')
    rows.assert_bound(rows.episodes)
    if not rows.online: raise ValueError('native/offline Critic-only data cannot become an online PPO batch')
    if len(rows.versions) != 1: raise ValueError('online data mixes policy versions')
    version = next(iter(rows.versions))
    raw = TrainingBatch(rows, rows.episodes, version, _raw_id(rows.episodes, version))
    raw.source_kind = source_kind
    return raw
