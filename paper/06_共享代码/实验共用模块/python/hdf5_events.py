"""Lossless, opt-in event archives and bounded streaming evidence readers.

The archive preserves source identity and the source JSONL byte hash.  Its own
file hash and canonical decoded-event hash are separate: rewriting JSON spacing
cannot reproduce an original byte hash.  No simulator or training is started.
"""
from collections import OrderedDict, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path


CHUNK_ROWS = 4096
FORMAT = 'hdf5_typed_events_v1'
ENVELOPE = frozenset(('schema_version', 'episode_id', 'config_sha256', 'event_seq',
                      'sim_time_seconds', 'event_type', 'data'))


class EventStorageError(ValueError):
    """Incomplete, corrupt, unsupported or unbound event storage."""


def _require(condition, message):
    if not condition:
        raise EventStorageError(message)


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False)


def _parse(value):
    def pairs(items):
        result = {}
        for key, item in items:
            _require(key not in result, 'duplicate JSON field: '+key)
            result[key] = item
        return result
    def constant(value):
        raise EventStorageError('nonfinite JSON constant: '+value)
    return json.loads(value, object_pairs_hook=pairs, parse_constant=constant)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _hash(value, name):
    _require(isinstance(value, str) and len(value) == 64 and
             all(c in '0123456789abcdef' for c in value), 'invalid '+name)
    return value


def _manifest(directory, manifest=None):
    if manifest is None:
        try:
            manifest = _parse((directory / 'events_manifest.json').read_bytes())
        except (OSError, ValueError) as exc:
            raise EventStorageError('cannot read event manifest: '+str(exc)) from exc
    _require(type(manifest) is dict and manifest.get('schema_version') == 1 and
             manifest.get('record_kind') == 'anylogic_boundary_events',
             'unsupported event manifest')
    _require(manifest.get('complete') is True and manifest.get('terminated') is True
             and manifest.get('truncated') is False, 'event recording is incomplete or abnormal')
    _require(manifest.get('source_kind') in ('actual_anylogic', 'synthetic_unit_test'),
             'unknown event source')
    _require(type(manifest.get('event_count')) is int and manifest['event_count'] >= 2,
             'invalid event count')
    _hash(manifest.get('events_sha256'), 'events_sha256')
    _hash(manifest.get('config_sha256'), 'config_sha256')
    _require(isinstance(manifest.get('episode_id'), str) and manifest['episode_id'],
             'missing episode identity')
    return manifest


def _path(directory, manifest):
    storage = manifest.get('event_storage', {})
    _require(type(storage) is dict, 'invalid event_storage')
    default = 'events.jsonl' if (directory / 'events.jsonl').exists() else 'events.jsonl.gz'
    filename = storage.get('filename', default)
    _require(isinstance(filename, str) and filename and
             Path(filename).name == filename, 'event filename must be a local basename')
    path = directory / filename
    _require(path.is_file(), 'missing event stream: '+filename)
    if 'sha256' in storage:
        _require(_sha256(path) == _hash(storage['sha256'], 'event_storage.sha256'),
                 'event storage file hash mismatch')
    return path, storage


def _validate_event(event, seq, manifest):
    _require(type(event) is dict and type(event.get('schema_version')) is int and
             event['schema_version'] == 1, 'invalid event envelope at '+str(seq))
    _require(type(event.get('event_seq')) is int and event['event_seq'] == seq,
             'event sequence gap/duplicate at '+str(seq))
    _require(event.get('episode_id') == manifest['episode_id'] and
             event.get('config_sha256') == manifest['config_sha256'],
             'event identity changed at '+str(seq))
    _require(isinstance(event.get('event_type'), str) and event['event_type'] and
             type(event.get('data')) is dict, 'invalid event type/data at '+str(seq))
    value = event.get('sim_time_seconds')
    _require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
             'invalid event sim_time_seconds')


def iter_event_bundle(bundle_dir, manifest=None, *, chunk_rows=CHUNK_ROWS):
    """Yield original event objects; verify hashes/counts before exhaustion.

    JSONL ``events_sha256`` always hashes the *decompressed original bytes*,
    including final newlines.  A gzip file can additionally bind compressed bytes
    with ``event_storage.sha256``.  HDF5 requires both file and logical hashes.
    Callers must exhaust this iterator before accepting evidence.
    """
    _require(type(chunk_rows) is int and 1 <= chunk_rows <= CHUNK_ROWS, 'invalid chunk_rows')
    directory = Path(bundle_dir)
    manifest = _manifest(directory, manifest)
    try:
        path, storage = _path(directory, manifest)
        if storage.get('format') == FORMAT:
            events = _iter_hdf5(path, manifest, storage, chunk_rows)
        else:
            _require(storage.get('format') in (None, 'jsonl', 'jsonl_gzip_v1'),
                     'unsupported event storage format')
            events = _iter_jsonl(path, manifest)
        count = 0
        for count, event in enumerate(events, 1):
            _validate_event(event, count - 1, manifest)
            yield event
        _require(count == manifest['event_count'], 'event count mismatch')
    except (OSError, EOFError, KeyError, TypeError, UnicodeError) as exc:
        raise EventStorageError('cannot read event stream: '+str(exc)) from exc


def _iter_jsonl(path, manifest):
    digest, count = hashlib.sha256(), 0
    opener = gzip.open if path.suffix == '.gz' else Path.open
    with opener(path, 'rb') as stream:
        for seq, line in enumerate(stream):
            _require(line.endswith(b'\n'), 'event stream lacks final newline')
            digest.update(line)
            try:
                event = _parse(line)
            except (ValueError, UnicodeError) as exc:
                raise EventStorageError(f'bad JSON at event {seq}: {exc}') from exc
            count += 1
            yield event
    _require(digest.hexdigest() == manifest['events_sha256'], 'event stream hash mismatch')
    _require(count == manifest['event_count'], 'event count mismatch')


def _libraries():
    try:
        import h5py
        import numpy as np
    except ImportError as exc:
        raise EventStorageError('HDF5 events require h5py in anylogic-env') from exc
    return h5py, np


def _dataset(group, name, dtype, *, length=0, chunk_rows=CHUNK_ROWS):
    return group.create_dataset(name, shape=(length,), maxshape=(None,), dtype=dtype,
                                chunks=(chunk_rows,), compression='gzip',
                                compression_opts=6, shuffle=True)


def _append(dataset, values):
    offset = len(dataset)
    dataset.resize((offset + len(values),))
    dataset[offset:] = values


class _Dictionary:
    def __init__(self, group, chunk_rows):
        h5py, _ = _libraries()
        self.dataset = _dataset(group, 'dictionary', h5py.string_dtype('utf-8'),
                                chunk_rows=chunk_rows)
        # Dictionary values live in HDF5.  Keep only a bounded lookup cache in
        # RAM; an evicted value can safely receive another equivalent ID later.
        self.ids = OrderedDict()

    def encode(self, values):
        additions, encoded = [], []
        for value in values:
            if value not in self.ids:
                self.ids[value] = len(self.dataset) + len(additions)
                additions.append(value)
                if len(self.ids) > CHUNK_ROWS:
                    self.ids.popitem(last=False)
            else:
                self.ids.move_to_end(value)
            encoded.append(self.ids[value])
        if additions:
            _append(self.dataset, additions)
        return encoded


def _value_kind(value):
    if type(value) is bool:
        return 'bool'
    if type(value) is int and -(2 ** 63) <= value < 2 ** 63:
        return 'int64'
    if type(value) is float:
        return 'float64'
    if type(value) is str and '\x00' not in value:
        return 'string'
    return 'json'


class _Column:
    def __init__(self, group, key, example, prior_rows, chunk_rows):
        self.group, self.kind = group, _value_kind(example)
        group.attrs['key'], group.attrs['kind'] = key, self.kind
        self.present = _dataset(group, 'present', 'u1', length=prior_rows, chunk_rows=chunk_rows)
        dtype = {'bool': '?', 'int64': '<i8', 'float64': '<f8', 'string': '<u8', 'json': '<u8'}[self.kind]
        self.values = _dataset(group, 'values', dtype, length=prior_rows, chunk_rows=chunk_rows)
        self.dictionary = _Dictionary(group, chunk_rows) if self.kind in ('string', 'json') else None
        self.fallback, self.chunk_rows = None, chunk_rows

    def append(self, rows, key):
        mask, values, fallback = [], [], []
        for row in rows:
            if key not in row:
                mask.append(0); values.append(0 if self.dictionary is None else '')
            elif _value_kind(row[key]) != self.kind:
                mask.append(2); values.append(0 if self.dictionary is None else '')
            else:
                mask.append(1)
                values.append(_json(row[key]) if self.kind == 'json' else row[key])
            fallback.append(_json(row[key]) if mask[-1] == 2 else '')
        if 2 in mask and self.fallback is None:
            h5py, _ = _libraries()
            self.fallback = _dataset(self.group, 'fallback_json', h5py.string_dtype('utf-8'),
                                     length=len(self.present), chunk_rows=self.chunk_rows)
        if self.fallback is not None:
            _append(self.fallback, fallback)
        _append(self.present, mask)
        _append(self.values, self.dictionary.encode(values) if self.dictionary else values)


class _Table:
    def __init__(self, group, chunk_rows):
        self.group, self.chunk_rows, self.columns, self.rows = group, chunk_rows, {}, 0

    def append(self, rows):
        for row in rows:
            for key, value in row.items():
                if key not in self.columns:
                    subgroup = self.group.create_group(f'c{len(self.columns):04d}')
                    self.columns[key] = _Column(subgroup, key, value, self.rows, self.chunk_rows)
        for key, column in self.columns.items():
            column.append(rows, key)
        self.rows += len(rows)
        self.group.attrs['row_count'] = self.rows


class _Writer:
    def __init__(self, path, manifest, chunk_rows, source_manifest_sha256):
        h5py, _ = _libraries()
        self.file = h5py.File(path, 'x')
        self.file.attrs.update(format=FORMAT, complete=False, episode_id=manifest['episode_id'],
            config_sha256=manifest['config_sha256'], source_kind=manifest['source_kind'],
            source_events_sha256=manifest['events_sha256'], schema_version=1,
            config_json=_json(manifest.get('config')),
            source_manifest_sha256=source_manifest_sha256)
        for key in ('recording_id', 'instrumentation_revision', 'seed', 'run_index'):
            self.file.attrs[key] = manifest[key]
        self.order = self.file.create_group('event_order')
        for name, dtype in (('event_seq', '<u8'), ('sim_time_seconds', '<f8'),
                            ('time_is_int', '?'), ('event_type_id', '<u4'), ('type_row', '<u8')):
            _dataset(self.order, name, dtype, chunk_rows=chunk_rows)
        self.kinds = self.file.create_group('event_types')
        self.tables, self.kind_ids, self.buffer = {}, {}, []
        self.chunk_rows, self.count, self.digest = chunk_rows, 0, hashlib.sha256()

    def append(self, event):
        self.digest.update((_json(event)+'\n').encode('utf-8'))
        self.buffer.append(event)
        if len(self.buffer) >= self.chunk_rows:
            self.flush()

    def flush(self):
        per_kind, order = defaultdict(list), defaultdict(list)
        for event in self.buffer:
            kind = event['event_type']
            if kind not in self.tables:
                kind_id = len(self.kind_ids)
                self.kind_ids[kind] = kind_id
                group = self.kinds.create_group(f'k{kind_id:04d}')
                group.attrs['event_type'] = kind
                self.tables[kind] = (_Table(group.create_group('data'), self.chunk_rows),
                                     _Table(group.create_group('envelope_extra'), self.chunk_rows))
            data, extra = self.tables[kind]
            order['event_seq'].append(event['event_seq'])
            order['sim_time_seconds'].append(event['sim_time_seconds'])
            order['time_is_int'].append(type(event['sim_time_seconds']) is int)
            order['event_type_id'].append(self.kind_ids[kind])
            order['type_row'].append(data.rows + len(per_kind[kind]))
            per_kind[kind].append(event)
        for name, values in order.items():
            _append(self.order[name], values)
        for kind, rows in per_kind.items():
            data, extra = self.tables[kind]
            data.append([event['data'] for event in rows])
            extra.append([{k: v for k, v in event.items() if k not in ENVELOPE} for event in rows])
        self.count += len(self.buffer)
        self.buffer.clear()
        self.file.flush()

    def finish(self):
        self.flush()
        self.file.attrs.update(complete=True, event_count=self.count,
                               logical_sha256=self.digest.hexdigest())
        self.file.close()
        return self.digest.hexdigest()


def _decode_table(group, first, last):
    rows = [{} for _ in range(last - first)]
    _require(last <= group.attrs['row_count'], 'event table row out of range')
    for column in group.values():
        key, kind = column.attrs['key'], column.attrs['kind']
        mask, values = column['present'][first:last], column['values'][first:last]
        dictionary = {}
        for row, present, value, index in zip(rows, mask, values, range(first, last)):
            _require(int(present) in (0, 1, 2), 'invalid event column presence')
            if present == 0:
                continue
            if present == 2:
                _require('fallback_json' in column, 'missing mixed-type fallback')
                row[key] = _parse(column['fallback_json'].asstr()[index])
            elif kind in ('string', 'json'):
                value = int(value)
                if value not in dictionary:
                    dictionary[value] = column['dictionary'].asstr()[value]
                row[key] = _parse(dictionary[value]) if kind == 'json' else dictionary[value]
            else:
                row[key] = {'bool': bool, 'float64': float, 'int64': int}[kind](value)
    return rows


def _iter_hdf5(path, manifest, storage, chunk_rows):
    h5py, _ = _libraries()
    _hash(storage.get('sha256'), 'event_storage.sha256')
    logical = _hash(storage.get('logical_sha256'), 'event_storage.logical_sha256')
    digest = hashlib.sha256()
    with h5py.File(path, 'r') as file:
        _require(file.attrs.get('format') == FORMAT and bool(file.attrs.get('complete', False)),
                 'incomplete or unsupported HDF5 event archive')
        for key in ('episode_id', 'config_sha256', 'source_kind', 'recording_id',
                    'instrumentation_revision', 'seed', 'run_index'):
            _require(file.attrs.get(key) == manifest[key], 'HDF5 event identity mismatch: '+key)
        _require(file.attrs.get('config_json') == _json(manifest.get('config')),
                 'HDF5 event configuration mismatch')
        _require(file.attrs.get('source_events_sha256') == manifest['events_sha256'],
                 'HDF5 original source hash mismatch')
        provenance = manifest.get('archive_provenance')
        _require(type(provenance) is dict and provenance.get('source_manifest_sha256') ==
                 file.attrs.get('source_manifest_sha256') and
                 provenance.get('source_events_sha256') == manifest['events_sha256'] and
                 provenance.get('source_event_count') == manifest['event_count'] and
                 provenance.get('source_kind') == manifest['source_kind'],
                 'HDF5 source provenance mismatch')
        count = manifest['event_count']
        _require(file.attrs.get('event_count') == count and
                 file.attrs.get('logical_sha256') == logical, 'HDF5 event count/logical hash mismatch')
        order = file['event_order']
        _require(all(len(order[key]) == count for key in
            ('event_seq', 'sim_time_seconds', 'time_is_int', 'event_type_id', 'type_row')),
            'HDF5 event order count mismatch')
        for start in range(0, count, chunk_rows):
            stop = min(count, start + chunk_rows)
            arrays = {key: value[start:stop] for key, value in order.items()}
            decoded = {}
            for kind_id in set(map(int, arrays['event_type_id'])):
                positions = [i for i, value in enumerate(arrays['event_type_id']) if int(value) == kind_id]
                indexes = [int(arrays['type_row'][i]) for i in positions]
                _require(indexes == list(range(indexes[0], indexes[0]+len(indexes))),
                         'HDF5 event type rows regressed or skipped')
                group = file['event_types'][f'k{kind_id:04d}']
                data = _decode_table(group['data'], indexes[0], indexes[-1]+1)
                extra = _decode_table(group['envelope_extra'], indexes[0], indexes[-1]+1)
                for position, row, envelope in zip(positions, data, extra):
                    event = dict(envelope, schema_version=1, episode_id=manifest['episode_id'],
                        config_sha256=manifest['config_sha256'],
                        event_seq=int(arrays['event_seq'][position]),
                        sim_time_seconds=(int if arrays['time_is_int'][position] else float)(
                            arrays['sim_time_seconds'][position]), event_type=group.attrs['event_type'], data=row)
                    decoded[position] = event
            for position in range(stop - start):
                event = decoded[position]
                digest.update((_json(event)+'\n').encode('utf-8'))
                yield event
    _require(digest.hexdigest() == logical, 'HDF5 decoded event logical hash mismatch')


def archive_event_bundle(source_dir, output_dir, *, chunk_rows=CHUNK_ROWS):
    """Create a separate verified HDF5 derivative; never overwrite source files.

    All source events are verified before the output directory is created. A
    second streaming pass writes at most ``chunk_rows`` event objects at once.
    Any interrupted derivative remains incomplete and is rejected by readers.
    """
    _require(type(chunk_rows) is int and 1 <= chunk_rows <= CHUNK_ROWS, 'invalid chunk_rows')
    _libraries()
    source, output = Path(source_dir), Path(output_dir)
    manifest = _manifest(source)
    manifest_hash = _sha256(source / 'events_manifest.json')
    source_path, _ = _path(source, manifest)
    for _ in iter_event_bundle(source, manifest, chunk_rows=chunk_rows):
        pass
    _require(_sha256(source / 'events_manifest.json') == manifest_hash,
             'source manifest changed during verification')
    output.mkdir(parents=True, exist_ok=False)
    writer = _Writer(output / 'events.h5', manifest, chunk_rows, manifest_hash)
    try:
        for event in iter_event_bundle(source, manifest, chunk_rows=chunk_rows):
            writer.append(event)
        logical = writer.finish()
    except BaseException:
        writer.file.close()
        raise
    derived = dict(manifest)
    derived['event_storage'] = dict(format=FORMAT, filename='events.h5',
        sha256=_sha256(output / 'events.h5'), logical_sha256=logical,
        hash_semantics='file SHA256 plus canonical decoded JSON event SHA256; events_sha256 retains original decompressed JSONL byte hash')
    derived['archive_provenance'] = dict(source_directory=str(source.resolve()),
        source_manifest_sha256=manifest_hash, source_file=source_path.name,
        source_file_sha256=_sha256(source_path), source_events_sha256=manifest['events_sha256'],
        source_event_count=manifest['event_count'], source_kind=manifest['source_kind'],
        transformation='lossless typed columns; no event filtering, merging or reordering')
    (output / 'events_manifest.json').write_text(_json(derived)+'\n', encoding='utf-8')
    for _ in iter_event_bundle(output, derived, chunk_rows=chunk_rows):
        pass
    return derived
