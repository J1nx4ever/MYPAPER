"""Bounded, typed HDF5 storage shared by collection, training, and audit.

No pickle or per-transition JSON is used. JSON describes the column schema,
shared constants, and small irregular lists (for example closed windows).
The receiving journal deliberately remains distinct from a sealed dataset.
"""
from collections import OrderedDict
from collections.abc import Sequence
import copy
import hashlib
import json
import numbers
import os
from pathlib import Path
import uuid

import numpy as np

SCHEMA_VERSION = 'mining-hdf5-typed-v1'
OBSERVATION_SIZE = 1519
_MISSING = object()


def _h5py():
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError('HDF5 storage requires h5py in Conda environment anylogic-env') from exc
    return h5py


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        if not all(isinstance(k, str) for k in value):
            raise ValueError('HDF5 field names must be strings')
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    return value


def _json(value):
    return json.dumps(_plain(value), ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False)


def _text(value):
    return value.decode('utf-8') if isinstance(value, bytes) else str(value)


def _dataset(group, name, array):
    array = np.asarray(array)
    options = dict(compression='gzip', compression_opts=4, shuffle=True, fletcher32=True)
    if not array.size:
        options = {}
    return group.create_dataset(name, data=array, **options)


class _Constants:
    def __init__(self, file, write=False):
        self.file, self.write = file, write
        self.ids, self.values = {}, []
        if not write:
            self.values = [_text(v) for v in file['constants'][...]]

    def intern(self, value):
        value = _json(value)
        if value not in self.ids:
            self.ids[value] = len(self.values)
            self.values.append(value)
        return self.ids[value]

    def get(self, index):
        if not 0 <= index < len(self.values):
            raise ValueError('HDF5 constant reference out of bounds')
        return json.loads(self.values[index])

    def save(self):
        dt = _h5py().string_dtype('utf-8')
        if 'constants' in self.file:
            del self.file['constants']
        self.file.create_dataset('constants', data=np.asarray(self.values, dtype=object), dtype=dt)


class _Encoder:
    def __init__(self, group, constants, count):
        self.group, self.constants, self.count = group, constants, count
        self.columns = {'float64': [], 'int64': [], 'uint64': [], 'uint8': []}

    def column(self, values, dtype):
        a = np.asarray(values, dtype=dtype).reshape(self.count, -1)
        start = sum(x.shape[1] for x in self.columns[dtype])
        self.columns[dtype].append(a)
        return dict(dtype=dtype, offset=start, width=a.shape[1])

    def node(self, values, *, root=False):
        present = [v is not _MISSING and v is not None for v in values]
        actual = [v for v, ok in zip(values, present) if ok]
        result = {}
        statuses = [0 if v is _MISSING else 1 if v is None else 2 for v in values]
        if not all(s == 2 for s in statuses):
            result['status'] = self.column(statuses, 'uint8')
        if not actual:
            return dict(result, kind='empty')
        # Dictionaries are always structural: a one-row block must never turn
        # its entire transition or request into a giant JSON value.
        if all(isinstance(v, dict) for v in actual):
            keys = sorted({k for v in actual for k in v})
            if not all(isinstance(k, str) for k in keys):
                raise ValueError('HDF5 field names must be strings')
            fields = {k: self.node([v.get(k, _MISSING) if ok else _MISSING
                                   for v, ok in zip(values, present)]) for k in keys}
            return dict(result, kind='dict', fields=fields)
        normalized = [_plain(v) for v in actual]
        first = _json(normalized[0])
        if all(_json(v) == first for v in normalized):
            if isinstance(normalized[0],list) and len(first.encode()) > 65536:
                raise ValueError('irregular HDF5 list exceeds 64 KiB; add a typed table')
            constant=dict(result, kind='constant', ref=self.constants.intern(normalized[0]))
            if all(isinstance(v,np.ndarray) for v in actual):
                constant.update(container='ndarray',original_dtype=str(actual[0].dtype))
            return constant
        def category(v):
            if isinstance(v, (bool, np.bool_)): return 'bool'
            if isinstance(v, numbers.Integral): return 'int'
            if isinstance(v, numbers.Real): return 'float'
            if isinstance(v, str): return 'str'
            if isinstance(v, (list, tuple, np.ndarray)): return 'array'
            raise ValueError('unsupported HDF5 value type: ' + type(v).__name__)
        kinds = {category(v) for v in actual}
        if len(kinds) != 1:
            # Preserve integer/float/boolean distinctions instead of casting
            # IDs through float64, which can lose integers above 2**53.
            tags = sorted(kinds)
            result.update(kind='union', tags=self.column([tags.index(category(v))+1 if ok else 0
                          for v, ok in zip(values, present)], 'uint8'),
                          variants={k: self.node([v if ok and category(v)==k else _MISSING
                                    for v, ok in zip(values, present)]) for k in tags}, names=tags)
            return result
        kind = next(iter(kinds))
        if kind in ('bool', 'int', 'float'):
            dtype = 'uint8' if kind == 'bool' else 'float64' if kind == 'float' else 'int64'
            if kind == 'int' and max(int(v) for v in actual) > np.iinfo(np.int64).max:
                if min(int(v) for v in actual) < 0: raise ValueError('integer range exceeds 64 bits')
                dtype = 'uint64'
            raw = [v if ok else 0 for v, ok in zip(values, present)]
            a = np.asarray(raw, dtype=dtype)
            if kind == 'float' and not np.all(np.isfinite(a)):
                raise ValueError('non-finite floating point data')
            return dict(result, kind=kind, column=self.column(a, dtype))
        if kind == 'array':
            arrays = [np.asarray(v) for v in actual]
            def homogeneous(value):
                if isinstance(value,np.ndarray):return value.dtype.kind in 'biuf'
                def leaves(v):
                    if isinstance(v,(tuple,list)):
                        for child in v:yield from leaves(child)
                    else:yield category(v)
                return len(set(leaves(value)))<=1
            if (all(a.dtype.kind in 'biuf' and a.shape == arrays[0].shape for a in arrays)
                    and all(homogeneous(v) for v in actual) and arrays[0].size
                    and len({a.dtype.kind for a in arrays})==1):
                dtype = ('float64' if any(a.dtype.kind == 'f' for a in arrays) else
                         'uint8' if all(a.dtype.kind == 'b' for a in arrays) else 'int64')
                if dtype=='int64' and any(np.max(a)>np.iinfo(np.int64).max for a in arrays):
                    if any(np.min(a)<0 for a in arrays):raise ValueError('integer range exceeds 64 bits')
                    dtype='uint64'
                shape = arrays[0].shape
                raw = [np.asarray(v, dtype=dtype) if ok else np.zeros(shape, dtype=dtype)
                       for v, ok in zip(values, present)]
                a = np.stack(raw)
                if dtype == 'float64' and not np.all(np.isfinite(a)):
                    raise ValueError('non-finite floating point arrays')
                return dict(result, kind='array', shape=shape,
                            container='ndarray' if all(isinstance(v,np.ndarray) for v in actual) else 'list',
                            original_dtype=str(arrays[0].dtype), column=self.column(a, dtype))
        # Only strings and small irregular lists reach the shared dictionary.
        # Full dictionaries have already been expanded structurally above.
        if kind != 'str' and any(len(_json(v).encode()) > 65536 for v in actual):
            raise ValueError('irregular HDF5 list exceeds 64 KiB; add a typed table')
        ids = [self.constants.intern(_plain(v)) if ok else 0 for v, ok in zip(values, present)]
        return dict(result, kind='dictionary', column=self.column(ids, 'uint64'))

    def finish(self, spec):
        for dtype, columns in self.columns.items():
            if columns:
                _dataset(self.group, dtype, np.concatenate(columns, axis=1))
        payload = np.frombuffer(_json(spec).encode('utf-8'), dtype=np.uint8)
        _dataset(self.group, 'schema', payload)
        self.group.attrs['count'] = self.count


def _encode(group, rows, constants):
    encoder = _Encoder(group, constants, len(rows))
    encoder.finish(encoder.node(rows, root=True))


class _Decoder:
    def __init__(self, group, constants):
        self.spec = json.loads(bytes(group['schema'][...]).decode('utf-8'))
        self.columns = {k: group[k][...] for k in ('float64','int64','uint64','uint8') if k in group}
        self.constants = constants
        self.count = int(group.attrs['count'])
        if any(v.ndim != 2 or len(v) != self.count for v in self.columns.values()):
            raise ValueError('HDF5 typed column length mismatch')

    def value(self, column, index):
        a = self.columns[column['dtype']]
        start, width = column['offset'], column['width']
        if start < 0 or width < 1 or start+width > a.shape[1]:
            raise ValueError('HDF5 column reference out of bounds')
        return a[index, start:start+width]

    def node(self, spec, index):
        if 'status' in spec:
            state = int(self.value(spec['status'], index)[0])
            if state == 0: return _MISSING
            if state == 1: return None
            if state != 2: raise ValueError('invalid HDF5 presence flag')
        kind = spec['kind']
        if kind == 'empty': return None
        if kind == 'dict':
            result = {}
            for key, child in spec['fields'].items():
                value = self.node(child, index)
                if value is not _MISSING: result[key] = value
            return result
        if kind == 'constant':
            value=self.constants.get(spec['ref'])
            return np.asarray(value,dtype=spec['original_dtype']) if spec.get('container')=='ndarray' else value
        if kind == 'union':
            tag = int(self.value(spec['tags'], index)[0])-1
            if not 0 <= tag < len(spec['names']): raise ValueError('invalid HDF5 union tag')
            return self.node(spec['variants'][spec['names'][tag]], index)
        value = self.value(spec['column'], index)
        if kind == 'bool': return bool(value[0])
        if kind == 'int': return int(value[0])
        if kind == 'float': return float(value[0])
        if kind == 'array':
            array = value.reshape(spec['shape']).astype(spec['original_dtype'])
            return array if spec['container']=='ndarray' else array.tolist()
        if kind == 'dictionary': return self.constants.get(int(value[0]))
        raise ValueError('unsupported HDF5 node kind: ' + str(kind))

    def row(self, index):
        return self.node(self.spec, index)


def _content_digest(file):
    digest = hashlib.sha256()
    def emit(value):
        data = value if isinstance(value, bytes) else _json(value).encode('utf-8')
        digest.update(len(data).to_bytes(8,'big')); digest.update(data)
    def visit(name, obj):
        emit(name)
        emit({k: _plain(v) for k,v in sorted(obj.attrs.items()) if k != 'content_sha256'})
        if not isinstance(obj, _h5py().Dataset): return
        emit(str(obj.dtype)); emit(list(obj.shape))
        if obj.shape == ():
            chunks = [obj[()]]
        else:
            chunks = (obj[start:start+128] for start in range(0,len(obj),128))
        for chunk in chunks:
            a = np.asarray(chunk)
            if a.dtype.kind in 'OUS':
                for item in a.flat: emit(_text(item))
            else: emit(a.tobytes(order='C'))
    visit('/',file); file.visititems(visit)
    return digest.hexdigest()


def _bounds(rows, episodes):
    cursor = 0
    for episode in episodes:
        start,stop = episode['start'],episode['stop']
        if type(start) is not int or type(stop) is not int or start != cursor or stop <= start or stop > len(rows):
            raise ValueError('episode boundary gap/overlap')
        if episode.get('normal_completion') is not True:
            raise ValueError('incomplete episode cannot be sealed')
        cursor = stop
    if not episodes or not rows or cursor != len(rows):
        raise ValueError('empty or unassigned HDF5 transitions')


def _observation(value):
    array = np.asarray(value)
    converted = np.asarray(value,dtype=np.float32)
    if converted.shape != (OBSERVATION_SIZE,) or not np.all(np.isfinite(converted)):
        raise ValueError('HDF5 observation must be finite 1519-dimensional state')
    if not np.array_equal(array,converted):
        raise ValueError('observation is not losslessly representable as float32')
    return converted


def write_dataset(path, rows, episodes, mc_reward, mc_grade, *, chunk_rows=64):
    """Atomically seal a complete dataset; read at most ``chunk_rows`` rows.

    Domain/reward validation remains the caller's responsibility. This layer
    enforces storage precision, continuity, boundaries, labels, and integrity.
    """
    if type(chunk_rows) is not int or chunk_rows <= 0: raise ValueError('invalid chunk_rows')
    path=Path(path)
    if path.exists(): raise FileExistsError('refusing HDF5 dataset overwrite: '+str(path))
    _bounds(rows,episodes)
    yr,yg=np.asarray(mc_reward,dtype=np.float64),np.asarray(mc_grade,dtype=np.float64)
    if yr.shape != (len(rows),) or yg.shape != (len(rows),4) or not np.all(np.isfinite(yr)) or not np.all(np.isfinite(yg)):
        raise ValueError('HDF5 label shape/finiteness mismatch')
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.partial-'+uuid.uuid4().hex)
    try:
        with _h5py().File(temporary,'x') as file:
            file.attrs.update(schema_version=SCHEMA_VERSION,kind='dataset',complete=False,
                              transitions=len(rows),episodes=len(episodes),chunk_rows=chunk_rows)
            constants=_Constants(file,write=True)
            states=file.create_dataset('states',shape=(len(rows)+len(episodes),OBSERVATION_SIZE),
                                      dtype='float32',chunks=(min(chunk_rows,len(rows)+len(episodes)),OBSERVATION_SIZE),
                                      compression='gzip',compression_opts=4,fletcher32=True)
            episode_index=np.empty(len(rows),dtype=np.int64)
            state_index=np.empty(len(rows),dtype=np.int64)
            transitions=file.create_group('transitions')
            # State writing is streamed by episode and exact continuity is
            # checked before either observation alias is discarded.
            for ei,e in enumerate(episodes):
                initial=_observation(rows[e['start']]['observation'])
                if 'initial_observation' in e and not np.array_equal(initial,_observation(e['initial_observation'])):
                    raise ValueError('episode initial observation mismatch')
                base=e['start']+ei; states[base]=initial; previous=initial
                for index in range(e['start'],e['stop']):
                    row=rows[index]
                    if not np.array_equal(previous,_observation(row['observation'])):
                        raise ValueError('discontinuous states inside episode')
                    previous=_observation(row['next_observation']); states[index+ei+1]=previous
                    episode_index[index]=ei; state_index[index]=index+ei
            # Request predecessor references remove action_info duplication.
            action_ref=np.full(len(rows),-2,dtype=np.int64)
            for start in range(0,len(rows),chunk_rows):
                batch=[]
                for index in range(start,min(start+chunk_rows,len(rows))):
                    source=rows[index]
                    row={k:v for k,v in source.items() if k not in ('observation','next_observation')}
                    if 'action_info' in row:
                        e=episodes[int(episode_index[index])]
                        predecessor=e['start_info'] if index==e['start'] else rows[index-1]['settlement_info']
                        if _json(row['action_info']) == _json(predecessor):
                            action_ref[index]=-1 if index==e['start'] else index-1
                            row.pop('action_info')
                    batch.append(row)
                group=transitions.create_group(f'{start:012d}')
                group.attrs.update(start=start,stop=start+len(batch))
                _encode(group,batch,constants)
            _dataset(file,'episode_index',episode_index); _dataset(file,'state_index',state_index)
            _dataset(file,'action_info_reference',action_ref)
            episode_rows=[{k:v for k,v in e.items() if k!='initial_observation'} for e in episodes]
            _encode(file.create_group('episode_ledger'),episode_rows,constants)
            windows=[]
            for index,e in enumerate(episodes):
                windows.extend(dict(episode_index=index,window=w) for w in e.get('windows',[]))
            if windows: _encode(file.create_group('window_ledger'),windows,constants)
            labels=file.create_group('labels'); _dataset(labels,'mc_reward',yr); _dataset(labels,'mc_grade',yg)
            constants.save(); file.attrs['complete']=True
            file.attrs['content_sha256']=_content_digest(file); file.flush()
        # link refuses an overwrite even if another writer won the race.
        os.link(temporary,path); temporary.unlink()
    except BaseException:
        if temporary.exists(): temporary.unlink()
        raise


class RowSequence(Sequence):
    """Independent rows decoded from a bounded number of typed HDF5 blocks."""
    def __init__(self,path,*,cache_blocks=2,verify=True):
        self.path=Path(path); self.cache_blocks=max(1,int(cache_blocks)); self._cache=OrderedDict()
        self._file=_h5py().File(self.path,'r')
        try:
            f=self._file
            if f.attrs.get('schema_version') != SCHEMA_VERSION or f.attrs.get('kind') != 'dataset' or not bool(f.attrs.get('complete',False)):
                raise ValueError('incomplete/unsupported HDF5 dataset')
            if verify and f.attrs.get('content_sha256') != _content_digest(f):
                raise ValueError('HDF5 content checksum mismatch')
            self.constants=_Constants(f)
            self._count=int(f.attrs['transitions']); self.chunk_rows=int(f.attrs['chunk_rows'])
            self.episode_index=f['episode_index'][...]; self.state_index=f['state_index'][...]
            self.action_ref=f['action_info_reference'][...]
            decoder=_Decoder(f['episode_ledger'],self.constants)
            self.episodes=[decoder.row(i) for i in range(decoder.count)]
            self._validate_layout()
            for i,e in enumerate(self.episodes):
                e['initial_observation']=f['states'][e['start']+i].tolist()
        except BaseException:
            self._file.close(); raise

    def _validate_layout(self):
        f=self._file
        if len(self.episodes)!=int(f.attrs['episodes']): raise ValueError('HDF5 episode count mismatch')
        _bounds(self,self.episodes)
        if f['states'].shape != (self._count+len(self.episodes),OBSERVATION_SIZE) or f['states'].dtype != np.dtype('float32'):
            raise ValueError('HDF5 state layout mismatch')
        if any(len(v)!=self._count for v in (self.episode_index,self.state_index,self.action_ref)):
            raise ValueError('HDF5 transition index count mismatch')
        expected_names=[f'{i:012d}' for i in range(0,self._count,self.chunk_rows)]
        if sorted(f['transitions'])!=expected_names: raise ValueError('HDF5 transition block inventory mismatch')
        for start in range(0,self._count,self.chunk_rows):
            group=f['transitions'][f'{start:012d}']; stop=min(self._count,start+self.chunk_rows)
            if (group.attrs['start'],group.attrs['stop'],group.attrs['count'])!=(start,stop,stop-start):
                raise ValueError('HDF5 transition block boundary mismatch')
        for ei,e in enumerate(self.episodes):
            a,b=e['start'],e['stop']
            if not np.all(self.episode_index[a:b]==ei) or not np.array_equal(self.state_index[a:b],np.arange(a,b)+ei):
                raise ValueError('HDF5 episode/state index mismatch')
            for index in range(a,b):
                if self.action_ref[index] not in (-2,-1 if index==a else index-1):
                    raise ValueError('HDF5 predecessor reference crosses episode')
        if f['labels/mc_reward'].shape!=(self._count,) or f['labels/mc_grade'].shape!=(self._count,4):
            raise ValueError('HDF5 label layout mismatch')

    def __len__(self): return self._count

    def _row(self,index,*,action_info=True):
        start=index//self.chunk_rows*self.chunk_rows
        if start not in self._cache:
            self._cache[start]=_Decoder(self._file['transitions'][f'{start:012d}'],self.constants)
            while len(self._cache)>self.cache_blocks: self._cache.popitem(last=False)
        self._cache.move_to_end(start)
        row=self._cache[start].row(index-start)
        state=int(self.state_index[index])
        row['observation']=self._file['states'][state].copy()
        row['next_observation']=self._file['states'][state+1].copy()
        ref=int(self.action_ref[index])
        if action_info and ref!=-2:
            row['action_info']=(copy.deepcopy(self.episodes[int(self.episode_index[index])]['start_info'])
                                if ref==-1 else self._row(ref,action_info=False)['settlement_info'])
        return row

    def __getitem__(self,index):
        if isinstance(index,slice): return [self[i] for i in range(*index.indices(len(self)))]
        if not isinstance(index,numbers.Integral): raise TypeError('row index must be an integer')
        index=int(index)
        if index<0:index+=len(self)
        if not 0<=index<len(self):raise IndexError(index)
        if not self._file.id.valid: raise ValueError('HDF5 row sequence is closed')
        return self._row(index)

    def __deepcopy__(self,memo):
        result=type(self)(self.path,cache_blocks=self.cache_blocks,verify=False)
        memo[id(self)]=result
        return result

    def close(self):
        self._cache.clear()
        if getattr(self,'_file',None) is not None:self._file.close()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
    def __del__(self):
        try:self.close()
        except Exception:pass


def read_dataset(path, *, lazy=True, cache_blocks=2):
    rows=RowSequence(path,cache_blocks=cache_blocks)
    try:
        episodes=copy.deepcopy(rows.episodes)
        yr=rows._file['labels/mc_reward'][...]; yg=rows._file['labels/mc_grade'][...]
        if not np.all(np.isfinite(yr)) or not np.all(np.isfinite(yg)):
            raise ValueError('non-finite HDF5 labels')
        if not lazy:
            materialized=list(rows); rows.close(); rows=materialized
        return rows,episodes,yr,yg
    except BaseException:
        rows.close(); raise


class HDF5RowJournal(Sequence):
    """Receiving-time typed spill file; only the active chunk stays as dicts.

    A journal is not accepted by ``read_dataset`` and cannot masquerade as a
    complete shift. Call ``write_dataset`` with this sequence after the caller
    has checked complete episode/MC semantics. Keep an interrupted journal as
    diagnostic evidence; it has complete=False and no trainable seal.
    """
    def __init__(self,path,*,chunk_rows=64):
        if type(chunk_rows) is not int or chunk_rows<=0:raise ValueError('invalid chunk_rows')
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True)
        self._file=_h5py().File(self.path,'x')
        self._file.attrs.update(schema_version=SCHEMA_VERSION,kind='receiving_journal',complete=False)
        self._file.create_group('blocks'); self.constants=_Constants(self._file,write=True)
        self.chunk_rows=chunk_rows; self._pending=[]; self._committed=0; self._cache=OrderedDict(); self._closed=False

    def append(self,row):
        if self._closed:raise ValueError('closed HDF5 receiving journal')
        self._pending.append(copy.deepcopy(row))
        if len(self._pending)>=self.chunk_rows:self.flush()

    def flush(self):
        if not self._pending:return
        group=self._file['blocks'].create_group(f'{self._committed:012d}')
        _encode(group,self._pending,self.constants)
        self._committed+=len(self._pending);self._pending=[]
        self.constants.save();self._file.attrs['transitions']=self._committed;self._file.flush()

    def __len__(self):return self._committed+len(self._pending)

    def __getitem__(self,index):
        if isinstance(index,slice):return [self[i] for i in range(*index.indices(len(self)))]
        if not isinstance(index,numbers.Integral):raise TypeError('row index must be an integer')
        index=int(index)
        if index<0:index+=len(self)
        if not 0<=index<len(self):raise IndexError(index)
        if self._closed:raise ValueError('closed HDF5 receiving journal')
        if index>=self._committed:return copy.deepcopy(self._pending[index-self._committed])
        # A caller can flush a short chunk, so find the actual block start.
        names=list(self._file['blocks'])
        start=max(int(name) for name in names if int(name)<=index)
        if start not in self._cache:
            self._cache[start]=_Decoder(self._file['blocks'][f'{start:012d}'],self.constants)
            while len(self._cache)>2:self._cache.popitem(last=False)
        self._cache.move_to_end(start)
        return self._cache[start].row(index-start)

    def seal(self,path,episodes,mc_reward,mc_grade):
        self.flush()
        write_dataset(path,self,episodes,mc_reward,mc_grade,chunk_rows=self.chunk_rows)
        self._file.attrs['sealed_dataset']=str(Path(path))
        self._file.flush()
        return read_dataset(path)[0]

    def close(self):
        if not getattr(self,'_closed',True):
            self.flush();self._file.close();self._closed=True;self._cache.clear()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
    def __del__(self):
        try:self.close()
        except Exception:pass
