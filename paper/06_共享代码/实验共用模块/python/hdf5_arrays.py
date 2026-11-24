"""Lossless typed numeric array bundles for versioned PPO target evidence."""
import json
import os
from pathlib import Path
import uuid

import numpy as np

from .hdf5_storage import _content_digest, _h5py


SCHEMA_VERSION = 'mining-hdf5-array-bundle-v1'
MAX_METADATA_BYTES = 65536


def _metadata(value):
    if type(value) is not dict:
        raise ValueError('array bundle metadata must be a JSON object')
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                     separators=(',', ':'), allow_nan=False).encode('utf-8')
    if len(raw) > MAX_METADATA_BYTES:
        raise ValueError('array bundle metadata exceeds 64 KiB')
    return raw


def _array(value):
    array = np.asarray(value)
    if array.dtype.kind not in 'biuf' or array.dtype.hasobject:
        raise ValueError('array bundles support only bool, integer and floating arrays')
    if array.dtype.kind == 'f' and not np.all(np.isfinite(array)):
        raise ValueError('array bundle contains non-finite values')
    return array


def write_array_bundle(path, arrays, metadata):
    """Atomically seal numeric arrays without casting; refuse any overwrite."""
    if type(arrays) is not dict or not arrays or not all(type(k) is str and k for k in arrays):
        raise ValueError('array bundle requires named arrays')
    prepared = {key: _array(value) for key, value in arrays.items()}
    raw_metadata = _metadata(metadata)
    path = Path(path)
    if path.exists():
        raise FileExistsError('refusing array bundle overwrite: '+str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.partial-'+uuid.uuid4().hex)
    try:
        with _h5py().File(temporary, 'x') as file:
            file.attrs.update(schema_version=SCHEMA_VERSION, kind='numeric_array_bundle',
                              complete=False, array_count=len(prepared))
            file.create_dataset('metadata_json', data=np.frombuffer(raw_metadata, dtype=np.uint8),
                                compression='gzip', fletcher32=True)
            group = file.create_group('arrays')
            for index, (key, array) in enumerate(prepared.items()):
                options = dict(compression='gzip', compression_opts=4,
                               shuffle=True, fletcher32=True) if array.ndim and array.size else {}
                dataset = group.create_dataset(f'a{index:04d}', data=array, dtype=array.dtype, **options)
                dataset.attrs['array_name'] = key
            file.attrs['complete'] = True
            file.attrs['content_sha256'] = _content_digest(file)
            file.flush()
        os.link(temporary, path)
        temporary.unlink()
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def read_array_bundle(path):
    """Verify a complete bundle before returning arrays and their provenance."""
    with _h5py().File(path, 'r') as file:
        if (file.attrs.get('schema_version') != SCHEMA_VERSION or
                file.attrs.get('kind') != 'numeric_array_bundle' or
                file.attrs.get('complete') != np.bool_(True)):
            raise ValueError('incomplete or unsupported array bundle')
        if file.attrs.get('content_sha256') != _content_digest(file):
            raise ValueError('array bundle content checksum mismatch')
        if set(file) != {'arrays', 'metadata_json'}:
            raise ValueError('array bundle dataset inventory mismatch')
        raw = file['metadata_json'][...]
        if raw.dtype != np.dtype('uint8') or raw.ndim != 1 or len(raw) > MAX_METADATA_BYTES:
            raise ValueError('invalid array bundle metadata layout')
        metadata = json.loads(bytes(raw).decode('utf-8'))
        _metadata(metadata)
        arrays = {}
        for dataset in file['arrays'].values():
            key = dataset.attrs.get('array_name')
            if not isinstance(key, str) or not key or key in arrays:
                raise ValueError('missing or duplicate array name')
            arrays[key] = _array(dataset[...])
        if not arrays or len(arrays) != file.attrs.get('array_count'):
            raise ValueError('array bundle array count mismatch')
    return arrays, metadata
