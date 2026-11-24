"""Storage selection and immutable composition, independent of PPO semantics."""
from bisect import bisect_right
from collections.abc import Sequence
import copy


HDF5 = 'hdf5-v1'
LEGACY = 'npz-v1'
FORMATS = (HDF5, LEGACY)


def storage_options(value=None, *, default=LEGACY):
    if value is None:
        value = dict(format=default, chunk_rows=64)
    if not isinstance(value, dict) or set(value) != {'format', 'chunk_rows'}:
        raise ValueError('storage requires exactly format and chunk_rows')
    if value['format'] not in FORMATS:
        raise ValueError('unsupported dataset storage format')
    if type(value['chunk_rows']) is not int or not 1 <= value['chunk_rows'] <= 4096:
        raise ValueError('storage.chunk_rows must be an integer in [1, 4096]')
    return dict(value)


class CompositeRows(Sequence):
    """A complete batch of independent row sequences without expanding HDF5 rows."""
    def __init__(self, parts):
        self.parts = tuple(parts)
        self.stops = []
        count = 0
        for part in self.parts:
            count += len(part)
            self.stops.append(count)

    def __len__(self):
        return self.stops[-1] if self.stops else 0

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        part = bisect_right(self.stops, index)
        start = self.stops[part - 1] if part else 0
        return self.parts[part][index - start]

    def __iter__(self):
        for part in self.parts:
            yield from part

    def __deepcopy__(self, memo):
        result = type(self)([copy.deepcopy(part, memo) for part in self.parts])
        memo[id(self)] = result
        return result

    def close(self):
        for part in self.parts:
            if hasattr(part, 'close'):
                part.close()
