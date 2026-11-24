"""Causal, padded material prefixes for a future GRU; no fitting or inference."""
from dataclasses import dataclass
import numpy as np

from .events import MemoryTape, ContractError, build_memory

FEATURE_NAMES = tuple(['kind.'+k for k in ('UNLOAD','PROCESS')] +
    ['window.'+str(i) for i in range(4)] +
    ['source.'+str(i) for i in range(4)]+['source.unknown'] +
    ['truck.'+str(i) for i in range(9)]+['truck.unknown'] +
    ['tons_500','metal_500','material_grade','visible_first_28800','visible_last_28800',
     'physical_start_28800','physical_end_28800','q_before_5000','m_before_5000',
     'q_after_5000','m_after_5000'])
FEATURE_VERSION = 'material-prefix-features-32-v1'


def _require(condition, message):
    if not condition: raise ContractError(message)


def _feature(token):
    _require(token.kind in ('UNLOAD','PROCESS'), 'only material events enter GRU features')
    _require(-1 <= token.source_id < 4 and -1 <= token.truck_id < 9,
             'source/truck category outside fixed 4-site/9-truck schema')
    result = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    result[0 if token.kind=='UNLOAD' else 1] = 1
    result[2+token.window] = 1
    result[6+(4 if token.source_id==-1 else token.source_id)] = 1
    result[11+(9 if token.truck_id==-1 else token.truck_id)] = 1
    result[21:] = [token.tons/500,token.metal/500,token.grade,
        token.time_first/28800,token.time/28800,token.interval_start/28800,token.interval_end/28800,
        token.q_before/5000,token.m_before/5000,token.q_after/5000,token.m_after/5000]
    _require(np.isfinite(result).all(), 'nonfinite event features')
    return result


@dataclass(frozen=True)
class PrefixBatch:
    features: np.ndarray
    valid: np.ndarray
    lengths: np.ndarray
    tape_cursors: tuple
    request_ids: tuple
    feature_version: str = FEATURE_VERSION


def pack_query_prefixes(memory, request_ids, *, next_state=False, max_tokens=4096,
                        max_batch_bytes=256*1024*1024):
    """Each query reads only its own prefix; reset is at the real shift boundary.

    WINDOW_CLOSE is a structural barrier in the tape, not a GRU input token.
    Future networks must skip padding (including zero-length prefixes) rather
    than advancing a GRU with zero vectors. Batch/batch_id/seq never become
    numeric features. No hidden-state cache is retained across parameter steps.
    """
    _require(type(memory) is MemoryTape, 'validated MemoryTape required')
    _require(build_memory(memory.events)==memory, 'material prefix/query certificate differs')
    _require(type(max_tokens) is int and max_tokens>0 and type(max_batch_bytes) is int and max_batch_bytes>0,
             'explicit positive sequence resource limits required')
    ids = tuple(request_ids)
    _require(bool(ids) and len(ids)==len(set(ids)), 'unique nonempty query ID list required')
    lookup = {q.request_id:q for q in memory.queries}
    _require(all(key in lookup for key in ids), 'unknown or terminal-only request ID')
    rows, cursors = [], []
    for request in ids:
        query = lookup[request]
        cursor = query.next_cursor if next_state else query.current_cursor
        seq = query.next_seq if next_state else query.current_seq
        prefix = memory.tokens[:cursor]
        _require(all(token.seq_last < seq for token in prefix), 'prefix includes query or future event')
        material=[t for t in prefix if t.kind in ('UNLOAD','PROCESS')]
        _require(len(material)<=max_tokens,'material prefix exceeds resource limit; do not truncate history silently')
        rows.append(material)
        cursors.append(cursor)
    lengths = np.asarray([len(row) for row in rows],dtype=np.int64)
    _require(len(rows)*int(lengths.max())*len(FEATURE_NAMES)*4<=max_batch_bytes,
             'padded prefix batch exceeds configured memory limit')
    padded = np.zeros((len(rows),int(lengths.max()),len(FEATURE_NAMES)),dtype=np.float32)
    valid = np.zeros(padded.shape[:2],dtype=bool)
    for i,row in enumerate(rows):
        if row: padded[i,:len(row)] = [_feature(t) for t in row];valid[i,:len(row)] = True
    # Byte-backed arrays cannot be changed or have writeability re-enabled.
    def immutable(value): return np.frombuffer(value.tobytes(),dtype=value.dtype).reshape(value.shape)
    return PrefixBatch(immutable(padded),immutable(valid),immutable(lengths),tuple(cursors),ids)
