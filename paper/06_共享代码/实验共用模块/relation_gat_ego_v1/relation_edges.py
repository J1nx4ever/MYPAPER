"""Pure 1519D -> 27-type directed multigraph, with no learned-state cache.

The caller validates the complete shared observation contract; this boundary
also validates every field used to construct the graph. All canonical indices
are physical rows (31 per sample), never scene rows (32 per sample).
"""
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
import hashlib
import numpy as np
import torch

from .contract import require

RELATION_NAMES = (
    'loadTarget', 'dumpTarget', 'segment', 'route', 'initialPredecessor',
    'queue', 'seize', 'capacityWait', 'service', 'admission', 'road_next',
    'truck_node_at_road_from', 'truck_node_at_road_to',
)
EDGE_TYPE_NAMES = tuple(name + suffix for name in RELATION_NAMES
                        for suffix in ('', '_reverse')) + ('self',)
MAX_EDGES = 1811
TYPE_LIMITS = (9, 9, 9, 144, 9, 54, 54, 18, 54, 2, 240, 144, 144)
INCOMING_LIMITS = (9, 1, 9, 1, 9, 1, 9, 16, 9, 1, 54, 9, 54, 9,
                   18, 9, 54, 9, 2, 1, 15, 15, 9, 16, 9, 16, 1)


def _tensor_signature(t):
    return (id(t), t.data_ptr(), t._version, tuple(t.shape), t.dtype, t.device)


def _edge_signature(edges):
    tensors = (edges.edge_index, edges.edge_type, edges.edge_attr, edges.count, edges.degree_features)
    if edges.canonical_ids is not None:
        tensors += (edges.canonical_ids,)
    return tuple(_tensor_signature(t) for t in tensors)


def _role(node):
    return 'truck' if node < 9 else 'load' if node < 13 else 'dump' if node < 15 else 'road'


_ROLES = (('truck', 'load'), ('truck', 'dump'), ('truck', 'road'),
          ('truck', 'road'), ('truck', 'truck'), ('site', 'truck'),
          ('site', 'truck'), ('dump', 'truck'), ('site', 'truck'),
          ('dump', 'truck'), ('road', 'road'), ('truck', 'road'), ('truck', 'road'))


@dataclass(frozen=True)
class GraphEdges:
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    edge_attr: torch.Tensor
    count: torch.Tensor
    degree_features: torch.Tensor
    # Optional storage-row -> canonical-row map, for a synchronously permuted
    # graph-interface fixture. Normal raw-state construction always uses None.
    canonical_ids: torch.Tensor | None = None
    _validated_signature: tuple | None = field(default=None, init=False, repr=False, compare=False)

    @property
    def batch_size(self):
        return self.count.shape[0]

    @property
    def counts(self):
        """Raw int64 counts; normalized model input is degree_features."""
        return self.count

    def validate(self):
        tensors = (self.edge_index, self.edge_type, self.edge_attr, self.count, self.degree_features)
        if self.canonical_ids is not None:
            tensors += (self.canonical_ids,)
        require(all(isinstance(t, torch.Tensor) for t in tensors), 'edge fields must be tensors')
        signature = tuple(_tensor_signature(t) for t in tensors)
        if signature == self._validated_signature:
            return self
        require(all(t.device.type == 'cpu' and not t.requires_grad for t in tensors),
                'edge metadata must be CPU tensors without gradients')
        ix, typ, attr, count, degree = tensors[:5]
        require(ix.dtype == torch.int64 and ix.ndim == 2 and ix.shape[0] == 2, 'edge_index must be int64[2,E]')
        e = ix.shape[1]
        require(typ.dtype == torch.int64 and typ.shape == (e,), 'edge_type must be int64[E]')
        require(attr.dtype == torch.float32 and attr.shape == (e, 2), 'edge_attr must be float32[E,2]')
        require(count.dtype == torch.int64 and count.ndim == 3 and count.shape[1:] == (31, 27)
                and count.shape[0] > 0, 'count must be int64[B,31,27]')
        b = count.shape[0]
        require(degree.dtype == torch.float32 and degree.shape == count.shape, 'degree_features shape/dtype differs')
        require(bool(torch.isfinite(attr).all() and torch.isfinite(degree).all()), 'nonfinite edge metadata')
        require(bool(((ix >= 0) & (ix < 31*b)).all()), 'edge endpoint outside physical graph')
        require(bool(((typ >= 0) & (typ < 27)).all()), 'edge_type outside 0..26')
        require(bool((ix[0]//31 == ix[1]//31).all()), 'cross-sample graph edge')
        if self.canonical_ids is None:
            canonical = torch.arange(31).repeat(b, 1)
        else:
            canonical = self.canonical_ids
            require(canonical.dtype == torch.int64 and canonical.shape == (b, 31)
                    and torch.equal(canonical.sort(dim=1).values, torch.arange(31).repeat(b, 1)),
                    'canonical_ids must permute all 31 physical rows within each sample')
        observed = torch.bincount(ix[1]*27 + typ, minlength=b*31*27).reshape(b, 31, 27)
        require(torch.equal(observed, count), 'incoming counts disagree with edge records')
        require(torch.equal(degree, count.to(torch.float32)/64), 'degree_features must equal count/64 exactly')
        require(bool((count >= 0).all()) and bool((count <= torch.tensor(INCOMING_LIMITS)).all()),
                'per-type incoming count exceeds contract')
        records = [[] for _ in range(b)]
        for (src, dst), t, (pos, valid) in zip(ix.T.tolist(), typ.tolist(), attr.tolist()):
            sample = src//31
            cs, cd = canonical[sample, src%31].item(), canonical[sample, dst%31].item()
            if t == 26:
                require(src == dst and pos == 0 and valid == 0, 'malformed SELF record')
            else:
                sr, dr = _ROLES[t//2]
                if t % 2:
                    sr, dr = dr, sr
                require((_role(cs) == sr or sr == 'site' and 9 <= cs < 15)
                        and (_role(cd) == dr or dr == 'site' and 9 <= cd < 15),
                        'relation endpoint canonical type differs')
                if t in (20, 21):
                    require(src != dst, 'road_next excludes identical Road')
                denominator = 16 if t in (6, 7) else 9 if 10 <= t <= 15 else None
                if denominator is None:
                    require(pos == 0 and valid == 0, 'unordered edge attribute must be zero')
                else:
                    scaled = pos * denominator
                    require(valid == 1 and 1 <= round(scaled) <= denominator
                            and abs(scaled-round(scaled)) <= 2e-6, 'invalid ordered edge attribute')
            records[sample].append((src%31, dst%31, t, pos, valid))
        require(bool((count[:, :, 26] == 1).all()), 'each physical node needs exactly one SELF')
        for rows in records:
            require(31 <= len(rows) <= MAX_EDGES, 'edge count outside [31,1811]')
            counter = Counter(rows)
            for k, limit in enumerate(TYPE_LIMITS):
                require(sum(n for (_, _, t, _, _), n in counter.items() if t == 2*k) <= limit,
                        'relation record count exceeds contract')
            for (src, dst, typ, pos, valid), n in counter.items():
                if typ < 26:
                    require(counter[(dst, src, typ ^ 1, pos, valid)] == n,
                            'reverse edge multiset or copied attributes differ')
        object.__setattr__(self, '_validated_signature', signature)
        return self

    def records(self, sample=0):
        """Explicit diagnostic metadata for rendering, never emitted by default."""
        self.validate()
        require(type(sample) is int and 0 <= sample < self.batch_size, 'invalid graph sample')
        rows = []
        for (src, dst), typ, attr in zip(self.edge_index.T.tolist(), self.edge_type.tolist(), self.edge_attr.tolist()):
            if src//31 == sample:
                rows.append(dict(source=src%31, destination=dst%31, type=typ,
                                 relation=EDGE_TYPE_NAMES[typ], position_norm=attr[0], position_valid=attr[1]))
        return rows


def _as_numpy(states):
    if isinstance(states, torch.Tensor):
        require(states.dtype == torch.float32 and states.device.type == 'cpu', 'states must be CPU float32')
        states = states.detach().numpy()
    require(isinstance(states, np.ndarray) and states.dtype == np.float32 and states.ndim == 2
            and states.shape[1] == 1519 and len(states) > 0, 'states must be float32[B,1519]')
    require(bool(np.isfinite(states).all()), 'nonfinite raw states')
    return states


def _onehot(values, label):
    require(bool(np.isin(values, (0, 1)).all()) and bool((values.sum(axis=-1) <= 1).all()),
            'invalid one-hot ' + label)
    return np.where(values.sum(axis=-1) > 0, values.argmax(axis=-1), -1)


def _grid(values, denominator, label):
    scaled = values.astype(np.float64)*denominator
    rounded = np.rint(scaled)
    require(bool(((values >= 0) & (values <= 1)).all())
            and bool((np.abs(scaled-rounded) <= 2e-6).all()), 'invalid scaled identity/rank ' + label)
    return rounded.astype(np.int64)


def build_edges(states):
    """Build an independent numeric graph; no model values or RNG are used."""
    raw = _as_numpy(states)
    truck = raw[:, 27:1107].reshape(-1, 9, 120)
    load = raw[:, 1107:1295].reshape(-1, 4, 47)
    dump = raw[:, 1295:1391].reshape(-1, 2, 48)
    road = raw[:, 1439:1519].reshape(-1, 16, 5)
    load_target = _onehot(truck[:, :, 26:30], 'loadTarget')
    dump_target = _onehot(truck[:, :, 30:32], 'dumpTarget')
    segment = _onehot(truck[:, :, 45:61], 'segment')
    truck_node = _onehot(truck[:, :, 32:41], 'truck.node')
    ranks = _grid(truck[:, :, 63:79], 16, 'route')
    predecessor = _grid(truck[:, :, 81], 9, 'initialPredecessor') - 1
    endpoints = _grid(road[:, :, :2], 9, 'road endpoints') - 1
    admission = _grid(dump[:, :, 11], 9, 'admission') - 1
    lists = (
        (load, 20, 5, 9, True), (dump, 12, 5, 13, True),
        (load, 29, 6, 9, True), (dump, 21, 6, 13, True),
        (dump, 30, 7, 13, True), (load, 38, 8, 9, False), (dump, 39, 8, 13, False),
    )
    decoded_lists = [(start_row, family, ordered, _grid(block[:, :, start:start+9], 9, 'site list')-1)
                     for block, start, family, start_row, ordered in lists]
    rows = []
    for b in range(len(raw)):
        offset = 31*b
        def add(src, dst, family, position=0., valid=0.):
            rows.append((offset+src, offset+dst, 2*family, position, valid))
            rows.append((offset+dst, offset+src, 2*family+1, position, valid))
        for t in range(9):
            for target, start, family in ((load_target, 9, 0), (dump_target, 13, 1), (segment, 15, 2)):
                if target[b, t] >= 0:
                    add(t, start+int(target[b, t]), family)
            route = ranks[b, t]
            present = route[route > 0]
            require(np.array_equal(np.sort(present), np.arange(1, len(present)+1)),
                    'route ranks must be unique and contiguous 1..K')
            for r in np.flatnonzero(route):
                add(t, 15+int(r), 3, float(route[r])/16, 1.)
            if predecessor[b, t] >= 0:
                add(t, int(predecessor[b, t]), 4)
        for start_row, family, ordered, ids in decoded_lists:
            for site in range(ids.shape[1]):
                for slot, t in enumerate(ids[b, site]):
                    if t >= 0:
                        add(start_row+site, int(t), family, (slot+1)/9 if ordered else 0., float(ordered))
        for d in range(2):
            if admission[b, d] >= 0:
                add(13+d, int(admission[b, d]), 9)
        for r in range(16):
            for s in range(16):
                if r != s and endpoints[b, r, 1] >= 0 and endpoints[b, r, 1] == endpoints[b, s, 0]:
                    add(15+r, 15+s, 10)
            for t in range(9):
                if truck_node[b, t] >= 0:
                    for endpoint, family in ((0, 11), (1, 12)):
                        if truck_node[b, t] == endpoints[b, r, endpoint]:
                            add(t, 15+r, family)
        rows.extend((offset+i, offset+i, 26, 0., 0.) for i in range(31))
    packed = np.asarray(rows, dtype=np.float64)
    ix = torch.from_numpy(packed[:, :2].astype(np.int64).T.copy())
    typ = torch.from_numpy(packed[:, 2].astype(np.int64))
    attr = torch.from_numpy(packed[:, 3:].astype(np.float32))
    count = torch.bincount(ix[1]*27+typ, minlength=len(raw)*31*27).reshape(len(raw), 31, 27)
    return GraphEdges(ix, typ, attr, count, count.to(torch.float32)/64).validate()


def batch_edges(samples):
    """Compose validated single-sample graphs using the exact 31-row offset.

    No raw parsing or learned representation is cached. A composition of
    separately validated samples preserves all per-sample invariants, so its
    validation certificate does not need another Python edge-record walk.
    """
    require(bool(samples) and all(isinstance(e, GraphEdges) and e.batch_size == 1 for e in samples),
            'batch_edges requires nonempty single-sample graphs')
    for e in samples:
        e.validate()
    if len(samples) == 1:
        return samples[0]
    canonical = None
    if any(e.canonical_ids is not None for e in samples):
        canonical = torch.cat([torch.arange(31)[None] if e.canonical_ids is None else e.canonical_ids
                               for e in samples], dim=0)
    result = GraphEdges(torch.cat([e.edge_index+31*b for b,e in enumerate(samples)],dim=1),
                        torch.cat([e.edge_type for e in samples]),torch.cat([e.edge_attr for e in samples]),
                        torch.cat([e.count for e in samples]),torch.cat([e.degree_features for e in samples]),canonical)
    object.__setattr__(result, '_validated_signature', _edge_signature(result))
    return result


class GraphEdgeCache:
    """Bounded per-row numeric LRU; minibatch reordering reuses raw graphs.

    Keys are complete current float32 snapshot bytes; values contain only
    topology/type/rank/count. Learned H/Z, RNG, module weights and autograd
    graphs are never cached. Public edge tensors are read-only by contract;
    normal tensor mutation invalidates the entry and forces raw reconstruction.
    """
    def __init__(self, max_entries=4096, *, max_bytes=64*1024*1024):
        require(type(max_entries) is int and max_entries > 0, 'invalid edge cache size')
        require(type(max_bytes) is int and max_bytes > 0, 'invalid edge cache byte limit')
        self.max_entries, self.max_bytes = max_entries, max_bytes
        self._entries = OrderedDict()
        self.bytes = self.builds = self.hits = 0

    def _row(self, raw):
        key = hashlib.sha256(raw.tobytes(order='C')).digest()
        if key in self._entries:
            edges, signature, size = self._entries[key]
            if signature == _edge_signature(edges):
                self.hits += 1
                self._entries.move_to_end(key)
                return edges
            self._entries.pop(key)
            self.bytes -= size
        edges = build_edges(raw[None])
        self.builds += 1
        size = sum(t.numel()*t.element_size() for t in
                   (edges.edge_index,edges.edge_type,edges.edge_attr,edges.count,edges.degree_features))
        if size <= self.max_bytes:
            self._entries[key] = (edges,_edge_signature(edges),size)
            self.bytes += size
            while len(self._entries)>self.max_entries or self.bytes>self.max_bytes:
                _, (_, _, old_size) = self._entries.popitem(last=False)
                self.bytes -= old_size
        return edges

    def get(self, states):
        raw = _as_numpy(states)
        return batch_edges([self._row(row) for row in raw])
