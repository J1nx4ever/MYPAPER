"""Current-snapshot factual graph: 19 nodes, directed roads, four relation groups.

No candidate edge is inserted into factual message passing. Candidate mask and
tons remain readout inputs; changing them cannot create commitments. Missing
references stay missing. The physical map is pinned to the observed production
road order, not to the older synthetic fixture map.

All returned tensors are CPU tensors. Only discrete references are detached;
continuous edge attributes and quantity sums retain input autograd. Edge quantities use
the raw schema scales (distance /10000 m, speed /10 m/s, tons /193 t, truck
counts /9, ordered slots /9). Node quantities use QUANTITY_SCALES below; they
are sums/counts, never attention-weighted averages. No clipping is applied.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch

from relation_gat_ego_v1.adapter import as_numpy, validate_states

NODE_NAMES = tuple([f'Truck{i}' for i in range(9)] + [f'Load{i}' for i in range(4)]
                   + [f'Dump{i}' for i in range(2)] + ['Junction2', 'Junction3', 'Junction4', 'Global'])
NUM_NODES = len(NODE_NAMES)
PHYSICAL_NODE_TO_TOKEN = (13, 14, 15, 16, 17, 9, 10, 11, 12)
ROAD_ENDPOINTS = ((0, 2), (5, 3), (6, 4), (7, 4), (8, 3), (1, 3), (2, 0), (2, 3),
                  (3, 5), (3, 8), (3, 1), (3, 2), (3, 4), (4, 6), (4, 7), (4, 3))
GROUP_NAMES = ('transport', 'commitment', 'service', 'material')
_PAIRS = (
    ('truck_at_node', 0), ('truck_segment_from', 0), ('truck_segment_to', 0),
    ('truck_predecessor', 0), ('reserved_load', 1), ('loaded_dump', 1),
    ('queue', 2), ('seize', 2), ('capacity_wait', 2), ('service', 2), ('admission', 2),
    ('compatible_material', 3),
)
RELATION_NAMES = ('road',) + tuple(name + suffix for name, _ in _PAIRS
                                   for suffix in ('', '_reverse'))
RELATION_GROUP_IDS = (0,) + tuple(group for _, group in _PAIRS for _ in range(2))
EDGE_FEATURE_NAMES = (
    'road_length_10000m', 'road_limit_10mps', 'road_closed', 'road_occupancy_9', 'road_valid',
    'segment_remaining_10000m', 'segment_remaining_valid', 'tons_193', 'tons_valid',
    'ore', 'payload_grade', 'payload_grade_known', 'position_9', 'position_valid',
)
EDGE_ATTRIBUTE_NAMES = EDGE_FEATURE_NAMES
QUANTITY_NAMES = (
    'reserved_truck_count', 'reserved_tons', 'inbound_loaded_truck_count', 'inbound_loaded_tons',
    'inbound_known_grade_truck_count', 'inbound_known_grade_tons', 'inbound_known_metal_tons',
    'queue_unique_count', 'queue_instance_count', 'seize_unique_count', 'seize_instance_count',
    'capacity_wait_unique_count', 'capacity_wait_instance_count', 'service_unique_count',
    'service_instance_count', 'admission_count', 'at_node_truck_count', 'incoming_road_count',
    'outgoing_road_count', 'incoming_segment_truck_count', 'outgoing_segment_truck_count',
)
QUANTITY_SCALES = tuple(1737.0 if name.endswith('_tons') else
                        16.0 if name in ('incoming_road_count', 'outgoing_road_count') else 9.0
                        for name in QUANTITY_NAMES)
_QI = {name: i for i, name in enumerate(QUANTITY_NAMES)}
_RI = {name: i for i, name in enumerate(RELATION_NAMES)}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _ids(one_hot):
    return np.where(one_hot.sum(-1) > 0, one_hot.argmax(-1), -1)


def _wire(values):
    return np.rint(values.astype(np.float64) * 9).astype(np.int64) - 1


@dataclass(frozen=True)
class GraphBatch:
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    edge_group: torch.Tensor
    edge_attr: torch.Tensor
    quantities: torch.Tensor
    candidate_dump_ids: torch.Tensor
    ego_ids: torch.Tensor

    @property
    def batch_size(self):
        return self.quantities.shape[0]

    @property
    def candidate_dump_indices(self):
        return self.candidate_dump_ids - 13

    @property
    def raw_quantities(self):
        """Human-readable counts/tons; returned tensor is a fresh scaled view."""
        return self.quantities * self.quantities.new_tensor(QUANTITY_SCALES)

    def validate(self):
        fields = (self.edge_index, self.edge_type, self.edge_group, self.edge_attr,
                  self.quantities, self.candidate_dump_ids, self.ego_ids)
        _require(all(isinstance(x, torch.Tensor) and x.device.type == 'cpu'
                     for x in fields), 'graph tensors require CPU tensors')
        ix, typ, group, attr, quantities, candidates, ego = fields
        _require(ix.dtype == torch.int64 and ix.ndim == 2 and ix.shape[0] == 2, 'invalid edge_index')
        b, e = self.batch_size, ix.shape[1]
        _require(b > 0 and quantities.dtype == torch.float32
                 and quantities.shape == (b, NUM_NODES, len(QUANTITY_NAMES)), 'invalid quantities')
        _require(typ.dtype == group.dtype == torch.int64 and typ.shape == group.shape == (e,), 'invalid edge labels')
        _require(attr.dtype == torch.float32 and attr.shape == (e, len(EDGE_FEATURE_NAMES)), 'invalid edge_attr')
        _require(bool(torch.isfinite(attr).all() and torch.isfinite(quantities).all()), 'nonfinite graph metadata')
        _require(bool((quantities >= 0).all()), 'negative graph quantity')
        _require(bool(((ix >= 0) & (ix < b * NUM_NODES)).all()), 'edge endpoint outside graph')
        _require(bool((ix[0] // NUM_NODES == ix[1] // NUM_NODES).all()), 'cross-sample graph edge')
        _require(bool(((typ >= 0) & (typ < len(RELATION_NAMES))).all()), 'invalid relation type')
        _require(torch.equal(group, torch.tensor(RELATION_GROUP_IDS)[typ]), 'relation/group mismatch')
        _require(candidates.dtype == torch.int64 and candidates.shape == (b, 4)
                 and bool(((candidates == 13) | (candidates == 14)).all()), 'invalid candidate dump ids')
        _require(ego.dtype == torch.int64 and ego.shape == (b,)
                 and bool(((ego >= -1) & (ego < 9)).all()), 'invalid ego ids')
        _require(not bool((ix % NUM_NODES == 18).any()), 'Global has no factual physical edge')
        return self

    def records(self, sample=0):
        """Explicit diagnostic records, never emitted automatically."""
        _require(type(sample) is int and 0 <= sample < self.batch_size, 'invalid sample')
        rows = []
        for ends, typ, group, attr in zip(self.edge_index.T.tolist(), self.edge_type.tolist(),
                                         self.edge_group.tolist(), self.edge_attr.tolist()):
            if ends[0] // NUM_NODES == sample:
                rows.append(dict(source=ends[0] % NUM_NODES, destination=ends[1] % NUM_NODES,
                                 relation=RELATION_NAMES[typ], group=GROUP_NAMES[group],
                                 attributes=dict(zip(EDGE_FEATURE_NAMES, attr))))
        return rows


def build_graph(states):
    """Construct only observed facts; candidate Load/Dump binding is readout-only.

    Reservation statistics count each non-ego, not-yet-loaded truck with positive reservation
    and an observed load target once, independently of membership in queue,
    seize and service lists. Loaded inbound statistics count each loaded truck
    with an observed dump target once. Unknown dump targets are not imputed.
    Known-grade mass is separated from total mass. Service lists are multisets;
    ordered queue/seize/capacity-wait slots retain their original positions.
    """
    validate_states(states)
    raw = as_numpy(states)
    live = states if isinstance(states, torch.Tensor) else torch.from_numpy(states)
    bsize = len(raw)
    truck = raw[:, 27:1107].reshape(bsize, 9, 120)
    load = raw[:, 1107:1295].reshape(bsize, 4, 47)
    dump = raw[:, 1295:1391].reshape(bsize, 2, 48)
    road = raw[:, 1439:1519].reshape(bsize, 16, 5)
    live_truck = live[:, 27:1107].reshape(bsize, 9, 120)
    live_road = live[:, 1439:1519].reshape(bsize, 16, 5)
    endpoints = _wire(road[:, :, :2])
    _require(bool((endpoints == np.asarray(ROAD_ENDPOINTS)).all()), 'road endpoints differ from frozen production map')
    _require(bool((road[:, :, 2:4] >= 0).all()), 'negative road length/speed')
    load_target, dump_target = _ids(truck[:, :, 26:30]), _ids(truck[:, :, 30:32])
    segment, node = _ids(truck[:, :, 45:61]), _ids(truck[:, :, 32:41])
    predecessor = _wire(truck[:, :, 81])
    ego = _ids(truck[:, :, 0])
    candidate_dump = np.where(load[:, :, 1] == 1, 13, 14).astype(np.int64)
    _require(bool((dump[:, 0, 1] == 1).all() and (dump[:, 1, 1] == 0).all()),
             'dump ore/waste roles differ from frozen material mapping')
    quantity_terms = {}
    rows = []

    for b in range(bsize):
        offset = b * NUM_NODES

        def add(src, dst, relation, attr=None, reverse=True):
            attr = live.new_zeros(len(EDGE_FEATURE_NAMES)) if attr is None else attr
            typ = _RI[relation]
            rows.append((offset + src, offset + dst, typ, attr.clone()))
            if reverse:
                rows.append((offset + dst, offset + src, _RI[relation + '_reverse'], attr.clone()))

        def inc(n, name, value=1):
            quantity_terms.setdefault((b, n, _QI[name]), []).append(value)

        occupancy = np.bincount(segment[b, segment[b] >= 0], minlength=16)
        for rid, (a, z) in enumerate(ROAD_ENDPOINTS):
            src, dst = PHYSICAL_NODE_TO_TOKEN[a], PHYSICAL_NODE_TO_TOKEN[z]
            attr = live.new_zeros(len(EDGE_FEATURE_NAMES))
            attr[:3] = live_road[b, rid, 2:5]
            attr[3:5] = live.new_tensor([occupancy[rid] / 9, 1])
            add(src, dst, 'road', attr, reverse=False)  # physical reverse roads already have their own rows
            inc(src, 'outgoing_road_count')
            inc(dst, 'incoming_road_count')
            inc(src, 'outgoing_segment_truck_count', occupancy[rid])
            inc(dst, 'incoming_segment_truck_count', occupancy[rid])

        for tid, t in enumerate(truck[b]):
            lt = live_truck[b, tid]
            # A segment identifies an in-transit vehicle. A stale node reference
            # must not create a false "arrived at station" relation.
            if segment[b, tid] >= 0:
                rid = int(segment[b, tid])
                _require(t[61] >= 0, 'negative segment remaining distance')
                attr = live.new_zeros(len(EDGE_FEATURE_NAMES))
                attr[:3] = live_road[b, rid, 2:5]
                attr[3:5] = live.new_tensor([occupancy[rid] / 9, 1])
                attr[5], attr[6] = lt[61], 1
                a, z = ROAD_ENDPOINTS[rid]
                add(tid, PHYSICAL_NODE_TO_TOKEN[a], 'truck_segment_from', attr)
                add(tid, PHYSICAL_NODE_TO_TOKEN[z], 'truck_segment_to', attr)
            elif node[b, tid] >= 0:
                n = PHYSICAL_NODE_TO_TOKEN[int(node[b, tid])]
                add(tid, n, 'truck_at_node')
                inc(n, 'at_node_truck_count')
            if predecessor[b, tid] >= 0:
                add(tid, int(predecessor[b, tid]), 'truck_predecessor')

            if tid != ego[b] and t[17] == 0 and t[20] > 0 and load_target[b, tid] >= 0:
                n = 9 + int(load_target[b, tid])
                _require(t[18] == load[b, n - 9, 1],
                         'reserved material differs from observed load target')
                attr = live.new_zeros(len(EDGE_FEATURE_NAMES))
                attr[7], attr[8], attr[9] = lt[20], 1, lt[18]
                add(tid, n, 'reserved_load', attr)
                inc(n, 'reserved_truck_count')
                inc(n, 'reserved_tons', lt[20] * 193)
            if t[17] == 1 and dump_target[b, tid] >= 0:
                n = 13 + int(dump_target[b, tid])
                _require(int(dump_target[b, tid]) == (0 if t[18] == 1 else 1),
                         'loaded material differs from observed dump target')
                attr = live.new_zeros(len(EDGE_FEATURE_NAMES))
                attr[7], attr[8], attr[9] = lt[19], 1, lt[18]
                if t[21] == 1:
                    attr[10], attr[11] = lt[22], 1
                add(tid, n, 'loaded_dump', attr)
                inc(n, 'inbound_loaded_truck_count')
                inc(n, 'inbound_loaded_tons', lt[19] * 193)
                if t[21] == 1:
                    inc(n, 'inbound_known_grade_truck_count')
                    inc(n, 'inbound_known_grade_tons', lt[19] * 193)
                    inc(n, 'inbound_known_metal_tons', lt[19] * 193 * lt[22])

        lists = ((load[b], 9, 20, 'queue', True), (dump[b], 13, 12, 'queue', True),
                 (load[b], 9, 29, 'seize', True), (dump[b], 13, 21, 'seize', True),
                 (dump[b], 13, 30, 'capacity_wait', True),
                 (load[b], 9, 38, 'service', False), (dump[b], 13, 39, 'service', False))
        for sites, start, col, relation, ordered in lists:
            for site, values in enumerate(sites[:, col:col + 9]):
                ids = _wire(values)
                present = ids[ids >= 0]
                inc(start + site, relation + '_unique_count', len(set(present.tolist())))
                inc(start + site, relation + '_instance_count', len(present))
                # Canonicalizing unordered multiset storage makes service-slot
                # permutations produce exactly identical graph tensors.
                if not ordered:
                    ids = np.sort(present)
                for slot, tid in enumerate(ids):
                    if tid < 0:
                        continue
                    attr = live.new_zeros(len(EDGE_FEATURE_NAMES))
                    if ordered:
                        attr[12:14] = live.new_tensor([(slot + 1) / 9, 1])
                    add(start + site, int(tid), relation, attr)
        for did, tid in enumerate(_wire(dump[b, :, 11])):
            if tid >= 0:
                add(13 + did, int(tid), 'admission')
                inc(13 + did, 'admission_count')
        for lid in range(4):
            attr = live.new_zeros(len(EDGE_FEATURE_NAMES))
            attr[9] = live[b, 1107 + 47 * lid + 1]
            add(9 + lid, int(candidate_dump[b, lid]), 'compatible_material', attr)

    edge_index = torch.tensor([(s, d) for s, d, _, _ in rows], dtype=torch.int64).T.contiguous()
    edge_type = torch.tensor([typ for _, _, typ, _ in rows], dtype=torch.int64)
    edge_attr = torch.stack([attr for _, _, _, attr in rows])
    quantities = live.new_zeros((bsize, NUM_NODES, len(QUANTITY_NAMES)))
    for index, terms in quantity_terms.items():
        quantities[index] = torch.stack([term if isinstance(term, torch.Tensor) else live.new_tensor(term)
                                        for term in terms]).sum()
    normalized = quantities / live.new_tensor(QUANTITY_SCALES)
    return GraphBatch(edge_index, edge_type, torch.tensor(RELATION_GROUP_IDS)[edge_type], edge_attr,
                      normalized, torch.from_numpy(candidate_dump),
                      torch.from_numpy(ego.astype(np.int64))).validate()


__all__ = ['GraphBatch', 'build_graph', 'NODE_NAMES', 'NUM_NODES', 'GROUP_NAMES', 'RELATION_NAMES',
           'RELATION_GROUP_IDS', 'EDGE_FEATURE_NAMES', 'EDGE_ATTRIBUTE_NAMES', 'QUANTITY_NAMES',
           'QUANTITY_SCALES', 'ROAD_ENDPOINTS', 'PHYSICAL_NODE_TO_TOKEN']
