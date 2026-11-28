"""Opt-in road adapter for the 9/16-slot layout prototype.

The legacy builder supplies business relations and station quantities. Its road
and truck-on-road relations are replaced before reaching the neural encoder.
This module is not registered in the production trainer.
"""

from dataclasses import replace

import numpy as np
import torch

from business_graph_summary_v1 import models
from business_graph_summary_v1.graph import (
    NUM_NODES, PHYSICAL_NODE_TO_TOKEN, QUANTITY_NAMES, QUANTITY_SCALES,
    RELATION_GROUP_IDS, RELATION_NAMES, ROAD_ENDPOINTS, build_graph,
)
from relation_gat_ego_v1.adapter import validate_states


_ROAD_LABELS = {
    name: RELATION_NAMES.index(name)
    for name in ('road', 'truck_segment_from', 'truck_segment_from_reverse',
                 'truck_segment_to', 'truck_segment_to_reverse')
}
_ROAD_QUANTITIES = {
    name: QUANTITY_NAMES.index(name)
    for name in ('incoming_road_count', 'outgoing_road_count',
                 'incoming_segment_truck_count', 'outgoing_segment_truck_count')
}


def road_header(edges):
    encoded = np.zeros((16, 4), dtype=np.float32)
    for i, (a, b, length, speed) in enumerate(edges):
        encoded[i] = ((a + 1) / 9, (b + 1) / 9, length / 10000, speed / 10)
    return encoded


def require_layout(states, expected_edges):
    rows = states[:, 1439:1519].reshape(-1, 16, 5)
    if not np.allclose(rows[:, :, :4], road_header(expected_edges), rtol=0.0, atol=2e-7):
        raise ValueError('observation roads differ from the admitted scene')


def build_topology_graph(states):
    validate_states(states)
    roads = states[:, 1439:1519].reshape(-1, 16, 5)
    raw = roads.detach().numpy()
    endpoints = np.rint(raw[:, :, :2] * 9).astype(np.int64) - 1
    if np.all(endpoints == np.asarray(ROAD_ENDPOINTS)):
        return build_graph(states)
    present = (endpoints >= 0).all(-1)
    trucks = states[:, 27:1107].reshape(-1, 9, 120).detach().numpy()
    if np.any(raw[~present]) or np.any(raw[present, 2:4] <= 0):
        raise ValueError('invalid padded or active road')

    # The stock graph encoder has a frozen physical-map contract. Use it only to
    # derive the unchanged business relations from the original business fields.
    proxy = states.clone()
    proxy[:, 1439:1519].reshape(-1, 16, 5)[:, :, :2] = (states.new_tensor(ROAD_ENDPOINTS) + 1) / 9
    base = build_graph(proxy)
    keep = ~torch.isin(base.edge_type, torch.tensor(tuple(_ROAD_LABELS.values())))
    quantities = base.quantities.clone()
    for quantity in _ROAD_QUANTITIES.values():
        quantities[:, :, quantity] = 0
    additions = []
    for sample, slots in enumerate(endpoints):
        roads_live = np.flatnonzero(present[sample])
        pairs = [tuple(slots[i]) for i in roads_live]
        nodes = {node for edge in pairs for node in edge}
        if len(set(pairs)) != len(pairs) or any(a == b for a, b in pairs) or not {0, 1, 5, 6, 7, 8} <= nodes:
            raise ValueError('invalid relation topology or missing station')
        active_truck_nodes = np.nonzero(trucks[sample, :, 32:41])[1]
        used_roads = (trucks[sample, :, 45:61] != 0) | (trucks[sample, :, 63:79] != 0)
        if not set(active_truck_nodes.tolist()) <= nodes or used_roads[:, ~present[sample]].any():
            raise ValueError('vehicle references padded topology')
        offset = sample * NUM_NODES

        def add(a, b, name, attributes):
            additions.append((offset + a, offset + b, _ROAD_LABELS[name], attributes))

        for rid, (a, b) in zip(roads_live, pairs):
            source = PHYSICAL_NODE_TO_TOKEN[a]
            destination = PHYSICAL_NODE_TO_TOKEN[b]
            occupants = np.flatnonzero(trucks[sample, :, 45 + rid])
            attrs = states.new_zeros(14)
            attrs[:3] = roads[sample, rid, 2:5]
            attrs[3:5] = states.new_tensor((len(occupants) / 9, 1))
            add(source, destination, 'road', attrs)
            for token, key, value in (
                (source, 'outgoing_road_count', 1), (destination, 'incoming_road_count', 1),
                (source, 'outgoing_segment_truck_count', len(occupants)),
                (destination, 'incoming_segment_truck_count', len(occupants)),
            ):
                idx = _ROAD_QUANTITIES[key]
                quantities[sample, token, idx] += value / QUANTITY_SCALES[idx]
            for truck in occupants:
                in_transit = attrs.clone()
                in_transit[5] = states[sample, 27 + truck * 120 + 61]
                in_transit[6] = 1
                for node, name in ((source, 'truck_segment_from'), (destination, 'truck_segment_to')):
                    add(truck, node, name, in_transit)
                    add(node, truck, name + '_reverse', in_transit)
    ends, types, attributes = (
        torch.tensor([(a, b) for a, b, _, _ in additions], dtype=torch.int64).T,
        torch.tensor([typ for _, _, typ, _ in additions], dtype=torch.int64),
        torch.stack([value for _, _, _, value in additions]),
    )
    edge_type = torch.cat((base.edge_type[keep], types))
    return replace(base, edge_index=torch.cat((base.edge_index[:, keep], ends), dim=1),
                   edge_type=edge_type, edge_group=torch.tensor(RELATION_GROUP_IDS)[edge_type],
                   edge_attr=torch.cat((base.edge_attr[keep], attributes)), quantities=quantities).validate()


class TopologyStateEncoder(models.GraphStateEncoder):
    def scene(self, states, *, diagnostic=False):
        graph = build_topology_graph(states)
        h = self.token_encoder(states, graph)
        details = dict(z0=h, graph=graph, edges=graph)
        for number, layer in enumerate(self.graph_layers, 1):
            if diagnostic:
                h, detail = layer(h, graph, diagnostic=True)
                details.update({f'layer{number}': detail, f'h{number}': h})
            else:
                h = layer(h, graph)
        details['z2'] = h
        return (h, details) if diagnostic else h

    def forward(self, states, *, diagnostic=False):
        context, details = super().forward(states, diagnostic=True)
        junctions = models.TYPE_SLICES[-1]
        index = [_ROAD_QUANTITIES['incoming_road_count'], _ROAD_QUANTITIES['outgoing_road_count']]
        present = details['graph'].quantities[:, junctions, index].sum(-1, keepdim=True) > 0
        old = details['typed_pools']
        mean = (details['z2'][:, junctions] * present).sum(1) / present.sum(1).clamp_min(1)
        pooled = torch.where(present.all(1), old[:, -1], mean)
        pools = torch.cat((old[:, :-1], pooled[:, None]), dim=1)
        context = torch.cat((pools.flatten(1), context[:, old.shape[1] * old.shape[2]:]), dim=-1)
        details.update(typed_pools=pools, raw_context=context)
        return (context, details) if diagnostic else context


def attach_prototype_encoder(module):
    """Reuse trained graph weights in memory. No old checkpoint is overwritten."""
    encoder = TopologyStateEncoder()
    encoder.load_state_dict(module.state_encoder.state_dict(), strict=True)
    module.state_encoder = encoder
