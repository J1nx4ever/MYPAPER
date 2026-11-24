"""The complete revision002 entity encoder: original S1519 -> Z0[32,64].

Only discrete reference decoding is detached. Every continuous content path is
recomputed under the current module parameters. No C summary, selected action,
future state, or learned graph output enters this acyclic encoder.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from .adapter import validate_states

_REFERENCE_RANGES = ((26, 41), (45, 61), (63, 79), (81, 82))
TRUCK_LOCAL_INDICES = tuple(i for i in range(120)
                          if not any(a <= i < b for a, b in _REFERENCE_RANGES))
assert len(TRUCK_LOCAL_INDICES) == 72
TOKEN_NAMES = tuple([f'truck.{i}' for i in range(9)] +
                    [f'load.{i}' for i in range(4)] +
                    [f'dump.{i}' for i in range(2)] +
                    [f'road.{i}' for i in range(16)] + ['global'])


def _onehot_ids(value):
    raw = value.detach()
    return torch.where(raw.sum(-1) > 0, raw.argmax(-1), -1)


def _wire_ids(value):
    return torch.round(value.detach().double() * 9).long() - 1


def _identity(ids, width, dtype=torch.float32):
    return F.one_hot(ids.clamp_min(0), width).to(dtype) * (ids >= 0).unsqueeze(-1)


def _reference(base, ids):
    """Base content, typed canonical identity, present; missing never aliases ID0."""
    batch = torch.arange(base.shape[0], device=base.device)
    batch = batch.reshape((-1,) + (1,) * (ids.ndim - 1)).expand_as(ids)
    present = (ids >= 0).to(base.dtype).unsqueeze(-1)
    content = base[batch, ids.clamp_min(0)] * present
    return torch.cat((content, _identity(ids, base.shape[1], base.dtype), present), -1)


def _finite(value, name):
    if not bool(torch.isfinite(value).all()):
        raise ValueError('nonfinite token encoder intermediate: ' + name)
    return value


def _normalize(module, value, name):
    value = _finite(value, name + '_input')
    centered = _finite(value - _finite(value.mean(-1, keepdim=True), name + '_mean'), name + '_centered')
    _finite(centered.square(), name + '_variance_terms')
    return _finite(module(value), name)


class TokenEncoder(nn.Module):
    PARAMETERS = 48432

    def __init__(self):
        super().__init__()
        for name, ni, no in (
            ('truck_base', 72, 16), ('site_base', 64, 16), ('road_base', 4, 16),
            ('node_content', 54, 16), ('ordered_truck_refs', 234, 16),
            ('service_set', 26, 16), ('route_ordered', 544, 16),
            ('fifo_ordered', 48, 32), ('truck_final', 223, 64),
            ('site_final', 150, 64), ('road_final', 72, 64), ('global_final', 19, 64),
        ):
            setattr(self, name, nn.Linear(ni, no))
        self.semantic_type_embedding = nn.Parameter(torch.empty(5, 64))
        nn.init.normal_(self.semantic_type_embedding)
        self.truck_ln = nn.LayerNorm(64, eps=1e-5)
        self.site_ln = nn.LayerNorm(64, eps=1e-5)
        self.road_ln = nn.LayerNorm(64, eps=1e-5)
        self.global_ln = nn.LayerNorm(64, eps=1e-5)
        assert sum(p.numel() for p in self.parameters()) == self.PARAMETERS

    def _project(self, name, value):
        return _finite(F.relu(getattr(self, name)(_finite(value, name + '_input'))), name)

    def forward(self, states, *, diagnostic=False):
        if torch.is_autocast_enabled('cpu'):
            raise ValueError('token encoder autocast must be disabled')
        validate_states(states)
        if not isinstance(states, torch.Tensor):
            # numpy is accepted at the external interface; no tensor path is detached.
            states = torch.tensor(states, dtype=torch.float32)
        b = states.shape[0]
        t = states[:, 27:1107].reshape(b, 9, 120)
        loads = states[:, 1107:1295].reshape(b, 4, 47)
        dumps = states[:, 1295:1391].reshape(b, 2, 48)
        fifo = states[:, 1391:1439].reshape(b, 16, 3)
        road = states[:, 1439:1519].reshape(b, 16, 5)
        motion = t[..., 92:120].reshape(b, 9, 4, 7)
        valid = motion[..., :1]
        front = valid * motion[..., 4:5]
        gated_motion = torch.cat((valid, motion[..., 1:4] * valid, front,
                                  motion[..., 5:7] * front), -1)
        gated = torch.cat((t[..., :22], t[..., 22:23] * t[..., 21:22],
                           t[..., 23:92], gated_motion.flatten(-2)), -1)
        local = gated[..., TRUCK_LOCAL_INDICES]
        truck_base = self._project('truck_base', local)
        zeros = states.new_zeros
        site_local = torch.cat((torch.cat((loads[..., :20], zeros((b, 4, 11))), -1),
                                torch.cat((zeros((b, 2, 20)), dumps[..., :11]), -1)), 1)
        site_mask = states.new_zeros((6, 31))
        site_mask[:4, :20] = 1
        site_mask[4:, 20:] = 1
        site_kind = F.one_hot(torch.tensor([0, 0, 0, 0, 1, 1]), 2).to(states.dtype)
        site_input = torch.cat((site_local, site_mask.expand(b, -1, -1),
                                site_kind.expand(b, -1, -1)), -1)
        site_base = self._project('site_base', site_input)
        segment = _onehot_ids(t[..., 45:61])
        segment_map = _identity(segment, 16, states.dtype)
        road_intrinsic = torch.cat((road[..., 2:5], segment_map.sum(1).unsqueeze(-1) / 9), -1)
        road_base = self._project('road_base', road_intrinsic)
        road_nodes = _wire_ids(road[..., :2])
        node_ids = _onehot_ids(t[..., 32:41])
        incoming = _identity(road_nodes[..., 1], 9, states.dtype).transpose(1, 2)
        outgoing = _identity(road_nodes[..., 0], 9, states.dtype).transpose(1, 2)
        at_node = _identity(node_ids, 9, states.dtype).transpose(1, 2)
        counts = torch.stack((incoming.sum(-1), outgoing.sum(-1), at_node.sum(-1)), -1)
        means = [torch.bmm(mask, base) / mask.sum(-1, keepdim=True).clamp_min(1)
                 for mask, base in ((incoming, road_base), (outgoing, road_base), (at_node, truck_base))]
        node_input = torch.cat((*means, counts / states.new_tensor([16, 16, 9]),
                                (counts > 0).to(states.dtype)), -1)
        node_content = self._project('node_content', node_input)
        load_ref = _reference(site_base[:, :4], _onehot_ids(t[..., 26:30]))
        dump_ref = _reference(site_base[:, 4:], _onehot_ids(t[..., 30:32]))
        node_ref = _reference(node_content, node_ids)
        segment_ref = _reference(road_base, segment)
        predecessor_ref = _reference(truck_base, _wire_ids(t[..., 81]))
        rank = torch.round(t[..., 63:79].detach() * 16).long()
        # rank-major slots, with road identity retained even for identical bases.
        wanted = torch.arange(1, 17).reshape(1, 1, 16, 1)
        match = rank.unsqueeze(-2) == wanted
        route_ids = torch.where(match.any(-1), match.long().argmax(-1), -1)
        route_refs = _reference(road_base, route_ids)
        route_present_slots = (route_ids >= 0).to(states.dtype).unsqueeze(-1)
        slot_rank = torch.arange(1, 17, dtype=states.dtype).reshape(1, 1, 16, 1) / 16
        route_input = torch.cat((route_refs[..., :32], slot_rank * route_present_slots,
                                 route_present_slots), -1)
        route_present = route_present_slots.any(-2).to(states.dtype)
        route_content = self._project('route_ordered', route_input.flatten(-2)) * route_present
        truck_input = torch.cat((local, torch.eye(9).expand(b, -1, -1), load_ref,
                                 dump_ref, node_ref, segment_ref, predecessor_ref,
                                 route_content, route_present), -1)
        # Site lists share one ordered projection. Load capacityWait/admission absent.
        empty_slots = zeros((b, 4, 9))
        queue = torch.cat((loads[..., 20:29], dumps[..., 12:21]), 1)
        seize = torch.cat((loads[..., 29:38], dumps[..., 21:30]), 1)
        wait = torch.cat((empty_slots, dumps[..., 30:39]), 1)
        service = torch.cat((loads[..., 38:47], dumps[..., 39:48]), 1)
        ordered_inputs, ordered_groups = [], []
        for slots in (queue, seize, wait):
            ids = _wire_ids(slots)
            refs = _reference(truck_base, ids)
            present = (ids >= 0).any(-1, keepdim=True).to(states.dtype)
            content = self._project('ordered_truck_refs', refs.flatten(-2)) * present
            ordered_inputs.append(refs)
            ordered_groups.append(torch.cat((content, present), -1))
        service_ids = _wire_ids(service)
        service_refs = _reference(truck_base, service_ids)
        service_count = (service_ids >= 0).sum(-1, keepdim=True).to(states.dtype)
        service_input = torch.cat((service_refs[..., :16].sum(-2) / service_count.clamp_min(1),
                                   service_refs[..., 16:25].sum(-2) / 9,
                                   service_count / 9), -1)
        service_present = (service_count > 0).to(states.dtype)
        service_content = self._project('service_set', service_input) * service_present
        admission = torch.cat((zeros((b, 4)), dumps[..., 11]), 1)
        admission_ref = _reference(truck_base, _wire_ids(admission))
        any_valid = fifo[..., 0].bool().any(-1, keepdim=True).to(states.dtype)
        fifo_content = self._project('fifo_ordered', fifo.flatten(-2)) * any_valid
        attached = zeros((b, 6, 1))
        attached[:, 4] = 1
        site_fifo = attached * fifo_content.unsqueeze(1)
        site_input_final = torch.cat((site_base, torch.eye(6).expand(b, -1, -1),
                                      *ordered_groups, torch.cat((service_content, service_present), -1),
                                      admission_ref, site_fifo, attached,
                                      attached * any_valid.unsqueeze(1)), -1)
        road_input = torch.cat((road_intrinsic, torch.eye(16).expand(b, -1, -1),
                                _reference(node_content, road_nodes[..., 0]),
                                _reference(node_content, road_nodes[..., 1])), -1)
        truck_tokens = _normalize(self.truck_ln, self._project('truck_final', truck_input) + self.semantic_type_embedding[0], 'truck_ln')
        site_type = self.semantic_type_embedding[torch.tensor([1, 1, 1, 1, 2, 2])]
        site_tokens = _normalize(self.site_ln, self._project('site_final', site_input_final) + site_type, 'site_ln')
        road_tokens = _normalize(self.road_ln, self._project('road_final', road_input) + self.semantic_type_embedding[3], 'road_ln')
        global_token = _normalize(self.global_ln, self._project('global_final', states[:, :19]) + self.semantic_type_embedding[4], 'global_ln')
        z0 = _finite(torch.cat((truck_tokens, site_tokens, road_tokens, global_token.unsqueeze(1)), 1), 'Z0')
        if not diagnostic:
            return z0
        details = dict(token_names=TOKEN_NAMES, truck_local=local, truck_base=truck_base,
                       site_base_input=site_input, site_base=site_base,
                       road_intrinsic=road_intrinsic, road_base=road_base,
                       node_input=node_input, node_content=node_content,
                       load_ref=load_ref, dump_ref=dump_ref, node_ref=node_ref,
                       segment_ref=segment_ref, predecessor_ref=predecessor_ref,
                       route_ids=route_ids, route_input=route_input, route_content=route_content,
                       route_present=route_present, queue_input=ordered_inputs[0],
                       seize_input=ordered_inputs[1], capacity_wait_input=ordered_inputs[2],
                       service_input=service_input, admission_ref=admission_ref,
                       fifo_raw=fifo, fifo_content=fifo_content, fifo_any_valid=any_valid,
                       truck_input=truck_input, site_input=site_input_final, road_input=road_input,
                       global_input=states[:, :19], action_sidecar=states[:, 19:27],
                       ego_ids=_onehot_ids(t[..., 0].unsqueeze(1)).squeeze(1),
                       road_node_ids=road_nodes, truck_node_ids=node_ids)
        return z0, details


__all__ = ['TokenEncoder', 'TOKEN_NAMES', 'TRUCK_LOCAL_INDICES']
