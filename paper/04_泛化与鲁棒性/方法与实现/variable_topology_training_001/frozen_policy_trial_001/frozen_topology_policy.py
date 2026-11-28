"""Sealed dynamic u9 inference on the real A/B physical observation."""

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import sys

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
TOPOLOGY_PLAN = HERE.parent
FLEET_PLAN = TOPOLOGY_PLAN.parent / 'fleet_outage_resilience_001'
PROJECT = HERE.parents[4]
for directory in (PROJECT, TOPOLOGY_PLAN, FLEET_PLAN):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from business_graph_summary_v1.graph import QUANTITY_NAMES
from compile_layouts import compile_layout, load_layouts
from fleet_policy import (
    FleetLearner, NODES, RAW, SummaryFeatureCache, episode_node_order,
    models, pack_states, require, validate_sequence,
)
from relation_gat_ego_v1.adapter import validate_state
from topology_policy import build_topology_graph, require_layout


CHECKPOINT_PATH = PROJECT / 'runs/bg128-fleet-resilience-20261124-repair01/models/dynamic/u0009/checkpoint.pt'
CHECKPOINT_SHA256 = 'b13e86feb74656def9077e412dbf4443857fba777ecec5e8ac758fd79a191f36'
CHECKPOINT_MANIFEST_SHA256 = '24f4cd40396be250583ea6cac2a7a44131c2878059f212c3307b2339ecb79baf'
CHECKPOINT_SEAL_SHA256 = 'adb7badc0477c1bbd676d82641b29949f5f9ae2c8004da706a3e3e2e217939e5'
ACTOR_DIGEST = '16b82688d37c0786d5b1510f0598763feb3d37cef590a68c4d5f05aa8b55e5a6'
_JUNCTION_ROAD_COUNTS = [QUANTITY_NAMES.index(name) for name in
                         ('incoming_road_count', 'outgoing_road_count')]


def _scene(encoder, status_projection, packed):
    """Retain the deployed fleet encoder and apply the observed road relations."""
    raw = packed[:, :RAW].contiguous()
    eligible, present = packed[:, RAW:RAW+9], packed[:, RAW+9:RAW+18]
    order = packed[:, RAW+18:].long()
    rows = torch.arange(len(raw))
    inverse = order.argsort(1)
    status = torch.stack((eligible-1, present-1), dim=-1)
    graph = build_topology_graph(raw)
    tokens = encoder.token_encoder(raw, graph)
    trucks = (tokens[:, :9] + status_projection(status)) * present.unsqueeze(-1)
    global_token = tokens[:, 18:] + status_projection(status.mean(1, keepdim=True))
    h = torch.cat((trucks, tokens[:, 9:18], global_token), dim=1)[rows[:, None], order]
    mask = present.gather(1, order[:, :9]).unsqueeze(-1)
    sample, physical = graph.edge_index // NODES, graph.edge_index % NODES
    graph = replace(graph, edge_index=sample * NODES + inverse[sample, physical],
                    quantities=graph.quantities[rows[:, None], order],
                    candidate_dump_ids=graph.candidate_dump_ids.gather(1, order[:, 9:13]-9),
                    ego_ids=inverse[rows, graph.ego_ids])
    for layer in encoder.graph_layers:
        h = layer(h, graph)
        h = torch.cat((h[:, :9] * mask, h[:, 9:]), dim=1)

    physical_h = h[rows[:, None], inverse]
    quantities = graph.quantities
    physical_quantities = quantities[rows[:, None], inverse]
    active_mean = (physical_h[:, :9] * present.unsqueeze(-1)).sum(1) / present.sum(1, keepdim=True)
    truck_pool = torch.where(present.bool().all(1, keepdim=True), physical_h[:, :9].mean(1), active_mean)
    junction_slice = models.TYPE_SLICES[-1]
    junctions = physical_h[:, junction_slice]
    junction_mask = (physical_quantities[:, junction_slice, _JUNCTION_ROAD_COUNTS]
                     .sum(-1, keepdim=True) > 0)
    active_junction_mean = (junctions * junction_mask).sum(1) / junction_mask.sum(1).clamp_min(1)
    junction_pool = torch.where(junction_mask.all(1), junctions.mean(1), active_junction_mean)
    pools = torch.stack([truck_pool, *(physical_h[:, s].mean(1) for s in models.TYPE_SLICES[1:-1]),
                         junction_pool], 1)
    ego = h[rows, graph.ego_ids]
    counts = torch.stack([physical_quantities[:, s].sum(1) for s in models.TYPE_SLICES], 1)
    context = torch.cat((pools.flatten(1), ego, h[:, 18], counts.flatten(1)), dim=-1)
    return context, ego, h, quantities, graph


def _actor_logits(actor, packed, prefixes):
    raw = packed[:, :RAW].contiguous()
    current = validate_sequence(prefixes, raw, actor.feature_config)
    context, ego, h, quantities, graph = _scene(actor.state_encoder, actor.status_projection, packed)
    task = actor.summary_task_encoder(current[:, :88])
    action = actor.summary_action_encoder(current[:, 88:].reshape(-1, 4, 10))
    context = actor.context_encoder(torch.cat((context, task), -1))
    rows = torch.arange(len(h))
    loads = packed[:, RAW+18+9:RAW+18+13].long()-9
    mask = raw[:, 19:23].gather(1, loads)
    dump_indices = graph.candidate_dump_ids
    candidate = torch.cat((ego[:, None].expand(-1, 4, -1), h[:, 9:13], h[rows[:, None], dump_indices],
                           quantities[rows, graph.ego_ids][:, None].expand(-1, 4, -1),
                           quantities[:, 9:13], quantities[rows[:, None], dump_indices],
                           mask.unsqueeze(-1), raw[:, 23:27].gather(1, loads).unsqueeze(-1),
                           action[rows[:, None], loads]), -1)
    scores = actor.score_candidates(actor.candidate_encoder(candidate), context)
    return scores.masked_fill(~mask.bool(), -torch.inf).gather(1, loads.argsort(1))


def _critic_value(critic, packed, prefixes):
    raw = packed[:, :RAW].contiguous()
    require(not bool(raw[:, 17].any()), 'terminal bootstrap is zero')
    current = validate_sequence(prefixes, raw, critic.feature_config)
    context, *_ = _scene(critic.state_encoder, critic.status_projection, packed)
    return critic.fusion(torch.cat((context, critic.summary_encoder(current)), -1)).squeeze(-1)


class FrozenTopologyPolicy:
    """One fixed roster/road layout per trial, with deterministic action readout."""

    def __init__(self, checkpoint_path, layout_id, *, actor_sampling_seed=0):
        layouts = {layout['id']: layout for layout in load_layouts()}
        require(layout_id in layouts, 'unknown frozen policy trial layout')
        layout = layouts[layout_id]
        scene, self.edges = compile_layout(layout)
        require((TOPOLOGY_PLAN / 'compiled' / f'{layout_id}.properties').read_bytes() == scene,
                'compiled topology scene bytes differ')
        self.layout_id = layout_id
        self.layout_sha256 = sha256(scene).hexdigest()
        self.opening_mask = sum(1 << slot for slot in layout['active_trucks'])
        self.order = episode_node_order(actor_sampling_seed, self.opening_mask)

        checkpoint = Path(checkpoint_path)
        for suffix, expected in (('', CHECKPOINT_SHA256), ('.manifest.json', CHECKPOINT_MANIFEST_SHA256),
                                 ('.seal.json', CHECKPOINT_SEAL_SHA256)):
            require(sha256(Path(str(checkpoint) + suffix).read_bytes()).hexdigest() == expected,
                    'sealed dynamic u9 checkpoint or metadata differs')
        restored = FleetLearner.load(checkpoint)
        require(restored.update_index == 9 and restored.policy_version == ACTOR_DIGEST,
                'restored policy is not the frozen dynamic u9')
        self.actor_digest = restored.policy_version
        self.actor, self.critic = restored.actor.eval(), restored.critic.eval()

    def _inputs(self, obs, action_mask, dispatchable_mask, in_system_mask, truck_id):
        raw = np.asarray(obs, dtype=np.float32)
        validate_state(raw, truck_id=truck_id)
        mask = np.asarray(action_mask)
        require(mask.shape == (4,) and mask.dtype.kind in 'biuf'
                and np.array_equal(mask, raw[19:23]), 'exported action mask differs from observation')
        require(all(type(value) in (int, np.int32, np.int64, np.uint32, np.uint64)
                    for value in (dispatchable_mask, in_system_mask))
                and dispatchable_mask == in_system_mask == self.opening_mask,
                'observed roster differs from the frozen trial layout')
        batch = raw[None, :]
        require_layout(batch, self.edges)
        packed = torch.from_numpy(pack_states(batch, [dispatchable_mask], [in_system_mask],
                                            node_order=self.order))
        prefix = SummaryFeatureCache(batch, feature_config=self.actor.feature_config).prefix()
        return packed, prefix

    @torch.inference_mode()
    def action(self, obs, action_mask, dispatchable_mask, in_system_mask, truck_id):
        packed, prefix = self._inputs(obs, action_mask, dispatchable_mask, in_system_mask, truck_id)
        return int(_actor_logits(self.actor, packed, prefix)[0].argmax().item())

    @torch.inference_mode()
    def evaluate(self, obs, action_mask, dispatchable_mask, in_system_mask, truck_id):
        """Single-decision diagnostic scores and value without sampling."""
        packed, prefix = self._inputs(obs, action_mask, dispatchable_mask, in_system_mask, truck_id)
        logits = _actor_logits(self.actor, packed, prefix)
        probabilities = torch.distributions.Categorical(logits=logits).probs[0]
        return probabilities, float(_critic_value(self.critic, packed, prefix)[0])
