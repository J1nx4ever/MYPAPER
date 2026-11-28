"""Check a topology batch and a read-only Full checkpoint without simulator runs."""

from hashlib import sha256
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3]
sys.path.insert(0, str(PROJECT))

import numpy as np
import torch

from business_graph_summary_v1 import models
from business_graph_summary_v1.features_cache import SummaryFeatureCache
from business_graph_summary_v1.graph import PHYSICAL_NODE_TO_TOKEN, RELATION_NAMES
from business_graph_summary_v1.test_support import graph_state
from compile_layouts import compile_layout, load_layouts
from topology_policy import attach_prototype_encoder, build_topology_graph, require_layout, road_header


CHECKPOINT = PROJECT / 'runs/bg128-gat-depth-20261125-full-s04/arms/s01_BG128/updates/0032/checkpoint.pt'
CHECKPOINT_SHA256 = '4f1171fb0873abb83d71124a3594ea1d70f2e871c242ab263c7daf82a4c1e182'


def clean_outputs(actor, critic, states):
    prefix = SummaryFeatureCache(states.numpy(), feature_config=actor.feature_config).prefix()
    return actor.distribution(states, prefix).probs, critic(states, prefix)


@torch.no_grad()
def run_probe():
    torch.set_num_threads(1)
    if sha256(CHECKPOINT.read_bytes()).hexdigest() != CHECKPOINT_SHA256:
        raise AssertionError('checkpoint bytes changed')
    saved = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    config = saved['config']
    actor = models.actor_from_config(config).eval()
    critic = models.MaterialCostCritic(config['memory_critic'], candidate_id=config['candidate_id'],
                                       feature_config=config['feature_config'], pair_seed=config['seed']).eval()
    actor.load_state_dict(saved['actor'], strict=True)
    critic.load_state_dict(saved['critic'], strict=True)

    nominal = torch.from_numpy(graph_state()[None])
    before = clean_outputs(actor, critic, nominal)
    for model in (actor, critic):
        attach_prototype_encoder(model)
        model.eval()
    after = clean_outputs(actor, critic, nominal)
    if not all(torch.equal(x, y) for x, y in zip(before, after)):
        raise AssertionError('the nominal contract no longer reproduces the same policy/value')

    layouts = []
    observations = []
    edge_sets = []
    for layout in load_layouts():
        payload, edges = compile_layout(layout)
        if (HERE / 'compiled' / f"{layout['id']}.properties").read_bytes() != payload:
            raise AssertionError('compiled scene changed')
        row = graph_state()
        road = row[1439:1519].reshape(16, 5)
        road.fill(0)
        road[:, :4] = road_header(edges)
        require_layout(row[None], edges)
        observations.append(row)
        edge_sets.append({(PHYSICAL_NODE_TO_TOKEN[a], PHYSICAL_NODE_TO_TOKEN[b]) for a,b,_,_ in edges})
        layouts.append({'id': layout['id'], 'nodes': len(layout['nodes_m']), 'roads': len(edges),
                        'scene_sha256': sha256(payload).hexdigest()})
    batch = torch.from_numpy(np.stack(observations))
    graph = build_topology_graph(batch)
    for i, expected in enumerate(edge_sets):
        measured = {(item['source'], item['destination']) for item in graph.records(i)
                    if item['relation'] == 'road'}
        if measured != expected:
            raise AssertionError('a different physical graph entered the encoder')
    probabilities, values = clean_outputs(actor, critic, batch)
    if not (torch.isfinite(probabilities).all() and torch.isfinite(values).all()):
        raise AssertionError('nonfinite topology batch')

    for i, layout in enumerate(layouts):
        layout.update(probabilities=probabilities[i].tolist(), value=values[i].item())
    return {'method': 'synthetic 1519D fixture per layout, nine populated truck slots',
            'physical_shifts_started': 0, 'checkpoint_sha256': CHECKPOINT_SHA256,
            'nominal_bit_exact_actor_and_critic': True, 'layouts': layouts}


if __name__ == '__main__':
    print(json.dumps(run_probe(), indent=2, allow_nan=False))
