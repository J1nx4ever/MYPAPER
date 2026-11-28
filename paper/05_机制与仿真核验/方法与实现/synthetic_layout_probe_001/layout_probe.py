"""Exercise frozen BG128 with manufactured layouts and no AnyLogic shifts."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3]
sys.path[:0] = [str(PROJECT), str(HERE.parent/'fleet_outage_resilience_001')]

import numpy as np
import torch

from business_graph_summary_v1.c_features import FEATURE_NAMES
from business_graph_summary_v1.graph import ROAD_ENDPOINTS
from business_graph_summary_v1.test_support import graph_state
from fleet_policy import FleetLearner, SummaryFeatureCache, pack_states


CHECKPOINT = PROJECT/'runs/bg128-fleet-resilience-20261124-repair01/models/dynamic/u0009/checkpoint.pt'
CHECKPOINT_SHA = 'b13e86feb74656def9077e412dbf4443857fba777ecec5e8ac758fd79a191f36'
NODE_NAMES = ('crusher', 'waste_dump', 'junction1', 'junction2', 'junction3',
              'load0', 'load1', 'load2', 'load3')
LAYOUT_A = ((0, 0), (1600, -900), (500, 0), (1500, 0), (2600, 0),
            (1500, 450), (2600, 600), (2850, 450), (1780, -700))
MOVED_NODES = {0: (0, -1200), 1: (1500, -1700), 5: (1500, 1050)}


def build_state(positions, edges):
    state = graph_state()
    roads = state[1439:1519].reshape(16, 5)
    lengths = []
    for i, (start, end) in enumerate(edges):
        length = float(np.linalg.norm(positions[start] - positions[end]))
        lengths.append(length)
        roads[i, :3] = [(start + 1) / 9, (end + 1) / 9, length / 10000]
    return state, lengths


@torch.no_grad()
def evaluate(model, state):
    batch = state[None]
    inputs = torch.from_numpy(pack_states(batch, [511], [511]))
    prefix = SummaryFeatureCache(batch, feature_config=model.actor.feature_config).prefix()
    probs = model.actor.distribution(inputs, prefix).probs[0]
    assert probs.shape == (4,) and bool(probs.isfinite().all())
    assert torch.allclose(probs.sum(), torch.tensor(1.0))
    feature_config = model.actor.feature_config['legacy_config']
    eta_index = FEATURE_NAMES.index('action.0.nominal_eta')
    valid_index = FEATURE_NAMES.index('action.0.eta_valid')
    assert prefix.features[0, valid_index] == 1
    eta = float(prefix.features[0, eta_index]) * feature_config['window_seconds']
    return dict(load0_nominal_eta_seconds=eta, candidate_probabilities=probs.tolist())


def probe():
    path_sha = sha256(CHECKPOINT.read_bytes()).hexdigest()
    if path_sha != CHECKPOINT_SHA:
        raise ValueError('frozen checkpoint differs')
    torch.set_num_threads(1)
    model = FleetLearner.load(CHECKPOINT)
    model.actor.eval()

    a = np.array(LAYOUT_A, dtype=float)
    b = a.copy()
    for node, coordinates in MOVED_NODES.items():
        b[node] = coordinates
    # Reattach Load0 to another junction while preserving 16 directed roads.
    new_links = {(3, 5): (4, 5), (5, 3): (5, 4)}
    c_edges = tuple(new_links.get(edge, edge) for edge in ROAD_ENDPOINTS)
    a_state, a_lengths = build_state(a, ROAD_ENDPOINTS)
    b_state, b_lengths = build_state(b, ROAD_ENDPOINTS)
    c_state, c_lengths = build_state(b, c_edges)

    original = evaluate(model, a_state)
    moved = evaluate(model, b_state)
    assert moved['load0_nominal_eta_seconds'] > original['load0_nominal_eta_seconds']
    assert not np.allclose(original['candidate_probabilities'], moved['candidate_probabilities'])
    try:
        evaluate(model, c_state)
    except ValueError as error:
        if str(error) != 'road endpoints differ from frozen production map':
            raise
        rejection = str(error)
    else:
        raise AssertionError('the frozen model accepted a different road topology')

    return dict(
        source='synthetic layouts; offline frozen inference',
        physical_shifts_started=0,
        checkpoint=dict(path=str(CHECKPOINT), sha256=path_sha),
        node_order=NODE_NAMES,
        directed_edges=dict(layout_a=ROAD_ENDPOINTS, layout_b=ROAD_ENDPOINTS, layout_c=c_edges),
        layout_a=dict(positions_m=a.tolist(), road_lengths_m=a_lengths, **original),
        layout_b=dict(positions_m=b.tolist(), road_lengths_m=b_lengths, **moved),
        layout_c=dict(positions_m=b.tolist(), road_lengths_m=c_lengths, rejection=rejection),
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = json.dumps(probe(), indent=2, allow_nan=False)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as file:
            file.write(result)
    else:
        print(result, end='')
