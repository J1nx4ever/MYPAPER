"""Causal analytical fixtures with the actual production map; no simulator."""
import numpy as np
import torch
from grade_commitment_ppo_v1.test_features import synthetic_state, put_truck
from .graph import ROAD_ENDPOINTS


def graph_state(*, terminal=False):
    state = synthetic_state(terminal=terminal, window_index=4 if terminal else 0)
    roads = state[1439:1519].reshape(16, 5)
    roads[:, :2] = (np.asarray(ROAD_ENDPOINTS, dtype=np.float32) + 1) / 9
    return state


def business_state():
    """Two Load0 commitments, one known loaded ore, one loaded waste.

    Truck1 waits at Load0; Truck2 approaches Load0 on road8 (3 -> 5).
    Truck3 carries ore toward Dump0 on road0 (0 -> 2) only as an analytical
    edge/route fixture; routes are not future labels. Truck4 is at Dump1.
    """
    state = put_truck(graph_state(), tid=1, phase=3, q=90, loaded=False, ore=True, target=0)
    state = put_truck(state, tid=2, phase=2, q=120, loaded=False, ore=True, target=0)
    state = put_truck(state, tid=3, phase=5, q=140, loaded=True, ore=True, grade=.3)
    state = put_truck(state, tid=4, phase=8, q=150, loaded=True, ore=False, grade=0)
    t = state[27:1107].reshape(9, 120)
    t[1:5, 32:41] = 0
    t[1, 32 + 5] = 1
    t[4, 32 + 1] = 1
    for tid, rid in ((2, 8), (3, 6)):
        t[tid, 45 + rid] = 1
        t[tid, 61:63] = [.04, .1]
        t[tid, 63 + rid] = 1 / 16
    t[2, 81] = 2 / 9  # Truck1 predecessor reference, separately typed.
    loads = state[1107:1295].reshape(4, 47)
    loads[0, 6:8] = [210 / 30000, 1 - 210 / 30000]
    loads[0, 17] = 2 / 9
    loads[0, 20] = 2 / 9
    dumps = state[1295:1391].reshape(2, 48)
    dumps[1, 30] = 5 / 9
    return state


def tensor(states=None, *, requires_grad=False):
    values = np.asarray(graph_state() if states is None else states)
    if values.ndim == 1:
        values = values[None]
    return torch.tensor(values, dtype=torch.float32, requires_grad=requires_grad)


__all__ = ['graph_state', 'business_state', 'tensor', 'put_truck']
