"""Offline checkpoint, mask and physical graph regression checks."""

import pytest
import torch

from frozen_topology_policy import (
    ACTOR_DIGEST, CHECKPOINT_PATH, FrozenTopologyPolicy, SummaryFeatureCache,
    _actor_logits, _critic_value, _scene, models, pack_states,
)
from business_graph_summary_v1.graph import PHYSICAL_NODE_TO_TOKEN
from business_graph_summary_v1.test_support import business_state
from topology_policy import road_header


@pytest.fixture(scope='module', params=[('tree_A', 511), ('loop_B', 223)])
def case(request):
    name, opening_mask = request.param
    torch.set_num_threads(1)
    policy = FrozenTopologyPolicy(CHECKPOINT_PATH, name)
    assert policy.opening_mask == opening_mask
    state = business_state()
    for slot, truck in enumerate(state[27:1107].reshape(9, 120)):
        if opening_mask & (1 << slot) == 0:
            truck[4:16] = 0
            truck[15] = 1
            truck[17:41] = 0
            truck[43:] = 0
    return policy, state


@torch.inference_mode()
def test_restored_weights_match_original_nominal_actor_and_critic(case):
    policy, state = case
    assert models.digest(policy.actor) == ACTOR_DIGEST
    raw = state[None, :]
    fleet = [policy.opening_mask]
    packed = torch.from_numpy(pack_states(raw, fleet, fleet, node_order=policy.order))
    prefix = SummaryFeatureCache(raw, feature_config=policy.actor.feature_config).prefix()
    assert policy.actor.feature_config['enabled']
    logits = _actor_logits(policy.actor, packed, prefix)
    assert torch.equal(policy.actor(packed, prefix), logits)
    assert torch.equal(policy.actor.distribution(packed, prefix).probs,
                       torch.distributions.Categorical(logits=logits).probs)
    assert torch.equal(policy.critic(packed, prefix), _critic_value(policy.critic, packed, prefix))


def physical_state(policy, nominal):
    state = nominal.copy()
    roads = state[1439:1519].reshape(16, 5)
    roads.fill(0)
    roads[:, :4] = road_header(policy.edges)
    return state


@torch.inference_mode()
def test_adapted_policy_uses_actual_edges_and_excludes_parked_vehicles(case):
    policy, nominal = case
    state = physical_state(policy, nominal)
    state[19] = state[23] = 0
    fleet = policy.opening_mask
    mask = [bool(flag) for flag in state[19:23]]
    context, _, h, _, graph = _scene(policy.actor.state_encoder, policy.actor.status_projection,
                                    policy._inputs(state, mask, fleet, fleet, 0)[0])
    roads = {(policy.order[row['source']], policy.order[row['destination']])
             for row in graph.records() if row['relation'] == 'road'}
    assert roads == {(PHYSICAL_NODE_TO_TOKEN[src], PHYSICAL_NODE_TO_TOKEN[dst])
                     for src, dst, _, _ in policy.edges}
    for slot in range(9):
        if fleet & (1 << slot) == 0:
            assert torch.count_nonzero(h[0, slot]) == 0
            assert not torch.any(graph.edge_index == slot)
    assert torch.isfinite(context).all()

    probs, value = policy.evaluate(state, mask, fleet, fleet, 0)
    action = policy.action(state.tolist(), mask, fleet, fleet, 0)
    assert torch.isfinite(probs).all() and torch.isfinite(torch.tensor(value))
    assert probs[0] == 0
    assert mask[action] and action == int(probs.argmax())
    assert action == policy.action(state, mask, fleet, fleet, 0)


def test_foreign_road_fleet_ego_or_action_mask_is_rejected(case):
    policy, nominal = case
    state = physical_state(policy, nominal)
    fleet, mask = policy.opening_mask, state[19:23]
    with pytest.raises(ValueError, match='mask differs'):
        policy.action(state, [False] * 4, fleet, fleet, 0)
    with pytest.raises(ValueError, match='roster differs'):
        policy.action(state, mask, fleet ^ (1 << 8), fleet, 0)
    with pytest.raises(ValueError, match='disagrees with ego'):
        policy.action(state, mask, fleet, fleet, 1)
    state[1441] += 0.05
    with pytest.raises(ValueError, match='roads differ'):
        policy.action(state, mask, fleet, fleet, 0)
