"""Offline numerical/mechanism tests, never simulator or training evidence."""
from copy import deepcopy
from dataclasses import replace
from unittest.mock import patch
import numpy as np
import pytest
import torch

from .models import Actor, MaterialCostCritic, digest, ACTOR_SPEC, CRITIC_SPEC
from .encoder import TokenEncoder
from .graph_layers import GraphLayer
from .graph import build_graph, RELATION_NAMES, GROUP_NAMES, QUANTITY_NAMES, EDGE_FEATURE_NAMES
from .features_cache import SummaryFeatureCache
from .test_support import graph_state, business_state, tensor


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def pair(seed=373):
    return Actor(pair_seed=seed), MaterialCostCritic(pair_seed=seed)


def prefix(raw, model):
    raw = np.asarray(raw)
    return SummaryFeatureCache(raw[None] if raw.ndim == 1 else raw,
                               feature_config=model.feature_config).prefix()


def subset(graph, selection):
    return replace(graph, edge_index=graph.edge_index[:, selection],
                   edge_type=graph.edge_type[selection], edge_group=graph.edge_group[selection],
                   edge_attr=graph.edge_attr[selection]).validate()


def test_complete_direct_readout_inventory_and_c128_partition():
    actor, critic = pair()
    raw = business_state()
    x, p = tensor(raw), prefix(raw, actor)
    logits, a = actor(x, p, diagnostic=True)
    values, c = critic(x, p, diagnostic=True)
    assert logits.shape == (1, 4) and values.shape == (1,)
    for model, detail in ((actor, a), (critic, c)):
        assert detail['z0'].shape == detail['z2'].shape == (1, 19, 64)
        assert len(model.state_encoder.graph_layers) == 2
        assert not any('query' in name or 'retrieval' in name for name, _ in model.named_modules())
        assert not hasattr(model.state_encoder.token_encoder, 'road_final')
        assert torch.equal(detail['z0'][:, 18], detail['z2'][:, 18])
        assert detail['typed_pools'].shape == (1, 4, 64)
        assert detail['quantity_pools'].shape == (1, 4, len(QUANTITY_NAMES))
        for layer in ('layer1', 'layer2'):
            assert not detail[layer]['group_masks'][:, 18].any()
    torch.testing.assert_close(a['candidate_input'][:, :, :64],
                               a['ego_token'][:, None].expand(-1, 4, -1), atol=0, rtol=0)
    torch.testing.assert_close(a['candidate_input'][:, :, 64:128], a['z2'][:, 9:13], atol=0, rtol=0)
    torch.testing.assert_close(a['candidate_input'][:, :, 128:192],
                               a['z2'][torch.arange(1)[:, None], a['candidate_dump_ids']], atol=0, rtol=0)
    current = torch.tensor(p.features.copy())
    torch.testing.assert_close(a['task_summary'], actor.summary_task_encoder(current[:, :88]), atol=0, rtol=0)
    torch.testing.assert_close(a['action_summary'], actor.summary_action_encoder(current[:, 88:].reshape(1, 4, 10)), atol=0, rtol=0)
    torch.testing.assert_close(c['summary'], critic.summary_encoder(current), atol=0, rtol=0)


def test_group_softmax_oracle_and_transport_multiplicity_cannot_dilute_business():
    graph = build_graph(tensor(business_state()))
    layer = GraphLayer()
    h = torch.linspace(-1., 1., 19 * 64).reshape(1, 19, 64)
    _, d = layer(h, graph, diagnostic=True)
    for bucket in d['buckets'].unique():
        selected = d['buckets'] == bucket
        torch.testing.assert_close(d['weights'][selected], d['scores'][selected].softmax(0), atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(d['weights'][selected].sum(0), torch.ones(4), atol=1e-6, rtol=0)
    rid = torch.where((graph.edge_type == RELATION_NAMES.index('road')) &
                      (graph.edge_index[1] == 9))[0][0]
    indices = torch.cat((torch.arange(len(graph.edge_type)), rid.repeat(11)))
    larger = subset(graph, indices)
    _, changed = layer(h, larger, diagnostic=True)
    business = graph.edge_group != GROUP_NAMES.index('transport')
    assert torch.equal(d['weights'][business], changed['weights'][:len(business)][business])
    assert not d['group_messages'][~d['group_masks']].any()


def test_single_incoming_road_still_transmits_edge_values_and_has_parameter_gradient():
    graph = build_graph(tensor(graph_state()))
    graph = subset(graph, torch.tensor([0]))
    graph = replace(graph, edge_attr=graph.edge_attr.clone().requires_grad_())
    layer = GraphLayer()
    h = torch.linspace(-1., 1., 19 * 64).reshape(1, 19, 64)
    before, details = layer(h, graph, diagnostic=True)
    torch.testing.assert_close(details['weights'], torch.ones(1, 4), atol=0, rtol=0)
    attr = graph.edge_attr.clone()
    attr[0, EDGE_FEATURE_NAMES.index('road_length_10000m')] += .2
    changed = layer(h, replace(graph, edge_attr=attr))
    destination = int(graph.edge_index[1, 0])
    assert (changed[0, destination] - before[0, destination]).abs().max() > 1e-6
    (before[0, destination] * torch.arange(64)).sum().backward()
    column = len(RELATION_NAMES) + EDGE_FEATURE_NAMES.index('road_length_10000m')
    assert layer.edge_value.weight.grad[:, column].abs().sum() > 1e-6
    assert graph.edge_attr.grad[0, EDGE_FEATURE_NAMES.index('road_length_10000m')].abs() > 1e-6


def test_loaded_truck_to_dump_to_load_information_arrives_in_two_layers():
    graph = build_graph(tensor(business_state()))
    selected = (((graph.edge_type == RELATION_NAMES.index('loaded_dump')) &
                 (graph.edge_index[0] == 3) & (graph.edge_index[1] == 13)) |
                ((graph.edge_type == RELATION_NAMES.index('compatible_material_reverse')) &
                 (graph.edge_index[0] == 13) & (graph.edge_index[1] == 9)))
    graph = subset(graph, selected)
    assert len(graph.edge_type) == 2
    h = torch.linspace(-1., 1., 19 * 64).reshape(1, 19, 64).requires_grad_()
    first, second = GraphLayer(), GraphLayer()
    one = first(h, graph)
    loss1 = (one[0, 9] * torch.arange(64)).sum()
    grad1 = torch.autograd.grad(loss1, h, retain_graph=True)[0]
    assert not grad1[0, 3].any()
    two = second(one, graph)
    grad2 = torch.autograd.grad((two[0, 9] * torch.arange(64)).sum(), h)[0]
    assert grad2[0, 3].abs().sum() > 1e-6


def test_quantities_bypass_layernorm_and_enter_candidate_load_dump_scoring():
    actor, _ = pair()
    raw = business_state()
    logits, detail = actor(tensor(raw), prefix(raw, actor), diagnostic=True)
    q = len(QUANTITY_NAMES)
    expected = detail['quantity_pools'].flatten(1)
    assert torch.equal(detail['raw_context'][:, -4 * q:], expected)
    assert expected.abs().sum() > 0
    assert torch.equal(detail['candidate_input'][:, :, 192 + q:192 + 2 * q], detail['load_quantities'])
    assert torch.equal(detail['candidate_input'][:, :, 192 + 2 * q:192 + 3 * q], detail['dump_quantities'])
    detail['candidate_input'].retain_grad()
    detail['context_input'].retain_grad()
    (logits[0, 0] - logits[0, 3]).backward()
    grad = detail['candidate_input'].grad
    assert grad[:, :, 64:128].abs().sum() > 1e-7
    assert grad[:, :, 128:192].abs().sum() > 1e-7
    assert grad[:, :, 192 + q:192 + 3 * q].abs().sum() > 1e-7
    assert detail['context_input'].grad[:, 384:384 + 4 * q].abs().sum() > 1e-7
    assert actor.state_encoder.token_encoder.quantity_projection.weight.grad.abs().sum() > 1e-7


def test_common_context_changes_relative_scores_even_with_zero_mlp_scorer():
    actor, _ = pair()
    with torch.no_grad():
        for p in actor.scorer.parameters():
            p.zero_()
    candidates = torch.zeros(1, 4, 64)
    candidates[0, 0, 0] = 1
    candidates[0, 1, 0] = -1
    context = torch.zeros(1, 64, requires_grad=True)
    logits = actor.score_candidates(candidates, context)
    grad = torch.autograd.grad(logits[0, 0] - logits[0, 1], context)[0]
    assert grad[0, 0].item() == pytest.approx(.25)
    modified = context.detach().clone()
    modified[0, 0] = 2
    result = actor.score_candidates(candidates, modified)
    assert (result[0, 0] - result[0, 1]).item() == pytest.approx(.5)


def test_encoder_roads_routes_and_stale_node_occupancy_use_current_facts():
    encoder = TokenEncoder()
    raw = business_state()
    x = tensor(raw, requires_grad=True)
    z, detail = encoder(x, diagnostic=True)
    trucks = raw[27:1107].reshape(9, 120)
    at_node = (trucks[:, 45:61].sum(-1) == 0) & (trucks[:, 32:41].sum(-1) > 0)
    assert detail['node_input'][0, :, 50].sum().item() == pytest.approx(at_node.sum() / 9)
    (z[0, 15:18] * torch.arange(64)).sum().backward()
    road_grad = x.grad[:, 1439:1519].reshape(1, 16, 5)
    assert road_grad[:, :, 2:5].abs().sum() > 1e-7
    assert encoder.road_base.weight.grad.abs().sum() > 1e-7
    assert encoder.node_content.weight.grad.abs().sum() > 1e-7


def test_mask_terminal_mixed_batch_and_certified_prefix_alignment():
    actor, critic = pair()
    normal, terminal = business_state(), graph_state(terminal=True)
    changed = normal.copy()
    changed[19] = 0
    probs = actor.distribution(tensor(changed), prefix(changed, actor)).probs
    assert probs[0, 0] == 0 and probs.sum().item() == pytest.approx(1.)
    with patch.object(critic.state_encoder, 'forward', side_effect=AssertionError('terminal encoded')):
        assert torch.equal(critic(tensor(terminal)), torch.zeros(1))
    with pytest.raises(ValueError):
        actor(tensor(terminal))
    decision_raw = np.stack((normal, changed))
    p = prefix(decision_raw, actor)
    output = critic(tensor(np.stack((normal, terminal, changed))), p)
    torch.testing.assert_close(output[[0, 2]], critic(tensor(decision_raw), p), atol=1e-6, rtol=1e-5)
    assert output[1] == 0
    with pytest.raises(ValueError, match='current raw observation'):
        critic(tensor(np.stack((changed, terminal, normal))), p)
    with pytest.raises(ValueError, match='current raw observation'):
        actor(tensor(changed), prefix(normal, actor))


def test_initialization_rng_reproducibility_independent_parameters_and_checkpoint_rejection():
    before = torch.get_rng_state().clone()
    actor, critic = pair()
    again, _ = pair()
    assert torch.equal(before, torch.get_rng_state())
    assert digest(actor) == digest(again)
    different, _ = pair(374)
    assert digest(actor) != digest(different)
    assert not {p.data_ptr() for p in actor.parameters()} & {p.data_ptr() for p in critic.parameters()}
    raw = business_state()
    before_critic = digest(critic)
    actor.distribution(tensor(raw), prefix(raw, actor)).log_prob(torch.tensor([0])).sum().backward()
    assert all(p.grad is None for p in critic.parameters()) and digest(critic) == before_critic
    actor.zero_grad(set_to_none=True)
    critic(tensor(raw), prefix(raw, critic)).sum().backward()
    assert all(p.grad is None for p in actor.parameters())
    frozen = deepcopy(actor.state_dict())
    frozen['_architecture_contract'][0] ^= 1
    previous = digest(actor)
    with pytest.raises(ValueError, match='incompatible checkpoint'):
        actor.load_state_dict(frozen, strict=False)
    assert digest(actor) == previous


def test_current_parameter_recompute_and_batched_readonly_inference():
    actor, critic = pair()
    raw = np.stack((graph_state(), business_state()))
    cache = SummaryFeatureCache(raw, feature_config=actor.feature_config)
    x, snapshot = tensor(raw), tensor(raw)
    old_rng, old_actor = torch.get_rng_state().clone(), digest(actor)
    combined = actor(x, cache.prefix())
    separate = torch.cat([actor(x[i:i + 1], cache.prefix([i])) for i in range(2)])
    torch.testing.assert_close(combined, separate, atol=1e-6, rtol=1e-5)
    assert torch.equal(x, snapshot) and torch.equal(old_rng, torch.get_rng_state())
    assert digest(actor) == old_actor
    critic_before = digest(critic)
    optimizer = torch.optim.Adam(actor.parameters(), lr=.001)
    optimizer.zero_grad()
    (-actor.distribution(x, cache.prefix()).log_prob(torch.tensor([0, 1]))).sum().backward()
    optimizer.step()
    assert not torch.equal(actor(x, cache.prefix()), combined)
    assert digest(critic) == critic_before and all(p.grad is None for p in critic.parameters())


def test_nonfinite_state_and_layernorm_overflow_fail_closed():
    actor, _ = pair()
    raw = graph_state()
    raw[0] = np.nan
    with pytest.raises(ValueError):
        actor(tensor(raw))
    layer = GraphLayer()
    graph = build_graph(tensor(graph_state()))
    h = torch.ones(1, 19, 64) * 1e20
    h[:, :, ::2] *= -1
    with torch.no_grad():
        for p in layer.parameters():
            p.zero_()
        layer.norm.weight.fill_(1)
    with pytest.raises(ValueError, match='variance'):
        layer(h, graph)
