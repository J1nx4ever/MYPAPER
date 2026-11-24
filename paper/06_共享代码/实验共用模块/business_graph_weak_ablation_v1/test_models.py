"""Verify the localized mechanisms through the frozen Actor/Critic entry points."""
import numpy as np
import pytest
import torch

from business_graph_summary_v1.graph import GROUP_NAMES, NODE_NAMES
from business_graph_summary_v1.models import Actor, MaterialCostCritic
from business_graph_summary_v1.test_support import business_state, graph_state, tensor
from single_critic_task_query_v1.features_cache import SummaryFeatureCache

from .models import (
    ACTOR_SPECS, BLOCK_MATERIAL_DUMP_TO_LOAD_L2, CRITIC_SPECS, ONE_GAT_NODE_MLP,
    actor_from_config, architecture_contract, critic_from_config, feature_config_for,
)


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def _config(variant):
    return dict(ablation_variant=variant, ablation_contract=architecture_contract(variant),
                feature_config=feature_config_for(variant), candidate_id='s01_BG128', seed=373,
                network=ACTOR_SPECS[variant], memory_critic=CRITIC_SPECS[variant])


@pytest.fixture(params=[(actor_from_config, Actor, 260481), (critic_from_config, MaterialCostCritic, 185297)])
def role(request):
    return request.param


def _inputs(model):
    first = business_state()
    first[19] = 0
    raw = np.stack((first, graph_state()))
    return tensor(raw), SummaryFeatureCache(raw, feature_config=model.feature_config).prefix()


def test_material_block_identical_to_localized_frozen_probe(role):
    factory, baseline_type, size = role
    config = _config(BLOCK_MATERIAL_DUMP_TO_LOAD_L2)
    model = factory(config)
    baseline = baseline_type(pair_seed=config['seed'])
    states, prefix = _inputs(model)
    assert sum(parameter.numel() for parameter in model.parameters()) == size
    baseline_parameters = dict(baseline.named_parameters())
    assert all(torch.equal(parameter, baseline_parameters[name])
               for name, parameter in model.named_parameters())
    output, diagnosed = model(states, prefix, diagnostic=True)
    _, untouched = baseline(states, prefix, diagnostic=True)
    for key in ('z0', 'h1', 'quantity_pools', 'current_summary', 'candidate_dump_ids'):
        assert torch.equal(diagnosed[key], untouched[key]), key
    for key in ('buckets', 'scores', 'weights', 'group_masks', 'group_messages'):
        assert torch.equal(diagnosed['layer2'][key], untouched['layer2'][key]), key
    h2_delta = diagnosed['h2'] != untouched['h2']
    assert h2_delta[:, 9:13].any()
    assert not h2_delta[:, :9].any() and not h2_delta[:, 13:].any()

    def original_probe(_, args):
        fused = args[0].clone()
        is_load = torch.arange(len(fused)) % len(NODE_NAMES)
        is_load = (is_load >= 9) & (is_load < 13)
        start = GROUP_NAMES.index('material') * 64
        fused[is_load, start:start + 64] = 0
        return (fused,)

    output_layer = baseline.state_encoder.graph_layers[1].output
    hook = output_layer.register_forward_pre_hook(original_probe)
    try:
        assert torch.equal(output, baseline(states, prefix))
    finally:
        hook.remove()
    restored = factory(config)
    restored.load_state_dict(model.state_dict())
    assert torch.equal(restored(states, prefix), output)
    with pytest.raises(ValueError, match='incompatible checkpoint _architecture_contract'):
        model.load_state_dict(baseline.state_dict(), strict=False)


def test_nodewise_second_stage_retains_readouts_without_cross_node_exchange(role):
    factory, baseline_type, size = role
    config = _config(ONE_GAT_NODE_MLP)
    original_rng = torch.get_rng_state().clone()
    model = factory(config)
    baseline = baseline_type(pair_seed=config['seed'])
    assert torch.equal(original_rng, torch.get_rng_state())
    assert sum(parameter.numel() for parameter in model.parameters()) == size - 57
    baseline_parameters = dict(baseline.named_parameters())
    for name, parameter in model.named_parameters():
        if '.local_mlp.' not in name:
            assert torch.equal(parameter, baseline_parameters[name]), name
    states, prefix = _inputs(model)
    output, details = model(states, prefix, diagnostic=True)
    expected, untouched = baseline(states, prefix, diagnostic=True)
    assert output.shape == expected.shape
    assert torch.equal(torch.isfinite(output), torch.isfinite(expected))
    for key in ('z0', 'h1', 'quantity_pools', 'current_summary', 'candidate_dump_ids'):
        assert torch.equal(details[key], untouched[key]), key
    assert details['load_tokens'].shape[1:] == details['dump_tokens'].shape[1:] == (4, 64)
    assert torch.equal(details['h1'][:, 18], details['h2'][:, 18])

    stage = model.state_encoder.graph_layers[1]
    h1 = details['h1'].detach().clone().requires_grad_(True)
    h2 = stage(h1, details['graph'])
    grad, = torch.autograd.grad(h2[0, 9, 0], h1)
    assert grad[0, 9].abs().sum() > 0
    assert not grad[0, :9].any() and not grad[0, 10:].any() and not grad[1].any()
