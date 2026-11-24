"""Offline checks of the two architecture manipulations and contracts."""
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from business_graph_summary_v1.models import Actor, MaterialCostCritic
from business_graph_summary_v1.test_support import business_state, graph_state, tensor
from single_critic_task_query_v1.features_cache import SummaryFeatureCache, _certificate, immutable

from .models import (ACTOR_SPECS, CRITIC_SPECS, FLAT_MLP, GRAPH_NO_C128, VARIANTS,
                     actor_from_config, architecture_contract, critic_from_config, feature_config_for)


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def config(variant):
    return dict(ablation_variant=variant, ablation_contract=architecture_contract(variant),
                network=ACTOR_SPECS[variant], memory_critic=CRITIC_SPECS[variant],
                candidate_id='s01_BG128', seed=373, feature_config=feature_config_for(variant))


@pytest.mark.parametrize('variant', VARIANTS)
def test_certified_features_mask_and_terminal(variant):
    actor, critic = actor_from_config(config(variant)), critic_from_config(config(variant))
    raw = business_state()
    raw[19] = 0
    states = tensor(raw)
    prefix = SummaryFeatureCache(raw[None], feature_config=actor.feature_config).prefix()
    probabilities = actor.distribution(states, prefix).probs
    assert probabilities.shape == (1, 4) and probabilities[0, 0] == 0
    assert probabilities.sum().item() == pytest.approx(1.)
    terminal = graph_state(terminal=True)
    combined = critic(tensor(np.stack((raw, terminal))), prefix)
    torch.testing.assert_close(combined, torch.cat((critic(states, prefix), torch.zeros(1))))
    with pytest.raises(ValueError, match='terminal Actor request'):
        actor(tensor(terminal), prefix)
    features = immutable(prefix.features + 1.)
    with pytest.raises(ValueError, match='byte certificate differs'):
        actor(states, replace(prefix, features=features))
    certificate = _certificate(prefix.source_states, features, prefix.query_indices,
                               prefix.feature_config_sha256)
    changed = replace(prefix, features=features, certificate_sha256=certificate)
    for current in (actor, critic):
        unchanged_output = current(states, prefix)
        changed_output = current(states, changed)
        if variant == FLAT_MLP:
            assert not torch.equal(unchanged_output, changed_output)
        else:
            assert torch.equal(unchanged_output, changed_output)


def test_flat_parameters_and_no_graph_execution():
    actor, critic = actor_from_config(config(FLAT_MLP)), critic_from_config(config(FLAT_MLP))
    actor_count = sum(parameter.numel() for parameter in actor.parameters())
    critic_count = sum(parameter.numel() for parameter in critic.parameters())
    assert actor_count == 260_480 and critic_count == 185_285
    assert actor_count + 1 == sum(parameter.numel() for parameter in Actor().parameters())
    assert critic_count + 12 == sum(parameter.numel() for parameter in MaterialCostCritic().parameters())
    raw = business_state()
    states = tensor(raw)
    prefix = SummaryFeatureCache(raw[None], feature_config=actor.feature_config).prefix()
    with patch('business_graph_summary_v1.models.build_graph', side_effect=AssertionError('graph used')):
        for model in (actor, critic):
            _, diagnostics = model(states, prefix, diagnostic=True)
            assert diagnostics['raw_and_summary'].shape == (1, 1647)
            assert torch.equal(diagnostics['raw_and_summary'][:, :1519], states)
    with pytest.raises(ValueError, match='incompatible checkpoint _architecture_contract'):
        actor.load_state_dict(Actor().state_dict())


def test_graph_quantity_and_candidate_channels_match_frozen_background():
    cfg = config(GRAPH_NO_C128)
    actor, critic = actor_from_config(cfg), critic_from_config(cfg)
    reference_actor = Actor(pair_seed=cfg['seed'])
    reference_critic = MaterialCostCritic(pair_seed=cfg['seed'])
    assert sum(parameter.numel() for parameter in actor.parameters()) == 251_313
    assert sum(parameter.numel() for parameter in critic.parameters()) == 177_073
    raw = business_state()
    states = tensor(raw)
    prefix = SummaryFeatureCache(raw[None], feature_config=actor.feature_config).prefix()
    for model, reference in ((actor, reference_actor), (critic, reference_critic)):
        baseline_parameters = dict(reference.named_parameters())
        for name, parameter in model.named_parameters():
            other = baseline_parameters[name]
            if other.shape != parameter.shape:
                other = other[:, :parameter.shape[1]]
            assert torch.equal(parameter, other), name
        _, details = model(states, prefix, diagnostic=True)
        _, baseline = reference(states, prefix, diagnostic=True)
        for key in ('typed_pools', 'quantity_pools', 'load_quantities', 'dump_quantities',
                    'load_tokens', 'dump_tokens', 'candidate_dump_ids'):
            assert torch.equal(details[key], baseline[key]), key
        assert 'current_summary' not in details
    _, details = actor(states, prefix, diagnostic=True)
    _, baseline = reference_actor(states, prefix, diagnostic=True)
    assert torch.equal(details['candidate_input'], baseline['candidate_input'][:, :, :-16])
    assert torch.equal(details['context_input'], baseline['context_input'][:, :-32])
    _, details = critic(states, prefix, diagnostic=True)
    _, baseline = reference_critic(states, prefix, diagnostic=True)
    assert torch.equal(details['fusion_input'], baseline['fusion_input'][:, :-32])
