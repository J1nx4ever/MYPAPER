"""Parameter-matched Flat-MLP and BG128 with the learned C128 inputs removed."""
from copy import deepcopy

import torch
from torch import nn
from torch.distributions import Categorical

from business_graph_summary_v1.contract import ablation_contract as baseline_contract, require
from business_graph_summary_v1.graph_layers import finite
from business_graph_summary_v1.models import (
    ACTOR_SPEC as BASE_ACTOR_SPEC,
    CRITIC_SPEC as BASE_CRITIC_SPEC,
    Actor as BaselineActor,
    MaterialCostCritic as BaselineCritic,
    _ContractModule,
    _stamp,
    _states,
)
from relation_gat_ego_v1.adapter import validate_states
from relation_gat_ego_v1.initialization import initialize
from single_critic_task_query_v1.c_features import feature_contract, normalize_feature_config
from single_critic_task_query_v1.features_cache import validate_sequence


FLAT_MLP = 'flat_mlp'
GRAPH_NO_C128 = 'graph_no_c128'
VARIANTS = (FLAT_MLP, GRAPH_NO_C128)
ACTOR_SPECS = {
    FLAT_MLP: dict(raw_input=1519, summary_input=128, hidden=[142, 180],
                   output=4, activation='relu', initialization='named-paired-v1'),
    GRAPH_NO_C128: {**{key: value for key, value in BASE_ACTOR_SPEC.items()
                       if not key.startswith(('task_summary_', 'action_summary_'))},
                   'task_summary_input': 0, 'action_summary_input': 0,
                   'initialization': 'paired-bg128-with-summary-columns-removed-v1'},
}
CRITIC_SPECS = {
    FLAT_MLP: dict(raw_input=1519, summary_input=128, hidden=[103, 148],
                   output='single_joint_cost_value', activation='relu', initialization='named-paired-v1'),
    GRAPH_NO_C128: {**{key: value for key, value in BASE_CRITIC_SPEC.items()
                       if key != 'summary_embedding'}, 'summary_input': 0,
                   'initialization': 'paired-bg128-with-summary-columns-removed-v1'},
}


def feature_config_for(variant):
    require(variant in VARIANTS, 'unknown business graph ablation')
    # Sealed data/replay keep the BG128 carrier. Only Flat-MLP consumes it.
    return normalize_feature_config({'enabled': True})


def architecture_contract(variant):
    require(variant in VARIANTS, 'unknown business graph ablation')
    return dict(
        version='business-graph-ablation-v1', variant=variant,
        raw_dimension=1519, certified_summary_dimension=128,
        consumed_summary_dimension=128 if variant == FLAT_MLP else 0,
        graph_design_sha256=baseline_contract()['design_sha256'] if variant == GRAPH_NO_C128 else None,
        action_mask='raw_legal_actions', terminal_bootstrap=0.0,
        actor_critic_sharing=False,
        actor=deepcopy(ACTOR_SPECS[variant]), critic=deepcopy(CRITIC_SPECS[variant]),
    )


class _AblationModule(_ContractModule):
    def __init__(self, variant, role, config):
        super().__init__()
        architecture = architecture_contract(variant)
        require(config['ablation_contract'] == architecture, 'ablation architecture differs')
        self.candidate_id = config['candidate_id']
        self.spec = deepcopy(architecture[role])
        key = 'network' if role == 'actor' else 'memory_critic'
        require(config[key] == self.spec, f'ablation {role} specification differs')
        self.feature_config = normalize_feature_config(config['feature_config'])
        require(self.feature_config == feature_config_for(variant), 'C128 carrier contract differs')
        self.register_buffer('_feature_contract', _stamp(feature_contract(self.feature_config)))
        self.register_buffer('_architecture_contract', _stamp(dict(
            architecture=architecture, candidate_id=self.candidate_id, role=role)))


def _flat_network(spec, output_dim):
    first, second = spec['hidden']
    return nn.Sequential(nn.Linear(spec['raw_input'] + spec['summary_input'], first), nn.ReLU(),
                         nn.Linear(first, second), nn.ReLU(), nn.Linear(second, output_dim))


def _without_summary_columns(encoder, summary_width):
    original = encoder[0]
    projection = nn.Linear(original.in_features - summary_width, original.out_features)
    with torch.no_grad():
        projection.weight.copy_(original.weight[:, :-summary_width])
        projection.bias.copy_(original.bias)
    encoder[0] = projection
    return encoder


class _Actor(_AblationModule):
    def forward(self, states, prefixes=None, *, diagnostic=False):
        _states(states)
        mask = states[:, 19:23]
        require(bool((states[:, 17] == 0).all()) and bool(((mask == 0) | (mask == 1)).all())
                and bool(mask.bool().any(1).all()), 'invalid action mask or terminal Actor request')
        current = validate_sequence(prefixes, states, self.feature_config)
        logits, details = self._unmasked(states, mask, current, diagnostic=diagnostic)
        result = logits.masked_fill(~mask.bool(), -torch.inf)
        if diagnostic:
            details.update(unmasked_logits=logits, masked_logits=result)
            return result, details
        return result

    def distribution(self, states, prefixes=None):
        return Categorical(logits=self(states, prefixes))


class FlatActor(_Actor):
    def __init__(self, config):
        super().__init__(FLAT_MLP, 'actor', config)
        with torch.random.fork_rng(devices=[]):
            self.mlp = _flat_network(self.spec, 4)
            initialize(self, pair_seed=config['seed'], role='actor')

    def _unmasked(self, states, mask, current, *, diagnostic):
        inputs = torch.cat((states, current), dim=-1)
        logits = finite(self.mlp(inputs), 'Flat-MLP Actor logits')
        details = dict(raw_and_summary=inputs, current_summary=current) if diagnostic else None
        return logits, details


class GraphNoSummaryActor(_Actor):
    score_candidates = BaselineActor.score_candidates

    def __init__(self, config):
        super().__init__(GRAPH_NO_C128, 'actor', config)
        with torch.random.fork_rng(devices=[]):
            baseline = BaselineActor(pair_seed=config['seed'])
            self.state_encoder = baseline.state_encoder
            self.context_encoder = _without_summary_columns(
                baseline.context_encoder, BASE_ACTOR_SPEC['task_summary_embedding'])
            self.candidate_encoder = _without_summary_columns(
                baseline.candidate_encoder, BASE_ACTOR_SPEC['action_summary_embedding'])
            self.scorer = baseline.scorer

    def _unmasked(self, states, mask, _current, *, diagnostic):
        raw, details = self.state_encoder(states, diagnostic=True)
        context = finite(self.context_encoder(raw), 'Actor context')
        ego = details['ego_token'][:, None].expand(-1, 4, -1)
        ego_q = details['ego_quantities'][:, None].expand(-1, 4, -1)
        candidate_input = torch.cat((ego, details['load_tokens'], details['dump_tokens'],
                                     ego_q, details['load_quantities'], details['dump_quantities'],
                                     mask.unsqueeze(-1), states[:, 23:27, None]), dim=-1)
        candidates = finite(self.candidate_encoder(candidate_input), 'Actor candidates')
        if diagnostic:
            logits, score_details = self.score_candidates(candidates, context, diagnostic=True)
            details.update(candidate_input=candidate_input, context_input=raw,
                           candidates=candidates, context=context, **score_details)
            return logits, details
        return self.score_candidates(candidates, context), None


class _Critic(_AblationModule):
    def forward(self, states, prefixes=None, *, diagnostic=False):
        _states(states)
        validate_states(states)
        decisions = states[:, 17] == 0
        rows = torch.where(decisions)[0]
        result = states.new_zeros(len(states))
        details = dict(terminal_bootstrap=0., decision_rows=rows)
        if len(rows):
            decision_states = states[decisions]
            current = validate_sequence(prefixes, decision_states, self.feature_config)
            values, decision_details = self._decision_values(decision_states, current, diagnostic=diagnostic)
            result = result.index_copy(0, rows, values)
            if diagnostic:
                details.update(decision_details)
        return (result, details) if diagnostic else result


class FlatCostCritic(_Critic):
    def __init__(self, config):
        super().__init__(FLAT_MLP, 'critic', config)
        with torch.random.fork_rng(devices=[]):
            self.mlp = _flat_network(self.spec, 1)
            initialize(self, pair_seed=config['seed'], role='critic')

    def _decision_values(self, states, current, *, diagnostic):
        inputs = torch.cat((states, current), dim=-1)
        values = finite(self.mlp(inputs), 'Flat-MLP joint-cost Critic').squeeze(-1)
        details = dict(raw_and_summary=inputs, current_summary=current) if diagnostic else None
        return values, details


class GraphNoSummaryCostCritic(_Critic):
    def __init__(self, config):
        super().__init__(GRAPH_NO_C128, 'critic', config)
        with torch.random.fork_rng(devices=[]):
            baseline = BaselineCritic(pair_seed=config['seed'])
            self.state_encoder = baseline.state_encoder
            self.fusion = _without_summary_columns(baseline.fusion, BASE_CRITIC_SPEC['summary_embedding'])

    def _decision_values(self, states, _current, *, diagnostic):
        encoded = self.state_encoder(states, diagnostic=diagnostic)
        raw, details = encoded if diagnostic else (encoded, None)
        values = finite(self.fusion(raw), 'joint-cost Critic').squeeze(-1)
        if diagnostic:
            details['fusion_input'] = raw
        return values, details


def actor_from_config(config):
    variant = config['ablation_variant']
    require(variant in VARIANTS, 'unknown business graph ablation')
    factory = FlatActor if variant == FLAT_MLP else GraphNoSummaryActor
    return factory(config)


def critic_from_config(config):
    variant = config['ablation_variant']
    require(variant in VARIANTS, 'unknown business graph ablation')
    factory = FlatCostCritic if variant == FLAT_MLP else GraphNoSummaryCostCritic
    return factory(config)
