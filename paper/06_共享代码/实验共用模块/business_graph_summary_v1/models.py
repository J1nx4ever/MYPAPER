"""Independent graph+C128 Actor and scalar cost Critic without final queries.

The Actor reads its candidate Load and compatible Dump directly. C128 is split
into its unchanged 88 task features and four unchanged 10-feature action blocks.
Both candidate and common context have an explicit multiplicative scoring path.
The Critic consumes the complete certified C128 and typed graph pooling.
"""
import copy
import hashlib
import json
import torch
from torch import nn
from torch.distributions import Categorical

from relation_gat_ego_v1.adapter import validate_states
from relation_gat_ego_v1.initialization import initialize
from single_critic_task_query_v1.c_features import (FEATURE_NAMES, feature_contract,
                                                   normalize_feature_config)
from single_critic_task_query_v1.features_cache import validate_sequence, pack_summary_prefixes
from .contract import ablation_contract, require, feature_config_for
from .encoder import TokenEncoder
from .graph import build_graph, QUANTITY_NAMES, NODE_NAMES, GROUP_NAMES
from .graph_layers import GraphLayer, finite

ACTOR_SPEC = dict(scene_tokens=19, graph_width=64, graph_layers=2,
                  final_query=False, typed_pool='mean_by_type', quantity_pool='sum_by_type',
                  task_summary_input=88, task_summary_embedding=32,
                  action_summary_input=10, action_summary_embedding=16,
                  candidate_width=64, context_width=64, scorer_hidden=128,
                  scorer_interaction='concat_product_plus_dot_over_sqrt64',
                  initialization='named-paired-v1')
CRITIC_SPEC = dict(scene_tokens=19, graph_width=64, graph_layers=2,
                   final_query=False, typed_pool='mean_by_type', quantity_pool='sum_by_type',
                   summary_input=128, summary_embedding=32, fusion_hidden=128,
                   output='single_joint_cost_value', initialization='named-paired-v1')
TYPE_SLICES = (slice(0, 9), slice(9, 13), slice(13, 15), slice(15, 18))
ACTION_SUMMARY_NAMES = ('legal', 'dispatch_q', 'ore', 'grade_mean', 'grade_std',
                        'available_q', 'queue_count', 'busy_fraction', 'nominal_eta', 'eta_valid')
require(len(FEATURE_NAMES) == 128 and FEATURE_NAMES[88:] == tuple(
    f'action.{a}.{name}' for a in range(4) for name in ACTION_SUMMARY_NAMES),
    'C128 task/action partition differs from frozen named feature layout')


def _stamp(value):
    raw = (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    return torch.tensor(list(hashlib.sha256(raw).digest()), dtype=torch.uint8)


def digest(module):
    h = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def common_actor_digest(actor):
    """All parameters belong to the sole frozen BG128 architecture."""
    h = hashlib.sha256()
    for name, value in sorted(actor.named_parameters()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def feature_config_sha256(config=None):
    return feature_contract(normalize_feature_config(config))['config_sha256']


def _states(states):
    require(torch.is_tensor(states) and states.dtype == torch.float32 and
            states.device.type == 'cpu' and states.ndim == 2 and
            states.shape[1] == 1519 and len(states) > 0 and bool(torch.isfinite(states).all()),
            'finite CPU float32 states[N,1519] required')
    require(not torch.is_autocast_enabled('cpu'), 'CPU autocast is outside the float32 contract')


class GraphStateEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_encoder = TokenEncoder()
        self.graph_layers = nn.ModuleList([GraphLayer(), GraphLayer()])

    def scene(self, states, *, diagnostic=False):
        graph = build_graph(states)
        z0 = self.token_encoder(states, graph)
        h = z0
        details = dict(z0=z0, graph=graph, edges=graph)
        for number, layer in enumerate(self.graph_layers, 1):
            if diagnostic:
                h, layer_details = layer(h, graph, diagnostic=True)
                details['layer' + str(number)] = layer_details
                details['h' + str(number)] = h
            else:
                h = layer(h, graph)
        details['z2'] = h
        return (h, details) if diagnostic else h

    def forward(self, states, *, diagnostic=False):
        require(bool((states[:, 17] == 0).all()), 'graph readout requires decision rows')
        # The graph is built once; neural states are recomputed on every call.
        h, details = self.scene(states, diagnostic=True)
        graph = details['graph']
        row = torch.arange(len(states))
        ego = h[row, graph.ego_ids]
        pools = torch.stack([h[:, section].mean(1) for section in TYPE_SLICES], dim=1)
        # Quantities bypass node LayerNorm and attention averaging explicitly.
        quantity_pools = torch.stack([graph.quantities[:, section].sum(1)
                                      for section in TYPE_SLICES], dim=1)
        raw = torch.cat((pools.flatten(1), ego, h[:, 18], quantity_pools.flatten(1)), dim=-1)
        details.update(typed_pools=pools, quantity_pools=quantity_pools,
                       raw_context=raw, ego_ids=graph.ego_ids, ego_token=ego,
                       ego_quantities=graph.quantities[row, graph.ego_ids],
                       load_tokens=h[:, 9:13],
                       dump_tokens=h[row[:, None], graph.candidate_dump_ids],
                       load_quantities=graph.quantities[:, 9:13],
                       dump_quantities=graph.quantities[row[:, None], graph.candidate_dump_ids],
                       candidate_dump_ids=graph.candidate_dump_ids)
        return (raw, details) if diagnostic else raw


class _ContractModule(nn.Module):
    """Fail before loading tensors if a checkpoint uses another architecture/C."""
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        for name in ('_feature_contract', '_architecture_contract'):
            key = prefix + name
            incoming = state_dict.get(key)
            require(torch.is_tensor(incoming) and
                    incoming.dtype == getattr(self, name).dtype and
                    incoming.shape == getattr(self, name).shape and
                    torch.equal(incoming.cpu(), getattr(self, name).cpu()),
                    'incompatible checkpoint ' + key)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)


def _common(module, spec, candidate, feature_config, role):
    module.candidate_id = candidate
    module.spec = copy.deepcopy(spec)
    design = ablation_contract(candidate)
    module.feature_config = normalize_feature_config(
        feature_config_for(candidate) if feature_config is None else feature_config)
    require(module.feature_config['enabled'] and design['summary_enabled'],
            'business graph baseline requires certified C128 enabled')
    module.state_encoder = GraphStateEncoder()
    module.register_buffer('_feature_contract', _stamp(feature_contract(module.feature_config)))
    module.register_buffer('_architecture_contract', _stamp(dict(graph=design, spec=spec, role=role)))


class Actor(_ContractModule):
    def __init__(self, spec=ACTOR_SPEC, *, candidate_id='s01_BG128', feature_config=None, pair_seed=1):
        super().__init__()
        require(spec == ACTOR_SPEC, 'frozen business graph Actor specification required')
        with torch.random.fork_rng(devices=[]):
            _common(self, spec, candidate_id, feature_config, 'actor')
            q = len(QUANTITY_NAMES)
            self.summary_task_encoder = nn.Sequential(nn.Linear(88, 32), nn.ReLU())
            self.summary_action_encoder = nn.Sequential(nn.Linear(10, 16), nn.ReLU())
            self.context_encoder = nn.Sequential(nn.Linear(384 + 4 * q + 32, 128),
                                                  nn.ReLU(), nn.Linear(128, 64))
            self.candidate_encoder = nn.Sequential(nn.Linear(192 + 3 * q + 2 + 16, 128),
                                                    nn.ReLU(), nn.Linear(128, 64))
            self.scorer = nn.Sequential(nn.Linear(192, 128), nn.ReLU(), nn.Linear(128, 1))
            initialize(self, pair_seed=pair_seed, role='actor')

    def score_candidates(self, candidates, context, *, diagnostic=False):
        require(candidates.shape[1:] == (4, 64) and context.shape == (len(candidates), 64),
                'candidate/context score dimensions differ')
        shared = context[:, None].expand(-1, 4, -1)
        product = finite(candidates * shared, 'candidate context interaction')
        score_input = torch.cat((candidates, shared, product), dim=-1)
        # The dot term prevents a solely additive common-context scoring path.
        dot = product.sum(-1) / 8.
        result = finite(self.scorer(score_input).squeeze(-1) + dot, 'Actor logits')
        return (result, dict(score_input=score_input, interaction=product, interaction_dot=dot)) if diagnostic else result

    def forward(self, states, prefixes=None, *, diagnostic=False):
        _states(states)
        mask = states[:, 19:23]
        require(bool((states[:, 17] == 0).all()) and bool(((mask == 0) | (mask == 1)).all())
                and bool(mask.bool().any(1).all()), 'invalid action mask or terminal Actor request')
        current = validate_sequence(prefixes, states, self.feature_config)
        raw, details = self.state_encoder(states, diagnostic=True)
        task_summary = self.summary_task_encoder(current[:, :88])
        action_summary = self.summary_action_encoder(current[:, 88:].reshape(-1, 4, 10))
        context_input = torch.cat((raw, task_summary), dim=-1)
        context = finite(self.context_encoder(context_input), 'Actor context')
        ego = details['ego_token'][:, None].expand(-1, 4, -1)
        ego_q = details['ego_quantities'][:, None].expand(-1, 4, -1)
        candidate_input = torch.cat((ego, details['load_tokens'], details['dump_tokens'],
                                     ego_q, details['load_quantities'], details['dump_quantities'],
                                     mask.unsqueeze(-1), states[:, 23:27, None], action_summary), dim=-1)
        candidates = finite(self.candidate_encoder(candidate_input), 'Actor candidates')
        logits, score_details = self.score_candidates(candidates, context, diagnostic=True)
        result = logits.masked_fill(~mask.bool(), -torch.inf)
        if diagnostic:
            details.update(current_summary=current, task_summary=task_summary,
                           action_summary=action_summary, candidate_input=candidate_input,
                           context_input=context_input, candidates=candidates, context=context,
                           unmasked_logits=logits, masked_logits=result, **score_details)
            return result, details
        return result

    def distribution(self, states, prefixes=None):
        return Categorical(logits=self(states, prefixes))


class MaterialCostCritic(_ContractModule):
    def __init__(self, spec=CRITIC_SPEC, *, candidate_id='s01_BG128', feature_config=None, pair_seed=1):
        super().__init__()
        require(spec == CRITIC_SPEC, 'frozen business graph scalar Critic specification required')
        with torch.random.fork_rng(devices=[]):
            _common(self, spec, candidate_id, feature_config, 'critic')
            self.summary_encoder = nn.Sequential(nn.Linear(128, 32), nn.ReLU())
            self.fusion = nn.Sequential(nn.Linear(384 + 4 * len(QUANTITY_NAMES) + 32, 128),
                                        nn.ReLU(), nn.Linear(128, 1))
            initialize(self, pair_seed=pair_seed, role='critic')

    def forward(self, states, prefixes=None, *, diagnostic=False):
        _states(states)
        validate_states(states)
        decisions = states[:, 17] == 0
        result = states.new_zeros(len(states))
        details = dict(terminal_bootstrap=0., decision_rows=torch.where(decisions)[0])
        if bool(decisions.any()):
            # Certified prefixes contain only decision rows, in their batch order.
            current = validate_sequence(prefixes, states[decisions], self.feature_config)
            raw, scene_details = self.state_encoder(states[decisions], diagnostic=True)
            summary = self.summary_encoder(current)
            fusion_input = torch.cat((raw, summary), dim=-1)
            values = finite(self.fusion(fusion_input), 'joint-cost Critic').squeeze(-1)
            result = result.index_copy(0, torch.where(decisions)[0], values)
            details.update(scene_details, current_summary=current, summary=summary,
                           fusion_input=fusion_input)
        return (result, details) if diagnostic else result


CommitmentCostCritic = MaterialCostCritic


def actor_from_config(config):
    require(config['graph_contract'] == ablation_contract(config['candidate_id']),
            'Actor graph contract differs')
    return Actor(config['network'], candidate_id=config['candidate_id'],
                 feature_config=config['feature_config'], pair_seed=config['seed'])


__all__ = ['Actor', 'MaterialCostCritic', 'CommitmentCostCritic', 'GraphStateEncoder',
           'ACTOR_SPEC', 'CRITIC_SPEC', 'actor_from_config', 'digest',
           'common_actor_digest', 'feature_config_sha256', 'pack_summary_prefixes']
