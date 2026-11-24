"""Keep the BG128 encoder and readouts while changing one second-layer mechanism."""
from copy import deepcopy

import torch
from torch import nn

from business_graph_summary_v1.contract import ablation_contract as baseline_contract, require
from business_graph_summary_v1.graph import GROUP_NAMES, NODE_NAMES, GraphBatch
from business_graph_summary_v1.graph_layers import checked_norm, finite
from business_graph_summary_v1.models import (
    ACTOR_SPEC as BASE_ACTOR_SPEC,
    CRITIC_SPEC as BASE_CRITIC_SPEC,
    Actor,
    MaterialCostCritic,
    _stamp,
)
from relation_gat_ego_v1.initialization import initialize
from single_critic_task_query_v1.c_features import normalize_feature_config


BLOCK_MATERIAL_DUMP_TO_LOAD_L2 = 'block_material_dump_to_load_l2'
ONE_GAT_NODE_MLP = 'one_gat_node_mlp'
VARIANTS = (BLOCK_MATERIAL_DUMP_TO_LOAD_L2, ONE_GAT_NODE_MLP)

_WIDTH = 64
_LOCAL_HIDDEN = 263
_LOADS = slice(NODE_NAMES.index('Load0'), NODE_NAMES.index('Dump0'))
_GLOBAL = NODE_NAMES.index('Global')
_MATERIAL_START = GROUP_NAMES.index('material') * _WIDTH
_MATERIAL = slice(_MATERIAL_START, _MATERIAL_START + _WIDTH)


def _specifications(baseline):
    return {
        BLOCK_MATERIAL_DUMP_TO_LOAD_L2: dict(baseline,
            second_stage='gat_with_load_material_aggregate_zeroed_before_group_projection'),
        ONE_GAT_NODE_MLP: dict(baseline, graph_layers=1,
            second_stage='nodewise_residual_mlp_with_layer_norm_and_global_bypass',
            nodewise_hidden=_LOCAL_HIDDEN, nodewise_activation='leaky_relu_0.2'),
    }


ACTOR_SPECS = _specifications(BASE_ACTOR_SPEC)
CRITIC_SPECS = _specifications(BASE_CRITIC_SPEC)


def feature_config_for(variant):
    require(variant in VARIANTS, 'unknown localized graph ablation')
    return normalize_feature_config({'enabled': True})


def architecture_contract(variant):
    require(variant in VARIANTS, 'unknown localized graph ablation')
    intervention = {
        BLOCK_MATERIAL_DUMP_TO_LOAD_L2: dict(layer=2, group='material', source_type='Dump',
            destination_type='Load', stage='zero_aggregate_before_fusion', retain_group_presence=True),
        ONE_GAT_NODE_MLP: dict(layer=2, stage='replace_message_passing_with_nodewise_residual_mlp',
            widths=[_WIDTH, _LOCAL_HIDDEN, _WIDTH], layer_norm=True, global_bypass=True),
    }[variant]
    return dict(
        version='business-graph-weak-ablation-v1', variant=variant,
        baseline_graph_design_sha256=baseline_contract()['design_sha256'],
        raw_dimension=1519, certified_summary_dimension=128, consumed_summary_dimension=128,
        intervention=intervention,
        actor=deepcopy(ACTOR_SPECS[variant]), critic=deepcopy(CRITIC_SPECS[variant]),
    )


class _SuppressLoadMaterial(nn.Linear):
    """A linear group fusion with the Load material aggregate input zeroed."""
    def forward(self, aggregates):
        # The first four 64D blocks are group aggregates. The final four
        # columns are group presence and still reach the original projection.
        fused = aggregates.reshape(-1, len(NODE_NAMES), self.in_features).clone()
        fused[:, _LOADS, _MATERIAL] = 0
        return super().forward(fused.reshape_as(aggregates))


class _NodewiseResidual(nn.Module):
    """Second processing stage with no exchange of neighboring node states."""
    def __init__(self, norm, *, pair_seed, role):
        super().__init__()
        self.local_mlp = nn.Sequential(nn.Linear(_WIDTH, _LOCAL_HIDDEN), nn.LeakyReLU(.2),
                                       nn.Linear(_LOCAL_HIDDEN, _WIDTH))
        self.norm = norm
        initialize(self, pair_seed=pair_seed, role=role)

    def forward(self, h, graph, *, diagnostic=False):
        require(isinstance(h, torch.Tensor) and h.dtype == torch.float32
                and h.device.type == 'cpu' and h.ndim == 3
                and h.shape[1:] == (len(NODE_NAMES), _WIDTH) and len(h) > 0,
                'nodewise stage requires CPU float32 H[B,19,64]')
        require(not torch.is_autocast_enabled('cpu'), 'nodewise CPU autocast must be disabled')
        require(isinstance(graph, GraphBatch) and graph.batch_size == len(h),
                'matching business GraphBatch required')
        finite(h, 'nodewise input')
        physical, global_token = h[:, :_GLOBAL], h[:, _GLOBAL:]
        projected = finite(self.local_mlp(physical), 'nodewise projection')
        pre_norm = finite(physical + projected, 'nodewise residual')
        normalized = checked_norm(self.norm, pre_norm, 'physical norm')
        result = torch.cat((normalized, global_token), dim=1)
        if diagnostic:
            return result, dict(projected=torch.cat((projected, torch.zeros_like(global_token)), dim=1),
                                pre_norm=torch.cat((pre_norm, global_token), dim=1))
        return result


def _model_from_config(config, role):
    variant = config['ablation_variant']
    contract = architecture_contract(variant)
    require(config['ablation_contract'] == contract, 'localized graph ablation architecture differs')
    network_key = 'network' if role == 'actor' else 'memory_critic'
    require(config[network_key] == contract[role], f'localized ablation {role} specification differs')
    feature_config = normalize_feature_config(config['feature_config'])
    require(feature_config == feature_config_for(variant), 'C128 carrier must remain enabled')
    model_type = Actor if role == 'actor' else MaterialCostCritic
    seed = config['seed']
    with torch.random.fork_rng(devices=[]):
        model = model_type(candidate_id=config['candidate_id'], feature_config=feature_config, pair_seed=seed)
        layer = model.state_encoder.graph_layers[1]
        if variant == BLOCK_MATERIAL_DUMP_TO_LOAD_L2:
            original = layer.output
            projection = _SuppressLoadMaterial(original.in_features, original.out_features, bias=False)
            projection.load_state_dict(original.state_dict())
            layer.output = projection
        else:
            model.state_encoder.graph_layers[1] = _NodewiseResidual(layer.norm, pair_seed=seed, role=role)
    model.spec = contract[role]
    model._architecture_contract.copy_(_stamp(dict(
        ablation=contract, candidate_id=config['candidate_id'], role=role)))
    return model


def actor_from_config(config):
    return _model_from_config(config, 'actor')


def critic_from_config(config):
    return _model_from_config(config, 'critic')
