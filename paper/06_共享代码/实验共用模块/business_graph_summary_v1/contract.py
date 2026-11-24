"""Frozen graph/fusion identity, independent of historical task-query runs."""
import hashlib
import json
from pathlib import Path

VERSION = 'business-graph-summary-009-v1'
CANDIDATES = ('s01_BG128',)
GROUPS = ('BG128',)
PLAN = Path(__file__).resolve().parents[1] / 'change_plans/token_attention_001/business_graph_summary_009'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def feature_config_for(candidate='s01_BG128'):
    from single_critic_task_query_v1.c_features import normalize_feature_config
    require(candidate in CANDIDATES, 'unknown business-graph candidate')
    return normalize_feature_config({'enabled': True})


def ablation_contract(candidate='s01_BG128'):
    from .graph import (NODE_NAMES, GROUP_NAMES, RELATION_NAMES, EDGE_FEATURE_NAMES,
                        QUANTITY_NAMES, QUANTITY_SCALES, RELATION_GROUP_IDS,
                        ROAD_ENDPOINTS, PHYSICAL_NODE_TO_TOKEN)
    require(candidate in CANDIDATES, 'unknown business-graph candidate')
    design = dict(version=VERSION, candidate=candidate, raw_dimension=1519,
        physical_nodes=18, scene_tokens=19, width=64, heads=4, graph_layers=2,
        node_names=list(NODE_NAMES), groups=list(GROUP_NAMES), relations=list(RELATION_NAMES),
        edge_feature_names=list(EDGE_FEATURE_NAMES), quantity_names=list(QUANTITY_NAMES),
        quantity_scales=list(QUANTITY_SCALES), relation_group_ids=list(RELATION_GROUP_IDS),
        road_endpoints=[list(pair) for pair in ROAD_ENDPOINTS],
        physical_node_to_token=list(PHYSICAL_NODE_TO_TOKEN),
        geographic_mapping=dict(load=[5,6,7,8], dump=[0,1], junction=[2,3,4]),
        road_representation='16_directed_attribute_edges_no_road_tokens',
        graph_weighting='within_group_attention_then_masked_group_concatenation',
        physical_residual=True, continuous_edge_score_and_value=True,
        facts='current_observed_targets_and_service_memberships',
        candidates='readout_only_never_counted_as_commitments',
        quantities='separate_typed_deterministic_channels_no_cross_relation_double_count',
        final_readout='direct_candidate_load_dump_and_typed_scene_pool',
        actor_queries=0, critic_queries=0, actor_scorer_shared=True,
        actor_critic_sharing=False, history=False, action_id_embedding=False,
        summary_enabled=True, summary_dimension=128,
        actor_summary_partition=dict(task=88, candidates=4, fields_per_candidate=10),
        critic_summary_partition='complete_C128',
        fusion='candidate_context_elementwise_interaction_v1',
        missing_data='validity_preserved_never_fill_missing_ID_with_zero',
        numerical_cache='current_raw_and_C_only_never_neural_activations')
    design['design_sha256'] = hashlib.sha256(json.dumps(design,sort_keys=True,
        separators=(',',':'),allow_nan=False).encode()).hexdigest()
    return design
