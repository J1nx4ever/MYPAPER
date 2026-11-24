"""Paired task-query C128 ablation; both arms use the same two relation GATs."""
import hashlib
import json
from pathlib import Path

GROUPS = ('TQ0', 'TQ128')
CANDIDATES = tuple('s01_' + name for name in GROUPS)
VERSION = 'single-critic-task-query-008-v1'
S_PARAMETERS = dict(actor=161360, critic=161168)
PLAN = Path(__file__).resolve().parents[1] / 'change_plans/token_attention_001/single_critic_task_query_008'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def group(candidate):
    require(candidate in CANDIDATES, 'unknown task-query candidate')
    return candidate[4:]


def feature_config_for(candidate):
    from .c_features import normalize_feature_config
    return normalize_feature_config({'enabled': group(candidate) == 'TQ128'})


def ablation_contract(candidate):
    mode = group(candidate)
    design = dict(version=VERSION, group=mode, physical_nodes=31, scene_tokens=32,
        width=64, heads=4, graph_layers=2, graph_weighting='gatv2_style',
        final_readout='shared_action_queries_and_global_ego_value_query',
        retrieval='all_32_entities_no_extra_relation_bias',
        actor_queries=4, critic_queries=1, actor_query_input=130, critic_query_input=128,
        actor_query_mlp=[130,128,64], critic_query_mlp=[128,128,64],
        actor_scorer_shared=True, action_id_embedding=False,
        S_parameters=dict(S_PARAMETERS),
        total_parameters=dict(actor=190321 if mode == 'TQ128' else 182097,
                              critic=186033 if mode == 'TQ128' else 177809),
        actor_critic_sharing=False, history=False, physical_residual=False,
        summary_enabled=mode == 'TQ128', summary_dimension=128 if mode == 'TQ128' else 0,
        summary_embedding=32 if mode == 'TQ128' else 0,
        initialization='s1519-ab-init-v1',
        edge_schema_sha256=hashlib.sha256((Path(__file__).resolve().parents[1] /
            'change_plans/token_attention_001/relation_gat_ego_003/edge_schema.json').read_bytes()).hexdigest())
    design['design_sha256'] = hashlib.sha256(json.dumps(design,sort_keys=True,
        separators=(',',':')).encode()).hexdigest()
    return design


def updates(candidate):
    group(candidate)
    return 16


def global_update(candidate, local):
    group(candidate)
    require(type(local) is int and 0 <= local <= 16, 'invalid local update')
    return 0 if local == 0 else (local - 1) * len(CANDIDATES) + CANDIDATES.index(candidate) + 1


def schedule(engineering=False):
    require(type(engineering) is bool, 'explicit schedule mode required')
    if engineering:
        return [('train_u0001', 'train', 1, 5)]
    return [(f'train_u{u:04d}', 'train', u, 32) for u in range(1,17)] + [
        ('final_u0016', 'final', 16, 100)]


def seed(namespace, kind, ordinal):
    return int.from_bytes(hashlib.sha256(f'{namespace}:{kind}:{ordinal}'.encode()).digest()[:6], 'big') + 1


def parameter_digest(module):
    digest = hashlib.sha256()
    for name, value in module.named_parameters():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()
