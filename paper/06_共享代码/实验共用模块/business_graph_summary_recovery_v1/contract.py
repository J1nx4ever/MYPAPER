"""Single-seed, single-arm campaign around the frozen BG128 architecture."""
from pathlib import Path
from business_graph_summary_v1.contract import (CANDIDATES, GROUPS, VERSION,
    require, ablation_contract, feature_config_for)
from single_critic_task_query_v1.contract import seed, parameter_digest
PLAN = Path(__file__).resolve().parents[1] / 'change_plans/token_attention_001/business_graph_summary_009/formal_training_001'

def group(candidate):
    require(candidate in CANDIDATES, 'unknown BG128 candidate')
    return 'BG128'

def updates(candidate):
    group(candidate)
    return 16

def global_update(candidate, local):
    group(candidate)
    require(type(local) is int and 0 <= local <= 16, 'invalid local update')
    return local

def schedule(engineering=False):
    require(type(engineering) is bool, 'explicit schedule mode required')
    if engineering:
        return [('train_u0001', 'train', 1, 5)]
    return [(f'train_u{u:04d}', 'train', u, 32) for u in range(1,17)]
