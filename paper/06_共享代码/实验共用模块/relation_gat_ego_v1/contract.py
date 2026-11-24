"""Three representations, one fixed joint objective, one paired training seed."""
import hashlib
import json
from pathlib import Path

GROUPS = ('G0', 'U2', 'G2')
CANDIDATES = tuple('s01_' + name for name in GROUPS)
VERSION = 'relation-gat-ego-003'
S_PARAMETERS = dict(G0=92048, U2=116112, G2=136336)
PLAN = Path(__file__).resolve().parents[1] / 'change_plans/token_attention_001/relation_gat_ego_003'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def group(candidate):
    require(candidate in CANDIDATES, 'unknown relation graph candidate')
    return candidate[4:]


def ablation_contract(candidate):
    mode = group(candidate)
    return dict(version=VERSION, group=mode, physical_nodes=31, scene_tokens=32,
        width=64, heads=4, graph_layers=0 if mode == 'G0' else 2,
        graph_weighting={'G0':'strict_bypass','U2':'incoming_record_uniform','G2':'gatv2_style'}[mode],
        final_readout='ego_attention_and_graph_updated_load_actions',
        S_parameters=S_PARAMETERS[mode], actor_critic_sharing=False,
        history=False, physical_residual=False,
        initialization='s1519-ab-init-v1',
        design_sha256=hashlib.sha256((PLAN/'contract.json').read_bytes()).hexdigest(),
        edge_schema_sha256=hashlib.sha256((PLAN/'edge_schema.json').read_bytes()).hexdigest())


def updates(candidate):
    group(candidate)
    return 16


def global_update(candidate, local):
    group(candidate)
    require(type(local) is int and 0 <= local <= 16, 'invalid local update')
    return 0 if local == 0 else (local - 1) * 3 + CANDIDATES.index(candidate) + 1


def schedule(engineering=False):
    require(type(engineering) is bool, 'explicit schedule mode required')
    if engineering:
        return [('train_u0001','train',1,5)]
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
