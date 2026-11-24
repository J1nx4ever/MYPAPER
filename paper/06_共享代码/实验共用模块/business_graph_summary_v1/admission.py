"""Load a sealed BG009 Actor. This module starts no RPC or simulation."""
import torch
from .contract import require, ablation_contract
from .c_features import feature_contract


def frozen_actor(path, *, allow_small_synthetic=False):
    from .learner import (ACTOR_SCHEMA, validate_config, _validate_model_state,
                          _load_checkpoint, _hash, implementation_signature)
    from .models import actor_from_config, digest
    saved, manifest = _load_checkpoint(path)
    require(isinstance(saved,dict) and saved.get('schema') == manifest.get('schema') == ACTOR_SCHEMA,
            'sealed business-graph Actor artifact required')
    cfg = validate_config(saved['config'])
    require(allow_small_synthetic or cfg['source_kind'] == 'real_anylogic_rpc',
            'real Actor cannot use a synthetic configuration')
    require(saved.get('implementation') == implementation_signature(), 'Actor implementation differs')
    require(saved.get('graph_contract') == cfg['graph_contract'] == ablation_contract(cfg['candidate_id']),
            'Actor graph identity differs')
    require(saved.get('learner_config_sha256') == manifest.get('config_sha256') == _hash(cfg)
            and saved.get('network') == cfg['network']
            and saved.get('feature_contract') == feature_contract(cfg['feature_config'])
            and all(saved.get(k) == manifest.get(k) == cfg[k]
                    for k in ('candidate_id','execution_profile')), 'Actor configuration differs')
    require(type(saved.get('update_index')) is int and saved['update_index'] >= 0
            and saved['update_index'] == manifest.get('update_index'),
            'Actor update identity differs')
    with torch.random.fork_rng(devices=[]):
        actor = actor_from_config(cfg)
    _validate_model_state(saved['actor'],actor)
    actor.load_state_dict(saved['actor'],strict=True)
    actor.eval()
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    fingerprint = digest(actor)
    require(saved.get('actor_digest') == manifest.get('actor_digest') == fingerprint,
            'Actor fingerprint differs')
    actor._artifact_config = cfg
    return actor, fingerprint
