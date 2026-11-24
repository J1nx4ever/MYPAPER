"""Exact named CPU generator streams for cross-arm, same-role initialization."""
import hashlib
import json
import math
import torch
from torch import nn

from .contract import require

SCHEME = 's1519-ab-init-v1'


def parameter_seed(pair_seed, role, name, shape):
    require(type(pair_seed) is int and 0 <= pair_seed < 2**63, 'invalid initialization pair_seed')
    require(role in ('actor', 'critic'), 'initialization role must be actor or critic')
    payload = dict(scheme=SCHEME, pair_seed=pair_seed, role=role, parameter=name, shape=list(shape))
    data = json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    return int.from_bytes(hashlib.sha256(data).digest()[:8], 'big') & (2**63-1)


def initialize(module, pair_seed, role):
    """Initialize existing parameters only; return a compact reproducible manifest.

    Root module names must be canonical and contain no comparison-arm wrapper.
    All parameters use independent CPU streams, so creating G2 score modules
    cannot change any common U2/G0 parameter. Actor and Critic have separate
    streams and remain separate objects. Global torch RNG is never consumed.
    """
    require(isinstance(module, nn.Module), 'initialize requires a module')
    owners = dict(module.named_modules())
    parameters = list(module.named_parameters())
    # Reject aliases rather than letting named_parameters silently hide a tie.
    all_parameters = list(module.named_parameters(remove_duplicate=False))
    require(len(parameters) == len(all_parameters), 'initialization forbids shared parameter objects')
    pointers = [p.untyped_storage().data_ptr() for _, p in parameters]
    require(len(pointers) == len(set(pointers)), 'initialization forbids aliased parameter storage')
    specifications = []
    for name, parameter in parameters:
        require(parameter.dtype == torch.float32 and parameter.device.type == 'cpu',
                'initialization requires CPU float32 parameters')
        require(parameter.is_contiguous(), 'initialization requires canonical contiguous parameter layout')
        parent_name, _, local_name = name.rpartition('.')
        owner = owners[parent_name]
        if isinstance(owner, nn.Linear):
            require(local_name in ('weight', 'bias'), 'unknown Linear parameter')
            expected_shape = (owner.out_features, owner.in_features) if local_name == 'weight' else (owner.out_features,)
            require(tuple(parameter.shape) == expected_shape, 'Linear parameter shape differs from fan_in contract')
            bound = 1/math.sqrt(owner.in_features)
            specification = ('uniform', bound)
        elif isinstance(owner, nn.LayerNorm):
            require(local_name in ('weight', 'bias'), 'unknown LayerNorm parameter')
            specification = ('constant', 1. if local_name == 'weight' else 0.)
        elif name.split('.')[-1] == 'semantic_type_embedding':
            require(tuple(parameter.shape) == (5, 64), 'semantic type shape must be [5,64]')
            specification = ('normal', None)
        elif isinstance(owner, nn.Embedding):
            require(tuple(parameter.shape) == (5, 64), 'only the semantic type embedding is permitted')
            specification = ('normal', None)
        elif local_name == 'score_a':
            require(tuple(parameter.shape) == (4, 16), 'score_a shape must be [4,16]')
            specification = ('uniform', 0.25)
        else:
            raise ValueError('unclassified initialization parameter: ' + name)
        specifications.append((name, parameter, specification))
    manifest = []
    with torch.no_grad():
        for name, parameter, (distribution, argument) in specifications:
            seed = parameter_seed(pair_seed, role, name, parameter.shape)
            generator = torch.Generator(device='cpu').manual_seed(seed)
            if distribution == 'uniform':
                parameter.uniform_(-argument, argument, generator=generator)
            elif distribution == 'normal':
                parameter.normal_(0., 1., generator=generator)
            else:
                parameter.fill_(argument)
            manifest.append(dict(parameter=name, shape=list(parameter.shape), seed=seed,
                                 distribution=distribution, argument=argument,
                                 sha256=hashlib.sha256(parameter.detach().numpy().tobytes(order='C')).hexdigest()))
    return manifest
