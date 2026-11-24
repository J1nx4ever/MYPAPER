"""Independent summary/Actor-memory control with one scalar composite-cost Critic.

All original observations and causal decision-summary prefixes are replayed
under current parameters. Old values, MC labels and GAE cost advantages are
frozen before either optimizer. This module never starts RPC or a simulator.
"""
from __future__ import annotations

from collections import defaultdict
import copy
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import time

import numpy as np
import torch
from torch import nn

from grade_event_memory_v1.events import build_memory
from grade_event_memory_v1.journal import TOKENIZATION
from grade_event_memory_v1.storage import OBSERVATION_SCHEMA
from grade_event_memory_v1.targets import cost_gae
from .targets import REWARD_VERSION, build_composite_targets, composite_contract, verify_episode_targets, validate_composite_contract
from .features import FEATURE_VERSION, FEATURE_CONFIG, feature_contract, physical_contract
from .models import (MaterialCostCritic, digest, actor_from_config, common_actor_digest,
                     normalize_feature_config, pack_summary_prefixes, physical_branch, pack_batch_prefixes)
from .features_cache import BatchFeatureCache, EXECUTION_VERSION
from .waiting_diagnostics import WAIT_FIELDS, validate_wait_breakdown


SCHEMA = 'physical-curriculum-control-ppo-v1'
CHECKPOINT_SCHEMA = 'physical-curriculum-control-ppo-checkpoint-v1'
ACTOR_SCHEMA = 'physical-curriculum-control-actor-v1'
CANDIDATES = ('a0_summary', 'a1_actor_gru')
PROFILES = ('real_smoke_2ep_v1', 'formal_32ep_v1', 'synthetic_test_v1')
PARENT_CHECKPOINT = Path(__file__).resolve().parents[1]/'runs/physical-realization-ppo-seed1-20261110-001/arms/wait01/updates/0008/checkpoint.pt'
PARENT_SHA256 = '9c4afa8dd0012ce8e2ffdcfbc208b242d31a50803e9330991d5bd924e9e32e9f'
COMMON_INITIAL_CHECKPOINT = Path(__file__).resolve().parents[1]/'runs/physical-realization-ppo-seed1-20261110-001/arms/production/initial.pt'
COMMON_INITIAL_SHA256 = '8421e826a6c16cf38fb1a9b3d8d9f538a3895d0f655e664aa8c1c32ff15fab42'
COMMON_INITIAL_SEED = 93468885541233
CONTINUATION_SEED = 58688234705239
STAGES = ('reference','wait015','wait02','fixed015','fixed02')
FIXED_STAGES = ('fixed015','fixed02')
WAIT_WEIGHTS = dict(reference=.1,wait015=.15,wait02=.2,fixed015=.15,fixed02=.2)
STAGE_LIMITS = dict(reference=0,wait015=32,wait02=32,fixed015=48,fixed02=48)
# Compatibility constant is the largest local budget; enforce STAGE_LIMITS.
STAGE_UPDATES = 48


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def _hash(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for data in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def implementation_signature():
    """Bind independently restored models to the exact numerical implementation."""
    project = Path(__file__).resolve().parents[1]
    sources = [Path(__file__),Path(__file__).with_name('models.py'),Path(__file__).with_name('features.py'),
               Path(__file__).with_name('features_cache.py'),Path(__file__).with_name('policy.py'),Path(__file__).with_name('targets.py')]
    sources += [Path(__file__).with_name(name) for name in ('admission.py','collection.py','storage.py','evaluation.py','waiting_diagnostics.py')]
    sources += [project/'u8_wait_efficiency_v1'/'waiting_diagnostics.py']
    sources += [project/'physical_realization_v1'/name for name in
                ('engine.py','observation.py','risk.py','representation.py')]
    sources += [project/'grade_commitment_ppo_v1'/name for name in ('features.py','models.py','learner.py')]
    sources += [project/'grade_event_memory_v1'/name for name in
                ('events.py','targets.py','sequence.py','journal.py','storage.py','admission.py')]
    sources += [project/'python'/name for name in
                ('constrained_ppo.py','constrained_adapter.py','observation_contract.py','observation_schema.json')]
    sources += [project.parents[1]/'python'/'episode_budget_ppo.py']
    files = {str(path.relative_to(project.parents[1])):_file_sha(path) for path in sources}
    return dict(execution_version=EXECUTION_VERSION,sources_sha256=files,implementation_sha256=_hash(files),
                torch_version=str(torch.__version__),numpy_version=np.__version__,
                python_version=list(sys.version_info[:3]))


def learned_parameter_digest(module):
    """Hash learned tensors only; execution/reward contract buffers are excluded."""
    h=hashlib.sha256()
    for name,value in module.named_parameters():
        array=value.detach().cpu().contiguous().numpy()
        h.update(_canonical(dict(name=name,dtype=str(array.dtype),shape=list(array.shape))))
        h.update(array.tobytes())
    return h.hexdigest()


def optimizer_digest(optimizer):
    """Stable exact Adam signature used by the objective-switch receipt."""
    h=hashlib.sha256()
    def visit(value):
        if torch.is_tensor(value):
            a=value.detach().cpu().contiguous().numpy()
            h.update(_canonical(dict(dtype=str(a.dtype),shape=list(a.shape))));h.update(a.tobytes())
        elif isinstance(value,dict):
            for key in sorted(value,key=str):h.update(str(key).encode());visit(value[key])
        elif isinstance(value,(list,tuple)):
            for item in value:visit(item)
        else:h.update(_canonical(value))
    visit(optimizer.state_dict());return h.hexdigest()


def _finite_tree(value):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value,dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value,(list,tuple)):
        return all(_finite_tree(item) for item in value)
    if isinstance(value,(float,np.floating)):
        return math.isfinite(value)
    return True


def _validate_optimizer(saved, expected, parameters):
    _require(isinstance(saved,dict) and set(saved)=={'state','param_groups'} and
             saved['param_groups']==expected['param_groups'], 'checkpoint Adam param groups/config differ')
    ids = [item for group in saved['param_groups'] for item in group['params']]
    _require(len(ids)==len(parameters) and set(saved['state'])<=set(ids) and _finite_tree(saved['state']),
             'checkpoint Adam state is nonfinite or has unknown parameters')
    for index,parameter in zip(ids,parameters):
        if index not in saved['state']:
            continue
        state=saved['state'][index]
        _require(set(state)=={'step','exp_avg','exp_avg_sq'} and
                 torch.is_tensor(state['step']) and state['step'].numel()==1 and
                 state['step'].item()>=0 and state['step'].item()==int(state['step'].item()) and
                 all(torch.is_tensor(state[key]) and state[key].shape==parameter.shape and
                     state[key].dtype==parameter.dtype for key in ('exp_avg','exp_avg_sq')) and
                 bool((state['exp_avg_sq']>=0).all()), 'checkpoint Adam moments/step shape differs')


def _validate_model_state(saved,module):
    expected=module.state_dict()
    _require(isinstance(saved,dict) and set(saved)==set(expected) and
             all(torch.is_tensor(saved[k]) and saved[k].shape==expected[k].shape and
                 saved[k].dtype==expected[k].dtype for k in expected) and _finite_tree(saved),
             'checkpoint model state schema/dtype is incompatible or nonfinite')
    _require(all(torch.equal(saved[k],expected[k]) for k in expected if k.startswith('_')),
             'checkpoint model feature/architecture contract buffer differs')


def _keys(value, expected, name):
    _require(isinstance(value, dict) and set(value) == set(expected), name + ' fields differ')


def default_config(*, source_kind='real_anylogic_rpc', seed=None, candidate_id='a1_actor_gru',
                   execution_profile='formal_32ep_v1', stage_id='wait015', wait_weight=None):
    _require(candidate_id in CANDIDATES and execution_profile in PROFILES, 'unknown candidate/profile')
    _require(stage_id in STAGES,'unknown sequential stage')
    if seed is None:seed=COMMON_INITIAL_SEED if stage_id in FIXED_STAGES else CONTINUATION_SEED
    expected_wait=WAIT_WEIGHTS[stage_id]
    _require(wait_weight is None or (type(wait_weight) in (int,float) and wait_weight==expected_wait),
             'stage reward weight differs')
    wait_weight=expected_wait
    episodes = 32 if execution_profile=='formal_32ep_v1' else 2
    return dict(schema=SCHEMA, source_kind=source_kind, seed=seed, reward_version=REWARD_VERSION,
                reward_contract=composite_contract(wait_weight=wait_weight),stage_id=stage_id,
                physical_contract=physical_contract(),physical_branch=physical_branch(),
                candidate_id=candidate_id,execution_profile=execution_profile,
                actor_advantage=dict(estimator='gae_cost'),feature_config=copy.deepcopy(FEATURE_CONFIG),
                network=dict(hidden_sizes=[128, 128], activation='relu', initialization='torch_default',
                             summary_embedding=32,gru_hidden=64,fusion_hidden=128,
                             history_projection_init='zeros'),
                memory_critic=dict(state_hidden_sizes=[128, 128], summary_embedding=32,
                                   gru_hidden=64, fusion_hidden=128),
                ppo=dict(gamma=1., gae_lambda=.95, epochs=4, episodes_per_batch=episodes,
                         minibatch_size=32, critic_prefix_minibatch=4, actor_lr=3e-4,
                         critic_lr=1e-3, adam_eps=1e-5, clip_range=.2, entropy_coef=.01,
                         max_grad_norm=.5, advantage_processing='standardize'),
                resources=dict(max_tokens=4096, max_prefix_bytes=256*1024*1024))


def validate_config(config):
    config = copy.deepcopy(config)
    template = default_config()
    _keys(config, template, 'learner config')
    _require(config['schema'] == SCHEMA and config['reward_version'] == REWARD_VERSION,
             'explicit composite schema/reward version required')
    _require(isinstance(config['reward_contract'],dict) and 'wait_weight' in config['reward_contract'],
             'explicit frozen branch reward contract required')
    validate_composite_contract(config['reward_contract'])
    _require(config['stage_id'] in STAGES and config['reward_contract']['wait_weight']==
             WAIT_WEIGHTS[config['stage_id']],'stage reward contract differs')
    _require(_canonical(config['physical_contract'])==_canonical(physical_contract()),'physical feature contract differs')
    physical_branch(config['physical_branch'])
    _require(config['source_kind'] in ('synthetic', 'real_anylogic_rpc'), 'unknown learner source kind')
    _require(config['candidate_id'] in CANDIDATES and config['execution_profile'] in PROFILES,
             'unknown candidate/profile')
    _require(config['actor_advantage']==dict(estimator='gae_cost'), 'Actor advantage estimator must remain GAE cost')
    config['feature_config']=normalize_feature_config(config['feature_config'])
    _require(config['source_kind']=='synthetic' or config['execution_profile']!='synthetic_test_v1',
             'synthetic test profile cannot admit real source')
    _require(type(config['seed']) is int and 0 < config['seed'] < 2**63, 'positive learner seed required')
    for key in ('network', 'memory_critic', 'ppo', 'resources'):
        _keys(config[key], template[key], key)
    network, memory, ppo, resources = (config[k] for k in ('network', 'memory_critic', 'ppo', 'resources'))
    for sizes in (network['hidden_sizes'], memory['state_hidden_sizes']):
        _require(isinstance(sizes, list) and len(sizes) == 2 and
                 all(type(n) is int and 0 < n <= 2048 for n in sizes), 'two explicit finite hidden widths required')
    _require(network['activation'] in ('relu', 'tanh') and network['initialization']=='torch_default' and
             network['history_projection_init']=='zeros', 'unsupported Actor specification/initialization')
    for spec in (network,memory):
        for name in ('summary_embedding', 'gru_hidden', 'fusion_hidden'):
            _require(type(spec[name]) is int and 0 < spec[name] <= 2048, 'invalid summary-memory architecture')
    for name in ('epochs', 'episodes_per_batch', 'minibatch_size', 'critic_prefix_minibatch'):
        _require(type(ppo[name]) is int and 0 < ppo[name] <= 4096, 'invalid batch/epoch size')
    for name in ('gamma', 'gae_lambda', 'actor_lr', 'critic_lr', 'adam_eps', 'clip_range',
                 'entropy_coef', 'max_grad_norm'):
        _require(type(ppo[name]) in (int, float) and math.isfinite(ppo[name]), 'nonfinite PPO coefficient')
    _require(ppo['gamma'] == 1 and ppo['gae_lambda'] == .95,
             'gamma1 and GAE lambda .95 are frozen in both candidates')
    _require(all(ppo[k] > 0 for k in ('actor_lr', 'critic_lr', 'adam_eps', 'max_grad_norm')) and
             0 < ppo['clip_range'] < 1 and ppo['entropy_coef'] >= 0, 'invalid optimizer coefficient')
    _require(ppo['advantage_processing'] in ('none', 'center', 'standardize'), 'explicit advantage processing required')
    _require(type(resources['max_tokens']) is int and 0 < resources['max_tokens'] <= 4096 and
             type(resources['max_prefix_bytes']) is int and 0 < resources['max_prefix_bytes'] <= 256*1024*1024,
             'causal prefix resource limits exceed contract')
    if config['source_kind'] == 'real_anylogic_rpc':
        _require(config['seed']==(COMMON_INITIAL_SEED if config['stage_id'] in FIXED_STAGES else CONTINUATION_SEED),
                 'real arm initialization/private PPO seed differs')
        _require(config['candidate_id']=='a1_actor_gru', 'real composite experiment freezes Actor and Critic history inputs')
        _require(network == template['network'] and memory == template['memory_critic'],
                 'formal Actor/Critic architecture is frozen')
    if config['execution_profile']!='synthetic_test_v1':
        expected_count=32 if config['execution_profile']=='formal_32ep_v1' else 2
        _require(ppo['epochs']==4 and ppo['episodes_per_batch']==expected_count,
                 'profile requires exact complete-shift count and four epochs')
    _canonical(config)
    return config


def _episode_key(episode):
    identity = episode['identity']
    return ':'.join(str(identity[k]) for k in ('recording_id', 'run_index', 'environment_seed', 'actor_sampling_seed'))


def require_independent_evaluation(episodes, learner):
    """Reject held-out seeds/runs that the local child has already trained on."""
    training = [source['identity'] for receipt in learner.consumed_batches.values()
                for source in receipt['sources']]
    for name in ('run_index', 'environment_seed', 'actor_sampling_seed'):
        used = {identity[name] for identity in training}
        _require(all(episode['identity'][name] not in used for episode in episodes),
                 'independent evaluation leaks a training '+name)


def _episode_fingerprint(episode):
    h = hashlib.sha256()
    h.update(_canonical(episode['identity']))
    for key in ('states', 'actions', 'old_log_probs', 'wait_deltas'):
        value = np.asarray(episode[key])
        h.update(key.encode()); h.update(str(value.dtype).encode()); h.update(_canonical(list(value.shape)))
        h.update(value.tobytes())
    h.update(_canonical(asdict(episode['memory'])))
    h.update(_canonical(asdict(episode['targets'])))
    h.update(_canonical(episode['evaluation']))
    h.update(_canonical(episode['composite_contract']))
    return h.hexdigest()


def _target_hash(targets):
    return _hash({name: value.tolist() for name, value in sorted(targets.items())})


def _immutable(value):
    value = np.asarray(value)
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


def _atomic_torch(path, value, *, metadata):
    """No overwrite; manifest and seal last. Partial publications cannot resume."""
    path = Path(path)
    manifest_path, seal_path = Path(str(path)+'.manifest.json'), Path(str(path)+'.seal.json')
    _require(not any(p.exists() for p in (path, manifest_path, seal_path)), 'checkpoint/artifact path already exists')
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.'+path.name+'.', dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(value, temporary)
        with open(temporary, 'rb') as handle:
            os.fsync(handle.fileno())
        sha = _file_sha(temporary)
        manifest = dict(metadata, sha256=sha, bytes=Path(temporary).stat().st_size)
        manifest_bytes = _canonical(manifest)
        os.link(temporary, path)
        for destination, data in ((manifest_path, manifest_bytes),
                                  (seal_path, _canonical(dict(sha256=sha, manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest())))):
            with destination.open('xb') as handle:
                handle.write(data); handle.flush(); os.fsync(handle.fileno())
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return dict(path=str(path.resolve()), sha256=sha, **metadata)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _load_checkpoint(path):
    path = Path(path)
    manifest_path, seal_path = Path(str(path)+'.manifest.json'), Path(str(path)+'.seal.json')
    _require(all(p.is_file() for p in (path, manifest_path, seal_path)), 'checkpoint publication is incomplete')
    manifest_bytes = manifest_path.read_bytes()
    manifest, seal = json.loads(manifest_bytes), json.loads(seal_path.read_bytes())
    _require(seal['manifest_sha256'] == hashlib.sha256(manifest_bytes).hexdigest() and
             seal['sha256'] == manifest['sha256'] == _file_sha(path) and path.stat().st_size == manifest['bytes'],
             'checkpoint byte/manifest seal differs')
    return torch.load(path, map_location='cpu', weights_only=False), manifest


class MultiObjectivePPO:
    def __init__(self, config):
        self.config = validate_config(config)
        self.config_sha256 = _hash(self.config)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.config['seed'])
            self.actor = actor_from_config(self.config)
        # Optional Actor memory consumes extra random draws; the Critic has an
        # independent initialization stream, identical across the two arms.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.config['seed']+1)
            self.critic = MaterialCostCritic(self.config['memory_critic'],feature_config=self.config['feature_config'],
                physical_spec=self.config['physical_branch'])
        p = self.config['ppo']
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=p['actor_lr'], eps=p['adam_eps'])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=p['critic_lr'], eps=p['adam_eps'])
        self.generator = torch.Generator().manual_seed(self.config['seed'])
        self.sampling_generator = torch.Generator().manual_seed(self.config['seed'] + 1)
        self.update_index = 0
        self.stage_update_index = 0
        self.stage_configs = {self.config['stage_id']:copy.deepcopy(self.config)}
        self.stage_transition = None
        self.consumed_batches = {}
        self.consumed_episodes = {}
        self.last_targets = {}
        self.last_target_hash = None
        self._actor_artifact = None
        self._failed_update = False
        self.runtime_state = {}
        self.progress_callback = None
        self._reference_only = self.config['stage_id']=='reference'
        self._dense_batch = None
        self._mini_cache = None
        self._summary_cache = BatchFeatureCache(feature_config=self.config['feature_config'],
            max_tokens=self.config['resources']['max_tokens'],max_bytes=self.config['resources']['max_prefix_bytes'])

    @property
    def reference_only(self):
        return self._reference_only

    @property
    def cumulative_update(self):
        return self.update_index+(0 if self.config['stage_id'] in FIXED_STAGES else 16)

    @property
    def policy_version(self):
        return digest(self.actor)

    def initialization_fingerprints(self):
        projection=self.actor.history_projection
        return dict(actor=self.policy_version,actor_common=common_actor_digest(self.actor),
                    critic=digest(self.critic),actor_history_projection_zero=
                    projection is None or bool((projection.weight.detach()==0).all()),
                    feature_contract=feature_contract(self.config['feature_config']),
                    physical_contract=physical_contract(),physical_heads_zero=all(bool((p==0).all())
                        for module in (self.actor,self.critic) for p in module.physical_head.parameters()))

    def _validate_episodes(self, episodes, *, training):
        episodes = tuple(episodes)
        _require(bool(episodes), 'nonempty full-shift batch required')
        if training:
            _require(len(episodes) == self.config['ppo']['episodes_per_batch'], 'complete configured shift count required')
        else:
            require_independent_evaluation(episodes, self)
        expected_source = 'synthetic' if self.config['source_kind'] == 'synthetic' else 'fixed_actor'
        keys, seeds, snapshots = set(), set(), set()
        for episode in episodes:
            _require(isinstance(episode, dict) and {'identity','states','actions','old_log_probs','memory','targets','evaluation','wait_deltas'} <= set(episode),
                     'sealed episode fields missing')
            identity, memory = episode['identity'], episode['memory']
            validate_wait_breakdown(episode['evaluation'],total_wait_seconds=episode['evaluation']['total_wait_seconds'])
            _require(identity['source_kind'] == expected_source, 'synthetic/native/real sources cannot mix')
            _require(identity['split'] == ('synthetic' if expected_source == 'synthetic' else ('train' if training else 'validation')),
                     'training/independent validation partition differs')
            _require(identity['reward_version'] == REWARD_VERSION and identity['observation_schema'] == OBSERVATION_SCHEMA and
                     identity['tokenization_version'] == TOKENIZATION, 'episode label/state/tokenization contract differs')
            contract=feature_contract(self.config['feature_config'])
            if expected_source!='synthetic' or 'decision_feature_version' in identity or 'decision_feature_config_sha256' in identity:
                _require(identity.get('decision_feature_version')==contract['version'] and
                         identity.get('decision_feature_config_sha256')==contract['config_sha256'],
                         'episode decision-summary contract differs')
            if expected_source!='synthetic' or 'physical_feature_contract_sha256' in identity:
                _require(identity.get('physical_feature_contract_sha256')==physical_contract()['contract_sha256'],
                         'episode physical feature contract differs')
            _require(identity['policy_version'] == self.policy_version, 'stale/off-policy Actor trajectory')
            _require(build_memory(memory.events) == memory and self._composite_targets(episode) == episode['targets'],
                     'causal memory or MC label certificate differs')
            n = len(memory.queries)
            states, actions, logs = (np.asarray(episode[k]) for k in ('states','actions','old_log_probs'))
            _require(states.shape == (n+1,1519) and states.dtype == np.dtype('float32') and np.isfinite(states).all() and
                     actions.shape == (n,) and actions.dtype.kind in 'iu' and ((0 <= actions)&(actions < 4)).all() and
                     logs.shape == (n,) and np.isfinite(logs).all(), 'full shift state/action/probability columns differ')
            _require(states[-1,17] == 1 and (states[:-1,17] == 0).all(), 'one true terminal observation required')
            mask = states[:-1,19:23]
            _require(((mask==0)|(mask==1)).all() and mask.astype(bool).any(axis=1).all() and
                     (mask[np.arange(n),actions]==1).all(), 'invalid action mask or illegal sampled action')
            with torch.no_grad():
                current,_ = self._policy_rows(episode)
            _require(np.isfinite(current.numpy()).all() and np.allclose(current.numpy(), logs, rtol=1e-6, atol=1e-6),
                     'old action probabilities do not match frozen current Actor/mask')
            key = _episode_key(episode)
            _require(key not in keys, 'duplicate complete shift in batch')
            keys.add(key)
            if training:
                _require(key not in self.consumed_episodes, 'episode was already consumed')
            run_key = (identity['recording_id'], identity['run_index'])
            _require(run_key not in seeds, 'duplicate run identity in batch')
            seeds.add(run_key); snapshots.add(identity['policy_sha256'])
            # Full immutable tape admission validates all original observations
            # and its largest prefix/resource bound once. Every later prefix
            # is a subset of this certified tape, not a truncated history.
            self._prefix(states[:-1],list(range(n)))
        _require(len(snapshots) == 1, 'one frozen Actor artifact is required per batch')
        if expected_source != 'synthetic':
            _require(self._actor_artifact is not None and self._actor_artifact['actor_digest'] == self.policy_version and
                     snapshots == {self._actor_artifact['sha256']}, 'formal batch must bind the saved current Actor byte SHA')
            _require(Path(self._actor_artifact['path']).is_file() and
                     _file_sha(self._actor_artifact['path'])==self._actor_artifact['sha256'],
                     'bound Actor snapshot bytes changed')
            checked=self._validated_actor_snapshot(self._actor_artifact['path'])
            _require(checked['sha256']==self._actor_artifact['sha256'], 'bound Actor snapshot seal changed')
        return episodes

    def _composite_targets(self, episode):
        _require(episode.get('composite_contract')==self.config['reward_contract'],
                 'episode reward branch differs from learner contract')
        return verify_episode_targets(episode)

    def _prefix(self, states, ids):
        return self._summary_cache.get(states).prefix(ids)

    def _policy_rows(self,episode):
        """Current logs/entropy on original causal prefixes, in query order."""
        states=np.asarray(episode['states'])[:-1]
        actions=np.asarray(episode['actions'])
        prefix=self._prefix(states,list(range(len(states))))
        distribution=self.actor.distribution(torch.tensor(states.copy(),dtype=torch.float32),prefix)
        return distribution.log_prob(torch.tensor(actions,dtype=torch.int64)),distribution.entropy()

    def _values(self, episodes):
        result = []
        with torch.no_grad():
            for episode in episodes:
                n = len(episode['actions']); ids=list(range(n))
                prefix=self._prefix(episode['states'][:-1],ids)
                states=torch.tensor(episode['states'][:-1].copy(),dtype=torch.float32)
                values=self.critic(states,prefix).numpy().tolist()
                result.append(np.asarray(values + [0.], dtype=np.float64))
        return result

    def predict_values(self, episode):
        """Read-only N+1 causal cost predictions, with true terminal exactly zero."""
        _require(build_memory(episode['memory'].events) == episode['memory'], 'causal memory certificate differs')
        states = np.asarray(episode['states'])
        _require(states.shape == (len(episode['memory'].queries)+1,1519) and np.isfinite(states).all(),
                 'predict_values requires finite complete-shift observations')
        return _immutable(self._values((episode,))[0])

    def _freeze_targets(self, episodes):
        old_values = self._values(episodes)
        snapshots = defaultdict(list)
        for episode, values in zip(episodes, old_values):
            n = len(episode['actions'])
            gae = cost_gae(episode['targets'].step_costs, values,
                           terminated=[False]*(n-1)+[True], truncated=[False]*n,
                           gae_lambda=self.config['ppo']['gae_lambda'])
            _require(np.allclose(gae.mc_returns, episode['targets'].mc_returns, rtol=0, atol=1e-10), 'MC target mismatch')
            advantages=gae.advantages
            lambda_returns=gae.lambda_returns
            for key, value in (('old_values',values[:-1]),('next_values',values[1:]),('advantages',advantages),
                               ('mc_returns',gae.mc_returns),('lambda_returns',lambda_returns),
                               ('costs',episode['targets'].step_costs),
                               ('grade_costs',episode['targets'].grade_costs),
                               ('production_costs',episode['targets'].production_costs),
                               ('wait_costs',episode['targets'].wait_costs)):
                snapshots[key].extend(value)
        advantages = np.asarray(snapshots['advantages'])
        processing = self.config['ppo']['advantage_processing']
        if processing in ('center','standardize'):
            advantages = advantages - advantages.mean()
        if processing == 'standardize':
            advantages = advantages/(advantages.std()+1e-8)
        snapshots['actor_advantages'] = advantages
        self.last_targets = {key: _immutable(np.asarray(value,dtype=np.float64)) for key,value in snapshots.items()}
        self.last_target_hash = _target_hash(self.last_targets)
        return self.last_targets

    def _mini_context(self,episodes,mapping,ids):
        # One immutable numerical context serves the Actor and Critic step.
        # No learned encoding or autograd graph is cached across Adam updates.
        if self._mini_cache is not None:
            old_episodes,old_mapping,old_ids,context=self._mini_cache
            if old_episodes is episodes and old_mapping is mapping and old_ids is ids:
                return context
        groups=defaultdict(list)
        for global_index in ids.tolist():
            ei,li=mapping[global_index];groups[ei].append((global_index,li))
        contexts=[];states=[];selected=[]
        for ei,group in groups.items():
            global_ids,local_ids=zip(*group)
            contexts.append(self._prefix(episodes[ei]['states'][:-1],list(local_ids)))
            if self._dense_batch is None:
                states.append(episodes[ei]['states'][list(local_ids)])
            selected.extend(global_ids)
        if self._dense_batch is not None:
            dense_episodes,dense_mapping,dense_states=self._dense_batch
            _require(dense_episodes is episodes and dense_mapping is mapping,'dense batch identity differs')
            selected_states=dense_states[selected]
        else:
            selected_states=torch.tensor(np.concatenate(states),dtype=torch.float32)
        context=(pack_batch_prefixes(contexts),selected_states,selected)
        self._mini_cache=(episodes,mapping,ids,context)
        return context

    def _critic_grad(self,episodes,mapping,ids,target):
        prefix,states,selected=self._mini_context(episodes,mapping,ids)
        predicted=self.critic(states,prefix)
        loss=(predicted-target[selected]).square().sum()/len(ids)
        _require(bool(torch.isfinite(loss)),'nonfinite Critic loss')
        loss.backward();return float(loss.detach())

    def _apply_fixed_rng_boundary(self):
        """Match the pre-existing global16 fork without resetting Adam or LR.

        Applied once after fixed-arm update16, before checkpoint publication
        and before the next Actor collection. A restored update16 checkpoint
        already contains the post-boundary private generators and receipt.
        """
        if self.config['stage_id'] not in FIXED_STAGES or self.update_index!=16:return
        _require('fixed_rng_boundary' not in self.runtime_state,'fixed RNG boundary was already applied')
        before={name:optimizer_digest(getattr(self,name+'_optimizer')) for name in ('actor','critic')}
        previous=hashlib.sha256(self.generator.get_state().numpy().tobytes()).hexdigest()
        self.generator.manual_seed(CONTINUATION_SEED)
        self.sampling_generator.manual_seed(CONTINUATION_SEED+1)
        _require(before=={name:optimizer_digest(getattr(self,name+'_optimizer')) for name in ('actor','critic')},
                 'private RNG boundary changed Adam')
        self.runtime_state['fixed_rng_boundary']=dict(completed_local_update=16,cumulative_update=16,
            seed=CONTINUATION_SEED,sampling_seed=CONTINUATION_SEED+1,optimizer_reset=False,
            learning_rates_changed=False,actor_adam_sha256=before['actor'],critic_adam_sha256=before['critic'],
            private_rng_before_sha256=previous,
            private_rng_after_sha256=hashlib.sha256(self.generator.get_state().numpy().tobytes()).hexdigest())

    def _actor_grad(self,episodes,mapping,ids,actions,old_logs,advantages):
        prefix,states,selected=self._mini_context(episodes,mapping,ids)
        distribution=self.actor.distribution(states,prefix)
        p=self.config['ppo']
        ratio=(distribution.log_prob(actions[selected])-old_logs[selected]).exp()
        unclipped=ratio*advantages[selected]
        clipped=ratio.clamp(1-p['clip_range'],1+p['clip_range'])*advantages[selected]
        loss=(torch.maximum(unclipped,clipped).sum()-p['entropy_coef']*distribution.entropy().sum())/len(ids)
        _require(bool(torch.isfinite(loss)),'nonfinite Actor loss')
        loss.backward();return float(loss.detach())

    def update(self, episodes, *, batch_id):
        _require(not self._reference_only, 'frozen reference is evaluation only; optimization is prohibited')
        _require(self.stage_update_index<STAGE_LIMITS[self.config['stage_id']],'stage update budget exhausted')
        _require(not self._failed_update, 'previous update failed; restore the last sealed checkpoint')
        _require(_hash(validate_config(self.config))==self.config_sha256, 'learner configuration changed after initialization')
        _require(isinstance(batch_id,str) and batch_id.strip() == batch_id and bool(batch_id), 'explicit batch ID required')
        _require(batch_id not in self.consumed_batches, 'batch was already consumed')
        self._summary_cache.clear()
        self._dense_batch=None;self._mini_cache=None
        started_at=time.monotonic(); completed_steps=0
        p=self.config['ppo']
        transition_count=sum(len(e['actions']) for e in episodes)
        total_steps=p['epochs']*math.ceil(transition_count/p['minibatch_size'])
        def progress(status,completed_epochs):
            if self.progress_callback is not None:
                self.progress_callback(dict(status=status,epoch=min(completed_epochs+1,p['epochs']),
                    completed_epochs=completed_epochs,epochs=p['epochs'],completed_steps=completed_steps,
                    total_steps=total_steps,elapsed_seconds=time.monotonic()-started_at))
        progress('update_begin',0)
        episodes = self._validate_episodes(episodes,training=True)
        frozen_policy = self.policy_version
        targets = self._freeze_targets(episodes)  # exactly once, before either optimizer
        x = torch.tensor(np.concatenate([e['states'][:-1] for e in episodes]),dtype=torch.float32)
        actions = torch.tensor(np.concatenate([e['actions'] for e in episodes]),dtype=torch.int64)
        old_logs = torch.tensor(np.concatenate([e['old_log_probs'] for e in episodes]),dtype=torch.float32)
        advantages = torch.tensor(targets['actor_advantages'],dtype=torch.float32)
        mc = torch.tensor(targets['mc_returns'],dtype=torch.float32)
        mapping = [(ei,li) for ei,e in enumerate(episodes) for li in range(len(e['actions']))]
        self._dense_batch=(episodes,mapping,x)
        p, losses = self.config['ppo'], []
        before_mse = float(np.mean((targets['old_values']-targets['mc_returns'])**2))
        # Any partial optimizer failure blocks continuation. Recovery is explicitly
        # from the last seal; silently retrying would consume a batch twice.
        self._failed_update = True
        for epoch in range(p['epochs']):
            for ids in torch.randperm(len(x),generator=self.generator).split(p['minibatch_size']):
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss=self._actor_grad(episodes,mapping,ids,actions,old_logs,advantages)
                actor_grad = nn.utils.clip_grad_norm_(self.actor.parameters(),p['max_grad_norm'],error_if_nonfinite=True)
                self.actor_optimizer.step()
                self.critic_optimizer.zero_grad(set_to_none=True)
                critic_loss = self._critic_grad(episodes,mapping,ids,mc)
                critic_grad = nn.utils.clip_grad_norm_(self.critic.parameters(),p['max_grad_norm'],error_if_nonfinite=True)
                self.critic_optimizer.step()
                self._mini_cache=None
                completed_steps+=1
                losses.append(dict(actor_loss=actor_loss,critic_mc_mse=critic_loss,
                                   actor_grad_norm=float(actor_grad),critic_grad_norm=float(critic_grad)))
            progress('epoch_end',epoch+1)
        self._dense_batch=None;self._mini_cache=None
        _require(_target_hash(self.last_targets) == self.last_target_hash, 'frozen training targets changed')
        _require(all(bool(torch.isfinite(t).all()) for t in list(self.actor.parameters())+list(self.critic.parameters())),
                 'nonfinite trained parameters')
        self.update_index += 1
        self.stage_update_index += 1
        self._apply_fixed_rng_boundary()
        source_records = [dict(episode_key=_episode_key(e),episode_sha256=_episode_fingerprint(e),
                               identity=copy.deepcopy(e['identity'])) for e in episodes]
        receipt = dict(batch_id=batch_id,update_index=self.update_index,global_update=self.update_index,
                       stage_id=self.config['stage_id'],stage_update_index=self.stage_update_index,
                       policy_before=frozen_policy,
                       policy_after=self.policy_version,config_sha256=self.config_sha256,
                       candidate_id=self.config['candidate_id'],execution_profile=self.config['execution_profile'],
                       actor_advantage=copy.deepcopy(self.config['actor_advantage']),
                       target_sha256=self.last_target_hash,sources=source_records)
        self.consumed_batches[batch_id] = receipt
        for source in source_records:
            self.consumed_episodes[source['episode_key']] = batch_id
        self._actor_artifact = None
        self._failed_update = False
        # Evaluation of the just-trained batch uses the frozen old log probabilities;
        # it is an update diagnostic, not an independent policy evaluation.
        metrics = self._metrics(episodes,old_logs=old_logs)
        progress('update_end',p['epochs'])
        return dict(receipt,episodes=len(episodes),transitions=len(x),epochs=p['epochs'],
                    actor_optimizer_steps=len(losses),critic_optimizer_steps=len(losses),
                    critic_mc_mse_before=before_mse,metrics=metrics,
                    mean={key:float(np.mean([row[key] for row in losses])) for key in losses[0]})

    def _metrics(self, episodes, *, old_logs=None):
        values = np.concatenate([v[:-1] for v in self._values(episodes)])
        target = np.concatenate([np.asarray(e['targets'].mc_returns) for e in episodes])
        if old_logs is None:
            old_logs = torch.tensor(np.concatenate([e['old_log_probs'] for e in episodes]),dtype=torch.float32)
        with torch.no_grad():
            policy_rows=[self._policy_rows(e) for e in episodes]
            delta = torch.cat([row[0] for row in policy_rows])-old_logs
            ratio = delta.exp()
            # Nonnegative sampled reverse-KL approximation (r-1)-log(r).
            kl = float(((ratio-1)-delta).mean())
            entropy = float(torch.cat([row[1] for row in policy_rows]).mean())
            clip = float(((ratio-1).abs()>self.config['ppo']['clip_range']).float().mean())
        variance = float(np.var(target))
        return dict(critic_mc_mse=float(np.mean((values-target)**2)),
                    critic_mc_bias=float(np.mean(values-target)),
                    explained_variance=None if variance <= 1e-12 else float(1-np.var(target-values)/variance),
                    approx_kl=kl,entropy=entropy,clip_fraction=clip,
                    grade_cost_mean=float(np.mean([sum(e['targets'].grade_costs) for e in episodes])),
                    production_cost_mean=float(np.mean([sum(e['targets'].production_costs) for e in episodes])),
                    wait_cost_mean=float(np.mean([sum(e['targets'].wait_costs) for e in episodes])),
                    total_cost_mean=float(np.mean([sum(e['targets'].step_costs) for e in episodes])),
                    negative_prediction_fraction=float(np.mean(values<0)),
                    waiting_breakdown_available_episodes=sum(e['evaluation'].get('waiting_breakdown_available',True)
                                                             for e in episodes),
                    waiting_breakdown_mean=({name:float(np.mean([e['evaluation'][name] for e in episodes]))
                                             for name in WAIT_FIELDS} if all(e['evaluation'].get(
                                             'waiting_breakdown_available',True) for e in episodes) else None),
                    true_terminal_values=0.,transitions=len(target))

    def evaluate(self, episodes):
        episodes = self._validate_episodes(episodes,training=False)
        return self._metrics(episodes)

    def save_actor(self, path):
        _require(not self._failed_update, 'cannot save a partially failed update')
        _require(_finite_tree(self.actor.state_dict()), 'cannot seal nonfinite Actor state')
        value = dict(schema=ACTOR_SCHEMA,config=copy.deepcopy(self.config),network=copy.deepcopy(self.config['network']),
                     actor=copy.deepcopy(self.actor.state_dict()),actor_digest=self.policy_version,
                     update_index=self.update_index,stage_id=self.config['stage_id'],
                     stage_update_index=self.stage_update_index,learner_config_sha256=self.config_sha256,
                     physical_contract=physical_contract(),
                     feature_contract=feature_contract(self.config['feature_config']),
                     candidate_id=self.config['candidate_id'],execution_profile=self.config['execution_profile'])
        artifact = _atomic_torch(path,value,metadata=dict(schema=ACTOR_SCHEMA,actor_digest=self.policy_version,
                                                        config_sha256=self.config_sha256,
                                                        candidate_id=self.config['candidate_id'],
                                                        execution_profile=self.config['execution_profile'],
                                                        update_index=self.update_index))
        self._actor_artifact = artifact
        return artifact

    def _validated_actor_snapshot(self, path):
        from .admission import frozen_actor
        saved,manifest=_load_checkpoint(path)
        _require(saved.get('schema')==manifest.get('schema')==ACTOR_SCHEMA,
                 'bound Actor artifact schema differs')
        _require(all(saved.get(k)==manifest.get(k)==self.config[k] for k in ('candidate_id','execution_profile')),
                 'bound Actor candidate/profile differs')
        _require(saved.get('config')==self.config and saved.get('network')==self.config['network'] and
                 saved.get('feature_contract')==feature_contract(self.config['feature_config']) and
                 saved.get('learner_config_sha256')==manifest.get('config_sha256')==self.config_sha256,
                 'bound Actor network/configuration differs')
        _require(type(saved.get('update_index')) is int and
                 saved['update_index']==manifest.get('update_index')==self.update_index,
                 'bound Actor update identity differs')
        _validate_model_state(saved['actor'],self.actor)
        actor, fingerprint = frozen_actor(path, allow_small_synthetic=self.config['source_kind']=='synthetic')
        _require(fingerprint == manifest.get('actor_digest') == self.policy_version,
                 'bound Actor snapshot is stale or differs')
        _require(_file_sha(path)==manifest['sha256'], 'bound Actor snapshot bytes changed during validation')
        return dict(path=str(Path(path).resolve()),sha256=manifest['sha256'],actor_digest=fingerprint)

    def bind_actor_snapshot(self, path):
        """Bind only a complete, sealed snapshot; never reserialize or overwrite."""
        self._actor_artifact = self._validated_actor_snapshot(path)
        return copy.deepcopy(self._actor_artifact)

    def save_checkpoint(self, path):
        _require(not self._failed_update, 'cannot seal a partially failed update')
        _require(_hash(validate_config(self.config))==self.config_sha256, 'learner configuration changed after initialization')
        _require(_finite_tree(self.actor.state_dict()) and _finite_tree(self.critic.state_dict()) and
                 _finite_tree(self.actor_optimizer.state_dict()) and _finite_tree(self.critic_optimizer.state_dict()),
                 'cannot seal nonfinite model/optimizer state')
        _canonical(self.runtime_state)
        value = dict(schema=CHECKPOINT_SCHEMA,config=copy.deepcopy(self.config),config_sha256=self.config_sha256,
                     reference_only=self._reference_only,stage_update_index=self.stage_update_index,
                     stage_configs=copy.deepcopy(self.stage_configs),stage_transition=copy.deepcopy(self.stage_transition),
                     physical_contract=physical_contract(),
                     candidate_id=self.config['candidate_id'],execution_profile=self.config['execution_profile'],
                     actor_advantage=copy.deepcopy(self.config['actor_advantage']),
                     actor=copy.deepcopy(self.actor.state_dict()),critic=copy.deepcopy(self.critic.state_dict()),
                     actor_optimizer=copy.deepcopy(self.actor_optimizer.state_dict()),
                     critic_optimizer=copy.deepcopy(self.critic_optimizer.state_dict()),
                     actor_digest=self.policy_version,critic_digest=digest(self.critic),update_index=self.update_index,
                     consumed_batches=copy.deepcopy(self.consumed_batches),consumed_episodes=copy.deepcopy(self.consumed_episodes),
                     last_targets={k:np.array(v,copy=True) for k,v in self.last_targets.items()},
                     last_target_hash=self.last_target_hash,actor_artifact=copy.deepcopy(self._actor_artifact),
                     runtime_state=copy.deepcopy(self.runtime_state),
                     implementation=implementation_signature(),
                     generator=self.generator.get_state(),sampling_generator=self.sampling_generator.get_state(),
                     python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),
                     observation_schema=OBSERVATION_SCHEMA,reward_version=REWARD_VERSION,feature_version=FEATURE_VERSION,
                     feature_contract=feature_contract(self.config['feature_config']),
                     tokenization_version=TOKENIZATION)
        return _atomic_torch(path,value,metadata=dict(schema=CHECKPOINT_SCHEMA,config_sha256=self.config_sha256,
                                                     candidate_id=self.config['candidate_id'],
                                                     execution_profile=self.config['execution_profile'],
                                                     actor_digest=self.policy_version,update_index=self.update_index))

    @classmethod
    def from_parent_checkpoint(cls,path,*,config,expected_sha256=PARENT_SHA256,reference_only=False):
        """Strict physical global16 model+full Adam fork; fresh local RNG/ledger."""
        from physical_realization_ppo_v1.learner import (MultiObjectivePPO as ParentPPO,
            CHECKPOINT_SCHEMA as PARENT_CHECKPOINT_SCHEMA)
        _require(type(reference_only) is bool,'explicit boolean reference-only marker required')
        child=cls(config)
        _require(child.config['stage_id'] not in FIXED_STAGES,'fixed arm must use common initial checkpoint')
        _require(reference_only==(child.config['stage_id']=='reference'),'reference-only arm identity differs')
        if child.config['source_kind']=='real_anylogic_rpc':
            _require(expected_sha256==PARENT_SHA256,'real fork requires frozen physical global16 parent')
        parent=ParentPPO.from_checkpoint(path,expected_sha256=expected_sha256,restore_global_rng=False)
        _require(parent.update_index==16 and parent.stage_update_index==8 and
                 parent.config['stage_id']=='wait01' and parent.config['reward_contract']['wait_weight']==.1,
                 'parent must be physical wait01 local8/global16')
        for key in ('candidate_id','source_kind','feature_config','physical_contract','physical_branch','network',
                    'memory_critic','ppo','resources','execution_profile','actor_advantage'):
            _require(child.config[key]==parent.config[key],'parent/child architecture or optimizer contract differs: '+key)
        for name in ('actor','critic'):
            source=dict(getattr(parent,name).named_parameters());module=getattr(child,name)
            destination=dict(module.named_parameters())
            _require(source.keys()==destination.keys() and all(source[k].shape==destination[k].shape and
                     source[k].dtype==destination[k].dtype for k in source),'physical named-parameter schema differs')
            with torch.no_grad():
                for key,value in destination.items():value.copy_(source[key])
            _require(learned_parameter_digest(module)==learned_parameter_digest(getattr(parent,name)),
                     'physical parameter fork was not exact')
            source_state=getattr(parent,name).state_dict();child_state=module.state_dict()
            _require(source_state.keys()==child_state.keys() and all(torch.equal(source_state[k],child_state[k])
                     for k in source_state if k.startswith('_') and k!='_execution_contract'),
                     'physical architecture/feature buffers differ')
            saved=getattr(parent,name+'_optimizer').state_dict()
            _validate_optimizer(saved,getattr(child,name+'_optimizer').state_dict(),list(module.parameters()))
            _require(len(saved['state'])==sum(len(g['params']) for g in saved['param_groups']) and
                     all(v['step'].item()>0 for v in saved['state'].values()),'complete parent Adam required')
            getattr(child,name+'_optimizer').load_state_dict(copy.deepcopy(saved))
        child._reference_only=reference_only
        child.runtime_state=dict(parent_checkpoint=dict(path=str(Path(path).resolve()),sha256=expected_sha256,
            config_sha256=parent.config_sha256,update_index=parent.update_index,stage_update_index=parent.stage_update_index,
            actor_digest=parent.policy_version,critic_digest=digest(parent.critic),source_seed=parent.config['seed'],
            checkpoint_schema=PARENT_CHECKPOINT_SCHEMA,reward_version=parent.config['reward_version'],
            learned_actor_sha256=learned_parameter_digest(child.actor),learned_critic_sha256=learned_parameter_digest(child.critic)),
            initialization=child.initialization_fingerprints(),optimizers_reset=False,reference_only=reference_only,
            transfer='physical_global16_full_parameters_and_adam_fresh_local_rng_and_ledgers')
        child.runtime_state['initialization'].update(initial_actor_digest=child.policy_version,
            initial_critic_digest=digest(child.critic),learned_actor_sha256=learned_parameter_digest(child.actor),
            learned_critic_sha256=learned_parameter_digest(child.critic))
        _require(_file_sha(path)==expected_sha256,'physical parent changed during fork')
        return child

    @classmethod
    def from_common_initial_checkpoint(cls,path,*,config,expected_sha256=COMMON_INITIAL_SHA256):
        """Copy the frozen grade-trained + zero-output46D common initial state."""
        from physical_realization_ppo_v1.learner import (MultiObjectivePPO as ParentPPO,
            CHECKPOINT_SCHEMA as PARENT_CHECKPOINT_SCHEMA)
        child=cls(config)
        _require(child.config['stage_id'] in FIXED_STAGES,'common initial is only for fixed-weight arms')
        if child.config['source_kind']=='real_anylogic_rpc':
            _require(expected_sha256==COMMON_INITIAL_SHA256,'fixed arm common checkpoint SHA differs')
        parent=ParentPPO.from_checkpoint(path,expected_sha256=expected_sha256,restore_global_rng=False)
        _require(parent.update_index==parent.stage_update_index==0 and parent.config['stage_id']=='production'
                 and parent.config['seed']==COMMON_INITIAL_SEED and not parent.consumed_batches
                 and not parent.consumed_episodes and not parent.last_targets,
                 'common initial must be the sealed post-grade pretraining physical initial')
        for key in ('candidate_id','source_kind','seed','feature_config','physical_contract','physical_branch','network',
                    'memory_critic','ppo','resources','execution_profile','actor_advantage'):
            _require(child.config[key]==parent.config[key],'common initial/child contract differs: '+key)
        for name in ('actor','critic'):
            source=dict(getattr(parent,name).named_parameters());module=getattr(child,name)
            destination=dict(module.named_parameters())
            _require(source.keys()==destination.keys() and all(source[k].shape==destination[k].shape and
                     source[k].dtype==destination[k].dtype for k in source),'common initial parameter schema differs')
            with torch.no_grad():
                for key,value in destination.items():value.copy_(source[key])
            _require(learned_parameter_digest(module)==learned_parameter_digest(getattr(parent,name)),
                     'common initial parameter transfer differs')
            source_state=getattr(parent,name).state_dict();child_state=module.state_dict()
            _require(source_state.keys()==child_state.keys() and all(torch.equal(source_state[k],child_state[k])
                     for k in source_state if k.startswith('_') and k!='_execution_contract'),
                     'common initial feature/architecture buffers differ')
            saved=getattr(parent,name+'_optimizer').state_dict()
            _validate_optimizer(saved,getattr(child,name+'_optimizer').state_dict(),list(module.parameters()))
            _require(not saved['state'],'common initial Adam must be empty')
            getattr(child,name+'_optimizer').load_state_dict(copy.deepcopy(saved))
        child.generator.set_state(parent.generator.get_state())
        child.sampling_generator.set_state(parent.sampling_generator.get_state())
        child.runtime_state=dict(parent_checkpoint=dict(path=str(Path(path).resolve()),sha256=expected_sha256,
            config_sha256=parent.config_sha256,update_index=0,stage_update_index=0,
            actor_digest=parent.policy_version,critic_digest=digest(parent.critic),source_seed=parent.config['seed'],
            checkpoint_schema=PARENT_CHECKPOINT_SCHEMA,reward_version=parent.config['reward_version'],
            learned_actor_sha256=learned_parameter_digest(child.actor),learned_critic_sha256=learned_parameter_digest(child.critic)),
            initialization=child.initialization_fingerprints(),optimizers_reset=False,reference_only=False,
            initial_optimizer_state='empty_common_initial',
            transfer='physical_common_initial_exact_parameters_empty_adam_and_private_rng')
        child.runtime_state['initialization'].update(initial_actor_digest=child.policy_version,
            initial_critic_digest=digest(child.critic),learned_actor_sha256=learned_parameter_digest(child.actor),
            learned_critic_sha256=learned_parameter_digest(child.critic))
        _require(_file_sha(path)==expected_sha256,'common initial changed during fork')
        return child

    @classmethod
    def from_checkpoint(cls, path, *, expected_config=None, expected_sha256=None, restore_global_rng=False):
        if expected_sha256 is not None:
            _require(_file_sha(path) == expected_sha256, 'checkpoint expected byte SHA differs')
        saved, manifest = _load_checkpoint(path)
        _require(saved.get('schema') == manifest.get('schema') == CHECKPOINT_SCHEMA, 'checkpoint schema differs')
        _require(saved['implementation']==implementation_signature(), 'checkpoint implementation/library source binding differs')
        config = validate_config(saved['config'])
        _require(all(saved.get(k)==manifest.get(k)==config[k] for k in ('candidate_id','execution_profile')) and
                 saved.get('actor_advantage')==config['actor_advantage'],
                 'checkpoint candidate/profile/estimator binding differs')
        _require(saved['config_sha256'] == manifest.get('config_sha256') == _hash(config), 'checkpoint config SHA differs')
        if expected_config is not None:
            _require(config == validate_config(expected_config), 'resume configuration differs')
        _require(saved.get('physical_contract')==physical_contract(),'checkpoint physical feature contract differs')
        _require(saved['observation_schema'] == OBSERVATION_SCHEMA and saved['reward_version'] == REWARD_VERSION and
                 saved['feature_version'] == FEATURE_VERSION and saved['tokenization_version'] == TOKENIZATION and
                 saved.get('feature_contract')==feature_contract(config['feature_config']),
                 'checkpoint input/reward/schema versions differ')
        learner = cls(config)
        _require(type(saved.get('reference_only')) is bool, 'checkpoint reference-only marker missing')
        learner._reference_only=saved['reference_only']
        _require(not learner._reference_only or saved['update_index']==0, 'reference checkpoint cannot contain optimizer updates')
        _validate_model_state(saved['actor'],learner.actor)
        _validate_model_state(saved['critic'],learner.critic)
        _validate_optimizer(saved['actor_optimizer'],learner.actor_optimizer.state_dict(),list(learner.actor.parameters()))
        _validate_optimizer(saved['critic_optimizer'],learner.critic_optimizer.state_dict(),list(learner.critic.parameters()))
        learner.actor.load_state_dict(saved['actor'],strict=True)
        learner.critic.load_state_dict(saved['critic'],strict=True)
        learner.actor_optimizer.load_state_dict(saved['actor_optimizer'])
        learner.critic_optimizer.load_state_dict(saved['critic_optimizer'])
        _require(learner.policy_version == saved['actor_digest'] == manifest.get('actor_digest') and
                 digest(learner.critic) == saved['critic_digest'], 'checkpoint model fingerprint differs')
        _require(type(saved['update_index']) is int and saved['update_index'] >= 0 and
                 saved['update_index'] == manifest.get('update_index'), 'checkpoint update identity differs')
        learner.update_index = saved['update_index']
        if config['source_kind']=='real_anylogic_rpc':
            proof=saved.get('runtime_state',{})
            initial=config['stage_id'] in FIXED_STAGES
            _require(proof.get('optimizers_reset') is False and proof.get('parent_checkpoint',{}).get('sha256')==
                     (COMMON_INITIAL_SHA256 if initial else PARENT_SHA256) and
                     proof['parent_checkpoint'].get('update_index')==(0 if initial else 16),
                     'real continuation/common initial parent proof differs')
            if initial:
                _require(proof.get('initial_optimizer_state')=='empty_common_initial',
                         'fixed arm empty common initial Adam proof missing')
        requires_full_adam=(learner.update_index>0 or
            (config['stage_id'] not in FIXED_STAGES and saved.get('runtime_state',{}).get('optimizers_reset') is False))
        if requires_full_adam:
            for name in ('actor','critic'):
                optimizer=getattr(learner,name+'_optimizer').state_dict()
                _require(len(optimizer['state'])==sum(len(g['params']) for g in optimizer['param_groups']) and
                         all(v['step'].item()>0 for v in optimizer['state'].values()),
                         'trained checkpoint requires complete Adam state')
        elif config['stage_id'] in FIXED_STAGES:
            _require(not learner.actor_optimizer.state and not learner.critic_optimizer.state,
                     'fixed initial checkpoint Adam must be empty')
        learner.stage_update_index=saved.get('stage_update_index')
        learner.stage_configs=copy.deepcopy(saved.get('stage_configs'))
        learner.stage_transition=copy.deepcopy(saved.get('stage_transition'))
        _require(type(learner.stage_update_index) is int and learner.stage_update_index==learner.update_index and
                 0<=learner.update_index<=STAGE_LIMITS[config['stage_id']] and learner.stage_configs=={config['stage_id']:config} and
                 learner.stage_transition is None,'independent arm cursor/config differs')
        _require(learner._reference_only==(config['stage_id']=='reference'),'reference-only checkpoint arm differs')
        learner.consumed_batches = copy.deepcopy(saved['consumed_batches'])
        learner.consumed_episodes = copy.deepcopy(saved['consumed_episodes'])
        _require(len(learner.consumed_batches) == learner.update_index, 'checkpoint batch/update ledger differs')
        reconstructed = {}
        stage_counts={stage:0 for stage in STAGES}
        for global_index,(batch_id, receipt) in enumerate(learner.consumed_batches.items(),1):
            stage=receipt.get('stage_id')
            _require(stage in learner.stage_configs,'checkpoint batch stage absent')
            stage_counts[stage]+=1
            receipt_config=learner.stage_configs[stage]
            _require(receipt['batch_id'] == batch_id and receipt['config_sha256'] == _hash(receipt_config) and
                     receipt.get('global_update')==receipt.get('update_index')==global_index and
                     stage==config['stage_id'] and
                     receipt.get('stage_update_index')==stage_counts[stage],
                     'checkpoint consumed-batch identity differs')
            if global_index==1:
                _require(isinstance(receipt.get('policy_before'),str),'initial policy chain missing')
            else:
                _require(receipt.get('policy_before')==previous_policy,'checkpoint policy chain differs')
            previous_policy=receipt.get('policy_after')
            _require(isinstance(previous_policy,str) and len(previous_policy)==64,'checkpoint policy chain missing')
            _require(all(receipt.get(k)==receipt_config[k] for k in ('candidate_id','execution_profile','actor_advantage')),
                     'checkpoint consumed-batch candidate/profile/estimator differs')
            for source in receipt['sources']:
                key = source['episode_key']
                _require(key not in reconstructed, 'checkpoint duplicate episode consumption')
                reconstructed[key] = batch_id
        _require(not learner.consumed_batches or previous_policy==learner.policy_version,
                 'checkpoint final policy chain differs')
        _require(stage_counts[config['stage_id']]==learner.stage_update_index, 'checkpoint local stage count differs')
        _require(reconstructed == learner.consumed_episodes, 'checkpoint episode-consumption ledger differs')
        learner.last_targets = {k:_immutable(v) for k,v in saved['last_targets'].items()}
        learner.last_target_hash = saved['last_target_hash']
        _require((not learner.last_targets and learner.last_target_hash is None and learner.update_index == 0) or
                 (bool(learner.last_targets) and _target_hash(learner.last_targets) == learner.last_target_hash and
                  learner.last_target_hash == next(reversed(learner.consumed_batches.values()))['target_sha256']),
                 'checkpoint frozen targets differ')
        learner._actor_artifact = copy.deepcopy(saved['actor_artifact'])
        _canonical(saved['runtime_state'])
        learner.runtime_state = copy.deepcopy(saved['runtime_state'])
        learner.generator.set_state(saved['generator'])
        learner.sampling_generator.set_state(saved['sampling_generator'])
        boundary=learner.runtime_state.get('fixed_rng_boundary')
        if config['stage_id'] in FIXED_STAGES and learner.update_index>=16:
            _require(isinstance(boundary,dict) and boundary.get('completed_local_update')==16 and
                     boundary.get('cumulative_update')==16 and boundary.get('seed')==CONTINUATION_SEED and
                     boundary.get('sampling_seed')==CONTINUATION_SEED+1 and boundary.get('optimizer_reset') is False and
                     boundary.get('learning_rates_changed') is False,'fixed arm RNG boundary receipt missing or differs')
            if learner.update_index==16:
                _require(boundary['private_rng_after_sha256']==hashlib.sha256(learner.generator.get_state().numpy().tobytes()).hexdigest()
                         and torch.equal(learner.generator.get_state(),torch.Generator().manual_seed(CONTINUATION_SEED).get_state())
                         and torch.equal(learner.sampling_generator.get_state(),torch.Generator().manual_seed(CONTINUATION_SEED+1).get_state()),
                         'fixed arm update16 private RNG differs from matched fork')
        else:
            _require(boundary is None,'premature or foreign fixed RNG boundary receipt')
        if restore_global_rng:
            random.setstate(saved['python_rng']); np.random.set_state(saved['numpy_rng']); torch.set_rng_state(saved['torch_rng'])
        return learner


# Compatibility API for the isolated campaign orchestrator.
GradeOnlyPPO = MultiObjectivePPO
PhysicalRealizationPPO = MultiObjectivePPO
PhysicalWaitTradeoffPPO = MultiObjectivePPO
