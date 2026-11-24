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
from .features import FEATURE_VERSION, FEATURE_CONFIG, feature_contract
from .models import (MaterialCostCritic, digest, actor_from_config, common_actor_digest,
                     normalize_feature_config, pack_summary_prefixes)
from .features_cache import BatchFeatureCache, EXECUTION_VERSION


SCHEMA = 'grade-multiobjective-ppo-v1'
CHECKPOINT_SCHEMA = 'grade-multiobjective-ppo-checkpoint-v1'
ACTOR_SCHEMA = 'grade-multiobjective-actor-v1'
CANDIDATES = ('a0_summary', 'a1_actor_gru')
PROFILES = ('real_smoke_2ep_v1', 'formal_32ep_v1', 'synthetic_test_v1')
GRADE_PARENT_CHECKPOINT = Path(__file__).resolve().parents[1]/'runs'/'grade-commitment-gru-ab-seed1-20261108-001'/'arms'/'a1_actor_gru'/'updates'/'0008'/'checkpoint.pt'
GRADE_PARENT_SHA256 = 'e457765c16afd2d90023c3e214ffa0d8e8da86aa0ede9b8d154cc38db4693bdf'


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


def default_config(*, source_kind='real_anylogic_rpc', seed=2026110903, candidate_id='a1_actor_gru',
                   execution_profile='formal_32ep_v1', wait_weight=0.):
    _require(candidate_id in CANDIDATES and execution_profile in PROFILES, 'unknown candidate/profile')
    episodes = 32 if execution_profile=='formal_32ep_v1' else 2
    return dict(schema=SCHEMA, source_kind=source_kind, seed=seed, reward_version=REWARD_VERSION,
                reward_contract=composite_contract(wait_weight=wait_weight),
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
            self.critic = MaterialCostCritic(self.config['memory_critic'],feature_config=self.config['feature_config'])
        p = self.config['ppo']
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=p['actor_lr'], eps=p['adam_eps'])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=p['critic_lr'], eps=p['adam_eps'])
        self.generator = torch.Generator().manual_seed(self.config['seed'])
        self.sampling_generator = torch.Generator().manual_seed(self.config['seed'] + 1)
        self.update_index = 0
        self.consumed_batches = {}
        self.consumed_episodes = {}
        self.last_targets = {}
        self.last_target_hash = None
        self._actor_artifact = None
        self._failed_update = False
        self.runtime_state = {}
        self.progress_callback = None
        self._summary_cache = BatchFeatureCache(feature_config=self.config['feature_config'],
            max_tokens=self.config['resources']['max_tokens'],max_bytes=self.config['resources']['max_prefix_bytes'])

    @property
    def policy_version(self):
        return digest(self.actor)

    def initialization_fingerprints(self):
        projection=self.actor.history_projection
        return dict(actor=self.policy_version,actor_common=common_actor_digest(self.actor),
                    critic=digest(self.critic),actor_history_projection_zero=
                    projection is None or bool((projection.weight.detach()==0).all()),
                    feature_contract=feature_contract(self.config['feature_config']))

    def _validate_episodes(self, episodes, *, training):
        episodes = tuple(episodes)
        _require(bool(episodes), 'nonempty full-shift batch required')
        if training:
            _require(len(episodes) == self.config['ppo']['episodes_per_batch'], 'complete configured shift count required')
        expected_source = 'synthetic' if self.config['source_kind'] == 'synthetic' else 'fixed_actor'
        keys, seeds, snapshots = set(), set(), set()
        for episode in episodes:
            _require(isinstance(episode, dict) and {'identity','states','actions','old_log_probs','memory','targets','evaluation','wait_deltas'} <= set(episode),
                     'sealed episode fields missing')
            identity, memory = episode['identity'], episode['memory']
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
            # Validate every future prefix up front, before either optimizer can run.
            for start in range(0, n, self.config['ppo']['critic_prefix_minibatch']):
                self._prefix(states[:-1],list(range(start,min(n,start+self.config['ppo']['critic_prefix_minibatch']))))
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

    def _critic_grad(self, episodes, mapping, ids, target):
        groups = defaultdict(list)
        for global_index in ids.tolist():
            episode_index, local_index = mapping[global_index]
            groups[episode_index].append((global_index,local_index))
        total = 0.
        for episode_index, group in groups.items():
            episode = episodes[episode_index]
            global_ids, local_ids = zip(*group)
            prefix = self._prefix(episode['states'][:-1], list(local_ids))
            states = torch.tensor(episode['states'][list(local_ids)],dtype=torch.float32)
            predicted = self.critic(states,prefix)
            loss = (predicted-target[list(global_ids)]).square().sum()/len(ids)
            _require(bool(torch.isfinite(loss)), 'nonfinite Critic loss')
            loss.backward(); total += float(loss.detach())
        return total

    def _actor_grad(self,episodes,mapping,ids,actions,old_logs,advantages):
        """Rebuild each full prefix with current weights; no stale hidden cache.

        Query order can be shuffled, but prefixes and zero initial hidden state
        retain their original complete-shift context. Microbatches accumulate
        one minibatch gradient before its single optimizer step.
        """
        groups=defaultdict(list)
        for global_index in ids.tolist():
            episode_index,local_index=mapping[global_index]
            groups[episode_index].append((global_index,local_index))
        p=self.config['ppo'];total=0.
        for episode_index,group in groups.items():
            episode=episodes[episode_index]
            global_ids,local_ids=zip(*group)
            prefix=self._prefix(episode['states'][:-1],list(local_ids))
            states=torch.tensor(episode['states'][list(local_ids)],dtype=torch.float32)
            distribution=self.actor.distribution(states,prefix)
            selected=list(global_ids)
            ratio=(distribution.log_prob(actions[selected])-old_logs[selected]).exp()
            unclipped=ratio*advantages[selected]
            clipped=ratio.clamp(1-p['clip_range'],1+p['clip_range'])*advantages[selected]
            loss=(torch.maximum(unclipped,clipped).sum()-
                  p['entropy_coef']*distribution.entropy().sum())/len(ids)
            _require(bool(torch.isfinite(loss)),'nonfinite Actor loss')
            loss.backward();total+=float(loss.detach())
        return total

    def update(self, episodes, *, batch_id):
        _require(not self._failed_update, 'previous update failed; restore the last sealed checkpoint')
        _require(_hash(validate_config(self.config))==self.config_sha256, 'learner configuration changed after initialization')
        _require(isinstance(batch_id,str) and batch_id.strip() == batch_id and bool(batch_id), 'explicit batch ID required')
        _require(batch_id not in self.consumed_batches, 'batch was already consumed')
        self._summary_cache.clear()
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
                completed_steps+=1
                losses.append(dict(actor_loss=actor_loss,critic_mc_mse=critic_loss,
                                   actor_grad_norm=float(actor_grad),critic_grad_norm=float(critic_grad)))
            progress('epoch_end',epoch+1)
        _require(_target_hash(self.last_targets) == self.last_target_hash, 'frozen training targets changed')
        _require(all(bool(torch.isfinite(t).all()) for t in list(self.actor.parameters())+list(self.critic.parameters())),
                 'nonfinite trained parameters')
        self.update_index += 1
        source_records = [dict(episode_key=_episode_key(e),episode_sha256=_episode_fingerprint(e),
                               identity=copy.deepcopy(e['identity'])) for e in episodes]
        receipt = dict(batch_id=batch_id,update_index=self.update_index,policy_before=frozen_policy,
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
                    explained_variance=None if variance <= 1e-12 else float(1-np.var(target-values)/variance),
                    approx_kl=kl,entropy=entropy,clip_fraction=clip,
                    grade_cost_mean=float(np.mean([sum(e['targets'].grade_costs) for e in episodes])),
                    production_cost_mean=float(np.mean([sum(e['targets'].production_costs) for e in episodes])),
                    wait_cost_mean=float(np.mean([sum(e['targets'].wait_costs) for e in episodes])),
                    total_cost_mean=float(np.mean([sum(e['targets'].step_costs) for e in episodes])),
                    negative_prediction_fraction=float(np.mean(values<0)),
                    true_terminal_values=0.,transitions=len(target))

    def evaluate(self, episodes):
        episodes = self._validate_episodes(episodes,training=False)
        return self._metrics(episodes)

    def save_actor(self, path):
        _require(not self._failed_update, 'cannot save a partially failed update')
        _require(_finite_tree(self.actor.state_dict()), 'cannot seal nonfinite Actor state')
        value = dict(schema=ACTOR_SCHEMA,config=copy.deepcopy(self.config),network=copy.deepcopy(self.config['network']),
                     actor=copy.deepcopy(self.actor.state_dict()),actor_digest=self.policy_version,
                     update_index=self.update_index,learner_config_sha256=self.config_sha256,
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
    def from_parent_checkpoint(cls, path, *, config, expected_sha256):
        """Transfer learned weights from the completed historical grade-only B.

        The original loader verifies its own sealed checkpoint, configuration,
        architecture/feature buffers, implementation, optimizer and audit ledger.
        We then copy *only named parameters* into a fresh composite-cost model.
        Child execution/contract buffers, optimizers, RNG and local ledgers are
        rebuilt from its new config; no historical file is opened for writing.
        """
        from grade_commitment_ppo_v1.learner import (GradeOnlyPPO as HistoricalPPO,
            CHECKPOINT_SCHEMA as HISTORICAL_CHECKPOINT_SCHEMA)
        _require(isinstance(expected_sha256,str) and len(expected_sha256)==64 and
                 all(c in '0123456789abcdef' for c in expected_sha256),
                 'explicit parent checkpoint byte SHA256 required')
        child=cls(config)
        if child.config['source_kind']=='real_anylogic_rpc':
            _require(expected_sha256==GRADE_PARENT_SHA256,
                     'real branch must use the approved historical grade-only B update 8 byte SHA')
        parent=HistoricalPPO.from_checkpoint(path,expected_sha256=expected_sha256)
        _require(parent.update_index==8, 'parent must be the historical grade-only update 8 checkpoint')
        _require(parent.config['candidate_id']=='a1_actor_gru' and
                 parent.config['reward_version']=='grade-only-alp-clip-linear-v1',
                 'parent must be the completed grade-only Actor/Critic GRU B model')
        for key in ('candidate_id','source_kind','feature_config','network','memory_critic'):
            _require(child.config[key]==parent.config[key],
                     'parent and branch observation/network/source contracts differ')
        _require(feature_contract(child.config['feature_config'])==feature_contract(parent.config['feature_config']),
                 'parent and branch physical feature contracts differ')
        for name in ('actor','critic'):
            source=dict(getattr(parent,name).named_parameters())
            destination=dict(getattr(child,name).named_parameters())
            _require(source.keys()==destination.keys() and
                     all(source[k].shape==destination[k].shape and source[k].dtype==destination[k].dtype
                         for k in source), 'parent learned-parameter schema differs')
            with torch.no_grad():
                for key,value in destination.items():
                    value.copy_(source[key])
            _require(learned_parameter_digest(getattr(parent,name))==learned_parameter_digest(getattr(child,name)),
                     'learned parameter transfer was not exact')
        actor_weights=learned_parameter_digest(child.actor)
        critic_weights=learned_parameter_digest(child.critic)
        child.runtime_state=dict(parent_checkpoint=dict(
            path=str(Path(path).resolve()),sha256=expected_sha256,
            config_sha256=parent.config_sha256,update_index=parent.update_index,
            checkpoint_schema=HISTORICAL_CHECKPOINT_SCHEMA,reward_version=parent.config['reward_version'],
            source_seed=parent.config['seed'],actor_digest=parent.policy_version,critic_digest=digest(parent.critic),
            learned_actor_sha256=actor_weights,learned_critic_sha256=critic_weights),
            initialization=dict(initial_actor_digest=child.policy_version,initial_critic_digest=digest(child.critic),
                learned_actor_sha256=actor_weights,learned_critic_sha256=critic_weights),
            transfer='historical_grade_only_learned_parameters_only_fresh_optimizers_rng_and_ledgers')
        _require(_file_sha(path)==expected_sha256, 'parent checkpoint changed during parameter transfer')
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
        _require(saved['observation_schema'] == OBSERVATION_SCHEMA and saved['reward_version'] == REWARD_VERSION and
                 saved['feature_version'] == FEATURE_VERSION and saved['tokenization_version'] == TOKENIZATION and
                 saved.get('feature_contract')==feature_contract(config['feature_config']),
                 'checkpoint input/reward/schema versions differ')
        learner = cls(config)
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
        learner.consumed_batches = copy.deepcopy(saved['consumed_batches'])
        learner.consumed_episodes = copy.deepcopy(saved['consumed_episodes'])
        _require(len(learner.consumed_batches) == learner.update_index, 'checkpoint batch/update ledger differs')
        reconstructed = {}
        for batch_id, receipt in learner.consumed_batches.items():
            _require(receipt['batch_id'] == batch_id and receipt['config_sha256'] == learner.config_sha256,
                     'checkpoint consumed-batch identity differs')
            _require(all(receipt.get(k)==config[k] for k in ('candidate_id','execution_profile','actor_advantage')),
                     'checkpoint consumed-batch candidate/profile/estimator differs')
            for source in receipt['sources']:
                key = source['episode_key']
                _require(key not in reconstructed, 'checkpoint duplicate episode consumption')
                reconstructed[key] = batch_id
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
        if restore_global_rng:
            random.setstate(saved['python_rng']); np.random.set_state(saved['numpy_rng']); torch.set_rng_state(saved['torch_rng'])
        return learner


# Compatibility API for the isolated campaign orchestrator.
GradeOnlyPPO = MultiObjectivePPO
