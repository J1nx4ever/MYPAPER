"""Independent frozen-Actor collection, with per-job reproducible sampling.

This entry point waits for manually started AnyLogic replications. It neither
launches the simulator nor constructs an optimizer, PPO object, or multiplier.
Raw RPC rewards are preserved; V1.1 reward labels belong to later admission.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from python.acceptance_recording import BoundaryRecorder, config_bytes
from python.collect_offline import validate_runtime, verify_environment
from python.constrained_batch import EpisodeBatchBuffer, collect_batch
from python.constrained_ppo import Actor, digest
from python.dispatch_bridge import DispatchCallbackBridge
from python.dispatch_env import DispatchEnv
from python.dispatch_service import create_server
from python.episode_runner import ParameterVariationEpisodeRunner
from python.experiment_config import source_pins
from python.protocol_artifacts import verify_protocol_artifacts

from .contracts import ROOT, check_config, read, require, sha, write
from .prepare import verify_bundle


def _rng_sha(generator):
    return hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest()


class FrozenJobActor:
    """The unchanged Actor with an independent CPU RNG reset for each job."""
    def __init__(self, actor, jobs):
        self.actor = copy.deepcopy(actor).cpu().eval()
        for parameter in self.actor.parameters():
            parameter.requires_grad_(False)
        self.policy_version = digest(self.actor)
        self.jobs = copy.deepcopy(jobs)
        require(bool(self.jobs), 'explicit sampling jobs required')
        identities = [(j['run_index'], j['experiment_seed']) for j in self.jobs]
        require(len(set(identities)) == len(identities), 'duplicate sampling job identity')
        require(all(type(j.get('actor_sampling_seed')) is int and
                    j['actor_sampling_seed'] >= 0 for j in self.jobs),
                'explicit per-job Actor sampling seed required')
        self.generator = None
        self.current_job = -1
        self.current_episode = None
        self.records = []
        self.last_metadata = None

    def assert_frozen(self):
        require(digest(self.actor) == self.policy_version and not self.actor.training and
                all(not p.requires_grad and p.device.type == 'cpu'
                    for p in self.actor.parameters()), 'sampling Actor changed')

    def _job(self, info):
        identity = (info['experiment_run_index'], info['experiment_seed'])
        if self.current_job < 0 or identity != (
                self.jobs[self.current_job]['run_index'],
                self.jobs[self.current_job]['experiment_seed']):
            index = self.current_job + 1
            require(index < len(self.jobs) and identity == (
                    self.jobs[index]['run_index'], self.jobs[index]['experiment_seed']),
                    'sampling job order or identity differs')
            require(info['decision_sequence'] == 0 and
                    info['episode_id'] not in {r['episode_id'] for r in self.records},
                    'sampling job must start with a new episode')
            self.current_job = index
            self.current_episode = info['episode_id']
            self.generator = torch.Generator(device='cpu')
            self.generator.manual_seed(self.jobs[index]['actor_sampling_seed'])
            self.records.append(dict(run_index=identity[0], experiment_seed=identity[1],
                episode_id=info['episode_id'],
                actor_sampling_seed=self.jobs[index]['actor_sampling_seed'],
                actions=0, initial_rng_sha256=_rng_sha(self.generator),
                final_rng_sha256=_rng_sha(self.generator)))
        require(info['episode_id'] == self.current_episode and
                info['decision_sequence'] == self.records[-1]['actions'],
                'sampling episode or decision sequence differs')

    def sample_with_info(self, observation, mask, info):
        self.assert_frozen()
        values = np.asarray(observation)
        require(values.dtype == np.float32 and values.shape == (1519,),
                'Actor sampling requires original float32 1519D input')
        require(np.array_equal(values[19:23], mask),
                'sampling mask differs from stored network input')
        self._job(info)
        before = _rng_sha(self.generator)
        with torch.no_grad():
            dist = self.actor.distribution(torch.as_tensor(values.copy())[None, :])
            # This is Categorical.sample's multinomial operation with a private
            # generator; probabilities/log_prob retain the existing Actor path.
            action = torch.multinomial(dist.probs, 1, replacement=True,
                                       generator=self.generator).reshape(1)
            log_probability = float(dist.log_prob(action).item())
        chosen = int(action.item())
        record = self.records[-1]
        self.last_metadata = dict(behavior_kind='frozen_actor', critic_only=True,
            actor_sampling_seed=record['actor_sampling_seed'],
            actor_sampling_draw_index=record['actions'],
            actor_rng_before_sha256=before,
            actor_rng_after_sha256=_rng_sha(self.generator))
        record['actions'] += 1
        record['final_rng_sha256'] = self.last_metadata['actor_rng_after_sha256']
        require(mask[chosen] and np.isfinite(log_probability) and log_probability <= 0,
                'invalid Actor action or selected log probability')
        return chosen, log_probability

    def row_metadata(self, info):
        require(self.last_metadata is not None and info['episode_id'] == self.current_episode
                and info['decision_sequence'] == self.last_metadata['actor_sampling_draw_index'],
                'sampling metadata is not attached to its original decision')
        return copy.deepcopy(self.last_metadata)

    def summary(self):
        self.assert_frozen()
        require(len(self.records) == len(self.jobs) and
                all(record['actions'] > 0 for record in self.records),
                'sampling job list is incomplete')
        return copy.deepcopy(self.records)


def make_actor_behavior(cfg):
    """Read an explicitly pinned initialization bundle, preserving global RNG."""
    require(type(cfg.get('cpu_threads')) is int and cfg['cpu_threads'] > 0,
            'explicit CPU thread count required for exact Actor replay')
    torch.set_num_threads(cfg['cpu_threads'])
    reference = cfg['actor_bundle']
    path = Path(reference['path'])
    require(sha(path) == reference['sha256'], 'Actor bundle hash differs')
    bundle = torch.load(path, map_location='cpu', weights_only=True)
    verify_bundle(bundle)
    require(bundle['seed'] == cfg['actor_seed'] and bundle['network'] == cfg['network'],
            'Actor bundle seed or network differs')
    require(bundle['actor_digest'] == cfg['behavior']['actor_digest'],
            'Actor configuration digest differs')
    with torch.random.fork_rng(devices=[]):
        actor = Actor(bundle['network'])
    actor.load_state_dict(bundle['actor'], strict=True)
    require(all(bool(torch.isfinite(value).all()) for value in actor.state_dict().values()),
            'Actor bundle contains nonfinite parameters')
    behavior = FrozenJobActor(actor, cfg['jobs'])
    require(behavior.policy_version == bundle['actor_digest'], 'loaded Actor digest differs')
    require(sha(path) == reference['sha256'], 'Actor bundle changed while loading')
    return behavior


def replay_actions(cfg, rows, episodes):
    """Exactly reproduce chosen actions, selected log probabilities and RNG IDs."""
    behavior = make_actor_behavior(cfg)
    require(len(episodes) == len(cfg['jobs']), 'Actor replay episode count differs')
    cursor = 0
    for episode, job in zip(episodes, cfg['jobs']):
        require(episode['start'] == cursor and episode['stop'] > cursor and
                episode['stop'] <= len(rows), 'Actor replay episode bounds differ')
        require(episode['experiment_run_index'] == job['run_index'] and
                episode['experiment_seed'] == job['experiment_seed'] and
                episode['scene_id'] == job['scene_id'] and
                episode['scene_group'] == job['scene_group'], 'Actor replay job differs')
        info = episode['start_info']
        for row in rows[episode['start']:episode['stop']]:
            chosen, lp = behavior.sample_with_info(row['observation'], row['mask'], info)
            require(chosen == row['action'] and lp == row['old_log_prob'] and
                    row['policy_version'] == behavior.policy_version,
                    'Actor action/log probability reproduction failed')
            metadata = behavior.row_metadata(info)
            require(all(row.get(key) == value for key, value in metadata.items()),
                    'Actor RNG or Critic-only metadata reproduction failed')
            info = row['settlement_info']
        cursor = episode['stop']
    require(cursor == len(rows), 'Actor replay has unassigned rows')
    return dict(actor_digest=behavior.policy_version, episodes=len(episodes),
        actions=len(rows), job_rng_replayed=True, action_rng=behavior.summary(),
        exact_action_log_prob=True, actor_updates=0, optimizer_updates=0,
        ppo_updates=0, multiplier_updates=0)


def preflight(config_path):
    """Check everything without output creation, sockets, or environment reset."""
    config_path = Path(config_path).resolve()
    cfg = check_config(config_path, 'collection')
    require(config_path.read_bytes() == config_bytes(cfg),
            'collection configuration must use exact canonical recorded bytes')
    verify_protocol_artifacts()
    verify_environment(cfg.get('environment_acceptance'), cfg['reward'])
    behavior = make_actor_behavior(cfg)
    behavior.assert_frozen()
    return cfg


def validate_actor_runtime(info, job):
    """The external Actor path must contain no native-policy echo metadata."""
    validate_runtime(info, job)
    require(not any(key.startswith('config.offline_native.')
                    for key in info['reward_components']),
            'native-policy RPC metadata is forbidden on the frozen Actor path')


def _fresh_paths(cfg):
    output = (ROOT / cfg['output']).resolve()
    require(output.is_relative_to(ROOT / 'runs') and output != ROOT / 'runs',
            'collection output must be a dedicated runs directory')
    require(not output.exists(), 'refusing collection output reuse')
    recording_id = cfg['recording']['recording_id']
    for old in (ROOT / 'runs').glob('**/recording_manifest.json'):
        require(read(old).get('recording_id') != recording_id,
                'recording_id already used in project runs')
    for job in cfg['jobs']:
        require(not Path(job['event_bundle']).exists(), 'refusing existing event identity')
    return output


def run(config_path):
    """Listen for this one approved partition and seal a candidate, never admit it."""
    config_path = Path(config_path).resolve()
    cfg = preflight(config_path)
    output = _fresh_paths(cfg)
    behavior = make_actor_behavior(cfg)
    baseline = dict(source_pins())
    baseline.update(cfg['source_pins'])
    baseline[str(config_path)] = sha(config_path)
    for job in cfg['jobs']:
        Path(job['event_bundle']).parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=False)
    recorder = buffer = env = server = None
    try:
        write(output / 'acceptance_run_config.json', cfg)
        require((output / 'acceptance_run_config.json').read_bytes() == config_path.read_bytes(),
                'recorded configuration bytes differ from approved configuration')
        write(output / 'source_snapshot.json', baseline)
        recorder = BoundaryRecorder(output, cfg, cfg['jobs'],
            source_kind='actual_anylogic_offline', source_sha256=baseline)
        buffer = EpisodeBatchBuffer(output / 'candidate',
            expected_episodes=len(cfg['jobs']), source_kind='real_anylogic_rpc')
        bridge = DispatchCallbackBridge(runtime_configuration=True,
            expected_episode_metadata=[(job['run_index'], job['experiment_seed'])
                                       for job in cfg['jobs']], evidence_sink=recorder)
        env = DispatchEnv(bridge, ParameterVariationEpisodeRunner(len(cfg['jobs'])),
                          cfg['callback_timeout_seconds'])
        server = create_server(bridge, host='127.0.0.1', port=50051)
        server.start()
        print('LISTENING ' + json.dumps(dict(address='127.0.0.1:50051',
            recording_id=cfg['recording']['recording_id'], config_sha256=sha(config_path),
            output=str(output), actor_digest=behavior.policy_version,
            partition=cfg['jobs'][0]['partition'], jobs=len(cfg['jobs']),
            simulator='start manually; no Actor/Critic/PPO/multiplier updates'),
            ensure_ascii=False), flush=True)
        collect_batch(env, behavior, buffer, cfg['jobs'], reward_settings=cfg['reward'],
                      recorder=recorder, runtime_validator=validate_actor_runtime)
        server.stop(grace=2).wait()
        server = None
        recorder.complete()
        replay = replay_actions(cfg, buffer.rows, buffer.episodes)
        for path, expected in baseline.items():
            current_path = Path(path) if Path(path).is_absolute() else ROOT / path
            require(sha(current_path) == expected, 'collection source changed: ' + path)
        summary = dict(contract='mining-fixed-actor-collection-v1.1',
            status='candidate_requires_event_finalization', formal_training_data=False,
            recording_id=cfg['recording']['recording_id'], config_sha256=sha(config_path),
            actor_seed=cfg['actor_seed'], actor_digest=behavior.policy_version,
            bundle_sha256=cfg['actor_bundle']['sha256'],
            episodes=len(buffer.episodes), transitions=len(buffer.rows),
            policy_version=behavior.policy_version, behavior=copy.deepcopy(cfg['behavior']),
            partition=cfg['jobs'][0]['partition'], action_rng=behavior.summary(),
            action_replay=replay, source_hashes_unchanged=True,
            full_evidence_recording=True, record_bc_dataset=False,
            capture_rollout_snapshots=False, raw_rpc_reward_unchanged=True,
            model_updates=0, actor_updates=0, critic_updates=0, optimizer_updates=0,
            ppo_updates=0, multiplier_updates=0, test_queries=0)
        write(output / 'collection_summary.json', summary)
        return summary
    except BaseException as error:
        if env is not None:
            env.close()
        if server is not None:
            server.stop(grace=2).wait()
            server = None
        if buffer is not None and not (output / 'candidate' / 'ABORTED.json').exists():
            buffer.abort(error)
        if recorder is not None:
            recorder.abort(error)
        write(output / 'ABORTED.json', dict(status='collection_aborted', trainable=False,
            error=str(error), exception_type=type(error).__name__,
            actor_updates=0, critic_updates=0, optimizer_updates=0,
            ppo_updates=0, multiplier_updates=0))
        raise
    finally:
        if env is not None:
            env.close()
        if server is not None:
            server.stop(grace=2).wait()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('preflight', 'run'))
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'preflight':
        cfg = preflight(args.config)
        result = dict(status='preflight_passed_no_listener',
            recording_id=cfg['recording']['recording_id'], config_sha256=sha(args.config),
            output=cfg['output'], actor_digest=cfg['behavior']['actor_digest'],
            jobs=len(cfg['jobs']), listener_started=False)
    else:
        result = run(args.config)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
