"""Collect one 32-shift AnyLogic B batch, or verify it and perform its single cost-PPO update."""
import argparse
import fcntl
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from statistics import fmean
import sys
from tempfile import TemporaryDirectory
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parent / 'capacity_scaled_001')]
import trial_service
from trial_service import digest, encode, require
from scaled_service import ScaledTrial, check_observed_targets
import campaign
from topology_learner import ROW, SamplingActor, activate, checkpoint_path, checkpoint_sha, load_learner, publish
from frozen_topology_policy import validate_state
from topology_policy import require_layout
import collect_trial


def labels(batch):
    require(type(batch) is int and 1 <= batch <= 4, 'batch must be 1..4')
    return [f'T_{n:03d}' for n in range(32 * (batch - 1) + 1, 32 * batch + 1)]


def binding_for(spec, sha, update):
    raw = Path(spec['activation_path']).read_bytes()
    binding = json.loads(raw)
    require(binding['policy_update'] == spec['policy_update'] == update and
            binding['config_sha256'] == sha and binding['checkpoint_sha256'] == checkpoint_sha(update) and
            binding['checkpoint_path'] == str(checkpoint_path(update)) and
            all(binding[key] == spec[key] for key in ('run_index', 'recording_id')), 'wrong training activation')
    return binding


class TrainingTrial(ScaledTrial):
    def __init__(self, label, learner, update):
        spec, self.sha, self.edges = campaign.read_case(label)
        require(spec['phase'] == 'train' and spec['policy_update'] == update and spec['action_mode'] == 'sample',
                'batch or sample mode mismatch')
        self.output = Path(spec['python_output_directory'])
        self.events = Path(spec['event_directory'])
        require(not self.events.exists() and not self.output.exists(), 'existing training evidence for this shift')
        binding = activate(label, update)
        self.spec = dict(spec, checkpoint_sha256=binding['checkpoint_sha256'])
        self.policy = SamplingActor(learner, self.spec, self.edges, self.output)
        self.active = self.policy.opening_mask
        self.sequence = 0
        self.previous_time = -1.
        self.last_key = self.reply = self.writer = None
        self.sealed = False

    def decide(self, request):
        require(request['obs'][16] == 0, 'expected stochastic native B conditions')
        return super().decide(request)

    def close(self):
        super().close()
        self.policy.close()

    def try_seal(self):
        if self.sealed:
            return
        super().try_seal()
        if self.sealed:
            buffer = self.output / 'ppo_buffer.bin'
            require(buffer.stat().st_size == self.sequence * ROW.itemsize, 'incomplete PPO dispatch buffer')
            receipt = dict(recording_id=self.spec['recording_id'], run_index=self.spec['run_index'],
                           config_sha256=self.sha, checkpoint_sha256=self.spec['checkpoint_sha256'],
                           decisions=self.sequence, buffer_sha256=digest(buffer))
            publish(self.output / 'ppo_buffer_receipt.json', encode(receipt))


def serve(batch):
    learner = load_learner(batch - 1)
    trial_service.Trial = lambda case: TrainingTrial(case, learner, batch - 1)
    trial_service.serve(labels(batch))


def read_episode(label, update, actor_digest):
    spec, sha, edges = campaign.read_case(label)
    _, _, job = campaign.load_case(label)
    bound = binding_for(spec, sha, update)
    spec = dict(spec, checkpoint_sha256=bound['checkpoint_sha256'])
    output = Path(spec['python_output_directory'])
    event_dir = Path(spec['event_directory'])
    manifest_path = event_dir / 'events_manifest.json'
    manifest = json.loads(manifest_path.read_bytes())
    require(manifest['native_policy_update'] == update and manifest['config']['actual_runtime_scene'] ==
            spec['expected_runtime_scene'], 'unbound native fleet, model, task, or topology')
    evidence = collect_trial.witness.replay_event_bundle(
        event_dir, expected_episode_id=manifest['episode_id'], expected_config_sha256=sha,
        expected_recording_id=spec['recording_id'], expected_source_kind='actual_anylogic')
    require(evidence['run_index'] == spec['run_index'] and evidence['seed'] == spec['environment_seed'],
            'native training identity mismatch')
    audit = collect_trial.policy_audit(dict(spec, policy_id=spec.get('policy_id', 'bg128-B-finetune')), job,
                                       (manifest_path, manifest, evidence))
    receipt = json.loads((output / 'ppo_buffer_receipt.json').read_bytes())
    buffer = output / 'ppo_buffer.bin'
    require(all(receipt[k] == bound[k] for k in ('recording_id', 'run_index', 'config_sha256','checkpoint_sha256'))
            and receipt['buffer_sha256'] == digest(buffer) and
            receipt['decisions'] == audit['decision_count'] and buffer.stat().st_size == receipt['decisions'] * ROW.itemsize,
            'collection buffer does not match native/inference evidence')
    rows = np.fromfile(buffer, dtype=ROW)
    inferred = [json.loads(line) for line in (output / 'inference.jsonl').read_bytes().splitlines()]
    for row, entry in zip(rows, inferred, strict=True):
        require(row['action'] == entry['action'] and sha256(row['state'].tobytes()).hexdigest() ==
                entry['observation_sha256'], 'PPO inputs or action differ from applied dispatch')
    require(np.isfinite(rows['value']).all() and np.isfinite(rows['log_probability']).all(),
            'nonfinite rollout value or action probability')
    terminal_file = output / 'terminal_observation.json'
    require(digest(terminal_file) == manifest['terminal_observation_sha256'], 'terminal observation SHA mismatch')
    end = json.loads(terminal_file.read_bytes())
    raw = np.asarray(end['obs'], dtype=np.float32)
    request_end = evidence['requests'][-1]
    require(all(end[k] == bound[k] for k in ('recording_id','run_index','config_sha256','checkpoint_sha256','policy_update'))
            and end['layout_id'] == spec['layout_id'] and end['environment_seed'] == spec['environment_seed']
            and request_end['terminal'] and end['decision_sequence'] == request_end['decision_sequence'] == len(rows)
            and end['sim_time_seconds'] == request_end['sim_time_seconds'] == 28800
            and end['dispatchable_mask'] == end['in_system_mask'] == 223, 'invalid true terminal identity or roster')
    validate_state(raw)
    require(raw[0] == 1 and raw[17] == 1 and raw[16] == 0, 'not a true stochastic terminal observation')
    require_layout(raw[None], edges)
    check_observed_targets(raw, spec['task'])
    for name in ('ore_tons','waste_tons','metal_tons'):
        require(math.isclose(end[name], evidence[name], rel_tol=0., abs_tol=spec['task']['mass_balance_epsilon']),
                f'terminal {name} disagrees with physical mass replay')
    wait = evidence['independent_wait_seconds']
    require(math.isclose(end['cumulative_wait_seconds'], wait, rel_tol=1e-9, abs_tol=1e-6),
            'terminal waiting disagrees with physical replay')
    task, scene = spec['task'], evidence['config']['actual_runtime_scene']['exogenous_config']
    require(all(task[k] == scene[k] for k in ('ore_target_tons', 'waste_target_tons', 'waste_upper_tons',
                                               'waste_dispatch_stop_tons')), 'scaled run goal disagrees with native')
    material = evidence['material_totals']
    for stem, fifo_key, input_key in (('ore_tons','fifo_tons','enqueued_tons'),
                                      ('metal_tons','fifo_metal_tons','enqueued_metal_tons')):
        require(abs(material[input_key] - material[stem] - material[fifo_key]) <= task['mass_balance_epsilon'],
                'crusher input/output mass conservation failed')
    requests = evidence['requests']
    closures = []
    for window in evidence['windows']:
        index = window['window_index']
        step = next(r['decision_sequence'] - 1 for r in requests if index in r['window_indices'])
        closures.append(dict(window=index, step=step, tons=window['ore_tons'], metal_tons=window['metal_tons']))
    physical = collect_trial.shift_outcomes(evidence, task)
    physical['total_wait_seconds'] = wait
    masks = np.full(len(rows)+1, 223, dtype=np.int64)
    return dict(identity={k:spec[k] for k in ('run_index','recording_id','environment_seed','actor_sampling_seed')},
                policy_version=actor_digest, states=np.concatenate((rows['state'], raw[None])),
                actions=np.asarray(rows['action'], dtype=np.int64), old_log_probs=rows['log_probability'].copy(),
                collected_values=rows['value'].copy(), dispatchable_masks=masks, in_system_masks=masks.copy(),
                wait_deltas=np.asarray([r['delta_wait_seconds'] for r in requests[1:]]),
                window_closures=closures, evaluation=physical,
                rpc_config={'config.target_ore_tons_shift':scene['ore_target_tons'],
                            'config.target_waste_tons_shift':scene['waste_target_tons'],
                            'config.mass_balance_epsilon':evidence['config']['mass_balance_epsilon']})


def batch_outcomes(episodes):
    results = [e['evaluation'] for e in episodes]
    n = len(results)
    return dict(shifts=n, compliant_windows=sum(r['compliant_windows'] for r in results),
                four_windows=sum(r['compliant_windows']==4 for r in results),
                joint=sum(r['compliant_windows']==4 and r['ore_target_met'] and r['waste_target_met'] for r in results),
                ore_target_met=sum(r['ore_target_met'] for r in results), waste_target_met=sum(r['waste_target_met'] for r in results),
                mean_ore_tons=fmean(r['ore_tons'] for r in results), mean_waste_tons=fmean(r['waste_tons'] for r in results),
                compliant_by_window=[sum(r['windows'][j]['cost'] == 0 for r in results) for j in range(4)])


def update_batch(batch):
    final = checkpoint_path(batch).parent
    if final.exists():
        metrics = json.loads((final / 'metrics.json').read_bytes())
        require(metrics['batch'] == batch and metrics['checkpoint_sha256'] == checkpoint_sha(batch) and
                metrics['parent_checkpoint_sha256'] == checkpoint_sha(batch - 1), 'existing update lineage mismatch')
        print(encode(metrics).decode(), flush=True)
        return
    started = time.monotonic()
    learner = load_learner(batch - 1)
    old_digest = learner.policy_version
    episodes = [read_episode(label, batch - 1, old_digest) for label in labels(batch)]
    print(f'Verified {len(episodes)} physical B shifts. Starting PPO update {batch}.', flush=True)
    result = learner.update(episodes, batch_id=f'{campaign.CAMPAIGN}/update-{batch}')
    require(learner.update_index == 9 + batch and learner.policy_version != old_digest,
            'optimizer did not produce a new actor')
    collected_values = np.concatenate([ep['collected_values'] for ep in episodes])
    replay_values = learner.last_targets['old_values']
    require(np.allclose(collected_values, replay_values, rtol=2e-5, atol=2e-5),
            'frozen Critic predictions differ from rollout collection')
    final.parent.mkdir(parents=True, exist_ok=True)
    # Stage all four files together, then atomically make the whole generation eligible for the next batch.
    with TemporaryDirectory(dir=final.parent, prefix=f'.b-update-{batch}-') as tmp:
        stage = Path(tmp) / 'candidate'
        info = learner.save(stage / 'checkpoint.pt')
        metrics = dict(batch=batch, parent_checkpoint_sha256=checkpoint_sha(batch - 1),
                       checkpoint_sha256=info['sha256'], parent_actor_digest=old_digest, actor_digest=learner.policy_version,
                       actor_lr=1e-4, critic_lr=3e-4, physical=batch_outcomes(episodes),
                       max_old_value_replay_error=float(np.max(np.abs(collected_values-replay_values))),
                       elapsed_seconds=time.monotonic()-started, **result)
        publish(stage / 'metrics.json', encode(metrics))
        os.rename(stage, final)
    require(load_learner(batch).policy_version == learner.policy_version, 'sealed checkpoint cannot be restored')
    print(encode(metrics).decode(), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('serve','update'))
    parser.add_argument('--batch', required=True, type=int, choices=range(1,5))
    args = parser.parse_args()
    if args.command == 'serve':
        serve(args.batch)
    else:
        campaign.MODEL_OUTPUT.mkdir(parents=True, exist_ok=True)
        with (campaign.MODEL_OUTPUT / f'.batch-{args.batch}.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            update_batch(args.batch)

