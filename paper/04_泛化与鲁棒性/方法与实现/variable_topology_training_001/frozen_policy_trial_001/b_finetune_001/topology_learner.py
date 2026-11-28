"""Fine-tune the restored fleet policy using the road graph seen by the deployed B actor."""
import json
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile
from types import MethodType

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
TRIAL = HERE.parent
sys.path[:0] = [str(TRIAL), str(TRIAL.parents[1] / 'fleet_outage_resilience_001' / 'replay_repair_001')]
from frozen_topology_policy import (
    ACTOR_DIGEST, CHECKPOINT_PATH, CHECKPOINT_SHA256, CHECKPOINT_MANIFEST_SHA256, CHECKPOINT_SEAL_SHA256,
    FrozenTopologyPolicy, _actor_logits, _critic_value, episode_node_order,
)
from replay_policy import FleetLearner
from trial_service import digest, encode, require


def publish(path, data):
    """Publish immutable evidence without overwriting an existing result, including on a retry."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix='.' + path.name, delete=False) as tmp:
        try:
            tmp.write(data)
            tmp.flush()
            os.fsync(tmp.fileno())
            try:
                os.link(tmp.name, path)
            except FileExistsError:
                require(path.read_bytes() == data, f'conflicting sealed file {path}')
        finally:
            Path(tmp.name).unlink(missing_ok=True)


def checkpoint_path(update):
    from campaign import MODEL_OUTPUT
    require(type(update) is int and 0 <= update <= 4, 'invalid B checkpoint generation')
    return CHECKPOINT_PATH if update == 0 else MODEL_OUTPUT / f'u{update:04d}' / 'checkpoint.pt'


def checkpoint_sha(update):
    path = checkpoint_path(update)
    actual = digest(path)
    manifest = Path(str(path) + '.manifest.json')
    seal = Path(str(path) + '.seal.json')
    info = json.loads(manifest.read_bytes())
    sealed = json.loads(seal.read_bytes())
    require(info['update_index'] == 9 + update and info['bytes'] == path.stat().st_size and
            actual == info['sha256'] == sealed['sha256'] and digest(manifest) == sealed['manifest_sha256'],
            f'checkpoint u{update} has invalid seal')
    if update == 0:
        require(actual == CHECKPOINT_SHA256 and digest(manifest) == CHECKPOINT_MANIFEST_SHA256 and
                digest(seal) == CHECKPOINT_SEAL_SHA256 and info['actor_digest'] == ACTOR_DIGEST,
                'original dynamic u9 has changed')
    return actual


def activate(label, update):
    from campaign import read_case
    spec, sha, _ = read_case(label)
    require(spec['policy_update'] == update or (spec['phase'] == 'test' and spec['policy_update'] == -1),
            'activation differs from declared role')
    value = dict(run_index=spec['run_index'], recording_id=spec['recording_id'], config_sha256=sha,
                 policy_update=update, checkpoint_path=str(checkpoint_path(update)), checkpoint_sha256=checkpoint_sha(update))
    publish(Path(spec['activation_path']), encode(value))
    return value


class TopologyLearner(FleetLearner):
    def __init__(self, seed):
        super().__init__(seed)
        # Binding only the forward passes leaves all restored Parameter and optimizer identities intact.
        self.actor.forward = MethodType(_actor_logits, self.actor)
        self.critic.forward = MethodType(_critic_value, self.critic)


def load_learner(update):
    torch.set_num_threads(1)
    checkpoint_sha(update)
    learner = TopologyLearner.load(checkpoint_path(update))
    require(learner.update_index == 9 + update, 'restored checkpoint has unexpected update number')
    for role, rate in (('actor', 1e-4), ('critic', 3e-4)):
        for group in getattr(learner, role + '_optimizer').param_groups:
            group['lr'] = rate
        learner.config['ppo'][role + '_lr'] = rate
    return learner


class InferenceActor:
    _inputs = FrozenTopologyPolicy._inputs

    def __init__(self, learner, spec, edges):
        self.actor, self.critic = learner.actor, learner.critic
        self.edges = edges
        self.opening_mask = sum(1 << row[0] for row in spec['expected_runtime_scene']['opening_fleet'] if row[1])
        require(self.opening_mask == 223 and spec['layout_id'] == 'loop_B', 'unexpected topology or roster')
        self.order = episode_node_order(spec['slot_permutation_seed'], self.opening_mask)

    @torch.no_grad()
    def action(self, obs, mask, dispatchable, present, truck):
        packed, prefix = self._inputs(obs, mask, dispatchable, present, truck)
        return int(self.actor(packed, prefix)[0].argmax().item())


# Required training buffer; recorded once per physical dispatch and read by the separate optimizer process.
ROW = np.dtype([('state', '<f4', (1519,)), ('action', '<i4'), ('log_probability', '<f8'), ('value', '<f8')])


class SamplingActor(InferenceActor):
    def __init__(self, learner, spec, edges, output):
        super().__init__(learner, spec, edges)
        require(spec['actor_sampling_seed'] == spec['slot_permutation_seed'], 'sampling and node-order identity differ')
        self.rng = torch.Generator().manual_seed(spec['actor_sampling_seed'])
        self.output = output
        self.buffer = None

    @torch.no_grad()
    def action(self, obs, mask, dispatchable, present, truck):
        packed, prefix = self._inputs(obs, mask, dispatchable, present, truck)
        distribution = self.actor.distribution(packed, prefix)
        chosen = torch.multinomial(distribution.probs[0], 1, generator=self.rng)
        action = int(chosen.item())
        log_prob = float(distribution.log_prob(chosen).item())
        value = float(self.critic(packed, prefix)[0].item())
        require(np.isfinite([log_prob, value]).all(), 'nonfinite sampled likelihood or value')
        if self.buffer is None:
            self.output.mkdir(parents=True, exist_ok=True)
            self.buffer = (self.output / 'ppo_buffer.bin').open('xb')
        self.buffer.write(np.asarray((obs, action, log_prob, value), dtype=ROW).tobytes())
        self.buffer.flush()
        return action

    def close(self):
        if self.buffer is not None:
            self.buffer.close()

