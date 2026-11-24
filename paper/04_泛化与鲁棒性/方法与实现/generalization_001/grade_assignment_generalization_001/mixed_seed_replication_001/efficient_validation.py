"""Reuse an exact Actor proof only inside its already validated physical phase."""

from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path

import numpy as np

from replica_plan import parent
from baseline1_v1.collection import verify_durable_reference
from business_graph_summary_campaign_v1 import collection, policy
from business_graph_summary_campaign_v1.plan_io import encoded, reference


def _array_hash(*values):
    digest = sha256()
    for value in values:
        array = np.ascontiguousarray(value)
        digest.update(encoded([array.dtype.str, list(array.shape)]))
        digest.update(memoryview(array).cast('B'))
    return digest.hexdigest()


def _raw_manifest(job):
    return Path(job['python_output']) / 'episodes' / f"run_{job['run_index']}" / 'manifest.json'


class PhaseProof:
    def __init__(self, jobs, runtime):
        self.jobs = {job['run_index']: job for job in jobs}
        self.runtime = runtime
        self.certificates = {}
        self.last_batch = None
        self.validated_seal = None
        self.count = dict(exact_replays=0, exact_replays_reused=0, batched_replays_reused=0,
                          learner_proofs_reused=0)

    def batched(self, original):
        def replay(actor, states, *, actions=None, chunk_size=16):
            if actions is None:
                return original(actor, states, actions=actions, chunk_size=chunk_size)
            key = (self.runtime.digest(actor), encoded(actor.feature_config), chunk_size, _array_hash(states, actions))
            if self.last_batch is not None and self.last_batch[0] == key:
                self.count['batched_replays_reused'] += 1
                return self.last_batch[1]
            result = original(actor, states, actions=actions, chunk_size=chunk_size)
            self.last_batch = (key, result)
            return result
        return replay

    def verify(self, actor, states, actions, logs, replay=None, *, rows=None, job=None):
        parent.require(job is not None and self.jobs.get(job['run_index']) == job and rows is not None
                       and len(rows) == len(states) == len(actions) == len(logs) > 0,
                       'exact admission needs one sealed planned shift')
        if replay is not None:
            values = replay['probabilities'].detach().numpy()
            parent.require(values.shape == (len(rows), 4) and np.isfinite(values).all(), 'invalid batch replay')
        terminal = rows[-1]['next_observation']
        raw = reference(_raw_manifest(job))
        payload = _array_hash(states, terminal, np.asarray(actions, dtype=np.int64), np.asarray(logs, dtype=np.float64))
        draws = sha256(encoded([[row['action'], row.get('actor_sampling_seed'), row.get('actor_sampling_draw_index'),
                                row.get('actor_rng_before_sha256'), row.get('actor_rng_after_sha256')]
                               for row in rows])).hexdigest()
        key = (raw, self.runtime.digest(actor), payload, draws)
        index = job['run_index']
        if index in self.certificates:
            parent.require(self.certificates[index] == key, 'admission actions, inputs, RNG or source changed')
            self.count['exact_replays_reused'] += 1
        else:
            self.runtime.verify_online(actor, states, actions, logs, replay, rows=rows, job=job)
            self.certificates[index] = key
            self.count['exact_replays'] += 1
        return 'exact_online'

    def learner(self, model, episode, current):
        identity = episode['identity']
        index = identity['run_index']
        job = self.jobs[index]
        if index not in self.certificates:
            return self.runtime.verify_learner_behavior(model, episode, current, job)
        artifact = model._actor_artifact
        raw, policy_digest, payload, _ = self.certificates[index]
        parent.require(np.isfinite(current.detach().numpy()).all() and isinstance(artifact, dict)
                       and artifact['actor_digest'] == model.policy_version == identity['policy_version'] == policy_digest
                       and artifact['sha256'] == identity['policy_sha256']
                       and job['actor_sampling_seed'] == identity['actor_sampling_seed']
                       and identity['raw_manifest_sha256'] == raw['sha256'], 'learner Actor/run proof differs')
        seal = reference(artifact['path'])
        parent.require(seal['sha256'] == artifact['sha256'], 'learner Actor artifact bytes changed')
        if self.validated_seal != seal:
            parent.require(model._validated_actor_snapshot(artifact['path'])['sha256'] == seal['sha256'],
                           'learner Actor seal differs')
            self.validated_seal = seal
        verify_durable_reference(raw)
        states = episode['states']
        parent.require(payload == _array_hash(states[:-1], states[-1], np.asarray(episode['actions'], dtype=np.int64),
                                             np.asarray(episode['old_log_probs'], dtype=np.float64)),
                       'learner inputs no longer match exact admission')
        self.count['learner_proofs_reused'] += 1
        return 'exact_online'


@contextmanager
def validation_scope(jobs, runtime):
    """Keep all admission and target gates; deduplicate identical Actor forwards."""
    proof = PhaseProof(jobs, runtime)
    with runtime.VERIFY_LOCK:
        original = (
            (runtime.campaign_behavior, 'verify', proof.verify),
            (runtime.learner_behavior, 'verify_learner', proof.learner),
            (collection, 'replay_policy', proof.batched(collection.replay_policy)),
            (policy, 'replay_policy', proof.batched(policy.replay_policy)),
        )
        overrides = [(module, name, getattr(module, name), replacement)
                     for module, name, replacement in original]
        try:
            for module, name, _, replacement in overrides:
                setattr(module, name, replacement)
            yield proof
        finally:
            for module, name, saved, _ in reversed(overrides):
                setattr(module, name, saved)
