"""Resume the sealed BG128 run through absolute update 32."""
import copy
from functools import lru_cache
import hashlib
import inspect
import json
from pathlib import Path
import random
import textwrap

import numpy as np
import torch

from baseline1_v1.java_plan import COLUMNS
from business_graph_summary_campaign_v1.plan_io import encoded, load_phase, reference, sha
from business_graph_summary_campaign_v1.progress import update_callback, write_progress
from business_graph_summary_campaign_v1.runtime import _learning_contract, _targets, _verify_target_ref
from business_graph_summary_recovery_v1.collection import (
    MaterialPhaseCollector, OnlineActor, load_admitted_episode,
)
from business_graph_summary_v1 import learner as frozen
from business_graph_summary_v1.contract import require
from business_graph_summary_v1.models import digest
from grade_only_ppo_v2.runtime import _actor_snapshot, _equal_tree
from online_v1_1.operations import _publish


CANDIDATE = 's01_BG128'
FIRST_UPDATE = 17
LAST_UPDATE = 32
SHIFTS_PER_UPDATE = 32
FROZEN_LEARNER_SHA = 'e676fb335503154284b78fde66e0c0207fcba96825c684564029a444c2591def'
PARENT_CHECKPOINT_SHA = 'f0bf60a3091f85eb4f163806805f495ff70d0f7aa325906203c82c92ecc30e43'
PARENT_CAMPAIGN_SHA = '90ada18aa85cdba4b51b3c55e9a3326729662a1dcc1b53a4b331a029cd7614d3'


def _extend_method(method, original, replacement):
    source = textwrap.dedent(inspect.getsource(method))
    require(source.count(original) == 1, 'reviewed learner update bound changed')
    # Use a private namespace so the historical learner remains importable and unchanged.
    scope = dict(vars(frozen))
    exec(compile(source.replace(original, replacement), inspect.getsourcefile(method), 'exec'), scope)
    return scope[method.__name__]


@lru_cache(maxsize=1)
def _extension_learner():
    require(sha(frozen.__file__) == FROZEN_LEARNER_SHA, 'frozen BG128 learner source changed')

    class ExtensionLearner(frozen.JointCostPPO):
        update = _extend_method(frozen.JointCostPPO.update,
            "limit=1 if self.config['execution_profile']=='smoke_5ep_v1' else 16",
            "limit=1 if self.config['execution_profile']=='smoke_5ep_v1' else 32")
        from_checkpoint = _extend_method(frozen.JointCostPPO.from_checkpoint.__func__,
            "(1 if config['execution_profile']=='smoke_5ep_v1' else 16)",
            "(1 if config['execution_profile']=='smoke_5ep_v1' else 32)")

    return ExtensionLearner


def _lineage(model):
    config = model.config
    campaign = model.runtime_state['campaign']
    require(config['candidate_id'] == CANDIDATE and config['execution_profile'] == 'formal_32ep_v1' and
            config['ppo']['actor_lr'] == 3e-4 and config['ppo']['critic_lr'] == 1e-3 and
            campaign['sha256'] == PARENT_CAMPAIGN_SHA and reference(campaign['path']) == campaign and
            model.runtime_state['candidate_id'] == CANDIDATE and
            len(model.consumed_batches) == model.update_index and
            len(model.consumed_episodes) == model.update_index * SHIFTS_PER_UPDATE and
            [item['update_index'] for item in model.consumed_batches.values()] ==
            list(range(1, model.update_index + 1)), 'checkpoint is outside the original BG128 training lineage')


def load_checkpoint(ref, config, *, restore=True):
    """Load both Models, Adam states, counters and private RNGs from a sealed file."""
    model = _extension_learner().from_checkpoint(
        ref['path'], expected_config=config, expected_sha256=ref['sha256'], restore_global_rng=restore,
    )
    require(FIRST_UPDATE - 1 <= model.update_index <= LAST_UPDATE, 'checkpoint outside the extension interval')
    _lineage(model)
    if model.update_index == FIRST_UPDATE - 1:
        require(ref['sha256'] == PARENT_CHECKPOINT_SHA, 'expected the original complete u16 checkpoint')
    else:
        transaction = model.runtime_state['transaction']
        require(transaction['schema'] == 'bg128-extension-update-v1' and
                transaction['update'] == model.update_index and
                transaction['parent_checkpoint'] == reference(transaction['parent_checkpoint']['path']) and
                transaction['campaign'] == reference(transaction['campaign']['path']),
                'extension checkpoint provenance differs')
    return model


def _global_rng():
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def _checkpoint_checks(model, restored):
    return dict(actor=model.policy_version == restored.policy_version,
        critic=digest(model.critic) == digest(restored.critic),
        actor_optimizer=_equal_tree(model.actor_optimizer.state_dict(), restored.actor_optimizer.state_dict()),
        critic_optimizer=_equal_tree(model.critic_optimizer.state_dict(), restored.critic_optimizer.state_dict()),
        minibatch_rng=_equal_tree(model.generator.get_state(), restored.generator.get_state()),
        actor_sampling_rng=_equal_tree(model.sampling_generator.get_state(), restored.sampling_generator.get_state()),
        consumed_batches=_equal_tree(model.consumed_batches, restored.consumed_batches),
        consumed_episodes=_equal_tree(model.consumed_episodes, restored.consumed_episodes),
        frozen_targets=_equal_tree(model.last_targets, restored.last_targets),
        frozen_targets_sha256=model.last_target_hash == restored.last_target_hash,
        runtime_state=_equal_tree(model.runtime_state, restored.runtime_state),
        configuration=model.config == restored.config,
        update_index=model.update_index == restored.update_index)


def verify_initial_checkpoint(ref, config):
    """Read-only proof using the original and extension checkpoint loaders."""
    start_rng = _global_rng()
    original = frozen.JointCostPPO.from_checkpoint(
        ref['path'], expected_config=config, expected_sha256=ref['sha256'], restore_global_rng=False,
    )
    model = load_checkpoint(ref, config, restore=False)
    checks = _checkpoint_checks(original, model)
    checks['global_rng_unchanged'] = _equal_tree(start_rng, _global_rng())
    actor_steps = sorted({int(state['step']) for state in model.actor_optimizer.state.values()})
    critic_steps = sorted({int(state['step']) for state in model.critic_optimizer.state.values()})
    require(all(checks.values()) and model.update_index == FIRST_UPDATE - 1 and
            len(model.consumed_episodes) == 512 and actor_steps == critic_steps == [11720],
            'continued learner failed exact u16 restored-state verification')
    return dict(schema='bg128-extension-u16-restore-v1', checkpoint=ref, update=model.update_index,
        frozen_learner_sha256=FROZEN_LEARNER_SHA, checks=checks, actor_digest=model.policy_version,
        critic_digest=digest(model.critic), consumed_episodes=len(model.consumed_episodes),
        consumed_batches=len(model.consumed_batches), actor_adam_steps=actor_steps, critic_adam_steps=critic_steps,
        preserved_actor_adam_states=len(model.actor_optimizer.state),
        preserved_critic_adam_states=len(model.critic_optimizer.state), optimizer_updates=0)


def _immutable_json(path, value):
    data = encoded(value)
    if path.is_file():
        require(path.read_bytes() == data, 'immutable update artifact differs: ' + str(path))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        _publish(path, data)


def _publish_ready(suite, suite_ref, phase_ref, parent, ledger):
    from extension_common import check_sources

    control = Path(suite['control_directory'])

    def publish(job, config_path):
        check_sources(suite)
        row = ledger.job(job)
        require(reference(config_path) == phase_ref and reference(suite_ref['path']) == suite_ref and
                reference(parent['path']) == parent and row['status'] == 'CLAIMED' and
                row['identity'] == job and not (control / 'STOP.properties').exists(),
                'training ready requires the exact immutable claimed slot and parent')
        values = {key: str(job[key]) for key in COLUMNS}
        values.update(protocol='baseline1-one-click-ready-v1', campaign_id=suite['campaign_id'],
            master_plan_sha256=suite['master_plan']['sha256'], config_sha256=phase_ref['sha256'],
            snapshot_sha256=phase_ref['sha256'], snapshot_path=phase_ref['path'],
            instrumentation_revision='boundary-events-v1-native-rpc-runtime-scene-v1',
            all_comparison_checkpoints_frozen='false')
        require(all(value.isascii() and '\n' not in value and '\r' not in value for value in values.values()),
                'ambiguous training ready Properties')
        path = control / 'ready' / ('run_' + str(job['run_index']) + '.properties')
        path.parent.mkdir(parents=True, exist_ok=True)
        _publish(path, ''.join(key + '=' + value + '\n' for key, value in values.items()).encode('ascii'))

    return publish


def _collect_episodes(suite, suite_ref, cfg, phase_ref, parent, ledger, model):
    jobs = cfg['jobs']
    rows = [ledger.job(job) for job in jobs]
    require(all(row['status'] in ('PLANNED', 'COMPLETE') for row in rows),
            'prior training claim or UNKNOWN requires reconciliation')
    pending = [job for job, row in zip(jobs, rows) if row['status'] == 'PLANNED']
    actor_ref = _actor_snapshot(model, Path(cfg['python_output']) / 'actor.pt')
    fresh = {}
    if pending:
        active = dict(copy.deepcopy(cfg), jobs=pending)
        ready = _publish_ready(suite, suite_ref, phase_ref, parent, ledger)
        startup = dict(control_directory=suite['control_directory'], campaign_id=suite['campaign_id'],
                       master_plan_sha256=suite['master_plan']['sha256'])
        behavior = OnlineActor(model.actor, pending)
        with MaterialPhaseCollector(active, phase_ref['path'], publish_ready=ready, startup=startup) as collector:
            for job in pending:
                write_progress(suite, status='COLLECTING', phase_id=cfg['phase_id'],
                    candidate_id=CANDIDATE, update_index=cfg['arm_update'],
                    run_index=job['run_index'], ordinal=job['ordinal'])
                ledger.claim(job)
                try:
                    episode, source, receipt = collector.collect(job, behavior, actor_path=actor_ref['path'])
                    ledger.complete(job, source, receipt)
                except BaseException as error:
                    if ledger.job(job)['status'] == 'CLAIMED':
                        ledger.unknown(job, repr(error))
                    raise
                fresh[job['run_index']] = episode
                print('TRAINING_SHIFT_COMPLETE phase=' + cfg['phase_id'] + ' run=' + str(job['run_index']), flush=True)

    episodes, sources = [], []
    for job in jobs:
        row = ledger.job(job)
        require(row['status'] == 'COMPLETE' and row['identity'] == job and
                row['receipt']['phase_config'] == phase_ref and
                row['dataset_ref'] == reference(row['dataset_ref']['path']), 'training episode seal differs')
        episode = fresh.pop(job['run_index'], None)
        if episode is None:
            episode = load_admitted_episode(row['receipt'])
        identity = episode['identity']
        require(identity['run_index'] == job['run_index'] and identity['environment_seed'] == job['experiment_seed'] and
                identity['actor_sampling_seed'] == job['actor_sampling_seed'] and identity['split'] == 'train' and
                identity['policy_sha256'] == actor_ref['sha256'] and identity['policy_version'] == model.policy_version,
                'training episode identity or behavior differs')
        episodes.append(episode)
        sources.append(row['dataset_ref'])
    return episodes, sources


def _restore_proof(model, checkpoint, path):
    saved, manifest = frozen._load_checkpoint(checkpoint['path'])
    start_rng = _global_rng()
    restored = load_checkpoint(checkpoint, model.config, restore=False)
    checks = _checkpoint_checks(model, restored)
    checks['global_rng_unchanged'] = _equal_tree(start_rng, _global_rng())
    checks['global_rng_sealed'] = _equal_tree(start_rng, (saved['python_rng'], saved['numpy_rng'], saved['torch_rng']))
    require(all(checks.values()) and manifest['update_index'] == model.update_index, 'extension checkpoint restore differs')
    _immutable_json(path, dict(schema='bg128-extension-checkpoint-restore-v1', checkpoint=checkpoint,
        update=model.update_index, checks=checks, optimizer_updates=0,
        frozen_learner_sha256=FROZEN_LEARNER_SHA))


def train_phase(suite_path, suite, cfg, config_path, ledger, model=None):
    """Collect 32 on-policy shifts and atomically commit one absolute update."""
    from extension_common import check_sources

    check_sources(suite)
    config = suite['arms'][0]['learner_config']
    suite_ref, phase_ref = reference(suite_path), reference(config_path)
    update = cfg['arm_update']
    name = f'train_u{update:04d}'
    jobs = cfg['jobs']
    arm = suite['arms'][0]
    require(FIRST_UPDATE <= update <= LAST_UPDATE and len(suite['arms']) == 1 and
            arm['candidate_id'] == cfg['candidate_id'] == CANDIDATE and
            cfg['arm_phase_id'] == name and cfg['phase_id'] == CANDIDATE + '_' + name and
            cfg['campaign_id'] == suite['campaign_id'] and len(jobs) == SHIFTS_PER_UPDATE and
            Path(cfg['python_output']) == Path(arm['python_output']) / name and
            phase_ref in suite['phase_configs'] and cfg == load_phase(config_path) and
            all(job['role'] == job['partition'] == 'train' and job['checkpoint_update'] == update - 1 and
                job['target_training_update'] == job['arm_update'] == update and
                job['actor_seed'] == config['seed'] and job['phase_id'] == cfg['phase_id'] and
                job['recording_id'] == cfg['recording_id'] for job in jobs), 'invalid absolute training phase')
    parent = (suite['parent_checkpoint'] if update == FIRST_UPDATE
              else ledger.commit_record(update - 1)['checkpoint_ref'])
    folder = Path(arm['python_output']) / 'updates' / f'{update:04d}'
    checkpoint_path = folder / 'checkpoint.pt'
    record = ledger.commit_record(update)
    if record is not None:
        prepared = json.loads((folder / 'prepared.json').read_bytes())
        require(prepared['checkpoint'] == record['checkpoint_ref'] == reference(checkpoint_path) and
                prepared['intent']['parent_checkpoint'] == parent and prepared['intent']['campaign'] == suite_ref and
                prepared['intent']['phase_config'] == phase_ref and
                prepared['intent']['batch_id'] == record['batch_id'], 'committed extension batch provenance differs')
        restored = load_checkpoint(prepared['checkpoint'], config)
        require(restored.runtime_state['transaction'] == prepared['intent'] and restored.update_index == update
                and restored.runtime_state['training_targets'] == prepared['training_targets'] and
                reference(folder / 'training_summary.json') == restored.runtime_state['training_summary'],
                'committed extension checkpoint differs')
        _verify_target_ref(prepared['training_targets'], prepared['intent'])
        return restored, prepared['checkpoint']

    restored = load_checkpoint(parent, config)
    require(restored.update_index == update - 1 and
            (model is None or all(_checkpoint_checks(model, restored).values())),
            'training parent is not the previous sealed update')
    model = restored
    episodes, sources = _collect_episodes(suite, suite_ref, cfg, phase_ref, parent, ledger, model)
    batch_id = hashlib.sha256(encoded(dict(campaign=suite_ref, phase_config=phase_ref,
                                           parent_checkpoint=parent, sources=sources))).hexdigest()
    intent = dict(schema='bg128-extension-update-v1', campaign=suite_ref, phase_config=phase_ref,
        parent_checkpoint=parent, learning_contract=_learning_contract(suite, CANDIDATE),
        update=update, arm_update=update, phase_id=cfg['phase_id'], batch_id=batch_id,
        sources=sources, run_indices=[job['run_index'] for job in jobs])
    intent_path = folder / 'intent.json'
    had_intent = intent_path.exists()
    _immutable_json(intent_path, intent)
    summary_path = folder / 'training_summary.json'
    expected_sources = model.consumed_episodes.copy()
    expected_batches = list(model.consumed_batches)
    if checkpoint_path.is_file():
        model = load_checkpoint(reference(checkpoint_path), config)
        require(model.update_index == update and model.runtime_state['transaction'] == intent and
                model.runtime_state['training_summary'] == reference(summary_path),
                'uncommitted sealed checkpoint belongs to a different batch')
        _verify_target_ref(model.runtime_state['training_targets'], intent)
    else:
        require(not had_intent and not any((folder / name).exists() for name in (
            'training_targets.h5', 'training_summary.json', 'prepared.json', 'checkpoint_restore_verification.json',
        )), 'unsealed optimizer transaction requires reconciliation before continuation')
        model.progress_callback = update_callback(suite, cfg, update)
        try:
            result = model.update(episodes, batch_id=batch_id)
        finally:
            model.progress_callback = None
        require(model.update_index == update and list(model.consumed_batches) == expected_batches + [batch_id] and
                all(model.consumed_episodes.get(key) == value for key, value in expected_sources.items()) and
                len(model.consumed_episodes) == update * SHIFTS_PER_UPDATE, 'update took a different training path')
        targets = _targets(folder / 'training_targets.h5', model.last_targets, intent)
        _immutable_json(summary_path, dict(transaction=intent, update=result))
        model.runtime_state.update(transaction=intent, training_targets=targets, training_summary=reference(summary_path))
        model.save_checkpoint(checkpoint_path)
    checkpoint = reference(checkpoint_path)
    _restore_proof(model, checkpoint, folder / 'checkpoint_restore_verification.json')
    _immutable_json(folder / 'prepared.json', dict(checkpoint=checkpoint, intent=intent,
                                                     training_targets=model.runtime_state['training_targets']))
    ledger.commit(update, jobs, checkpoint, batch_id)
    write_progress(suite, status='UPDATE_COMMITTED', phase_id=cfg['phase_id'], candidate_id=CANDIDATE,
                   update_index=update, completed_epochs=config['ppo']['epochs'], epochs=config['ppo']['epochs'])
    print('TRAINING_UPDATE_COMMITTED update=' + str(update), flush=True)
    return model, checkpoint
