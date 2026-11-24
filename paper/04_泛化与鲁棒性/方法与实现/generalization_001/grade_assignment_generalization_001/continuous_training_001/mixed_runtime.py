"""Run the frozen mixed-grade BG128 campaign and paired fixed-policy evaluation."""
from collections import Counter
from contextlib import contextmanager
from importlib.util import module_from_spec, spec_from_file_location
from functools import lru_cache
import hashlib
from pathlib import Path
from statistics import fmean
from threading import RLock

import numpy as np
import torch

from baseline1_v1.evaluation import ReadOnlyTrainingGuard
from baseline1_v1.java_plan import COLUMNS
from business_graph_summary_campaign_v1 import behavior as campaign_behavior
from business_graph_summary_campaign_v1.admission import _check_actor_phase, _verify_sampling, frozen_actor
from business_graph_summary_campaign_v1.collection import MaterialPhaseCollector, OnlineActor, load_admitted_episode
from business_graph_summary_campaign_v1.gate import PROTECTED_SHA, check_phase
from business_graph_summary_campaign_v1.plan_io import encoded, load_phase, load_reference, reference
from business_graph_summary_campaign_v1.progress import update_callback, write_progress
from business_graph_summary_campaign_v1.runtime import (
    _compare_initializations, _learning_contract, _same_or_save, _targets, _verify_target_ref,
)
from business_graph_summary_evaluation_v1.metrics import build_report, write_reports
from business_graph_summary_evaluation_v1.runtime import _portable_state_fingerprint
from business_graph_summary_v1 import behavior as learner_behavior
from business_graph_summary_v1 import learner as frozen_learner
from business_graph_summary_v1.contract import require
from business_graph_summary_v1.evaluation import evaluate_episodes
from business_graph_summary_v1.models import digest
from grade_only_ppo_v2.runtime import _actor_snapshot, _equal_tree
from online_v1_1.operations import _publish
from python.constrained_batch import load_dataset
from python.observation_contract import FIELD_NAMES
from python.runtime_scenario import PREFIX, fingerprint


PLAN = Path(__file__).resolve().parent
ROOT = Path(frozen_learner.__file__).resolve().parents[1]
REPORT = ROOT / 'reports/token_attention_001/business_graph_summary_009/generalization_001/continuous_training_001'
EXTENSION = PLAN.parents[2] / 'training_extension_001/train_extension.py'
CANDIDATE = 's01_BG128'
UPDATE_COUNT = 32
BATCH_SIZE = 32
EVAL_SHIFTS = 100
SCENARIO_COUNT = 7
HISTORICAL_SHA = '5d42df7d8e33b5f11424094268a4da894a36454e46d86889fc0218ecce6b2914'
MATERIAL_FIELDS = ('isOre', 'meanGrade', 'gradeHalfRange')
MATERIAL_RPC_FIELDS = ('isOre', 'grade', 'gradeHalfRange')
MATERIAL_COLUMNS = [FIELD_NAMES.index(f'load.{site}.{field}') for site in range(4) for field in MATERIAL_FIELDS]
VERIFY_LOCK = RLock()


@lru_cache(maxsize=1)
def _extension_module():
    spec = spec_from_file_location('mixed_bg128_extension_bounds', EXTENSION)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _learner_class():
    return _extension_module()._extension_learner()


def _require_reference(ref):
    require(reference(ref['path']) == ref, 'frozen artifact changed: ' + ref['path'])
    return ref


def _check_sources(suite):
    for path, sha256 in suite['source_pins'].items():
        _require_reference(dict(path=path, sha256=sha256))
    require(reference(ROOT.parents[1] / 'python/episode_budget_ppo.py')['sha256'] == PROTECTED_SHA,
            'protected PPO core changed')


def _material_parameters(job):
    rpc = job['expected_rpc_config']
    values = np.array([rpc[f'{PREFIX}load.{site}.{field}']
                       for site in range(4) for field in MATERIAL_RPC_FIELDS], dtype=np.float32).reshape(4, 3)
    flags, grades, widths = values.T
    require(np.isin(flags, (0, 1)).all() and int(flags.sum()) == 3,
            'expected exactly three ore sites and one waste site')
    ore = flags.astype(bool)
    require(grades[~ore].item() == 0 and (widths >= 0).all()
            and (grades[ore] >= widths[ore]).all()
            and (grades[ore] <= 1 - widths[ore]).all(), 'invalid ore support or waste grade')
    return values, ore


def verify_material_observation(states, job):
    """Bind all four physical slots to the planned commodity flags and grade laws."""
    values, ore = _material_parameters(job)
    observations = np.asarray(states)
    require(observations.dtype == np.float32 and observations.ndim in (1, 2)
            and observations.shape[-1] == 1519, 'material validation needs the original physical observation')
    require(np.equal(observations[..., MATERIAL_COLUMNS], values.ravel()).all(),
            'observed ore flags or grade parameters differ from the frozen physical scene')
    return ore


def _verify_observation_sources(episode, job):
    ore = verify_material_observation(episode['states'], job)
    for event in episode['memory'].events:
        if event.kind in ('UNLOAD', 'PROCESS'):
            require(0 <= event.source_id < len(ore) and ore[event.source_id],
                    'crusher material came from a non-ore or unknown load site')


def verify_online(actor, states, actions, logs, replay=None, *, rows=None, job=None):
    """Authenticate the actual one-row Actor and every recorded private RNG draw."""
    require(rows is not None and job is not None and len(rows) == len(states) == len(actions) == len(logs) > 0,
            'exact online verification needs complete sealed decision rows and a frozen job')
    verify_material_observation(states, job)
    exact = learner_behavior.exact_replay(actor, states, actions)
    exact_logs = exact['log_probs'].detach().numpy().astype('>f8')
    recorded_logs = np.asarray(logs, dtype='>f8')
    require(np.isfinite(exact_logs).all() and exact_logs.shape == recorded_logs.shape
            and exact_logs.tobytes() == recorded_logs.tobytes(), 'exact online log probabilities differ from stored bits')
    _verify_sampling(rows, exact['probabilities'], job)
    if replay is not None:
        batch = replay['probabilities'].detach().numpy()
        require(batch.shape == (len(rows), 4) and np.isfinite(batch).all(), 'invalid batch replay probabilities')
    return 'exact_online'


def verify_sealed_behavior(actor_path, raw_directory, job):
    """Offline probe over a real sealed shift. No runtime, output or ledger side effects."""
    actor, version = frozen_actor(actor_path)
    reader, episodes, _, _, _ = load_dataset(raw_directory, require_real=True)
    try:
        rows = list(reader)
    finally:
        if hasattr(reader, 'close'):
            reader.close()
    require(bool(rows) and len(episodes) == 1 and episodes[0]['normal_completion']
            and episodes[0]['experiment_run_index'] == job['run_index']
            and episodes[0]['experiment_seed'] == job['experiment_seed']
            and episodes[0]['policy_version'] == version, 'offline probe requires one matching sealed physical shift')
    states = np.stack([rows[0]['observation']] + [row['next_observation'] for row in rows])
    verify_material_observation(states, job)
    actions = np.array([row['action'] for row in rows], dtype=np.int64)
    logs = [row['old_log_prob'] for row in rows]
    return dict(mode=verify_online(actor, states[:-1], actions, logs, rows=rows, job=job),
                decisions=len(rows), run_index=job['run_index'], actor_digest=version)


def verify_learner_behavior(model, episode, current, job):
    """Rebind the optimizer preflight to sealed RPC rows and exact online RNG."""
    identity = episode['identity']
    artifact = model._actor_artifact
    require(np.isfinite(current.detach().numpy()).all() and isinstance(artifact, dict)
            and artifact['actor_digest'] == model.policy_version == identity['policy_version']
            and artifact['sha256'] == identity['policy_sha256']
            and identity['run_index'] == job['run_index']
            and identity['actor_sampling_seed'] == job['actor_sampling_seed'],
            'learner trajectory is not from the current sealed Actor/run')
    require(model._validated_actor_snapshot(artifact['path'])['sha256'] == artifact['sha256'],
            'bound Actor seal changed')
    raw = Path(artifact['path']).parent / 'episodes' / f"run_{identity['run_index']}"
    require(reference(raw / 'manifest.json')['sha256'] == identity['raw_manifest_sha256'],
            'learner raw trajectory manifest changed')
    reader, _, _, _, _ = load_dataset(raw, require_real=True)
    try:
        rows = list(reader)
    finally:
        if hasattr(reader, 'close'):
            reader.close()
    require(bool(rows), 'learner trajectory is empty')
    states = np.stack([rows[0]['observation']] + [row['next_observation'] for row in rows])
    actions = np.array([row['action'] for row in rows], dtype=np.int64)
    logs = np.array([row['old_log_prob'] for row in rows], dtype=np.float64)
    require(states.tobytes() == episode['states'].tobytes() and np.array_equal(actions, episode['actions'])
            and logs.tobytes() == np.asarray(episode['old_log_probs'], dtype=np.float64).tobytes(),
            'learner inputs differ from sealed observations, actions or old log probabilities')
    verify_material_observation(states, job)
    return verify_online(model.actor, states[:-1], actions, logs, rows=rows, job=job)


@contextmanager
def _exact_verification(jobs):
    """Scope the two frozen replay callbacks to this serial controller phase."""
    by_run = {job['run_index']: job for job in jobs}
    with VERIFY_LOCK:
        old_collection = campaign_behavior.verify
        old_learner = learner_behavior.verify_learner
        def learner(model, episode, current):
            return verify_learner_behavior(model, episode, current, by_run[episode['identity']['run_index']])
        campaign_behavior.verify = verify_online
        learner_behavior.verify_learner = learner
        try:
            yield
        finally:
            campaign_behavior.verify = old_collection
            learner_behavior.verify_learner = old_learner


class MixedOnlineActor(OnlineActor):
    def __init__(self, actor, jobs):
        super().__init__(actor, jobs)
        self._mixed_jobs = {job['run_index']: job for job in jobs}

    def sample_with_info(self, observation, mask, info):
        job = self._mixed_jobs[info['experiment_run_index']]
        verify_material_observation(observation, job)
        actual = info['reward_components']
        expected = job['expected_rpc_config']
        require(all(actual.get(key) == expected[key] for key in
                    (f'{PREFIX}load.{site}.{field}' for site in range(4) for field in MATERIAL_RPC_FIELDS)),
                'decision RPC and frozen grade parameters differ')
        return super().sample_with_info(observation, mask, info)


def _old_model(suite):
    checkpoint = _require_reference(suite['old_checkpoint'])
    require(checkpoint['sha256'] == HISTORICAL_SHA, 'wrong historical u32 checkpoint')
    cfg = suite['old_learner_config']
    for ref in suite['old_artifacts'].values():
        _require_reference(ref)
    return _extension_module().load_checkpoint(checkpoint, cfg, restore=False)


def preflight(suite, configs):
    """Read-only check of frozen package, fresh learner contract and historical reference."""
    _check_sources(suite)
    require(len(suite['arms']) == 1 and suite['arms'][0]['candidate_id'] == CANDIDATE
            and len(configs) == len(suite['phase_configs']) == UPDATE_COUNT + 2 * SCENARIO_COUNT,
            'mixed campaign arm and phase inventory differ')
    jobs = [job for cfg in configs for job in cfg['jobs']]
    require(jobs == suite['jobs'] and len(jobs) == UPDATE_COUNT * BATCH_SIZE + 2 * SCENARIO_COUNT * EVAL_SHIFTS
            and len({job['run_index'] for job in jobs}) == len(jobs), 'mixed campaign job inventory differs')
    for job in jobs:
        _material_parameters(job)
        scene = {key: value for key, value in job['expected_rpc_config'].items() if key.startswith(PREFIX)}
        require(fingerprint(scene) == job['runtime_scene_sha256'], 'mixed job physical scene fingerprint differs')
    pairs = {}
    for index, (cfg, ref) in enumerate(zip(configs, suite['phase_configs'], strict=True)):
        _require_reference(ref)
        require(load_phase(ref['path']) == cfg and cfg['campaign_id'] == suite['campaign_id'],
                'phase differs from the frozen configuration')
        check_phase(cfg)
        arm, update = cfg['policy_arm'], cfg['arm_update']
        if index < UPDATE_COUNT:
            require(arm == 'train' and update == index + 1 and len(cfg['jobs']) == BATCH_SIZE,
                    'mixed training phase count or order differs')
            require(all(job['role'] == job['partition'] == 'train' and job['checkpoint_update'] == update - 1
                        and job['target_training_update'] == update for job in cfg['jobs']), 'training job binding differs')
        else:
            require(arm in ('old', 'new') and update == UPDATE_COUNT and len(cfg['jobs']) == EVAL_SHIFTS,
                    'fixed evaluation phase binding differs')
            scenario = cfg['scenario_id']
            require((scenario, arm) not in pairs, 'duplicate fixed evaluation phase')
            require(all(job['role'] == 'final' and job['partition'] == 'validation' and job['scene_id'] == scenario
                        and job['checkpoint_update'] == UPDATE_COUNT and job['target_training_update'] == 0
                        and job['pair_ordinal'] == i for i, job in enumerate(cfg['jobs'])), 'evaluation job binding differs')
            pairs[scenario, arm] = [_pair(job) for job in cfg['jobs']]
    scenarios = {key[0] for key in pairs}
    require(len(scenarios) == SCENARIO_COUNT and all((s, arm) in pairs for s in scenarios for arm in ('old', 'new'))
            and all(pairs[s, 'old'] == pairs[s, 'new'] for s in scenarios), 'fixed evaluation pairs differ')
    _require_reference(suite['master_plan'])
    with ReadOnlyTrainingGuard():
        historical = _old_model(suite)
        fresh = _learner_class()(suite['arms'][0]['learner_config'])
        require(fresh.update_index == 0 and not fresh.consumed_episodes and not fresh.actor_optimizer.state
                and not fresh.critic_optimizer.state and fresh.config['seed'] != historical.config['seed'],
                'fresh mixed learner origin differs')
    return dict(training_phases=UPDATE_COUNT, training_shifts=UPDATE_COUNT * BATCH_SIZE,
                evaluation_phases=2 * SCENARIO_COUNT, evaluation_shifts=2 * SCENARIO_COUNT * EVAL_SHIFTS,
                old_checkpoint=suite['old_checkpoint'], old_actor_digest=historical.policy_version)


def _pair(job):
    return job['experiment_seed'], job['actor_sampling_seed'], job['runtime_scene_sha256']


def _ready_publisher(suite, phase_ref, checkpoint, ledger):
    control = Path(suite['control_directory'])
    def publish(job, config_path):
        _require_reference(checkpoint)
        require(reference(config_path) == phase_ref and phase_ref in suite['phase_configs'],
                'unfrozen physical phase at READY')
        row = ledger.job(job)
        require(row['status'] == 'CLAIMED' and row['identity'] == job and not (control / 'STOP.properties').exists(),
                'READY requires a fresh live claim and no STOP')
        values = {name: str(job[name]) for name in COLUMNS}
        values.update(protocol='baseline1-one-click-ready-v1', campaign_id=suite['campaign_id'],
                      master_plan_sha256=suite['master_plan']['sha256'],
                      instrumentation_revision='boundary-events-v1-native-rpc-runtime-scene-v1',
                      config_sha256=phase_ref['sha256'], snapshot_sha256=phase_ref['sha256'],
                      all_comparison_checkpoints_frozen=str(job['role'] != 'train').lower(),
                      snapshot_path=phase_ref['path'])
        require(all(value.isascii() and '\n' not in value and '\r' not in value for value in values.values()),
                'invalid READY Properties')
        path = control / 'ready' / f"run_{job['run_index']}.properties"
        path.parent.mkdir(parents=True, exist_ok=True)
        _publish(path, ''.join(key + '=' + value + '\n' for key, value in values.items()).encode('ascii'))
    return publish


def _collect(suite, cfg, phase_ref, checkpoint, ledger, model, *, actor_ref=None):
    if actor_ref is None:
        actor_ref = _actor_snapshot(model, Path(cfg['python_output']) / 'actor.pt')
    _check_actor_phase(frozen_actor(actor_ref['path'])[0], cfg)
    rows = ledger.resume()['jobs']
    require(all(rows[str(job['run_index'])]['status'] in ('PLANNED', 'COMPLETE') for job in cfg['jobs']),
            'unresolved physical claim prevents collection')
    remaining = [job for job in cfg['jobs'] if rows[str(job['run_index'])]['status'] == 'PLANNED']
    fresh = {}
    if remaining:
        active = dict(cfg, jobs=remaining)
        behavior = MixedOnlineActor(model.actor, remaining)
        with MaterialPhaseCollector(active, phase_ref['path'], publish_ready=_ready_publisher(suite, phase_ref, checkpoint, ledger),
                startup=dict(control_directory=suite['control_directory'], campaign_id=suite['campaign_id'],
                             master_plan_sha256=suite['master_plan']['sha256'])) as collector:
            for job in remaining:
                write_progress(suite, status='COLLECTING', policy_arm=cfg['policy_arm'], phase_id=cfg['phase_id'],
                               run_index=job['run_index'], ordinal=job['ordinal'])
                ledger.claim(job)
                try:
                    episode, dataset_ref, receipt = collector.collect(job, behavior, actor_path=actor_ref['path'])
                    _verify_observation_sources(episode, job)
                    ledger.complete(job, dataset_ref, receipt)
                    fresh[job['run_index']] = episode
                except BaseException as error:
                    if ledger.job(job)['status'] == 'CLAIMED':
                        ledger.unknown(job, repr(error))
                    raise
    episodes, sources = [], []
    rows = ledger.resume()['jobs']
    for job in cfg['jobs']:
        row = rows[str(job['run_index'])]
        require(row['status'] == 'COMPLETE' and row['identity'] == job and row['receipt']['phase_config'] == phase_ref,
                'complete physical run and frozen receipt differ')
        sources.append(_require_reference(row['dataset_ref']))
        episode = fresh.pop(job['run_index'], None)
        if episode is None:
            episode = load_admitted_episode(row['receipt'])
            _verify_observation_sources(episode, job)
        identity = episode['identity']
        expected = dict(run_index=job['run_index'], environment_seed=job['experiment_seed'],
                        actor_sampling_seed=job['actor_sampling_seed'], scene_id=job['scene_id'],
                        scene_group=job['scene_group'], split=job['partition'], recording_id=job['recording_id'],
                        policy_sha256=actor_ref['sha256'], policy_version=model.policy_version)
        require(all(identity[key] == value for key, value in expected.items()), 'admitted episode identity differs')
        episodes.append(episode)
    return episodes, sources, actor_ref


def _load_new(suite, checkpoint, update, *, restore=False):
    config = suite['arms'][0]['learner_config']
    model = _learner_class().from_checkpoint(checkpoint['path'], expected_config=config,
        expected_sha256=checkpoint['sha256'], restore_global_rng=restore)
    require(model.runtime_state['campaign'] == reference(suite['suite_path'])
            and model.runtime_state['candidate_id'] == CANDIDATE and model.update_index == update
            and len(model.consumed_episodes) == update * BATCH_SIZE
            and len(model.consumed_batches) == update, 'mixed checkpoint lineage or consumption differs')
    return model


def _initial(suite):
    arm = suite['arms'][0]
    output = Path(arm['python_output'])
    ref_path, checkpoint = output / 'initial_checkpoint.json', output / 'initial.pt'
    if ref_path.is_file():
        ref = load_reference(reference(ref_path))
        model = _load_new(suite, ref, 0)
    else:
        require(not checkpoint.exists(), 'initial checkpoint exists without its immutable reference')
        model = _learner_class()(arm['learner_config'])
        model.runtime_state.update(campaign=reference(suite['suite_path']), candidate_id=CANDIDATE)
        model.save_checkpoint(checkpoint)
        ref = reference(checkpoint)
        _same_or_save(ref_path, ref)
    require(not model.last_targets and not model.actor_optimizer.state and not model.critic_optimizer.state,
            'mixed initial policy did not originate at fresh u0')
    _compare_initializations(suite, model, ref)
    return ref


def _check_saved_update(suite, prepared, folder, intent, update):
    require(prepared['intent'] == intent and prepared['checkpoint']['path'] == str(folder / 'checkpoint.pt'),
            'published mixed training transaction differs')
    model = _load_new(suite, _require_reference(prepared['checkpoint']), update, restore=True)
    require(model.runtime_state['transaction'] == intent
            and prepared['training_targets'] == model.runtime_state['training_targets']
            and model.runtime_state['training_summary'] == reference(folder / 'training_summary.json'),
            'mixed training transaction receipts differ')
    _verify_target_ref(prepared['training_targets'], intent)
    return prepared['checkpoint']


def _batch_scene_statistics(episodes, jobs):
    stats = {}
    for episode, job in zip(episodes, jobs, strict=True):
        row = stats.setdefault(job['scene_id'], Counter(shifts=0, decisions=0, complete_task=0))
        row['shifts'] += 1
        row['decisions'] += len(episode['actions'])
        physical = episode['evaluation']
        row['complete_task'] += int(physical['ore_tons'] >= 16000 and physical['waste_tons'] >= 4000
            and all(window.grade_valid and .26 - 1e-12 <= window.grade <= .28 + 1e-12
                    for window in episode['memory'].settlements))
    return {key: dict(value) for key, value in stats.items()}


def _train(suite, cfg, phase_ref, parent, ledger):
    update = cfg['arm_update']
    folder = Path(suite['arms'][0]['python_output']) / 'updates' / f'{update:04d}'
    checkpoint_path = folder / 'checkpoint.pt'
    record = ledger.commit_record(update)
    if record is None:
        model = _load_new(suite, parent, update - 1, restore=True)
        episodes, sources, _ = _collect(suite, cfg, phase_ref, parent, ledger, model)
    else:
        rows = ledger.resume()['jobs']
        physical = [rows[str(job['run_index'])] for job in cfg['jobs']]
        require(all(row['status'] == 'COMPLETE' and row['committed_update'] == update for row in physical),
                'committed mixed update physical ledger differs')
        sources = [_require_reference(row['dataset_ref']) for row in physical]
    transaction = dict(campaign=reference(suite['suite_path']), phase_config=phase_ref, parent_checkpoint=parent,
                       sources=sources, update=update)
    batch_id = hashlib.sha256(encoded(transaction)).hexdigest()
    intent = dict(schema='bg128-mixed-update-v1', **transaction, batch_id=batch_id,
                  learning_contract=_learning_contract(dict(suite, pair_id=suite.get('pair_id', suite['campaign_id'])), CANDIDATE),
                  run_indices=[job['run_index'] for job in cfg['jobs']])
    prepared_path = folder / 'prepared.json'
    if record is not None:
        prepared = load_reference(reference(prepared_path))
        require(record['checkpoint_ref'] == prepared['checkpoint'] and record['batch_id'] == batch_id,
                'mixed ledger commit differs from the prepared checkpoint')
        return _check_saved_update(suite, prepared, folder, intent, update)
    intent_path = folder / 'intent.json'
    had_intent = intent_path.exists()
    _same_or_save(intent_path, intent)
    if checkpoint_path.is_file():
        model = _load_new(suite, reference(checkpoint_path), update, restore=True)
        _check_saved_update(suite, dict(intent=intent, checkpoint=reference(checkpoint_path),
            training_targets=model.runtime_state['training_targets']), folder, intent, update)
    else:
        require(not had_intent and not prepared_path.exists(), 'unsealed optimizer transaction requires review')
        model.progress_callback = update_callback(suite, cfg, update)
        try:
            result = model.update(episodes, batch_id=batch_id)
        finally:
            model.progress_callback = None
        require(model.update_index == update and len(model.consumed_episodes) == update * BATCH_SIZE
                and next(reversed(model.consumed_batches)) == batch_id, 'mixed optimizer consumption differs')
        targets = _targets(folder / 'training_targets.h5', model.last_targets, intent)
        summary = folder / 'training_summary.json'
        _same_or_save(summary, dict(transaction=intent, update=result,
            scenario_statistics=_batch_scene_statistics(episodes, cfg['jobs']),
            exact_learner_decisions=sum(len(episode['actions']) for episode in episodes)))
        model.runtime_state.update(transaction=intent, training_targets=targets, training_summary=reference(summary))
        model.save_checkpoint(checkpoint_path)
    checkpoint = reference(checkpoint_path)
    restored = _load_new(suite, checkpoint, update)
    checks = _extension_module()._checkpoint_checks(model, restored)
    require(all(checks.values()), 'mixed checkpoint exact restore differs')
    _same_or_save(folder / 'checkpoint_restore_verification.json', dict(checkpoint=checkpoint, checks=checks))
    _same_or_save(prepared_path, dict(checkpoint=checkpoint, intent=intent,
                                    training_targets=model.runtime_state['training_targets']))
    ledger.commit(update, cfg['jobs'], checkpoint, batch_id)
    write_progress(suite, status='UPDATE_COMMITTED', phase_id=cfg['phase_id'], update_index=update)
    return checkpoint


def _evaluation_sources(episodes, jobs):
    decisions = [0] * 4
    ore_tons, ore_metal = [0.0] * 4, [0.0] * 4
    by_window = [[dict(tons=0.0, metal_tons=0.0) for _ in range(4)] for _ in range(4)]
    for episode, job in zip(episodes, jobs, strict=True):
        _verify_observation_sources(episode, job)
        for site, count in enumerate(np.bincount(episode['actions'], minlength=4).tolist()):
            decisions[site] += count
        for event in episode['memory'].events:
            if event.kind == 'UNLOAD':
                ore_tons[event.source_id] += event.tons
                ore_metal[event.source_id] += event.metal
            elif event.kind == 'PROCESS':
                row = by_window[event.window][event.source_id]
                row['tons'] += event.tons
                row['metal_tons'] += event.metal
    return dict(actions_by_site=decisions, ore_unloaded_tons_by_site=ore_tons,
                ore_unloaded_metal_tons_by_site=ore_metal, processed_window_by_site=by_window)


def _evaluate(suite, cfg, phase_ref, checkpoint, ledger):
    arm = cfg['policy_arm']
    with ReadOnlyTrainingGuard():
        model = _old_model(suite) if arm == 'old' else _load_new(suite, checkpoint, UPDATE_COUNT)
        output = Path(cfg['python_output'])
        actor_ref = _actor_snapshot(model, output / 'actor.pt')
        before = _portable_state_fingerprint(model)
        with ReadOnlyTrainingGuard(model):
            episodes, sources, actor = _collect(suite, cfg, phase_ref, checkpoint, ledger, model, actor_ref=actor_ref)
            write_progress(suite, status='EVALUATING', policy_arm=arm, phase_id=cfg['phase_id'])
            report = evaluate_episodes(episodes, model, expected_complete_shifts=EVAL_SHIFTS,
                                       checkpoint_provenance=checkpoint)
        require(before == _portable_state_fingerprint(model), 'fixed evaluation mutated training state')
        evidence = _evaluation_sources(episodes, cfg['jobs'])
        report.update(policy_arm=arm, scenario_id=cfg['scenario_id'], phase_config=phase_ref,
                      source_refs=sources, scenario_evidence=evidence, actor_artifact=actor,
                      model_state_fingerprint=before)
        require(report['fingerprints'] == dict(actor=digest(model.actor), critic=digest(model.critic)),
                'evaluation model fingerprints differ')
        _same_or_save(output / 'evaluation_complete.json', report)
        summary = build_report(report, expected_complete_shifts=EVAL_SHIFTS)
        summary.update(policy_arm=arm, scenario_id=cfg['scenario_id'], scenario_evidence=evidence)
        published = write_reports(summary, report, output / 'reports')
    return dict(checkpoint=checkpoint, reports=published, report=reference(output / 'evaluation_complete.json'))


def _window_outcomes(records):
    return dict(upper=sum(w['grade'] is not None and w['grade'] > .28 + 1e-12 for w in records),
                lower=sum(w['grade'] is not None and w['grade'] < .26 - 1e-12 for w in records),
                empty=sum(w['empty'] for w in records))


def _comparison(suite, configs, results):
    cases = {}
    for cfg, result in zip(configs, results, strict=True):
        report = load_reference(result['report'])
        key = cfg['scenario_id'], cfg['policy_arm']
        require(report['scenario_id'] == key[0] and report['policy_arm'] == key[1]
                and report['checkpoint_provenance'] == result['checkpoint'], 'paired evaluation report binding differs')
        cases.setdefault(key[0], {})[key[1]] = (cfg, result, report)
    summary = {}
    rows = []
    for scenario, arms in cases.items():
        (old_cfg, old_result, old_report), (new_cfg, new_result, new_report) = arms['old'], arms['new']
        paired = []
        for job_a, job_b, row_a, row_b in zip(old_cfg['jobs'], new_cfg['jobs'], old_report['per_episode'],
                                               new_report['per_episode'], strict=True):
            require(_pair(job_a) == _pair(job_b) and job_a['run_index'] != job_b['run_index']
                    and job_a['pair_ordinal'] == job_b['pair_ordinal'], 'evaluation pairing differs')
            item = dict(scenario_id=scenario, pair_ordinal=job_a['pair_ordinal'],
                        environment_seed=job_a['experiment_seed'], actor_sampling_seed=job_a['actor_sampling_seed'])
            for arm, job, record in (('old', job_a, row_a), ('new', job_b, row_b)):
                identity = record['identity']
                require(identity['run_index'] == job['run_index'] and identity['environment_seed'] == job['experiment_seed']
                        and identity['actor_sampling_seed'] == job['actor_sampling_seed'], 'paired record has the wrong identity')
                physical = record['physical']
                item[arm] = dict(run_index=job['run_index'],
                    **{field: physical[field] for field in ('complete_task', 'all_four_windows_compliant', 'ore_target_met',
                       'waste_target_met', 'ore_tons', 'waste_tons', 'total_wait_seconds', 'combined_cost')},
                    window_outcomes=_window_outcomes(physical['windows']))
            paired.append(item)
        rows.extend(paired)
        outcomes = {}
        for key in ('complete_task', 'all_four_windows_compliant', 'ore_target_met', 'waste_target_met'):
            a = [pair['old'][key] for pair in paired]
            b = [pair['new'][key] for pair in paired]
            outcomes[key] = dict(old=sum(a), new=sum(b), improved=sum(not x and y for x, y in zip(a, b)),
                                 worsened=sum(x and not y for x, y in zip(a, b)))
        summary[scenario] = dict(outcomes=outcomes, old=old_result['reports']['summary'], new=new_result['reports']['summary'],
            mean_new_minus_old={key: fmean(row['new'][key] - row['old'][key] for row in paired)
                                for key in ('total_wait_seconds', 'combined_cost', 'ore_tons', 'waste_tons')})
    directory = Path(suite.get('report_directory', REPORT))
    shape = dict(schema='bg128-mixed-paired-v1', campaign=reference(suite['suite_path']),
                 paired_slots_per_case=EVAL_SHIFTS, evaluation_shifts=len(rows) * 2, comparisons=summary, rows=rows,
                 uncertainty='one historical and one mixed training seed; pairing controls evaluation seeds')
    json_path = directory / 'paired_evaluation.json'
    _same_or_save(json_path, shape)
    lines = ['# BG128 混合品位训练固定策略对照', '', '每个工况采用 100 对新评估种子。', '',
             '| 工况 | 原 u32 完整任务 | 混合 u32 完整任务 | 改善 | 退化 | 平均等待差（车辆秒） |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for scenario, result in summary.items():
        task = result['outcomes']['complete_task']
        wait = result['mean_new_minus_old']['total_wait_seconds']
        lines.append(f"| {scenario} | {task['old']}/100 | {task['new']}/100 | {task['improved']} | {task['worsened']} | {wait:+.2f} |")
    markdown = directory / 'paired_evaluation.md'
    data = ('\n'.join(lines) + '\n').encode()
    if markdown.exists():
        require(markdown.read_bytes() == data, 'immutable paired evaluation Markdown changed')
    else:
        _publish(markdown, data)
    return dict(json=reference(json_path), markdown=reference(markdown))


def run(suite, configs, ledger):
    """Run an explicitly authorized, serial campaign through all fixed evaluations."""
    torch.set_num_threads(suite['cpu_threads'])
    preflight(suite, configs)
    ledger.register()
    initial = parent = _initial(suite)
    checkpoints = []
    results = []
    for cfg, phase_ref in zip(configs, suite['phase_configs'], strict=True):
        _check_sources(suite)
        with _exact_verification(cfg['jobs']):
            if cfg['policy_arm'] == 'train':
                parent = _train(suite, cfg, phase_ref, parent, ledger)
                checkpoints.append(parent)
            else:
                require(len(checkpoints) == UPDATE_COUNT, 'fixed evaluation requires committed mixed u32')
                checkpoint = suite['old_checkpoint'] if cfg['policy_arm'] == 'old' else parent
                results.append(_evaluate(suite, cfg, phase_ref, checkpoint, ledger))
    state = ledger.resume()
    commits = state['campaigns'][suite['campaign_id']]['commits']
    require(set(commits) == {str(i) for i in range(1, UPDATE_COUNT + 1)} and len(results) == 2 * SCENARIO_COUNT,
            'mixed campaign completion counts differ')
    require(all(state['jobs'][str(job['run_index'])]['status'] == 'COMPLETE'
                and ('committed_update' in state['jobs'][str(job['run_index'])]) == (job['role'] == 'train')
                for job in suite['jobs']), 'mixed physical ledger completion differs')
    comparison = _comparison(suite, configs[UPDATE_COUNT:], results)
    report = dict(status='COMPLETE', campaign=reference(suite['suite_path']), initial_checkpoint=initial,
                  new_checkpoint=parent, old_checkpoint=suite['old_checkpoint'], training_checkpoints=checkpoints,
                  training_updates=UPDATE_COUNT, training_shifts=UPDATE_COUNT * BATCH_SIZE,
                  evaluation_shifts=2 * SCENARIO_COUNT * EVAL_SHIFTS, evaluation_optimizer_updates=0,
                  phase_results=results, comparison=comparison)
    _check_sources(suite)
    _same_or_save(Path(suite['python_output']) / 'suite_complete.json', report)
    write_progress(suite, status='COMPLETE', training_updates=UPDATE_COUNT, completed_shifts=len(suite['jobs']))
    return report
