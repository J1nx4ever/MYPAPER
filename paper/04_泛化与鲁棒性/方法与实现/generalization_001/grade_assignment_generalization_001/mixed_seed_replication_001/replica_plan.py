"""Freeze four BG128 seed replications without starting a simulation."""

from collections import Counter
from hashlib import sha256
from pathlib import Path
from random import Random
import subprocess
import sys
from tempfile import TemporaryDirectory

PLAN = Path(__file__).resolve().parent
ROOT = PLAN.parents[5]
MIXED = PLAN.with_name('continuous_training_001')
for directory in (ROOT, MIXED, MIXED / 'recovery_003'):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import mixed_plan as parent
from mixed_java import _replace_once, scenario_table_bytes, verify_java, write_java
from business_graph_summary_campaign_v1.plan_io import load_reference
from python.collect_offline import validate_runtime

REPLICAS = ('s02', 's03', 's04', 's05')
TRAIN = parent.TRAINING_SHIFTS
EVAL = len(parent.EVAL_CASES) * parent.EVAL_COUNT
CAP = TRAIN + EVAL
PREPARED = PLAN / 'prepared_oneclick_003'
ENTRY = PLAN / 'replica_run.py'
ANCHOR = parent.OUTPUT / 'suite_complete.json'
ANCHOR_SHA = 'e32326901f5c070c5bd64160c8978ea0b491947da7eb159f380c8ddef945bc9d'
REPORT = parent.REPORT.with_name('mixed_seed_replication_001')
SPEED_STD = parent.PREFIX + 'env.freeSpeedFactorStd'
SPEED_FIELDS = dict(freeSpeedFactorStd=.05, freeSpeedFactorMin=.85, freeSpeedFactorMax=1.10,
                    useLocalDynamicSpeed=1., useEndogenousCongestion=1., deterministicCheck=0.)


def locations(replica):
    position = REPLICAS.index(replica)
    campaign = f'bg128-continuous-mixed-20261120-{replica}'
    output = ROOT / 'runs' / campaign
    return dict(campaign_id=campaign, python_output=str(output), control_directory=str(output / 'control'),
                budget_path=str(output / 'budget.sqlite'), material_root=str(output.with_name(campaign + '_materials')),
                report_directory=str(REPORT / replica), suite_path=str(PREPARED / replica / 'suite.json'),
                run_index_base=10003001 + position * 2000, rpc_port=50051)


def _check_speed(rpc):
    parent.require(all(rpc[parent.PREFIX + 'env.' + field] == expected
                       for field, expected in SPEED_FIELDS.items()), 'speed/congestion differs from seed 01')


def anchored_source(*, check_rpc=False):
    anchor = dict(path=str(ANCHOR), sha256=ANCHOR_SHA)
    completion = load_reference(anchor)
    original, phases = parent.load_suite(completion['campaign']['path'])
    parent.require(completion['status'] == 'COMPLETE' and parent.reference(parent.SUITE) == completion['campaign']
                   and original['source_pins'] == parent.source_pins(), 'seed 01 frozen training source differs')
    for phase in phases:
        for job in phase['jobs']:
            _check_speed(job['expected_rpc_config'])
    training = phases[:parent.UPDATES]
    evaluation = [phase for phase in phases if phase['policy_arm'] == 'new']
    parent.require(len(training) == 32 and [cfg['scenario_id'] for cfg in evaluation] == list(parent.EVAL_CASES),
                   'seed 01 phase order differs')
    baseline = {}
    for cfg, result in zip(phases[32:], completion['phase_results'], strict=True):
        report = load_reference(result['report'])
        parent.require(report['scenario_id'] == cfg['scenario_id'] and report['policy_arm'] == cfg['policy_arm']
                       and report['n_shifts'] == 100 and report['evaluation_optimizer_updates'] == 0
                       and report['training_state_unchanged'] is True, 'seed 01 fixed evaluation evidence differs')
        if cfg['policy_arm'] == 'old':
            baseline[cfg['scenario_id']] = result
        for job, episode in zip(cfg['jobs'], report['per_episode'], strict=True):
            identity = episode['identity']
            parent.require((identity['run_index'], identity['environment_seed'], identity['actor_sampling_seed'])
                           == (job['run_index'], job['experiment_seed'], job['actor_sampling_seed']),
                           'seed 01 fixed evaluation slot differs')
    evidence = []
    if check_rpc:
        for cfg in [*training, *evaluation]:
            job = cfg['jobs'][0]
            folder = Path(cfg['python_output']) / 'episodes' / f"run_{job['run_index']}"
            raw = parent.reference(folder / 'episodes.json')
            manifest = parent.read(folder / 'manifest.json')
            parent.require(manifest['files_sha256']['episodes.json'] == raw['sha256'], 'seed 01 RPC evidence changed')
            start = load_reference(raw)[0]['start_info']
            _check_speed(start['reward_components'])
            validate_runtime(start, job)
            evidence.append(raw)
    return original, training, evaluation, baseline, evidence


def _scenes(campaign, update):
    rng = Random(parent.seed(campaign, 'configuration', update))
    choices = [''.join(order) for order in parent.permutations(parent.ROLES)
               if ''.join(order) not in parent.HOLDOUTS]
    scenes = []
    for site in range(4):
        eligible = [order for order in choices if order[site] == 'W']
        for _ in range(8):
            means = tuple(rng.uniform(*bounds) for bounds in parent.MEAN_RANGES.values())
            width = rng.uniform(*parent.HALF_WIDTH_RANGE)
            scenes.append((rng.choice(eligible), means, width))
    rng.shuffle(scenes)
    return scenes


def documents(replica, source):
    original, training, evaluation, baseline, _ = source
    layout = locations(replica)
    campaign = layout['campaign_id']
    init = parent.seed(campaign, 'initialization', 1)
    learner = dict(original['learner_config'], seed=init)
    output, control, materials = (Path(layout[key]) for key in ('python_output', 'control_directory', 'material_root'))
    settings = dict(parent._suite_settings(learner, original['old_learner_config'], original['old_artifacts']),
                    **layout, replica=replica, schema='bg128-mixed-replication-v1',
                    source_completion=dict(path=str(ANCHOR), sha256=ANCHOR_SHA), reused_baseline=baseline)
    settings.update(arms=[dict(original['arms'][0], learner_config=learner, actor_seed=init,
                               python_output=str(output / 'arms' / parent.CANDIDATE))],
                    evaluation_shifts=EVAL, new_shift_cap=CAP, pair_id=original['pair_id'])
    for key in ('occupied_before', 'maximum_total_if_approved'):
        settings.pop(key)
    phases, ordinal = [], 0
    for template in [*training, *evaluation]:
        train = template['policy_arm'] == 'train'
        update, phase, name = (template[key] for key in ('arm_update', 'phase_id', 'arm_phase_id'))
        scenario = template['scenario_id']
        record = f'{campaign}-{phase}'
        out = output / 'arms' / parent.CANDIDATE / name if train else output / 'evaluation' / scenario / 'new'
        events = control / 'legacy_unused' / phase
        material = materials / phase
        scenes = _scenes(campaign, update) if train else [None] * len(template['jobs'])
        if train:
            parent.require(Counter(order.index('W') for order, _, _ in scenes) == {site: 8 for site in range(4)},
                           'waste classes must be balanced in each training batch')
        jobs = []
        for old, scene in zip(template['jobs'], scenes, strict=True):
            run = layout['run_index_base'] + ordinal
            slot = old['pair_ordinal']
            if train:
                env = parent.seed(campaign + '-train', 'environment', slot)
                sampling = parent.seed(campaign + '-train', 'sampling', slot)
                rpc, scene_sha = parent._runtime_config(old['expected_rpc_config'], scene)
                scene_id = 'mixed_' + scene[0].lower()
            else:
                env, sampling = old['experiment_seed'], old['actor_sampling_seed']
                rpc, scene_sha, scene_id = (old[key] for key in
                                            ('expected_rpc_config', 'runtime_scene_sha256', 'scene_id'))
            _check_speed(rpc)
            group = f'{campaign}-{scene_id}-{slot:04d}'
            leaf = f'dispatch-production-wait-shortfall-v2-run-{run}-seed-{env}'
            jobs.append(dict(old, ordinal=ordinal, run_index=run, experiment_seed=env,
                             actor_sampling_seed=sampling, actor_seed=init, recording_id=record,
                             source_recording_id=record, scene_id=scene_id, scene_group=group,
                             realization_group_id=group, expected_rpc_config=rpc, runtime_scene_sha256=scene_sha,
                             python_output=str(out), event_directory=str(events), event_bundle=str(events / leaf),
                             material_event_directory=str(material), material_event_bundle=str(material / leaf),
                             material_journal=str(material / leaf / 'material_events.bin')))
            ordinal += 1
        phases.append(dict(template, campaign_id=campaign, recording_id=record, python_output=str(out),
                           event_directory=str(events), execution=dict(template['execution'], port=layout['rpc_port']),
                           jobs=jobs))
    parent.require(ordinal == CAP and len(phases) == 39, 'replication physics inventory differs')
    scene_hashes = {job['runtime_scene_sha256'] for cfg in phases[:32] for job in cfg['jobs']}
    held_out = {job['runtime_scene_sha256'] for cfg in phases[32:] for job in cfg['jobs']}
    parent.require(len(scene_hashes) == TRAIN and len(held_out) == 7 and scene_hashes.isdisjoint(held_out),
                   'training and evaluation configurations overlap')
    return settings, phases


def _pins(original):
    extras = [*PLAN.glob('*.py'), MIXED / 'recovery_003/transport.py',
              MIXED / 'recovery_002/storage_scope.py']
    return dict(original['source_pins'], **{str(path): parent.sha(path) for path in extras})


def _render_helper(settings, jobs, master, directory):
    with TemporaryDirectory(prefix='bg128-replica-java-') as scratch:
        folder = Path(scratch)
        temp_table = folder / 'scenarios.csv'
        helper = write_java(settings, jobs, master, temp_table, folder, bridge_path=ENTRY)
        table = directory / temp_table.name
        source = _replace_once(helper.read_text(encoding='utf-8'), str(temp_table), str(table))
        anchor = '    private static final class Row {'
        source = _replace_once(source, anchor,
            '    public static double freeSpeedFactorStd(int index) { runIndex(index); return 0.05; }\n'
            f"    public static int rpcPort() {{ return {settings['rpc_port']}; }}\n\n" + anchor)
        with table.open('xb') as stream:
            stream.write(temp_table.read_bytes())
        with (directory / helper.name).open('x', encoding='utf-8') as stream:
            stream.write(source)


def _java_check(directory, port):
    helper, table = directory / 'Baseline1CampaignPlan.java', directory / 'scenarios.csv'
    evidence = verify_java(helper, table)
    with TemporaryDirectory(prefix='bg128-replica-freeform-') as scratch:
        folder = Path(scratch)
        probe = folder / 'FreeformProbe.java'
        probe.write_text(
            'public final class FreeformProbe { public static void main(String[] args) {'
            f'if (Baseline1CampaignPlan.rpcPort() != {port}) throw new AssertionError("port");'
            'for (int index=0; index<Baseline1CampaignPlan.jobCount(); index++) '
            'if (Baseline1CampaignPlan.freeSpeedFactorStd(index) != 0.05) throw new AssertionError("std"); }}',
            encoding='utf-8')
        subprocess.run(['javac', '-encoding', 'UTF-8', '-d', str(folder), str(helper), str(probe)], check=True)
        subprocess.run(['java', '-cp', str(folder), 'FreeformProbe'], check=True)
    parent.require(evidence['rows_checked'] == CAP, 'Java check incomplete')
    return dict(evidence, rpc_port=port, free_speed_std=.05, speed_rows_checked=CAP)


def load_package(replica, source=None):
    source = source or anchored_source()
    settings, expected = documents(replica, source)
    path = Path(settings['suite_path'])
    suite, configs = parent.load_suite(path)
    parent.require(all(suite.get(key) == value for key, value in settings.items()) and configs == expected
                   and suite['source_pins'] == _pins(source[0]), 'replica suite/implementation differs')
    for cfg in configs:
        parent.check_phase(cfg)
    for key, name in (('master_plan', 'master_plan.csv'), ('scenario_table', 'scenarios.csv'),
                      ('java_helper', 'Baseline1CampaignPlan.java'), ('identities', 'identities.csv')):
        parent.require(suite[key] == parent.reference(path.with_name(name)), 'replica artifact changed: ' + key)
    parent.require(path.with_name('scenarios.csv').read_bytes() == scenario_table_bytes(suite['jobs']),
                   'AnyLogic scenario sidecar differs from RPC config')
    return suite, configs


def prepare():
    from replica_budget import assert_available, preceding_snapshot

    source = anchored_source(check_rpc=True)
    prepared = []
    for replica in REPLICAS:
        settings, configs = documents(replica, source)
        directory = Path(settings['suite_path']).parent
        parent.require(not any(path.exists() for path in
                               (directory, Path(settings['python_output']), Path(settings['material_root']))),
                       'replication preparation must not replace any artifacts')
        for cfg in configs:
            parent.check_phase(cfg)
        prepared.append((settings, configs))
    jobs_by_suite = [dict(settings, jobs=[job for cfg in configs for job in cfg['jobs']])
                     for settings, configs in prepared]
    prior = preceding_snapshot()
    check = assert_available(jobs_by_suite, prior)
    index = []
    for (settings, configs), expanded in zip(prepared, jobs_by_suite, strict=True):
        directory = Path(settings['suite_path']).parent
        refs = []
        for cfg in configs:
            path = directory / 'configs' / (cfg['phase_id'] + '.json')
            parent.write(path, cfg)
            refs.append(parent.reference(path))
        master = directory / 'master_plan.csv'
        identities = directory / 'identities.csv'
        jobs = expanded['jobs']
        parent._write_rows(master, parent.COLUMNS, jobs)
        _render_helper(settings, jobs, master, directory)
        java = _java_check(directory, settings['rpc_port'])
        parent._write_rows(identities, parent.IDENTITY_COLUMNS, parent._identity_rows(configs, refs))
        suite = dict(settings, phase_configs=refs, source_pins=_pins(source[0]), master_plan=parent.reference(master),
                     scenario_table=parent.reference(directory / 'scenarios.csv'), identities=parent.reference(identities),
                     java_helper=parent.reference(directory / 'Baseline1CampaignPlan.java'), java_validation=java)
        parent.write(settings['suite_path'], suite)
        load_package(settings['replica'], source)
        path = directory / 'launch_identity.json'
        parent.write(path, dict(replica=settings['replica'], suite=parent.reference(suite['suite_path']),
            physical_shifts=CAP, training_shifts=TRAIN, evaluation_shifts=EVAL, rpc_port=settings['rpc_port'],
            free_speed_factor_std=.05, python_output=settings['python_output'], anylogic_event_root=settings['material_root'],
            first=next(parent._identity_rows(configs, refs)), identities=suite['identities'], java_helper=suite['java_helper'],
            phases=[dict(parent._phase_entry(cfg, ref), recording_id=cfg['recording_id'],
                         material_event_directory=cfg['jobs'][0]['material_event_directory'],
                         first_run=cfg['jobs'][0]['run_index'], last_run=cfg['jobs'][-1]['run_index'],
                         shifts=len(cfg['jobs'])) for cfg, ref in zip(configs, refs, strict=True)]))
        index.append(parent.reference(path))
        print('PREPARED', settings['replica'], settings['rpc_port'], path, flush=True)
    parent.require(parent.sha(parent.CORE) == parent.PROTECTED_SHA, 'protected PPO core changed')
    evidence = dict(status='PASS_OFFLINE_NO_LISTENER', replicas=index, rpc_start_fixtures=source[-1],
                    budgets=check, protected_core=parent.reference(parent.CORE), simulation_started=False,
                    budget_registered=False)
    from oneclick_java import prepare_combined

    evidence['combined_java'] = prepare_combined(index)
    parent.write(REPORT / 'oneclick_configuration_verification.json', evidence)
    return evidence
