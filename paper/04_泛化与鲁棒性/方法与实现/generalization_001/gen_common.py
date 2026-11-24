"""Sealed identities and scenario contract for fixed BG128 u32 generalization."""
import csv
import json
from pathlib import Path
import sys

PLAN = Path(__file__).resolve().parent
ROOT = PLAN.parents[3]
sys.path.insert(0, str(ROOT))

from baseline1_v1.java_plan import COLUMNS
from business_graph_summary_campaign_v1.gate import PROTECTED_SHA, check_phase
from business_graph_summary_campaign_v1.plan_io import load_phase, load_reference, reference, sha
from business_graph_summary_evaluation_v1.gate import require, seed
from business_graph_summary_evaluation_v1.gate import source_pins as evaluation_source_pins
from python.runtime_scenario import PREFIX, fingerprint

CAMPAIGN = 'bg128-u32-generalization-20261119-001'
CANDIDATE = 's01_BG128'
OUTPUT = ROOT / 'runs' / CAMPAIGN
CONTROL = OUTPUT / 'control'
MATERIALS = OUTPUT.with_name(CAMPAIGN + '_materials')
REPORT = ROOT / 'reports/token_attention_001/business_graph_summary_009/generalization_001'
PREPARED = PLAN / 'prepared_001'
SUITE = PREPARED / 'suite.json'
LAUNCH = OUTPUT / 'launch'
APPROVAL = LAUNCH / 'run_authority.json'
CORE = ROOT.parents[1] / 'python/episode_budget_ppo.py'
EXTENSION_PLAN = PLAN.parent / 'training_extension_001'
EXTENSION_OUT = ROOT / 'runs/bg128-u16-to-u32-20261119-001'
CHECKPOINT = EXTENSION_OUT / 'arms' / CANDIDATE / 'updates/0032/checkpoint.pt'
CHECKPOINT_SHA = '5d42df7d8e33b5f11424094268a4da894a36454e46d86889fc0218ecce6b2914'
TEMPLATE = EXTENSION_PLAN / 'prepared_001/configs/s01_BG128_eval_u0032.json'
FIRST_RUN = 9991001
COUNT = 100
SCENARIOS = {
    'nominal': {},
    'grade_swap': {'load.0.grade': .15, 'load.2.grade': .38},
    'speed_18kph': {'road.6.speedLimit_mps': 18.0 / 3.6},
    'speed_15kph': {'road.6.speedLimit_mps': 15.0 / 3.6},
    'grade_mean_shift': {'load.0.grade': .36, 'load.1.grade': .29, 'load.2.grade': .17},
    'grade_variability': {f'load.{i}.gradeHalfRange': .01 for i in range(3)},
}
TOTAL = COUNT * len(SCENARIOS)
PREVIOUS_CAP = 22611
USER_SCOPE = '因此进入单种子泛化性测试的实验，给出一键启动anylogic的具体方案，以最好的模型为推理模型'


def read(path):
    return json.loads(Path(path).read_bytes())


def checked(ref):
    path = Path(ref['path'])
    require(reference(path) == ref, 'sealed artifact changed: ' + str(path))
    return path


def source_pins():
    files = [p for p in PLAN.glob('*.py') if not p.name.startswith('test_')]
    files += [EXTENSION_PLAN / name for name in ('train_extension.py', 'extension_budget.py')]
    return {str(path): sha(path) for path in sorted(files)}


def check_sources(suite):
    require(suite['source_pins'] == source_pins()
            and suite['evaluation_source_pins'] == evaluation_source_pins()
            and sha(CORE) == PROTECTED_SHA, 'sealed scenario, evaluation or PPO sources changed')


def seeds_for(slot):
    return (seed(CAMPAIGN, 'environment', slot), seed(CAMPAIGN, 'sampling', slot))


def config_for(source, scenario):
    expected = dict(source, **{PREFIX + key: value for key, value in SCENARIOS[scenario].items()})
    scene = {key: value for key, value in expected.items() if key.startswith(PREFIX)}
    return expected, fingerprint(scene)


def phase_identity(scenario):
    name = 'eval_' + scenario
    tag = CANDIDATE + '_' + name
    return name, tag, CAMPAIGN + '-' + tag, OUTPUT / 'evaluation' / scenario, CONTROL / 'legacy_unused' / tag


def authorization_for(suite):
    return dict(schema='bg128-u32-generalization-launch-authority-v1',
                campaign_id=CAMPAIGN, suite_sha256=sha(SUITE), max_new_shifts=TOTAL,
                training_shifts=0, optimizer_updates=0, checkpoint_sha256=CHECKPOINT_SHA,
                authorized_by_anylogic_run=True, source_user_instruction=USER_SCOPE,
                paired_slots=COUNT, scenarios=list(SCENARIOS))


def review_selection(suite):
    completed = load_reference(suite['training_completion'])
    prior = load_reference(suite['training_suite'])
    artifacts = suite['parent_artifacts']
    require(checked(suite['training_completion']) == EXTENSION_OUT / 'suite_complete.json'
            and checked(suite['training_suite']) == EXTENSION_PLAN / 'prepared_001/suite.json'
            and completed['campaign'] == suite['training_suite'] and completed['status'] == 'COMPLETE'
            and completed['cumulative_updates'] == 32 and completed['cumulative_training_shifts'] == 1024
            and suite['learner_config'] == prior['learner_config']
            and suite['learner_config']['ppo']['actor_lr'] == 3e-4
            and suite['learner_config']['candidate_id'] == CANDIDATE,
            'complete original 3e-4 training lineage required')
    checkpoint = artifacts['checkpoint']
    require(set(artifacts) == {'checkpoint', 'manifest', 'seal'}
            and checked(checkpoint) == CHECKPOINT and checkpoint['sha256'] == CHECKPOINT_SHA
            and completed['checkpoints']['32'] == completed['fixed_evaluations']['32']['checkpoint'] == checkpoint,
            'selection must bind original completed u32')
    manifest = load_reference(artifacts['manifest'])
    require(manifest['update_index'] == 32 and manifest['sha256'] == CHECKPOINT_SHA
            and manifest['actor_digest'] == completed['fixed_evaluations']['32']['fingerprints']['actor']
            and load_reference(artifacts['seal']) ==
            {'sha256': CHECKPOINT_SHA, 'manifest_sha256': artifacts['manifest']['sha256']},
            'u32 seal or independent Actor fingerprint differs')
    paired = load_reference(suite['selection'])
    require(completed['paired_reports']['json'] == suite['selection'], 'sealed paired selection differs')
    summary = paired['metrics']
    final = summary['32']
    require(set(summary) == {'13', '16', '24', '32'} and paired['paired_shifts'] == 100
            and final['complete_task_shifts'] == max(row['complete_task_shifts'] for row in summary.values()) == 100
            and final['grade_rmse'] == min(row['grade_rmse'] for row in summary.values())
            and final['mean_total_wait_seconds'] == min(row['mean_total_wait_seconds'] for row in summary.values()),
            'u32 selection evidence differs')


def load_package():
    from java_scene import scenario_table_bytes

    suite = read(SUITE)
    check_sources(suite)
    require(suite['schema'] == 'bg128-u32-generalization-v1' and suite['campaign_id'] == CAMPAIGN
            and suite['suite_path'] == str(SUITE) and suite['python_output'] == str(OUTPUT)
            and suite['control_directory'] == str(CONTROL) and suite['material_root'] == str(MATERIALS)
            and suite['budget_path'] == str(OUTPUT / 'generalization_ledger.sqlite')
            and suite['checkpoint_update'] == 32 and suite['checkpoint_sha256'] == CHECKPOINT_SHA
            and suite['new_shift_cap'] == suite['evaluation_shifts'] == TOTAL
            and suite['training_shifts'] == suite['training_updates'] == suite['optimizer_steps'] == 0
            and suite['cpu_threads'] == 2 and suite['paired_slots'] == COUNT
            and suite['sampling'] == 'stochastic_private_rng', 'experiment scope differs')
    review_selection(suite)
    require(checked(suite['phase_template']) == TEMPLATE, 'expected u32 evaluation phase template')
    template = load_phase(TEMPLATE)
    require([p['scenario_id'] for p in suite['phases']] == list(SCENARIOS), 'scenario order differs')
    require(suite['scenario_overrides'] == SCENARIOS, 'frozen interventions differ')
    job_updates = {'ordinal', 'pair_ordinal', 'run_index', 'experiment_seed', 'actor_sampling_seed',
                   'scene_id', 'scene_group', 'realization_group_id', 'runtime_scene_sha256',
                   'phase_id', 'arm_phase_id', 'recording_id', 'source_recording_id', 'python_output',
                   'event_directory', 'event_bundle', 'material_event_directory', 'material_event_bundle',
                   'material_journal', 'expected_rpc_config'}
    phase_updates = {'campaign_id', 'phase_id', 'arm_phase_id', 'recording_id', 'python_output',
                     'event_directory', 'jobs'}
    phases, jobs = [], []
    for item in suite['phases']:
        scenario = item['scenario_id']
        name, tag, recording, output, events = phase_identity(scenario)
        ref = item['phase_config']
        require(checked(ref) == PREPARED / 'configs' / (tag + '.json'), 'scenario config location differs')
        cfg = load_phase(ref['path'])
        check_phase(cfg)
        require({k:v for k,v in cfg.items() if k not in phase_updates} ==
                {k:v for k,v in template.items() if k not in phase_updates},
                'original reward, graph, episode or RPC schema contract differs')
        require((cfg['campaign_id'], cfg['phase_id'], cfg['arm_phase_id'], cfg['recording_id'],
                 cfg['python_output'], cfg['event_directory'], len(cfg['jobs']))
                == (CAMPAIGN, tag, name, recording, str(output), str(events), COUNT),
                'evaluation phase identity differs')
        for slot, job in enumerate(cfg['jobs']):
            old = template['jobs'][slot]
            expected, scene_sha = config_for(old['expected_rpc_config'], scenario)
            require(item == dict(scenario_id=scenario, phase_id=tag, phase_config=ref,
                                 python_output=str(output), checkpoint_update=32, runtime_scene_sha256=scene_sha),
                    'phase metadata differs')
            ordinal = len(jobs)
            run = FIRST_RUN + ordinal
            env, sampling = seeds_for(slot)
            group = f'{CAMPAIGN}-{scenario}-{slot:03d}'
            leaf = f'dispatch-production-wait-shortfall-v2-run-{run}-seed-{env}'
            material = MATERIALS / tag
            require({k:v for k,v in job.items() if k not in job_updates} ==
                    {k:v for k,v in old.items() if k not in job_updates}, 'fixed-policy job contract differs')
            require(job == dict(old, ordinal=ordinal, pair_ordinal=slot, run_index=run,
                experiment_seed=env, actor_sampling_seed=sampling, scene_id=scenario,
                scene_group=group, realization_group_id=group, runtime_scene_sha256=scene_sha,
                phase_id=tag, arm_phase_id=name, recording_id=recording, source_recording_id=recording,
                python_output=str(output), event_directory=str(events), event_bundle=str(events / leaf),
                material_event_directory=str(material), material_event_bundle=str(material / leaf),
                material_journal=str(material / leaf / 'material_events.bin'), expected_rpc_config=expected),
                'exact scenario, seed, expected RPC or recording identity differs')
            jobs.append(job)
        phases.append(cfg)
    require('jobs' not in suite and len(jobs) == TOTAL, 'sealed phase inventory differs')
    with checked(suite['master_plan']).open(newline='') as stream:
        require(list(csv.DictReader(stream)) == [{key: str(job[key]) for key in COLUMNS} for job in jobs],
                'frozen Java master differs from phase jobs')
    require(checked(suite['scenario_table']).read_bytes() == scenario_table_bytes(jobs),
            'Java scenario values differ from expected RPC configuration')
    for key in ('java_helper', 'one_click_entry'):
        checked(suite[key])
    for ref in suite['manual_java_changes'].values():
        checked(ref)
    suite['jobs'] = jobs
    return suite, phases
