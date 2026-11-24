"""Prepare the two standard-scene speed perturbations and seal their run identities."""

import csv
import json
from pathlib import Path
import sys

PLAN = Path(__file__).resolve().parent
ROOT = PLAN.parents[4]
MIXED = PLAN.parent / 'grade_assignment_generalization_001/continuous_training_001'
for directory in (ROOT, MIXED, MIXED / 'recovery_003', PLAN):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from baseline1_v1.java_plan import COLUMNS
from business_graph_summary_campaign_v1.contract import require, seed
from business_graph_summary_campaign_v1.gate import PROTECTED_SHA, check_phase
from business_graph_summary_campaign_v1.plan_io import load_reference, load_suite, reference, sha, write
from mixed_plan import _write_rows, source_pins as mixed_source_pins
from python.runtime_scenario import PREFIX, fingerprint

CAMPAIGN = 'bg128-speed-congestion-20261120-001'
CAMPAIGN_PLAN = PLAN / 'campaign_plan.json'
PREPARED = PLAN / 'prepared_001'
SUITE = PREPARED / 'suite.json'
OUTPUT = ROOT / 'runs' / CAMPAIGN
MATERIALS = OUTPUT.with_name(CAMPAIGN + '_materials')
CONTROL = OUTPUT / 'control'
LAUNCH = OUTPUT / 'launch'
AUTHORITY = LAUNCH / 'run_authority.json'
LEDGER = OUTPUT / 'budget.sqlite'
REPORT = ROOT / 'reports/token_attention_001/business_graph_summary_009/generalization_001/speed_congestion_001'
CORE = ROOT.parents[1] / 'python/episode_budget_ppo.py'
MIXED_SUITE = MIXED / 'prepared_001/suite.json'
NOMINAL = MIXED / 'prepared_001/configs/s01_BG128_eval_nominal_new.json'
PARENT_ROOT = ROOT / 'runs/bg128-continuous-mixed-20261119-001'
CHECKPOINT = PARENT_ROOT / 'arms/s01_BG128/updates/0032/checkpoint.pt'
CHECKPOINT_SHA = 'b8b81c0821535cdfdff3c9352c11d69bc91d14fbe97694ab5f45c08fb97c3c9a'
MIXED_COMPLETE = PARENT_ROOT / 'suite_complete.json'
HISTORICAL_NOMINAL = PARENT_ROOT / 'evaluation/nominal/new/evaluation_complete.json'
STD_KEY = PREFIX + 'env.freeSpeedFactorStd'
PREVIOUS_CAP = 25636
FIRST_RUN = 10002426
SLOTS = 20
TOTAL = 2 * SLOTS
SCHEMA = 'bg128-speed-congestion-evaluation-v1'
IDENTITY_COLUMNS = (
    'ordinal', 'pair_ordinal', 'scene_id', 'free_speed_factor_std', 'recording_id',
    'config_sha256', 'runtime_scene_sha256', 'python_output', 'material_event_directory',
    'run_index', 'experiment_seed', 'actor_sampling_seed', 'actor_seed',
)


def read(path):
    return json.loads(Path(path).read_bytes())


def _recipe():
    plan = read(CAMPAIGN_PLAN)
    require(plan == dict(campaign_id=CAMPAIGN, base_scenario='nominal', first_run_index=FIRST_RUN,
                         scenarios=[dict(id='speed_std_015', freeSpeedFactorStd=.15, shifts=SLOTS),
                                    dict(id='speed_std_025', freeSpeedFactorStd=.25, shifts=SLOTS)]),
            'this campaign is scoped to the announced nominal 20 + 20 slots')
    return plan


def _settings():
    plan = _recipe()
    parent = read(MIXED_SUITE)
    checkpoint = reference(CHECKPOINT)
    manifest = reference(str(CHECKPOINT) + '.manifest.json')
    seal = reference(str(CHECKPOINT) + '.seal.json')
    complete = read(MIXED_COMPLETE)
    nominal = reference(NOMINAL)
    require(nominal in parent['phase_configs'], 'nominal template is outside the completed mixed campaign')
    require(checkpoint['sha256'] == CHECKPOINT_SHA and read(manifest['path'])['update_index'] == 32
            and read(manifest['path'])['sha256'] == CHECKPOINT_SHA
            and read(seal['path']) == {'sha256': CHECKPOINT_SHA, 'manifest_sha256': manifest['sha256']}
            and complete['status'] == 'COMPLETE' and complete['new_checkpoint'] == checkpoint
            and complete['campaign'] == reference(MIXED_SUITE), 'mixed u32 checkpoint lineage or seal differs')
    require(sha(CORE) == PROTECTED_SHA, 'protected PPO core byte SHA changed')
    return dict(
        schema=SCHEMA, campaign_id=CAMPAIGN, suite_path=str(SUITE), candidate_id='s01_BG128',
        parent_campaign=reference(MIXED_SUITE), parent_complete=reference(MIXED_COMPLETE),
        parent_artifacts=dict(checkpoint=checkpoint, manifest=manifest, seal=seal),
        nominal_template=nominal, historical_nominal=reference(HISTORICAL_NOMINAL),
        campaign_plan=reference(CAMPAIGN_PLAN), scenarios=plan['scenarios'],
        learner_config=parent['arms'][0]['learner_config'], actor_seed=parent['arms'][0]['actor_seed'],
        protected_core=reference(CORE), checkpoint_update=32, cpu_threads=2,
        sampling='stochastic_private_rng', pair_id=CAMPAIGN + '-paired', paired_slots=SLOTS,
        training_updates=0, training_shifts=0, optimizer_steps=0, evaluation_shifts=TOTAL,
        new_shift_cap=TOTAL, legacy_reserved_shifts=0, active_h2_reserved_shifts=0,
        occupied_before=PREVIOUS_CAP, maximum_total_if_approved=PREVIOUS_CAP + TOTAL,
        python_output=str(OUTPUT), control_directory=str(CONTROL), material_root=str(MATERIALS),
        report_directory=str(REPORT), budget_path=str(LEDGER),
    )


def _documents(settings):
    nominal = load_reference(settings['nominal_template'])
    baseline = nominal['jobs'][0]['expected_rpc_config']
    expected_environment = {
        STD_KEY: .05, PREFIX + 'env.freeSpeedFactorMin': .85, PREFIX + 'env.freeSpeedFactorMax': 1.10,
        PREFIX + 'env.useEndogenousCongestion': 1., PREFIX + 'env.useLocalDynamicSpeed': 1.,
        PREFIX + 'env.deterministicCheck': 0.,
    }
    require(all(baseline[key] == value for key, value in expected_environment.items())
            and all(job['expected_rpc_config'] == baseline for job in nominal['jobs']),
            'nominal physical environment or speed settings differ')
    phases = []
    for scene in settings['scenarios']:
        scenario, count = scene['id'], scene['shifts']
        phase = 's01_BG128_eval_' + scenario
        recording = CAMPAIGN + '-' + phase
        output = OUTPUT / 'evaluation' / scenario
        events = CONTROL / 'legacy_unused' / phase
        materials = MATERIALS / phase
        rpc = dict(baseline)
        rpc[STD_KEY] = scene['freeSpeedFactorStd']
        scene_sha = fingerprint({key: value for key, value in rpc.items() if key.startswith(PREFIX)})
        jobs = []
        for slot, source in enumerate(nominal['jobs'][:count]):
            ordinal = len(phases) * SLOTS + slot
            index = FIRST_RUN + ordinal
            environment = seed(CAMPAIGN, 'environment', slot)
            sampling = seed(CAMPAIGN, 'sampling', slot)
            leaf = f'dispatch-production-wait-shortfall-v2-run-{index}-seed-{environment}'
            group = f'{CAMPAIGN}-{scenario}-{slot:04d}'
            jobs.append(dict(
                source, ordinal=ordinal, pair_ordinal=slot, run_index=index, experiment_seed=environment,
                actor_sampling_seed=sampling, actor_seed=settings['actor_seed'],
                arm_phase_id='eval_' + scenario, phase_id=phase, recording_id=recording,
                source_recording_id=recording, scene_id=scenario, scene_group=group, realization_group_id=group,
                runtime_scene_sha256=scene_sha, expected_rpc_config=rpc, python_output=str(output),
                event_directory=str(events), event_bundle=str(events / leaf),
                material_event_directory=str(materials), material_event_bundle=str(materials / leaf),
                material_journal=str(materials / leaf / 'material_events.bin'),
            ))
        phases.append(check_phase(dict(
            nominal, campaign_id=CAMPAIGN, phase_id=phase, arm_phase_id='eval_' + scenario,
            recording_id=recording, python_output=str(output), event_directory=str(events),
            policy_arm='new', scenario_id=scenario, jobs=jobs,
        )))
    return phases


def _identities(phases, refs):
    return [dict(job, config_sha256=ref['sha256'], free_speed_factor_std=job['expected_rpc_config'][STD_KEY])
            for phase, ref in zip(phases, refs, strict=True) for job in phase['jobs']]


def _pins():
    base = mixed_source_pins()
    sources = [PLAN / name for name in ('speed_plan.py', 'speed_campaign.py', 'java_speed.py',
                                       'speed_budget.py', 'speed_report.py')]
    sources.extend([MIXED / 'recovery_003/transport.py', MIXED / 'recovery_002/storage_scope.py'])
    base.update({str(path): sha(path) for path in sources})
    return base


def authorization_for(suite):
    require(suite['campaign_id'] == CAMPAIGN and suite['suite_path'] == str(SUITE),
            'launch authority requires the canonical speed campaign')
    return dict(
        schema=SCHEMA + '-authority', campaign_id=CAMPAIGN, suite=reference(SUITE),
        authorized_by_anylogic_run=True, max_new_shifts=TOTAL, prior_registered_cap=PREVIOUS_CAP,
        maximum_total_after_registration=PREVIOUS_CAP + TOTAL, training_updates=0, optimizer_steps=0,
        checkpoint=suite['parent_artifacts']['checkpoint'],
    )


def prepare():
    """Freeze the handoff without creating an experiment output or listener."""
    from java_speed import verify_java, write_java
    from speed_budget import check_collisions

    require(not any(path.exists() for path in (PREPARED, OUTPUT, MATERIALS, REPORT)),
            'preparation or execution already exists')
    suite = _settings()
    pins = _pins()
    phases = _documents(suite)
    jobs = [job for phase in phases for job in phase['jobs']]
    check_collisions(dict(suite, jobs=jobs))
    refs = []
    for phase in phases:
        path = PREPARED / 'configs' / (phase['phase_id'] + '.json')
        write(path, phase)
        refs.append(reference(path))
    master = PREPARED / 'master_plan.csv'
    _write_rows(master, COLUMNS, jobs)
    bridge = PLAN / 'speed_campaign.py'
    helper = write_java(suite, jobs, master, PREPARED, bridge_path=bridge)
    identities = PREPARED / 'identities.csv'
    rows = _identities(phases, refs)
    _write_rows(identities, IDENTITY_COLUMNS, rows)
    checks = verify_java(PREPARED)
    suite.update(phase_configs=refs, source_pins=pins, master_plan=reference(master),
                 scenario_table=reference(PREPARED / 'scenarios.csv'),
                 speed_scenario_table=reference(PREPARED / 'speed_scenarios.csv'),
                 java_helper=reference(helper), one_click_entry=reference(bridge),
                 identities=reference(identities), java_validation=checks)
    write(SUITE, suite)
    identity = dict(
        campaign_id=CAMPAIGN, suite=reference(SUITE), python_output=str(OUTPUT),
        anylogic_event_root=str(MATERIALS), python_report_directory=str(REPORT),
        checkpoint=suite['parent_artifacts']['checkpoint'], operator_action='AnyLogic RL_Training Run',
        evaluation_shifts=TOTAL, optimizer_updates=0, first={key: rows[0][key] for key in IDENTITY_COLUMNS},
        phases=[dict(scenario_id=phase['scenario_id'], recording_id=phase['recording_id'], config=ref,
                     free_speed_factor_std=phase['jobs'][0]['expected_rpc_config'][STD_KEY],
                     python_output=phase['python_output'], first_run=phase['jobs'][0]['run_index'],
                     last_run=phase['jobs'][-1]['run_index'], shifts=len(phase['jobs']),
                     material_event_directory=phase['jobs'][0]['material_event_directory'])
                for phase, ref in zip(phases, refs, strict=True)],
        all_run_indices_and_seeds=suite['identities'],
    )
    write(PREPARED / 'launch_identity.json', identity)
    load_package()
    require(not any(path.exists() for path in (OUTPUT, MATERIALS, LEDGER)), 'offline preparation created run output')
    return identity


def load_package():
    suite, phases = load_suite(SUITE)
    settings = _settings()
    require(all(suite.get(key) == value for key, value in settings.items()), 'frozen campaign scope differs')
    require(suite['source_pins'] == _pins(), 'reviewed implementation bytes differ')
    require(phases == _documents(settings), 'frozen RPC phase or paired identity differs')
    artifacts = {
        'master_plan': PREPARED / 'master_plan.csv', 'scenario_table': PREPARED / 'scenarios.csv',
        'speed_scenario_table': PREPARED / 'speed_scenarios.csv', 'identities': PREPARED / 'identities.csv',
        'java_helper': PREPARED / 'Baseline1CampaignPlan.java', 'one_click_entry': PLAN / 'speed_campaign.py',
    }
    require(all(suite[key] == reference(path) for key, path in artifacts.items()), 'frozen handoff changed')
    require(suite['phase_configs'] == [reference(PREPARED / 'configs' / (phase['phase_id'] + '.json'))
                                      for phase in phases], 'noncanonical phase config references')
    with artifacts['master_plan'].open(newline='', encoding='utf-8') as stream:
        require(list(csv.DictReader(stream)) == [{key: str(job[key]) for key in COLUMNS} for job in suite['jobs']],
                'CampaignPlan and Python job inventory differ')
    return suite, phases
