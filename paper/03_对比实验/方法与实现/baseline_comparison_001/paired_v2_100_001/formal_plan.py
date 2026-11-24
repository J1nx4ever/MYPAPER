"""Freeze 100 fresh nominal slots for the five audited V2 policies."""
from hashlib import sha256
from pathlib import Path
import sys


PLAN = Path(__file__).resolve().parent
for directory in (PLAN.parent, PLAN.parent/'paired_100_001', PLAN.parent/'all_recheck_001',
                  PLAN.parent.parent/'gat_weak_ablation_001'):
    if str(directory) not in sys.path:
        sys.path.append(str(directory))

import recheck_plan as reviewed
from recheck_plan import (ROOT, EXECUTION, PYTHON, CORE, COLUMNS, STRATEGIES, METHOD, PHASE_SCHEMA, smoke,
                          encoded, load_reference, reference, require)
from python.offline_split import JAVA_MASK, episode_id


SCHEMA = 'bg128-traditional-paired100-v2'
CAMPAIGN = 'bg128-baseline-v2-paired100-20261122-001'
SLOTS = 100
COUNT = SLOTS*len(STRATEGIES)
FIRST_RUN = 10032001
PREPARED = PLAN/'prepared_001'
SUITE = PREPARED/'suite.json'
OUTPUT = ROOT/'runs'/CAMPAIGN
CONTROL = OUTPUT/'control'
EVENTS = OUTPUT/'anylogic_events'
REPORT = smoke.REPORT/'paired_v2_100_001'
REPORT_FILENAME = 'paired_comparison.json'
LAYOUT = dict(python_output=str(OUTPUT), control_directory=str(CONTROL), report_directory=str(REPORT),
              event_root=str(EVENTS))
PAIRING = dict(scope='this_campaign_only', slots=SLOTS, share='experiment_seed',
               policies=list(STRATEGIES), order='slot_then_policy')


def source_contract():
    refs, learner, rpc, scene = reviewed.source_contract()
    refs.update(v2_recheck_suite=reference(reviewed.SUITE),
                v2_recheck_completion=reference(reviewed.OUTPUT/'suite_complete.json'),
                v2_recheck_report=reference(reviewed.REPORT/reviewed.REPORT_FILENAME))
    complete = load_reference(refs['v2_recheck_completion'])
    report = load_reference(refs['v2_recheck_report'])
    require(complete['status'] == report['status'] == 'COMPLETE'
            and complete['suite'] == refs['v2_recheck_suite'] and complete['evaluation'] == refs['v2_recheck_report']
            and complete['physical_shifts'] == report['completed_shifts'] == len(STRATEGIES)
            and complete['training_shifts'] == complete['optimizer_updates'] == 0
            and tuple(row['policy_kind'] for row in report['shifts']) == STRATEGIES
            and all(row['engineering_acceptance']['passed'] for row in report['shifts']),
            'completed five-policy V2 recheck proof differs')
    return refs, learner, rpc, scene


def source_pins():
    verified = reviewed.load_suite()
    local = ('formal_plan.py', 'formal_ledger.py', 'java_adapter.py', 'prepare.py', 'launch.py',
             'paired_runtime.py', 'formal_report.py')
    weak = PLAN.parent.parent/'gat_weak_ablation_001'
    extra = (PLAN.parent/'paired_100_001/paired_report.py', weak/'weak_budget.py', weak/'weak_plan.py',
             *[PLAN/name for name in local])
    return [*verified['source_pins'], *[reference(path) for path in extra]]


def seed(stream, slot):
    return int.from_bytes(sha256(f'{CAMPAIGN}/{stream}/{slot}'.encode('ascii')).digest()[:6], 'big')


def job_for(ordinal, learner, rpc, scene):
    slot, arm = divmod(ordinal,len(STRATEGIES))
    policy = STRATEGIES[arm]
    actor = policy == 'bg128'
    run, environment = FIRST_RUN+ordinal, seed('environment', slot)
    leaf = f'run_{run}'
    event_dir = EVENTS/leaf
    recording = f'{CAMPAIGN}-{slot:03d}-{policy}'
    return dict(ordinal=ordinal, pair_ordinal=slot, policy_kind=policy, actor_seed=learner['seed'] if actor else 0,
                group='BASELINE', role='final', source_recording_id=recording, recording_id=recording,
                run_index=run, experiment_seed=environment, actor_sampling_seed=seed('actor-sampling', slot) if actor else 0,
                actor_sampling_seed_consumed=actor, event_directory=str(event_dir),
                event_bundle=str(event_dir/episode_id(run, environment)), python_output=str(OUTPUT/'shifts'/leaf),
                target_training_update=0, checkpoint_update=32 if actor else 0, scene_id='nominal',
                expected_rpc_config=rpc, runtime_scene_sha256=scene)


def phase_path(job):
    return PREPARED/'configs'/f"run_{job['run_index']}.json"


def config_for(job, refs):
    cfg = reviewed.config_for(job, refs)
    cfg.update(campaign_id=CAMPAIGN)
    return cfg


def validate_jobs(jobs):
    require(len(jobs) == COUNT, '500 physical V2 jobs required')
    for key in ('run_index', 'recording_id', 'source_recording_id', 'event_directory', 'event_bundle', 'python_output'):
        require(len({row[key] for row in jobs}) == COUNT, 'duplicate physical ' + key)
    used = set()
    for slot in range(SLOTS):
        start = slot*len(STRATEGIES)
        environment, sampling = seed('environment',slot), seed('actor-sampling',slot)
        require(0 < environment <= JAVA_MASK and 0 < sampling <= JAVA_MASK and environment != sampling
                and used.isdisjoint((environment,sampling)), 'seed identity or cross-slot isolation differs')
        used.update((environment,sampling))
        for i, policy in enumerate(STRATEGIES):
            job = jobs[start+i]
            actor = policy == 'bg128'
            require(job['ordinal'] == start+i and job['pair_ordinal'] == slot and job['policy_kind'] == policy
                    and job['run_index'] == FIRST_RUN+start+i and job['experiment_seed'] == environment
                    and job['actor_sampling_seed'] == (sampling if actor else 0)
                    and job['actor_sampling_seed_consumed'] is actor
                    and (job['actor_seed'] > 0 if actor else job['actor_seed'] == 0),
                    'physical slot policy order or scoped seed assignment differs')


def load_suite(path=SUITE, *, check_sources=True):
    path = Path(path)
    require(path.resolve() == SUITE, 'the single frozen V2 suite path is required')
    suite = load_reference(reference(path))
    require(path.read_bytes() == encoded(suite), 'noncanonical V2 evaluation suite bytes')
    require(suite['schema'] == SCHEMA and suite['campaign_id'] == CAMPAIGN and suite['execution'] == EXECUTION
            and suite['method'] == METHOD and all(suite.get(key) == val for key, val in LAYOUT.items())
            and suite['physical_shifts'] == COUNT and suite['evaluation_slots'] == SLOTS
            and suite['training_shifts'] == suite['optimizer_updates'] == 0 and suite['pairing'] == PAIRING
            and suite['checkpoint_selection'] == 'mixed-s02-u32', 'V2 evaluation suite scope or execution differs')
    refs, learner, rpc, scene = source_contract()
    require(suite['learner_config'] == learner and all(suite.get(key) == val for key, val in refs.items()),
            'V2 recheck, nominal scene or pinned Actor source differs')
    validate_jobs(suite['jobs'])
    for i, job in enumerate(suite['jobs']):
        require({key: val for key, val in job.items() if key != 'phase_config'} == job_for(i, learner, rpc, scene),
                'frozen V2 physical job differs')
        phase = phase_path(job)
        require(reference(phase) == job['phase_config'] and phase.read_bytes() == encoded(config_for(job, refs)),
                'frozen V2 phase bytes differ')
    for key, name in [('master_plan', 'master_plan.csv'), ('scenario_table', 'scenario_table.csv'),
                      ('java_helper', 'Baseline1CampaignPlan.java'), ('run_seed', 'run_seed.csv')]:
        require(reference(PREPARED/name) == suite[key], 'frozen launch artifact differs: ' + key)
    if check_sources:
        require(suite['source_pins'] == source_pins(), 'V2 evaluation source bytes differ')
    return suite
