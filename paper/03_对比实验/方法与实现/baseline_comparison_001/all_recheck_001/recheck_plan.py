"""Freeze a fresh nominal slot for independent rechecking of the five V2 policies."""
from hashlib import sha256
from pathlib import Path
import sys


PLAN = Path(__file__).resolve().parent
for directory in (PLAN.parent, PLAN.parent/'paired_100_001', PLAN.parent/'dispatch_inspired_001', PLAN.parent/'audit_002'):
    if str(directory) not in sys.path:
        sys.path.append(str(directory))

import smoke_plan as smoke
import paired_plan as previous
from smoke_plan import (ROOT, EXECUTION, PYTHON, CORE, COLUMNS, encoded, load_reference, reference, require)
from python.offline_split import JAVA_MASK, episode_id


SCHEMA = 'bg128-traditional-audit-v2'
PHASE_SCHEMA = 'bg128-traditional-audit-phase-v2'
CAMPAIGN = 'bg128-baseline-all-recheck-20261122-001'
STRATEGIES = ('unload_greedy', 'fifo_greedy', 'rolling', 'dispatch_inspired', 'bg128')
COUNT = len(STRATEGIES)
FIRST_RUN = 10025001
PREPARED = PLAN/'prepared_001'
SUITE = PREPARED/'suite.json'
OUTPUT = ROOT/'runs'/CAMPAIGN
CONTROL = OUTPUT/'control'
EVENTS = OUTPUT/'anylogic_events'
REPORT = smoke.REPORT/'all_recheck_001'
METHOD = dict(planning_ms=800, depth=3, rollout_limit=20, response_target_ms=1000, replan_seconds=900, match_trucks=3)
LAYOUT = dict(python_output=str(OUTPUT), control_directory=str(CONTROL), report_directory=str(REPORT),
              event_root=str(EVENTS))
PAIRING = dict(scope='this_campaign_only', slots=1, share='experiment_seed', policies=list(STRATEGIES))
REPORT_FILENAME = 'single_shift_validation.json'


def source_contract():
    refs, learner, rpc, scene = previous.source_contract()
    refs['checkpoint_selection_evidence'] = previous.selection_evidence(refs)
    return refs, learner, rpc, scene


def source_pins():
    revised = sorted(path for path in (ROOT/'traditional_dispatch_v2').glob('*.py')
                     if not path.name.startswith('test_'))
    require({'nominal.py', 'decision.py', 'runtime.py', 'dispatch_policy.py', 'flow_plan.py'} <= {path.name for path in revised},
            'complete the audited revised strategy runtime before freezing')
    names = ('recheck_plan.py', 'recheck_budget.py', 'java_adapter.py', 'prepare.py', 'launch.py')
    upstream = (previous.PLAN/'paired_plan.py', previous.PLAN/'paired_budget.py',
                PLAN.parent/'dispatch_inspired_001/dispatch_plan.py', PLAN.parent/'dispatch_inspired_001/dispatch_budget.py',
                PLAN.parent/'audit_002/audit_plan.py', PLAN.parent/'audit_002/audit_budget.py')
    return [*smoke.source_pins(), *[reference(path) for path in (*revised, *upstream)],
            *[reference(PLAN/name) for name in names]]


def seed(stream):
    return int.from_bytes(sha256(f'{CAMPAIGN}/{stream}/0'.encode('ascii')).digest()[:6], 'big')


def job_for(ordinal, learner, rpc, scene):
    policy = STRATEGIES[ordinal]
    actor = policy == 'bg128'
    run, environment = FIRST_RUN+ordinal, seed('environment')
    event_dir = EVENTS/f'run_{run}'
    recording = f'{CAMPAIGN}-{policy}'
    return dict(ordinal=ordinal, pair_ordinal=0, policy_kind=policy, actor_seed=learner['seed'] if actor else 0,
                group='BASELINE', role='final', source_recording_id=recording, recording_id=recording,
                run_index=run, experiment_seed=environment, actor_sampling_seed=seed('actor-sampling') if actor else 0,
                actor_sampling_seed_consumed=actor, event_directory=str(event_dir),
                event_bundle=str(event_dir/episode_id(run, environment)), python_output=str(OUTPUT/f'run_{run}'),
                target_training_update=0, checkpoint_update=32 if actor else 0, scene_id='nominal',
                expected_rpc_config=rpc, runtime_scene_sha256=scene)


def phase_path(job):
    return PREPARED/'configs'/f"run_{job['run_index']}.json"


def config_for(job, refs):
    result = smoke.config_for(job, refs)
    result.update(schema=PHASE_SCHEMA, campaign_id=CAMPAIGN, method=METHOD)
    if job['policy_kind'] == 'dispatch_inspired':
        from traditional_dispatch_v2.flow_plan import FLOW_PLAN_CONTRACT

        result['flow_plan_contract'] = FLOW_PLAN_CONTRACT
    return result


def validate_jobs(jobs):
    require(len(jobs) == COUNT and tuple(j['policy_kind'] for j in jobs) == STRATEGIES,
            'five frozen revised strategies in order required')
    for key in ('run_index', 'recording_id', 'source_recording_id', 'event_directory', 'event_bundle', 'python_output'):
        require(len({job[key] for job in jobs}) == COUNT, 'duplicate physical ' + key)
    environment, sampling = seed('environment'), seed('actor-sampling')
    require(0 < environment <= JAVA_MASK and 0 < sampling <= JAVA_MASK and environment != sampling,
            'the validation seeds must be positive, distinct Java-width values')
    for i, job in enumerate(jobs):
        actor = STRATEGIES[i] == 'bg128'
        require(job['ordinal'] == i and job['pair_ordinal'] == 0 and job['run_index'] == FIRST_RUN+i
                and job['experiment_seed'] == environment and job['actor_sampling_seed'] == (sampling if actor else 0)
                and job['actor_sampling_seed_consumed'] is actor and (job['actor_seed'] > 0 if actor else job['actor_seed'] == 0),
                'physical identity order or scoped seed assignment differs')


def load_suite(path=SUITE, *, check_sources=True):
    path = Path(path)
    require(path.resolve() == SUITE, 'frozen revised validation suite path required')
    suite = load_reference(reference(path))
    require(path.read_bytes() == encoded(suite), 'noncanonical revised validation suite bytes')
    require(suite['schema'] == SCHEMA and suite['campaign_id'] == CAMPAIGN and suite['execution'] == EXECUTION
            and suite['method'] == METHOD and all(suite.get(key) == val for key, val in LAYOUT.items())
            and suite['physical_shifts'] == COUNT and suite['evaluation_slots'] == 1
            and suite['training_shifts'] == suite['optimizer_updates'] == 0 and suite['pairing'] == PAIRING,
            'revised validation suite scope or execution contract differs')
    refs, learner, rpc, scene = source_contract()
    require(suite['learner_config'] == learner and all(suite.get(key) == val for key, val in refs.items()),
            'frozen nominal configuration or actor source differs')
    validate_jobs(suite['jobs'])
    for i, job in enumerate(suite['jobs']):
        require({key: val for key, val in job.items() if key != 'phase_config'} == job_for(i, learner, rpc, scene),
                'frozen revised physical job differs')
        phase = phase_path(job)
        require(reference(phase) == job['phase_config'] and phase.read_bytes() == encoded(config_for(job, refs)),
                'frozen revised phase bytes differ')
    for key, name in [('master_plan', 'master_plan.csv'), ('scenario_table', 'scenario_table.csv'),
                      ('java_helper', 'Baseline1CampaignPlan.java'), ('run_seed', 'run_seed.csv')]:
        require(reference(PREPARED/name) == suite[key], 'frozen launch artifact differs: ' + key)
    if check_sources:
        require(suite['source_pins'] == source_pins(), 'revised validation source bytes differ')
    return suite
