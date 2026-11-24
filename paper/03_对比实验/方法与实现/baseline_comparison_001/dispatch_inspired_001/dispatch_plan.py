"""Freeze a single deterministic DISPATCH-inspired AnyLogic validation shift."""
from hashlib import sha256
from pathlib import Path
import sys

PLAN = Path(__file__).resolve().parent
for directory in (PLAN.parent, PLAN.parent/'paired_100_001'):
    if str(directory) not in sys.path:
        sys.path.append(str(directory))

import smoke_plan as smoke
from smoke_plan import (ROOT, EXECUTION, PYTHON, CORE, CORE_SHA, COLUMNS,
                        encoded, load_reference, reference, require)
from python.offline_split import JAVA_MASK, episode_id


SCHEMA = 'bg128-dispatch-inspired-validation-v1'
PHASE_SCHEMA = 'bg128-dispatch-inspired-validation-phase-v1'
CAMPAIGN = 'bg128-dispatch-inspired-20261122-001'
POLICY = 'dispatch_inspired'
COUNT = 1
RUN = 10022001
PREPARED = PLAN/'prepared_001'
SUITE = PREPARED/'suite.json'
OUTPUT = ROOT/'runs'/CAMPAIGN
CONTROL = OUTPUT/'control'
REPORT = smoke.REPORT/'dispatch_inspired_001'
EVENTS = OUTPUT/'anylogic_events'
CONFIG = PREPARED/'phase_config.json'
METHOD = dict(planning_ms=800, response_target_ms=1000, replan_seconds=900, match_trucks=3)
LAYOUT = dict(python_output=str(OUTPUT), control_directory=str(CONTROL), report_directory=str(REPORT),
              event_root=str(EVENTS))


def source_contract():
    refs, _, rpc, scene = smoke.source_contract()
    selected = {key: refs[key] for key in ('protected_core', 'nominal_source')}
    selected.update(smoke_completion=reference(smoke.OUTPUT/'suite_complete.json'),
                    smoke_evaluation=reference(smoke.REPORT/'single_shift_validation.json'))
    completion = load_reference(selected['smoke_completion'])
    report = load_reference(selected['smoke_evaluation'])
    require(completion['status'] == report['status'] == 'COMPLETE'
            and completion['evaluation'] == selected['smoke_evaluation']
            and completion['physical_shifts'] == report['completed_shifts'] == len(smoke.STRATEGIES),
            'original low-frequency smoke contract did not complete')
    return selected, rpc, scene


def source_pins():
    local = sorted(path for path in PLAN.glob('*.py') if not path.name.startswith('test_'))
    require({'dispatch_plan.py', 'dispatch_budget.py', 'prepare.py', 'java_adapter.py', 'launch.py',
             'dispatch_runtime.py', 'dispatch_policy.py', 'flow_plan.py'} <= {path.name for path in local},
            'finish the DISPATCH runtime and policy before preparation')
    return [*smoke.source_pins(), *[reference(path) for path in local]]


def seed():
    return int.from_bytes(sha256(f'{CAMPAIGN}/environment/0'.encode('ascii')).digest()[:6], 'big')


def job_for(rpc, scene):
    environment = seed()
    return dict(ordinal=0, policy_kind=POLICY, actor_seed=0, group='BASELINE', role='final',
                source_recording_id=CAMPAIGN, recording_id=CAMPAIGN, run_index=RUN, experiment_seed=environment,
                actor_sampling_seed=0, actor_sampling_seed_consumed=False, event_directory=str(EVENTS),
                event_bundle=str(EVENTS/episode_id(RUN, environment)), python_output=str(OUTPUT/'shift'),
                target_training_update=0, checkpoint_update=0, scene_id='nominal',
                expected_rpc_config=rpc, runtime_scene_sha256=scene)


def config_for(job, refs):
    from flow_plan import FLOW_PLAN_CONTRACT

    cfg = smoke.config_for(job, refs)
    cfg.update(schema=PHASE_SCHEMA, campaign_id=CAMPAIGN, method=METHOD, flow_plan_contract=FLOW_PLAN_CONTRACT)
    return cfg


def validate_jobs(jobs):
    require(len(jobs) == COUNT, 'one physical DISPATCH validation shift is required')
    job = jobs[0]
    require(job['ordinal'] == 0 and job['run_index'] == RUN and job['recording_id'] == CAMPAIGN
            and job['source_recording_id'] == CAMPAIGN and job['policy_kind'] == POLICY
            and job['experiment_seed'] == seed() and 0 < job['experiment_seed'] <= JAVA_MASK,
            'DISPATCH validation identity differs')
    require(job['actor_seed'] == job['actor_sampling_seed'] == 0
            and job['actor_sampling_seed_consumed'] is False, 'deterministic strategy must have no policy RNG')


def load_suite(path=SUITE, *, check_sources=True):
    path = Path(path)
    require(path.resolve() == SUITE, 'frozen DISPATCH validation suite path required')
    suite = load_reference(reference(path))
    require(path.read_bytes() == encoded(suite), 'noncanonical DISPATCH validation suite bytes')
    require(suite['schema'] == SCHEMA and suite['campaign_id'] == CAMPAIGN and suite['execution'] == EXECUTION
            and suite['method'] == METHOD and all(suite.get(key) == val for key, val in LAYOUT.items())
            and suite['physical_shifts'] == COUNT and suite['training_shifts'] == suite['optimizer_updates'] == 0,
            'DISPATCH validation suite scope or execution contract differs')
    validate_jobs(suite['jobs'])
    job = suite['jobs'][0]
    refs, rpc, scene = source_contract()
    require(all(suite.get(key) == val for key, val in refs.items()), 'frozen nominal source contract differs')
    require({key: value for key, value in job.items() if key != 'phase_config'} == job_for(rpc, scene),
            'frozen DISPATCH physical job differs')
    require(reference(CONFIG) == job['phase_config'] and CONFIG.read_bytes() == encoded(config_for(job, refs)),
            'frozen DISPATCH phase bytes differ')
    for key, name in [('master_plan', 'master_plan.csv'), ('scenario_table', 'scenario_table.csv'),
                      ('java_helper', 'Baseline1CampaignPlan.java')]:
        require(reference(PREPARED/name) == suite[key], 'frozen launch artifact differs: ' + key)
    if check_sources:
        require(suite['source_pins'] == source_pins(), 'DISPATCH validation source bytes differ')
    return suite
