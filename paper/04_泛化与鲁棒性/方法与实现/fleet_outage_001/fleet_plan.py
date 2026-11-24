"""Freeze five BG128 evaluations with six active trucks and nine observation slots."""
from hashlib import sha256
from pathlib import Path
import sys


PLAN = Path(__file__).resolve().parent
FORMAL = PLAN.parent/'baseline_comparison_001/paired_v2_100_001'
if str(FORMAL) not in sys.path:
    sys.path.append(str(FORMAL))

import formal_plan as nominal
from formal_plan import (ROOT, CORE, EXECUTION, PYTHON, COLUMNS, METHOD, PHASE_SCHEMA, smoke,
                         encoded, load_reference, reference, require)
from python.offline_split import JAVA_MASK, episode_id


SCHEMA = 'bg128-fleet-outage-v1'
CAMPAIGN = 'bg128-baseline-v2-active6-smoke-20261122-001'
SCENE = 'nominal_fleet_outage_6'
POLICY = 'bg128'
STRATEGIES = (POLICY,)
SLOTS = COUNT = 5
FIRST_RUN = 10033001
DISABLED_TRUCK_IDS = (2, 5, 8)
DISABLED_MASK = sum(1 << truck for truck in DISABLED_TRUCK_IDS)
PHYSICAL_COUNT = 9
ACTIVE_COUNT = PHYSICAL_COUNT-len(DISABLED_TRUCK_IDS)
OUTAGE_RPC = {'config.fleet.outage_version':1.0, 'config.fleet.disabled_mask':float(DISABLED_MASK),
              'config.fleet.active_count':float(ACTIVE_COUNT)}
FLEET = dict(outage_version=1, disabled_truck_ids=list(DISABLED_TRUCK_IDS), disabled_mask=DISABLED_MASK,
             active_count=ACTIVE_COUNT, physical_count=PHYSICAL_COUNT, scene_id=SCENE)
PREPARED = PLAN/'prepared_001'
SUITE = PREPARED/'suite.json'
OUTPUT = ROOT/'runs'/CAMPAIGN
CONTROL = OUTPUT/'control'
EVENTS = OUTPUT/'anylogic_events'
REPORT = smoke.REPORT.parent/'fleet_outage_001'
REPORT_FILENAME = 'fleet_outage_evaluation.json'
LAYOUT = dict(python_output=str(OUTPUT), control_directory=str(CONTROL), report_directory=str(REPORT),
              event_root=str(EVENTS))
PAIRING = dict(scope='this_campaign_only', slots=SLOTS, policies=list(STRATEGIES), order='slot_then_policy')


def source_contract():
    refs, learner, rpc, scene = nominal.source_contract()
    refs['nominal_v2_suite'] = reference(nominal.SUITE)
    require(rpc['config.num_trucks'] == rpc['config.runtime_scene.trucks_count']
            == rpc['config.runtime_scene.env.numTrucks'] == PHYSICAL_COUNT and not rpc.keys() & OUTAGE_RPC.keys(),
            'pinned nominal fixture differs from the nine-truck physical contract')
    return refs, learner, {**rpc, **OUTAGE_RPC}, scene


def source_pins():
    prior = nominal.load_suite()
    files = ('fleet_plan.py', 'fleet_ledger.py', 'java_adapter.py', 'prepare.py', 'launch.py', 'fleet_runtime.py',
             'anylogic_fleet_outage.java.txt')
    return [*prior['source_pins'], *[reference(PLAN/name) for name in files]]


def seed(stream, slot):
    return int.from_bytes(sha256(f'{CAMPAIGN}/{stream}/{slot}'.encode('ascii')).digest()[:6], 'big')


def job_for(slot, learner, rpc, scene):
    run, environment, sampling = FIRST_RUN+slot, seed('environment',slot), seed('actor-sampling',slot)
    leaf = f'run_{run}'
    recording = f'{CAMPAIGN}-{slot:03d}-bg128'
    event_dir = EVENTS/leaf
    return dict(ordinal=slot, pair_ordinal=slot, policy_kind=POLICY, actor_seed=learner['seed'],
                group='BASELINE', role='final', source_recording_id=recording, recording_id=recording,
                run_index=run, experiment_seed=environment, actor_sampling_seed=sampling,
                actor_sampling_seed_consumed=True, event_directory=str(event_dir),
                event_bundle=str(event_dir/episode_id(run,environment)), python_output=str(OUTPUT/'shifts'/leaf),
                target_training_update=0, checkpoint_update=32, scene_id=SCENE, expected_rpc_config=rpc,
                runtime_scene_sha256=scene)


def phase_path(job):
    return PREPARED/'configs'/f"run_{job['run_index']}.json"


def config_for(job, refs):
    cfg = nominal.config_for(job, refs)
    cfg.update(campaign_id=CAMPAIGN, fleet=FLEET)
    return cfg


def validate_jobs(jobs):
    require(len(jobs) == COUNT, 'five physical fleet outage jobs required')
    for key in ('run_index', 'recording_id', 'source_recording_id', 'event_directory', 'event_bundle', 'python_output'):
        require(len({row[key] for row in jobs}) == COUNT, 'duplicate physical '+key)
    used = set()
    for slot, job in enumerate(jobs):
        env, actor = seed('environment',slot), seed('actor-sampling',slot)
        require(0 < env <= JAVA_MASK and 0 < actor <= JAVA_MASK and env != actor and used.isdisjoint((env,actor)),
                'five independent environment and Actor seed pairs required')
        used.update((env,actor))
        require(job['ordinal'] == job['pair_ordinal'] == slot and job['run_index'] == FIRST_RUN+slot
                and job['policy_kind'] == POLICY and job['scene_id'] == SCENE and job['actor_seed'] > 0
                and job['experiment_seed'] == env and job['actor_sampling_seed'] == actor
                and job['actor_sampling_seed_consumed'] is True
                and all(job['expected_rpc_config'].get(k) == v for k,v in OUTAGE_RPC.items()),
                'frozen fleet slot order, seed or RPC outage metadata differs')


def load_suite(path=SUITE, *, check_sources=True):
    path = Path(path)
    require(path.resolve() == SUITE, 'frozen fleet outage suite path required')
    suite = load_reference(reference(path))
    require(path.read_bytes() == encoded(suite), 'noncanonical fleet outage suite bytes')
    require(suite['schema'] == SCHEMA and suite['campaign_id'] == CAMPAIGN and suite['execution'] == EXECUTION
            and suite['method'] == METHOD and all(suite.get(key) == val for key, val in LAYOUT.items())
            and suite['physical_shifts'] == COUNT and suite['evaluation_slots'] == SLOTS
            and suite['training_shifts'] == suite['optimizer_updates'] == 0 and suite['pairing'] == PAIRING
            and suite['checkpoint_selection'] == 'mixed-s02-u32' and suite['fleet'] == FLEET,
            'fleet outage suite scope or execution differs')
    refs, learner, rpc, scene = source_contract()
    require(suite['learner_config'] == learner and all(suite.get(key) == val for key, val in refs.items()),
            'frozen nominal source or Actor checkpoint differs')
    validate_jobs(suite['jobs'])
    for ordinal, job in enumerate(suite['jobs']):
        require({key: val for key, val in job.items() if key != 'phase_config'} == job_for(ordinal,learner,rpc,scene),
                'frozen fleet physical job differs')
        config = phase_path(job)
        require(reference(config) == job['phase_config'] and config.read_bytes() == encoded(config_for(job,refs)),
                'frozen fleet phase bytes differ')
    for key, name in [('master_plan', 'master_plan.csv'), ('scenario_table', 'scenario_table.csv'),
                      ('java_helper', 'Baseline1CampaignPlan.java'), ('run_seed', 'run_seed.csv')]:
        require(reference(PREPARED/name) == suite[key], 'frozen launch artifact differs: '+key)
    if check_sources:
        require(suite['source_pins'] == source_pins(), 'fleet outage source bytes differ')
    return suite
