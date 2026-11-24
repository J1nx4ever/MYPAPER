"""Freeze five paired slots across four six-active BG128 truck rosters."""
from hashlib import sha256
from pathlib import Path
import sys


PLAN = Path(__file__).resolve().parent
PRIOR = PLAN.parent/'fleet_outage_001'
sys.path.insert(0, str(PRIOR))
try:
    import fleet_plan as previous
finally:
    sys.path.remove(str(PRIOR))

from fleet_plan import (ROOT, CORE, EXECUTION, PYTHON, METHOD, PHASE_SCHEMA, smoke,
                        encoded, load_reference, reference, require)
from python.offline_split import JAVA_MASK, episode_id


SCHEMA = 'bg128-fleet-outage-comparison-v1'
CAMPAIGN = 'bg128-baseline-v2-active6-mix-paired-20261122-001'
POLICY = 'bg128'
STRATEGIES = (POLICY,)
ROSTERS = {'heavy_out': (6, 7, 8), 'balanced_out': (2, 5, 8), 'small_out': (0, 1, 2), 'medium_out': (3, 4, 5)}
SCENARIOS = tuple(ROSTERS)
SLOTS, COUNT = 5, 5*len(SCENARIOS)
FIRST_RUN = 10034001
PHYSICAL_COUNT, ACTIVE_COUNT, OUTAGE_VERSION = 9, 6, 1
FLEET = {scenario: dict(scenario_id=scenario, scene_id=f'nominal_fleet_outage_{scenario}',
                        outage_version=OUTAGE_VERSION, disabled_truck_ids=list(ids),
                        disabled_mask=sum(1 << truck for truck in ids), active_count=ACTIVE_COUNT,
                        physical_count=PHYSICAL_COUNT) for scenario, ids in ROSTERS.items()}
MASKS = tuple(fleet['disabled_mask'] for fleet in FLEET.values())
PREPARED = PLAN/'prepared_001'
SUITE = PREPARED/'suite.json'
OUTPUT = ROOT/'runs'/CAMPAIGN
CONTROL = OUTPUT/'control'
EVENTS = OUTPUT/'anylogic_events'
REPORT = smoke.REPORT.parent/'fleet_outage_002'
REPORT_FILENAME = 'fleet_outage_comparison.json'
LAYOUT = dict(python_output=str(OUTPUT), control_directory=str(CONTROL), report_directory=str(REPORT),
              event_root=str(EVENTS))
PAIRING = dict(scope='this_campaign_only', slots=SLOTS, policies=list(STRATEGIES), scenarios=list(SCENARIOS),
               share=['experiment_seed', 'actor_sampling_seed'], order='slot_then_scenario')


def source_contract():
    refs, learner, rpc, scene = previous.nominal.source_contract()
    prior = previous.load_suite()
    refs.update(nominal_v2_suite=prior['nominal_v2_suite'], single_roster_suite=reference(previous.SUITE))
    require(rpc['config.num_trucks'] == rpc['config.runtime_scene.trucks_count']
            == rpc['config.runtime_scene.env.numTrucks'] == PHYSICAL_COUNT and not rpc.keys() & previous.OUTAGE_RPC.keys(),
            'nominal fixture differs from the nine-truck physical contract')
    return refs, learner, rpc, scene


def source_pins():
    prior = previous.load_suite()
    names = ('outage_plan.py', 'outage_ledger.py', 'outage_java.py', 'prepare.py', 'launch.py', 'fleet_runtime.py',
             'anylogic_fleet_outage.java.txt')
    return [*prior['source_pins'], prior['java_helper'], *[reference(PLAN/name) for name in names]]


def seed(stream, slot):
    return int.from_bytes(sha256(f'{CAMPAIGN}/{stream}/{slot}'.encode('ascii')).digest()[:6], 'big')


def outage_rpc(fleet):
    return {'config.fleet.outage_version':float(fleet['outage_version']), 'config.fleet.disabled_mask':float(fleet['disabled_mask']),
            'config.fleet.active_count':float(fleet['active_count'])}


def job_for(ordinal, learner, rpc, scene):
    slot, arm = divmod(ordinal, len(SCENARIOS))
    scenario = SCENARIOS[arm]
    fleet = FLEET[scenario]
    run, environment, sampling = FIRST_RUN+ordinal, seed('environment',slot), seed('actor-sampling',slot)
    leaf = f'run_{run}'
    recording = f'{CAMPAIGN}-{slot:03d}-{scenario}'
    event_dir = EVENTS/leaf
    return dict(ordinal=ordinal, pair_ordinal=slot, policy_kind=POLICY, actor_seed=learner['seed'],
                group='BASELINE', role='final', source_recording_id=recording, recording_id=recording,
                run_index=run, experiment_seed=environment, actor_sampling_seed=sampling,
                actor_sampling_seed_consumed=True, event_directory=str(event_dir),
                event_bundle=str(event_dir/episode_id(run,environment)), python_output=str(OUTPUT/'shifts'/leaf),
                target_training_update=0, checkpoint_update=32, scene_id=fleet['scene_id'], scenario_id=scenario,
                fleet=fleet, expected_rpc_config={**rpc, **outage_rpc(fleet)}, runtime_scene_sha256=scene)


def phase_path(job):
    return PREPARED/'configs'/f"run_{job['run_index']}.json"


def config_for(job, refs):
    cfg = previous.nominal.config_for(job, refs)
    cfg.update(campaign_id=CAMPAIGN, fleet=job['fleet'])
    return cfg


def validate_jobs(jobs):
    require(len(jobs) == COUNT, 'twenty physical fleet outage jobs required')
    for key in ('run_index', 'recording_id', 'source_recording_id', 'event_directory', 'event_bundle', 'python_output'):
        require(len({row[key] for row in jobs}) == COUNT, 'duplicate physical '+key)
    used = set()
    for slot in range(SLOTS):
        env, actor = seed('environment',slot), seed('actor-sampling',slot)
        require(0 < env <= JAVA_MASK and 0 < actor <= JAVA_MASK and env != actor and used.isdisjoint((env,actor)),
                'five independent environment and Actor seed pairs required')
        used.update((env,actor))
        for arm, scenario in enumerate(SCENARIOS):
            ordinal = slot*len(SCENARIOS)+arm
            job, fleet = jobs[ordinal], FLEET[scenario]
            require(job['ordinal'] == ordinal and job['pair_ordinal'] == slot and job['run_index'] == FIRST_RUN+ordinal
                    and job['policy_kind'] == POLICY and job['scenario_id'] == scenario and job['scene_id'] == fleet['scene_id']
                    and job['fleet'] == fleet and job['actor_seed'] > 0 and job['checkpoint_update'] == 32
                    and job['experiment_seed'] == env and job['actor_sampling_seed'] == actor
                    and job['actor_sampling_seed_consumed'] is True
                    and all(job['expected_rpc_config'].get(key) == val for key,val in outage_rpc(fleet).items()),
                    'frozen fleet slot, ordered roster, paired seed or expected RPC differs')


def load_suite(path=SUITE, *, check_sources=True):
    path = Path(path)
    require(path.resolve() == SUITE, 'frozen fleet outage comparison suite path required')
    suite = load_reference(reference(path))
    require(path.read_bytes() == encoded(suite), 'noncanonical fleet outage comparison suite bytes')
    require(suite['schema'] == SCHEMA and suite['campaign_id'] == CAMPAIGN and suite['execution'] == EXECUTION
            and suite['method'] == METHOD and all(suite.get(key) == val for key,val in LAYOUT.items())
            and suite['physical_shifts'] == COUNT and suite['evaluation_slots'] == SLOTS
            and suite['training_shifts'] == suite['optimizer_updates'] == 0 and suite['pairing'] == PAIRING
            and suite['checkpoint_selection'] == 'mixed-s02-u32' and suite['fleet'] == FLEET,
            'fleet outage comparison scope or execution differs')
    refs, learner, rpc, scene = source_contract()
    require(suite['learner_config'] == learner and all(suite.get(key) == val for key,val in refs.items()),
            'frozen nominal source or Actor checkpoint differs')
    validate_jobs(suite['jobs'])
    for ordinal, job in enumerate(suite['jobs']):
        require({key: val for key,val in job.items() if key != 'phase_config'} == job_for(ordinal,learner,rpc,scene),
                'frozen fleet physical job differs')
        config = phase_path(job)
        require(reference(config) == job['phase_config'] and config.read_bytes() == encoded(config_for(job,refs)),
                'frozen fleet phase bytes differ')
    for key, name in [('master_plan', 'master_plan.csv'), ('scenario_table', 'scenario_table.csv'),
                      ('java_helper', 'Baseline1CampaignPlan.java'), ('run_seed', 'run_seed.csv')]:
        require(reference(PREPARED/name) == suite[key], 'frozen launch artifact differs: '+key)
    if check_sources:
        require(suite['source_pins'] == source_pins(), 'fleet outage comparison source bytes differ')
    return suite
