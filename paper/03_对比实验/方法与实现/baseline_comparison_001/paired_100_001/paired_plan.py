"""One frozen nominal protocol with 100 paired slots and four policies per slot."""
from hashlib import sha256
from pathlib import Path
import sys

PLAN = Path(__file__).resolve().parent
if str(PLAN.parent) not in sys.path:
    sys.path.append(str(PLAN.parent))

import smoke_plan as smoke
from smoke_plan import (ROOT, STRATEGIES, EXECUTION, METHOD, PYTHON, CORE, CORE_SHA, COLUMNS,
                        encoded, load_reference, reference, require)
from python.offline_split import JAVA_MASK, episode_id
from python.runtime_scenario import PREFIX, fingerprint


SCHEMA = 'bg128-traditional-paired100-v1'
PHASE_SCHEMA = 'bg128-traditional-paired100-phase-v1'
CAMPAIGN = 'bg128-baseline-paired100-20261122-001'
SLOTS = 100
COUNT = SLOTS*len(STRATEGIES)
FIRST_RUN = 10021001
PREPARED = PLAN/'prepared_001'
SUITE = PREPARED/'suite.json'
OUTPUT = ROOT/'runs'/CAMPAIGN
CONTROL = OUTPUT/'control'
REPORT = smoke.REPORT/'paired_100_001'
EVENTS = OUTPUT/'anylogic_events'
SOURCE = ROOT/'runs/bg128-continuous-mixed-20261120-s02'
NOMINAL = SOURCE/'evaluation/nominal/new'
MODEL = SOURCE/'arms/s01_BG128/updates/0032'
MODEL_REF_FIELDS = ('actor', 'actor_manifest', 'checkpoint', 'checkpoint_manifest', 'checkpoint_seal')
LAYOUT = dict(python_output=str(OUTPUT), control_directory=str(CONTROL), report_directory=str(REPORT),
              event_root=str(EVENTS))
PAIRING = dict(scope='this_campaign_only', slots=SLOTS, share='experiment_seed',
               policies=list(STRATEGIES), order='slot_then_policy')


def source_contract():
    refs = dict(protected_core=reference(CORE), nominal_source=reference(NOMINAL/'phase_config.json'),
                checkpoint=reference(MODEL/'checkpoint.pt'), actor=reference(NOMINAL/'actor.pt'),
                actor_manifest=reference(NOMINAL/'actor.pt.manifest.json'),
                source_completion=reference(SOURCE/'suite_complete.json'),
                checkpoint_manifest=reference(MODEL/'checkpoint.pt.manifest.json'),
                checkpoint_seal=reference(MODEL/'checkpoint.pt.seal.json'),
                smoke_completion=reference(smoke.OUTPUT/'suite_complete.json'),
                smoke_evaluation=reference(smoke.REPORT/'single_shift_validation.json'))
    expected = dict(protected_core=CORE_SHA,
                    nominal_source='f3ff8a2a712812231f6e6d4af82704e1c750128d112073c5cdd133201e2666c0',
                    checkpoint='67b7b2c7104079836ccef8f5747e7aebb4e9e5a92dcaac67c94e42412f104f41',
                    actor='e1575fec566c647a882c63934ec57c908cef28d43f076431ef9103cc9b73e0a9',
                    source_completion='45d7735b6612b55404bdf240f2c7885667f5e80ba07ffaea07aa38eefdb808fc')
    require(all(refs[name]['sha256'] == digest for name, digest in expected.items()), 'frozen s02 or PPO bytes differ')
    completion = load_reference(refs['source_completion'])
    require(completion['status'] == 'COMPLETE' and completion['new_checkpoint'] == refs['checkpoint']
            and completion['training_shifts'] == 1024 and completion['training_updates'] == 32
            and completion['evaluation_shifts'] == 700 and completion['evaluation_optimizer_updates'] == 0,
            's02 fixed u32 training/evaluation completion differs')
    refs['source_suite'] = completion['suite']
    learner = load_reference(refs['source_suite'])['learner_config']
    nominal = load_reference(refs['nominal_source'])
    template = nominal['jobs'][0]
    require(nominal['scenario_id'] == 'nominal' and template['actor_seed'] == learner['seed'],
            'selected checkpoint and nominal scenario are not bound to s02')
    rpc = dict(template['expected_rpc_config'], **{'config.material_event_memory_enabled': 0.0,
                                                 'config.wait_breakdown_enabled': 1.0,
                                                 'config.wait_breakdown_version': 1.0})
    scene = fingerprint({key: val for key, val in rpc.items() if key.startswith(PREFIX)})
    _, _, smoke_rpc, smoke_scene = smoke.source_contract()
    require(template['runtime_scene_sha256'] == scene == smoke_scene and rpc == smoke_rpc,
            'formal comparison physics or low-frequency diagnostics differ from the validated nominal setup')
    smoke_completion = load_reference(refs['smoke_completion'])
    smoke_report = load_reference(refs['smoke_evaluation'])
    require(smoke_completion['status'] == smoke_report['status'] == 'COMPLETE'
            and smoke_completion['physical_shifts'] == smoke_report['completed_shifts'] == len(STRATEGIES)
            and smoke_completion['evaluation'] == refs['smoke_evaluation']
            and smoke_completion['optimizer_updates'] == 0
            and {row['policy_kind'] for row in smoke_report['shifts']} == set(STRATEGIES),
            'four-strategy smoke completion does not match the physical report')
    return refs, learner, rpc, scene


def source_pins():
    local = ('paired_plan.py', 'paired_budget.py', 'java_adapter.py', 'prepare.py', 'launch.py', 'paired_runtime.py',
             'paired_report.py', 'selection.py')
    return [*smoke.source_pins(), *[reference(PLAN/name) for name in local]]


def selection_evidence(refs):
    ref = reference(REPORT/'checkpoint_selection.json')
    evidence = load_reference(ref)
    require(evidence['schema'] == 'bg128-nominal-paired-100-checkpoint-selection-v1'
            and evidence['selected_seed'] == 's02' and evidence['selected_update'] == 32
            and evidence['selected_checkpoint'] == refs['checkpoint']
            and evidence['selected_checkpoint_manifest'] == refs['checkpoint_manifest']
            and evidence['selected_actor'] == refs['actor'] and evidence['selected_actor_manifest'] == refs['actor_manifest'],
            'checkpoint selection evidence must bind the frozen nominal s02 policy')
    return ref


def seed(stream, slot):
    return int.from_bytes(sha256(f'{CAMPAIGN}/{stream}/{slot}'.encode('ascii')).digest()[:6], 'big')


def job_for(ordinal, learner, rpc, scene):
    slot, strategy = divmod(ordinal, len(STRATEGIES))
    policy = STRATEGIES[strategy]
    actor = policy == 'bg128'
    environment = seed('environment', slot)
    run = FIRST_RUN+ordinal
    directory = f'run_{run}'
    current_output, event_directory = OUTPUT/'shifts'/directory, EVENTS/directory
    recording = f'{CAMPAIGN}-{slot:03d}-{policy}'
    return dict(ordinal=ordinal, pair_ordinal=slot, policy_kind=policy, actor_seed=learner['seed'] if actor else 0,
                group='BASELINE', role='final', source_recording_id=recording, recording_id=recording,
                run_index=run, experiment_seed=environment, actor_sampling_seed=seed('actor-sampling', slot) if actor else 0,
                actor_sampling_seed_consumed=actor, event_directory=str(event_directory),
                event_bundle=str(event_directory/episode_id(run, environment)), python_output=str(current_output),
                target_training_update=0, checkpoint_update=32 if actor else 0, scene_id='nominal',
                expected_rpc_config=rpc, runtime_scene_sha256=scene)


def config_for(job, refs):
    cfg = smoke.config_for(job, refs)
    cfg.update(schema=PHASE_SCHEMA, campaign_id=CAMPAIGN)
    return cfg


def phase_path(job):
    return PREPARED/'configs'/f"run_{job['run_index']}.json"


def validate_jobs(jobs):
    require(len(jobs) == COUNT, '400 physical jobs required')
    for key in ('run_index', 'recording_id', 'source_recording_id', 'event_directory', 'event_bundle', 'python_output'):
        require(len({row[key] for row in jobs}) == COUNT, 'duplicate physical ' + key)
    distinct = set()
    for slot in range(SLOTS):
        start = slot*len(STRATEGIES)
        group = jobs[start:start+len(STRATEGIES)]
        environment, sampling = seed('environment', slot), seed('actor-sampling', slot)
        require(0 < environment <= JAVA_MASK and 0 < sampling <= JAVA_MASK and environment != sampling
                and distinct.isdisjoint((environment, sampling)), 'slot seed isolation differs')
        distinct.update((environment, sampling))
        for i, job in enumerate(group):
            actor = STRATEGIES[i] == 'bg128'
            require(job['ordinal'] == start+i and job['pair_ordinal'] == slot and job['policy_kind'] == STRATEGIES[i]
                    and job['run_index'] == FIRST_RUN+start+i and job['experiment_seed'] == environment
                    and job['actor_sampling_seed'] == (sampling if actor else 0)
                    and job['actor_sampling_seed_consumed'] is actor
                    and (job['actor_seed'] > 0 if actor else job['actor_seed'] == 0),
                    'physical identity order or scoped seed assignment differs')


def load_suite(path=SUITE, *, check_sources=True):
    require(Path(path).resolve() == SUITE, 'frozen paired suite path required')
    path = Path(path)
    suite = load_reference(reference(path))
    require(path.read_bytes() == encoded(suite), 'noncanonical paired suite bytes')
    require(suite['schema'] == SCHEMA and suite['campaign_id'] == CAMPAIGN and suite['execution'] == EXECUTION
            and suite['method'] == METHOD and all(suite.get(key) == value for key, value in LAYOUT.items()),
            'paired suite campaign or execution contract differs')
    require(suite['pairing'] == PAIRING and suite['evaluation_slots'] == SLOTS and suite['physical_shifts'] == COUNT
            and suite['training_shifts'] == suite['optimizer_updates'] == 0
            and suite['checkpoint_selection'] == 'mixed-s02-u32', 'paired evaluation scope differs')
    refs, learner, rpc, scene = source_contract()
    require(suite['learner_config'] == learner and all(suite.get(key) == val for key, val in refs.items()),
            'frozen source contract differs')
    require(selection_evidence(refs) == suite['checkpoint_selection_evidence'], 'frozen selection evidence differs')
    jobs = suite['jobs']
    validate_jobs(jobs)
    for i, job in enumerate(jobs):
        actual = {key: value for key, value in job.items() if key != 'phase_config'}
        require(actual == job_for(i, learner, rpc, scene), 'frozen paired job differs')
        phase = phase_path(job)
        require(reference(phase) == job['phase_config'] and phase.read_bytes() == encoded(config_for(job, refs)),
                'frozen paired phase bytes differ')
    for key, name in [('master_plan', 'master_plan.csv'), ('scenario_table', 'scenario_table.csv'),
                      ('java_helper', 'Baseline1CampaignPlan.java'), ('run_seed', 'run_seed.csv')]:
        require(reference(PREPARED/name) == suite[key], 'frozen paired launch artifact differs: ' + key)
    if check_sources:
        require(suite['source_pins'] == source_pins(), 'paired implementation source bytes differ')
    return suite
