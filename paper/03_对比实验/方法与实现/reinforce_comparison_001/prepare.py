"""Freeze a training-only BG128 REINFORCE package from the completed s02 PPO schedule."""

import csv
from functools import lru_cache
from pathlib import Path
import sys

PLAN = Path(__file__).resolve().parent
ROOT = PLAN.parents[3]
MIXED = PLAN.parent / 'generalization_001/grade_assignment_generalization_001/continuous_training_001'
for directory in (ROOT, MIXED):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import mixed_plan as original
from business_graph_summary_campaign_v1.plan_io import load_reference, load_suite, reference, write


CAMPAIGN = 'bg128-reinforce-full-shift-20261127-s02'
SCHEMA = 'bg128-reinforce-full-shift-suite-v1'
FIRST_RUN = 10090001
UPDATES = 32
BATCH_SIZE = 32
COUNT = UPDATES * BATCH_SIZE
RPC_PORT = 50051
PREPARED = PLAN / 'prepared_001'
SUITE = PREPARED / 'suite.json'
OUTPUT = ROOT / 'runs' / CAMPAIGN
CONTROL = OUTPUT / 'control'
MATERIAL = OUTPUT.with_name(CAMPAIGN + '_materials')
REPORT = ROOT / 'reports' / PLAN.relative_to(ROOT / 'change_plans')
LEDGER = OUTPUT / 'budget.sqlite'
ARM = OUTPUT / 'arms/s01_BG128'
ORIGINAL_OUTPUT = ROOT / 'runs/bg128-continuous-mixed-20261120-s02'
COMPLETION = dict(path=str(ORIGINAL_OUTPUT / 'suite_complete.json'),
                  sha256='45d7735b6612b55404bdf240f2c7885667f5e80ba07ffaea07aa38eefdb808fc')
INITIAL_ACTOR = dict(path=str(ORIGINAL_OUTPUT / 'arms/s01_BG128/initial.pt'),
                     sha256='418cd7a5edd9a6b8f25b1892b6a7e9456a164a98f9cde0e3100fdc99fbf08a26')
INITIAL_DIGEST = '2e3685082d4bbf607153062c7da0e820aef94ae168ec5cbf88cadd0165b4ed32'
FINAL_PPO = dict(path=str(ORIGINAL_OUTPUT / 'arms/s01_BG128/updates/0032/checkpoint.pt'),
                 sha256='67b7b2c7104079836ccef8f5747e7aebb4e9e5a92dcaac67c94e42412f104f41')
PROTECTED_CORE = dict(path=str(ROOT.parents[1] / 'python/episode_budget_ppo.py'),
                      sha256='8e4cfa31c622f5e33756e304eea3f7526547db72ae93f7d9f00808845d61ef96')


@lru_cache(maxsize=1)
def historical():
    """Bind the historical package, initial weights and the original 1,024 slots."""
    completed = load_reference(COMPLETION)
    source_ref = completed['suite']
    original.require(reference(source_ref['path']) == source_ref
                     and completed['status'] == 'COMPLETE'
                     and completed['training_shifts'] == COUNT and completed['training_updates'] == UPDATES
                     and completed['initial_checkpoint'] == INITIAL_ACTOR
                     and reference(INITIAL_ACTOR['path']) == INITIAL_ACTOR
                     and completed['new_checkpoint'] == FINAL_PPO and reference(FINAL_PPO['path']) == FINAL_PPO,
                     'historical mixed s02 training or checkpoint differs')
    suite, all_phases = load_suite(source_ref['path'])
    train = all_phases[:UPDATES]
    original.require(all(phase['policy_arm'] == 'train' and phase['arm_update'] == update
                         and len(phase['jobs']) == BATCH_SIZE
                         for update, phase in enumerate(train, 1))
                     and len(train) == UPDATES and suite['replica'] == 's02'
                     and suite['candidate_id'] == 's01_BG128' and suite['actor_seed'] == suite['arms'][0]['actor_seed'],
                     'historical s02 training phase or Actor identity differs')
    checkpoint = Path(INITIAL_ACTOR['path'])
    manifest_ref = reference(str(checkpoint) + '.manifest.json')
    seal_ref = reference(str(checkpoint) + '.seal.json')
    manifest = load_reference(manifest_ref)
    original.require(manifest['sha256'] == INITIAL_ACTOR['sha256'] and manifest['update_index'] == 0
                     and manifest['actor_digest'] == INITIAL_DIGEST
                     and load_reference(seal_ref) == {'sha256': INITIAL_ACTOR['sha256'],
                                                       'manifest_sha256': manifest_ref['sha256']},
                     'archived Actor weight or checkpoint seal differs')
    return suite, train, completed, manifest_ref, seal_ref


def documents():
    source, training, completed, manifest, seal = historical()
    copied_fields = ('candidate_id', 'arm_order', 'stage', 'execution_profile', 'training_updates',
                     'first_update', 'final_update', 'episodes_per_batch', 'training_seed_count',
                     'cpu_threads', 'sampling', 'pair_id', 'paired_slots', 'active_h2_reserved_shifts',
                     'legacy_reserved_shifts', 'mean_ranges', 'half_width_range', 'heldout_orders',
                     'learner_config', 'actor_seed')
    fixed = {key: source[key] for key in copied_fields}
    fixed.update(schema=SCHEMA, campaign_id=CAMPAIGN, baseline_seed='s02', baseline_completion=COMPLETION,
                 baseline_suite=completed['suite'], baseline_final_checkpoint=FINAL_PPO,
                 initial_actor=INITIAL_ACTOR, initial_actor_manifest=manifest, initial_actor_seal=seal,
                 initial_actor_digest=INITIAL_DIGEST, protected_core=PROTECTED_CORE,
                 python_output=str(OUTPUT), control_directory=str(CONTROL), material_root=str(MATERIAL),
                 report_directory=str(REPORT), suite_path=str(SUITE), budget_path=str(LEDGER),
                 run_index_base=FIRST_RUN, rpc_port=RPC_PORT, training_shifts=COUNT, evaluation_shifts=0,
                 new_shift_cap=COUNT, evaluation_cases=[], arms=[dict(source['arms'][0], python_output=str(ARM))],
                 reinforce=dict(actor_lr=3e-4, adam_eps=1e-5, baseline_decay=.5, episodes_per_batch=BATCH_SIZE))
    phases = []
    for historic_phase in training:
        phase_id = historic_phase['phase_id']
        recording = f'{CAMPAIGN}-{phase_id}'
        output = ARM / historic_phase['arm_phase_id']
        events = CONTROL / 'legacy_unused' / phase_id
        material = MATERIAL / phase_id
        jobs = []
        for prior in historic_phase['jobs']:
            ordinal = prior['ordinal']
            run = FIRST_RUN + ordinal
            env = original.seed(CAMPAIGN, 'environment', ordinal)
            actor = original.seed(CAMPAIGN, 'sampling', ordinal)
            group = f"{CAMPAIGN}-{prior['scene_id']}-{prior['pair_ordinal']:04d}"
            leaf = f'dispatch-production-wait-shortfall-v2-run-{run}-seed-{env}'
            jobs.append(dict(prior, run_index=run, experiment_seed=env, actor_sampling_seed=actor,
                             recording_id=recording, source_recording_id=recording, scene_group=group,
                             realization_group_id=group, python_output=str(output), event_directory=str(events),
                             event_bundle=str(events / leaf), material_event_directory=str(material),
                             material_event_bundle=str(material / leaf),
                             material_journal=str(material / leaf / 'material_events.bin')))
        phase = dict(historic_phase, campaign_id=CAMPAIGN, recording_id=recording, python_output=str(output),
                     event_directory=str(events), execution=dict(historic_phase['execution'], port=RPC_PORT),
                     jobs=jobs)
        original.check_phase(phase)
        phases.append(phase)
    return fixed, phases


def source_pins():
    from business_graph_summary_v1.learner import implementation_signature

    runtime = ROOT / 'business_graph_reinforce_v1'
    sources = [PLAN / name for name in ('prepare.py', 'prepare_java.py', 'budget.py', 'run.py')]
    sources.extend(path for path in runtime.glob('*.py') if not path.name.startswith('test_'))
    sources.extend(MIXED / name for name in ('mixed_plan.py', 'mixed_java.py', 'mixed_runtime.py',
                                              'recovery_002/storage_scope.py', 'recovery_003/transport.py'))
    for package in ('business_graph_summary_campaign_v1', 'business_graph_summary_storage_v1',
                    'baseline1_v1', 'grade_only_ppo_v2'):
        sources.extend(path for path in (ROOT / package).glob('*.py') if not path.name.startswith('test_'))
    sources.extend(ROOT / 'python' / name for name in ('collect_offline.py', 'constrained_batch.py', 'dispatch_env.py',
                   'dispatch_bridge.py', 'dispatch_service.py', 'runtime_scenario.py', 'native_behavior.py',
                   'reward_config.py', 'reward_contract.py', 'generated/dispatch_pb2_grpc.py',
                   'generated/dispatch_pb2.py'))
    sources.append(ROOT / 'proto/dispatch.proto')
    actor_files = implementation_signature()['sources_sha256']
    pins = {str(ROOT.parents[1] / path): sha256 for path, sha256 in actor_files.items()}
    pins.update({str(path): reference(path)['sha256'] for path in sources})
    return dict(sorted(pins.items()))


def _csv(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def load_package():
    fixed, expected = documents()
    suite, phases = load_suite(SUITE)
    original.require(all(suite.get(key) == value for key, value in fixed.items()) and phases == expected,
                     'frozen REINFORCE campaign or original physical schedule differs')
    original.require(suite['source_pins'] == source_pins() and reference(PROTECTED_CORE['path']) == PROTECTED_CORE,
                     'REINFORCE source pins or protected PPO core changed')
    for key, name in (('master_plan', 'master_plan.csv'), ('identities', 'identities.csv'),
                      ('scenario_table', 'scenarios.csv'), ('java_helper', 'Baseline1CampaignPlan.java')):
        original.require(suite[key] == reference(PREPARED / name), 'frozen artifact differs: ' + key)
    phase_paths = [PREPARED / 'configs' / f"{phase['phase_id']}.json" for phase in phases]
    original.require(suite['phase_configs'] == [reference(path) for path in phase_paths]
                     and suite['phases'] == [original._phase_entry(phase, ref)
                                              for phase, ref in zip(phases, suite['phase_configs'], strict=True)],
                     'frozen phase references differ')
    original.require(_csv(PREPARED / 'master_plan.csv') == [
        {key: str(job[key]) for key in original.COLUMNS} for job in suite['jobs']]
        and _csv(PREPARED / 'identities.csv') == [{key: str(value) for key, value in row.items()}
              for row in original._identity_rows(phases, suite['phase_configs'])], 'frozen identity table differs')
    return suite, phases


def prepare():
    from .budget import assert_available, budget_snapshot
    from .prepare_java import render

    original.require(not any(path.exists() for path in (PREPARED, OUTPUT, MATERIAL)),
                     'preparation needs fresh immutable package and runtime paths')
    settings, phases = documents()
    jobs = [job for phase in phases for job in phase['jobs']]
    check = assert_available(dict(settings, jobs=jobs), budget_snapshot())
    pins = source_pins()
    refs = []
    for phase in phases:
        path = PREPARED / 'configs' / (phase['phase_id'] + '.json')
        write(path, phase)
        refs.append(reference(path))
    master, identities = PREPARED / 'master_plan.csv', PREPARED / 'identities.csv'
    original._write_rows(master, original.COLUMNS, jobs)
    original._write_rows(identities, original.IDENTITY_COLUMNS, original._identity_rows(phases, refs))
    helper, java_check = render(settings, jobs, master, PREPARED)
    phase_entries = [original._phase_entry(phase, ref) for phase, ref in zip(phases, refs, strict=True)]
    stored = dict(settings, phase_configs=refs, phases=phase_entries, source_pins=pins, master_plan=reference(master),
                  scenario_table=reference(PREPARED / 'scenarios.csv'), java_helper=reference(helper),
                  identities=reference(identities), java_validation=java_check)
    write(SUITE, stored)
    load_package()
    launch = dict(campaign_id=CAMPAIGN, suite=reference(SUITE), training_shifts=COUNT, evaluation_shifts=0,
                  python_output=str(OUTPUT), anylogic_event_root=str(MATERIAL), rpc_port=RPC_PORT,
                  first=next(original._identity_rows(phases, refs)), identities=stored['identities'],
                  master_plan=stored['master_plan'], java_helper=stored['java_helper'], budget=check,
                  phases=[dict(row, recording_id=phase['recording_id'],
                               material_event_directory=phase['jobs'][0]['material_event_directory'],
                               first_run=phase['jobs'][0]['run_index'], last_run=phase['jobs'][-1]['run_index'])
                          for row, phase in zip(phase_entries, phases, strict=True)])
    write(PREPARED / 'launch_identity.json', launch)
    return launch
