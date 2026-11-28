"""Freeze the seven physical Full S02 evaluation panels for its matched REINFORCE Actor."""
from functools import lru_cache
import json
from pathlib import Path

from .. import prepare as training
from .. import budget as training_budget
from business_graph_summary_campaign_v1.plan_io import load_reference, load_suite, reference, write
from python.collect_offline import validate_runtime
from weak_plan import bind

PLAN = Path(__file__).resolve().parent
ROOT = training.ROOT
original = training.original
require = original.require
CAMPAIGN = 'bg128-reinforce-s02-u32-paired700-20261127-001'
FIRST_RUN = 10092001
COUNT, PER_CASE, RPC_PORT = 700, 100, 50051
PREPARED = PLAN / 'prepared_001'
SUITE = PREPARED / 'suite.json'
OUTPUT = ROOT / 'runs' / CAMPAIGN
MATERIAL = OUTPUT.with_name(CAMPAIGN + '_materials')
CONTROL = OUTPUT / 'control'
REPORT = ROOT / 'reports' / PLAN.relative_to(ROOT / 'change_plans')
LEDGER = OUTPUT / 'budget.sqlite'
COMPLETION = dict(path=str(training.OUTPUT / 'suite_complete.json'),
                  sha256='6fabcaa84bfb9e9efdda903fd4967db9cbeac3c1a3617d6227d5eaddee3b8b8e')
PAIR_KEYS = ('scene_id', 'pair_ordinal', 'expected_rpc_config', 'runtime_scene_sha256',
             'experiment_seed', 'actor_sampling_seed', 'checkpoint_update', 'role', 'partition')


@lru_cache(maxsize=1)
def historical():
    full = load_reference(training.COMPLETION)
    suite, phases = load_suite(full['suite']['path'])
    require(reference(suite['suite_path']) == full['suite'] and full['status'] == 'COMPLETE'
            and suite['replica'] == 's02' and full['new_checkpoint'] == training.FINAL_PPO
            and reference(training.FINAL_PPO['path']) == training.FINAL_PPO
            and full['evaluation_shifts'] == COUNT and full['evaluation_optimizer_updates'] == 0,
            'Full S02 completion or evaluated u32 differs')
    evals = [phase for phase in phases if phase['policy_arm'] == 'new']
    require(list(original.EVAL_CASES) == [phase['scenario_id'] for phase in evals], 'Full S02 panel order differs')
    first_completion = load_reference(suite['source_completion'])
    first_suite = first_completion['campaign']
    require(reference(first_suite['path']) == first_suite, 'shared S01 source suite changed')
    common = [p for p in load_suite(first_suite['path'])[1] if p['policy_arm'] == 'new']
    for src, phase, phase_ref, result in zip(common, evals, suite['phase_configs'][-len(evals):],
                                            full['phase_results'], strict=True):
        report = load_reference(result['report'])
        require(report['phase_config'] == phase_ref and report['scenario_id'] == phase['scenario_id']
                and report['checkpoint_provenance'] == result['checkpoint'] == training.FINAL_PPO
                and report['n_shifts'] == len(phase['jobs']) == PER_CASE and report['training_state_unchanged']
                and report['evaluation_optimizer_updates'] == 0, 'historical inference phase seal differs')
        for index, (s01, job, episode) in enumerate(zip(src['jobs'], phase['jobs'], report['per_episode'], strict=True)):
            ident = episode['identity']
            require(job['pair_ordinal'] == index and all(s01[key] == job[key] for key in PAIR_KEYS)
                    and (ident['run_index'], ident['environment_seed'], ident['actor_sampling_seed'],
                         ident['scene_id'], ident['recording_id']) == (job['run_index'], job['experiment_seed'],
                         job['actor_sampling_seed'], job['scene_id'], job['recording_id']),
                    'Full S02 physical slot or common evaluation seeds differ')
    return suite, evals, full


def verify_physical_starts():
    """Check authoritative historical START RPC for every paired physical shift."""
    _, phases, _ = historical()
    for phase in phases:
        for job in phase['jobs']:
            directory = Path(job['python_output']) / 'episodes' / f"run_{job['run_index']}"
            receipt = json.loads((directory / 'material_receipt.json').read_bytes())
            raw = load_reference(receipt['references']['raw_manifest'])
            source = reference(directory / 'episodes.json')
            require(raw['files_sha256']['episodes.json'] == source['sha256'] and receipt['normal_completion']
                    and receipt['run_index'] == job['run_index'] and receipt['four_windows'] == 4,
                    'Full S02 physical manifest or shift completion differs')
            start = load_reference(source)[0]['start_info']
            validate_runtime(start, job)
    return COUNT


def documents():
    source, templates, full = historical()
    finished = load_reference(COMPLETION)
    profile = finished['final_actor']
    actor = reference(profile['path'])
    ckpt = finished['final_checkpoint']
    require(finished['status'] == 'COMPLETE' and finished['training_shifts'] == 1024
            and finished['training_updates'] == profile['update_index'] == 32
            and finished['suite'] == reference(training.SUITE) and actor['sha256'] == profile['sha256']
            and reference(ckpt['path']) == ckpt and finished['protected_ppo_sha256'] == training.PROTECTED_CORE['sha256']
            and load_reference(finished['report']) == {k: v for k, v in finished.items() if k != 'report'},
            'REINFORCE training completion or u32 seal differs')
    fixed = {key: source[key] for key in ('candidate_id', 'stage', 'execution_profile', 'actor_seed',
             'learner_config', 'cpu_threads', 'sampling', 'evaluation_cases', 'episodes_per_batch')}
    fixed.update(schema='bg128-full-s02-reinforce-u32-fixed-paired-v1', campaign_id=CAMPAIGN,
                 replica='s02', baseline_seed='s02', training_completion=COMPLETION, frozen_actor=actor,
                 frozen_checkpoint=ckpt, baseline_completion=training.COMPLETION, baseline_suite=full['suite'],
                 baseline_phase_results=full['phase_results'], protected_core=training.PROTECTED_CORE,
                 arms=[dict(source['arms'][0], python_output=str(OUTPUT / 'evaluation'))],
                 report_directory=str(REPORT), python_output=str(OUTPUT), material_root=str(MATERIAL),
                 control_directory=str(CONTROL), budget_path=str(LEDGER), suite_path=str(SUITE),
                 rpc_port=RPC_PORT, training_shifts=0, training_updates=0, evaluation_shifts=COUNT,
                 evaluation_optimizer_updates=0, new_shift_cap=COUNT)
    phases = []
    for case, template in enumerate(templates):
        phase_id = template['phase_id']
        recording = f'{CAMPAIGN}-{phase_id}'
        material = MATERIAL / phase_id
        events = CONTROL / 'legacy_unused' / phase_id
        out = OUTPUT / 'evaluation' / template['scenario_id'] / 'new'
        jobs = []
        for i, src in enumerate(template['jobs']):
            ordinal = case * PER_CASE + i
            run = FIRST_RUN + ordinal
            group = f"{CAMPAIGN}-{src['scene_id']}-{i:04d}"
            leaf = f"dispatch-production-wait-shortfall-v2-run-{run}-seed-{src['experiment_seed']}"
            jobs.append(dict(src, ordinal=ordinal, run_index=run, scene_group=group, realization_group_id=group,
                             recording_id=recording, source_recording_id=recording,
                             python_output=str(out), event_directory=str(events), event_bundle=str(events / leaf),
                             material_event_directory=str(material), material_event_bundle=str(material / leaf),
                             material_journal=str(material / leaf / 'material_events.bin')))
        cfg = dict(template, jobs=jobs, campaign_id=CAMPAIGN, recording_id=recording,
                   python_output=str(out), event_directory=str(events), execution=dict(template['execution'], port=RPC_PORT))
        original.check_phase(cfg)
        phases.append(cfg)
    return fixed, phases


def source_pins():
    pins = training.source_pins()
    local = ['prepare.py', 'budget.py', 'run.py', 'runtime.py', 'paired_metrics.py', 'prepare_java.py']
    for filename in local:
        path = PLAN / filename
        pins[str(path)] = reference(path)['sha256']
    return dict(sorted(pins.items()))


# Same phase, source and Java sidecar validation as the frozen training package.
load_package = bind(training.load_package, documents=documents, SUITE=SUITE, PREPARED=PREPARED, source_pins=source_pins)


def prepare():
    from . import budget
    from .prepare_java import render
    require(not any(path.exists() for path in (OUTPUT, MATERIAL, PREPARED)), 'inference requires fresh output/package paths')
    fixed, phases = documents()
    starts = verify_physical_starts()
    jobs = [job for phase in phases for job in phase['jobs']]
    check = budget.assert_available(dict(fixed, jobs=jobs), budget.preceding_snapshot())
    pins = source_pins()
    refs = []
    for phase in phases:
        path = PREPARED / 'configs' / (phase['phase_id'] + '.json')
        write(path, phase)
        refs.append(reference(path))
    master, identities = PREPARED / 'master_plan.csv', PREPARED / 'identities.csv'
    original._write_rows(master, original.COLUMNS, jobs)
    original._write_rows(identities, original.IDENTITY_COLUMNS, original._identity_rows(phases, refs))
    helper, java_check = render(fixed, jobs, master, PREPARED)
    entries = [original._phase_entry(cfg, ref) for cfg, ref in zip(phases, refs, strict=True)]
    stored = dict(fixed, phase_configs=refs, phases=entries, source_pins=pins, master_plan=reference(master),
                  scenario_table=reference(PREPARED / 'scenarios.csv'), java_helper=reference(helper),
                  identities=reference(identities), java_validation=java_check)
    write(SUITE, stored)
    load_package()
    identity_rows = list(original._identity_rows(phases, refs))
    launch = dict(campaign_id=CAMPAIGN, suite=reference(SUITE), training_shifts=0, evaluation_shifts=COUNT,
                  frozen_actor=fixed['frozen_actor'], checkpoint=fixed['frozen_checkpoint'],
                  historical_start_rpcs_verified=starts, python_output=str(OUTPUT), anylogic_event_root=str(MATERIAL),
                  rpc_port=RPC_PORT, first=identity_rows[0], last=identity_rows[-1], identities=reference(identities),
                  master_plan=reference(master), java_helper=reference(helper), budget=check,
                  phases=[dict(entry, recording_id=phase['recording_id'], first_run=phase['jobs'][0]['run_index'],
                               last_run=phase['jobs'][-1]['run_index'],
                               material_event_directory=phase['jobs'][0]['material_event_directory'])
                          for entry, phase in zip(entries, phases, strict=True)])
    write(PREPARED / 'launch_identity.json', launch)
    return launch
