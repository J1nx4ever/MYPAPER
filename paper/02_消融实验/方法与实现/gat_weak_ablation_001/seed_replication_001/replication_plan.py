"""Freeze the GAT1 s02 and s03 repeats against their completed Full campaigns."""

import csv
from functools import lru_cache
from pathlib import Path
import sys

PLAN = Path(__file__).resolve().parent
WEAK_PLAN = PLAN.parent
ROOT = PLAN.parents[4]
for directory in (ROOT, WEAK_PLAN):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import weak_plan as weak
from weak_plan import CAP, EVAL, GOLDEN_EVAL_SHA, RPC_PORT, SCENARIOS, TRAIN, bind, parent, replica
from weak_runtime import source_files
from business_graph_summary_campaign_v1.plan_io import load_reference, reference
from business_graph_weak_ablation_v1.models import architecture_contract

UNITS = ('gat1-s02', 'gat1-s03')
VARIANTS = {'gat1': 'one_gat_node_mlp'}
PREPARED = PLAN / 'prepared_001'
REPORT = weak.REPORT / PLAN.name
ENTRY = PLAN / 'replication_run.py'
FIRST_RUN = 10038001
ALP_RUNS = CAP * len(UNITS)
GAT1_S01_COMPLETE = dict(path=str(ROOT / 'runs/bg128-gat-weak-20261122-gat1-s01/suite_complete.json'),
                        sha256='27e5629ac6395dc7e011c3d1be421543772ec6af5e972eb5658eca8dfd951a4d')
FULL_COMPLETION_SHA = {
    's02': '45d7735b6612b55404bdf240f2c7885667f5e80ba07ffaea07aa38eefdb808fc',
    's03': '799afbff30c8ec2ba52d8b54747b9bed0fae78dac21ea4ad8bd8063b962d81ee',
}
FULL_CHECKPOINT_SHA = {
    's02': '67b7b2c7104079836ccef8f5747e7aebb4e9e5a92dcaac67c94e42412f104f41',
    's03': 'f0844556cfb277f3292319bc66a8c3f24fa3d581110269d2f4c97e0c1b80281d',
}


def locations(unit):
    campaign = f'bg128-gat1-replication-20261123-{unit.rsplit("-", 1)[1]}'
    output = ROOT / 'runs' / campaign
    return dict(campaign_id=campaign, python_output=str(output), control_directory=str(output / 'control'),
                budget_path=str(output / 'budget.sqlite'), material_root=str(output.with_name(campaign + '_materials')),
                report_directory=str(REPORT / unit), suite_path=str(PREPARED / unit / 'suite.json'),
                run_index_base=FIRST_RUN + UNITS.index(unit) * 2000, rpc_port=RPC_PORT)


@lru_cache(maxsize=1)
def historical_sources():
    sources = dict(weak.historical_sources())
    for seed, completion_sha in FULL_COMPLETION_SHA.items():
        original, phases = replica.load_package(seed)
        completion_ref = dict(path=str(Path(original['python_output']) / 'suite_complete.json'),
                              sha256=completion_sha)
        complete = load_reference(completion_ref)
        checkpoint = complete['new_checkpoint']
        parent.require(complete['status'] == 'COMPLETE' and complete['suite'] == reference(original['suite_path'])
                       and complete['training_shifts'] == TRAIN and complete['training_updates'] == parent.UPDATES
                       and complete['evaluation_shifts'] == EVAL and complete['evaluation_optimizer_updates'] == 0
                       and checkpoint['sha256'] == FULL_CHECKPOINT_SHA[seed]
                       and reference(checkpoint['path']) == checkpoint, 'Full training completion or checkpoint differs')
        training, evaluation = phases[:parent.UPDATES], phases[parent.UPDATES:]
        parent.require(tuple(phase['scenario_id'] for phase in evaluation) == SCENARIOS
                       and weak.eval_sha(phases) == GOLDEN_EVAL_SHA, 'Full evaluation differs from the frozen panel')
        reports = {}
        for phase, item in zip(evaluation, complete['phase_results'], strict=True):
            report_ref = item['report']
            report = load_reference(report_ref)
            parent.require(report['scenario_id'] == phase['scenario_id']
                           and report['phase_config'] == original['phase_configs'][parent.UPDATES + len(reports)]
                           and report['checkpoint_provenance'] == checkpoint == item['checkpoint']
                           and report['n_shifts'] == parent.EVAL_COUNT
                           and report['training_state_unchanged'] and report['evaluation_optimizer_updates'] == 0,
                           'Full fixed evaluation report differs')
            for job, row in zip(phase['jobs'], report['per_episode'], strict=True):
                identity = row['identity']
                parent.require((identity['run_index'], identity['environment_seed'], identity['actor_sampling_seed'])
                               == (job['run_index'], job['experiment_seed'], job['actor_sampling_seed']),
                               'Full per-episode identity differs')
            reports[phase['scenario_id']] = report_ref
        sources[seed] = dict(suite=original, completion=completion_ref, training=training, evaluation=evaluation,
                             checkpoint=checkpoint, reports=reports)
    return sources


def documents(unit, sources):
    build = bind(weak.previous.documents, VARIANTS=VARIANTS, locations=locations,
                 architecture_contract=architecture_contract)
    settings, phases = build(unit, sources)
    settings.update(schema='bg128-gat1-seed-replication-v1',
                    fixed_evaluation_slot_sha256=GOLDEN_EVAL_SHA, gat1_s01_completion=GAT1_S01_COMPLETE)
    parent.require(weak.eval_sha(phases) == GOLDEN_EVAL_SHA, 'fixed 700-slot evaluation panel differs')
    return settings, phases


def _pins(sources):
    predecessor = WEAK_PLAN.with_name('baseline_comparison_001') / 'dispatch_flow_matching_001'
    extras = (*PLAN.glob('*.py'), *source_files(), WEAK_PLAN / 'weak_plan.py', WEAK_PLAN / 'weak_budget.py',
              predecessor / 'flow_matching_campaign.py', predecessor / 'flow_matching_plan.py')
    return dict(weak.previous._pins(sources), **{str(path): parent.sha(path) for path in extras})


def load_package(unit, sources=None):
    load = bind(weak.previous.load_package, documents=documents, _pins=_pins, historical_sources=historical_sources)
    suite, phases = load(unit, sources)
    with Path(suite['master_plan']['path']).open(encoding='utf-8', newline='') as stream:
        expected = [{key: str(job[key]) for key in parent.COLUMNS} for job in suite['jobs']]
        parent.require(list(csv.DictReader(stream)) == expected, 'Java master physical identities differ')
    return suite, phases


def prepare():
    from replication_budget import assert_available, budget_snapshot

    parent.require(not PREPARED.exists(), 'preparation must not overwrite an existing package')
    s01 = load_reference(GAT1_S01_COMPLETE)
    parent.require(s01['status'] == 'COMPLETE' and s01['training_updates'] == parent.UPDATES
                   and s01['training_shifts'] == TRAIN and s01['evaluation_shifts'] == EVAL
                   and s01['evaluation_optimizer_updates'] == 0, 'archived GAT1 s01 is incomplete')
    load_reference(s01['comparison'])
    sources = historical_sources()
    work = [documents(unit, sources) for unit in UNITS]
    batches = [dict(suite, jobs=[job for phase in phases for job in phase['jobs']]) for suite, phases in work]
    for suite in batches:
        parent.require(not Path(suite['python_output']).exists() and not Path(suite['material_root']).exists(),
                       'new GAT1 run paths already contain evidence')
    budget = assert_available(batches, budget_snapshot())
    pins = _pins(sources)
    render = bind(weak.previous._render_helper, ENTRY=ENTRY)
    manifests = []
    for (settings, phases), suite in zip(work, batches, strict=True):
        directory = Path(settings['suite_path']).parent
        refs = []
        for phase in phases:
            config = directory / 'configs' / (phase['phase_id'] + '.json')
            parent.write(config, phase)
            refs.append(reference(config))
        jobs = suite['jobs']
        master, identities = (directory / file for file in ('master_plan.csv', 'identities.csv'))
        parent._write_rows(master, parent.COLUMNS, jobs)
        helper = render(settings, jobs, master, directory)
        java = replica._java_check(directory, RPC_PORT)
        parent._write_rows(identities, parent.IDENTITY_COLUMNS, parent._identity_rows(phases, refs))
        stored = dict(settings, phase_configs=refs, source_pins=pins, master_plan=reference(master),
                      scenario_table=reference(directory / 'scenarios.csv'), java_helper=reference(helper),
                      identities=reference(identities), java_validation=java)
        parent.write(settings['suite_path'], stored)
        load_package(settings['unit'], sources)
        path = directory / 'launch_identity.json'
        parent.write(path, dict(unit=settings['unit'], ablation=settings['ablation'],
            baseline_seed=settings['baseline_seed'], baseline_completion=settings['baseline_completion'],
            suite=reference(settings['suite_path']), physical_shifts=CAP, python_output=settings['python_output'],
            anylogic_event_root=settings['material_root'], first=next(parent._identity_rows(phases, refs)),
            identities=stored['identities'], java_helper=stored['java_helper'],
            phases=[dict(parent._phase_entry(phase, ref), recording_id=phase['recording_id'],
                         material_event_directory=phase['jobs'][0]['material_event_directory'],
                         first_run=phase['jobs'][0]['run_index'], last_run=phase['jobs'][-1]['run_index'])
                    for phase, ref in zip(phases, refs, strict=True)]))
        manifests.append(reference(path))
        print('PREPARED', settings['unit'], jobs[0]['run_index'], flush=True)
    identities = [load_reference(ref) for ref in manifests]
    helper = PREPARED / 'Baseline1CampaignPlan.java'
    with helper.open('x', encoding='utf-8') as stream:
        stream.write(weak.previous._combined_source(identities))
    weak.previous._verify_combined(helper, identities)
    launch = PREPARED / 'launch_identity.json'
    parent.write(launch, dict(order=list(UNITS), physical_shifts=ALP_RUNS, training_shifts=TRAIN * len(UNITS),
                             evaluation_shifts=EVAL * len(UNITS), rpc_port=RPC_PORT, java_helper=reference(helper),
                             launch_identities=manifests, fixed_evaluation_slot_sha256=GOLDEN_EVAL_SHA))
    parent.require(parent.sha(parent.CORE) == parent.PROTECTED_SHA, 'protected PPO core SHA changed')
    result = dict(status='PASS_OFFLINE_NO_LISTENER', launch_identity=reference(launch),
                  java_indices_checked=ALP_RUNS, matched_training_scenes_per_unit=TRAIN,
                  fixed_evaluation_slots_per_unit=EVAL, fixed_evaluation_slot_sha256=GOLDEN_EVAL_SHA,
                  sources={key: sources[key]['completion'] for key in FULL_COMPLETION_SHA},
                  original_gat1_s01=GAT1_S01_COMPLETE, budgets=budget, protected_core=reference(parent.CORE),
                  simulation_started=False, budget_registered=False)
    parent.write(REPORT / 'configuration_verification.json', result)
    return result
