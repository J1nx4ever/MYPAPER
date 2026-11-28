"""Freeze four paired BG128 depth-study runs on the original evaluation panel."""

import csv
from functools import lru_cache
from pathlib import Path
import sys

PLAN = Path(__file__).resolve().parent
sys.path.insert(0, str(PLAN.parent / 'seed_replication_001'))

import replication_plan as previous
import weak_plan as weak
from weak_plan import CAP, EVAL, GOLDEN_EVAL_SHA, RPC_PORT, TRAIN, bind, parent, replica
from weak_runtime import source_files
from business_graph_summary_campaign_v1.plan_io import load_reference, reference
from business_graph_summary_v1.contract import ablation_contract as full_graph_contract
from business_graph_weak_ablation_v1.models import architecture_contract


UNITS = ('full-s04', 'gat1-s04', 'full-s05', 'gat1-s05')
VARIANTS = {'full': 'full_business_graph', 'gat1': 'one_gat_node_mlp'}
PREPARED = PLAN / 'prepared_001'
REPORT = weak.REPORT / PLAN.name
ENTRY = PLAN / 'extension_run.py'
FIRST_RUN = 10046001
ALP_RUNS = len(UNITS) * CAP
historical_sources = previous.historical_sources


def locations(unit):
    campaign = f'bg128-gat-depth-20261125-{unit}'
    output = previous.ROOT / 'runs' / campaign
    return dict(campaign_id=campaign, python_output=str(output),
                control_directory=str(output / 'control'), budget_path=str(output / 'budget.sqlite'),
                material_root=str(output.with_name(campaign + '_materials')),
                report_directory=str(REPORT / unit), suite_path=str(PREPARED / unit / 'suite.json'),
                run_index_base=FIRST_RUN + UNITS.index(unit) * CAP, rpc_port=RPC_PORT)


@lru_cache(maxsize=2)
def training_template(seed):
    parent.require(seed in ('s04', 's05'), 'unknown extension training schedule')
    _, phases = replica.load_package(seed)
    parent.require(weak.eval_sha(phases) == GOLDEN_EVAL_SHA, 'archived replica panel differs')
    return phases[:parent.UPDATES]


def _architecture(variant):
    if variant == VARIANTS['full']:
        return full_graph_contract()
    return architecture_contract(variant)


def documents(unit, sources):
    seed = unit.rsplit('-', 1)[1]
    anchor = sources['s01']
    controls = dict(sources)
    controls[seed] = dict(anchor, training=training_template(seed), completion=None,
                          checkpoint=None, reports={})
    build = bind(weak.previous.documents, VARIANTS=VARIANTS, locations=locations,
                 architecture_contract=_architecture)
    settings, phases = build(unit, controls)
    settings.update(schema='bg128-gat-depth-seed-extension-v1',
                    training_schedule=reference(replica.locations(seed)['suite_path']),
                    fixed_evaluation_slot_sha256=GOLDEN_EVAL_SHA)
    if unit.startswith('gat1-'):
        settings['matched_full_suite_path'] = locations(f'full-{seed}')['suite_path']
    parent.require(weak.eval_sha(phases) == GOLDEN_EVAL_SHA, 'fixed evaluation panel differs')
    return settings, phases


def _pins(sources):
    extras = [*PLAN.glob('*.py'), *source_files(), *previous.PLAN.glob('*.py'),
              weak.PLAN / 'weak_plan.py', weak.PLAN / 'weak_budget.py']
    return dict(previous._pins(sources), **{str(path): parent.sha(path) for path in extras})


def load_package(unit, sources=None):
    load = bind(weak.previous.load_package, documents=documents, _pins=_pins,
                historical_sources=historical_sources)
    suite, phases = load(unit, sources)
    with Path(suite['master_plan']['path']).open(encoding='utf-8', newline='') as stream:
        expected = [{key: str(job[key]) for key in parent.COLUMNS} for job in suite['jobs']]
        parent.require(list(csv.DictReader(stream)) == expected, 'Java master physical identities differ')
    return suite, phases


def _freeze_unit(settings, phases, suite, pins):
    directory = Path(settings['suite_path']).parent
    refs = []
    for phase in phases:
        config = directory / 'configs' / (phase['phase_id'] + '.json')
        parent.write(config, phase)
        refs.append(reference(config))
    jobs = suite['jobs']
    master, identities = (directory / name for name in ('master_plan.csv', 'identities.csv'))
    parent._write_rows(master, parent.COLUMNS, jobs)
    render = bind(weak.previous._render_helper, ENTRY=ENTRY)
    helper = render(settings, jobs, master, directory)
    java = replica._java_check(directory, RPC_PORT)
    parent._write_rows(identities, parent.IDENTITY_COLUMNS, parent._identity_rows(phases, refs))
    stored = dict(settings, phase_configs=refs, source_pins=pins, master_plan=reference(master),
                  scenario_table=reference(directory / 'scenarios.csv'), java_helper=reference(helper),
                  identities=reference(identities), java_validation=java)
    parent.write(settings['suite_path'], stored)
    path = directory / 'launch_identity.json'
    parent.write(path, dict(
        unit=settings['unit'], ablation=settings['ablation'], baseline_seed=settings['baseline_seed'],
        training_schedule=settings['training_schedule'], suite=reference(settings['suite_path']),
        physical_shifts=CAP, python_output=settings['python_output'],
        anylogic_event_root=settings['material_root'], first=next(parent._identity_rows(phases, refs)),
        identities=stored['identities'], java_helper=stored['java_helper'],
        phases=[dict(parent._phase_entry(phase, ref), recording_id=phase['recording_id'],
                     material_event_directory=phase['jobs'][0]['material_event_directory'],
                     first_run=phase['jobs'][0]['run_index'], last_run=phase['jobs'][-1]['run_index'])
                for phase, ref in zip(phases, refs, strict=True)]))
    return reference(path)


def prepare():
    from extension_budget import assert_available, budget_snapshot

    parent.require(not PREPARED.exists(), 'preparation must not overwrite an existing package')
    parent.require(parent.sha(parent.CORE) == parent.PROTECTED_SHA, 'protected PPO core SHA changed')
    sources = historical_sources()
    work = [documents(unit, sources) for unit in UNITS]
    batches = [dict(settings, jobs=[job for phase in phases for job in phase['jobs']])
               for settings, phases in work]
    for suite in batches:
        parent.require(not Path(suite['python_output']).exists() and not Path(suite['material_root']).exists(),
                       'new campaign already has output or material evidence')
    budget = assert_available(batches, budget_snapshot())
    pins = _pins(sources)
    manifests = []
    for (settings, phases), suite in zip(work, batches, strict=True):
        manifests.append(_freeze_unit(settings, phases, suite, pins))
        load_package(settings['unit'], sources)
    identities = [load_reference(ref) for ref in manifests]
    helper = PREPARED / 'Baseline1CampaignPlan.java'
    combined = bind(weak.previous._combined_source, ALP_RUNS=ALP_RUNS)
    with helper.open('x', encoding='utf-8') as stream:
        stream.write(combined(identities))
    verify = bind(weak.previous._verify_combined, ALP_RUNS=ALP_RUNS)
    verify(helper, identities)
    launch = PREPARED / 'launch_identity.json'
    parent.write(launch, dict(order=list(UNITS), physical_shifts=ALP_RUNS,
                             training_shifts=TRAIN * len(UNITS), evaluation_shifts=EVAL * len(UNITS),
                             rpc_port=RPC_PORT, java_helper=reference(helper), launch_identities=manifests,
                             fixed_evaluation_slot_sha256=GOLDEN_EVAL_SHA))
    parent.require(parent.sha(parent.CORE) == parent.PROTECTED_SHA, 'protected PPO core SHA changed')
    result = dict(status='PASS_OFFLINE_NO_LISTENER', launch_identity=reference(launch),
                  java_indices_checked=ALP_RUNS, matched_training_scenes_per_pair=TRAIN,
                  fixed_evaluation_slots_per_unit=EVAL, fixed_evaluation_slot_sha256=GOLDEN_EVAL_SHA,
                  budgets=budget, protected_core=reference(parent.CORE),
                  simulation_started=False, budget_registered=False)
    parent.write(REPORT / 'configuration_verification.json', result)
    return result
