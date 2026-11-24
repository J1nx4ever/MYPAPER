"""Seal two localized GAT changes against the original mixed s01 schedule."""

import csv
from hashlib import sha256
from pathlib import Path
import sys
from types import FunctionType

PLAN = Path(__file__).resolve().parent
ROOT = PLAN.parents[3]
STRONG_PLAN = PLAN.with_name('structural_ablation_001')
for directory in (ROOT, STRONG_PLAN):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import ablation_plan as previous
from ablation_plan import CAP, EVAL, RPC_PORT, SCENARIOS, TRAIN, historical_sources, parent, replica
from business_graph_summary_campaign_v1.plan_io import encoded, load_reference, reference

VARIANTS = {'nodump': 'block_material_dump_to_load_l2', 'gat1': 'one_gat_node_mlp'}
UNITS = tuple(f'{group}-s01' for group in VARIANTS)
PREPARED = PLAN / 'prepared_001'
REPORT = previous.REPORT.with_name(PLAN.name)
ENTRY = PLAN / 'weak_run.py'
FIRST_RUN = 10027001
ALP_RUNS = CAP * len(UNITS)
GRADE_ARCHIVE = dict(path=str(previous.REPORT / 'mechanism_grade_response.json'),
                     sha256='0880b6a8a47a21145f99505817a66f31aaa8a6c425bcef0daa06724b75f83524')
GOLDEN_EVAL_SHA = '82942133ea60c29ee0bb524d4eb936eb76941070b1abc4d36b680c289c842d92'
EVAL_FIELDS = ('scene_id', 'pair_ordinal', 'experiment_seed', 'actor_sampling_seed',
               'runtime_scene_sha256', 'expected_rpc_config')


def bind(function, **overrides):
    """Bind the original workflow to this campaign in a private namespace."""
    result = FunctionType(function.__code__, dict(function.__globals__, **overrides),
                          function.__name__, function.__defaults__, function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    return result


def locations(unit):
    campaign = f'bg128-gat-weak-20261122-{unit}'
    out = ROOT / 'runs' / campaign
    return dict(campaign_id=campaign, python_output=str(out), control_directory=str(out / 'control'),
                budget_path=str(out / 'budget.sqlite'), material_root=str(out.with_name(campaign + '_materials')),
                report_directory=str(REPORT / unit), suite_path=str(PREPARED / unit / 'suite.json'),
                run_index_base=FIRST_RUN + UNITS.index(unit) * 2000, rpc_port=RPC_PORT)


def documents(unit, sources):
    from business_graph_weak_ablation_v1.models import architecture_contract

    build = bind(previous.documents, VARIANTS=VARIANTS, locations=locations,
                 architecture_contract=architecture_contract)
    suite, phases = build(unit, sources)
    suite.update(schema='bg128-gat-weak-ablation-v1', grade_archive=GRADE_ARCHIVE,
                 fixed_evaluation_slot_sha256=GOLDEN_EVAL_SHA)
    parent.require(eval_sha(phases) == GOLDEN_EVAL_SHA, 'fixed 700-slot evaluation panel differs')
    return suite, phases


def eval_sha(phases):
    panel = [{field: job[field] for field in EVAL_FIELDS}
             for phase in phases if phase['policy_arm'] == 'new' for job in phase['jobs']]
    parent.require(len(panel) == EVAL, 'fixed evaluation slot inventory changed')
    return sha256(encoded(panel)).hexdigest()


def _pins(sources):
    from weak_runtime import source_files

    baseline = PLAN.with_name('baseline_comparison_001')
    budget_files = list(baseline.rglob('*budget.py'))
    files = {*PLAN.glob('*.py'), *source_files(), *budget_files}
    return dict(previous._pins(sources), **{str(path): parent.sha(path) for path in files
                                           if not path.name.startswith('test_')})


def load_package(unit, sources=None):
    load = bind(previous.load_package, documents=documents, _pins=_pins)
    suite, phases = load(unit, sources)
    with Path(suite['master_plan']['path']).open(encoding='utf-8', newline='') as stream:
        expected = [{key: str(job[key]) for key in parent.COLUMNS} for job in suite['jobs']]
        parent.require(list(csv.DictReader(stream)) == expected, 'Java master physical identities differ')
    return suite, phases


def prepare():
    from weak_budget import assert_available, budget_snapshot

    parent.require(not PREPARED.exists(), 'preparation must not overwrite a frozen package')
    archive = load_reference(GRADE_ARCHIVE)
    parent.require(archive['analyzed_decisions'] == 400 and len(archive['sampled_episodes']) == 50,
                   'frozen grade replay inventory differs')
    sources = historical_sources()
    work = [documents(unit, sources) for unit in UNITS]
    batches = [dict(suite, jobs=[job for phase in phases for job in phase['jobs']]) for suite, phases in work]
    for suite in batches:
        parent.require(not Path(suite['python_output']).exists() and not Path(suite['material_root']).exists(),
                       'new weak ablation run paths already contain evidence')
    budget = assert_available(batches, budget_snapshot())
    pins = _pins(sources)
    render = bind(previous._render_helper, ENTRY=ENTRY)
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
            baseline_seed=settings['baseline_seed'], suite=reference(settings['suite_path']),
            physical_shifts=CAP, python_output=settings['python_output'],
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
        stream.write(previous._combined_source(identities))
    previous._verify_combined(helper, identities)
    launch = PREPARED / 'launch_identity.json'
    parent.write(launch, dict(order=list(UNITS), physical_shifts=ALP_RUNS, training_shifts=TRAIN * len(UNITS),
                             evaluation_shifts=EVAL * len(UNITS), rpc_port=RPC_PORT, java_helper=reference(helper),
                             launch_identities=manifests, fixed_evaluation_slot_sha256=GOLDEN_EVAL_SHA))
    parent.require(parent.sha(parent.CORE) == parent.PROTECTED_SHA, 'protected PPO core SHA changed')
    result = dict(status='PASS_OFFLINE_NO_LISTENER', launch_identity=reference(launch),
                  java_indices_checked=ALP_RUNS, fixed_evaluation_slots_per_unit=EVAL,
                  fixed_evaluation_slot_sha256=GOLDEN_EVAL_SHA, grade_archive=GRADE_ARCHIVE,
                  budgets=budget, protected_core=reference(parent.CORE),
                  simulation_started=False, budget_registered=False)
    parent.write(REPORT / 'configuration_verification.json', result)
    return result
