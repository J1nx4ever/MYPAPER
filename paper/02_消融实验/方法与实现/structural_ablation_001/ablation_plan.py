"""Freeze paired mixed-training schedules and fixed-u32 evaluation slots."""

import csv
from functools import lru_cache
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from textwrap import indent

PLAN = Path(__file__).resolve().parent
ROOT = PLAN.parents[3]
REPLICATION = PLAN.parent / 'generalization_001/grade_assignment_generalization_001/mixed_seed_replication_001'
for directory in (ROOT, REPLICATION):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import replica_plan as replica
from replica_plan import CAP, EVAL, TRAIN, parent
from mixed_java import _replace_once, scenario_table_bytes, write_java
from oneclick_java import INDEX_METHODS, RUN_METHODS
from business_graph_ablation_v1.models import architecture_contract
from business_graph_summary_campaign_v1.plan_io import load_reference, reference

VARIANTS = {'flat': 'flat_mlp', 'noc128': 'graph_no_c128'}
UNITS = tuple(f'{group}-s01' for group in VARIANTS)
PREPARED = PLAN / 'prepared_001'
REPORT = ROOT / 'reports/token_attention_001/business_graph_summary_009/structural_ablation_001'
ENTRY = PLAN / 'ablation_run.py'
FIRST_RUN = 10014001
RPC_PORT = 50051
ALP_RUNS = CAP * len(UNITS)
CHECKPOINT_UPDATE = parent.UPDATES
SCENARIOS = tuple(parent.EVAL_CASES)


def locations(unit):
    campaign = f'bg128-ablation-20261121-{unit}'
    out = ROOT / 'runs' / campaign
    return dict(campaign_id=campaign, python_output=str(out), control_directory=str(out / 'control'),
                budget_path=str(out / 'budget.sqlite'), material_root=str(out.with_name(campaign + '_materials')),
                report_directory=str(REPORT / unit), suite_path=str(PREPARED / unit / 'suite.json'),
                run_index_base=FIRST_RUN + UNITS.index(unit) * 2000, rpc_port=RPC_PORT)


@lru_cache(maxsize=1)
def historical_sources():
    source, train, evaluate, _, _ = replica.anchored_source()
    ref = dict(path=str(replica.ANCHOR), sha256=replica.ANCHOR_SHA)
    completion = load_reference(ref)
    checkpoint = completion['new_checkpoint']
    parent.require(completion['training_shifts'] == TRAIN and completion['training_updates'] == CHECKPOINT_UPDATE
                   and completion['evaluation_optimizer_updates'] == 0
                   and reference(checkpoint['path']) == checkpoint, 'Full u32 completion or bytes differ')
    configs = parent.load_suite(completion['campaign']['path'])[1]
    reports = {}
    for cfg, result in zip(configs[CHECKPOINT_UPDATE:], completion['phase_results'], strict=True):
        if cfg['policy_arm'] != 'new':
            continue
        scenario = cfg['scenario_id']
        report = load_reference(result['report'])
        parent.require(report['scenario_id'] == scenario and report['checkpoint_provenance'] == checkpoint,
                       'Full evaluation u32 provenance differs')
        reports[scenario] = result['report']
    parent.require(tuple(reports) == SCENARIOS, 'Full evaluation scenarios differ from original order')
    return {'s01': dict(suite=source, completion=ref, training=train, evaluation=evaluate,
                        reports=reports, checkpoint=checkpoint)}


def documents(unit, sources):
    group, baseline_seed = unit.split('-')
    variant = VARIANTS[group]
    control = sources[baseline_seed]
    base = control['suite']
    layout = locations(unit)
    campaign = layout['campaign_id']
    init = parent.seed(campaign, 'initialization', 1)
    learner = dict(base['learner_config'], seed=init)
    output, controls, materials = (Path(layout[key]) for key in ('python_output', 'control_directory', 'material_root'))
    arm = dict(base['arms'][0], learner_config=learner, actor_seed=init,
               python_output=str(output / 'arms' / parent.CANDIDATE))
    settings = dict(unit=unit, ablation_variant=variant, ablation=architecture_contract(variant),
                    baseline_seed=baseline_seed, baseline_completion=control['completion'],
                    baseline_checkpoint=control['checkpoint'], baseline_reports=control['reports'],
                    baseline_suite=reference(base['suite_path']),
                    source_completion=sources['s01']['completion'], schema='bg128-structural-ablation-v1',
                    **layout)
    for key in ('candidate_id', 'arm_order', 'stage', 'execution_profile', 'training_updates', 'first_update',
                'final_update', 'episodes_per_batch', 'training_seed_count', 'cpu_threads', 'sampling',
                'pair_id', 'paired_slots', 'active_h2_reserved_shifts', 'legacy_reserved_shifts', 'mean_ranges',
                'half_width_range', 'heldout_orders', 'evaluation_cases', 'protected_core'):
        settings[key] = base[key]
    settings.update(arms=[arm], learner_config=learner, actor_seed=init, training_shifts=TRAIN,
                    evaluation_shifts=EVAL, new_shift_cap=CAP)
    phases = []
    ordinal = 0
    for template in [*control['training'], *control['evaluation']]:
        phase = template['phase_id']
        train = template['policy_arm'] == 'train'
        recording = f'{campaign}-{phase}'
        out = (Path(arm['python_output']) / template['arm_phase_id'] if train
               else output / 'evaluation' / template['scenario_id'] / 'new')
        events = controls / 'legacy_unused' / phase
        material = materials / phase
        jobs = []
        for old in template['jobs']:
            run = layout['run_index_base'] + ordinal
            env = parent.seed(campaign + '-train', 'environment', ordinal) if train else old['experiment_seed']
            action = parent.seed(campaign + '-train', 'sampling', ordinal) if train else old['actor_sampling_seed']
            group = f"{campaign}-{old['scene_id']}-{old['pair_ordinal']:04d}"
            leaf = f'dispatch-production-wait-shortfall-v2-run-{run}-seed-{env}'
            jobs.append(dict(old, ordinal=ordinal, run_index=run, experiment_seed=env, actor_sampling_seed=action,
                             actor_seed=init, recording_id=recording, source_recording_id=recording,
                             scene_group=group, realization_group_id=group, python_output=str(out),
                             event_directory=str(events), event_bundle=str(events / leaf),
                             material_event_directory=str(material), material_event_bundle=str(material / leaf),
                             material_journal=str(material / leaf / 'material_events.bin')))
            ordinal += 1
        cfg = dict(template, campaign_id=campaign, recording_id=recording, python_output=str(out),
                   event_directory=str(events), execution=dict(template['execution'], port=RPC_PORT), jobs=jobs)
        parent.check_phase(cfg)
        phases.append(cfg)
    parent.require(ordinal == CAP and len(phases) == CHECKPOINT_UPDATE + len(SCENARIOS),
                   'ablation phase inventory differs')
    return settings, phases


def _pins(sources):
    original = sources['s01']['suite']['source_pins']
    extras = [*PLAN.glob('*.py'), *(ROOT / 'business_graph_ablation_v1').glob('*.py'),
              *REPLICATION.glob('*.py'), replica.MIXED / 'recovery_003/transport.py',
              replica.MIXED / 'recovery_002/storage_scope.py']
    return dict(original, **{str(path): parent.sha(path) for path in extras})


def load_package(unit, sources=None):
    sources = historical_sources() if sources is None else sources
    settings, expected = documents(unit, sources)
    path = Path(settings['suite_path'])
    suite, phases = parent.load_suite(path)
    parent.require(all(suite.get(key) == value for key, value in settings.items())
                   and phases == expected and suite['source_pins'] == _pins(sources),
                   'frozen ablation suite or sources differ')
    for key, name in (('master_plan', 'master_plan.csv'), ('scenario_table', 'scenarios.csv'),
                      ('java_helper', 'Baseline1CampaignPlan.java'), ('identities', 'identities.csv')):
        parent.require(suite[key] == reference(path.with_name(name)), 'ablation artifact differs: ' + key)
    parent.require(path.with_name('scenarios.csv').read_bytes() == scenario_table_bytes(suite['jobs']),
                   'AnyLogic physical scenario table differs from RPC config')
    with path.with_name('identities.csv').open(newline='', encoding='utf-8') as stream:
        parent.require(list(csv.DictReader(stream)) == [{key: str(value) for key, value in row.items()}
                       for row in parent._identity_rows(phases, suite['phase_configs'])],
                       'ablation identity list differs from configuration')
    return suite, phases


def _render_helper(settings, jobs, master, directory):
    helper = write_java(settings, jobs, master, directory / 'scenarios.csv', directory, bridge_path=ENTRY)
    anchor = '    private static final class Row {'
    source = _replace_once(helper.read_text(encoding='utf-8'), anchor,
                           f'    public static int rpcPort() {{ return {RPC_PORT}; }}\n'
                           '    public static double freeSpeedFactorStd(int index) { runIndex(index); return 0.05; }\n'
                           + anchor)
    helper.write_text(source, encoding='utf-8')
    return helper


def _combined_source(identities):
    chunks = [f'''public final class Baseline1CampaignPlan {{
    private Baseline1CampaignPlan() {{ }}
    private static final int PER_UNIT = {CAP};
    public static int rpcPort() {{ return {RPC_PORT}; }}
    public static int jobCount() {{ return {ALP_RUNS}; }}
    public static int frozenPlanJobCount() {{ return jobCount(); }}
    public static int resumeOffset() {{ return 0; }}
    public static boolean preparationOnly() {{ return false; }}
    private static int forIndex(int index) {{
        if (index < 0 || index >= jobCount()) throw new IllegalStateException("index outside ablation plan");
        return index / PER_UNIT;
    }}
    private static int forRun(long run) {{
''']
    for index, item in enumerate(identities):
        first = item['first']['run_index']
        chunks.append(f'        if (run >= {first}L && run < {first + CAP}L) return {index};\n')
    chunks.append('        throw new IllegalStateException("run outside ablation plan");\n    }\n')
    methods = [(name, result, 'int index', 'forIndex(index)', 'index % PER_UNIT')
               for name, result in INDEX_METHODS.items()]
    methods += [(name, result, 'long run' + params, 'forRun(run)', 'run' + args)
                for name, (result, params, args) in RUN_METHODS.items()]
    for name, result, params, route, args in methods:
        chunks.append(f'    public static {result} {name}({params}) {{\n        switch ({route}) {{\n')
        for index in range(len(identities)):
            chunks.append(f'            case {index}: return Unit{index}.{name}({args});\n')
        chunks.append('            default: throw new IllegalStateException("invalid ablation route");\n        }\n    }\n')
    for index, item in enumerate(identities):
        artifact = item['java_helper']
        parent.require(reference(artifact['path']) == artifact, 'Java phase helper bytes changed')
        nested = Path(artifact['path']).read_text(encoding='utf-8')
        starter = 'public final class Baseline1CampaignPlan {'
        parent.require(nested.count(starter) == 1 and not any(line.startswith(('package ', 'import '))
                                                             for line in nested.splitlines()), 'unexpected Java helper')
        nested = nested.replace(starter, f'private static final class Unit{index} {{', 1)
        nested = nested.replace('Baseline1CampaignPlan', f'Unit{index}')
        chunks.append(indent(nested, '    '))
    return ''.join(chunks) + '}\n'


def _verify_combined(helper, identities):
    probe = f'''public final class AblationPlanProbe {{
    private static void check(boolean ok) {{
        if (!ok) throw new AssertionError("ablation route differs");
    }}
    public static void main(String[] paths) throws Exception {{
        check(Baseline1CampaignPlan.rpcPort() == {RPC_PORT} && Baseline1CampaignPlan.jobCount() == {ALP_RUNS});
        int index = 0;
        for (String path : paths) {{
            java.util.List<String> rows = java.nio.file.Files.readAllLines(java.nio.file.Path.of(path));
            check(rows.size() == {CAP + 1});
            for (String row : rows.subList(1, rows.size())) {{
                String[] fields = row.split(",", -1);
                long run = Baseline1CampaignPlan.runIndex(index);
                check(run == Long.parseLong(fields[6])
                    && Baseline1CampaignPlan.environmentSeed(index) == Long.parseLong(fields[7])
                    && Baseline1CampaignPlan.actorSamplingSeed(index) == Long.parseLong(fields[8])
                    && Baseline1CampaignPlan.recordingId(index).equals(fields[5])
                    && Baseline1CampaignPlan.eventRoot(index).equals(fields[9])
                    && Baseline1CampaignPlan.materialEnabled(index)
                    && Baseline1CampaignPlan.dispatchPolicy(index) == 3
                    && !Baseline1CampaignPlan.offlineEnabled(index)
                    && Baseline1CampaignPlan.freeSpeedFactorStd(index) == 0.05
                    && Baseline1CampaignPlan.scenarioId(run) != null);
                index++;
            }}
            try {{
                Baseline1CampaignPlan.awaitConfigSha256(
                    Baseline1CampaignPlan.runIndex(index - {CAP}), 0L, "wrong", "wrong", "wrong");
                throw new AssertionError("wrong launch identity accepted");
            }} catch (IllegalStateException expected) {{
                check(expected.getMessage().contains("Main parameters differ"));
            }}
        }}
        check(index == {ALP_RUNS});
        try {{
            Baseline1CampaignPlan.runIndex(index);
            throw new AssertionError("out-of-range index accepted");
        }} catch (IllegalStateException expected) {{
            check(expected.getMessage().contains("outside ablation plan"));
        }}
    }}
}}
'''
    with TemporaryDirectory(prefix='bg128-ablation-java-') as scratch:
        test = Path(scratch) / 'AblationPlanProbe.java'
        test.write_text(probe, encoding='utf-8')
        subprocess.run(['javac', '-encoding', 'UTF-8', '-d', scratch, str(helper), str(test)], check=True)
        masters = [str(Path(item['suite']['path']).with_name('master_plan.csv')) for item in identities]
        subprocess.run(['java', '-cp', scratch, 'AblationPlanProbe', *masters], check=True)


def prepare():
    from ablation_budget import assert_available, budget_snapshot

    parent.require(not PREPARED.exists(), 'ablation preparation must not overwrite an existing package')
    sources = historical_sources()
    work = [documents(unit, sources) for unit in UNITS]
    batches = [dict(settings, jobs=[job for phase in phases for job in phase['jobs']]) for settings, phases in work]
    for suite in batches:
        parent.require(not Path(suite['python_output']).exists() and not Path(suite['material_root']).exists(),
                       'new ablation paths already contain evidence')
    check = assert_available(batches, budget_snapshot())
    pins = _pins(sources)
    launch = []
    for (settings, phases), suite in zip(work, batches, strict=True):
        path = Path(settings['suite_path'])
        directory = path.parent
        refs = []
        for cfg in phases:
            config_path = directory / 'configs' / (cfg['phase_id'] + '.json')
            parent.write(config_path, cfg)
            refs.append(reference(config_path))
        jobs = suite['jobs']
        master = directory / 'master_plan.csv'
        identities = directory / 'identities.csv'
        parent._write_rows(master, parent.COLUMNS, jobs)
        helper = _render_helper(settings, jobs, master, directory)
        java = replica._java_check(directory, RPC_PORT)
        parent._write_rows(identities, parent.IDENTITY_COLUMNS, parent._identity_rows(phases, refs))
        stored = dict(settings, phase_configs=refs, source_pins=pins, master_plan=reference(master),
                      scenario_table=reference(directory / 'scenarios.csv'), java_helper=reference(helper),
                      identities=reference(identities), java_validation=java)
        parent.write(path, stored)
        load_package(settings['unit'], sources)
        info = dict(unit=settings['unit'], ablation=settings['ablation'], baseline_seed=settings['baseline_seed'],
                    suite=reference(path), physical_shifts=CAP, python_output=settings['python_output'],
                    anylogic_event_root=settings['material_root'], first=next(parent._identity_rows(phases, refs)),
                    identities=stored['identities'], java_helper=stored['java_helper'],
                    phases=[dict(parent._phase_entry(cfg, ref), recording_id=cfg['recording_id'],
                                 material_event_directory=cfg['jobs'][0]['material_event_directory'],
                                 first_run=cfg['jobs'][0]['run_index'], last_run=cfg['jobs'][-1]['run_index'])
                            for cfg, ref in zip(phases, refs, strict=True)])
        identity = directory / 'launch_identity.json'
        parent.write(identity, info)
        launch.append(reference(identity))
        print('PREPARED', settings['unit'], info['first']['run_index'], flush=True)
    identities = [load_reference(ref) for ref in launch]
    combined = PREPARED / 'Baseline1CampaignPlan.java'
    with combined.open('x', encoding='utf-8') as stream:
        stream.write(_combined_source(identities))
    _verify_combined(combined, identities)
    launch_path = PREPARED / 'launch_identity.json'
    parent.write(launch_path, dict(order=list(UNITS), physical_shifts=ALP_RUNS, training_shifts=TRAIN * len(UNITS),
                                  evaluation_shifts=EVAL * len(UNITS), rpc_port=RPC_PORT,
                                  java_helper=reference(combined), launch_identities=launch))
    parent.require(parent.sha(parent.CORE) == parent.PROTECTED_SHA, 'protected PPO core SHA changed')
    evidence = dict(status='PASS_OFFLINE_NO_LISTENER', launch_identity=reference(launch_path),
                    java_indices_checked=ALP_RUNS, budgets=check, protected_core=reference(parent.CORE),
                    simulation_started=False, budget_registered=False)
    parent.write(REPORT / 'configuration_verification.json', evidence)
    return evidence
