"""Bind the replacement shift and verify the new AnyLogic helper offline."""

import csv
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

from baseline1_v1.java_plan import _replace_once, _require
from business_graph_summary_campaign_v1.plan_io import sha


PLAN = Path(__file__).resolve().parent
OFFSET = 2
FULL_COUNT = 2424
FAILED_RUN = 10000003
REPLACEMENT_RUN = 10002425
SECOND_RUN = 10000004
REPLACEMENT_PHASE = 's01_BG128_train_u0001_replacement'
REPLACEMENT_FIELDS = {
    'source_recording_id', 'recording_id', 'run_index', 'experiment_seed',
    'actor_sampling_seed', 'event_directory', 'event_bundle', 'python_output',
}

_resume_spec = spec_from_file_location('mixed_recovery_002_java', PLAN.parent / 'recovery_002' / 'recovery_java.py')
_resume_java = module_from_spec(_resume_spec)
_resume_spec.loader.exec_module(_resume_java)
_quoted = _resume_java._quoted
_literal = _resume_java._literal


def _table(path):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


def _check_tables(master, scenarios, effective_master, effective_scenarios, campaign):
    master_header, original_jobs = _table(master)
    effective_header, jobs = _table(effective_master)
    original_scene_header, original_scenes = _table(scenarios)
    effective_scene_header, scenes = _table(effective_scenarios)
    _require(master_header == effective_header and original_scene_header == effective_scene_header,
             'recovery tables must retain the original headers')
    _require(len(original_jobs) == len(jobs) == len(original_scenes) == len(scenes) == FULL_COUNT,
             'recovery tables must retain the original row count')
    _require(original_jobs[:OFFSET] == jobs[:OFFSET] and original_jobs[OFFSET + 1:] == jobs[OFFSET + 1:],
             'recovery changed a master row outside the replacement ordinal')
    _require(original_scenes[:OFFSET] == scenes[:OFFSET] and original_scenes[OFFSET + 1:] == scenes[OFFSET + 1:],
             'recovery changed a scenario row outside the replacement ordinal')
    failed, replacement = original_jobs[OFFSET], jobs[OFFSET]
    _require(int(failed['run_index']) == FAILED_RUN and int(replacement['run_index']) == REPLACEMENT_RUN,
             'unexpected failed or replacement physical run')
    _require(all(replacement[key] == failed[key] for key in master_header if key not in REPLACEMENT_FIELDS),
             'replacement changed logical phase membership or checkpoint target')
    _require(replacement['recording_id'] == campaign + '-' + REPLACEMENT_PHASE,
             'replacement recording ID must map to its distinct material directory')
    previous_seeds = {row[key] for row in original_jobs for key in ('experiment_seed', 'actor_sampling_seed')}
    replacement_seeds = {replacement['experiment_seed'], replacement['actor_sampling_seed']}
    _require(len(replacement_seeds) == 2 and not previous_seeds.intersection(replacement_seeds),
             'replacement environment and actor seeds must both be fresh')
    _require(scenes[OFFSET] == dict(original_scenes[OFFSET], run_index=str(REPLACEMENT_RUN)),
             'replacement must preserve the failed ordinal scenario')
    _require([row['run_index'] for row in jobs] == [row['run_index'] for row in scenes],
             'effective master and scenario orders differ')


def write_java(original_helper, destination, original_control, recovery_control,
               original_bridge, recovery_bridge, original_suite, recovery_json,
               effective_master, effective_scenarios):
    """Render an isolated startup with the first two ordinals closed and ordinal two replaced."""
    original_helper, destination = (Path(path).resolve() for path in (original_helper, destination))
    original_control, recovery_control = (Path(path).resolve() for path in (original_control, recovery_control))
    original_bridge, recovery_bridge = (Path(path).resolve() for path in (original_bridge, recovery_bridge))
    original_suite, recovery_json = (Path(path).resolve() for path in (original_suite, recovery_json))
    effective_master, effective_scenarios = (Path(path).resolve() for path in (effective_master, effective_scenarios))
    _require(destination.name == 'Baseline1CampaignPlan.java' and not destination.exists(),
             'new Baseline1CampaignPlan.java destination required')
    _require(original_bridge != recovery_bridge and original_suite != recovery_json,
             'recovery startup must use a separate bridge and descriptor')
    _require(recovery_control == original_control.parent / PLAN.name / 'control',
             'third recovery control must be isolated beside the original control')
    original = original_helper.read_text(encoding='utf-8')
    master = Path(_literal(original, r'MASTER_PLAN_PATH = ("[^"\n]+");'))
    scenarios = Path(_literal(original, r'SCENARIO_TABLE_PATH = ("[^"\n]+");'))
    campaign = _literal(original, r'CAMPAIGN_ID = ("[^"\n]+");')
    _require(effective_master != master and effective_scenarios != scenarios,
             'effective tables must preserve the original files')
    _check_tables(master, scenarios, effective_master, effective_scenarios, campaign)
    pairs = (
        (f'MASTER_PLAN_PATH = {_quoted(master)};', f'MASTER_PLAN_PATH = {_quoted(effective_master)};'),
        (f'MASTER_PLAN_SHA256 = "{sha(master)}";', f'MASTER_PLAN_SHA256 = "{sha(effective_master)}";'),
        (f'SCENARIO_TABLE_PATH = {_quoted(scenarios)};', f'SCENARIO_TABLE_PATH = {_quoted(effective_scenarios)};'),
        (f'SCENARIO_TABLE_SHA256 = "{sha(scenarios)}";', f'SCENARIO_TABLE_SHA256 = "{sha(effective_scenarios)}";'),
        (f'CONTROL_DIRECTORY = {_quoted(original_control)};',
         f'CONTROL_DIRECTORY = {_quoted(recovery_control)};'),
        (f'absolute({_quoted(original_bridge)}, "bridge path")',
         f'absolute({_quoted(recovery_bridge)}, "bridge path")'),
        (f'sha256(java.nio.file.Files.readAllBytes(bridge)).equals("{sha(original_bridge)}")',
         f'sha256(java.nio.file.Files.readAllBytes(bridge)).equals("{sha(recovery_bridge)}")'),
        (f'"--suite", {_quoted(original_suite)}, "--run-index"',
         f'"--suite", {_quoted(recovery_json)}, "--run-index"'),
        (f'private static final int JOB_COUNT = {FULL_COUNT};',
         f'private static final int JOB_COUNT = {FULL_COUNT};\n'
         f'    private static final int RESUME_OFFSET = {OFFSET};'),
        ('require(row.ordinal == 0, "one-click startup requires index 0; recovery must be reviewed");',
         'require(row.ordinal == RESUME_OFFSET, "recovery startup requires first remaining ordinal");'),
        ('require(index >= 0 && index < JOB_COUNT, "Freeform index outside frozen plan");\n'
         '        return plan().rows[index];',
         'require(index >= 0 && index < jobCount(), "Freeform index outside remaining plan");\n'
         '        return plan().rows[index + RESUME_OFFSET];'),
        ('public static int jobCount() { return JOB_COUNT; }',
         'public static int jobCount() { return JOB_COUNT - RESUME_OFFSET; }'),
        ('public static int resumeOffset() { return 0; }',
         'public static int resumeOffset() { return RESUME_OFFSET; }'),
        ('require(row != null, "run_index absent from frozen master");',
         'require(row != null, "run_index absent from frozen master");\n'
         '        require(row.ordinal >= RESUME_OFFSET, "settled run cannot start again");'),
        ('require(!java.nio.file.Files.exists(control.resolve("STOP.properties")), "STOP.properties exists");',
         '''java.nio.file.Path stop = control.resolve("STOP.properties");
        if (!java.nio.file.Files.exists(stop)) return;
        String reason;
        try (java.io.BufferedReader input = java.nio.file.Files.newBufferedReader(stop)) {
            reason = input.readLine();
        } catch (java.io.IOException | SecurityException error) {
            reason = "STOP reason unavailable";
        }
        require(false, "STOP.properties exists at " + stop
            + (reason == null || reason.isBlank() ? "" : " (" + reason + ")"));'''),
    )
    source = original
    for before, after in pairs:
        source = _replace_once(source, before, after)
    restored = source
    for before, after in reversed(pairs):
        restored = _replace_once(restored, after, before)
    _require(restored == original, 'third recovery changed more than the frozen startup and table bindings')
    _require(all(_quoted(path) not in source for path in
                 (master, scenarios, original_control, original_bridge, original_suite)),
             'an original startup or table binding remains embedded in the helper')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x', encoding='utf-8') as stream:
        stream.write(source)
    return destination


def _probe():
    source = _resume_java._probe()
    source = _replace_once(source, 'resumeOffset() == 1', f'resumeOffset() == {OFFSET}')
    source = _replace_once(source, 'jobCount() == rows.size() - 2', f'jobCount() == rows.size() - {OFFSET + 1}')
    source = _replace_once(source, 'if (i > 1) {\n                int index = i - 2;',
                           f'if (i > {OFFSET}) {{\n                int index = i - {OFFSET + 1};')
    source = _replace_once(source, 'if (args[0].equals("expect-sha-failure")) {',
        '''if (args[0].equals("expect-master-sha-failure")) {
            try {
                Baseline1CampaignPlan.runIndex(0);
                throw new AssertionError("altered master was accepted");
            } catch (IllegalStateException error) {
                require(error.getMessage().contains("master plan byte SHA256 differs"), "wrong master SHA rejection");
            }
            return;
        }
        if (args[0].equals("expect-sha-failure")) {''')
    return _replace_once(source, 'System.out.println(rows.size() - 1);',
        f'''require(Baseline1CampaignPlan.runIndex(0) == {REPLACEMENT_RUN}L, "replacement first");
        require(Baseline1CampaignPlan.runIndex(1) == {SECOND_RUN}L, "next original run");
        require(Baseline1CampaignPlan.phaseId(0).equals("{REPLACEMENT_PHASE}"), "replacement phase");
        require(Baseline1CampaignPlan.stageId(0).equals("BG128"), "replacement stage");
        require(!Baseline1CampaignPlan.materialRoot(0).equals(Baseline1CampaignPlan.materialRoot(1)),
            "distinct replacement material root");
        try {{
            Baseline1CampaignPlan.scenarioId({FAILED_RUN}L);
            throw new AssertionError("failed physical run accepted");
        }} catch (IllegalStateException error) {{
            require(error.getMessage().contains("run_index absent from frozen scenario table"),
                "wrong failed run rejection");
        }}
        System.out.println(rows.size() - 1);''')


def verify_java(helper, scenarios_path):
    """Compile and check all effective rows and the exact startup with a temporary stub."""
    helper, scenarios_path = (Path(path).resolve() for path in (helper, scenarios_path))
    source = helper.read_text(encoding='utf-8')
    master = Path(_literal(source, r'MASTER_PLAN_PATH = ("[^"\n]+");'))
    bound_scenarios = Path(_literal(source, r'SCENARIO_TABLE_PATH = ("[^"\n]+");'))
    _require(scenarios_path == bound_scenarios, 'helper is bound to another scenario table')
    campaign = _literal(source, r'CAMPAIGN_ID = ("[^"\n]+");')
    material_root = Path(_literal(source, r'CONTROL_DIRECTORY = ("[^"\n]+");')).parents[1].with_name(
        campaign + '_materials')
    _, jobs = _table(master)
    _require(len(jobs) == FULL_COUNT, 'effective master size differs')
    with TemporaryDirectory(prefix='bg128-third-recovery-java-') as folder:
        temp = Path(folder)
        probe = temp / 'SceneProbe.java'
        probe.write_text(_probe(), encoding='utf-8')
        javac = ['javac', '-encoding', 'UTF-8', '-d', str(temp)]
        java = ['java', '-cp', str(temp), 'SceneProbe']
        subprocess.run([*javac, str(helper), str(probe)], check=True)
        result = subprocess.run([*java, str(scenarios_path), str(master), campaign, str(material_root)],
                                check=True, capture_output=True, text=True, timeout=30)
        count = int(result.stdout.strip())
        _require(count == FULL_COUNT, 'Java effective scenario coverage differs')
        for constant, path, command in (('SCENARIO_TABLE_PATH', scenarios_path, 'expect-sha-failure'),
                                        ('MASTER_PLAN_PATH', master, 'expect-master-sha-failure')):
            altered = temp / f'tampered-{constant}.csv'
            altered.write_bytes(path.read_bytes() + b'\n')
            tampered = temp / 'Baseline1CampaignPlan.java'
            tampered.write_text(_replace_once(source, f'{constant} = {_quoted(path)};',
                                              f'{constant} = {_quoted(altered)};'), encoding='utf-8')
            subprocess.run([*javac, str(tampered), str(probe)], check=True)
            subprocess.run([*java, command], check=True, timeout=15)
        _resume_java._verify_startup(source, temp, jobs[OFFSET - 1:OFFSET + 1])
    return dict(compiled=True, rows_checked=count, remaining_rows_checked=count - OFFSET,
                first_remaining_run=int(jobs[OFFSET]['run_index']),
                second_remaining_run=int(jobs[OFFSET + 1]['run_index']),
                last_remaining_run=int(jobs[-1]['run_index']),
                prior_run_excluded=True, failed_run_excluded=True, material_paths_checked=True,
                startup_arguments_checked=True, stop_diagnostics_checked=True,
                altered_master_rejected=True, altered_sidecar_rejected=True, simulation_started=False)
