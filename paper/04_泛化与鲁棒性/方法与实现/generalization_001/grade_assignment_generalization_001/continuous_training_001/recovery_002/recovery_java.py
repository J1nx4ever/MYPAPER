"""Render and verify the second recovery helper with one excluded physical run."""

import csv
import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import re
import subprocess
from tempfile import TemporaryDirectory

from baseline1_v1.java_plan import _replace_once, _require
from business_graph_summary_campaign_v1.plan_io import sha


PLAN = Path(__file__).resolve().parent
OFFSET = 1
FULL_COUNT = 2424


def _quoted(path):
    return json.dumps(str(Path(path).resolve()))


def write_java(original_helper, destination, original_control, recovery_control,
               original_bridge, recovery_bridge, original_suite, recovery_json):
    """Keep immutable job/scenario bindings and remap Freeform index zero to row one."""
    original_helper, destination = (Path(path).resolve() for path in (original_helper, destination))
    original_control, recovery_control = (Path(path).resolve() for path in (original_control, recovery_control))
    original_bridge, recovery_bridge = (Path(path).resolve() for path in (original_bridge, recovery_bridge))
    original_suite, recovery_json = (Path(path).resolve() for path in (original_suite, recovery_json))
    _require(destination.name == 'Baseline1CampaignPlan.java' and not destination.exists(),
             'new Baseline1CampaignPlan.java destination required')
    _require(original_control != recovery_control and original_bridge != recovery_bridge
             and original_suite != recovery_json, 'recovery startup must use a separate identity')
    _require(recovery_control == original_control.parent / 'recovery_002' / 'control',
             'second recovery control must be isolated beside the original control')
    original = original_helper.read_text(encoding='utf-8')
    pairs = (
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
    _require(restored == original, 'second recovery changed the original job bindings')
    _require(all(_quoted(path) not in source for path in (original_control, original_bridge, original_suite)),
             'original startup location remains embedded in the recovery helper')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x', encoding='utf-8') as stream:
        stream.write(source)
    return destination


def _literal(source, pattern):
    values = re.findall(pattern, source)
    _require(len(values) == 1, 'missing or duplicate Java source binding')
    return json.loads(values[0])


def _probe():
    path = PLAN.parent / 'mixed_java.py'
    spec = spec_from_file_location('mixed_java_resume_probe', path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    source = module._probe_java()
    source = _replace_once(
        source, 'require(Baseline1CampaignPlan.jobCount() == rows.size() - 1, "master row count");',
        '''List<String> master = Files.readAllLines(Path.of(args[1]));
        require(master.size() == rows.size(), "same original master and sidecar row counts");
        require(Baseline1CampaignPlan.frozenPlanJobCount() == rows.size() - 1, "original master row count");
        require(Baseline1CampaignPlan.resumeOffset() == 1, "resume offset");
        require(Baseline1CampaignPlan.jobCount() == rows.size() - 2, "remaining run count");''')
    source = _replace_once(
        source, 'require(Baseline1CampaignPlan.runIndex(i - 1) == run, "master order");',
        '''String[] job = master.get(i).split(",", -1);
            require(Long.parseLong(job[6]) == run, "original scenario/master order");
            if (i > 1) {
                int index = i - 2;
                require(Baseline1CampaignPlan.runIndex(index) == run, "remaining master run order");
                require(Baseline1CampaignPlan.environmentSeed(index) == Long.parseLong(job[7]), "environment seed");
                require(Baseline1CampaignPlan.actorSamplingSeed(index) == Long.parseLong(job[8]), "actor seed");
                require(Baseline1CampaignPlan.recordingId(index).equals(job[5]), "recording ID");
                require(Baseline1CampaignPlan.eventRoot(index).equals(job[9]), "event root");
                String phase = job[5].substring(args[2].length() + 1);
                require(Baseline1CampaignPlan.phaseId(index).equals(phase), "phase");
                require(Baseline1CampaignPlan.materialRoot(index).equals(Path.of(args[3], phase).toString()),
                    "material root");
                require(Baseline1CampaignPlan.dispatchPolicy(index) == 3
                    && Baseline1CampaignPlan.materialEnabled(index), "original BG128 dispatcher");
            }''')
    source = _replace_once(source, 'System.out.println(rows.size() - 1);',
        '''try {
            Baseline1CampaignPlan.runIndex(Baseline1CampaignPlan.jobCount());
            throw new AssertionError("out-of-range Freeform index accepted");
        } catch (IllegalStateException error) {
            require(error.getMessage().contains("Freeform index outside remaining plan"), "wrong index rejection");
        }
        System.out.println(rows.size() - 1);''')
    return source


def _verify_startup(source, temp, jobs):
    """Check the Java startup wiring using a temporary recorder instead of a controller."""
    declared = {
        'control': _literal(source, r'CONTROL_DIRECTORY = ("[^"\n]+");'),
        'bridge': _literal(source, r'Path bridge = absolute\(("[^"\n]+"), "bridge path"\)'),
        'suite': _literal(source, r'"--suite", ("[^"\n]+"), "--run-index"'),
    }
    launch = temp / 'startup'
    launch.mkdir()
    control = launch / 'control'
    bridge = launch / 'stub_bridge.py'
    bridge.write_text(
        'import json, sys\nfrom pathlib import Path\n'
        'Path(__file__).with_name("argv.json").write_text(json.dumps(sys.argv[1:]))\n',
        encoding='utf-8')
    descriptor = launch / 'recovery.json'
    descriptor.write_text('{}\n', encoding='utf-8')
    fake = source
    for key, path in zip(('control', 'bridge', 'suite'), (control, bridge, descriptor), strict=True):
        fake = _replace_once(fake, _quoted(declared[key]), _quoted(path))
    fake = _replace_once(fake,
        f'sha256(java.nio.file.Files.readAllBytes(bridge)).equals("{sha(declared["bridge"])}")',
        f'sha256(java.nio.file.Files.readAllBytes(bridge)).equals("{sha(bridge)}")')
    java_helper = launch / 'Baseline1CampaignPlan.java'
    java_helper.write_text(fake, encoding='utf-8')
    probe = launch / 'StartupProbe.java'
    probe.write_text('''
public final class StartupProbe {
    public static void main(String[] args) throws Exception {
        if (args[0].equals("check-settled")) {
            try {
                Baseline1CampaignPlan.awaitConfigSha256(Long.parseLong(args[1]), Long.parseLong(args[2]),
                    args[3], args[4], "boundary-events-v1-native-rpc-runtime-scene-v1");
                throw new AssertionError("settled run was accepted");
            } catch (IllegalStateException error) {
                if (!error.getMessage().contains("settled run cannot start again")) throw error;
            }
            return;
        }
        java.lang.reflect.Method row = Baseline1CampaignPlan.class.getDeclaredMethod("row", int.class);
        row.setAccessible(true);
        Object selected = row.invoke(null, Integer.parseInt(args[0]));
        java.lang.reflect.Method startup = Baseline1CampaignPlan.class
            .getDeclaredMethod("ensureController", selected.getClass());
        startup.setAccessible(true);
        startup.invoke(null, selected);
    }
}
''', encoding='utf-8')
    subprocess.run(['javac', '-encoding', 'UTF-8', '-d', str(launch), str(java_helper), str(probe)], check=True)
    command = ['java', '-cp', str(launch), 'StartupProbe']
    previous, first = jobs[:2]
    subprocess.run([*command, 'check-settled', previous['run_index'], previous['experiment_seed'],
                    previous['recording_id'], previous['event_directory']], check=True, timeout=15)
    argv = launch / 'argv.json'
    wrong_slot = subprocess.run([*command, '1'], capture_output=True, text=True, timeout=15)
    _require(wrong_slot.returncode != 0 and 'startup requires first remaining ordinal' in wrong_slot.stderr
             and not argv.exists(), 'non-first remaining run invoked the recovery bridge')
    subprocess.run([*command, '0'], check=True, timeout=15)
    _require(json.loads(argv.read_bytes()) == ['--suite', str(descriptor), '--run-index', first['run_index'],
             '--environment-seed', first['experiment_seed'], '--recording-id', first['recording_id'],
             '--from-anylogic-run'], 'recovery bridge arguments differ from the first remaining slot')
    control.mkdir()
    stop = control / 'STOP.properties'
    details = 'reason=synthetic stop\nsecond line should not appear\n'
    stop.write_text(details, encoding='utf-8')
    stopped = subprocess.run([*command, '0'], capture_output=True, text=True, timeout=15)
    _require(stopped.returncode != 0 and str(stop) in stopped.stderr and 'reason=synthetic stop' in stopped.stderr
             and 'second line should not appear' not in stopped.stderr and stop.read_text() == details,
             'STOP diagnostics must report the new control path and first line without editing the record')


def verify_java(helper, scenarios_path):
    """Compile, check every row, reject altered bytes and probe startup offline."""
    helper, scenarios_path = (Path(path).resolve() for path in (helper, scenarios_path))
    source = helper.read_text(encoding='utf-8')
    master = _literal(source, r'MASTER_PLAN_PATH = ("[^"\n]+");')
    campaign = _literal(source, r'CAMPAIGN_ID = ("[^"\n]+");')
    material_root = Path(_literal(source, r'CONTROL_DIRECTORY = ("[^"\n]+");')).parents[1].with_name(
        campaign + '_materials')
    with Path(master).open(newline='') as stream:
        jobs = list(csv.DictReader(stream))
    _require(len(jobs) == FULL_COUNT, 'original master size differs')
    with TemporaryDirectory(prefix='bg128-second-recovery-java-') as folder:
        temp = Path(folder)
        probe = temp / 'SceneProbe.java'
        probe.write_text(_probe(), encoding='utf-8')
        javac = ['javac', '-encoding', 'UTF-8', '-d', str(temp)]
        java = ['java', '-cp', str(temp), 'SceneProbe']
        subprocess.run([*javac, str(helper), str(probe)], check=True)
        result = subprocess.run([*java, str(scenarios_path), master, campaign, str(material_root)],
                                check=True, capture_output=True, text=True, timeout=30)
        count = int(result.stdout.strip())
        _require(count == FULL_COUNT, 'Java original scenario coverage differs')
        altered = temp / 'tampered.csv'
        altered.write_bytes(scenarios_path.read_bytes() + b'\n')
        tampered = temp / 'Baseline1CampaignPlan.java'
        tampered.write_text(_replace_once(
            source, f'SCENARIO_TABLE_PATH = {_quoted(scenarios_path)};',
            f'SCENARIO_TABLE_PATH = {_quoted(altered)};'), encoding='utf-8')
        subprocess.run([*javac, str(tampered), str(probe)], check=True)
        subprocess.run([*java, 'expect-sha-failure'], check=True, timeout=15)
        _verify_startup(source, temp, jobs)
    return dict(compiled=True, rows_checked=count, remaining_rows_checked=count - OFFSET,
                first_remaining_run=int(jobs[OFFSET]['run_index']), last_remaining_run=int(jobs[-1]['run_index']),
                prior_run_excluded=True, material_paths_checked=True, startup_arguments_checked=True,
                stop_diagnostics_checked=True,
                altered_sidecar_rejected=True, simulation_started=False)
