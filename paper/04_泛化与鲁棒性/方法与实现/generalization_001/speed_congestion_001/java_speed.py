"""Freeze the AnyLogic speed-variation table and verify it without a simulator."""

from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

from baseline1_v1.java_plan import _replace_once, _require


PLAN = Path(__file__).resolve().parent
MIXED_JAVA = PLAN.parent / "grade_assignment_generalization_001/continuous_training_001/mixed_java.py"
SCENARIOS = "scenarios.csv"
SPEED_SCENARIOS = "speed_scenarios.csv"
HELPER = "Baseline1CampaignPlan.java"
HEADER = "run_index,free_speed_factor_std"
SPEEDS = (0.15,) * 20 + (0.25,) * 20


def _mixed_java():
    spec = spec_from_file_location("speed_congestion_mixed_java", MIXED_JAVA)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _table_bytes(jobs):
    _require(len(jobs) == len(SPEEDS), "speed campaign requires exactly 40 rows")
    rows = [HEADER]
    for job, std in zip(jobs, SPEEDS, strict=True):
        rpc = job["expected_rpc_config"]
        _require(
            rpc["config.runtime_scene.env.freeSpeedFactorStd"] == std
            and rpc["config.runtime_scene.env.freeSpeedFactorMin"] == 0.85
            and rpc["config.runtime_scene.env.freeSpeedFactorMax"] == 1.10
            and rpc["config.runtime_scene.env.useLocalDynamicSpeed"] == 1.0
            and rpc["config.runtime_scene.env.useEndogenousCongestion"] == 1.0
            and rpc["config.runtime_scene.env.deterministicCheck"] == 0.0,
            "speed values, support or enabled traffic model differ from the campaign",
        )
        rows.append(f"{job['run_index']},{std:.2f}")
    return ("\n".join(rows) + "\n").encode("ascii")


def _path_constant(name, path):
    return f"    private static final String {name} = {json.dumps(str(Path(path).resolve()))};"


def _speed_java(path, data):
    return _path_constant("SPEED_TABLE_PATH", path) + r'''
    private static final String SPEED_TABLE_SHA256 = "@SHA@";

    /**
     * Freeform read only. The matching physical identity is checked by
     * Main startup; reading this value never claims or starts a shift.
     */
    public static double freeSpeedFactorStd(int index) {
        long run = runIndex(index);
        return plan().speedFactors.get(run);
    }

    private static java.util.Map<Long, Double> readSpeedFactors(Row[] rows) throws java.io.IOException {
        digestValue(SPEED_TABLE_SHA256, "speed table SHA256");
        byte[] bytes = java.nio.file.Files.readAllBytes(absolute(SPEED_TABLE_PATH, "speed table path"));
        require(sha256(bytes).equals(SPEED_TABLE_SHA256), "speed table byte SHA256 differs");
        String csv = java.nio.charset.StandardCharsets.US_ASCII.newDecoder()
            .decode(java.nio.ByteBuffer.wrap(bytes)).toString();
        String[] lines = csv.split("\n", -1);
        require(rows.length == 40 && lines.length == rows.length + 2
                && lines[rows.length + 1].isEmpty() && lines[0].equals("run_index,free_speed_factor_std"),
            "speed table header or row count differs");
        java.util.Map<Long, Double> speeds = new java.util.HashMap<>();
        for (int i = 0; i < rows.length; ++i) {
            String[] values = lines[i + 1].split(",", -1);
            require(values.length == 2 && positive(values[0], "speed table run_index") == rows[i].run,
                "speed table does not match frozen master order");
            require(values[1].equals(i < 20 ? "0.15" : "0.25"), "unexpected speed standard deviation");
            require(speeds.put(rows[i].run, scenarioNumber(values[1])) == null, "duplicate speed table run");
        }
        return java.util.Collections.unmodifiableMap(speeds);
    }

'''.replace("@SHA@", sha256(data).hexdigest())


def write_java(suite, jobs, master_path, directory, *, bridge_path):
    """Publish the helper and both SHA-pinned tables exclusively."""
    jobs = list(jobs)
    directory = Path(directory).resolve()
    scenarios, speeds, helper = (directory / name for name in (SCENARIOS, SPEED_SCENARIOS, HELPER))
    _require(not any(path.exists() for path in (scenarios, speeds, helper)), "output already sealed")
    speed_bytes = _table_bytes(jobs)
    with TemporaryDirectory(prefix="bg128-speed-java-generate-") as scratch:
        temp = Path(scratch)
        old_scenarios = temp / SCENARIOS
        base = _mixed_java().write_java(suite, jobs, master_path, old_scenarios, temp, bridge_path=bridge_path)
        source = _replace_once(
            base.read_text(encoding="utf-8"),
            _path_constant("SCENARIO_TABLE_PATH", old_scenarios),
            _path_constant("SCENARIO_TABLE_PATH", scenarios),
        )
        scenario_bytes = old_scenarios.read_bytes()
    anchor = "    private static final class Row {"
    source = _replace_once(source, anchor, _speed_java(speeds, speed_bytes) + anchor)
    source = _replace_once(source, "        final java.util.Map<Long, Scenario> scenarios;",
                           "        final java.util.Map<Long, Scenario> scenarios;\n"
                           "        final java.util.Map<Long, Double> speedFactors;")
    source = _replace_once(source, "            this.scenarios = readScenarios(rows);",
                           "            this.scenarios = readScenarios(rows);\n"
                           "            this.speedFactors = readSpeedFactors(rows);")
    directory.mkdir(parents=True, exist_ok=True)
    for target, data in ((scenarios, scenario_bytes), (speeds, speed_bytes), (helper, source.encode("utf-8"))):
        with target.open("xb") as stream:
            stream.write(data)
    return helper


def verify_java(directory):
    """Check grade/road and speed tables, zero-based Freeform and SHA rejection."""
    directory = Path(directory).resolve()
    helper, speeds = directory / HELPER, directory / SPEED_SCENARIOS
    result = _mixed_java().verify_java(helper, directory / SCENARIOS)
    with TemporaryDirectory(prefix="bg128-speed-java-check-") as scratch:
        temp = Path(scratch)
        probe = temp / "SpeedProbe.java"
        probe.write_text(_probe_java(), encoding="utf-8")
        command = ["javac", "-encoding", "UTF-8", "-d", str(temp)]
        subprocess.run([*command, str(helper), str(probe)], check=True)
        java = ["java", "-cp", str(temp), "SpeedProbe"]
        checked = subprocess.run([*java, str(speeds)], check=True, capture_output=True, text=True)
        count = int(checked.stdout.strip())
        _require(count == result["rows_checked"] == len(SPEEDS), "verification did not cover 40 rows")
        altered = temp / "changed_speed.csv"
        altered.write_bytes(speeds.read_bytes().replace(b",0.15\n", b",0.16\n", 1))
        changed = temp / HELPER
        changed.write_text(_replace_once(
            helper.read_text(encoding="utf-8"),
            _path_constant("SPEED_TABLE_PATH", speeds),
            _path_constant("SPEED_TABLE_PATH", altered),
        ), encoding="utf-8")
        subprocess.run([*command, str(changed), str(probe)], check=True)
        subprocess.run([*java, "expected-speed-integrity-failure"], check=True)
    return {
        **result,
        "speed_rows_checked": count,
        "speed_0_15_rows": 20,
        "speed_0_25_rows": 20,
        "freeform_boundaries_checked": True,
        "altered_speed_sidecar_rejected": True,
    }


def _probe_java():
    return r'''
public final class SpeedProbe {
    private static void require(boolean value, String label) {
        if (!value) throw new AssertionError(label);
    }

    private static void expectInvalidIndex(int index) {
        try {
            Baseline1CampaignPlan.freeSpeedFactorStd(index);
            throw new AssertionError("out-of-range Freeform index accepted");
        } catch (IllegalStateException error) {
            require(error.getMessage().contains("Freeform index outside frozen plan"),
                "wrong Freeform rejection");
        }
    }

    public static void main(String[] args) throws Exception {
        if (args[0].equals("expected-speed-integrity-failure")) {
            try {
                Baseline1CampaignPlan.freeSpeedFactorStd(0);
                throw new AssertionError("changed speed table accepted");
            } catch (IllegalStateException error) {
                require(error.getMessage().contains("speed table byte SHA256 differs"), "wrong speed rejection");
            }
            return;
        }
        java.util.List<String> rows = java.nio.file.Files.readAllLines(java.nio.file.Path.of(args[0]));
        require(rows.size() == 41 && Baseline1CampaignPlan.jobCount() == 40, "unexpected campaign count");
        for (int i = 0; i < 40; ++i) {
            String[] cells = rows.get(i + 1).split(",", -1);
            require(Baseline1CampaignPlan.runIndex(i) == Long.parseLong(cells[0]), "physical identity");
            require(Baseline1CampaignPlan.freeSpeedFactorStd(i) == Double.parseDouble(cells[1])
                    && Baseline1CampaignPlan.freeSpeedFactorStd(i) == (i < 20 ? 0.15 : 0.25),
                "frozen standard deviation");
        }
        expectInvalidIndex(-1);
        expectInvalidIndex(40);
        System.out.println(40);
    }
}
'''
