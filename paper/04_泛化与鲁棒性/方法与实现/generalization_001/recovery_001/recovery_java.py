"""Render the remaining-run helper against the original frozen master."""

import json
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from subprocess import run
from tempfile import TemporaryDirectory

from baseline1_v1.java_plan import _replace_once

RESUME_OFFSET = 101
TOTAL = 600
REMAINING = TOTAL - RESUME_OFFSET


def render_java(original_source, *, old_suite, recovery_path, control_directory, bridge_path):
    """Keep original identities and translate AnyLogic index 0 to ordinal 101."""
    helper = old_suite["java_helper"]
    if sha256(original_source.encode("utf-8")).hexdigest() != helper["sha256"]:
        raise ValueError("original Java helper does not match the frozen suite")
    path = Path(recovery_path).resolve()
    control = Path(control_directory).resolve()
    if control == Path(old_suite["control_directory"]).resolve():
        raise ValueError("recovery requires an isolated control directory")
    bridge = Path(bridge_path).resolve()
    replacements = (
        (
            "private static final String CONTROL_DIRECTORY = "
            + json.dumps(old_suite["control_directory"]) + ";",
            "private static final String CONTROL_DIRECTORY = " + json.dumps(str(control)) + ";",
        ),
        (
            f"    private static final int JOB_COUNT = {TOTAL};",
            f"    private static final int JOB_COUNT = {TOTAL};\n"
            f"    private static final int RESUME_OFFSET = {RESUME_OFFSET};\n"
            "    private static final int REMAINING_JOB_COUNT = JOB_COUNT - RESUME_OFFSET;",
        ),
        (
            'require(row.ordinal == 0, "one-click startup requires index 0; recovery must be reviewed");',
            'require(row.ordinal == RESUME_OFFSET, "recovery must start at original ordinal 101");',
        ),
        (json.dumps(old_suite["one_click_entry"]["path"]), json.dumps(str(bridge))),
        (json.dumps(old_suite["one_click_entry"]["sha256"]),
         json.dumps(sha256(bridge.read_bytes()).hexdigest())),
        ('"--suite", ' + json.dumps(old_suite["suite_path"]), '"--recovery", ' + json.dumps(str(path))),
        (
            'require(index >= 0 && index < JOB_COUNT, "Freeform index outside frozen plan");\n'
            "        return plan().rows[index];",
            'require(index >= 0 && index < REMAINING_JOB_COUNT, "Freeform index outside remaining plan");\n'
            "        return plan().rows[index + RESUME_OFFSET];",
        ),
        ("public static int jobCount() { return JOB_COUNT; }",
         "public static int jobCount() { return REMAINING_JOB_COUNT; }"),
        ("public static int resumeOffset() { return 0; }",
         "public static int resumeOffset() { return RESUME_OFFSET; }"),
        (
            '        require(row != null, "run_index absent from frozen master");',
            '        require(row != null, "run_index absent from frozen master");\n'
            '        require(row.ordinal >= RESUME_OFFSET, "run_index was already consumed");',
        ),
        (
            "        Scenario scene = plan().scenarios.get(runIndex);\n"
            '        require(scene != null, "run_index absent from frozen scenario table: " + runIndex);\n'
            "        return scene;",
            "        Plan frozen = plan();\n"
            "        Row row = frozen.byRun.get(runIndex);\n"
            '        require(row != null, "run_index absent from frozen scenario table: " + runIndex);\n'
            '        require(row.ordinal >= RESUME_OFFSET, "run_index was already consumed");\n'
            "        return frozen.scenarios.get(runIndex);",
        ),
    )
    source = original_source
    for before, after in replacements:
        source = _replace_once(source, before, after)
    return source


def _probe_java():
    """Extend the previously tested stage probe with the original master offsets."""
    path = Path(__file__).resolve().parents[1] / "java_scene.py"
    spec = spec_from_file_location("frozen_generalization_java_scene", path)
    upstream = module_from_spec(spec)
    spec.loader.exec_module(upstream)
    source = upstream._probe_java()
    replacements = (
        (
            '        require(Baseline1CampaignPlan.jobCount() == rows.size() - 1, "job count");',
            '        List<String> master = Files.readAllLines(Path.of(args[1]));\n'
            '        int offset = Baseline1CampaignPlan.resumeOffset();\n'
            '        require(offset == 101 && Baseline1CampaignPlan.frozenPlanJobCount() == 600\n'
            '            && Baseline1CampaignPlan.jobCount() == 499 && rows.size() == 601\n'
            '            && master.size() == 601, "job counts or offset differ");',
        ),
        ("        for (int i = 1; i < rows.size(); ++i) {",
         "        for (int i = offset + 1; i < rows.size(); ++i) {"),
        (
            '            require(Baseline1CampaignPlan.runIndex(i - 1) == run, "master run order " + i);',
            '            int index = i - offset - 1;\n'
            '            String[] identity = master.get(i).split(",", -1);\n'
            '            require(Baseline1CampaignPlan.runIndex(index) == run\n'
            '                && run == Long.parseLong(identity[6])\n'
            '                && Baseline1CampaignPlan.environmentSeed(index) == Long.parseLong(identity[7])\n'
            '                && Baseline1CampaignPlan.actorSamplingSeed(index) == Long.parseLong(identity[8])\n'
            '                && Baseline1CampaignPlan.recordingId(index).equals(identity[5])\n'
            '                && Baseline1CampaignPlan.eventRoot(index).equals(identity[9]),\n'
            '                "original master identity " + i);',
        ),
        (
            "        System.out.println(rows.size() - 1);",
            '        require(Baseline1CampaignPlan.runIndex(0) == 9991102L\n'
            '            && Baseline1CampaignPlan.runIndex(498) == 9991600L, "remaining endpoints");\n'
            '        for (int index : new int[] {-1, 499}) {\n'
            '            try {\n'
            '                Baseline1CampaignPlan.runIndex(index);\n'
            '                throw new AssertionError("index outside remainder was accepted");\n'
            '            } catch (IllegalStateException error) {\n'
            '                require(error.getMessage().contains("Freeform index outside remaining plan"),\n'
            '                    "wrong index rejection");\n'
            '            }\n'
            '        }\n'
            '        for (long run : new long[] {9991001L, 9991100L, 9991101L}) {\n'
            '            try {\n'
            '                Baseline1CampaignPlan.loadGrade(run, 0);\n'
            '                throw new AssertionError("consumed scenario was accepted");\n'
            '            } catch (IllegalStateException error) {\n'
            '                require(error.getMessage().contains("run_index was already consumed"),\n'
            '                    "wrong consumed scenario rejection");\n'
            '            }\n'
            '        }\n'
            '        System.out.println(rows.size() - offset - 1);',
        ),
    )
    for before, after in replacements:
        source = _replace_once(source, before, after)
    return source


def verify_java(helper_path, master, scenario_table):
    """Compile and verify every remaining mapping without starting AnyLogic or RPC."""
    helper_path = Path(helper_path).resolve()
    master, scenario_table = Path(master).resolve(), Path(scenario_table).resolve()
    with TemporaryDirectory(prefix="bg128-generalization-recovery-java-") as directory:
        temp = Path(directory)
        probe = temp / "SceneProbe.java"
        probe.write_text(_probe_java(), encoding="utf-8")
        run(["javac", "-encoding", "UTF-8", "-d", str(temp), str(helper_path), str(probe)], check=True)
        result = run(
            ["java", "-cp", str(temp), "SceneProbe", str(scenario_table), str(master)],
            check=True,
            capture_output=True,
            text=True,
        )
        rows = int(result.stdout.strip().splitlines()[-1])
        if rows != REMAINING:
            raise ValueError("Java probe did not verify every remaining identity")
    return {
        "compiled": True, "rows_checked": rows, "resume_offset": RESUME_OFFSET,
        "first_run": 9991102, "last_run": 9991600, "consumed_runs_rejected": True,
        "simulation_started": False,
    }
