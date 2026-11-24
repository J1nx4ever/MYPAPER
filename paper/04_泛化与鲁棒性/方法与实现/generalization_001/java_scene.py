"""Render the existing one-click gate with a separately frozen scene table."""

import csv
from importlib.util import module_from_spec, spec_from_file_location
import io
import json
from math import isfinite
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from baseline1_v1.java_plan import _replace_once, _require
from business_graph_summary_campaign_v1.plan_io import sha


PLAN = Path(__file__).resolve().parent
SCENARIO_FIELDS = (
    'run_index', 'scenario_id',
    'load0_grade', 'load1_grade', 'load2_grade',
    'load0_grade_half_range', 'load1_grade_half_range', 'load2_grade_half_range',
    'road_6_speed_limit_mps',
)
ROAD_SEGMENT = ('intersection1', 'Crusher')


def scenario_table_bytes(jobs):
    """Project the Java table directly from the Python expected-RPC values."""
    text = io.StringIO(newline='')
    writer = csv.writer(text, lineterminator='\n')
    writer.writerow(SCENARIO_FIELDS)
    for job in jobs:
        run, scene_id = job['run_index'], job['scene_id']
        _require(type(run) is int and run > 0, 'invalid scenario run_index')
        _require(isinstance(scene_id, str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', scene_id)
                 is not None, 'unsafe scenario ID')
        rpc = job['expected_rpc_config']
        grades = [float(rpc[f'config.runtime_scene.load.{i}.grade']) for i in range(3)]
        half_ranges = [float(rpc[f'config.runtime_scene.load.{i}.gradeHalfRange']) for i in range(3)]
        limit = float(rpc['config.runtime_scene.road.6.speedLimit_mps'])
        _require(all(isfinite(grade) and isfinite(half_range) and 0 <= half_range
                     and half_range <= grade <= 1 - half_range
                     for grade, half_range in zip(grades, half_ranges)),
                 'invalid scene ore grades or half-ranges')
        _require(isfinite(limit) and limit > 0, 'invalid scene road speed limit')
        writer.writerow((run, scene_id, *grades, *half_ranges, limit))
    return text.getvalue().encode('ascii')


def render_helper(suite, jobs, master, scenario_table, suite_path, bridge_path):
    """Use the established master, ready, and claim protocol for paired jobs."""
    master, scenario_table = Path(master).resolve(), Path(scenario_table).resolve()
    _require(bool(jobs) and [job['ordinal'] for job in jobs] == list(range(len(jobs)))
             and len({job['run_index'] for job in jobs}) == len(jobs),
             'invalid generalization master order or duplicate run_index')
    _require(scenario_table.read_bytes() == scenario_table_bytes(jobs),
             'Java scenario table differs from Python expected-RPC values')

    creator = PLAN.parent / 'formal_training_001/prepare_training.py'
    spec = spec_from_file_location('frozen_bg128_java_source', creator)
    original = module_from_spec(spec)
    spec.loader.exec_module(original)
    _require(master.read_bytes() == original.master_bytes(jobs),
             'Java master differs from the scenario jobs')
    first_phase = [job for job in jobs if job['recording_id'] == jobs[0]['recording_id']]
    _require(first_phase == jobs[:len(first_phase)], 'first scenario phase must be contiguous')
    # The base renderer checks that seeds are unique. The first phase has unique
    # pairs, while the frozen full master pairs each slot across later phases.
    with TemporaryDirectory(prefix='bg128-generalization-java-') as folder:
        projection = Path(folder) / 'first_phase.csv'
        projection.write_bytes(original.master_bytes(first_phase))
        source = original.render_java(dict(suite, jobs=first_phase), projection,
                                      suite_path=suite_path, bridge_path=bridge_path)
        source = _replace_once(
            source,
            'private static final String MASTER_PLAN_PATH = ' + json.dumps(str(projection)) + ';',
            'private static final String MASTER_PLAN_PATH = ' + json.dumps(str(master)) + ';',
        )
        source = _replace_once(
            source, 'private static final String MASTER_PLAN_SHA256 = ' + json.dumps(sha(projection)) + ';',
            'private static final String MASTER_PLAN_SHA256 = ' + json.dumps(sha(master)) + ';',
        )
    source = _replace_once(source, f'private static final int JOB_COUNT = {len(first_phase)};',
                           f'private static final int JOB_COUNT = {len(jobs)};')
    anchor = '    private static final class Row {'
    source = _replace_once(source, anchor, _scenario_java(scenario_table) + '\n' + anchor)
    source = _replace_once(
        source,
        '        Plan(Row[] rows, java.util.Map<Long, Row> byRun) { this.rows = rows; this.byRun = byRun; }',
        '''        final java.util.Map<Long, Scenario> scenarios;
        Plan(Row[] rows, java.util.Map<Long, Row> byRun) throws java.io.IOException {
            this.rows = rows;
            this.byRun = byRun;
            this.scenarios = readScenarios(rows);
        }''',
    )
    return source


def _scenario_java(scenario_table):
    replacements = {
        '@SCENARIO_TABLE@': json.dumps(str(scenario_table)),
        '@SCENARIO_SHA@': json.dumps(sha(scenario_table)),
        '@SCENARIO_HEADER@': json.dumps(','.join(SCENARIO_FIELDS)),
        '@ROAD_FROM@': json.dumps(ROAD_SEGMENT[0]),
        '@ROAD_TO@': json.dumps(ROAD_SEGMENT[1]),
    }
    source = r'''
    private static final String SCENARIO_TABLE_PATH = @SCENARIO_TABLE@;
    private static final String SCENARIO_TABLE_SHA256 = @SCENARIO_SHA@;
    private static final String SCENARIO_HEADER = @SCENARIO_HEADER@;

    private static final class Scenario {
        final String id;
        final double[] grades = new double[3];
        final double[] halfRanges = new double[3];
        final double road6Limit;

        Scenario(String[] values) {
            id = values[1];
            require(id.matches("[A-Za-z0-9][A-Za-z0-9._-]*"), "invalid scenario ID");
            for (int i = 0; i < 3; ++i) {
                grades[i] = scenarioNumber(values[i + 2]);
                halfRanges[i] = scenarioNumber(values[i + 5]);
                require(halfRanges[i] >= 0.0 && grades[i] >= halfRanges[i]
                    && grades[i] <= 1.0 - halfRanges[i], "invalid scenario ore grade or half-range");
            }
            road6Limit = scenarioNumber(values[8]);
            require(road6Limit > 0.0, "scenario road speed limit must be positive");
        }
    }

    private static double scenarioNumber(String value) {
        require(value.matches("[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?"),
            "invalid scenario numeric syntax");
        double number = Double.parseDouble(value);
        require(Double.isFinite(number), "nonfinite scenario number");
        return number;
    }

    private static java.util.Map<Long, Scenario> readScenarios(Row[] rows) throws java.io.IOException {
        digestValue(SCENARIO_TABLE_SHA256, "scenario table SHA256");
        byte[] bytes = java.nio.file.Files.readAllBytes(absolute(SCENARIO_TABLE_PATH, "scenario table path"));
        require(sha256(bytes).equals(SCENARIO_TABLE_SHA256), "scenario table byte SHA256 differs");
        String csv = java.nio.charset.StandardCharsets.US_ASCII.newDecoder()
            .decode(java.nio.ByteBuffer.wrap(bytes)).toString();
        String[] lines = csv.split("\\n", -1);
        require(lines.length == JOB_COUNT + 2 && lines[JOB_COUNT + 1].isEmpty()
            && lines[0].equals(SCENARIO_HEADER), "scenario table header or row count differs");
        java.util.Map<Long, Scenario> scenes = new java.util.HashMap<Long, Scenario>();
        for (int i = 0; i < JOB_COUNT; ++i) {
            String[] values = lines[i + 1].split(",", -1);
            require(values.length == 9, "invalid scenario column count at ordinal " + i);
            long run = positive(values[0], "scenario run_index");
            require(run == rows[i].run, "scenario run_index differs from master at ordinal " + i);
            require(scenes.put(run, new Scenario(values)) == null, "duplicate scenario run_index");
        }
        return java.util.Collections.unmodifiableMap(scenes);
    }

    private static Scenario scenarioForRun(long runIndex) {
        Scenario scene = plan().scenarios.get(runIndex);
        require(scene != null, "run_index absent from frozen scenario table: " + runIndex);
        return scene;
    }

    public static String scenarioId(long runIndex) {
        return scenarioForRun(runIndex).id;
    }

    public static double loadGrade(long runIndex, int siteId) {
        require(siteId >= 0 && siteId < 3, "scenario load site must be 0, 1, or 2");
        return scenarioForRun(runIndex).grades[siteId];
    }

    public static double loadGradeHalfRange(long runIndex, int siteId) {
        require(siteId >= 0 && siteId < 3, "scenario load site must be 0, 1, or 2");
        return scenarioForRun(runIndex).halfRanges[siteId];
    }

    /** All speed limits here and in the table are in m/s. */
    public static double roadSpeedLimit(long runIndex, String from, String to, double nominal) {
        Scenario scene = scenarioForRun(runIndex);
        require(Double.isFinite(nominal) && nominal > 0.0, "invalid nominal speed limit");
        if (@ROAD_FROM@.equals(from) && @ROAD_TO@.equals(to)) {
            return scene.road6Limit;
        }
        return nominal;
    }
'''
    for before, after in replacements.items():
        source = source.replace(before, after)
    return source

def verify_java_scene(helper_path, scenario_table):
    """Compile the exact helper and check all slots without starting AnyLogic."""
    from subprocess import run

    helper_path, scenario_table = Path(helper_path).resolve(), Path(scenario_table).resolve()
    with TemporaryDirectory(prefix='bg128-generalization-java-check-') as directory:
        temp = Path(directory)
        probe = temp / 'SceneProbe.java'
        probe.write_text(_probe_java(), encoding='utf-8')
        run(['javac', '-encoding', 'UTF-8', '-d', str(temp), str(helper_path), str(probe)], check=True)
        result = run(['java', '-cp', str(temp), 'SceneProbe', str(scenario_table)], check=True,
                     capture_output=True, text=True)
        # Exercise the SHA gate against an isolated table copy. The prepared
        # helper and real table remain byte-identical throughout this check.
        tampered = temp / 'tampered_scenarios.csv'
        tampered.write_bytes(scenario_table.read_bytes() + b'\n')
        isolated_helper = temp / 'Baseline1CampaignPlan.java'
        isolated_helper.write_text(_replace_once(
            helper_path.read_text(encoding='utf-8'),
            'private static final String SCENARIO_TABLE_PATH = ' + json.dumps(str(scenario_table)) + ';',
            'private static final String SCENARIO_TABLE_PATH = ' + json.dumps(str(tampered)) + ';',
        ), encoding='utf-8')
        run(['javac', '-encoding', 'UTF-8', '-d', str(temp), str(isolated_helper)], check=True)
        run(['java', '-cp', str(temp), 'SceneProbe', 'expect-sha-failure'], check=True)
    return {'compiled': True, 'rows_checked': int(result.stdout.strip().splitlines()[-1]),
            'paste_fragments_stub_checked': True, 'unknown_run_rejected': True,
            'altered_sidecar_rejected': True, 'simulation_started': False}


def _probe_java():
    fragments = {
        '@LOAD_SITE_BODY@': (PLAN / 'Main.init1.LoadSites.after.java').read_text(encoding='utf-8'),
        '@ROAD_BODY@': (PLAN / 'Main.addBidirectionalPath.after.java').read_text(encoding='utf-8'),
    }
    source = r'''
import java.nio.file.*;
import java.util.*;

public final class SceneProbe {
    private static void require(boolean value, String label) {
        if (!value) throw new AssertionError(label);
    }

    static final class Node {
        double getX() { return 1.0; }
        double getY() { return 2.0; }
    }
    static final class LoadSite {
        double gradeHalfRange = 0.005;
        double effectiveGrade, effectiveHalfRange;
        Node loadNode;
        void setLocation(Node node) { }
        void initLoadSite(int id, String name, boolean ore, double grade,
                          int reserve, double service, double x, double y, double bucket, double density) {
            effectiveGrade = grade;
            effectiveHalfRange = gradeHalfRange;
        }
    }
    static final class PathSegment {
        final String key, directionName = "stub";
        final double limit;
        PathSegment(SceneProbe main, String from, String to, String positiveFrom, String positiveTo, double limit) {
            this.key = from + "_" + to;
            this.limit = limit;
        }
    }

    final long experimentRunIndex;
    final Node ShovelOreHigh1 = new Node(), ShovelOreHigh2 = new Node(), ShovelOreLow = new Node();
    final List<LoadSite> loads = new ArrayList<>();
    final Map<String, PathSegment> segmentMap = new HashMap<>();
    SceneProbe(long experimentRunIndex) { this.experimentRunIndex = experimentRunIndex; }
    LoadSite add_loadSites() { LoadSite site = new LoadSite(); loads.add(site); return site; }
    void traceln(String message) { }
    void createLoads() {
@LOAD_SITE_BODY@
    }
    void addBidirectionalPath(String positiveFrom, String positiveTo, double speedLimit) {
@ROAD_BODY@
    }

    public static void main(String[] args) throws Exception {
        if (args[0].equals("expect-sha-failure")) {
            try {
                Baseline1CampaignPlan.runIndex(0);
                throw new AssertionError("altered table was accepted");
            } catch (IllegalStateException error) {
                require(error.getMessage().contains("scenario table byte SHA256 differs"), "wrong rejection");
            }
            return;
        }
        List<String> rows = Files.readAllLines(Path.of(args[0]));
        require(Baseline1CampaignPlan.jobCount() == rows.size() - 1, "job count");
        double nominal = 30.0 / 3.6;
        for (int i = 1; i < rows.size(); ++i) {
            String[] values = rows.get(i).split(",", -1);
            long run = Long.parseLong(values[0]);
            require(Baseline1CampaignPlan.runIndex(i - 1) == run, "master run order " + i);
            require(Baseline1CampaignPlan.scenarioId(run).equals(values[1]), "scenario ID " + i);
            SceneProbe probe = new SceneProbe(run);
            probe.createLoads();
            probe.addBidirectionalPath("Crusher", "intersection1", nominal);
            probe.addBidirectionalPath("intersection1", "intersection2", 20.0 / 3.6);
            for (int site = 0; site < 3; ++site) {
                LoadSite load = probe.loads.get(site);
                require(load.effectiveGrade == Double.parseDouble(values[site + 2]), "grade " + i);
                require(load.effectiveHalfRange == Double.parseDouble(values[site + 5]), "grade range " + i);
            }
            require(probe.segmentMap.get("intersection1_Crusher").limit == Double.parseDouble(values[8]),
                "changed direction " + i);
            require(probe.segmentMap.get("Crusher_intersection1").limit == nominal, "reverse direction " + i);
            require(probe.segmentMap.get("intersection1_intersection2").limit == 20.0 / 3.6,
                "unaffected segment " + i);
        }
        try {
            Baseline1CampaignPlan.loadGrade(0L, 0);
            throw new AssertionError("unknown run was accepted");
        } catch (IllegalStateException error) {
            require(error.getMessage().contains("run_index absent from frozen scenario table"), "wrong rejection");
        }
        System.out.println(rows.size() - 1);
    }
}
'''
    for marker, fragment in fragments.items():
        source = source.replace(marker, fragment)
    return source
