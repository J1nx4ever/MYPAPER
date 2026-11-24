"""Generate the four-site AnyLogic scenario helper and verify it offline."""

import csv
import hashlib
from importlib.util import module_from_spec, spec_from_file_location
import io
import json
from math import isfinite
from pathlib import Path
import re
import subprocess
from tempfile import TemporaryDirectory

from baseline1_v1.java_plan import _replace_once, _require
from business_graph_summary_campaign_v1.plan_io import sha


PLAN = Path(__file__).resolve().parent
SITE_COUNT = 4
RPC_PREFIX = 'config.runtime_scene.'
SCENARIO_FIELDS = (
    'run_index', 'scenario_id',
    *(f'load{site}_is_ore' for site in range(SITE_COUNT)),
    *(f'load{site}_grade' for site in range(SITE_COUNT)),
    *(f'load{site}_grade_half_range' for site in range(SITE_COUNT)),
    'road_6_speed_limit_mps',
)


def scenario_table_bytes(jobs):
    """Use the Python authority for every AnyLogic scene value."""
    stream = io.StringIO(newline='')
    writer = csv.writer(stream, lineterminator='\n')
    writer.writerow(SCENARIO_FIELDS)
    for job in jobs:
        run = job['run_index']
        scene_id = job['scene_id']
        _require(type(run) is int and run > 0, 'invalid scenario run_index')
        _require(isinstance(scene_id, str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', scene_id),
                 'unsafe scenario ID')
        rpc = job['expected_rpc_config']
        roles = tuple(rpc[f'{RPC_PREFIX}load.{site}.isOre'] for site in range(SITE_COUNT))
        grades = tuple(float(rpc[f'{RPC_PREFIX}load.{site}.grade']) for site in range(SITE_COUNT))
        ranges = tuple(float(rpc[f'{RPC_PREFIX}load.{site}.gradeHalfRange']) for site in range(SITE_COUNT))
        limit = float(rpc[f'{RPC_PREFIX}road.6.speedLimit_mps'])
        _require(all(role in (0.0, 1.0) for role in roles) and sum(roles) == 3,
                 'exactly three ore sources and one waste source required')
        for ore, grade, half_range in zip(roles, grades, ranges, strict=True):
            _require(isfinite(grade) and isfinite(half_range) and 0 <= half_range <= 1,
                     'invalid source grade or half-range')
            _require(half_range <= grade <= 1 - half_range if ore else grade == 0,
                     'ore support must stay in [0, 1] and waste grade must be zero')
        _require(isfinite(limit) and limit > 0, 'invalid road speed limit')
        writer.writerow((run, scene_id, *map(int, roles), *grades, *ranges, limit))
    return stream.getvalue().encode('ascii')


def _base_renderer():
    path = PLAN.parents[2] / 'formal_training_001/prepare_training.py'
    spec = spec_from_file_location('continuous_bg128_base_java', path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_java(suite, jobs, master_path, scenario_table_path, service_dir, *,
               suite_path=None, bridge_path=None):
    """Write the sealed sidecar and helper for a new campaign."""
    jobs = list(jobs)
    master_path = Path(master_path).resolve()
    scenario_table_path = Path(scenario_table_path).resolve()
    helper_path = Path(service_dir).resolve() / 'Baseline1CampaignPlan.java'
    suite_path = Path(suite_path or suite['suite_path']).resolve()
    bridge_path = Path(bridge_path or PLAN / 'mixed_bootstrap.py').resolve()
    _require(bool(jobs) and [job['ordinal'] for job in jobs] == list(range(len(jobs)))
             and len({job['run_index'] for job in jobs}) == len(jobs)
             and len({job['event_bundle'] for job in jobs}) == len(jobs),
             'each Java master row needs a distinct ordered physical identity')
    _require(not scenario_table_path.exists() and not helper_path.exists(),
             'Java helper or sidecar already exists')
    base = _base_renderer()
    _require(master_path.read_bytes() == base.master_bytes(jobs),
             'Java master differs from the supplied jobs')
    first_phase = [job for job in jobs if job['recording_id'] == jobs[0]['recording_id']]
    _require(first_phase == jobs[:len(first_phase)], 'first Java phase must be contiguous')
    data = scenario_table_bytes(jobs)

    # As in the original generalization helper, seed uniqueness is checked on
    # the leading phase by the base renderer. The campaign planner independently
    # validates the sole exception for paired evaluation in the full master.
    with TemporaryDirectory(prefix='bg128-continuous-java-') as folder:
        projection = Path(folder) / 'first_phase.csv'
        projection.write_bytes(base.master_bytes(first_phase))
        source = base.render_java(dict(suite, jobs=first_phase), projection,
                                  suite_path=suite_path, bridge_path=bridge_path)
        source = _replace_once(
            source, 'private static final String MASTER_PLAN_PATH = ' + json.dumps(str(projection)) + ';',
            'private static final String MASTER_PLAN_PATH = ' + json.dumps(str(master_path)) + ';')
        source = _replace_once(
            source, 'private static final String MASTER_PLAN_SHA256 = ' + json.dumps(sha(projection)) + ';',
            'private static final String MASTER_PLAN_SHA256 = ' + json.dumps(sha(master_path)) + ';')
    source = _replace_once(source, f'private static final int JOB_COUNT = {len(first_phase)};',
                           f'private static final int JOB_COUNT = {len(jobs)};')
    anchor = '    private static final class Row {'
    source = _replace_once(source, anchor, _scenario_java(scenario_table_path, data) + '\n' + anchor)
    source = _replace_once(
        source,
        '        Plan(Row[] rows, java.util.Map<Long, Row> byRun) { this.rows = rows; this.byRun = byRun; }',
        '''        final java.util.Map<Long, Scenario> scenarios;
        Plan(Row[] rows, java.util.Map<Long, Row> byRun) throws java.io.IOException {
            this.rows = rows;
            this.byRun = byRun;
            this.scenarios = readScenarios(rows);
        }''')
    scenario_table_path.parent.mkdir(parents=True, exist_ok=True)
    helper_path.parent.mkdir(parents=True, exist_ok=True)
    with scenario_table_path.open('xb') as stream:
        stream.write(data)
    with helper_path.open('x', encoding='utf-8') as stream:
        stream.write(source)
    return helper_path


def _scenario_java(path, data):
    source = r'''
    private static final int SCENE_LOAD_SITES = 4;
    private static final String SCENARIO_TABLE_PATH = @SCENARIO_TABLE@;
    private static final String SCENARIO_TABLE_SHA256 = @SCENARIO_SHA@;
    private static final String SCENARIO_HEADER = @SCENARIO_HEADER@;

    private static final class Scenario {
        final String id;
        final boolean[] ore = new boolean[SCENE_LOAD_SITES];
        final double[] grades = new double[SCENE_LOAD_SITES];
        final double[] halfRanges = new double[SCENE_LOAD_SITES];
        final double road6Limit;

        Scenario(String[] values) {
            id = values[1];
            require(id.matches("[A-Za-z0-9][A-Za-z0-9._-]*"), "invalid scenario ID");
            int oreSites = 0;
            for (int site = 0; site < SCENE_LOAD_SITES; ++site) {
                String role = values[2 + site];
                require(role.equals("0") || role.equals("1"), "invalid source role");
                ore[site] = role.equals("1");
                if (ore[site]) ++oreSites;
                grades[site] = scenarioNumber(values[2 + SCENE_LOAD_SITES + site]);
                halfRanges[site] = scenarioNumber(values[2 + 2 * SCENE_LOAD_SITES + site]);
                require(halfRanges[site] >= 0.0 && halfRanges[site] <= 1.0,
                    "invalid grade half-range");
                if (ore[site]) {
                    require(grades[site] >= halfRanges[site] && grades[site] <= 1.0 - halfRanges[site],
                        "ore grade support outside [0,1]");
                } else {
                    require(grades[site] == 0.0, "waste source grade must be zero");
                }
            }
            require(oreSites == 3, "exactly three ore sources required");
            road6Limit = scenarioNumber(values[2 + 3 * SCENE_LOAD_SITES]);
            require(road6Limit > 0.0, "invalid scenario road speed");
        }
    }

    private static double scenarioNumber(String text) {
        require(text.matches("[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?"), "invalid scenario numeric syntax");
        double value = Double.parseDouble(text);
        require(Double.isFinite(value), "nonfinite scenario number");
        return value;
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
        java.util.Map<Long, Scenario> scenes = new java.util.HashMap<>();
        for (int i = 0; i < JOB_COUNT; ++i) {
            String[] values = lines[i + 1].split(",", -1);
            require(values.length == 3 + 3 * SCENE_LOAD_SITES, "invalid scenario column count at ordinal " + i);
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

    private static void requireSceneSite(int siteId) {
        require(siteId >= 0 && siteId < SCENE_LOAD_SITES, "scenario load site must be in [0,3]");
    }

    public static String scenarioId(long runIndex) { return scenarioForRun(runIndex).id; }

    public static boolean loadIsOre(long runIndex, int siteId) {
        requireSceneSite(siteId);
        return scenarioForRun(runIndex).ore[siteId];
    }

    public static double loadGrade(long runIndex, int siteId) {
        requireSceneSite(siteId);
        return scenarioForRun(runIndex).grades[siteId];
    }

    public static double loadGradeHalfRange(long runIndex, int siteId) {
        requireSceneSite(siteId);
        return scenarioForRun(runIndex).halfRanges[siteId];
    }

    /** The scene and this method both use m/s. */
    public static double roadSpeedLimit(long runIndex, String from, String to, double nominal) {
        Scenario scene = scenarioForRun(runIndex);
        require(Double.isFinite(nominal) && nominal > 0.0, "invalid nominal speed limit");
        return "intersection1".equals(from) && "Crusher".equals(to) ? scene.road6Limit : nominal;
    }
'''
    return (source.replace('@SCENARIO_TABLE@', json.dumps(str(path)))
            .replace('@SCENARIO_SHA@', json.dumps(hashlib.sha256(data).hexdigest()))
            .replace('@SCENARIO_HEADER@', json.dumps(','.join(SCENARIO_FIELDS))))


def verify_java(helper_path, scenario_table_path):
    """Check generated Java and every scenario row without starting AnyLogic or the bridge."""
    helper_path = Path(helper_path).resolve()
    scenario_table_path = Path(scenario_table_path).resolve()
    with TemporaryDirectory(prefix='bg128-continuous-java-check-') as folder:
        temp = Path(folder)
        probe = temp / 'SceneProbe.java'
        probe.write_text(_probe_java(), encoding='utf-8')
        command = ['javac', '-encoding', 'UTF-8', '-d', str(temp)]
        subprocess.run([*command, str(helper_path), str(probe)], check=True)
        java = ['java', '-cp', str(temp), 'SceneProbe']
        result = subprocess.run([*java, str(scenario_table_path)], check=True, capture_output=True, text=True)
        count = int(result.stdout.strip())
        changed_table = temp / 'tampered.csv'
        changed_table.write_bytes(scenario_table_path.read_bytes() + b'\n')
        changed_helper = temp / 'Baseline1CampaignPlan.java'
        changed_helper.write_text(_replace_once(
            helper_path.read_text(encoding='utf-8'),
            'private static final String SCENARIO_TABLE_PATH = ' + json.dumps(str(scenario_table_path)) + ';',
            'private static final String SCENARIO_TABLE_PATH = ' + json.dumps(str(changed_table)) + ';'),
            encoding='utf-8')
        subprocess.run([*command, str(changed_helper), str(probe)], check=True)
        subprocess.run([*java, 'expect-sha-failure'], check=True)
    return dict(compiled=True, rows_checked=count, source_roles_and_grades_checked=True,
                paste_fragment_stub_checked=True, unknown_run_rejected=True,
                altered_sidecar_rejected=True, simulation_started=False)


def _probe_java():
    fragment = (PLAN / 'Main.init1.LoadSites.after.java').read_text(encoding='utf-8')
    source = r'''
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public final class SceneProbe {
    private static void require(boolean value, String label) {
        if (!value) throw new AssertionError(label);
    }

    static final class Node {
        final int index;
        Node(int index) { this.index = index; }
        double getX() { return index; }
        double getY() { return 2.0 * index; }
    }

    static final class LoadSite {
        double gradeHalfRange, effectiveHalfRange, grade, bucket, density;
        boolean ore;
        int id;
        Node loadNode, location;
        void setLocation(Node node) { location = node; }
        void initLoadSite(int id, String name, boolean ore, double grade,
                          int reserve, double service, double x, double y, double bucket, double density) {
            this.id = id;
            this.ore = ore;
            this.grade = grade;
            this.bucket = bucket;
            this.density = density;
            effectiveHalfRange = gradeHalfRange;
            require(reserve == 30000 && service == 0.45, "station operating parameters");
            require(loadNode == location && loadNode.index == id && x == location.getX()
                    && y == location.getY(), "station physical identity");
        }
    }

    final long experimentRunIndex;
    final Node ShovelOreHigh1 = new Node(0), ShovelOreHigh2 = new Node(1);
    final Node ShovelOreLow = new Node(2), ShovelWaste = new Node(3);
    final List<LoadSite> loads = new ArrayList<>();

    SceneProbe(long run) { experimentRunIndex = run; }
    LoadSite add_loadSites() {
        LoadSite site = new LoadSite();
        loads.add(site);
        return site;
    }
    void createLoads() {
@LOAD_SITES@
    }

    public static void main(String[] args) throws Exception {
        if (args[0].equals("expect-sha-failure")) {
            try {
                Baseline1CampaignPlan.runIndex(0);
                throw new AssertionError("altered table was accepted");
            } catch (IllegalStateException error) {
                require(error.getMessage().contains("scenario table byte SHA256 differs"), "wrong SHA rejection");
            }
            return;
        }
        List<String> rows = Files.readAllLines(Path.of(args[0]));
        require(Baseline1CampaignPlan.jobCount() == rows.size() - 1, "master row count");
        double nominal = 30.0 / 3.6;
        for (int i = 1; i < rows.size(); ++i) {
            String[] values = rows.get(i).split(",", -1);
            long run = Long.parseLong(values[0]);
            require(Baseline1CampaignPlan.runIndex(i - 1) == run, "master order");
            require(Baseline1CampaignPlan.scenarioId(run).equals(values[1]), "scenario ID");
            SceneProbe probe = new SceneProbe(run);
            probe.createLoads();
            require(probe.loads.size() == 4, "four physical stations");
            for (int site = 0; site < 4; ++site) {
                LoadSite load = probe.loads.get(site);
                require(load.id == site && load.ore == values[2 + site].equals("1"), "source role");
                require(load.grade == Double.parseDouble(values[6 + site]), "source grade");
                require(load.effectiveHalfRange == Double.parseDouble(values[10 + site]), "source half-range");
                require(load.bucket == (site == 3 ? 15.0 : 13.5)
                        && load.density == (site == 3 ? 2.0 : 2.2), "physical shovel parameters");
            }
            require(Baseline1CampaignPlan.roadSpeedLimit(run, "intersection1", "Crusher", nominal)
                    == Double.parseDouble(values[14]), "scenario road value");
            require(Baseline1CampaignPlan.roadSpeedLimit(run, "Crusher", "intersection1", nominal)
                    == nominal, "reverse travel unchanged");
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
    return source.replace('@LOAD_SITES@', fragment)
