"""Render the three-arm 15-shift engineering helper and its fixed one-click bridge."""
import csv
import io
import json
import tempfile
from pathlib import Path
from baseline1_v1.java_plan import COLUMNS, render_java, _replace_once
from baseline1_v1.prepare import _reference_configs
from python.native_behavior import identity
from .gate import check_jobs
from .plan_io import sha

def master_bytes(jobs):
    stream=io.StringIO(newline='')
    writer=csv.DictWriter(stream,fieldnames=COLUMNS,lineterminator='\n')
    writer.writeheader();writer.writerows({k:j[k] for k in COLUMNS} for j in jobs)
    return stream.getvalue().encode()

def _with_startup(source, *, suite_path, bridge_path):
    """Add the bridge only to Main startup; Freeform accessors stay read-only."""
    bridge_path=Path(bridge_path).resolve()
    extra='''
    private static boolean controllerStarted = false;
    /** Only Main startup invokes this; all Freeform accessors remain read-only. */
    private static synchronized void ensureController(Row row) {
        if (controllerStarted) return;
        require(row.ordinal == 0, "one-click startup requires index 0; recovery must be reviewed");
        checkStop(absolute(CONTROL_DIRECTORY, "control directory"));
        try {
            java.nio.file.Path bridge = absolute(@BRIDGE@, "bridge path");
            require(sha256(java.nio.file.Files.readAllBytes(bridge)).equals(@BRIDGESHA@),
                "one-click bridge byte SHA256 differs");
            Process process = new ProcessBuilder(
                "/home/jinx/miniconda3/envs/anylogic-env/bin/python", "-B", bridge.toString(),
                "--suite", @SUITE@, "--run-index", Long.toString(row.run),
                "--environment-seed", Long.toString(row.seed), "--recording-id", row.cells[5],
                "--from-anylogic-run")
                .directory(new java.io.File(@ROOT@)).inheritIO().start();
            require(process.waitFor(15L, java.util.concurrent.TimeUnit.MINUTES),
                "controller startup timed out; inspect launch record before any retry");
            require(process.exitValue() == 0,
                "controller startup failed; inspect AnyLogic console; do not resubmit");
            controllerStarted = true;
        } catch (java.io.IOException error) {
            throw new IllegalStateException("Baseline 1 one-click STOP: cannot start fixed Python bridge", error);
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("Baseline 1 one-click STOP: controller startup interrupted", error);
        }
    }
'''
    root=Path(__file__).resolve().parents[1]
    for token,value in {'@BRIDGE@':str(bridge_path),'@BRIDGESHA@':sha(bridge_path),
                        '@SUITE@':str(Path(suite_path).resolve()),'@ROOT@':str(root)}.items():
        extra=extra.replace(token,json.dumps(value))
    anchor='    private Baseline1CampaignPlan() { }'
    source=_replace_once(source,anchor,anchor+'\n'+extra)
    anchor='        java.nio.file.Path ready = control.resolve("ready").resolve("run_" + runIndex + ".properties");'
    return _replace_once(source,anchor,'        ensureController(row);\n'+anchor)


def render(suite,master,*,suite_path=None,bridge_path=None):
    check_jobs(suite)
    # The base renderer checks contiguous ordinals and unique seeds. Render
    # the leading phase only; later arms intentionally share paired seeds.
    # The resulting Java reader is then pinned to the complete frozen CSV.
    first=[j for j in suite['jobs'] if j['phase_id']==suite['jobs'][0]['phase_id']]
    digests={'P'+str(k):identity(v['behavior']) for k,v in _reference_configs().items()}
    with tempfile.TemporaryDirectory(prefix='single-critic-task-query-java-') as folder:
        projection=Path(folder)/'master.csv';projection.write_bytes(master_bytes(first))
        source=render_java(master_path=str(projection),master_sha256=sha(projection),
            control_directory=suite['control_directory'],campaign_id=suite['campaign_id'],
            jobs=[{k:j[k] for k in COLUMNS} for j in first],behavior_digests=digests)
        source=_replace_once(source,'private static final String MASTER_PLAN_PATH = '+json.dumps(str(projection))+';',
            'private static final String MASTER_PLAN_PATH = '+json.dumps(str(master))+';')
        source=_replace_once(source,'private static final String MASTER_PLAN_SHA256 = '+json.dumps(sha(projection))+';',
            'private static final String MASTER_PLAN_SHA256 = '+json.dumps(sha(master))+';')
    source=_replace_once(source,f'private static final int JOB_COUNT = {len(first)};',
                         f"private static final int JOB_COUNT = {len(suite['jobs'])};")
    anchor='    public static String eventRoot(int index) { return row(index).cells[9]; }'
    material=str(Path(suite['python_output']).with_name(suite['campaign_id']+'_materials'))
    extra='''
    public static boolean materialEnabled(int index) { row(index); return true; }
    public static String phaseId(int index) {
        String recording = row(index).cells[5];
        require(recording.startsWith(CAMPAIGN_ID + "-"), "ablation recording prefix differs");
        return recording.substring(CAMPAIGN_ID.length() + 1);
    }
    public static String materialRoot(int index) { return MATERIAL_ROOT + "/" + phaseId(index); }
    public static String stageId(int index) { return phaseId(index).split("_", 3)[1]; }
    /** Metadata compatibility only. Python uses fixed combined costs. */
    public static double waitWeight(int index) { row(index); return 0.2; }
    public static long actorSamplingSeed(int index) { return positive(row(index).cells[8], "actor seed"); }
    public static boolean preparationOnly() { return false; }
    public static int jobCount() { return JOB_COUNT; }
    public static int resumeOffset() { return 0; }
    public static int frozenPlanJobCount() { return JOB_COUNT; }
'''.replace('MATERIAL_ROOT',json.dumps(material))
    source=_replace_once(source,anchor,anchor+extra)
    if (suite_path is None)!=(bridge_path is None):
        raise ValueError('suite and bridge paths must be supplied together')
    if suite_path is not None:
        source=_with_startup(source,suite_path=suite_path,bridge_path=bridge_path)
    return source
