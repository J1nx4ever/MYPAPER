"""Render the unchanged Java ready/claim protocol for the scoped four-row pairing."""
import csv
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from smoke_plan import CAMPAIGN, CONTROL, STRATEGIES, COLUMNS, reference, require, validate_jobs

import ablation_plan  # Establishes the reviewed mixed-Java module path.
from baseline1_v1.java_plan import NATIVE_GROUPS, _replace_once, render_java
from mixed_java import _scenario_java, scenario_table_bytes, verify_java
from single_critic_task_query_v1.java_plan import _with_startup


def master_bytes(jobs):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator='\n')
    writer.writeheader()
    writer.writerows({key: job[key] for key in COLUMNS} for job in jobs)
    data = stream.getvalue().encode('ascii')
    require(b'"' not in data and b'\r' not in data, 'unsupported master CSV quoting or newlines')
    return data


def render(jobs, master, scenarios, *, suite_path, bridge_path):
    validate_jobs(jobs)
    require(master.read_bytes() == master_bytes(jobs) and scenarios.read_bytes() == scenario_table_bytes(jobs),
            'sealed Java master or scene table differs from the four validated rows')

    # Obtain the audited generic reader from a one-row projection. Only the
    # smoke helper then permits zero unused seeds and its one shared environment.
    projection_job = {key: jobs[0][key] for key in COLUMNS}
    projection_job.update(actor_seed=jobs[-1]['actor_seed'], actor_sampling_seed=jobs[-1]['actor_sampling_seed'])
    with TemporaryDirectory(prefix='bg128-smoke-java-') as folder:
        projection = Path(folder) / 'master.csv'
        projection.write_bytes(master_bytes([projection_job]))
        preview_ref = reference(projection)
        frozen_ref = reference(master)
        source = render_java(master_path=preview_ref['path'], master_sha256=preview_ref['sha256'],
                             control_directory=str(CONTROL), campaign_id=CAMPAIGN, jobs=[projection_job],
                             behavior_digests={key: '1' * 64 for key in NATIVE_GROUPS})
        for name, field in [('MASTER_PLAN_PATH', 'path'), ('MASTER_PLAN_SHA256', 'sha256')]:
            source = _replace_once(source, f'private static final String {name} = {json.dumps(preview_ref[field])};',
                                   f'private static final String {name} = {json.dumps(frozen_ref[field])};')
    source = _replace_once(source, 'private static final int JOB_COUNT = 1;',
                           f'private static final int JOB_COUNT = {len(STRATEGIES)};')
    source = _replace_once(source, '            positive(values[1], "actor_seed");',
                           '''            require(values[2].equals("BASELINE") && values[3].equals("final"),
                "smoke rows all use the final RPC policy");
            if (ordinal == JOB_COUNT - 1) positive(values[1], "actor_seed");
            else require(values[1].equals("0"), "classical policy cannot have an Actor initialization");''')
    source = _replace_once(source, '            positive(values[8], "actor_sampling_seed");',
                           '''            if (ordinal == JOB_COUNT - 1) positive(values[8], "actor_sampling_seed");
            else require(values[8].equals("0"), "classical policy must not consume a sampling seed");''')
    source = _replace_once(source, '                    rows[i] = row;',
                           '''                    require(i == 0 || row.seed == rows[0].seed,
                        "smoke rows must share this campaign's fresh environment seed");
                    rows[i] = row;''')
    anchor = '    private static final class Row {'
    source = _replace_once(source, anchor, _scenario_java(scenarios, scenarios.read_bytes()) + '\n' + anchor)
    source = _replace_once(source,
        '        Plan(Row[] rows, java.util.Map<Long, Row> byRun) { this.rows = rows; this.byRun = byRun; }',
        '''        final java.util.Map<Long, Scenario> scenarios;
        Plan(Row[] rows, java.util.Map<Long, Row> byRun) throws java.io.IOException {
            this.rows = rows;
            this.byRun = byRun;
            this.scenarios = readScenarios(rows);
        }''')
    start = source.index('    public static int dispatchPolicy(int index) {')
    end = source.index('    /** Called once at the very beginning of Main On startup', start)
    source = source[:start] + '''    public static int dispatchPolicy(int index) { row(index); return 3; }
    public static boolean materialEnabled(int index) { row(index); return false; }
    public static boolean eventRecordingEnabled(int index) { row(index); return false; }
    public static boolean offlineEnabled(int index) { row(index); return false; }
    public static String offlineSha256(int index) { row(index); return "@ZERO@"; }
    public static String materialRoot(int index) { return eventRoot(index); }
    public static String phaseId(int index) { return recordingId(index).substring(CAMPAIGN_ID.length() + 1); }
    public static String stageId(int index) { return phaseId(index); }
    public static long actorSamplingSeed(int index) { return Long.parseLong(row(index).cells[8]); }
    public static double waitWeight(int index) { row(index); return 0.2; }
    public static double freeSpeedFactorStd(int index) { row(index); return @SPEED_STD@; }
    public static boolean preparationOnly() { return false; }
    public static int rpcPort() { return 50051; }
    public static int jobCount() { return JOB_COUNT; }
    public static int frozenPlanJobCount() { return JOB_COUNT; }
    public static int resumeOffset() { return 0; }

'''.replace('@ZERO@', '0'*64).replace('@SPEED_STD@', str(jobs[0]['expected_rpc_config']['config.runtime_scene.env.freeSpeedFactorStd'])) + source[end:]
    return _with_startup(source, suite_path=suite_path, bridge_path=bridge_path)
