"""Build the AnyLogic plan for an externally started training controller."""

import json
import sys
from pathlib import Path

PLAN = Path(__file__).resolve().parent
ROOT = PLAN.parents[3]
MIXED = PLAN.parent / 'generalization_001/grade_assignment_generalization_001/continuous_training_001'
for directory in (ROOT, MIXED):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from baseline1_v1.java_plan import COLUMNS, _replace_once, render_java
from baseline1_v1.prepare import _reference_configs
from business_graph_summary_campaign_v1.plan_io import sha
from mixed_java import _scenario_java, scenario_table_bytes, verify_java
from python.native_behavior import identity


def render(suite, jobs, master_path, output_dir):
    """Bind all slots to the master and scene tables and exercise them in Java."""
    directory = Path(output_dir)
    master = Path(master_path)
    table = directory / 'scenarios.csv'
    helper = directory / 'Baseline1CampaignPlan.java'
    root = Path(suite['material_root'])
    for job in jobs:
        material_directory = root / job['phase_id']
        if (job['role'] != 'train' or job['group'] != 'BASELINE'
                or job['recording_id'] != suite['campaign_id'] + '-' + job['phase_id']
                or Path(job['material_event_directory']) != material_directory):
            raise ValueError('Java material path or training phase differs from the plan')
    source = render_java(
        master_path=str(master), master_sha256=sha(master),
        control_directory=suite['control_directory'], campaign_id=suite['campaign_id'],
        jobs=[{key: job[key] for key in COLUMNS} for job in jobs],
        behavior_digests={'P' + str(k): identity(v['behavior']) for k, v in _reference_configs().items()},
    )
    data = scenario_table_bytes(jobs)
    # The standard Java plan already waits for a sealed READY and claims it.
    # Freeform expressions require access to the mixed scenario and material data.
    accessors = '''
    public static int jobCount() { return JOB_COUNT; }
    public static int rpcPort() { return @PORT@; }
    public static double freeSpeedFactorStd(int index) { row(index); return 0.05; }
    public static boolean materialEnabled(int index) { row(index); return true; }
    public static String materialRoot(int index) {
        return @MATERIALS@ + "/" + recordingId(index).substring(CAMPAIGN_ID.length() + 1);
    }
'''
    source = _replace_once(source, '    private static final class Row {',
                           accessors.replace('@PORT@', str(suite['rpc_port']))
                           .replace('@MATERIALS@', json.dumps(str(root)))
                           + _scenario_java(table, data) + '\n    private static final class Row {')
    source = _replace_once(source,
        '        Plan(Row[] rows, java.util.Map<Long, Row> byRun) { this.rows = rows; this.byRun = byRun; }',
        '''        final java.util.Map<Long, Scenario> scenarios;
        Plan(Row[] rows, java.util.Map<Long, Row> byRun) throws java.io.IOException {
            this.rows = rows;
            this.byRun = byRun;
            this.scenarios = readScenarios(rows);
        }''')
    with table.open('xb') as stream:
        stream.write(data)
    with helper.open('x', encoding='utf-8') as stream:
        stream.write(source)
    evidence = verify_java(helper, table)
    if evidence['rows_checked'] != len(jobs):
        raise ValueError('Java probe did not verify every training slot')
    return helper, evidence
