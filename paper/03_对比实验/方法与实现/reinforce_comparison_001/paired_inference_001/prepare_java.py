"""Bind the tested external-controller Java helper to the seven evaluation phases."""

import csv
from hashlib import sha256
import json
from pathlib import Path
import sys

PLAN = Path(__file__).resolve().parent
TRAIN = PLAN.parent / 'prepared_001'
ROOT = PLAN.parents[4]
MIXED = PLAN.parents[1] / 'generalization_001/grade_assignment_generalization_001/continuous_training_001'
for directory in (ROOT, MIXED):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from baseline1_v1.java_plan import COLUMNS, _replace_once
from business_graph_summary_campaign_v1.plan_io import sha
from mixed_java import scenario_table_bytes, verify_java


def render(suite, jobs, master_path, output_dir):
    """Preserve the external READY/CLAIM handshake and load the frozen paired roster."""
    directory, master = Path(output_dir).resolve(), Path(master_path).resolve()
    table, helper = directory / 'scenarios.csv', directory / 'Baseline1CampaignPlan.java'
    train_helper = TRAIN / helper.name
    if sha(train_helper) != 'a28f19182713f1d925b0095390c0a022f9ee5bfafabebe59239ae7b33813ed6d':
        raise ValueError('previously validated Java helper changed')
    with master.open(encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames) != COLUMNS or list(reader) != [
                {key: str(job[key]) for key in COLUMNS} for job in jobs]:
            raise ValueError('master differs from the evaluation roster')
    if len(jobs) != 700 or not all(
            job['ordinal'] == i and job['role'] == 'final' and job['group'] == 'BASELINE'
            and job['checkpoint_update'] == 32
            and job['recording_id'] == suite['campaign_id'] + '-' + job['phase_id']
            and Path(job['material_event_directory']) == Path(suite['material_root']) / job['phase_id']
            for i, job in enumerate(jobs)):
        raise ValueError('evaluation Java material or physical identity differs')
    data = scenario_table_bytes(jobs)
    previous = json.loads((TRAIN / 'suite.json').read_bytes())
    source = train_helper.read_text(encoding='utf-8')
    for before, after in (
            (str(TRAIN / 'master_plan.csv'), str(master)),
            (previous['master_plan']['sha256'], sha(master)),
            (previous['control_directory'], suite['control_directory']),
            (previous['campaign_id'], suite['campaign_id']),
            (previous['material_root'], suite['material_root']),
            (str(TRAIN / 'scenarios.csv'), str(table)),
            (previous['scenario_table']['sha256'], sha256(data).hexdigest())):
        source = _replace_once(source, json.dumps(before), json.dumps(after))
    source = _replace_once(source, 'private static final int JOB_COUNT = 1024;',
                           'private static final int JOB_COUNT = 700;')
    source = _replace_once(source, 'public static int rpcPort() { return 50051; }',
                           f"public static int rpcPort() {{ return {int(suite['rpc_port'])}; }}")
    with table.open('xb') as stream:
        stream.write(data)
    with helper.open('x', encoding='utf-8') as stream:
        stream.write(source)
    evidence = verify_java(helper, table)
    if evidence['rows_checked'] != len(jobs):
        raise ValueError('AnyLogic scene probe did not verify every paired slot')
    return helper, evidence
