"""Rebind the tested Java controller handshake to the 658 recovery slots."""

import csv
import json
from pathlib import Path

from . import prepare as plan
from baseline1_v1.java_plan import COLUMNS, _replace_once
from business_graph_summary_campaign_v1.plan_io import reference, sha
from mixed_java import scenario_table_bytes, verify_java


def render(suite, jobs, master_path, output_dir):
    directory, master = Path(output_dir).resolve(), Path(master_path).resolve()
    table, helper = directory / 'scenarios.csv', directory / 'Baseline1CampaignPlan.java'
    previous, _ = plan.original_package()
    old_helper = Path(previous['java_helper']['path'])
    plan.require(reference(old_helper) == previous['java_helper'], 'original java helper changed')
    with master.open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        plan.require(tuple(reader.fieldnames) == COLUMNS and list(reader) == [
            {key: str(job[key]) for key in COLUMNS} for job in jobs], 'master file differs from the recovered slots')
    plan.require(len(jobs) == plan.COUNT and all(
        job['ordinal'] == i and job['role'] == 'final' and job['group'] == 'BASELINE'
        and job['checkpoint_update'] == 32 and job['recording_id'] == suite['campaign_id'] + '-' + job['phase_id']
        and Path(job['material_event_directory']) == Path(suite['material_root']) / job['phase_id']
        for i, job in enumerate(jobs)), 'Java recovery material or physical identity differs')
    with table.open('xb') as stream:
        stream.write(scenario_table_bytes(jobs))
    source = old_helper.read_text(encoding='utf-8')
    for before, after in ((previous['master_plan']['path'], str(master)),
                          (previous['master_plan']['sha256'], sha(master)),
                          (previous['control_directory'], suite['control_directory']),
                          (previous['campaign_id'], suite['campaign_id']),
                          (previous['material_root'], suite['material_root']),
                          (previous['scenario_table']['path'], str(table)),
                          (previous['scenario_table']['sha256'], sha(table))):
        source = _replace_once(source, json.dumps(before), json.dumps(after))
    source = _replace_once(source, 'private static final int JOB_COUNT = 700;',
                           f'private static final int JOB_COUNT = {len(jobs)};')
    with helper.open('x', encoding='utf-8') as stream:
        stream.write(source)
    evidence = verify_java(helper, table)
    plan.require(evidence['rows_checked'] == len(jobs), 'Java scene probe completeness differs')
    return helper, evidence
