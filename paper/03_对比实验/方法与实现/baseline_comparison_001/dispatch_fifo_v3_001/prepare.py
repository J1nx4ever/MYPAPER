"""Freeze the 100 matched-seed FIFO DISPATCH evaluations without registering them."""
import csv
import io
import json
from pathlib import Path

from v3_plan import (CAMPAIGN, COUNT, EXECUTION, LAYOUT, METHOD, OUTPUT, PAIRING, PREPARED, SCHEMA, SLOTS, SUITE,
                     algorithm_contract, config_for, job_for, load_suite, phase_path, previous, reference, require,
                     source_contract, source_pins)
from v3_ledger import check_fresh, preceding_snapshot
from business_graph_summary_campaign_v1.plan_io import write
from v3_java import master_bytes, render, scenario_table_bytes, verify_java


IDENTITY_COLUMNS = ('ordinal','pair_ordinal','policy_kind','run_index','recording_id','experiment_seed',
                    'actor_sampling_seed','actor_sampling_seed_consumed','config_path','config_sha256',
                    'python_output','anylogic_event_directory','v2_reference_run_index','v2_recording_id','v2_config_sha256')


def identity_row(job):
    return dict({key: job[key] for key in IDENTITY_COLUMNS if key in job},
                config_path=job['phase_config']['path'], config_sha256=job['phase_config']['sha256'],
                anylogic_event_directory=job['event_directory'], v2_config_sha256=job['v2_phase_config']['sha256'])


def seed_bytes(rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=IDENTITY_COLUMNS, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode('ascii')


def verify_launch_identity(suite):
    card = json.loads((PREPARED/'launch_identity.json').read_bytes())
    jobs = suite['jobs']
    rows = [identity_row(job) for job in jobs]
    require(card['status'] == 'PREPARED' and card['campaign_id'] == CAMPAIGN
            and card['suite'] == reference(SUITE) and card['jobs'] == rows
            and all(card[key] == suite[key] for key in (*LAYOUT, 'physical_shifts', 'evaluation_slots',
                    'training_shifts', 'optimizer_updates', 'method', 'protected_core', 'java_helper',
                    'run_seed', 'previous_formal_suite')),
            'launch identity card differs from the frozen roster')
    require(Path(suite['run_seed']['path']).read_bytes() == seed_bytes(rows)
            and Path(suite['master_plan']['path']).read_bytes() == master_bytes(jobs)
            and Path(suite['scenario_table']['path']).read_bytes() == scenario_table_bytes(jobs),
            'launch table does not reproduce the 100 frozen jobs')


def prepare():
    require(not PREPARED.exists() and not OUTPUT.exists(), 'fresh FIFO DISPATCH preparation and output directories required')
    formal, anchors = source_contract()
    pins, contract = source_pins(), algorithm_contract()
    protected, previous_suite = formal['protected_core'], reference(previous.SUITE)
    jobs = [job_for(slot,anchor) for slot,anchor in enumerate(anchors)]
    readiness = check_fresh({'jobs':jobs}, preceding_snapshot())
    PREPARED.mkdir()
    for job in jobs:
        path = phase_path(job)
        write(path, config_for(job, protected, contract))
        job['phase_config'] = reference(path)
    master, scenarios, helper, seeds = (PREPARED/name for name in ('master_plan.csv','scenario_table.csv',
                                                                  'Baseline1CampaignPlan.java','run_seed.csv'))
    master.write_bytes(master_bytes(jobs))
    scenarios.write_bytes(scenario_table_bytes(jobs))
    helper.write_text(render(jobs, master, scenarios), encoding='utf-8')
    java = verify_java(helper, scenarios, jobs)
    require(java['rows_checked'] == COUNT and java['paired_environment_rows_checked'] == COUNT,
            'incomplete generated Java verification')
    rows = [identity_row(job) for job in jobs]
    seeds.write_bytes(seed_bytes(rows))
    suite = dict(schema=SCHEMA, campaign_id=CAMPAIGN, **LAYOUT, execution=EXECUTION, method=METHOD, pairing=PAIRING,
                 **contract, protected_core=protected, previous_formal_suite=previous_suite,
                 master_plan=reference(master), scenario_table=reference(scenarios), java_helper=reference(helper),
                 run_seed=reference(seeds), source_pins=pins, evaluation_slots=SLOTS, physical_shifts=COUNT,
                 training_shifts=0, optimizer_updates=0, jobs=jobs)
    write(SUITE,suite)
    load_suite()
    identity = dict(status='PREPARED', campaign_id=CAMPAIGN, suite=reference(SUITE), java_helper=suite['java_helper'],
                    run_seed=suite['run_seed'], **LAYOUT, jobs=rows, physical_shifts=COUNT, budget=readiness,
                    evaluation_slots=SLOTS, training_shifts=0, optimizer_updates=0, java_verification=java,
                    method=METHOD, previous_formal_suite=previous_suite, protected_core=protected,
                    start_condition='sealed_formal_v2_completion_and_100_dispatch_receipts')
    write(PREPARED/'launch_identity.json',identity)
    verify_launch_identity(suite)
    return identity


if __name__ == '__main__':
    print(json.dumps(prepare(), ensure_ascii=False, indent=2))
