"""Freeze the 400-row paired package. Preparation writes no physical run or ledger."""
import csv
import json

from paired_plan import (CAMPAIGN, COUNT, EXECUTION, LAYOUT, METHOD, OUTPUT, PAIRING, PREPARED, SCHEMA, SLOTS,
                         STRATEGIES, SUITE, config_for, job_for, load_suite, phase_path, reference, require, selection_evidence,
                         source_contract, source_pins)
from paired_budget import budget_snapshot, check_fresh
from business_graph_summary_campaign_v1.plan_io import write
from java_adapter import master_bytes, render, scenario_table_bytes, verify_java


SEED_COLUMNS = ('ordinal', 'pair_ordinal', 'policy_kind', 'run_index', 'recording_id', 'experiment_seed',
                'actor_sampling_seed', 'actor_sampling_seed_consumed', 'config_path', 'config_sha256',
                'python_output', 'anylogic_event_directory')


def launch_row(job):
    row = {key: job[key] for key in SEED_COLUMNS if key in job}
    return dict(row, config_path=job['phase_config']['path'], config_sha256=job['phase_config']['sha256'],
                anylogic_event_directory=job['event_directory'])


def prepare():
    require(not PREPARED.exists() and not OUTPUT.exists(), 'fresh paired preparation and run directory required')
    pins = source_pins()
    refs, learner, rpc, scene = source_contract()
    selection = selection_evidence(refs)
    jobs = [job_for(ordinal, learner, rpc, scene) for ordinal in range(COUNT)]
    readiness = check_fresh({'jobs':jobs}, budget_snapshot())
    PREPARED.mkdir()
    for job in jobs:
        path = phase_path(job)
        write(path, config_for(job, refs))
        job['phase_config'] = reference(path)
    master, scenarios, helper, seeds = (PREPARED/name for name in ('master_plan.csv', 'scenario_table.csv',
                                                                  'Baseline1CampaignPlan.java', 'run_seed.csv'))
    master.write_bytes(master_bytes(jobs))
    scenarios.write_bytes(scenario_table_bytes(jobs))
    helper.write_text(render(jobs, master, scenarios), encoding='utf-8')
    java_check = verify_java(helper, scenarios)
    require(java_check['rows_checked'] == COUNT, 'incomplete generated Java verification')
    with seeds.open('x', encoding='ascii', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=SEED_COLUMNS, lineterminator='\n')
        writer.writeheader()
        writer.writerows(launch_row(job) for job in jobs)

    suite = dict(schema=SCHEMA, campaign_id=CAMPAIGN, **LAYOUT, execution=EXECUTION, method=METHOD, pairing=PAIRING,
                 checkpoint_selection='mixed-s02-u32', checkpoint_selection_evidence=selection,
                 learner_config=learner, **refs, master_plan=reference(master),
                 scenario_table=reference(scenarios), java_helper=reference(helper), run_seed=reference(seeds),
                 source_pins=pins, evaluation_slots=SLOTS, physical_shifts=COUNT, training_shifts=0,
                 optimizer_updates=0, jobs=jobs)
    write(SUITE, suite)
    load_suite()
    identity = dict(status='PREPARED', campaign_id=CAMPAIGN, suite=reference(SUITE), java_helper=suite['java_helper'],
                    run_seed=suite['run_seed'], **LAYOUT, evaluation_slots=SLOTS, physical_shifts=COUNT,
                    training_shifts=0, optimizer_updates=0, budget=readiness, java_verification=java_check,
                    checkpoint_selection=suite['checkpoint_selection'], checkpoint_selection_evidence=selection,
                    checkpoint=refs['checkpoint'],
                    actor=refs['actor'], protected_core=refs['protected_core'],
                    first_slot=[launch_row(job) for job in jobs[:len(STRATEGIES)]],
                    last_slot=[launch_row(job) for job in jobs[-len(STRATEGIES):]])
    write(PREPARED/'launch_identity.json', identity)
    return identity


if __name__ == '__main__':
    print(json.dumps(prepare(), ensure_ascii=False, indent=2))
