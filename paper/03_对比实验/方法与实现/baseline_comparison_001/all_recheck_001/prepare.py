"""Seal the new five-row package without creating physical run directories."""
import csv
import json

from recheck_plan import (CAMPAIGN, COUNT, EXECUTION, LAYOUT, METHOD, OUTPUT, PAIRING, PREPARED, SCHEMA, SUITE,
                        config_for, job_for, load_suite, phase_path, reference, require, source_contract, source_pins)
from recheck_budget import budget_snapshot, check_fresh
from business_graph_summary_campaign_v1.plan_io import write
from java_adapter import master_bytes, render, scenario_table_bytes, verify_java


IDENTITY_COLUMNS = ('ordinal', 'pair_ordinal', 'policy_kind', 'run_index', 'recording_id', 'experiment_seed',
                    'actor_sampling_seed', 'actor_sampling_seed_consumed', 'config_path', 'config_sha256',
                    'python_output', 'anylogic_event_directory')


def identity_row(job):
    return dict({key: job[key] for key in IDENTITY_COLUMNS if key in job}, config_path=job['phase_config']['path'],
                config_sha256=job['phase_config']['sha256'], anylogic_event_directory=job['event_directory'])


def prepare():
    require(not PREPARED.exists() and not OUTPUT.exists(), 'fresh revised preparation and output directories required')
    pins = source_pins()
    refs, learner, rpc, scene = source_contract()
    jobs = [job_for(i, learner, rpc, scene) for i in range(COUNT)]
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
    java = verify_java(helper, scenarios)
    require(java['rows_checked'] == COUNT, 'incomplete generated Java verification')
    rows = [identity_row(job) for job in jobs]
    with seeds.open('x', encoding='ascii', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=IDENTITY_COLUMNS, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    suite = dict(schema=SCHEMA, campaign_id=CAMPAIGN, **LAYOUT, execution=EXECUTION, method=METHOD, pairing=PAIRING,
                 learner_config=learner, **refs, master_plan=reference(master), scenario_table=reference(scenarios),
                 java_helper=reference(helper), run_seed=reference(seeds), source_pins=pins, evaluation_slots=1,
                 physical_shifts=COUNT, training_shifts=0, optimizer_updates=0, jobs=jobs)
    write(SUITE, suite)
    load_suite()
    identity = dict(status='PREPARED', campaign_id=CAMPAIGN, suite=reference(SUITE), java_helper=suite['java_helper'],
                    run_seed=suite['run_seed'], **LAYOUT, jobs=rows, physical_shifts=COUNT, budget=readiness,
                    evaluation_slots=1, training_shifts=0, optimizer_updates=0, java_verification=java,
                    method=METHOD, protected_core=refs['protected_core'], checkpoint=refs['checkpoint'])
    write(PREPARED/'launch_identity.json', identity)
    return identity


if __name__ == '__main__':
    print(json.dumps(prepare(), ensure_ascii=False, indent=2))
