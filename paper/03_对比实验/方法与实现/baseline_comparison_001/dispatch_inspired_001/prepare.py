"""Write a sealed single-shift package without creating a run or ledger."""
import json

from dispatch_plan import (CAMPAIGN, CONFIG, COUNT, EXECUTION, LAYOUT, METHOD, OUTPUT, PREPARED, SCHEMA, SUITE,
                           config_for, job_for, load_suite, reference, require, source_contract, source_pins)
from dispatch_budget import budget_snapshot, check_fresh
from business_graph_summary_campaign_v1.plan_io import write
from java_adapter import master_bytes, render, scenario_table_bytes, verify_java


def prepare():
    require(not PREPARED.exists() and not OUTPUT.exists(), 'fresh DISPATCH preparation and run directory required')
    pins = source_pins()
    refs, rpc, scene = source_contract()
    job = job_for(rpc, scene)
    jobs = [job]
    readiness = check_fresh({'jobs':jobs}, budget_snapshot())
    PREPARED.mkdir()
    write(CONFIG, config_for(job, refs))
    job['phase_config'] = reference(CONFIG)
    master, scenarios, helper = (PREPARED/name for name in ('master_plan.csv', 'scenario_table.csv',
                                                          'Baseline1CampaignPlan.java'))
    master.write_bytes(master_bytes(jobs))
    scenarios.write_bytes(scenario_table_bytes(jobs))
    helper.write_text(render(jobs, master, scenarios), encoding='utf-8')
    java_check = verify_java(helper, scenarios)
    require(java_check['rows_checked'] == COUNT, 'incomplete generated Java verification')
    suite = dict(schema=SCHEMA, campaign_id=CAMPAIGN, **LAYOUT, execution=EXECUTION, method=METHOD, **refs,
                 master_plan=reference(master), scenario_table=reference(scenarios), java_helper=reference(helper),
                 source_pins=pins, physical_shifts=COUNT, training_shifts=0, optimizer_updates=0, jobs=jobs)
    write(SUITE, suite)
    load_suite()
    identity = dict(status='PREPARED', campaign_id=CAMPAIGN, suite=reference(SUITE), java_helper=suite['java_helper'],
                    **LAYOUT, physical_shifts=COUNT, training_shifts=0, optimizer_updates=0, budget=readiness,
                    java_verification=java_check, protected_core=refs['protected_core'],
                    nominal_source=refs['nominal_source'], job={key: job[key] for key in (
                        'ordinal', 'run_index', 'recording_id', 'experiment_seed', 'actor_sampling_seed',
                        'actor_sampling_seed_consumed', 'python_output', 'event_directory', 'phase_config', 'policy_kind')})
    write(PREPARED/'launch_identity.json', identity)
    return identity


if __name__ == '__main__':
    print(json.dumps(prepare(), ensure_ascii=False, indent=2))
