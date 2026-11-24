"""Freeze the four-strategy validation package without reserving physical runs."""
import json

from smoke_plan import (
    CAMPAIGN, CONTROL, EVENTS, EXECUTION, LAYOUT, METHOD, OUTPUT, PLAN, PREPARED, SCHEMA, STRATEGIES,
    SUITE, config_for, job_for, load_suite, reference, require, seed, source_contract, source_pins,
)
from smoke_budget import check_fresh, preceding_snapshot
from business_graph_summary_campaign_v1.plan_io import write
from java_helper import master_bytes, render, scenario_table_bytes, verify_java


def prepare():
    require(not PREPARED.exists() and not OUTPUT.exists(), 'fresh unused preparation and output directory required')
    pins = source_pins()
    refs, learner, rpc, scene = source_contract()
    env, sample = seed('environment'), seed('actor-sampling')
    jobs = [job_for(i, env, sample, learner, rpc, scene) for i in range(len(STRATEGIES))]
    summary = check_fresh({'jobs': jobs}, preceding_snapshot())
    PREPARED.mkdir()
    for job in jobs:
        path = PREPARED/'configs'/f"{job['policy_kind']}.json"
        write(path, config_for(job, refs))
        job['phase_config'] = reference(path)
    master, scenarios, helper = (PREPARED/name for name in ('master_plan.csv', 'scenario_table.csv', 'Baseline1CampaignPlan.java'))
    master.write_bytes(master_bytes(jobs))
    scenarios.write_bytes(scenario_table_bytes(jobs))
    helper.write_text(render(jobs, master, scenarios, suite_path=SUITE, bridge_path=PLAN/'launch.py'), encoding='utf-8')
    java_check = verify_java(helper, scenarios)
    require(java_check['rows_checked'] == len(STRATEGIES), 'compiled Java helper inventory differs')
    suite = dict(schema=SCHEMA, campaign_id=CAMPAIGN, **LAYOUT, execution=EXECUTION, method=METHOD,
                 learner_config=learner, **refs, master_plan=reference(master), scenario_table=reference(scenarios),
                 java_helper=reference(helper), source_pins=pins, pairing=dict(scope='this_campaign_only', slots=1,
                     share='experiment_seed', policies=list(STRATEGIES)), physical_shifts=len(STRATEGIES),
                 training_shifts=0, optimizer_updates=0, jobs=jobs)
    write(SUITE, suite)
    load_suite()
    identity = dict(status='PREPARED', suite=reference(SUITE), campaign_id=CAMPAIGN,
                    java_helper=suite['java_helper'], checkpoint=refs['checkpoint'], protected_core=refs['protected_core'],
                    nominal_source=refs['nominal_source'], control_directory=str(CONTROL), event_root=str(EVENTS),
                    physical_shifts=len(STRATEGIES), training_shifts=0, optimizer_updates=0, budget=summary,
                    java_verification=java_check,
                    jobs=[dict({key: job[key] for key in ('ordinal', 'policy_kind', 'run_index', 'recording_id',
                                'experiment_seed', 'python_output', 'actor_sampling_seed',
                                'actor_sampling_seed_consumed', 'phase_config')},
                               anylogic_event_directory=job['event_directory']) for job in jobs])
    write(PREPARED/'launch_identity.json', identity)
    return identity


if __name__ == '__main__':
    print(json.dumps(prepare(), ensure_ascii=False, indent=2))
