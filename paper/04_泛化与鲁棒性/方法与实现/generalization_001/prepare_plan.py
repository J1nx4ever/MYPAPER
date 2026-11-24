"""Freeze the six-scene one-click package without launching or registering it."""
import csv

from gen_common import (
    CAMPAIGN, CHECKPOINT, CHECKPOINT_SHA, CONTROL, COUNT, EXTENSION_OUT, EXTENSION_PLAN,
    FIRST_RUN, MATERIALS, OUTPUT, PLAN, PREPARED, PREVIOUS_CAP, SCENARIOS, SUITE, TEMPLATE, TOTAL,
    config_for, evaluation_source_pins, load_package, phase_identity, read, reference, require,
    seeds_for, source_pins,
)
from baseline1_v1.java_plan import COLUMNS
from business_graph_summary_campaign_v1.plan_io import load_phase, write
from gen_budget import budget_snapshot, check_collisions
from java_scene import render_helper, scenario_table_bytes


def scene_phases():
    template = load_phase(TEMPLATE)
    jobs, phases, identities = [], [], []
    for scenario in SCENARIOS:
        name, tag, recording, output, events = phase_identity(scenario)
        material = MATERIALS / tag
        cfg = dict(template, campaign_id=CAMPAIGN, phase_id=tag, arm_phase_id=name, recording_id=recording,
                   python_output=str(output), event_directory=str(events), jobs=[])
        scene_sha = None
        for slot, old in enumerate(template['jobs']):
            expected, scene_sha = config_for(old['expected_rpc_config'], scenario)
            ordinal = len(jobs)
            env, sampling = seeds_for(slot)
            run = FIRST_RUN + ordinal
            group = f'{CAMPAIGN}-{scenario}-{slot:03d}'
            leaf = f'dispatch-production-wait-shortfall-v2-run-{run}-seed-{env}'
            job = dict(old, ordinal=ordinal, pair_ordinal=slot, run_index=run, experiment_seed=env,
                       actor_sampling_seed=sampling, scene_id=scenario, scene_group=group,
                       realization_group_id=group, runtime_scene_sha256=scene_sha,
                       phase_id=tag, arm_phase_id=name, recording_id=recording, source_recording_id=recording,
                       python_output=str(output), event_directory=str(events), event_bundle=str(events / leaf),
                       material_event_directory=str(material), material_event_bundle=str(material / leaf),
                       material_journal=str(material / leaf / 'material_events.bin'),
                       expected_rpc_config=expected)
            cfg['jobs'].append(job)
            jobs.append(job)
        path = PREPARED / 'configs' / (tag + '.json')
        write(path, cfg)
        item = dict(scenario_id=scenario, phase_id=tag, phase_config=reference(path), python_output=str(output),
                    checkpoint_update=32, runtime_scene_sha256=scene_sha)
        phases.append(item)
        identities.append(dict(item, recording_id=recording, anylogic_event_directory=str(material),
                               shifts=COUNT, first_run=cfg['jobs'][0]['run_index'], last_run=cfg['jobs'][-1]['run_index']))
    return jobs, phases, identities


def write_rows(path, keys, rows):
    with path.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, lineterminator='\n')
        writer.writeheader()
        writer.writerows({key: row[key] for key in keys} for row in rows)


def prepare():
    require(not any(p.exists() for p in (PREPARED, OUTPUT, MATERIALS)), 'new independent experiment identity required')
    previous = budget_snapshot()
    require(previous['maximum_total_shifts'] == PREVIOUS_CAP, 'preceding experiment budget differs')
    parent = EXTENSION_PLAN / 'prepared_001/suite.json'
    completed = EXTENSION_OUT / 'suite_complete.json'
    suite = dict(
        schema='bg128-u32-generalization-v1', campaign_id=CAMPAIGN, suite_path=str(SUITE),
        new_shift_cap=TOTAL, evaluation_shifts=TOTAL, training_shifts=0, training_updates=0,
        optimizer_steps=0, checkpoint_update=32, checkpoint_sha256=CHECKPOINT_SHA, cpu_threads=2,
        sampling='stochastic_private_rng', paired_slots=COUNT, training_seed_count=1,
        expected_preceding_cap=PREVIOUS_CAP, maximum_total_if_started=PREVIOUS_CAP + TOTAL,
        python_output=str(OUTPUT), control_directory=str(CONTROL), material_root=str(MATERIALS),
        budget_path=str(OUTPUT / 'generalization_ledger.sqlite'),
        source_pins=source_pins(), evaluation_source_pins=evaluation_source_pins(),
        training_suite=reference(parent), training_completion=reference(completed),
        learner_config=read(parent)['learner_config'], selection=read(completed)['paired_reports']['json'],
        parent_artifacts=dict(checkpoint=reference(CHECKPOINT), manifest=reference(str(CHECKPOINT) + '.manifest.json'),
                              seal=reference(str(CHECKPOINT) + '.seal.json')),
        phase_template=reference(TEMPLATE), one_click_entry=reference(PLAN / 'bootstrap.py'),
        scenario_overrides=SCENARIOS, scenario_scope='one frozen actor; fixed observation and objective; six physical settings',
        manual_java_changes={name: reference(PLAN / name) for name in (
            'Main.init1.LoadSites.before.java', 'Main.init1.LoadSites.after.java',
            'Main.addBidirectionalPath.before.java', 'Main.addBidirectionalPath.after.java',
        )},
    )
    jobs, phases, identities = scene_phases()
    suite['phases'] = phases
    check_collisions(dict(suite, jobs=jobs), previous)
    master = PREPARED / 'master_plan.csv'
    table = PREPARED / 'scenarios.csv'
    helper = PREPARED / 'Baseline1CampaignPlan.java'
    write_rows(master, COLUMNS, jobs)
    table.write_bytes(scenario_table_bytes(jobs))
    helper.write_text(render_helper(suite, jobs, master, table, SUITE, PLAN / 'bootstrap.py'), encoding='utf-8')
    suite.update(master_plan=reference(master), scenario_table=reference(table), java_helper=reference(helper))
    write(SUITE, suite)
    write(PREPARED / 'launch_identity.json',
          dict(suite=reference(SUITE), checkpoint=suite['parent_artifacts']['checkpoint'],
               java_helper=suite['java_helper'], scenario_table=suite['scenario_table'],
               python_output=str(OUTPUT), anylogic_event_root=str(MATERIALS), shifts=TOTAL, configurations=identities))
    columns = ('ordinal', 'pair_ordinal', 'scene_id', 'recording_id', 'config_sha256', 'runtime_scene_sha256',
               'python_output', 'material_event_directory', 'run_index', 'experiment_seed',
               'actor_sampling_seed', 'actor_seed', 'checkpoint_update')
    digests = {item['phase_id']: item['phase_config']['sha256'] for item in identities}
    write_rows(PREPARED / 'identities.csv', columns,
               [dict(job, config_sha256=digests[job['phase_id']]) for job in jobs])
    load_package()
    print(f'PREPARED_ONLY suite={SUITE} sha256={reference(SUITE)["sha256"]} shifts={TOTAL}')


if __name__ == '__main__':
    prepare()
