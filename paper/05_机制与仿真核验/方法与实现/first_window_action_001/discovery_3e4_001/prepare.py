"""Prepare the authorized new-seed series without reserving or launching slots."""
import copy
import csv
import importlib.util
import json
from pathlib import Path

from discovery_common import (APPROVAL, CAMPAIGN, CONTROL, COUNT, FIRST_RUN, MATERIALS, OLD, OUTPUT,
    PHASE, PLAN, PREPARED, RECORDING, SUITE, approval_for, check_fresh_jobs, check_package, checked,
    previous_budget, read, reference, require, seed, write)
from business_graph_summary_evaluation_v1.gate import check_phase


def prepare():
    require(not PREPARED.exists() and not OUTPUT.exists() and not MATERIALS.parent.exists(),
            'fresh discovery identity required')
    before = previous_budget()
    require(before['maximum_total_shifts']==21699,'expected complete prior budget including first four interventions')
    old = read(OLD)
    cfg = read(checked(old['phase_config']))
    template = copy.deepcopy(cfg['jobs'][0])
    cfg.update(campaign_id=CAMPAIGN,phase_id=PHASE,arm_phase_id='discover0100',recording_id=RECORDING,
        python_output=str(OUTPUT/'evaluation'),event_directory=str(CONTROL/'legacy_unused'/PHASE),jobs=[])
    for ordinal in range(COUNT):
        run = FIRST_RUN+ordinal
        env = seed(CAMPAIGN,'environment',ordinal)
        sample = seed(CAMPAIGN,'sampling',ordinal)
        episode = f'dispatch-production-wait-shortfall-v2-run-{run}-seed-{env}'
        job = dict(template,ordinal=ordinal,run_index=run,experiment_seed=env,actor_sampling_seed=sample,
            phase_id=PHASE,arm_phase_id='discover0100',recording_id=RECORDING,source_recording_id=RECORDING,
            scene_group=CAMPAIGN+f'-validation-{ordinal:04d}',realization_group_id=CAMPAIGN+f'-validation-{ordinal:04d}',
            python_output=cfg['python_output'],event_directory=cfg['event_directory'],
            event_bundle=str(Path(cfg['event_directory'])/episode),material_event_directory=str(MATERIALS),
            material_event_bundle=str(MATERIALS/episode),material_journal=str(MATERIALS/episode/'material_events.bin'))
        cfg['jobs'].append(job)
    check_phase(cfg)
    check_fresh_jobs(cfg['jobs'],before)
    PREPARED.mkdir()
    phase_path = PREPARED/'phase_config.json'
    write(phase_path,cfg)
    keep = ('learner_config','parent_artifacts','selection','training_suite','training_completion','cpu_threads',
            'source_pins','checkpoint_update','actor_seed','candidate_id','sampling','training_updates',
            'optimizer_steps','training_shifts','new_shift_cap','evaluation_shifts')
    suite = {field:old[field] for field in keep}
    suite.update(schema='bg128-u13-first-window-discovery-v1',campaign_id=CAMPAIGN,
        original_suite=reference(OLD),protocol=reference(PLAN/'protocol.md'),
        phase_config=reference(phase_path),python_output=str(OUTPUT),control_directory=str(CONTROL),jobs=cfg['jobs'],
        occupied_before=before['maximum_total_shifts'],maximum_total_if_approved=before['maximum_total_shifts']+COUNT)
    old_renderer = PLAN.parents[1]/'optimization_stability_001/prepare_training.py'
    loader = importlib.util.spec_from_file_location('bg128_java_renderer',old_renderer)
    renderer = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(renderer)
    master = PREPARED/'master_plan.csv'
    master.write_bytes(renderer.master_bytes(cfg['jobs']))
    suite['master_plan'] = reference(master)
    helper = PREPARED/'Baseline1CampaignPlan.java'
    helper.write_text(renderer.render_java(suite,master,suite_path=SUITE,bridge_path=PLAN/'bootstrap.py'))
    suite['java_helper'] = reference(helper)
    closure = ['bootstrap.py','discovery_budget.py','discovery_common.py','run_discovery.py','prepare.py']
    suite['discovery_sources'] = {name:reference(PLAN/name) for name in closure}
    suite['discovery_sources']['preceding_budget'] = reference(PLAN.parent/'followup_001/merged_budget.py')
    write(SUITE,suite)
    write(APPROVAL,approval_for(suite))
    columns = ('ordinal','run_index','experiment_seed','actor_sampling_seed','actor_seed','recording_id',
               'python_output','material_event_directory')
    seed_file = PREPARED/'run_seed.csv'
    with seed_file.open('x',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=columns,extrasaction='ignore',lineterminator='\n')
        writer.writeheader()
        writer.writerows(cfg['jobs'])
    write(PREPARED/'launch_identity.json',dict(status='PREPARED',recording_id=RECORDING,
        suite=reference(SUITE),config=suite['phase_config'],python_output=cfg['python_output'],
        anylogic_event_directory=str(MATERIALS),control_directory=str(CONTROL),
        first_run=FIRST_RUN,last_run=FIRST_RUN+COUNT-1,run_seed=reference(seed_file),checkpoint=suite['parent_artifacts']['checkpoint'],
        budget_before=suite['occupied_before'],budget_after_registration=suite['maximum_total_if_approved'],
        java_helper=suite['java_helper'],physical_shifts=COUNT,optimizer_updates=0))
    check_package()
    print(json.dumps(read(PREPARED/'launch_identity.json'),ensure_ascii=False))


if __name__=='__main__':
    prepare()
