"""Prepare100 new fixed-checkpoint BG128 shifts; never launch or register."""
import argparse
import copy
import csv
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path('/home/jinx/Faith/experiments/lagrangian_ppo_v1')
sys.path.insert(0,str(ROOT))


def prepare():
    from business_graph_stability_evaluation_v1.gate import (
        PLAN,REPORT,CAMPAIGN_ID,PHASE_ID,RUN_COUNT,SUITE_CONTRACT,ACTOR_SEED,
        CHECKPOINT_UPDATE,source_pins,check_jobs,check_phase,require,seed,reference,encoded)
    from business_graph_stability_campaign_v1.plan_io import write,load_suite
    from business_graph_stability_evaluation_v1.budget import (
        GLOBAL_PATH,budget_snapshot,preceding_reference,check_collisions)
    directory=PLAN/'prepared_001'
    require(not directory.exists(),'new immutable preparation required')
    from business_graph_stability_campaign_v1.pairing import evaluation_source, contract as pairing_contract
    paired_source=evaluation_source()
    pending=json.loads((PLAN/'seed_plan.json').read_bytes())
    require(pending['status']=='AWAITING_NEW_U13_CHECKPOINT' and pending['checkpoint_sha256'] is None and
        pending['seed_manifest']==reference(PLAN.parent/'pairing_manifest.json'), 'paired pending identity changed')
    selection_path=REPORT/'selection_review.json'
    training_path=ROOT/'change_plans/token_attention_001/business_graph_summary_009/optimization_stability_001/prepared_001/suite.json'
    parent,phases=load_suite(training_path)
    arm=parent['arms'][0]
    checkpoint=Path(arm['python_output'])/'updates/0013/checkpoint.pt'
    require(checkpoint.is_file(), 'new paired training u13 is not available; never fall back to historical u13')
    parent_artifacts=dict(checkpoint=reference(checkpoint),manifest=reference(str(checkpoint)+'.manifest.json'),
        seal=reference(str(checkpoint)+'.seal.json'))
    completion=Path(parent['python_output'])/'suite_complete.json'
    require(completion.is_file(),'training must complete before binding the new u13 SHA')
    complete=json.loads(completion.read_bytes())
    require(complete['global_updates']==16 and complete['physical_shifts']==512 and
        complete['campaign']==reference(training_path),'paired training completion differs')
    from business_graph_stability_v1.learner import JointCostPPO
    model=JointCostPPO.from_checkpoint(checkpoint,expected_config=arm['learner_config'],
        expected_sha256=parent_artifacts['checkpoint']['sha256'],restore_global_rng=False)
    require(model.update_index==13,'only the new paired training u13 may be evaluated')
    selected=dict(status='PASS',selected_update=13,selected_checkpoint=parent_artifacts['checkpoint'],
        selected_learner_config=arm['learner_config'],training_suite=reference(training_path),
        selected_actor_policy_digest=model.policy_version,
        selection_frozen_before_100_new_evaluation_shifts=True,
        selection_rule='preregistered_fixed_u13_no_outcome_selection',
        protocol=parent['optimization_protocol'])
    write(selection_path,selected)
    require(selected['status']=='PASS' and selected['selected_update']==CHECKPOINT_UPDATE and
        selected['selected_checkpoint']==parent_artifacts['checkpoint'] and
        selected['selected_learner_config']==arm['learner_config'],
        'frozen training-only selection differs from the requested parent model')
    output=ROOT/'runs'/CAMPAIGN_ID
    materials=ROOT/'runs'/(CAMPAIGN_ID+'_materials')
    require(not output.exists() and not materials.exists(),'new physical identity occupied')
    before=budget_snapshot();state=preceding_reference()
    phase=copy.deepcopy(phases[13]);template=copy.deepcopy(phase['jobs'][0])
    phase.update(campaign_id=CAMPAIGN_ID,phase_id=PHASE_ID,arm_phase_id='eval0100',arm_update=CHECKPOINT_UPDATE,
        recording_id=CAMPAIGN_ID+'-'+PHASE_ID,python_output=str(output/'evaluation'),
        event_directory=str(output/'control/legacy_unused'/PHASE_ID),evaluation_only=True,optimizer_steps=0,jobs=[])
    for ordinal in range(RUN_COUNT):
        run=9970001+ordinal;environment=paired_source['jobs'][ordinal]['experiment_seed']
        episode=f'dispatch-production-wait-shortfall-v2-run-{run}-seed-{environment}'
        job=copy.deepcopy(template);job.pop('pair_ordinal',None)
        job.update(ordinal=ordinal,stage_id='BG128',arm_phase_id='eval0100',arm_update=CHECKPOINT_UPDATE,
            actor_seed=ACTOR_SEED,role='final',target_training_update=0,checkpoint_update=CHECKPOINT_UPDATE,
            run_index=run,experiment_seed=environment,actor_sampling_seed=paired_source['jobs'][ordinal]['actor_sampling_seed'],
            scene_group=CAMPAIGN_ID+f'-validation-{ordinal:04d}',
            realization_group_id=CAMPAIGN_ID+f'-validation-{ordinal:04d}',partition='validation',
            phase_id=PHASE_ID,recording_id=phase['recording_id'],source_recording_id=phase['recording_id'],
            event_directory=phase['event_directory'],event_bundle=str(Path(phase['event_directory'])/episode),
            python_output=phase['python_output'],material_event_directory=str(materials/PHASE_ID),
            material_event_bundle=str(materials/PHASE_ID/episode),
            material_journal=str(materials/PHASE_ID/episode/'material_events.bin'))
        phase['jobs'].append(job)
    check_phase(phase)
    directory.mkdir(parents=True)
    phase_path=directory/'configs'/f'{PHASE_ID}.json';write(phase_path,phase)
    suite=dict(contract=SUITE_CONTRACT,kind='independent_evaluation',campaign_id=CAMPAIGN_ID,
        suite_path=str(directory/'suite.json'),execution_profile='fixed_checkpoint_evaluation_v1',
        candidate_id='s01_BG128',checkpoint_update=CHECKPOINT_UPDATE,actor_seed=ACTOR_SEED,
        learner_config=arm['learner_config'],parent_artifacts=parent_artifacts,
        training_suite=reference(training_path),training_completion=reference(Path(parent['python_output'])/'suite_complete.json'),
        selection=reference(selection_path),new_shift_cap=100,evaluation_shifts=100,training_shifts=0,
        training_updates=0,optimizer_steps=0,sampling='stochastic_private_rng',cpu_threads=2,
        python_output=str(output),control_directory=str(output/'control'),
        phase_config=reference(phase_path),jobs=phase['jobs'],source_pins=source_pins(),
        one_click_entry=reference(PLAN/'bootstrap.py'),legacy_reserved_shifts=0,active_h2_reserved_shifts=0,
        budget_state_before=state,evaluation_budget_before=reference(GLOBAL_PATH) if GLOBAL_PATH.exists() else None,
        evaluation_budget_path=str(GLOBAL_PATH),maximum_total_if_approved=before['maximum_total_shifts']+100,
        selection_frozen_before_evaluation=True,checkpoint_selection_uses_evaluation_results=False,
        high_frequency_diagnostics=False,pairing=pairing_contract())
    check_jobs(suite);check_collisions(suite,before)
    old_prepare=ROOT/'change_plans/token_attention_001/business_graph_summary_009/optimization_stability_001/prepare_training.py'
    spec=importlib.util.spec_from_file_location('bg128_original_preparation',old_prepare)
    renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)
    master=directory/'master_plan.csv';master.write_bytes(renderer.master_bytes(suite['jobs']))
    suite['master_plan']=reference(master)
    helper=directory/'Baseline1CampaignPlan.java'
    helper.write_text(renderer.render_java(suite,master,suite_path=directory/'suite.json',bridge_path=PLAN/'bootstrap.py'))
    suite['java_helper']=reference(helper)
    write(directory/'suite.json',suite)
    approval=dict(decision='approved',approved_by='user',execution_authorized=True,
        source_user_instruction='用和这次相同的种子进行训练和推理。只是降低学习率吗；同时应该从原始共同初始化 Checkpoint 重新开始',
        campaign_id=CAMPAIGN_ID,campaign_sha256=reference(directory/'suite.json')['sha256'],max_new_shifts=100,
        training_updates=0,optimizer_steps=0,checkpoint=parent_artifacts['checkpoint'],
        maximum_total_after_registration=suite['maximum_total_if_approved'])
    write(directory/'approval.json',approval)
    identity=dict(status='PREPARED_NOT_REGISTERED_NOT_STARTED',suite=reference(directory/'suite.json'),
        checkpoint=parent_artifacts['checkpoint'],recording_id=phase['recording_id'],config=reference(phase_path),
        python_output=phase['python_output'],anylogic_event_directory=str(materials/PHASE_ID),
        run_index_first=9970001,run_index_last=9970100,jobs=[{k:j[k] for k in
            ('ordinal','run_index','experiment_seed','actor_sampling_seed','actor_seed')} for j in suite['jobs']],
        physical_shifts=100,training_updates=0,optimizer_steps=0,sampling='stochastic_private_rng')
    write(directory/'launch_identity.json',identity)
    with (directory/'identities.csv').open('x',newline='') as stream:
        keys=('ordinal','run_index','experiment_seed','actor_sampling_seed','actor_seed','recording_id',
            'python_output','material_event_directory')
        writer=csv.DictWriter(stream,fieldnames=keys,lineterminator='\n');writer.writeheader()
        writer.writerows({k:j[k] for k in keys} for j in suite['jobs'])
    require(source_pins()==suite['source_pins'] and preceding_reference()==state,'sources or preceding budget changed')
    return reference(directory/'suite.json')


if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(prepare()))
