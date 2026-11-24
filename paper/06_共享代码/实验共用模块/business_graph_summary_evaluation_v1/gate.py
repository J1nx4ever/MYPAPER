"""Read-only identity, graph, cost and independent-evaluation gates."""
import csv
import json
from pathlib import Path

from business_graph_summary_campaign_v1.plan_io import ROOT, encoded, sha, reference, load_reference, load_suite
from business_graph_summary_campaign_v1.gate import PROTECTED_SHA, check_phase
from business_graph_summary_campaign_v1.contract import seed, require

PLAN=ROOT/'change_plans/token_attention_001/business_graph_summary_009/checkpoint_evaluation_001'
REPORT=ROOT/'reports/token_attention_001/business_graph_summary_009/checkpoint_evaluation_001'
SUITE_CONTRACT='business-graph-summary-independent-evaluation-v1'
CAMPAIGN_ID='business-graph-summary-best-checkpoint-eval100-20261118-001'
PHASE_ID='s01_BG128_eval0100'
RUN_COUNT=100
CHECKPOINT_UPDATE=13
ACTOR_SEED=75597143230906


def source_pins():
    from business_graph_summary_recovery_v1.gate import source_pins as recovery_sources
    from business_graph_summary_campaign_v1.gate import source_pins as original_sources
    pins=original_sources();pins.update(recovery_sources())
    files=[p for p in (ROOT/'business_graph_summary_evaluation_v1').glob('*.py') if not p.name.startswith('test_')]
    files += [PLAN/'bootstrap.py',PLAN/'prepare_evaluation.py']
    for path in files:
        require(path.is_file(),'evaluation source missing: '+str(path))
        pins[str(path.resolve())]=sha(path)
    return dict(sorted(pins.items()))


def check_sources(pins):
    require(pins==source_pins(),'evaluation implementation changed; freeze a new package')
    require(sha(ROOT.parents[1]/'python/episode_budget_ppo.py')==PROTECTED_SHA,'protected PPO core changed')


def check_jobs(suite):
    require(suite['contract']==SUITE_CONTRACT and suite['kind']=='independent_evaluation' and
        suite['campaign_id']==CAMPAIGN_ID and suite['new_shift_cap']==suite['evaluation_shifts']==RUN_COUNT and
        suite['training_shifts']==suite['training_updates']==suite['optimizer_steps']==0,
        'exact100 independent shifts with zero optimization required')
    require(suite['checkpoint_update']==CHECKPOINT_UPDATE and suite['actor_seed']==ACTOR_SEED and
        suite['sampling']=='stochastic_private_rng' and suite['cpu_threads']==2,
        'selected frozen checkpoint or sampling protocol differs')
    require(suite['legacy_reserved_shifts']==suite['active_h2_reserved_shifts']==0,
        'cancelled H2 reservations cannot be reused')
    output=ROOT/'runs'/CAMPAIGN_ID
    material=ROOT/'runs'/(CAMPAIGN_ID+'_materials')/PHASE_ID
    recording=CAMPAIGN_ID+'-'+PHASE_ID
    require(Path(suite['python_output'])==output and Path(suite['control_directory'])==output/'control',
        'independent output/control identity differs')
    require(len(suite['jobs'])==100,'exact100 canonical jobs required')
    for ordinal,j in enumerate(suite['jobs']):
        require(j['ordinal']==ordinal and j['run_index']==9950001+ordinal and
            j['experiment_seed']==seed(CAMPAIGN_ID,'environment',ordinal) and
            j['actor_sampling_seed']==seed(CAMPAIGN_ID,'sampling',ordinal) and j['actor_seed']==ACTOR_SEED,
            'fresh evaluation run or seed differs')
        require(j['role']=='final' and j['partition']=='validation' and
            j['target_training_update']==0 and j['checkpoint_update']==CHECKPOINT_UPDATE and
            j['candidate_id']=='s01_BG128' and j['phase_id']==PHASE_ID,
            'evaluation cannot request training or another candidate')
        require(j['recording_id']==j['source_recording_id']==recording and
            Path(j['python_output'])==output/'evaluation' and
            Path(j['material_event_directory'])==material,
            'evaluation recording/output identity differs')
        episode=f"dispatch-production-wait-shortfall-v2-run-{j['run_index']}-seed-{j['experiment_seed']}"
        require(Path(j['material_event_bundle'])==material/episode and
            Path(j['material_journal'])==material/episode/'material_events.bin',
            'job-local material identity differs')
    streams=[{j[k] for j in suite['jobs']} for k in ('experiment_seed','actor_sampling_seed')]
    require(all(len(x)==100 for x in streams) and streams[0].isdisjoint(streams[1]) and
        ACTOR_SEED not in streams[0]|streams[1], 'evaluation seed reuse')
    return suite


def check_approval(approval,suite,suite_sha256):
    require(approval.get('decision')=='approved' and approval.get('approved_by')=='user' and
        approval.get('execution_authorized') is True and bool(approval.get('source_user_instruction')) and
        approval.get('campaign_id')==suite['campaign_id'] and approval.get('campaign_sha256')==suite_sha256 and
        approval.get('max_new_shifts')==100==suite['new_shift_cap'] and
        approval.get('training_updates')==approval.get('optimizer_steps')==0 and
        approval.get('checkpoint')==suite['parent_artifacts']['checkpoint'] and
        approval.get('maximum_total_after_registration')==suite['maximum_total_if_approved'],
        'user100-shift authorization must bind this frozen checkpoint and exact plan')
    return approval


def preflight(path, *, approval_path=None):
    from .budget import GLOBAL_PATH, budget_snapshot, preceding_reference, check_collisions
    from business_graph_summary_recovery_v1.admission import _phase
    path=Path(path).resolve();suite=json.loads(path.read_bytes())
    require(suite['suite_path']==str(path),'suite path differs')
    check_sources(suite['source_pins']);check_jobs(suite)
    for key in ('phase_config','master_plan','java_helper','one_click_entry','selection','training_suite','training_completion'):
        require(reference(suite[key]['path'])==suite[key],'frozen evaluation artifact changed: '+key)
    for ref in suite['parent_artifacts'].values():
        require(reference(ref['path'])==ref,'fixed checkpoint artifact changed')
    require(set(suite['parent_artifacts'])=={'checkpoint','manifest','seal'},'sealed checkpoint required')
    parent,_=load_suite(suite['training_suite']['path'])
    require(suite['learner_config']==parent['arms'][0]['learner_config'],'frozen learner configuration differs')
    expected=Path(parent['arms'][0]['python_output'])/'updates/0013/checkpoint.pt'
    require(Path(suite['parent_artifacts']['checkpoint']['path'])==expected,'selected checkpoint is not update13')
    selection=load_reference(suite['selection'])
    require(selection['status']=='PASS' and selection['selected_update']==CHECKPOINT_UPDATE and
        selection['selected_checkpoint']==suite['parent_artifacts']['checkpoint'] and
        selection['selected_learner_config']==suite['learner_config'] and
        selection['training_suite']==suite['training_suite'] and
        selection['selection_frozen_before_100_new_evaluation_shifts'] is True,
        'training-only selection must bind this exact checkpoint before sampling')
    manifest=load_reference(suite['parent_artifacts']['manifest'])
    require(manifest['actor_digest']==selection['selected_actor_policy_digest'] and
        manifest['sha256']==suite['parent_artifacts']['checkpoint']['sha256'] and
        manifest['update_index']==CHECKPOINT_UPDATE,'selected Actor and checkpoint manifest differ')
    complete=load_reference(suite['training_completion'])
    require(complete['global_updates']==16 and complete['physical_shifts']==512 and
        complete['campaign']==suite['training_suite'],'parent training incomplete')
    cfg=_phase(suite['phase_config']['path']);check_phase(cfg)
    require(cfg['jobs']==suite['jobs'] and cfg['evaluation_only'] is True and cfg['optimizer_steps']==0,
        'phase and independent jobs differ')
    from baseline1_v1.java_plan import COLUMNS
    with Path(suite['master_plan']['path']).open(newline='') as handle:
        require(list(csv.DictReader(handle))==[{k:str(j[k]) for k in COLUMNS} for j in suite['jobs']],
            'Java master differs from canonical jobs')
    require(Path(suite['evaluation_budget_path'])==GLOBAL_PATH,'independent evaluation extension path differs')
    state=budget_snapshot()
    if suite['campaign_id'] not in state['campaigns']:
        check_collisions(suite,state)
        require(preceding_reference()==suite['budget_state_before'],'preceding complete merged budget changed')
        require(state['maximum_total_shifts']+100==suite['maximum_total_if_approved'],'exact100 budget differs')
        require(suite['evaluation_budget_before']==(reference(GLOBAL_PATH) if GLOBAL_PATH.exists() else None),
            'evaluation extension changed')
        require(not Path(suite['python_output']).exists() and
            not (ROOT/'runs'/(suite['campaign_id']+'_materials')).exists(),'new evaluation output occupied')
    else:
        require(state['campaigns'][suite['campaign_id']]['campaign_sha256']==sha(path),'registered evaluation differs')
    approval_path=Path(approval_path) if approval_path is not None else path.parent/'approval.json'
    approval=None if not approval_path.exists() else check_approval(json.loads(approval_path.read_bytes()),suite,sha(path))
    return suite,cfg,approval
