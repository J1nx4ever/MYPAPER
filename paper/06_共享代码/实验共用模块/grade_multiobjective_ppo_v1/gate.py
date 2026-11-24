"""Paired576-shift reward fine-tuning from the immutable historical B model."""
import copy
import csv
import json
from pathlib import Path

from baseline1_v1 import gate as baseline
from baseline1_v1.prepare import _seed
from grade_commitment_ppo_fast_v1.gate import source_pins as prior_pins
from grade_only_ppo_v1.gate import MEMORY
from .budget_policy import GLOBAL_PATH,budget_snapshot,check_collisions,budget_state_reference

ROOT,REPO,PROTECTED_SHA=baseline.ROOT,baseline.REPO,baseline.PROTECTED_SHA
encoded,reference,sha,require,load_reference=(baseline.encoded,baseline.reference,baseline.sha,
    baseline.require,baseline.load_reference)
PLAN=ROOT/'change_plans/grade_production_wait_finetune_001'
CAMPAIGN_CONTRACT='grade-production-wait-finetune-suite-v1'
PHASE_CONTRACT='grade-three-objective-rpc-phase-v1'
CAMPAIGN_ID='grade-production-wait-finetune-seed1-20261109-001'
EXPERIMENT_ID=CAMPAIGN_ID
PAIR_ID='grade-production-wait-pair-seed1-20261109-001'
STAGES=('wait0','wait01')
WAIT_WEIGHTS={'wait0':0.0,'wait01':0.1}
PROFILE='formal_32ep_v1'
PURPOSE='historical_grade_policy_production_penalty_and_paired_wait_finetuning'
CANDIDATE='a1_actor_gru'
INITIALIZATION='historical_grade_only_B_u8_learnable_parameters_reset_optimizers'
PARENT_PATH=ROOT/'runs/grade-commitment-gru-ab-seed1-20261108-001/arms/a1_actor_gru/updates/0008/checkpoint.pt'
PARENT_SHA='e457765c16afd2d90023c3e214ffa0d8e8da86aa0ede9b8d154cc38db4693bdf'
TRAINING_REWARD='negative_combined_grade_production_wait_cost_v1'
DIAGNOSTIC_SCOPE=dict(scope='paired_wait_finetuning_from_historical_grade_policy',production='training_and_evaluation',
    waiting='explicit_arm_weight_0_or_0.1',production_guarantee=False)


def parent_references():
    require(sha(PARENT_PATH)==PARENT_SHA,'historical parent checkpoint bytes changed')
    return {key:reference(path) for key,path in dict(checkpoint=PARENT_PATH,
        manifest=Path(str(PARENT_PATH)+'.manifest.json'),seal=Path(str(PARENT_PATH)+'.seal.json')).items()}


def source_pins():
    pins=prior_pins()
    pins.update({str(p.resolve()):sha(p) for p in (ROOT/'grade_multiobjective_ppo_v1').glob('*.py')
                 if not p.name.startswith('test_')})
    return dict(sorted(pins.items()))


def check_sources(pins):
    require(pins==source_pins(),'composite source inventory changed; freeze a new package')
    require(sha(REPO/'python/episode_budget_ppo.py')==PROTECTED_SHA,'protected PPO core changed')


def schedule():
    return [('t0','t0',0,16)]+[(f'train_u{i:04d}','train',i,32) for i in range(1,9)]+[('final_u0008','final',8,16)]


def check_phase(cfg):
    from .features import feature_contract
    from .targets import composite_contract
    require(cfg.get('contract')==PHASE_CONTRACT and cfg.get('execution_profile')==PROFILE and
        cfg.get('candidate_id')==CANDIDATE and cfg.get('purpose')==PURPOSE and
        cfg.get('experiment_id')==EXPERIMENT_ID,'composite phase scope differs')
    compatible=copy.deepcopy(cfg);compatible['contract']=baseline.PHASE_CONTRACT
    baseline.check_phase(compatible)
    require(cfg.get('material_memory')==MEMORY and cfg.get('feature_contract')==feature_contract() and
        cfg.get('stage_id') in STAGES and cfg.get('composite_contract')==composite_contract(WAIT_WEIGHTS[cfg['stage_id']]) and cfg.get('training_reward')==TRAINING_REWARD and
        cfg.get('diagnostic_scope')==DIAGNOSTIC_SCOPE,'explicit material/feature/composite contract differs')
    require(cfg.get('history_contract')==dict(scope='whole_shift',window_reset=False,terminal_reset=True,
        source='causal_decision_summaries',critic='decision_gru'),'full causal history required')
    require(cfg['reward']['kappa']==4e-5 and cfg['reward']['ore_weight']==4 and
        cfg['reward']['waste_weight']==4,'raw RPC4/4 and waiting contract must stay unchanged')
    return cfg


def check_approval(approval,suite,suite_sha256):
    require(approval.get('decision')=='approved' and approval.get('approved_by')=='user' and
        approval.get('execution_authorized') is True and approval.get('ack_composite_reward') is True and
        approval.get('ack_historical_parent_preserved') is True and bool(approval.get('source_user_instruction')) and
        approval.get('experiment_id')==EXPERIMENT_ID and approval.get('execution_profile')==PROFILE and
        approval.get('campaign_id')==suite['campaign_id'] and approval.get('campaign_sha256')==suite_sha256 and
        approval.get('max_new_shifts')==576 and approval.get('legacy_reserved_shifts')==0 and
        approval.get('active_h2_reserved_shifts')==0 and approval.get('budget_policy')==suite['budget_policy'] and
        approval.get('accounting')=='additional_to_active_approved_campaigns' and
        approval.get('maximum_total_after_registration')==suite['maximum_total_if_approved'],
        'exact576 paired fine-tuning budget/execution and immutable parent authorization required')
    return approval


def preflight(path,*,approval_path=None):
    from .features import feature_contract
    from .targets import composite_contract
    from .learner import default_config,validate_config
    from .admission import _phase
    path=Path(path).resolve();suite=json.loads(path.read_bytes())
    require(suite.get('contract')==CAMPAIGN_CONTRACT and suite.get('campaign_id')==CAMPAIGN_ID and
        suite.get('experiment_id')==EXPERIMENT_ID and suite.get('kind')=='single_seed_trial' and
        suite.get('purpose')==PURPOSE and suite.get('execution_profile')==PROFILE and
        suite.get('new_shift_cap')==576 and suite.get('legacy_reserved_shifts')==0 and
        suite.get('active_h2_reserved_shifts')==0 and
        suite.get('initialization')==INITIALIZATION,'exact paired576 suite required')
    require(suite.get('training_shifts')==512 and suite.get('initial_evaluation_shifts')==32 and
        suite.get('final_evaluation_shifts')==32 and suite.get('training_updates')==16 and
        suite.get('anylogic_start_count')==1 and suite.get('arm_order')==list(STAGES) and
        suite.get('acceptance_standard')==dict(evaluation_shifts=16,required_complete_successes=16,
            requirement='all_four_windows_and_ore16000_waste4000',usage='final_quality_assessment_only_no_prerequisite_stage') and
        suite.get('parent_artifacts')==parent_references(),
        'paired budget, historical parent or final quality criterion differs')
    check_sources(suite['source_pins'])
    require(suite.get('feature_contract')==feature_contract() and
        suite.get('stage_reward_contracts')=={s:composite_contract(WAIT_WEIGHTS[s]) for s in STAGES} and
        suite.get('diagnostic_scope')==DIAGNOSTIC_SCOPE,'stage coefficients/input definition differs')
    output=ROOT/'runs'/CAMPAIGN_ID
    require(Path(suite['python_output'])==output and Path(suite['control_directory'])==output/'control',
        'dedicated new output/control required')
    require(len(suite['arms'])==2,'exactly two waiting-weight branches required')
    for arm,stage in zip(suite['arms'],STAGES):
        require(arm['stage_id']==stage and arm['candidate_id']==CANDIDATE and arm['actor_seed']==2026110903 and
            Path(arm['python_output'])==output/'arms'/stage and
            validate_config(arm['learner_config'])==default_config(seed=2026110903,candidate_id=CANDIDATE,
                wait_weight=WAIT_WEIGHTS[stage],source_kind='real_anylogic_rpc',execution_profile=PROFILE),
            'fixed B architecture or stage learner differs')
    phases=[check_phase(load_reference(ref)) for ref in suite['phase_configs']]
    for ref in suite['phase_configs']:_phase(ref['path'])
    require(len(phases)==20,'all20 phases required')
    jobs=[j for cfg in phases for j in cfg['jobs']]
    require(jobs==suite['jobs'] and len(jobs)==576,'canonical576 jobs differ')
    ordinal=0
    for si,stage in enumerate(STAGES):
        pair=0;arm=suite['arms'][si]
        for phase,(name,role,local,count) in zip(phases[si*10:(si+1)*10],schedule()):
            phase_id=stage+'_'+name;recording=CAMPAIGN_ID+'-'+phase_id
            require(phase['stage_id']==stage and phase['campaign_id']==CAMPAIGN_ID and
                phase['phase_id']==phase_id and phase['arm_phase_id']==name and phase['arm_update']==local and
                len(phase['jobs'])==count and phase['recording_id']==recording and
                Path(phase['python_output'])==Path(arm['python_output'])/name,'stage phase schedule differs')
            namespace=PAIR_ID
            for job in phase['jobs']:
                require(job['ordinal']==ordinal and job['pair_ordinal']==pair and job['stage_id']==stage and
                    job['run_index']==9030001+si*10000+pair and
                    job['experiment_seed']==_seed(namespace,'environment',pair) and
                    job['actor_sampling_seed']==_seed(namespace,'actor',pair) and
                    job['actor_seed']==2026110903 and job['candidate_id']==CANDIDATE and
                    job['arm_phase_id']==name and job['arm_update']==local and job['role']==role and
                    job['target_training_update']==si*8+local and job['checkpoint_update']==(local-1 if role=='train' else local) and
                    job['recording_id']==job['source_recording_id']==recording and
                    job['partition']==('train' if role=='train' else 'validation') and
                    Path(job['material_event_directory'])==ROOT/'runs'/(CAMPAIGN_ID+'_materials')/stage/name,
                    'job identity/paired seed/partition/update differs')
                ordinal+=1;pair+=1
    require(len({j['run_index'] for j in jobs})==576,'duplicate physical run')
    for i in range(288):
        a,b=jobs[i],jobs[288+i]
        require(all(a[k]==b[k] for k in ('experiment_seed','actor_sampling_seed','scene_group','partition')),
            'wait comparison must use corresponding paired scenes/seeds')
    for key in ('master_plan','java_helper'):
        require(reference(suite[key]['path'])==suite[key],key+' bytes changed')
    from baseline1_v1.java_plan import COLUMNS
    with Path(suite['master_plan']['path']).open(newline='') as f:master=list(csv.DictReader(f))
    require(master==[{k:str(j[k]) for k in COLUMNS} for j in jobs],'canonical CSV differs')
    from python.protocol_artifacts import verify_protocol_artifacts
    from python.collect_offline import verify_environment
    verify_protocol_artifacts()
    for phase in phases:verify_environment(phase['environment_acceptance'],phase['reward'],storage_migration=True)
    acceptance=load_reference(suite['engineering_acceptance'])
    require(acceptance.get('status')=='PASS_SYNTHETIC_ONLY' and acceptance.get('source_pins')==suite['source_pins'] and
        all(acceptance.get('gates',{}).get(k)=='PASS_SYNTHETIC_ONLY' for k in ('G1','G2','G3','G4')),
        'matching fine-tuning engineering gates required')
    budget=budget_snapshot();cid=suite['campaign_id']
    current=budget_state_reference()
    require(suite.get('budget_policy')==current['policy'],'active budget policy changed')
    if cid not in budget['campaigns']:
        require(current==suite['budget_state_before'],'active/history budget changed since freeze')
        check_collisions(suite,budget)
        require(not output.exists() and not Path(str(output)+'_materials').exists(),'output identity already used')
    else:require(budget['campaigns'][cid]['campaign_sha256']==sha(path),'registered suite changed')
    prior=budget['additional_reserved_shifts']-(576 if cid in budget['campaigns'] else 0)
    require(suite['prior_baseline_reserved_shifts']==prior and suite['maximum_total_if_approved']==prior+576,
        'independent576 cap must exclude cancelled H2 future reservations')
    approval=None if approval_path is None else check_approval(json.loads(Path(approval_path).read_bytes()),suite,sha(path))
    return suite,phases,approval
