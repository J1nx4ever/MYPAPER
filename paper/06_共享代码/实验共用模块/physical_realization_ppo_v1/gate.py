"""Read-only exact608 single-model curriculum gates; never start or register."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path

from baseline1_v1 import gate as baseline
from grade_only_ppo_v1.gate import MEMORY
from u8_wait_efficiency_v1.gate import source_pins as preceding_source_pins
from u8_wait_efficiency_v1.waiting_diagnostics import WAIT_DIAGNOSTICS_CONTRACT as OLD_WAIT, WAIT_RPC_FIELDS

ROOT, REPO, PROTECTED_SHA = baseline.ROOT, baseline.REPO, baseline.PROTECTED_SHA
encoded, reference, sha, require, load_reference = (baseline.encoded, baseline.reference,
    baseline.sha, baseline.require, baseline.load_reference)
PLAN = ROOT/'change_plans/physical_realization_ppo_001'
CAMPAIGN_ID = EXPERIMENT_ID = 'physical-realization-ppo-seed1-20261110-001'
SUITE_CONTRACT = CAMPAIGN_CONTRACT = 'physical-realization-ppo-single-model-suite-v1'
PHASE_CONTRACT = 'physical-realization-ppo-rpc-phase-v1'
PROFILE, CANDIDATE = 'formal_32ep_v1', 'a1_actor_gru'
PURPOSE = 'single_model_physical46_production_then_wait_curriculum'
INITIALIZATION = 'pure_grade_u8_named_weights_new_adam_then_stage1_full_adam_wait01'
TRAINING_REWARD = 'negative_combined_grade_production_wait_cost_v1'
STAGES = TRAIN_STAGES = ('production', 'wait01')
WAIT_WEIGHTS = dict(production=0., wait01=.1)
CAP, TRAIN_UPDATES = 608, 16
PARENT_PATH = ROOT/'runs/grade-commitment-gru-ab-seed1-20261108-001/arms/a1_actor_gru/updates/0008/checkpoint.pt'
PARENT_SHA = 'e457765c16afd2d90023c3e214ffa0d8e8da86aa0ede9b8d154cc38db4693bdf'
TEMPLATE_PATH = ROOT/'change_plans/grade_wait01_sampling_ab_001/prepared/configs/stochastic_eval0032.json'
JAVA_TEMPLATE_PATH = ROOT/'change_plans/u8_wait_efficiency_001/prepared_004/Baseline1CampaignPlan.java'
COLUMNS = ('ordinal','actor_seed','group','role','source_recording_id','recording_id',
    'run_index','experiment_seed','actor_sampling_seed','event_directory','event_bundle',
    'python_output','target_training_update','checkpoint_update','stage_id','phase_id',
    'material_directory','wait_weight','pair_id','partition')
HISTORY_CONTRACT = dict(scope='whole_shift',window_reset=False,terminal_reset=True,
    source='causal_decision_summaries',critic='decision_gru')
WAIT_DIAGNOSTICS_CONTRACT = dict(mode='optional_if_present',schema=OLD_WAIT)
DIAGNOSTIC_SCOPE = dict(scope=PURPOSE,production='training_and_evaluation',
    waiting='raw_fleet_wait_optional_terminal_breakdown',production_guarantee=False,
    action_selection='categorical')


def schedule():
    return [('production','t0','t0',0,16)] + [
        ('production','train_u%04d'%u,'train',u,32) for u in range(1,9)
    ] + [('production','monitor_u0008','monitor',8,16)] + [
        ('wait01','train_u%04d'%u,'train',u,32) for u in range(1,9)
    ] + [('wait01','final_u0008','final',8,64)]


def seed_value(kind, ordinal, nonce=0):
    digest=hashlib.sha256(('%s|%s|%s|%s'%(CAMPAIGN_ID,kind,nonce,ordinal)).encode()).digest()
    return int.from_bytes(digest[:6],'big') or 1


def parent_references():
    require(sha(PARENT_PATH)==PARENT_SHA,'immutable pure-grade u8 parent bytes changed')
    result={key:reference(path) for key,path in dict(checkpoint=PARENT_PATH,
        manifest=Path(str(PARENT_PATH)+'.manifest.json'),seal=Path(str(PARENT_PATH)+'.seal.json')).items()}
    manifest,seal=load_reference(result['manifest']),load_reference(result['seal'])
    require(manifest['sha256']==seal['sha256']==PARENT_SHA and
        seal['manifest_sha256']==result['manifest']['sha256'] and manifest['update_index']==8,
        'pure-grade parent manifest/seal mismatch')
    return result


def source_pins():
    pins=preceding_source_pins()
    for directory in (ROOT/'physical_realization_v1',Path(__file__).parent):
        for path in directory.glob('*.py'):
            if not path.name.startswith('test_'):
                pins[str(path.resolve())]=sha(path)
    pins[str(JAVA_TEMPLATE_PATH.resolve())]=sha(JAVA_TEMPLATE_PATH)
    return dict(sorted(pins.items()))


def check_sources(pins):
    require(pins==source_pins(),'physical PPO implementation changed; freeze a new package')
    require(sha(REPO/'python/episode_budget_ppo.py')==PROTECTED_SHA,'protected PPO changed')


def canonical_master_row(job):
    return {key:str(job[key]) for key in COLUMNS}


def check_jobs(suite):
    jobs=suite.get('jobs');nonce=suite.get('seed_plan_nonce',0)
    require(suite.get('campaign_id')==CAMPAIGN_ID and suite.get('new_shift_cap')==CAP and
        isinstance(jobs,list) and len(jobs)==CAP and type(nonce) is int and nonce>=0,
        'exact608 single-model physical job manifest required')
    pos=0;seen_seeds=set()
    for stage,name,role,local,count in schedule():
        phase_id=stage+'_'+name
        global_update=local+(8 if stage=='wait01' else 0)
        for index,job in enumerate(jobs[pos:pos+count],pos):
            require(job.get('ordinal')==index and job.get('run_index')==9090001+index and
                job.get('actor_seed')==seed_value('initialization',0,nonce) and
                job.get('experiment_seed')==seed_value('environment',index,nonce) and
                job.get('actor_sampling_seed')==seed_value('sampling',index,nonce),
                'canonical fresh run/seed/initialization differs')
            for key in ('experiment_seed','actor_sampling_seed'):
                require(job[key] not in seen_seeds,'cross-phase/cross-type seed collision')
                seen_seeds.add(job[key])
            require(job.get('stage_id')==stage and job.get('phase_id')==phase_id and
                job.get('arm_phase_id')==name and job.get('arm_update')==local and
                job.get('role')==role and job.get('candidate_id')==CANDIDATE and
                job.get('target_training_update')==(global_update if role=='train' else 0) and
                job.get('checkpoint_update')==(local-1 if role=='train' else local) and
                job.get('checkpoint_global_update')==(global_update-1 if role=='train' else global_update) and
                job.get('evaluation_only') is (role!='train') and
                job.get('ppo_trainable') is (role=='train') and
                job.get('partition')==('train' if role=='train' else 'validation') and
                job.get('action_selection')=='categorical' and
                job.get('actor_sampling_seed_usage')=='consumed' and
                job.get('recording_id')==job.get('source_recording_id')==CAMPAIGN_ID+'-'+phase_id and
                float(job.get('wait_weight',-1))==WAIT_WEIGHTS[stage],
                'stage/global/local/checkpoint/train-evaluation identity differs')
        pos+=count
    require(pos==CAP and seed_value('initialization',0,nonce) not in seen_seeds,'canonical schedule or initialization/cross-type seed collision')
    return jobs


def check_phase(cfg):
    from .features import feature_contract, physical_contract
    from .targets import composite_contract
    stage,name,local=cfg.get('stage_id'),cfg.get('arm_phase_id'),cfg.get('arm_update')
    matched=[x for x in schedule() if (x[0],x[1],x[3])==(stage,name,local)]
    require(len(matched)==1 and cfg.get('contract')==PHASE_CONTRACT and
        cfg.get('campaign_id')==cfg.get('experiment_id')==CAMPAIGN_ID and
        cfg.get('phase_id')==stage+'_'+name and cfg.get('candidate_id')==CANDIDATE and
        cfg.get('execution_profile')==PROFILE and cfg.get('purpose')==PURPOSE,
        'new physical single-model RPC phase contract required')
    role=matched[0][2]
    output=ROOT/'runs'/CAMPAIGN_ID
    require(Path(cfg['python_output'])==output/'arms'/stage/name and
        Path(cfg['event_directory'])==output/'control'/'legacy_events_disabled'/stage/name,
        'phase output must use its unique campaign/stage/phase directory')
    compatible=copy.deepcopy(cfg);compatible['contract']=baseline.PHASE_CONTRACT
    baseline.check_phase(compatible)
    require(len(cfg['jobs'])==matched[0][4] and cfg.get('evaluation_only') is (role!='train') and
        cfg.get('ppo_trainable') is (role=='train') and cfg.get('action_selection')=='categorical' and
        cfg.get('optimizer_steps')==(None if role=='train' else 0),
        'phase exact count/train-evaluation/zero evaluation optimizer scope differs')
    require(cfg.get('feature_contract')==feature_contract() and
        cfg.get('physical_feature_contract')==physical_contract() and cfg.get('material_memory')==MEMORY and
        cfg.get('composite_contract')==composite_contract(WAIT_WEIGHTS[stage]) and
        cfg.get('history_contract')==HISTORY_CONTRACT and cfg.get('diagnostic_scope')==DIAGNOSTIC_SCOPE and
        cfg.get('wait_diagnostics_contract')==WAIT_DIAGNOSTICS_CONTRACT and
        cfg.get('training_reward')==TRAINING_REWARD,'input/physical46/reward/history contract changed')
    template=json.loads(TEMPLATE_PATH.read_bytes())
    require(all(cfg.get(k)==template[k] for k in ('reward','environment_acceptance','execution','storage',
        'capture_physical_events','diagnostic_recording')),'physical/RPC/storage/logging contract changed')
    expected_rpc=copy.deepcopy(template['jobs'][0]['expected_rpc_config']);expected_rpc.update(WAIT_RPC_FIELDS)
    require(cfg.get('wait_diagnostics_required') is True,'formal608 requires enabled waiting diagnostics')
    for job in cfg['jobs']:
        episode='dispatch-production-wait-shortfall-v2-run-%d-seed-%d'%(job['run_index'],job['experiment_seed'])
        material=ROOT/'runs'/(CAMPAIGN_ID+'_materials')/stage/name
        require(job.get('phase_id')==cfg['phase_id'] and job.get('stage_id')==stage and job.get('role')==role and
            job.get('expected_rpc_config')==expected_rpc and
            all(job.get(k)==template['jobs'][0][k] for k in ('scene_id','runtime_scene_sha256')) and
            Path(job['material_directory'])==Path(job['material_event_directory'])==material and
            Path(job['material_event_bundle'])==material/episode and
            Path(job['material_journal'])==material/episode/'material_events.bin',
            'RPC configuration/material identity differs')
    return cfg


def check_approval(approval,suite,suite_sha256):
    require(isinstance(approval,dict) and approval.get('decision')=='approved' and
        approval.get('approved_by')=='user' and approval.get('execution_authorized') is True and
        isinstance(approval.get('source_user_instruction'),str) and bool(approval['source_user_instruction'].strip()) and
        approval.get('campaign_id')==approval.get('experiment_id')==CAMPAIGN_ID and
        approval.get('campaign_sha256')==suite_sha256 and approval.get('max_new_shifts')==CAP and
        approval.get('maximum_total_after_registration')==suite['maximum_total_if_approved'] and
        approval.get('legacy_reserved_shifts')==approval.get('active_h2_reserved_shifts')==0 and
        approval.get('approved_stages')==list(STAGES) and approval.get('ack_wait_weights')==WAIT_WEIGHTS and
        all(approval.get(k) is True for k in ('ack_single_model_P_only','ack_physical46_residual',
          'ack_pure_grade_named_weights_new_adam','ack_stage2_full_adam_continuation',
          'ack_fixed_stage1_and_final_checkpoints','ack_no_automatic_retries_or_extra_training')),
        'explicit608 singleP/physical46/staged optimizer and execution approval required')
    return approval


def preflight(path,*,approval_path=None):
    from .budget import budget_snapshot,budget_state_reference,check_collisions
    from .learner import default_config,validate_config
    from .features import feature_contract,physical_contract
    from .targets import composite_contract
    path=Path(path).resolve();suite=json.loads(path.read_bytes())
    require(suite.get('contract')==SUITE_CONTRACT and suite.get('experiment_id')==CAMPAIGN_ID and
        suite.get('single_model') is True and suite.get('logical_model_id')=='P' and
        suite.get('legacy_reserved_shifts')==suite.get('active_h2_reserved_shifts')==0 and
        type(suite.get('cpu_threads')) is int and suite.get('cpu_threads')==2 and
        suite.get('kind')=='single_seed_trial' and suite.get('execution_profile')==PROFILE and
        suite.get('purpose')==PURPOSE and suite.get('initialization')==INITIALIZATION and
        suite.get('training_shifts')==512 and suite.get('training_updates')==16 and
        suite.get('initial_evaluation_shifts')==16 and suite.get('monitor_evaluation_shifts')==16 and
        suite.get('final_evaluation_shifts')==64 and suite.get('anylogic_start_count')==1 and
        suite.get('arm_order')==list(STAGES) and suite.get('feature_contract')==feature_contract() and
        suite.get('physical_feature_contract')==physical_contract() and
        suite.get('stage_reward_contracts')=={s:composite_contract(WAIT_WEIGHTS[s]) for s in STAGES} and
        suite.get('parent_artifacts')==parent_references(),'exact608 single-model staged suite differs')
    check_jobs(suite);check_sources(suite['source_pins'])
    output=ROOT/'runs'/CAMPAIGN_ID
    require(Path(suite['python_output'])==output and Path(suite['control_directory'])==output/'control',
        'independent physical PPO output identity required')
    for arm,stage in zip(suite['arms'],STAGES):
        expected=default_config(stage_id=stage,seed=suite['actor_seed'],candidate_id=CANDIDATE,
            source_kind='real_anylogic_rpc',execution_profile=PROFILE)
        require(arm['stage_id']==stage and validate_config(arm['learner_config'])==expected and
            Path(arm['python_output'])==output/'arms'/stage,'stage learner config/optimizer changed')
    require(len(suite['arms'])==2 and suite['actor_seed']==seed_value('initialization',0,suite['seed_plan_nonce']),
        'single model stage initialization differs')
    phases=[check_phase(load_reference(r)) for r in suite['phase_configs']]
    require(len(phases)==19 and [j for p in phases for j in p['jobs']]==suite['jobs'],'phase/job schedule changed')
    for key in ('master_plan','java_helper','rpc_template_reference','engineering_acceptance'):
        require(reference(suite[key]['path'])==suite[key],'frozen reference changed: '+key)
    with Path(suite['master_plan']['path']).open(newline='') as stream:
        require(list(csv.DictReader(stream))==[canonical_master_row(j) for j in suite['jobs']],
            'master identity bytes differ from suite')
    from .java_plan import render_helper
    require(Path(suite['java_helper']['path']).read_text()==render_helper(suite['master_plan']['path'],suite['jobs']),
        'Java helper differs from reviewed renderer')
    acceptance=load_reference(suite['engineering_acceptance'])
    require(acceptance.get('status')=='PASS_SYNTHETIC_ONLY' and acceptance.get('source_pins')==suite['source_pins'] and
        all(acceptance.get('gates',{}).get(k)=='PASS_SYNTHETIC_ONLY' for k in ('G1','G2','G3','G4')),
        'matching G1-G4 engineering acceptance required')
    from python.protocol_artifacts import verify_protocol_artifacts
    from python.collect_offline import verify_environment
    verify_protocol_artifacts()
    for p in phases:verify_environment(p['environment_acceptance'],p['reward'],storage_migration=True)
    budget=budget_snapshot();state=budget_state_reference()
    if CAMPAIGN_ID not in budget['campaigns']:
        require(state==suite['budget_state_before'],'current merged budget changed since preparation')
        check_collisions(suite,budget)
        require(not output.exists() and not Path(str(output)+'_materials').exists(),'output identity already used')
        previous=budget['maximum_total_shifts']
    else:
        require(budget['campaigns'][CAMPAIGN_ID]['campaign_sha256']==sha(path),'registered suite changed')
        previous=budget['maximum_total_shifts']-CAP
    require(suite['maximum_total_if_approved']==previous+CAP,'new608 budget arithmetic differs')
    approval=None if approval_path is None else check_approval(json.loads(Path(approval_path).read_bytes()),suite,sha(path))
    return suite,phases,approval
