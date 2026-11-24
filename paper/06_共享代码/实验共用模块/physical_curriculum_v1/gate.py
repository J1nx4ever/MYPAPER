"""Immutable 6,844-slot curriculum experiment contracts; reads never execute."""
from __future__ import annotations
import copy
import csv
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from baseline1_v1 import gate as baseline
from physical_wait_tradeoff_v1 import gate as previous
from grade_only_ppo_v1.gate import MEMORY
ROOT, REPO, PROTECTED_SHA = baseline.ROOT, baseline.REPO, baseline.PROTECTED_SHA
require, sha, reference, load_reference = baseline.require, baseline.sha, baseline.reference, baseline.load_reference

def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()

def digest(value): return hashlib.sha256(encoded(value)).hexdigest()
PLAN=ROOT/'change_plans/physical_realization_curriculum_control_001'
CAP, TRAIN_UPDATES=6844,160
INITIALIZATION_SEED=58688234705239
ORIGINAL_SEED=93468885541233
CAMPAIGN_ID=EXPERIMENT_ID='physical-curriculum-control-seed1-20261110-001'
SUITE_CONTRACT=CAMPAIGN_CONTRACT='physical-curriculum-suite-v1'
PHASE_CONTRACT='physical-curriculum-rpc-phase-v1'
READY_PROTOCOL='physical-curriculum-ready-v1'
PROFILE='formal_32ep_v1'
PURPOSE='matched_curriculum_vs_fixed_wait_physical46'
CANDIDATE='a1_actor_gru'
INITIALIZATION='shared_pure_grade_u8_plus_physical46_common_initial_full_adam_continuation'
TRAINING_REWARD=previous.TRAINING_REWARD
STAGES=('reference','wait015','wait02','fixed015','fixed02')
TRAIN_STAGES=STAGES[1:]
WAIT_WEIGHTS=dict(reference=.1,wait015=.15,wait02=.2,fixed015=.15,fixed02=.2)
PARENT_PATH=previous.PARENT_PATH; PARENT_SHA=previous.PARENT_SHA
COMMON_INITIAL_PATH=ROOT/'runs/physical-realization-ppo-seed1-20261110-001/arms/production/initial.pt'
COMMON_INITIAL_SHA='8421e826a6c16cf38fb1a9b3d8d9f538a3895d0f655e664aa8c1c32ff15fab42'
HISTORICAL_SUITE_PATH=ROOT/'change_plans/physical_realization_wait_tradeoff_001/prepared_002/suite.json'
HISTORICAL_SUITE_SHA='0270fa512e4f2b6954110730b72a550f48c31637190b22ecc672cfbd53f6d5d5'
PHYSICAL_SUITE_PATH=ROOT/'change_plans/physical_realization_ppo_001/prepared/suite.json'
TEMPLATE_PATH=previous.TEMPLATE_PATH
JAVA_TEMPLATE_PATH=previous.JAVA_TEMPLATE_PATH
COLUMNS=previous.COLUMNS
HISTORY_CONTRACT=previous.HISTORY_CONTRACT
DIAGNOSTIC_SCOPE=dict(previous.DIAGNOSTIC_SCOPE,scope=PURPOSE)
ACCEPTANCE=dict(stop_rule='fixed_common_1536_budget_no_automatic_plateau_stop',
    monitoring_nodes=[768,1024,1280,1536],monitor_scenarios=64,blind_scenarios=100,
    benchmark_scenarios=100,benchmark_models=['reference','wait015','wait02'],
    four_models=list(TRAIN_STAGES),platform_claim_requires_separate_statistical_review=True,
    unknown_and_claimed_never_automatically_retried=True,action_selection='categorical')

def actor_seed(stage): return ORIGINAL_SEED if stage.startswith('fixed') else INITIALIZATION_SEED

def cumulative_update(stage, local): return 16 if stage=='reference' else local+(16 if stage in ('wait015','wait02') else 0)

def schedule():
    phases=[]; prior=0
    for node in (24,32,40,48):
        for stage in TRAIN_STAGES:
            start=max(prior,16) if stage in ('wait015','wait02') else prior
            for cumulative in range(start+1,node+1):
                local=cumulative-(16 if stage in ('wait015','wait02') else 0)
                phases.append((stage,'train_u%04d'%local,'train',local,32))
        if node==24:
            phases.extend((stage,'benchmark_0100','final',0 if stage=='reference' else 8,100)
                          for stage in ('reference','wait015','wait02'))
        phases.extend((stage,'monitor_n%04d'%(node*32),'monitor',
                       node-(16 if stage in ('wait015','wait02') else 0),64) for stage in TRAIN_STAGES)
        prior=node
    phases.extend((stage,'blind_0100','final',32 if stage in ('wait015','wait02') else 48,100)
                  for stage in TRAIN_STAGES)
    return phases

@lru_cache(maxsize=1)
def historical_suite():
    require(sha(HISTORICAL_SUITE_PATH)==HISTORICAL_SUITE_SHA,'preserved RAB suite bytes changed')
    return json.loads(HISTORICAL_SUITE_PATH.read_bytes())

@lru_cache(maxsize=1)
def physical_suite(): return json.loads(PHYSICAL_SUITE_PATH.read_bytes())

def historical_pairing_contract():
    return dict(schema='physical-curriculum-seed-segments-v1',
        original_physical=reference(PHYSICAL_SUITE_PATH),RAB=reference(HISTORICAL_SUITE_PATH),
        original_optimizer_seed=ORIGINAL_SEED,optimizer_seed_after_512=INITIALIZATION_SEED,
        shared_train_positions=[1,1536],benchmark='historical_RAB_100_development_only',
        monitor='fresh64_repeated_by_model_and_node',blind='fresh100_disjoint_once_after_all_frozen',
        initial_pure_grade_budget_recounted=False)

@lru_cache(maxsize=1)
def _sources_index():
    physical=physical_suite(); rab=historical_suite()
    first=[j for j in physical['jobs'] if j['role']=='train' and j['stage_id']=='production']
    second=[j for j in physical['jobs'] if j['role']=='train' and j['stage_id']=='wait01']
    third=[j for j in rab['jobs'] if j['role']=='train' and j['stage_id']=='wait015']
    bench=[j for j in rab['jobs'] if j['role']=='final' and j['stage_id']=='reference']
    require(len(first)==len(second)==len(third)==256 and len(bench)==100,'source seed inventory differs')
    return first+second+third,bench

@lru_cache(maxsize=1)
def _physical_sha(): return sha(PHYSICAL_SUITE_PATH)

def _new_seed(kind,index,field):
    return 1+int.from_bytes(hashlib.sha256((CAMPAIGN_ID+'/'+kind+'/'+str(index)+'/'+field).encode()).digest()[:6],'big')%(2**48-1)

def seed_record(stage,name,local,slot):
    history,bench=_sources_index()
    if name.startswith('train_'):
        index=(cumulative_update(stage,local)-1)*32+slot
        old=history[index] if index<768 else None; kind='train'
    elif name=='benchmark_0100': index=slot;old=bench[slot];kind='benchmark'
    else: index=slot;old=None;kind='monitor' if name.startswith('monitor_') else 'blind'
    result=dict(pair_id=('%s-%04d'%(kind,index)),pair_ordinal=index,
                experiment_seed=_new_seed(kind,index,'environment'),actor_sampling_seed=_new_seed(kind,index,'action'))
    if old:
        result.update(experiment_seed=old['experiment_seed'],actor_sampling_seed=old['actor_sampling_seed'],
          historical_source_run_index=old['run_index'],historical_source_recording_id=old['recording_id'],
          historical_source_suite_sha256=_physical_sha() if kind=='train' and index<512 else HISTORICAL_SUITE_SHA)
    return result

def historical_job(stage,role,local,index):
    return seed_record(stage,'train_u%04d'%local if role=='train' else 'benchmark_0100',local,index)

def _parent(path, expected):
    require(sha(path)==expected,'immutable parent checkpoint changed')
    refs={k:reference(p) for k,p in dict(checkpoint=path,manifest=Path(str(path)+'.manifest.json'),seal=Path(str(path)+'.seal.json')).items()}
    manifest,seal=load_reference(refs['manifest']),load_reference(refs['seal'])
    require(manifest['sha256']==seal['sha256']==expected and seal['manifest_sha256']==refs['manifest']['sha256'],'parent manifest/seal changed')
    return refs

def parent_references():
    result=_parent(PARENT_PATH,PARENT_SHA)
    common=_parent(COMMON_INITIAL_PATH,COMMON_INITIAL_SHA)
    result.update(common_initial=common['checkpoint'],common_initial_manifest=common['manifest'],common_initial_seal=common['seal'])
    return result

def source_pins():
    pins=previous.source_pins()
    pins.update({str(p.resolve()):sha(p) for p in Path(__file__).parent.glob('*.py') if not p.name.startswith('test_')})
    return dict(sorted(pins.items()))

def check_sources(pins):
    require(pins==source_pins(),'curriculum implementation changed; freeze a new package')
    require(sha(REPO/'python/episode_budget_ppo.py')==PROTECTED_SHA,'protected PPO changed')

def common_config():
    from .features import feature_contract,physical_contract
    from .waiting_diagnostics import WAIT_DIAGNOSTICS_CONTRACT,WAIT_RPC_FIELDS
    template=json.loads(TEMPLATE_PATH.read_bytes())
    result={k:copy.deepcopy(template[k]) for k in ('reward','environment_acceptance','execution','storage','capture_physical_events','diagnostic_recording')}
    result.update(material_memory=copy.deepcopy(MEMORY),feature_contract=feature_contract(),physical_feature_contract=physical_contract(),
        history_contract=HISTORY_CONTRACT,diagnostic_scope=DIAGNOSTIC_SCOPE,wait_diagnostics_contract=WAIT_DIAGNOSTICS_CONTRACT,
        wait_diagnostics_required=True,training_reward=TRAINING_REWARD,rpc_configuration_authoritative=True,
        require_local_alp_audit=False,kind='single_seed_trial',execution_profile=PROFILE,purpose=PURPOSE,
        scene_id=template['jobs'][0]['scene_id'],runtime_scene_sha256=template['jobs'][0]['runtime_scene_sha256'],
        expected_rpc_config=dict(template['jobs'][0]['expected_rpc_config'],**WAIT_RPC_FIELDS))
    return result

def expand_phase(spec, common):
    from .targets import composite_contract
    stage,name,role,local,count=spec['stage_id'],spec['arm_phase_id'],spec['role'],spec['arm_update'],spec['count']
    qualified=stage+'_'+name; output=ROOT/'runs'/CAMPAIGN_ID
    material=ROOT/'runs'/(CAMPAIGN_ID+'_materials')/stage/name
    events=output/'control/legacy_events_disabled'/stage/name
    evaluation=role!='train'; kind='train' if not evaluation else ('benchmark' if name.startswith('benchmark') else ('blind' if name.startswith('blind') else 'monitor'))
    cfg=copy.deepcopy(common)
    cfg.update(contract=PHASE_CONTRACT,campaign_id=CAMPAIGN_ID,experiment_id=EXPERIMENT_ID,stage_id=stage,phase_id=qualified,
        arm_phase_id=name,arm_update=local,cumulative_update=cumulative_update(stage,local),candidate_id=CANDIDATE,
        recording_id=CAMPAIGN_ID+'-'+qualified,python_output=str(output/'arms'/stage/name),event_directory=str(events),
        composite_contract=composite_contract(WAIT_WEIGHTS[stage]),historical_pairing_contract=historical_pairing_contract(),
        evaluation_only=evaluation,ppo_trainable=not evaluation,action_selection='categorical',actor_sampling_seed_usage='consumed',
        optimizer_steps=0 if evaluation else None,eval_kind=kind,behavior=None,jobs=[],
        cumulative_shifts=768 if kind=='benchmark' else cumulative_update(stage,local)*32,
        required_committed_updates=spec['required_committed_updates'])
    for slot in range(count):
        seed=seed_record(stage,name,local,slot);ordinal=spec['ordinal_start']+slot;run=9110001+ordinal
        episode='dispatch-production-wait-shortfall-v2-run-%d-seed-%d'%(run,seed['experiment_seed'])
        job=dict(seed,ordinal=ordinal,actor_seed=actor_seed(stage),group='BASELINE',role=role,
            source_recording_id=cfg['recording_id'],recording_id=cfg['recording_id'],run_index=run,
            event_directory=str(events),event_bundle=str(events/('run_%d.events.jsonl'%run)),python_output=cfg['python_output'],
            target_training_update=spec['target_training_update'],checkpoint_update=local-1 if not evaluation else local,
            stage_id=stage,phase_id=qualified,arm_phase_id=name,arm_update=local,cumulative_update=cfg['cumulative_update'],
            candidate_id=CANDIDATE,material_directory=str(material),wait_weight=WAIT_WEIGHTS[stage],
            partition='validation' if evaluation else 'train',scene_id=common['scene_id'],
            scene_group=CAMPAIGN_ID+'-'+seed['pair_id'],realization_group_id=CAMPAIGN_ID+'-'+seed['pair_id'],
            runtime_scene_sha256=common['runtime_scene_sha256'],expected_rpc_config=common['expected_rpc_config'],
            material_event_directory=str(material),material_event_bundle=str(material/episode),material_journal=str(material/episode/'material_events.bin'),
            evaluation_only=evaluation,ppo_trainable=not evaluation,action_selection='categorical',actor_sampling_seed_usage='consumed',eval_kind=kind)
        cfg['jobs'].append(job)
    cfg['freeze_group_id']=kind+'_%04d'%cfg['cumulative_shifts'] if evaluation else 'train'
    return cfg

def phase_specs():
    result=[];ordinal=0;update=0
    for stage,name,role,local,count in schedule():
        if role=='train': update+=1
        result.append(dict(stage_id=stage,arm_phase_id=name,role=role,arm_update=local,count=count,
            ordinal_start=ordinal,target_training_update=update if role=='train' else 0,required_committed_updates=update))
        ordinal+=count
    require(ordinal==CAP and update==TRAIN_UPDATES and len(result)==183,'canonical curriculum count differs')
    return result

def load_phase(path_or_ref):
    raw=load_reference(path_or_ref) if isinstance(path_or_ref,dict) else json.loads(Path(path_or_ref).read_bytes())
    require(raw.get('schema')=='physical-curriculum-compact-phase-v1','compact phase contract required')
    common=load_reference(raw['common'])
    require(raw['spec'] in phase_specs(),'noncanonical phase spec')
    cfg=expand_phase(raw['spec'],common)
    require(digest(cfg)==raw['expanded_sha256'],'expanded phase semantic SHA differs')
    return cfg

def load_suite(path):
    raw=json.loads(Path(path).read_bytes())
    require(raw.get('contract')==SUITE_CONTRACT and 'jobs' not in raw,'compact suite contract required')
    phases=[load_phase(ref) for ref in raw['phase_configs']]
    jobs=[j for p in phases for j in p['jobs']]
    require(digest(jobs)==raw['jobs_sha256'],'expanded suite job digest differs')
    return dict(raw,jobs=jobs,_suite_path=str(Path(path).resolve()))

def canonical_master_row(job): return {k:str(job[k]) for k in COLUMNS}

def check_jobs(suite):
    require(suite.get('campaign_id')==CAMPAIGN_ID and suite.get('new_shift_cap')==CAP,'exact6844 identity required')
    jobs=suite.get('jobs');expected=[j for spec in phase_specs() for j in expand_phase(spec,common_config())['jobs']]
    require(jobs==expected,'jobs must exactly match canonical train/monitor/benchmark/blind schedule')
    train={j[k] for j in jobs if j['eval_kind']=='train' for k in ('experiment_seed','actor_sampling_seed')}
    sets={kind:{j[k] for j in jobs if j['eval_kind']==kind for k in ('experiment_seed','actor_sampling_seed')} for kind in ('monitor','blind','benchmark')}
    require(len(train)==3072 and all(train.isdisjoint(v) for v in sets.values()),'training/evaluation seed overlap')
    require(len(sets['monitor'])==128 and len(sets['blind'])==len(sets['benchmark'])==200 and
        all(sets[a].isdisjoint(sets[b]) for a,b in (('monitor','blind'),('monitor','benchmark'),('blind','benchmark'))),'evaluation split leakage')
    return jobs

def check_phase(cfg):
    spec=next((s for s in phase_specs() if s['stage_id']==cfg.get('stage_id') and s['arm_phase_id']==cfg.get('arm_phase_id')),None)
    require(spec is not None and cfg==expand_phase(spec,common_config()),'frozen phase differs')
    compatible=dict(cfg,contract=baseline.PHASE_CONTRACT);baseline.check_phase(compatible)
    return cfg

def approval_contract(suite,suite_reference,user_instruction):
    from datetime import datetime,timezone
    require(isinstance(user_instruction,str) and bool(user_instruction.strip()),'explicit Build-complete execution instruction required')
    return dict(decision='approved',approved_by='user',execution_authorized=True,
        source_user_instruction=user_instruction.strip(),campaign_id=CAMPAIGN_ID,experiment_id=EXPERIMENT_ID,
        campaign_sha256=suite_reference['sha256'],max_new_shifts=CAP,maximum_total_after_registration=suite['maximum_total_if_approved'],
        legacy_reserved_shifts=0,active_h2_reserved_shifts=0,budget_policy=suite['budget_policy'],
        approved_stages=list(STAGES),ack_wait_weights=WAIT_WEIGHTS,
        ack_common_grade_pretrained_initial=True,ack_fixed_1536_budget=True,ack_monitoring_not_blind=True,
        ack_historical_seed_reuse_same_partition=True,historical_pairing_contract=historical_pairing_contract(),
        ack_all_physical_results_newly_simulated=True,ack_no_automatic_retries_or_extra_training=True,
        ack_old_RAB_not_executed_separately=True,
        training_shifts=5120,evaluation_shifts=1724,monitor_evaluation_shifts=1024,
        blind_evaluation_shifts=400,development_evaluation_shifts=300,
        execution_profile=PROFILE,accounting='additional_to_completed4620_includes_unstarted_RAB812',
        approved_at_utc=datetime.now(timezone.utc).isoformat(),approval_scope='one_frozen_campaign_budget_and_execution')

def check_approval(approval,suite,suite_sha256):
    require(approval.get('decision')=='approved' and approval.get('approved_by')=='user' and
        approval.get('execution_authorized') is True and bool(approval.get('source_user_instruction')) and
        approval.get('campaign_id')==CAMPAIGN_ID and approval.get('campaign_sha256')==suite_sha256 and
        approval.get('max_new_shifts')==CAP and approval.get('maximum_total_after_registration')==suite['maximum_total_if_approved'] and
        approval.get('active_h2_reserved_shifts')==approval.get('legacy_reserved_shifts')==0 and approval.get('approved_stages')==list(STAGES) and
        approval.get('ack_wait_weights')==WAIT_WEIGHTS and approval.get('historical_pairing_contract')==historical_pairing_contract() and
        all(approval.get(k) is True for k in ('ack_common_grade_pretrained_initial','ack_fixed_1536_budget','ack_monitoring_not_blind',
            'ack_historical_seed_reuse_same_partition','ack_all_physical_results_newly_simulated',
            'ack_no_automatic_retries_or_extra_training','ack_old_RAB_not_executed_separately')) and
        all(approval.get(k)==v for k,v in dict(training_shifts=5120,evaluation_shifts=1724,monitor_evaluation_shifts=1024,
            blind_evaluation_shifts=400,development_evaluation_shifts=300).items()),
        'explicit frozen6844 scope and Build-complete execution instruction required')
    return approval

def preflight(path,*,approval_path=None):
    from .budget import budget_snapshot,budget_state_reference,check_collisions,GLOBAL_PATH
    from .learner import default_config,validate_config
    path=Path(path).resolve();suite=load_suite(path);phases=[load_phase(ref) for ref in suite['phase_configs']]
    require(suite.get('contract')==SUITE_CONTRACT and suite.get('training_shifts')==5120 and suite.get('training_updates')==160
        and suite.get('monitoring_shifts')==1024 and suite.get('final_evaluation_shifts')==700 and suite.get('benchmark_evaluation_shifts')==300
        and suite.get('blind_evaluation_shifts')==400 and suite.get('anylogic_start_count')==1
        and suite.get('acceptance_standard')==ACCEPTANCE and suite.get('parent_artifacts')==parent_references()
        and suite.get('arm_order')==list(STAGES) and suite.get('cpu_threads')==2,'frozen curriculum scope differs')
    check_jobs(suite);check_sources(suite['source_pins'])
    require(len(phases)==183 and all(check_phase(p) for p in phases),'phase contract differs')
    require(suite.get('historical_pairing_contract')==historical_pairing_contract(),'historical pairing changed')
    for arm,stage in zip(suite['arms'],STAGES):
        expected=default_config(seed=actor_seed(stage),candidate_id=CANDIDATE,stage_id=stage,wait_weight=WAIT_WEIGHTS[stage],source_kind='real_anylogic_rpc',execution_profile=PROFILE)
        require(arm['stage_id']==stage and validate_config(arm['learner_config'])==expected,'frozen arm config differs')
    output=ROOT/'runs'/CAMPAIGN_ID
    require(suite['python_output']==str(output) and suite['control_directory']==str(output/'control') and suite['budget_path']==str(GLOBAL_PATH),'dedicated campaign paths differ')
    for key in ('master_plan','java_helper','common_config','engineering_acceptance'):
        require(reference(suite[key]['path'])==suite[key],'frozen file changed: '+key)
    engineering=load_reference(suite['engineering_acceptance'])
    require(engineering.get('status')=='PASS_SYNTHETIC_ONLY' and engineering.get('source_pins')==suite['source_pins'] and
        all(engineering.get('gates',{}).get(k)=='PASS_SYNTHETIC_ONLY' for k in ('G1','G2','G3','G4')),'engineering acceptance differs')
    with Path(suite['master_plan']['path']).open(newline='') as handle:
        require(list(csv.DictReader(handle))==[canonical_master_row(j) for j in suite['jobs']],'master rows differ')
    from .java_plan import render_helper
    require(Path(suite['java_helper']['path']).read_text()==render_helper(suite['master_plan']['path'],suite['jobs']),'helper differs')
    snapshot=budget_snapshot();state=budget_state_reference()
    if CAMPAIGN_ID not in snapshot['campaigns']:
        require(state==suite['budget_state_before'],'budget changed after freeze');check_collisions(suite,snapshot)
        require(suite['maximum_total_if_approved']==snapshot['maximum_total_shifts']+CAP,'capacity differs')
    else:
        require(snapshot['campaigns'][CAMPAIGN_ID]['campaign_sha256']==sha(path) and
            suite['maximum_total_if_approved']==snapshot['maximum_total_shifts'],'registered suite differs')
    approval=None if approval_path is None else check_approval(json.loads(Path(approval_path).read_bytes()),suite,sha(path))
    return suite,phases,approval
