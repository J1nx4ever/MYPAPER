"""Frozen 576-shift summary/ActorGRU study; configuration admission only."""
import copy
import csv
import json
from pathlib import Path

from grade_only_ab_v1 import gate as inherited
from grade_only_ppo_v1 import gate as frozen
from baseline1_v1.budget import GLOBAL_PATH

ROOT,REPO,PROTECTED_SHA=inherited.ROOT,inherited.REPO,inherited.PROTECTED_SHA
encoded,reference,sha,require=inherited.encoded,inherited.reference,inherited.sha,inherited.require
load_reference=inherited.load_reference
MEMORY,DIAGNOSTIC_SCOPE,TRAINING_REWARD=inherited.MEMORY,inherited.DIAGNOSTIC_SCOPE,inherited.TRAINING_REWARD
budget_snapshot=inherited.budget_snapshot
PROFILE='formal_32ep_v1'
CAMPAIGN_CONTRACT='grade-commitment-fast-paired-suite-v1'
PHASE_CONTRACT='grade-commitment-fast-paired-phase-v1'
PURPOSE='summary_actor_recurrence_comparison_shared_decision_gru_critic'
CANDIDATES=('a0_summary','a1_actor_gru')
PLAN=ROOT/'change_plans/grade_commitment_training_speed_001'
PAIR_ID='grade-commitment-fast-pair-seed1-20261109-001'
PAIR_PATH=PLAN/'prepared/paired_seeds.json'
INITIALIZATION='fresh_shared_actor_base_and_critic_zero_residual_gru_no_pretraining'


def source_pins():
    from grade_commitment_ppo_v1.gate import source_pins as legacy_pins
    pins=legacy_pins()
    pins.update({str(p.resolve()):sha(p) for p in (ROOT/'grade_commitment_ppo_fast_v1').glob('*.py')
                 if not p.name.startswith('test_')})
    return dict(sorted(pins.items()))


def check_sources(pins):
    require(pins==source_pins(),'paired-suite source inventory changed; freeze a new package')
    require(sha(REPO/'python/episode_budget_ppo.py')==PROTECTED_SHA,'protected PPO core changed')


def load_paired_seeds(ref):
    proposal=load_reference(ref)
    return validate_paired_seed_document(proposal)


def validate_paired_seed_document(proposal):
    require(proposal.get('schema')=='grade-commitment-fast-paired-seeds-v1'
            and proposal.get('seed_namespace')==PAIR_ID and proposal.get('actor_initialization_seed')==2026110901
            and proposal.get('columns')==['ordinal','environment_seed','actor_sampling_seed','a_run_index','b_run_index'],
            'paired-seed contract differs')
    rows=proposal.get('rows')
    require(isinstance(rows,list) and len(rows)==288 and all(isinstance(r,list) and len(r)==5 for r in rows),
            'complete 288-row paired seed table required')
    require([r[0] for r in rows]==list(range(288)) and
            all(all(type(x) is int and x>0 for x in r[1:]) for r in rows), 'paired ordinal/identity malformed')
    require(len({r[1] for r in rows})==len({r[2] for r in rows})==288
            and len({r[k] for r in rows for k in (3,4)})==576,
            'paired table reuses seed outside the corresponding A/B ordinal')
    from baseline1_v1.prepare import _seed
    require(rows==[[i,_seed(PAIR_ID,'environment',i),_seed(PAIR_ID,'actor',i),8970001+i,8980001+i]
                   for i in range(288)],'new commitment study must use its exact independent deterministic identities')
    return proposal


def _local_schedule(pair_ordinal):
    if pair_ordinal<16:return 't0','t0',0
    if pair_ordinal>=272:return 'final_u0008','final',8
    update=(pair_ordinal-16)//32+1
    return 'train_u%04d'%update,'train',update


def validate_pair_jobs(jobs,paired_seed_reference=None,*,pair_id=PAIR_ID,proposal=None):
    """Only this exact A/B pair may share seeds, once per corresponding row."""
    require(pair_id==PAIR_ID,'unknown pairing authorization scope')
    proposal=load_paired_seeds(paired_seed_reference) if proposal is None else validate_paired_seed_document(proposal)
    require(isinstance(jobs,list) and len(jobs)==576,'exact 576 physical jobs required')
    for ordinal,job in enumerate(jobs):
        arm=ordinal//288;pair=ordinal%288;candidate=CANDIDATES[arm];row=proposal['rows'][pair]
        arm_phase,role,local=_local_schedule(pair)
        require(job.get('ordinal')==ordinal and job.get('pair_ordinal')==pair
                and job.get('candidate_id')==candidate and job.get('arm_phase_id')==arm_phase
                and job.get('phase_id')==candidate+'_'+arm_phase and job.get('arm_update')==local
                and job.get('role')==role and job.get('group')=='BASELINE'
                and job.get('target_training_update')==arm*8+local
                and job.get('checkpoint_update')==(local-1 if role=='train' else local)
                and job.get('actor_seed')==2026110901
                and job.get('partition')==('train' if role=='train' else 'validation'),
                'paired job candidate/ordinal/role/update differs at '+str(ordinal))
        require(job.get('run_index')==row[3+arm] and job.get('experiment_seed')==row[1]
                and job.get('actor_sampling_seed')==row[2],
                'paired seed/run permission does not cover this exact job')
    require(len({j['run_index'] for j in jobs})==576,'physical run IDs must remain globally unique')
    return jobs


def check_collisions(suite,budget):
    # The inherited check compares only with historical ledger identities.
    # Internal duplicate seeds are admitted only by validate_pair_jobs above;
    # original/new run, recording and filesystem identities are never reusable.
    return frozen.check_collisions(suite,budget)


def check_phase(cfg):
    require(cfg.get('contract')==PHASE_CONTRACT and cfg.get('execution_profile')==PROFILE
            and cfg.get('purpose')==PURPOSE and cfg.get('candidate_id') in CANDIDATES
            and cfg.get('pair_id')==PAIR_ID,
            'formal paired phase scope required')
    require(cfg.get('phase_id')==cfg['candidate_id']+'_'+cfg.get('arm_phase_id',''),
            'phase must explicitly bind its candidate and local phase')
    compatible=copy.deepcopy(cfg);compatible['contract']=frozen.PHASE_CONTRACT
    frozen.check_phase(compatible)
    from .features import feature_contract
    require(cfg.get('feature_contract')==feature_contract(), 'decision-summary feature contract differs')
    require(cfg.get('history_contract')==dict(scope='whole_shift',window_reset=False,terminal_reset=True,
            source='causal_decision_summaries',critic='decision_gru_both_arms'),
            'whole-shift decision-history contract required')
    return cfg


def check_approval(approval,suite,suite_sha256):
    require(approval.get('decision')=='approved' and approval.get('approved_by')=='user'
            and approval.get('execution_authorized') is True and bool(approval.get('source_user_instruction'))
            and approval.get('ack_grade_only_diagnostic') is True and approval.get('ack_summary_actor_gru_comparison') is True
            and approval.get('execution_profile')==suite.get('execution_profile')==PROFILE
            and approval.get('pair_id')==suite.get('pair_id')==PAIR_ID
            and approval.get('campaign_id')==suite['campaign_id'] and approval.get('campaign_sha256')==suite_sha256
            and approval.get('max_new_shifts')==suite.get('new_shift_cap')==576
            and approval.get('legacy_reserved_shifts')==11280
            and approval.get('accounting')=='additional_to_h2_reserved'
            and approval.get('maximum_total_after_registration')==suite['maximum_total_if_approved'],
            'exact single-campaign 576-shift paired budget/execution approval required')
    return approval


def _pair_reference(suite):
    ref=suite.get('paired_seed_reference',suite.get('paired_seeds'))
    if 'paired_seeds' in suite:require(suite['paired_seeds']==ref,'paired seed reference aliases disagree')
    return ref


def preflight(path,*,approval_path=None):
    """Never constructs a learner, opens RPC, creates runs, or claims budget."""
    from .learner import default_config,validate_config
    from .features import feature_contract
    from .admission import _phase
    path=Path(path).resolve();suite=json.loads(path.read_bytes())
    require(suite.get('contract')==CAMPAIGN_CONTRACT and suite.get('kind')=='single_seed_trial'
            and suite.get('execution_profile')==PROFILE and suite.get('purpose')==PURPOSE
            and suite.get('new_shift_cap')==576 and suite.get('legacy_reserved_shifts')==11280,
            'one exact formal 576-shift paired suite required')
    require(suite.get('initialization')==INITIALIZATION and suite.get('diagnostic_scope')==DIAGNOSTIC_SCOPE,
            'fresh common base/Critic with zero residual ActorGRU initialization required')
    require(suite.get('feature_contract')==feature_contract(), 'suite summary contract differs')
    require(suite.get('history_contract')==dict(scope='whole_shift',window_reset=False,terminal_reset=True,
            source='causal_decision_summaries',critic='decision_gru_both_arms'),
            'suite must preserve full causal decision history through window boundaries')
    require(suite.get('training_shifts')==512 and suite.get('initial_evaluation_shifts')==32
            and suite.get('final_evaluation_shifts')==32 and suite.get('training_updates')==16
            and suite.get('anylogic_start_count')==1 and suite.get('arm_order')==list(CANDIDATES),
            'one-start suite totals/arm order must match the exact576 timeline')
    check_sources(suite['source_pins'])
    pair_ref=_pair_reference(suite);load_paired_seeds(pair_ref)
    output=ROOT/'runs'/suite['campaign_id'];materials=ROOT/'runs'/(suite['campaign_id']+'_materials')
    require(Path(suite['python_output'])==output and Path(suite['control_directory'])==output/'control',
            'one new global suite output/control required')
    arms=suite.get('arms');require(isinstance(arms,list) and len(arms)==2,'two separate learner arms required')
    for arm,candidate in zip(arms,CANDIDATES):
        require(arm.get('candidate_id')==candidate and arm.get('actor_seed')==2026110901
                and Path(arm['python_output'])==output/'arms'/candidate,'independent arm identity/output differs')
        config=validate_config(arm['learner_config'])
        require(config==default_config(source_kind='real_anylogic_rpc',seed=2026110901,
                    candidate_id=candidate,execution_profile=PROFILE),
                'formal arm must preserve frozen feature/network/reward/PPO architecture contract')
        require(feature_contract(config['feature_config'])==suite['feature_contract'],
                'learner feature configuration differs from suite contract')
    phases=[check_phase(load_reference(ref)) for ref in suite['phase_configs']]
    for ref in suite['phase_configs']:_phase(ref['path'])
    jobs=[j for p in phases for j in p['jobs']]
    require(jobs==suite['jobs'],'suite/phase canonical job inventory differs')
    validate_pair_jobs(jobs,pair_ref,pair_id=suite.get('pair_id'))
    require(len(phases)==20,'exact two-arm independent evaluation timeline required')
    expected=[]
    for candidate in CANDIDATES:
        expected += [(candidate,'t0','t0',0,16)]
        expected += [(candidate,'train_u%04d'%i,'train',i,32) for i in range(1,9)]
        expected += [(candidate,'final_u0008','final',8,16)]
    for phase,(candidate,arm_phase,role,local,count) in zip(phases,expected):
        prefixed=candidate+'_'+arm_phase
        require(phase['phase_id']==prefixed and phase['arm_phase_id']==arm_phase
                and phase['candidate_id']==candidate and phase['arm_update']==local
                and phase['campaign_id']==suite['campaign_id'] and len(phase['jobs'])==count
                and phase['recording_id']==suite['campaign_id']+'-'+prefixed
                and phase.get('pair_ordinals')==[j['pair_ordinal'] for j in phase['jobs']]
                and Path(phase['python_output'])==output/'arms'/candidate/arm_phase
                and Path(phase['event_directory'])==output/'control'/'legacy_unused'/prefixed,
                'paired phase timeline/count/output identity differs')
        for job in phase['jobs']:
            require(job['phase_id']==prefixed and job['recording_id']==phase['recording_id']
                    and job['source_recording_id']==phase['recording_id']
                    and Path(job['material_event_directory'])==materials/candidate/arm_phase,
                    'paired job recording/material identity differs')
    require(len({j['recording_id'] for j in jobs})==20,'each arm phase needs its own recording')
    # Evaluation and training never share the same scene/seed across roles.
    for key in ('experiment_seed','actor_sampling_seed','scene_group'):
        partitions={}
        for job in jobs:
            value=job.get(key);require(value is not None,'explicit '+key+' required')
            require(value not in partitions or partitions[value]==job['partition'],
                    'paired train/evaluation leakage: '+key)
            partitions[value]=job['partition']
    for key in ('master_plan','java_helper'):
        require(sha(suite[key]['path'])==suite[key]['sha256'],key+' byte SHA changed')
    from baseline1_v1.java_plan import COLUMNS
    with Path(suite['master_plan']['path']).open(newline='') as stream:master=list(csv.DictReader(stream))
    require(master==[{k:str(j[k]) for k in COLUMNS} for j in jobs],'576-row master differs from canonical suite')
    from python.protocol_artifacts import verify_protocol_artifacts
    from python.collect_offline import verify_environment
    verify_protocol_artifacts()
    for phase in phases:verify_environment(phase['environment_acceptance'],phase['reward'],storage_migration=True)
    acceptance=load_reference(suite['engineering_acceptance'])
    require(acceptance.get('status')=='PASS_SYNTHETIC_ONLY' and acceptance.get('source_pins')==suite['source_pins']
            and all(acceptance.get('gates',{}).get(k)=='PASS_SYNTHETIC_ONLY' for k in ('G1','G2','G3','G4')),
            'matching new paired-suite G1-G4 engineering acceptance required')
    b=budget_snapshot();cid=suite['campaign_id']
    if cid not in b['campaigns']:
        require(reference(GLOBAL_PATH)==suite['budget_before'],'shared budget changed since suite freeze')
        check_collisions(suite,b)
        require(not output.exists() and not materials.exists(),'suite output/material identity already used')
    else:require(b['campaigns'][cid]['campaign_sha256']==sha(path),'registered suite changed')
    require(suite['prior_baseline_reserved_shifts']==b['additional_reserved_shifts']-(576 if cid in b['campaigns'] else 0)
            and suite['maximum_total_if_approved']==11280+suite['prior_baseline_reserved_shifts']+576,
            'exact additional suite cap must preserve every historical allocation')
    approval=None if approval_path is None else check_approval(json.loads(Path(approval_path).read_bytes()),suite,sha(path))
    return suite,phases,approval
