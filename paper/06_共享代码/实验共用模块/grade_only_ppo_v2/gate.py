"""Exact four-shift REAL smoke contract; formal budgets stay unsupported here."""
import copy
import csv
import json
from pathlib import Path

from grade_only_ppo_v1 import gate as frozen
from baseline1_v1.budget import GLOBAL_PATH

ROOT, REPO, PROTECTED_SHA = frozen.ROOT, frozen.REPO, frozen.PROTECTED_SHA
encoded, reference, require, sha = frozen.encoded, frozen.reference, frozen.require, frozen.sha
load_reference = frozen.load_reference
budget_snapshot, check_collisions = frozen.budget_snapshot, frozen.check_collisions
DIAGNOSTIC_SCOPE, MEMORY, TRAINING_REWARD = frozen.DIAGNOSTIC_SCOPE, frozen.MEMORY, frozen.TRAINING_REWARD
CAMPAIGN_CONTRACT = 'grade-only-adv-smoke-campaign-v2'
PHASE_CONTRACT = 'grade-only-adv-smoke-rpc-phase-v2'
PROFILE = 'real_smoke_2ep_v1'
PURPOSE = 'engineering_smoke_not_a_b_effectiveness_comparison'
PLAN = ROOT/'change_plans/grade_only_mc_advantage_ab_001/real_smoke_001'
CANDIDATES = ('a0_gae095','a1_mc100')


def source_pins():
    pins=frozen.source_pins()
    pins.update({str(p.resolve()):sha(p) for p in (ROOT/'grade_only_ppo_v2').glob('*.py')
                 if not p.name.startswith('test_')})
    return dict(sorted(pins.items()))


def check_sources(pins):
    require(pins==source_pins(),'smoke implementation/source inventory changed; freeze a new package')
    require(sha(REPO/'python/episode_budget_ppo.py')==PROTECTED_SHA,'protected PPO core changed')


def check_phase(cfg):
    require(cfg.get('contract')==PHASE_CONTRACT,'independent v2 smoke phase contract required')
    require(cfg.get('execution_profile')==PROFILE and cfg.get('purpose')==PURPOSE
            and cfg.get('candidate_id') in CANDIDATES,'exact real smoke scope/candidate required')
    compatible=copy.deepcopy(cfg);compatible['contract']=frozen.PHASE_CONTRACT
    frozen.check_phase(compatible)
    return cfg


def check_approval(approval,campaign,campaign_sha256):
    require(approval.get('decision')=='approved' and approval.get('approved_by')=='user'
            and approval.get('execution_authorized') is True and bool(approval.get('source_user_instruction'))
            and approval.get('ack_grade_only_diagnostic') is True and approval.get('ack_real_smoke_only') is True
            and approval.get('execution_profile')==campaign.get('execution_profile')==PROFILE
            and approval.get('candidate_id')==campaign.get('candidate_id') in CANDIDATES
            and approval.get('campaign_id')==campaign['campaign_id']
            and approval.get('campaign_sha256')==campaign_sha256
            and approval.get('max_new_shifts')==campaign.get('new_shift_cap')==4
            and approval.get('legacy_reserved_shifts')==11280
            and approval.get('accounting')=='additional_to_h2_reserved'
            and approval.get('maximum_total_after_registration')==campaign['maximum_total_if_approved'],
            'exact four-shift real-smoke budget/execution approval required')
    return approval


def preflight(campaign_path,*,approval_path=None):
    """Only reads: no constructor, listener, claim, output mkdir or training."""
    from .learner import validate_config
    from grade_event_memory_v1.admission import _phase
    path=Path(campaign_path).resolve();c=json.loads(path.read_bytes())
    require(c.get('contract')==CAMPAIGN_CONTRACT and c.get('kind')=='single_seed_trial'
            and c.get('execution_profile')==PROFILE and c.get('purpose')==PURPOSE
            and c.get('new_shift_cap')==4 and c.get('legacy_reserved_shifts')==11280,
            'exact four-job real-smoke campaign required; formal execution unsupported')
    require(c.get('diagnostic_scope')==DIAGNOSTIC_SCOPE and c.get('candidate_id') in CANDIDATES,
            'grade-only smoke scope/candidate differs')
    require(c.get('initialization')=='fresh_random_actor_and_event_critic_no_pretraining',
            'fresh independent initialization required')
    check_sources(c['source_pins'])
    learner=validate_config(c['learner_config'])
    require(learner['source_kind']=='real_anylogic_rpc' and learner['seed']==c['actor_seed']
            and learner['execution_profile']==PROFILE and learner['candidate_id']==c['candidate_id'],
            'real smoke learner identity differs')
    output=ROOT/'runs'/c['campaign_id']
    require(Path(c['python_output'])==output and Path(c['control_directory'])==output/'control',
            'new independent output/control identity required')
    phases=[check_phase(load_reference(ref)) for ref in c['phase_configs']]
    for ref in c['phase_configs']: _phase(ref['path'])
    jobs=[j for p in phases for j in p['jobs']]
    require(jobs==c['jobs'] and len(jobs)==4 and [j['ordinal'] for j in jobs]==list(range(4)),
            'four immutable smoke jobs required')
    for key in ('run_index','experiment_seed','actor_sampling_seed'):
        require(len({j[key] for j in jobs})==4,'duplicate smoke identity: '+key)
    expected=[('t0','t0',0,1),('train_u0001','train',1,2),('final_u0001','final',1,1)]
    require(len(phases)==3,'smoke requires initial eval, one train batch and final eval')
    partitions={}
    for phase,(phase_id,role,update,count) in zip(phases,expected):
        require(phase['phase_id']==phase_id and phase['campaign_id']==c['campaign_id']
                and phase['candidate_id']==c['candidate_id'] and len(phase['jobs'])==count
                and phase['recording_id']==c['campaign_id']+'-'+phase_id
                and Path(phase['python_output'])==output/phase_id
                and Path(phase['event_directory'])==output/'control'/'legacy_unused'/phase_id,
                'exact three-phase smoke schedule/identity differs')
        for job in phase['jobs']:
            require(job['role']==role and job['target_training_update']==update
                    and job['checkpoint_update']==(0 if role=='train' else update)
                    and job['actor_seed']==c['actor_seed'] and job['phase_id']==phase_id
                    and job['source_recording_id']==phase['recording_id']
                    and Path(job['material_event_directory'])==ROOT/'runs'/(c['campaign_id']+'_materials')/phase_id,
                    'smoke job update/material path differs')
            group=job.get('scene_group')
            require(isinstance(group,str) and group and
                    (group not in partitions or partitions[group]==job['partition']),
                    'train/independent-evaluation scene leakage')
            partitions[group]=job['partition']
    for key in ('master_plan','java_helper'):
        require(sha(c[key]['path'])==c[key]['sha256'],key+' byte SHA changed')
    from baseline1_v1.java_plan import COLUMNS
    with Path(c['master_plan']['path']).open(newline='') as stream: master=list(csv.DictReader(stream))
    require(master==[{k:str(j[k]) for k in COLUMNS} for j in jobs],'master CSV differs from smoke jobs')
    from python.protocol_artifacts import verify_protocol_artifacts
    from python.collect_offline import verify_environment
    verify_protocol_artifacts()
    for phase in phases: verify_environment(phase['environment_acceptance'],phase['reward'],storage_migration=True)
    acceptance=load_reference(c['engineering_acceptance'])
    require(acceptance.get('status')=='PASS_SYNTHETIC_ONLY' and acceptance.get('source_pins')==c['source_pins']
            and all(acceptance.get('gates',{}).get(k)=='PASS_SYNTHETIC_ONLY' for k in ('G1','G2','G3','G4')),
            'current independent smoke G1-G4 engineering acceptance required')
    budget=budget_snapshot();cid=c['campaign_id']
    if cid not in budget['campaigns']:
        require(reference(GLOBAL_PATH)==c['budget_before'],'budget changed since preparation; reconcile first')
        check_collisions(c,budget)
        require(not output.exists() and all(not Path(j['material_event_directory']).exists() for j in jobs),
                'smoke run/material identity already used')
    else:
        require(budget['campaigns'][cid]['campaign_sha256']==sha(path),'registered smoke campaign changed')
    require(c['prior_baseline_reserved_shifts']==budget['additional_reserved_shifts']-(4 if cid in budget['campaigns'] else 0)
            and c['maximum_total_if_approved']==11280+c['prior_baseline_reserved_shifts']+4,
            'smoke must preserve all prior allocations and add exactly four shifts')
    approval=None if approval_path is None else check_approval(json.loads(Path(approval_path).read_bytes()),c,sha(path))
    return c,phases,approval
