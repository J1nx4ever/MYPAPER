"""Read-only admission for the frozen eight-arm, 16-update paired matrix."""
import copy
import csv
import json
from pathlib import Path
from .contract import GROUPS,CANDIDATES,ablation_contract,schedule,global_update,seed,require
from .plan_io import ROOT,PLAN,REPORT,encoded,sha,reference,load_reference,load_suite
from grade_only_ppo_v1.gate import MEMORY,DIAGNOSTIC_SCOPE,TRAINING_REWARD
REPO=ROOT.parents[1]
PROTECTED_SHA='8e4cfa31c622f5e33756e304eea3f7526547db72ae93f7d9f00808845d61ef96'
CAMPAIGN_CONTRACT='state-branch-matrix-suite-v1'
PHASE_CONTRACT='state-branch-matrix-phase-v1'
PROFILE='formal_32ep_v1'

def evaluation_contract():
    return dict(updates=[8,16],shifts_per_checkpoint_per_arm=100,
        final_checkpoint='fixed_update16_no_best_selection',sampling='stochastic_private_rng',
        pairing='same_initialization_environment_sampling_within_s01_ordinal_across_eight_arms',
        panels='u8_and_u16_disjoint_from_each_other_and_training',
        all_comparison_checkpoints_frozen_before_evaluation=True,primary='joint_four_window_compliance_rate',
        formal_training_seeds=1,stability_claim=False,initial_evaluation=False,other_evaluations=False,
        learning_speed='not_estimated_from_only_two_evaluation_checkpoints',
        no_curriculum=True,no_eta=True,reward_version='grade-only-alp-clip-linear-v1')


def source_pins():
    from representation_only128_v1.gate import source_pins as inherited
    pins=inherited()
    for package in ('state_branch_matrix_v1','physical_realization_v1'):
        pins.update({str(p.resolve()):sha(p) for p in (ROOT/package).glob('*.py') if not p.name.startswith('test_')})
    return dict(sorted(pins.items()))


def check_sources(pins):
    require(pins==source_pins(),'source inventory changed; freeze a new reviewed matrix')
    require(sha(REPO/'python/episode_budget_ppo.py')==PROTECTED_SHA,'protected PPO core changed')
    from python import observation_contract
    from .features import FEATURE_CONFIG
    path=Path(observation_contract.__file__).with_name('observation_schema.json').resolve()
    require(path==ROOT/'python/observation_schema.json' and sha(path)==FEATURE_CONFIG['schema_sha256'],
            'launch from the fixed experiment directory/schema')


def check_phase(cfg):
    from grade_only_ppo_v1 import gate as old
    from .features import feature_contract,physical_contract
    require(cfg['contract']==PHASE_CONTRACT and cfg['candidate_id'] in CANDIDATES and
            cfg['state_branch_contract']==ablation_contract(cfg['candidate_id']),'phase branch contract differs')
    compatible=copy.deepcopy(cfg);compatible['contract']=old.PHASE_CONTRACT;old.check_phase(compatible)
    require(cfg['feature_contract']==feature_contract() and cfg['physical_feature_contract']==physical_contract()
            and cfg['execution_profile']==PROFILE,'summary/physical/profile contract differs')
    require(cfg['diagnostic_recording']==dict(transition_jsonl=False,rpc_jsonl=False,rollout_snapshots=False),
            'high-frequency diagnostics must remain off')
    return cfg


def check_jobs(suite):
    require(suite['stage']=='formal','matrix uses one reviewed formal schedule')
    arms=['s01_'+g for g in GROUPS]
    require(suite['arm_order']==arms and [a['candidate_id'] for a in suite['arms']]==arms,'eight-arm inventory differs')
    require(suite['new_shift_cap']==len(suite['jobs'])==5696 and suite['training_shifts']==4096
            and suite['training_updates']==128 and suite['training_seed_count']==1,'matrix physical/update budget differs')
    init=seed(suite['campaign_id'],'initialization',1)
    for a in suite['arms']:
        require(a['actor_seed']==init and a['representation']==a['candidate_id'].split('_')[1]
                and a['replicate']=='s01' and a['new_shift_cap']==712,'paired initialization/arm metadata differs')
    seen_runs=set();seen_material=set();roles={};pairs={};position=0;pair_base=0
    for phase,role,u,count in schedule(False):
        for arm in suite['arms']:
            candidate=arm['candidate_id'];phase_id=candidate+'_'+phase
            for offset in range(count):
                job=suite['jobs'][position];pair=pair_base+offset
                require(job['ordinal']==position and job['candidate_id']==candidate and job['pair_ordinal']==pair
                        and job['phase_id']==phase_id and job['arm_phase_id']==phase and job['arm_update']==u
                        and job['role']==role and job['actor_seed']==init
                        and job['target_training_update']==global_update(candidate,u)
                        and job['checkpoint_update']==(u-1 if role=='train' else u)
                        and job['partition']==('train' if role=='train' else 'validation'),'phase-major timeline differs')
                require(job['run_index']==suite['run_index_base']+position,'run identity differs')
                ns=suite['campaign_id']+'-s01'
                require(job['experiment_seed']==seed(ns,'environment',pair) and
                        job['actor_sampling_seed']==seed(ns,'sampling',pair),'paired new RNG identity differs')
                require(job['run_index'] not in seen_runs and job['material_journal'] not in seen_material,'physical identity reused')
                seen_runs.add(job['run_index']);seen_material.add(job['material_journal'])
                pairs.setdefault(pair,[]).append(candidate)
                for field in ('experiment_seed','actor_sampling_seed'):
                    require(roles.setdefault((field,job[field]),(pair,role))==(pair,role),'RNG crosses train/evaluation roles')
                output=ROOT/'runs'/suite['campaign_id'];events=output/'control/legacy_unused'/phase_id
                material=ROOT/'runs'/(suite['campaign_id']+'_materials')/phase_id
                episode=f"dispatch-production-wait-shortfall-v2-run-{job['run_index']}-seed-{job['experiment_seed']}"
                require(job['group']=='BASELINE' and job['recording_id']==job['source_recording_id']==suite['campaign_id']+'-'+phase_id
                        and Path(job['python_output'])==output/'arms'/candidate/phase
                        and Path(job['event_directory'])==events and Path(job['event_bundle'])==events/episode
                        and Path(job['material_event_directory'])==material
                        and Path(job['material_event_bundle'])==material/episode
                        and Path(job['material_journal'])==material/episode/'material_events.bin','output/material identity differs')
                position+=1
        pair_base+=count
    require(position==5696 and pair_base==712 and all(len(v)==8 and len(set(v))==8 for v in pairs.values()),
            'each paired scenario must have exactly eight independent physical runs')
    require(len({j['experiment_seed'] for j in suite['jobs']})==712 and
            len({j['actor_sampling_seed'] for j in suite['jobs']})==712,'unexpected seed reuse')
    sets=[{j[k] for j in suite['jobs']} for k in ('actor_seed','experiment_seed','actor_sampling_seed')]
    require(all(sets[i].isdisjoint(sets[j]) for i in range(3) for j in range(i)),'cross-type seed collision')
    return suite


def require_prior_complete(suite):
    from .budget import budget_snapshot
    b=budget_snapshot()
    require(len(suite['predecessor_suites'])==2,'both completed engineering suites required')
    for prior_ref,cap,updates in zip(suite['predecessor_suites'],(180,36),(5,1)):
        prior=load_reference(prior_ref);entry=b['campaigns'].get(prior['campaign_id'])
        require(entry is not None and entry['campaign_sha256']==prior_ref['sha256'] and entry['cap']==cap
                and len(entry['commits'])==updates,'preceding engineering commits incomplete')
        slots=[r for r in b['jobs'].values() if r['campaign_id']==prior['campaign_id']]
        require(len(slots)==cap and all(r['status']=='COMPLETE' for r in slots),'preceding engineering UNKNOWN/incomplete')
        complete=Path(prior['python_output'])/'suite_complete.json'
        require(complete.is_file(),'preceding engineering completion absent')
        proof=json.loads(complete.read_bytes())
        require(proof.get('campaign')==prior_ref and proof.get('physical_shifts')==cap
                and proof.get('global_updates')==updates,'preceding engineering identity differs')
        require(not (Path(prior['control_directory'])/'STOP.properties').exists(),'preserved predecessor STOP')
    return True


def check_approval(approval,suite,suite_sha256):
    require(approval.get('decision')=='approved' and approval.get('approved_by')=='user'
            and approval.get('execution_authorized') is True and bool(approval.get('source_user_instruction'))
            and approval.get('ack_grade_only_diagnostic') is True and approval.get('ack_state_branch_matrix') is True
            and approval.get('campaign_id')==suite['campaign_id'] and approval.get('campaign_sha256')==suite_sha256
            and approval.get('max_new_shifts')==suite['new_shift_cap']==5696
            and approval.get('maximum_total_after_registration')==suite['maximum_total_if_approved']
            and approval.get('active_h2_reserved_shifts')==approval.get('legacy_reserved_shifts')==0
            and approval.get('accounting')=='merged_all_ledgers_no_h2_future_reservation',
            'exact matrix budget/execution authorization required')
    return approval


def preflight(path,*,approval_path=None):
    from .learner import default_config,validate_config
    from .budget import budget_snapshot,check_collisions,preceding_reference
    from .admission import _phase
    suite,phases=load_suite(path)
    require(suite.get('contract')==CAMPAIGN_CONTRACT and suite['suite_path']==str(Path(path).resolve())
            and suite['stage']=='formal' and suite['legacy_reserved_shifts']==suite['active_h2_reserved_shifts']==0,
            'scope/path/budget policy differs')
    check_sources(suite['source_pins']);check_jobs(suite)
    require(suite['evaluation_contract']==evaluation_contract(),'fixed evaluation protocol differs')
    require(reference(suite['one_click_entry']['path'])==suite['one_click_entry'],'fixed bridge changed')
    for prior in suite['predecessor_suites']:load_reference(prior)
    for arm in suite['arms']:
        require(validate_config(arm['learner_config'])==default_config(seed=arm['actor_seed'],candidate_id=arm['candidate_id']),
                'network/Reward/PPO/init rules differ')
    for cfg,ref in zip(phases,suite['phase_configs']):check_phase(cfg);_phase(ref['path'])
    from baseline1_v1.java_plan import COLUMNS
    for key in ('master_plan','java_helper','common_config'):
        require(reference(suite[key]['path'])==suite[key],'frozen '+key+' changed')
    with Path(suite['master_plan']['path']).open(newline='') as stream:
        require(list(csv.DictReader(stream))==[{k:str(j[k]) for k in COLUMNS} for j in suite['jobs']],'master inventory differs')
    from python.protocol_artifacts import verify_protocol_artifacts
    from python.collect_offline import verify_environment
    verify_protocol_artifacts();verify_environment(phases[0]['environment_acceptance'],phases[0]['reward'],storage_migration=True)
    acceptance=load_reference(suite['engineering_acceptance'])
    require(acceptance['status']=='PASS_OFFLINE_ONLY' and acceptance['source_pins']==suite['source_pins'],
            'current offline acceptance required')
    b=budget_snapshot();cid=suite['campaign_id']
    if cid not in b['campaigns']:
        check_collisions(suite,b)
        require(not Path(suite['python_output']).exists() and not (ROOT/'runs'/(cid+'_materials')).exists(),
                'new physical output identity occupied')
        require(preceding_reference()==suite['budget_state_before'],'preceding budget evidence changed')
        require(b['maximum_total_shifts']+5696==suite['maximum_total_if_approved'],'merged budget changed')
    else:require(b['campaigns'][cid]['campaign_sha256']==sha(path),'registered suite differs')
    approval=None
    if approval_path is not None:
        approval=check_approval(json.loads(Path(approval_path).read_bytes()),suite,sha(path));require_prior_complete(suite)
    return suite,phases,approval
