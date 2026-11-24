"""Read-only exact design admission; execution still needs explicit approval."""
import copy
import csv
import json
from pathlib import Path
from .contract import GROUPS, CANDIDATES, ablation_contract, schedule, global_update, seed, require
from .plan_io import ROOT, PLAN, REPORT, encoded, sha, reference, load_reference, load_suite, load_phase
from grade_only_ppo_v1.gate import MEMORY, DIAGNOSTIC_SCOPE, TRAINING_REWARD

REPO = ROOT.parents[1]
PROTECTED_SHA = '8e4cfa31c622f5e33756e304eea3f7526547db72ae93f7d9f00808845d61ef96'
CAMPAIGN_CONTRACT = 'representation-only128-suite-v1'
PHASE_CONTRACT = 'representation-only128-phase-v1'
PROFILE = 'formal_32ep_v1'

def source_pins():
    from representation_ablation_v1.gate import source_pins as inherited
    pins = inherited()
    pins.update({str(p.resolve()): sha(p) for p in (ROOT/'representation_only128_v1').glob('*.py')
                 if not p.name.startswith('test_')})
    # Pin the newer accounting modules that merge ALL preceding allocations.
    for package in ('grade_sampling_ab_v1','grade_revalidation_v1','grade_multiobjective_ppo_v1',
                    'physical_curriculum_v1','physical_realization_ppo_v1','u8_wait_efficiency_v1'):
        for name in ('budget.py','budget_policy.py','gate.py'):
            p=ROOT/package/name
            if p.exists(): pins[str(p.resolve())]=sha(p)
    return dict(sorted(pins.items()))

def check_sources(pins):
    require(pins == source_pins(), 'source inventory changed; freeze a new reviewed package')
    require(sha(REPO/'python/episode_budget_ppo.py') == PROTECTED_SHA, 'protected PPO core changed')
    from python import observation_contract
    from .features import FEATURE_CONFIG
    path=Path(observation_contract.__file__).with_name('observation_schema.json').resolve()
    require(path == ROOT/'python/observation_schema.json' and sha(path)==FEATURE_CONFIG['schema_sha256'],
            'wrong python package: launch from the experiment directory with the fixed controller')

def check_phase(cfg):
    from grade_only_ppo_v1 import gate as old
    require(cfg['contract']==PHASE_CONTRACT and cfg['candidate_id'] in CANDIDATES and
            cfg['ablation_contract']==ablation_contract(cfg['candidate_id']), 'phase ablation contract differs')
    compatible=copy.deepcopy(cfg); compatible['contract']=old.PHASE_CONTRACT
    old.check_phase(compatible)
    from .features import feature_contract
    require(cfg['feature_contract']==feature_contract() and cfg['execution_profile']==PROFILE,
            'feature/profile changed')
    require(cfg['diagnostic_recording']==dict(transition_jsonl=False,rpc_jsonl=False,rollout_snapshots=False),
            'high frequency diagnostic recording must remain off')
    return cfg

def check_jobs(suite):
    require(suite['stage']=='engineering', 'only the supplementary engineering36 schedule is prepared')
    engineering=True
    names=['e01']
    arms=[f'{name}_{g}' for name in names for g in GROUPS]
    require(suite['arm_order']==arms and [a['candidate_id'] for a in suite['arms']]==arms,
            'single R-128 arm inventory differs')
    cap=36
    require(suite['new_shift_cap']==len(suite['jobs'])==cap, 'physical budget differs')
    require(suite['training_shifts']==32 and
            suite['training_updates']==1 and suite['training_seed_count']==1,
            'training count/seed contract differs')
    seen={k:set() for k in ('run_index','material_journal')}
    pairs={}; roles={}; position=0
    for arm in suite['arms']:
        candidate=arm['candidate_id']; rep=candidate.split('_')[0]
        require(arm['actor_seed']==seed(suite['campaign_id'],'initialization',int(rep[1:])), 'initialization stream differs')
        require(arm['representation']==candidate.split('_',1)[1] and arm['replicate']==rep
                and arm['new_shift_cap']==(36 if engineering else 432), 'arm metadata/count differs')
        pair=0
        for phase,role,u,count in schedule(engineering):
            for _ in range(count):
                job=suite['jobs'][position]
                require(job['ordinal']==position and job['candidate_id']==candidate and job['pair_ordinal']==pair
                        and job['phase_id']==candidate+'_'+phase and job['arm_phase_id']==phase
                        and job['arm_update']==u and job['role']==role and job['actor_seed']==arm['actor_seed']
                        and job['target_training_update']==global_update(candidate,u)
                        and job['checkpoint_update']==(u-1 if role=='train' else u)
                        and job['partition']==('train' if role=='train' else 'validation'), 'canonical timeline differs')
                require(job['run_index']==suite['run_index_base']+position, 'run identity differs')
                ns=suite['campaign_id']+'-'+rep
                require(job['experiment_seed']==seed(ns,'environment',pair) and
                        job['actor_sampling_seed']==seed(ns,'sampling',pair), 'new paired RNG identity differs')
                for key in seen:
                    require(job[key] not in seen[key], 'physical identity reused'); seen[key].add(job[key])
                key=(rep,pair)
                pairs.setdefault(key,[]).append(candidate)
                for field in ('experiment_seed','actor_sampling_seed'):
                    value=job[field]
                    require(roles.setdefault((field,value),(rep,pair,role))==(rep,pair,role), 'seed leaks across roles or pairs')
                phase_id=job['phase_id']; output=ROOT/'runs'/suite['campaign_id']
                require(job['recording_id']==job['source_recording_id']==suite['campaign_id']+'-'+phase_id and
                        Path(job['python_output'])==output/'arms'/candidate/phase and
                        Path(job['material_event_directory'])==ROOT/'runs'/(suite['campaign_id']+'_materials')/phase_id,
                        'output/recording/material identity differs')
                position+=1; pair+=1
    require(all(len(v)==1 for v in pairs.values()), 'single-arm ordinal reused')
    require(len({j['experiment_seed'] for j in suite['jobs']})==cap and
            len({j['actor_sampling_seed'] for j in suite['jobs']})==cap, 'independent shift seed reuse')
    sets=[{j[k] for j in suite['jobs']} for k in ('actor_seed','experiment_seed','actor_sampling_seed')]
    require(all(sets[i].isdisjoint(sets[j]) for i in range(3) for j in range(i)), 'cross-type seed collision')
    return suite

def require_prior_complete(suite):
    """Execution barrier only: preparation never waits for or changes the live180."""
    from .budget import budget_snapshot
    prior=load_reference(suite['predecessor_suite'])
    b=budget_snapshot();entry=b['campaigns'].get(prior['campaign_id'])
    require(entry is not None and entry['campaign_sha256']==suite['predecessor_suite']['sha256']
            and len(entry['commits'])==5 and entry['cap']==180,
            'R-128 start waits for all five prior180 engineering commits')
    slots=[row for row in b['jobs'].values() if row['campaign_id']==prior['campaign_id']]
    require(len(slots)==180 and all(row['status']=='COMPLETE' for row in slots),
            'prior180 slots incomplete or UNKNOWN; finish the existing experiment first')
    complete=Path(prior['python_output'])/'suite_complete.json'
    require(complete.is_file(), 'R-128 start waits for prior180 suite_complete.json')
    proof=json.loads(complete.read_bytes())
    require(proof.get('campaign')==suite['predecessor_suite'] and proof.get('physical_shifts')==180
            and proof.get('global_updates')==5, 'prior180 completion identity differs')
    require(not (Path(prior['control_directory'])/'STOP.properties').exists(),
            'preserved prior180 STOP requires review')
    return proof

def check_approval(approval,suite,suite_sha256):
    require(approval.get('decision')=='approved' and approval.get('approved_by')=='user'
            and approval.get('execution_authorized') is True and bool(approval.get('source_user_instruction'))
            and approval.get('ack_grade_only_diagnostic') is True
            and approval.get('ack_representation_only128') is True
            and approval.get('campaign_id')==suite['campaign_id'] and approval.get('campaign_sha256')==suite_sha256
            and approval.get('max_new_shifts')==suite['new_shift_cap']
            and approval.get('maximum_total_after_registration')==suite['maximum_total_if_approved']
            and approval.get('active_h2_reserved_shifts')==0 and approval.get('legacy_reserved_shifts')==0
            and approval.get('accounting')=='merged_all_ledgers_no_h2_future_reservation',
            'exact new-suite physical budget and execution authorization required')
    return approval

def preflight(path, *, approval_path=None):
    from .learner import default_config,validate_config
    from .budget import budget_snapshot,check_collisions,preceding_reference
    from .admission import _phase
    suite,phases=load_suite(path)
    require(suite.get('contract')==CAMPAIGN_CONTRACT and suite['suite_path']==str(Path(path).resolve())
            and suite['stage']=='engineering' and suite['active_h2_reserved_shifts']==0
            and suite['legacy_reserved_shifts']==0, 'scope/path/budget policy differs')
    check_sources(suite['source_pins']); check_jobs(suite)
    prior=load_reference(suite['predecessor_suite'])
    require(prior['new_shift_cap']==180 and prior['training_updates']==5 and
            prior['contract']=='representation-ablation-suite-v1', 'prior180 contract differs')
    require(reference(suite['one_click_entry']['path'])==suite['one_click_entry'], 'fixed one-click bridge changed')
    for arm in suite['arms']:
        require(validate_config(arm['learner_config'])==default_config(seed=arm['actor_seed'],candidate_id=arm['candidate_id']),
                'network/Reward/PPO/init budget must be identical across arms')
    for phase,ref in zip(phases,suite['phase_configs']):
        check_phase(phase); _phase(ref['path'])
    from baseline1_v1.java_plan import COLUMNS
    for key in ('master_plan','java_helper','common_config'):
        require(reference(suite[key]['path'])==suite[key], 'frozen '+key+' changed')
    with Path(suite['master_plan']['path']).open(newline='') as stream:
        require(list(csv.DictReader(stream))==[{k:str(j[k]) for k in COLUMNS} for j in suite['jobs']], 'master inventory differs')
    from python.protocol_artifacts import verify_protocol_artifacts
    from python.collect_offline import verify_environment
    verify_protocol_artifacts()
    verify_environment(phases[0]['environment_acceptance'],phases[0]['reward'],storage_migration=True)
    acceptance=load_reference(suite['engineering_acceptance'])
    require(acceptance['status']=='PASS_OFFLINE_ONLY' and acceptance['source_pins']==suite['source_pins'],
            'current offline engineering acceptance required')
    b=budget_snapshot();cid=suite['campaign_id']
    if cid not in b['campaigns']:
        check_collisions(suite,b)
        require(not Path(suite['python_output']).exists() and
                not (ROOT/'runs'/(cid+'_materials')).exists(), 'new output identity already exists')
        require(preceding_reference()==suite['budget_state_before'], 'preceding capacity evidence changed')
    else:
        require(b['campaigns'][cid]['campaign_sha256']==sha(path), 'registered suite differs')
    approval=None
    if approval_path is not None:
        approval=check_approval(json.loads(Path(approval_path).read_bytes()),suite,sha(path))
        if cid not in b['campaigns']:
            require(b['maximum_total_shifts']+suite['new_shift_cap']==suite['maximum_total_if_approved'],
                    'expected preceding engineering allocation absent or total changed')
        require_prior_complete(suite)
    return suite,phases,approval
