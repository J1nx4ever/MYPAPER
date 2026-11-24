"""Read-only strict admission for the explicitly authorized 3 x 5 smoke."""
import copy
import csv
import json
from pathlib import Path
from .contract import CANDIDATES, GROUPS, ablation_contract, global_update, seed, require
from .plan_io import ROOT, PLAN, REPORT, encoded, sha, reference, load_reference, load_suite
from .targets import composite_contract, REWARD_VERSION

REPO = ROOT.parents[1]
PROTECTED_SHA = '8e4cfa31c622f5e33756e304eea3f7526547db72ae93f7d9f00808845d61ef96'
CAMPAIGN_CONTRACT = 'relation-gat-ego-smoke-suite-v1'
PHASE_CONTRACT = 'relation-gat-ego-rpc-phase-v1'
PROFILE = 'smoke_5ep_v1'
TRAINING_REWARD = 'negative_combined_grade_production_wait_cost_v1'
DIAGNOSTIC_SCOPE = dict(scope='joint_objective_engineering_smoke',production='training_and_evaluation',
                       waiting='training_and_evaluation',production_guarantee=False)


def evaluation_contract():
    return dict(independent_physical_evaluation_shifts=0, fixed_observation_replay=True,
        checkpoint_restore_required=True,performance_conclusion=False,
        sampling='stochastic_private_rng',training_seed_count=1,
        fresh_formal_restart_required=True,additional_formal_campaign_auto_start=False)


def source_pins():
    from state_branch_matrix_v1.gate import source_pins as inherited
    pins = inherited()
    for package in ('relation_gat_ego_v1','physical_curriculum_v1'):
        pins.update({str(p.resolve()):sha(p) for p in (ROOT/package).glob('*.py') if not p.name.startswith('test_')})
    for folder in ('relation_gat_ego_003','s1519_ego_revision_002','schema_audit_001'):
        for p in (ROOT/'change_plans/token_attention_001'/folder).iterdir():
            if p.is_file() and p.suffix in ('.json','.py','.md') and not p.name.startswith('test_'):
                pins[str(p.resolve())] = sha(p)
    return dict(sorted(pins.items()))


def check_sources(pins):
    require(pins==source_pins(),'source inventory changed; freeze a new smoke package')
    require(sha(REPO/'python/episode_budget_ppo.py')==PROTECTED_SHA,'protected PPO core changed')
    from python import observation_contract
    require(Path(observation_contract.__file__).resolve().parent==ROOT/'python',
            'fixed experiment Python/schema import path required')
    require(sha(ROOT/'python/observation_schema.json')=='c242ce76856d1cd929af738da3942bab312749712603626d6c2747ac47d12637',
            'physical observation schema changed')


def check_phase(cfg):
    from baseline1_v1 import gate as transport
    from grade_only_ppo_v1.gate import MEMORY
    from .c_features import feature_contract
    require(cfg['contract']==PHASE_CONTRACT and cfg['candidate_id'] in CANDIDATES and
            cfg['graph_contract']==ablation_contract(cfg['candidate_id']),'phase graph contract differs')
    raw = copy.deepcopy(cfg); raw['contract'] = transport.PHASE_CONTRACT
    transport.check_phase(raw)
    require(cfg['execution_profile']==PROFILE and cfg['feature_contract']==feature_contract() and
            cfg['composite_contract']==composite_contract(.1),'phase feature/objective/profile differs')
    require(cfg['material_memory']==dict(MEMORY,reward_version=REWARD_VERSION), 'joint material-memory identity differs')
    require(cfg['training_reward']==TRAINING_REWARD and cfg['diagnostic_scope']==DIAGNOSTIC_SCOPE,
            'joint target meaning differs')
    require(cfg['diagnostic_recording']==dict(transition_jsonl=False,rpc_jsonl=False,rollout_snapshots=False) and
            cfg['storage']==dict(format='hdf5-v1',chunk_rows=64),'compact storage/diagnostics required')
    require(cfg['rpc_configuration_authoritative'] is True and cfg['require_local_alp_audit'] is False,
            'runtime RPC configuration authority required')
    for job in cfg['jobs']:
        for key,value in dict(shift_seconds=28800.,window_seconds=7200.,target_ore_tons_shift=16000.,
            target_waste_tons_shift=4000.,target_grade=.27,w_production=4.,waste_shortfall_weight=4.,
            grade_lower=.26,grade_upper=.28,wait_kappa=.00004,material_event_memory_enabled=1.,
            material_event_contract_version=1.).items():
            require(job['expected_rpc_config'].get('config.'+key)==value,'explicit RPC contract differs: '+key)
        require(job['partition']=='train' and job['role']=='train', 'smoke contains only the authorized training shifts')
    return cfg


def check_jobs(suite):
    require(suite['stage']=='engineering' and suite['execution_profile']==PROFILE,
            'only the 5-shift engineering profile is released')
    require(suite['arm_order']==list(CANDIDATES) and [a['candidate_id'] for a in suite['arms']]==list(CANDIDATES),
            'three-arm inventory differs')
    require(suite['new_shift_cap']==len(suite['jobs'])==suite['training_shifts']==15 and
            suite['training_updates']==3 and suite['training_seed_count']==1 and
            suite['episodes_per_batch']==5 and suite['cpu_threads']==2,'smoke physical/update/resource budget differs')
    init=seed(suite['campaign_id'],'initialization',1)
    output=ROOT/'runs'/suite['campaign_id']; material_base=output.with_name(suite['campaign_id']+'_materials')
    for a in suite['arms']:
        require(a['actor_seed']==init and a['replicate']=='s01' and a['new_shift_cap']==5 and
                a['representation']==a['candidate_id'][4:] and
                Path(a['python_output'])==output/'arms'/a['candidate_id'],'paired initialization/output metadata differs')
    require(Path(suite['python_output'])==output and Path(suite['control_directory'])==output/'control',
            'campaign output identity differs')
    pairs={}; seen=set(); ns=suite['campaign_id']+'-s01'
    for ordinal,j in enumerate(suite['jobs']):
        candidate=CANDIDATES[ordinal//5]; pair=ordinal%5; phase=candidate+'_train_u0001'
        recording=suite['campaign_id']+'-'+phase
        events=output/'control/legacy_unused'/phase; materials=material_base/phase
        episode=f"dispatch-production-wait-shortfall-v2-run-{j['run_index']}-seed-{j['experiment_seed']}"
        require(j['ordinal']==ordinal and j['run_index']==suite['run_index_base']+ordinal and
            j['candidate_id']==candidate and j['phase_id']==phase and j['arm_phase_id']=='train_u0001' and
            j['pair_ordinal']==pair and j['role']=='train' and j['partition']=='train' and j['arm_update']==1 and
            j['checkpoint_update']==0 and j['target_training_update']==global_update(candidate,1),
            'smoke timeline differs')
        require(j['actor_seed']==init and j['experiment_seed']==seed(ns,'environment',pair) and
                j['actor_sampling_seed']==seed(ns,'sampling',pair),'new paired seed differs')
        require(j['recording_id']==j['source_recording_id']==recording and
            Path(j['python_output'])==output/'arms'/candidate/'train_u0001' and
            Path(j['event_directory'])==events and Path(j['event_bundle'])==events/episode and
            Path(j['material_event_directory'])==materials and Path(j['material_event_bundle'])==materials/episode and
            Path(j['material_journal'])==materials/episode/'material_events.bin','physical output/recording identity differs')
        require(j['run_index'] not in seen,'duplicate physical run'); seen.add(j['run_index'])
        pairs.setdefault(pair,[]).append(candidate)
    require(all(v==list(CANDIDATES) for v in pairs.values()) and len(pairs)==5,'paired five-scenario inventory differs')
    streams=[{j[k] for j in suite['jobs']} for k in ('actor_seed','experiment_seed','actor_sampling_seed')]
    require([len(s) for s in streams]==[1,5,5] and all(streams[i].isdisjoint(streams[j])
            for i in range(3) for j in range(i)),'cross-type seed collision')
    return suite


def require_prior_complete(suite):
    from .acceptance import verify_acceptance
    proof=load_reference(suite['engineering_acceptance'])
    verify_acceptance(proof)
    require(proof['status']=='PASS_OFFLINE_ONLY' and proof['source_pins']==suite['source_pins'],
            'current offline implementation acceptance required before physical collection')
    return True


def check_approval(approval,suite,suite_sha256):
    require(approval.get('decision')=='approved' and approval.get('approved_by')=='user' and
        approval.get('execution_authorized') is True and bool(approval.get('source_user_instruction')) and
        approval.get('ack_relation_graph_smoke') is True and approval.get('campaign_id')==suite['campaign_id'] and
        approval.get('execution_profile')==suite['execution_profile'] and approval.get('pair_id')==suite['pair_id'] and
        approval.get('campaign_sha256')==suite_sha256 and approval.get('max_new_shifts')==suite['new_shift_cap']==15 and
        approval.get('maximum_total_after_registration')==suite['maximum_total_if_approved'] and
        approval.get('active_h2_reserved_shifts')==approval.get('legacy_reserved_shifts')==0 and
        approval.get('accounting')=='merged_all_ledgers_no_h2_future_reservation',
        'exact authorized 15-shift smoke identity required')
    return approval


def preflight(path,*,approval_path=None):
    from .learner import default_config,validate_config
    from .budget import budget_snapshot,check_collisions,preceding_reference
    from .admission import _phase
    suite,phases=load_suite(path)
    require(suite['contract']==CAMPAIGN_CONTRACT and suite['suite_path']==str(Path(path).resolve()) and
            suite['legacy_reserved_shifts']==suite['active_h2_reserved_shifts']==0,'suite path/scope differs')
    check_sources(suite['source_pins']);check_jobs(suite);require_prior_complete(suite)
    require(suite['evaluation_contract']==evaluation_contract(),'smoke evaluation boundary differs')
    for arm in suite['arms']:
        require(validate_config(arm['learner_config'])==default_config(seed=arm['actor_seed'],
            candidate_id=arm['candidate_id'],execution_profile=PROFILE),'learner objective/PPO/init differs')
    require(len(phases)==len(suite['phase_configs'])==3,'exactly three five-shift phases required')
    for index,(cfg,ref) in enumerate(zip(phases,suite['phase_configs'])):
        candidate=CANDIDATES[index]
        require(cfg['candidate_id']==candidate and len(cfg['jobs'])==5 and cfg['arm_update']==1 and
                cfg['arm_phase_id']=='train_u0001' and cfg['phase_id']==candidate+'_train_u0001' and
                cfg['campaign_id']==suite['campaign_id'] and
                Path(cfg['python_output'])==Path(suite['arms'][index]['python_output'])/'train_u0001',
                'phase metadata/output/five-shift boundary differs')
        check_phase(cfg);_phase(ref['path'])
    from baseline1_v1.java_plan import COLUMNS
    for key in ('master_plan','java_helper','common_config','one_click_entry'):
        require(reference(suite[key]['path'])==suite[key],'frozen '+key+' changed')
    with Path(suite['master_plan']['path']).open(newline='') as stream:
        require(list(csv.DictReader(stream))==[{k:str(j[k]) for k in COLUMNS} for j in suite['jobs']],
                'master CSV inventory differs')
    from python.protocol_artifacts import verify_protocol_artifacts
    from python.collect_offline import verify_environment
    verify_protocol_artifacts()
    verify_environment(phases[0]['environment_acceptance'],phases[0]['reward'],storage_migration=True)
    b=budget_snapshot();cid=suite['campaign_id']
    if cid not in b['campaigns']:
        check_collisions(suite,b)
        require(not Path(suite['python_output']).exists() and not (ROOT/'runs'/(cid+'_materials')).exists(),
                'new physical identity already occupied')
        require(preceding_reference()==suite['budget_state_before'] and
                b['maximum_total_shifts']+15==suite['maximum_total_if_approved'],'merged budget changed')
    else:
        require(b['campaigns'][cid]['campaign_sha256']==sha(path),'registered suite differs')
    approval=None
    if approval_path is not None:
        approval=check_approval(json.loads(Path(approval_path).read_bytes()),suite,sha(path))
    return suite,phases,approval
