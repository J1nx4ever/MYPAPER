"""Read-only gates for one frozen BG128 seed and 512 on-policy training shifts."""
import copy
import csv
import json
from pathlib import Path

from .contract import CANDIDATES, ablation_contract, feature_config_for, global_update, schedule, seed, require
from .plan_io import ROOT, PLAN, REPORT, encoded, sha, reference, load_reference, load_suite
from business_graph_stability_v1.targets import composite_contract, REWARD_VERSION

REPO=ROOT.parents[1]
PROTECTED_SHA='8e4cfa31c622f5e33756e304eea3f7526547db72ae93f7d9f00808845d61ef96'
CAMPAIGN_CONTRACT='business-graph-stability-single-seed-suite-v1'
PHASE_CONTRACT='business-graph-stability-phase-v1'
TRAINING_REWARD='negative_combined_grade_production_wait_cost_v1'
DIAGNOSTIC_SCOPE='disabled_high_frequency_diagnostics'


def evaluation_contract(engineering=False):
    require(engineering is False, 'this package authorizes formal training only')
    return dict(independent_physical_evaluation_shifts_per_arm=0,
        sampling='stochastic_private_rng',fixed_final_update=13,archive_final_update=16,
        prospective_independent_shifts=100,optimizer_updates=0,
        training_seed_count=1,multi_seed_claim=False,smoke_checkpoint_reused_for_formal=False,
        additional_campaign_auto_start=False,
        rmse_empty_window_policy='undefined_excluded_with_explicit_empty_count',
        metrics=['four_window_grade_rmse','joint_four_window_compliance_rate','ore_target_met_rate',
                 'waste_target_met_rate','complete_task_rate','fleet_wait_vehicle_seconds',
                 'critic_mc_mse','explained_variance'])


def source_pins():
    # Reuse all transport/target/schema pins; never edit their sealed sources.
    from single_critic_task_query_v1.gate import source_pins as inherited
    pins=inherited()
    for package in ('business_graph_summary_v1','business_graph_summary_campaign_v1',
                    'business_graph_summary_evaluation_v1',
                    'business_graph_summary_storage_v1','business_graph_summary_recovery_v1',
                    'business_graph_stability_v1','business_graph_stability_campaign_v1'):
        pins.update({str(p.resolve()):sha(p) for p in (ROOT/package).glob('*.py')
                     if not p.name.startswith('test_')})
    for name in ('bootstrap.py','prepare_training.py','protocol.json','pairing_manifest.json'):
        p=PLAN/name
        require(p.is_file(),'campaign preparation source missing: '+str(p))
        pins[str(p.resolve())]=sha(p)
    return dict(sorted(pins.items()))


def check_sources(pins):
    require(pins==source_pins(),'implementation changed; prepare a new immutable package')
    require(sha(REPO/'python/episode_budget_ppo.py')==PROTECTED_SHA,'protected PPO core changed')
    from python import observation_contract
    require(Path(observation_contract.__file__).resolve().parent==ROOT/'python','experiment Python import path differs')
    require(sha(ROOT/'python/observation_schema.json')=='c242ce76856d1cd929af738da3942bab312749712603626d6c2747ac47d12637',
        'observation schema changed')


def check_phase(cfg):
    from baseline1_v1 import gate as transport
    from business_graph_stability_v1.c_features import feature_contract
    from grade_only_ppo_v1.gate import MEMORY
    require(cfg['contract']==PHASE_CONTRACT and cfg['candidate_id'] in CANDIDATES,'unknown phase/candidate')
    raw=copy.deepcopy(cfg);raw['contract']=transport.PHASE_CONTRACT
    transport.check_phase(raw)
    require(cfg['feature_contract']==feature_contract(feature_config_for(cfg['candidate_id'])) and
        cfg['graph_contract']==ablation_contract(cfg['candidate_id']) and
        cfg['composite_contract']==composite_contract(),'phase frozen graph/C128/cost differs')
    require(cfg['material_memory']==dict(MEMORY,reward_version=REWARD_VERSION),'journal identity differs')
    require(cfg['execution_profile']=='formal_32ep_v1' and cfg['training_reward']==TRAINING_REWARD and
        cfg['diagnostic_scope']==DIAGNOSTIC_SCOPE,'training profile/meaning differs')
    require(cfg['diagnostic_recording']==dict(transition_jsonl=False,rpc_jsonl=False,rollout_snapshots=False) and
        cfg['storage']==dict(format='hdf5-v1',chunk_rows=64),'high frequency diagnostics must remain off')
    require(cfg['rpc_configuration_authoritative'] is True and cfg['require_local_alp_audit'] is False,
        'RPC configuration authority required')
    expected=dict(shift_seconds=28800.,window_seconds=7200.,target_ore_tons_shift=16000.,
        target_waste_tons_shift=4000.,target_grade=.27,grade_lower=.26,grade_upper=.28,
        grade_epsilon=1e-12,tonnage_epsilon=1e-6,w_production=4.,waste_shortfall_weight=4.,wait_kappa=.00004,
        material_event_memory_enabled=1.,material_event_contract_version=1.,waste_task_limit_enabled=1.,
        waste_task_limit_version=2.,waste_task_dispatch_stop_tons=6000.,waste_task_upper_tons=6193.,waste_task_overrun_tons=193.)
    for job in cfg['jobs']:
        for key,value in expected.items():
            require(job['expected_rpc_config'].get('config.'+key)==value,'explicit live RPC field differs: '+key)
    return cfg


def check_jobs(suite):
    require(suite['stage']=='formal' and suite['execution_profile']=='formal_32ep_v1','only frozen formal profile supported')
    require(suite['arm_order']==list(CANDIDATES) and [a['candidate_id'] for a in suite['arms']]==list(CANDIDATES),
        'exactly one BG128 arm required')
    require(suite['new_shift_cap']==len(suite['jobs'])==512 and suite['training_shifts']==512 and
        suite['training_updates']==16 and suite['evaluation_shifts']==0 and suite['training_seed_count']==1 and
        suite['episodes_per_batch']==32 and suite['cpu_threads']==2,'frozen single-seed training scope differs')
    expected=[(candidate,name,role,local,n) for name,role,local,n in schedule(False) for candidate in CANDIDATES]
    require(len(expected)==len(suite['phase_configs'])==16,'phase inventory differs')
    output=ROOT/'runs'/suite['campaign_id'];materials=output.with_name(suite['campaign_id']+'_materials')
    from .pairing import check_seed_pairing
    source=check_seed_pairing(suite)
    init=source['arms'][0]['actor_seed']
    ordinal=0
    for candidate,name,role,local,n in expected:
        phase=candidate+'_'+name;recording=suite['campaign_id']+'-'+phase
        for inside in range(n):
            j=suite['jobs'][ordinal];pair=(local-1)*32+inside;ns=suite['campaign_id']+'-s01-'+role
            require(j['ordinal']==ordinal and j['run_index']==suite['run_index_base']+ordinal and
                j['candidate_id']==candidate and j['phase_id']==phase and j['arm_phase_id']==name and
                j['arm_update']==local and j['role']==role=='train' and j['partition']=='train' and
                j['pair_ordinal']==pair and j['checkpoint_update']==local-1 and
                j['target_training_update']==global_update(candidate,local),'job timeline differs')
            original=source['jobs'][ordinal]
            require(j['actor_seed']==init and j['experiment_seed']==original['experiment_seed'] and
                j['actor_sampling_seed']==original['actor_sampling_seed'],'corresponding paired seed stream differs')
            require(j['recording_id']==j['source_recording_id']==recording and
                Path(j['python_output'])==output/'arms'/candidate/name and
                Path(j['material_event_directory'])==materials/phase,'recording/output identity differs')
            ordinal+=1
    require(ordinal==512 and len({j['run_index'] for j in suite['jobs']})==512,'physical run reused')
    streams=[{j[key] for j in suite['jobs']} for key in ('actor_seed','experiment_seed','actor_sampling_seed')]
    require([len(s) for s in streams]==[1,512,512] and
        all(streams[i].isdisjoint(streams[j]) for i in range(3) for j in range(i)),'cross-type seed collision')
    require(suite['legacy_reserved_shifts']==suite['active_h2_reserved_shifts']==0,'cancelled H2 reservation reintroduced')
    require(suite.get('predecessor_suites')==[] and suite.get('pending_predecessor_shifts',0)==0,
        'no separate smoke/evaluation campaign is included')
    require(suite['formal_model_inherits_smoke_weights'] is False and
        suite['initialization']=='original_common_initial_checkpoint_zero_updates','original common initialization required')
    return suite


def require_prior_complete(suite, *, allow_pending_smoke=False):
    acceptance=load_reference(suite['engineering_acceptance'])
    require(acceptance['status']=='PASS_OFFLINE_ONLY' and acceptance['source_pins']==suite['source_pins'] and
        acceptance.get('schema')=='bg128-stability-campaign-offline-acceptance-v1' and
        bool(acceptance.get('checks')) and all(acceptance['checks'].values()),'current BG128 campaign offline acceptance required')
    frozen=load_reference(acceptance['frozen_architecture_validation'])
    require(frozen['tests_passed'] and all(frozen['real_forward_checks'].values()) and
        frozen['protected_ppo_sha256']['before']==frozen['protected_ppo_sha256']['after']==PROTECTED_SHA,
        'frozen BG128 architecture evidence differs')
    for artifact in acceptance['verification_artifacts']:
        require(reference(artifact['path'])==artifact,'campaign verification evidence changed')
    require(acceptance['formal_first_batch_is_end_to_end_validation'] is True and
        acceptance['separate_physical_smoke_shifts']==0,'first formal batch is not an additional smoke campaign')
    return True


def check_approval(approval,suite,suite_sha256):
    require(approval.get('decision')=='approved' and approval.get('approved_by')=='user' and
        approval.get('execution_authorized') is True and approval.get('ack_frozen_bg128_single_seed') is True and
        bool(approval.get('source_user_instruction')) and approval.get('campaign_id')==suite['campaign_id'] and
        approval.get('execution_profile')==suite['execution_profile'] and approval.get('pair_id')==suite['pair_id'] and
        approval.get('campaign_sha256')==suite_sha256 and approval.get('max_new_shifts')==512==suite['new_shift_cap'] and
        approval.get('maximum_total_after_registration')==suite['maximum_total_if_approved'] and
        approval.get('accounting')=='merged_all_ledgers_no_h2_future_reservation',
        'authorization must bind this frozen one-seed training scope and identities')
    return approval


def preflight(path, *, approval_path=None, allow_pending_smoke=False):
    from business_graph_stability_v1.learner import default_config,validate_config
    from .budget import budget_snapshot,check_collisions,preceding_reference
    from .admission import _phase
    suite,phases=load_suite(path)
    require(suite['contract']==CAMPAIGN_CONTRACT and suite['suite_path']==str(Path(path).resolve()),'suite identity differs')
    check_sources(suite['source_pins']);check_jobs(suite);require_prior_complete(suite)
    require(suite.get('optimization_protocol')==reference(PLAN/'protocol.json'), 'optimization protocol missing or changed')
    protocol=load_reference(suite['optimization_protocol'])
    require(protocol['intervention']==dict(actor_lr_before=3e-4,actor_lr_after=1e-4) and
        protocol['selection']['primary_checkpoint_update']==13 and
        protocol['training']['updates']==16 and protocol['training']['physical_shifts']==512 and
        protocol['training']['adaptive_kl_stop'] is False,'registered LR/stop/selection protocol differs')
    require(suite['evaluation_contract']==evaluation_contract(False),'evaluation definition/scope differs')
    for arm in suite['arms']:
        require(validate_config(arm['learner_config'])==default_config(seed=arm['actor_seed'],
            candidate_id=arm['candidate_id'],execution_profile=suite['execution_profile']),
            'learner differs from frozen graph/target/PPO')
    for cfg,ref in zip(phases,suite['phase_configs']):check_phase(cfg);_phase(ref['path'])
    for key in ('master_plan','java_helper','common_config','one_click_entry'):
        require(reference(suite[key]['path'])==suite[key],'frozen '+key+' changed')
    from baseline1_v1.java_plan import COLUMNS
    with Path(suite['master_plan']['path']).open(newline='') as handle:
        require(list(csv.DictReader(handle))==[{k:str(j[k]) for k in COLUMNS} for j in suite['jobs']],'master CSV differs')
    from python.protocol_artifacts import verify_protocol_artifacts
    from single_critic_task_query_v1.environment_review import verify_environment
    verify_protocol_artifacts();verify_environment(phases[0]['environment_acceptance'],phases[0]['reward'])
    state=budget_snapshot()
    if suite['campaign_id'] not in state['campaigns']:
        check_collisions(suite,state)
        require(not Path(suite['python_output']).exists() and not (ROOT/'runs'/(suite['campaign_id']+'_materials')).exists(),
            'new physical identity occupied')
        require(preceding_reference()==suite['budget_state_before'],'preceding merged ledgers changed')
        require(state['maximum_total_shifts']+512==suite['maximum_total_if_approved'],'merged budget scope changed')
    else:
        require(state['campaigns'][suite['campaign_id']]['campaign_sha256']==sha(path),'registered campaign bytes differ')
    approval=None if approval_path is None else check_approval(json.loads(Path(approval_path).read_bytes()),suite,sha(path))
    return suite,phases,approval
