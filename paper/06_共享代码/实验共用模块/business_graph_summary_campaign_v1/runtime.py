"""Frozen BG128 single-seed online PPO campaign, 16 fresh 32-shift batches.

Import/preflight never opens RPC or fits a model. Only the explicitly authorized
run command registers the fixed campaign and starts the transport listener.
There is no historical-behavior replay or automatic additional evaluation.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from online_v1_1.operations import _publish
from baseline1_v1.java_plan import COLUMNS
from grade_only_ppo_v2.runtime import _controller_lock, _actor_snapshot, _equal_tree
from .collection import MaterialPhaseCollector, load_admitted_episode
from .gate import encoded, preflight, reference, require, sha, check_sources
from .progress import write_progress, update_callback

from .contract import CANDIDATES, GROUPS, global_update as _global_update, updates, parameter_digest, ablation_contract, schedule


def _save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _publish(path, encoded(value))


def _same_or_save(path, value):
    path = Path(path)
    if path.exists():
        require(path.read_bytes() == encoded(value), 'immutable AB transaction differs: '+str(path))
    else:
        _save(path, value)


def _arm(suite,candidate):
    chosen=[a for a in suite['arms'] if a['candidate_id']==candidate]
    require(len(chosen)==1,'unique frozen arm required')
    return chosen[0]

def _phase_contract(cfg,suite):
    candidate=cfg['candidate_id'];arm=_arm(suite,candidate)
    local,name=cfg['arm_update'],cfg['arm_phase_id'];jobs=cfg['jobs']
    require(suite['stage']=='formal', 'BG128 campaign is formal training only')
    role=jobs[0]['role'];expected=schedule(False)
    require((name,role,local,len(jobs)) in expected and cfg['phase_id']==candidate+'_'+name
            and cfg['campaign_id']==suite['campaign_id']
            and Path(cfg['python_output'])==Path(arm['python_output'])/name,'phase timeline/output differs')
    for job in jobs:
        require(job['candidate_id']==candidate and job['phase_id']==cfg['phase_id'] and job['role']==role
                and job['arm_update']==local and job['target_training_update']==_global_update(candidate,local)
                and job['checkpoint_update']==(local-1 if role=='train' else local),'job checkpoint differs')
    return arm,role,local,_global_update(candidate,local)

def _learning_contract(suite,candidate):
    config=_arm(suite,candidate)['learner_config']
    from .c_features import feature_contract
    from .targets import composite_contract
    return dict(pair_id=suite['pair_id'],candidate_id=candidate,
        execution_profile=config['execution_profile'],estimator=config['actor_advantage']['estimator'],
        gamma=config['ppo']['gamma'],gae_lambda=config['ppo']['gae_lambda'],
        feature_contract=feature_contract(config['feature_config']),graph_contract=ablation_contract(candidate),
        actor_has_history=False,critic_has_history=False,history_scope='none',
        reward_contract=composite_contract(),critic_target='combined_undiscounted_MC',
        actor_target='combined_cost_GAE_lambda_0.95')


def _targets(path,arrays,intent):
    from python.hdf5_arrays import write_array_bundle,read_array_bundle
    metadata=dict(schema='business-graph-summary-campaign-joint-frozen-targets-v1',transaction=intent,
        reward_version='grade-deadband004-production8-8-wait02-v1',waiting_in_targets=True,production_in_targets=True)
    path=Path(path)
    if path.exists():
        stored,observed=read_array_bundle(path)
        require(observed==metadata and set(stored)==set(arrays)
                and all(np.array_equal(stored[k],arrays[k]) for k in arrays),
                'uncommitted commitment targets differ; reconcile')
    else:write_array_bundle(path,arrays,metadata)
    return reference(path)


def _verify_target_ref(ref,intent):
    from python.hdf5_arrays import read_array_bundle
    require(reference(ref['path'])==ref,'commitment frozen target bytes changed')
    _,metadata=read_array_bundle(ref['path'])
    require(metadata.get('schema')=='business-graph-summary-campaign-joint-frozen-targets-v1' and metadata['transaction']==intent
            and metadata['waiting_in_targets'] is True and metadata['production_in_targets'] is True,
            'commitment target contract/feature/transaction changed')


def _load_model(ref,config,*,restore=True):
    from .learner import JointCostPPO
    return JointCostPPO.from_checkpoint(ref['path'],expected_config=config,
        expected_sha256=ref['sha256'],restore_global_rng=restore)


def _publisher(suite, ledger):
    control=Path(suite['control_directory'])
    def publish(job, config_path):
        check_sources(suite['source_pins'])
        require(reference(config_path) in suite['phase_configs'], 'unfrozen AB phase before ready')
        row=ledger.job(job)
        require(row['status']=='CLAIMED' and row['identity']==job, 'ready needs exact claimed AB slot')
        require(not (control/'STOP.properties').exists(), 'preserved STOP; do not resubmit')
        candidate=job['candidate_id']
        comparison=Path(_arm(suite,candidate)['python_output'])/'initialization_comparison.json'
        require(comparison.is_file(),'capacity and fresh initialization proof required before ready')
        proof=json.loads(comparison.read_bytes())
        require(bool(proof.get('checks')) and all(proof['checks'].values()) and
                proof['input_contract']==ablation_contract(candidate) and
                reference(proof['initial_b']['path'])==proof['initial_b'], 'initialization proof changed')
        require(job['role']=='train', 'this campaign only publishes fresh on-policy training jobs')
        values={key:str(job[key]) for key in COLUMNS}
        values.update(protocol='baseline1-one-click-ready-v1',campaign_id=suite['campaign_id'],
            master_plan_sha256=suite['master_plan']['sha256'],
            instrumentation_revision='boundary-events-v1-native-rpc-runtime-scene-v1',
            config_sha256=sha(config_path),snapshot_sha256=sha(config_path),
            all_comparison_checkpoints_frozen=str(job['role']!='train').lower(),
            snapshot_path=str(Path(config_path).resolve()))
        require(all(value.isascii() and '\n' not in value and '\r' not in value for value in values.values()),
                'ambiguous ready Properties')
        path=control/'ready'/('run_'+str(job['run_index'])+'.properties')
        path.parent.mkdir(parents=True,exist_ok=True)
        _publish(path,''.join(key+'='+value+'\n' for key,value in values.items()).encode('ascii'))
    return publish


def _phase_data(cfg, config_path, suite, ledger, model):
    from .collection import OnlineActor
    _,role,local,_=_phase_contract(cfg,suite)
    require(model.config==_arm(suite,cfg['candidate_id'])['learner_config'] and
            model.update_index==(local-1 if role=='train' else local), 'phase loaded wrong arm/local policy')
    check_sources(suite['source_pins'])
    require(reference(config_path) in suite['phase_configs'], 'phase config source changed')
    actor_ref=_actor_snapshot(model,Path(cfg['python_output'])/'actor.pt')
    state=ledger.resume()
    remaining=[job for job in cfg['jobs'] if state['jobs'][str(job['run_index'])]['status']=='PLANNED']
    fresh_episodes={}
    if remaining:
        active=copy.deepcopy(cfg);active['jobs']=remaining
        behavior=OnlineActor(model.actor,remaining)
        with MaterialPhaseCollector(active,config_path,publish_ready=_publisher(suite,ledger),
                startup=dict(control_directory=suite['control_directory'],campaign_id=suite['campaign_id'],
                             master_plan_sha256=suite['master_plan']['sha256'])) as collector:
            for job in remaining:
                write_progress(suite, status='COLLECTING', phase_id=cfg['phase_id'],
                    candidate_id=cfg['candidate_id'], update_index=local,
                    run_index=job['run_index'], ordinal=job.get('ordinal'))
                ledger.claim(job)
                try:
                    episode,dataset_ref,receipt=collector.collect(job,behavior,actor_path=actor_ref['path'])
                    fresh_episodes[job['run_index']]=episode
                    ledger.complete(job,dataset_ref,receipt)
                except BaseException as error:
                    if ledger.job(job)['status']=='CLAIMED':
                        ledger.unknown(job,repr(error))
                    raise
                print('COMPLETE arm='+cfg['candidate_id']+' phase='+cfg['arm_phase_id']+
                      ' run_index='+str(job['run_index']),flush=True)
    episodes,refs=[],[]
    state=ledger.resume()
    for job in cfg['jobs']:
        row=state['jobs'][str(job['run_index'])]
        require(row['status']=='COMPLETE' and row['identity']==job and
                row['receipt']['phase_config']==reference(config_path), 'durable AB physical slot differs')
        require(reference(row['dataset_ref']['path'])==row['dataset_ref'], 'derived source manifest changed')
        from .collection import _same_references
        _same_references(row['receipt']['references'])
        episode=fresh_episodes.pop(job['run_index'],None)
        if episode is None: episode=load_admitted_episode(row['receipt'])
        identity=episode['identity']
        require(identity['run_index']==job['run_index'] and identity['environment_seed']==job['experiment_seed'] and
                identity['actor_sampling_seed']==job['actor_sampling_seed'] and
                identity['policy_sha256']==actor_ref['sha256'] and identity['policy_version']==model.policy_version,
                'AB phase Actor/episode identity differs')
        episodes.append(episode);refs.append(row['dataset_ref'])
    return episodes,refs,actor_ref


def _batch_id(suite,cfg,refs):
    return hashlib.sha256(encoded(dict(campaign_id=suite['campaign_id'],candidate_id=cfg['candidate_id'],
        phase_id=cfg['phase_id'],arm_update=cfg['arm_update'],sources=refs))).hexdigest()


def _intent(suite,suite_path,cfg,refs):
    _,_,local,global_update=_phase_contract(cfg,suite)
    return dict(learning_contract=_learning_contract(suite,cfg['candidate_id']),campaign=reference(suite_path),
        phase_id=cfg['phase_id'],arm_update=local,update=global_update,batch_id=_batch_id(suite,cfg,refs),
        sources=refs,run_indices=[job['run_index'] for job in cfg['jobs']])


def _verify_checkpoint_restore(model,ref,path,episodes=None):
    from .models import digest
    restored=_load_model(ref,model.config,restore=False)
    checks=dict(actor=restored.policy_version==model.policy_version,critic=digest(restored.critic)==digest(model.critic),
        actor_optimizer=_equal_tree(restored.actor_optimizer.state_dict(),model.actor_optimizer.state_dict()),
        critic_optimizer=_equal_tree(restored.critic_optimizer.state_dict(),model.critic_optimizer.state_dict()),
        generator=_equal_tree(restored.generator.get_state(),model.generator.get_state()),
        sampling_generator=_equal_tree(restored.sampling_generator.get_state(),model.sampling_generator.get_state()),
        consumed_batches=_equal_tree(restored.consumed_batches,model.consumed_batches),
        consumed_episodes=_equal_tree(restored.consumed_episodes,model.consumed_episodes),
        runtime_state=_equal_tree(restored.runtime_state,model.runtime_state),
        frozen_targets=_equal_tree(restored.last_targets,model.last_targets),
        target_hash=restored.last_target_hash==model.last_target_hash,update_index=restored.update_index==model.update_index)
    require(all(checks.values()), 'AB checkpoint exact restore differs')
    _same_or_save(path,dict(schema='business-graph-summary-campaign-joint-checkpoint-restore-v1',checkpoint=ref,
        candidate_id=model.config['candidate_id'],execution_profile=model.config['execution_profile'],
        arm_update=model.update_index,optimizer_updates_in_verification=0,checks=checks))
    if episodes is not None:
        from .evaluation import offline_observation_check
        report=offline_observation_check(model,restored,episodes)
        _same_or_save(Path(path).with_name('offline_observation_consistency.json'),report)


def _train_phase(model,episodes,refs,*,suite,suite_path,cfg,ledger):
    arm,role,local,global_update=_phase_contract(cfg,suite)
    require(role=='train' and model.config==arm['learner_config'], 'training wrong AB candidate')
    require(len(refs)==len(cfg['jobs']) and all(reference(ref['path'])==ref for ref in refs),
            'AB training source bytes changed')
    state=ledger.resume()
    commits=state['campaigns'][suite['campaign_id']]['commits']
    require(global_update==len(commits)+1, 'AB global commit is not the next complete batch')
    require(all(state['jobs'][str(job['run_index'])]['status']=='COMPLETE' and
                state['jobs'][str(job['run_index'])]['identity']==job and
                state['jobs'][str(job['run_index'])]['dataset_ref']==ref for job,ref in zip(cfg['jobs'],refs)),
            'AB training physical sources differ')
    folder=Path(arm['python_output'])/'updates'/('%04d'%local)
    intent=_intent(suite,suite_path,cfg,refs);had_intent=(folder/'intent.json').exists()
    _same_or_save(folder/'intent.json',intent)
    checkpoint=folder/'checkpoint.pt'
    if checkpoint.exists():
        model=_load_model(reference(checkpoint),arm['learner_config'])
        require(model.update_index==local and model.runtime_state.get('transaction')==intent,
                'uncommitted checkpoint belongs to another arm/transaction')
        _verify_target_ref(model.runtime_state['training_targets'],intent)
        require(reference(folder/'training_summary.json')==model.runtime_state['training_summary'],
                'uncommitted AB summary changed')
    else:
        require(not had_intent and not any((folder/name).exists() for name in
                ('training_targets.h5','training_summary.json','prepared.json','checkpoint_restore_verification.json')),
                'incomplete optimizer publication; reconcile without repeating update')
        require(model.update_index==local-1, 'AB local PPO is not the next complete batch')
        model.progress_callback=update_callback(suite,cfg,local)
        try:
            result=model.update(episodes,batch_id=intent['batch_id'])
        finally:
            model.progress_callback=None
        require(model.update_index==local, 'learner advanced wrong arm-local update')
        targets=_targets(folder/'training_targets.h5',model.last_targets,intent)
        _same_or_save(folder/'training_summary.json',dict(transaction=intent,update=result))
        model.runtime_state.update(transaction=intent,training_targets=targets,
                                   training_summary=reference(folder/'training_summary.json'))
        model.save_checkpoint(checkpoint)
    _verify_checkpoint_restore(model,reference(checkpoint),folder/'checkpoint_restore_verification.json',
                               episodes if local==1 else None)
    prepared=dict(checkpoint=reference(checkpoint),intent=intent,training_targets=model.runtime_state['training_targets'])
    if local==1:
        prepared['offline_observation_consistency']=reference(folder/'offline_observation_consistency.json')
    _same_or_save(folder/'prepared.json',prepared)
    ledger.commit(global_update,cfg['jobs'],prepared['checkpoint'],intent['batch_id'])
    write_progress(suite,status='UPDATE_COMMITTED',phase_id=cfg['phase_id'],
        candidate_id=cfg['candidate_id'],update_index=local,completed_epochs=model.config['ppo']['epochs'],
        epochs=model.config['ppo']['epochs'])
    print('COMMITTED arm='+cfg['candidate_id']+' local='+str(local)+' global='+str(global_update),flush=True)
    return model,prepared['checkpoint']


def _initial_reference(suite,candidate):
    return json.loads((Path(_arm(suite,candidate)['python_output'])/'initial_checkpoint.json').read_bytes())


def _checkpoint_for(suite,ledger,candidate,local):
    if local==0:return _initial_reference(suite,candidate)
    return ledger.commit_record(_global_update(candidate,local))['checkpoint_ref']


def _verify_committed(cfg,suite,suite_path,ledger):
    arm,role,local,global_update=_phase_contract(cfg,suite)
    require(role=='train', 'only training phases have global commits')
    folder=Path(arm['python_output'])/'updates'/('%04d'%local)
    prepared=json.loads((folder/'prepared.json').read_bytes())
    commit=ledger.snapshot()['campaigns'][suite['campaign_id']]['commits'][str(global_update)]
    require(prepared['checkpoint']==commit['checkpoint_ref'], 'AB committed checkpoint differs')
    archived=_load_model(prepared['checkpoint'],arm['learner_config'],restore=False)
    require(archived.update_index==local, 'committed model has wrong local update')
    _verify_target_ref(prepared['training_targets'],prepared['intent'])
    require(archived.runtime_state.get('transaction')==prepared['intent'] and
            archived.runtime_state.get('training_targets')==prepared['training_targets'] and
            reference(folder/'training_summary.json')==archived.runtime_state['training_summary'],
            'committed AB transaction sources changed')
    refs=[];episodes=[];state=ledger.resume()
    for job in cfg['jobs']:
        row=state['jobs'][str(job['run_index'])]
        require(row['status']=='COMPLETE' and row['identity']==job and
                row.get('committed_update')==global_update, 'committed AB physical identity differs')
        episodes.append(load_admitted_episode(row['receipt']));refs.append(row['dataset_ref'])
    expected=_intent(suite,suite_path,cfg,refs)
    require(prepared['intent']==expected and commit['batch_id']==expected['batch_id'],
            'committed AB durable inputs changed')
    if local==1:
        proof=folder/'offline_observation_consistency.json'
        require(prepared.get('offline_observation_consistency')==reference(proof),
                'committed captured-observation proof changed')
        _verify_checkpoint_restore(archived,prepared['checkpoint'],
                                   folder/'checkpoint_restore_verification.json',episodes)


def _initial_arm(suite,suite_path,arm):
    from .learner import JointCostPPO
    output=Path(arm['python_output']);initial=output/'initial_checkpoint.json';config=arm['learner_config']
    _same_or_save(output/'authorized_learner_config.json',dict(config=config,campaign=reference(suite_path)))
    if initial.exists():
        ref=json.loads(initial.read_bytes());model=_load_model(ref,config,restore=False)
    else:
        require(not (output/'initial.pt').exists(), 'unpublished arm initial checkpoint; reconcile')
        model=JointCostPPO(config)
        model.runtime_state.update(campaign=reference(suite_path),candidate_id=arm['candidate_id'])
        model.save_checkpoint(output/'initial.pt');ref=reference(output/'initial.pt');_save(initial,ref)
    require(model.update_index==0 and not model.consumed_batches and not model.consumed_episodes and
            not model.last_targets and model.runtime_state==dict(campaign=reference(suite_path),candidate_id=arm['candidate_id']),
            'arm initial checkpoint is not the independent fresh origin')
    return model,ref


def _compare_initializations(suite,model_b,ref_b):
    # Recreate the same frozen BG128 configuration without loading weights.
    # All learned tensors must match, with independent storage and empty Adam.
    from .learner import JointCostPPO
    import torch
    checks={}
    comparisons=[]
    candidate=model_b.config['candidate_id']
    for reference_candidate in CANDIDATES:
        cfg=copy.deepcopy(model_b.config)
        cfg['candidate_id']=reference_candidate
        cfg['graph_contract']=ablation_contract(reference_candidate)
        from .contract import feature_config_for
        cfg['feature_config']=feature_config_for(reference_candidate)
        other=JointCostPPO(cfg)
        for role in ('actor','critic'):
            a=dict(getattr(other,role).named_parameters())
            b=dict(getattr(model_b,role).named_parameters())
            names=sorted(set(a)&set(b))
            key=reference_candidate+'_'+role
            names=[n for n in names if a[n].shape==b[n].shape]
            checks[key+'_common_bytes']=bool(names) and all(torch.equal(a[n],b[n]) for n in names)
            checks[key+'_storage_independent']=all(a[n].untyped_storage().data_ptr()!=b[n].untyped_storage().data_ptr() for n in names)
            comparisons.append(dict(reference=reference_candidate,role=role,names=names))
        checks[reference_candidate+'_fresh']=other.update_index==model_b.update_index==0 and not other.consumed_batches and not model_b.consumed_batches
        checks[reference_candidate+'_empty_optimizers']=not other.actor_optimizer.state and not other.critic_optimizer.state and not model_b.actor_optimizer.state and not model_b.critic_optimizer.state
        checks[reference_candidate+'_rng']=torch.equal(other.generator.get_state(),model_b.generator.get_state()) and torch.equal(other.sampling_generator.get_state(),model_b.sampling_generator.get_state())
    actor_pointers={p.untyped_storage().data_ptr() for p in model_b.actor.parameters()}
    critic_pointers={p.untyped_storage().data_ptr() for p in model_b.critic.parameters()}
    checks['actor_critic_storage_disjoint']=not actor_pointers.intersection(critic_pointers)
    require(all(checks.values()),'named paired fresh initialization mismatch')
    path=Path(_arm(suite,candidate)['python_output'])/'initialization_comparison.json'
    _same_or_save(path,dict(initial_b=ref_b,checks=checks,input_contract=ablation_contract(candidate),
        common_named_comparisons=comparisons,parameter_digest=parameter_digest(model_b.actor),
        optimizer_updates_in_comparison=0,old_weights_loaded=False,parameter_counts_matched=True))
    return reference(path)


def _finish_arm(suite_path,suite,phases,arm,ledger):
    candidate=arm['candidate_id']
    state=ledger.resume();commits=state['campaigns'][suite['campaign_id']]['commits']
    require(all(str(_global_update(candidate,u)) in commits for u in range(1,17)),
            'BG128 training updates incomplete')
    require(len(phases)==16 and all(cfg['jobs'][0]['role']=='train' for cfg in phases),
            'this campaign only authorizes 16 training batches')
    result=dict(status='BUSINESS_GRAPH_SUMMARY_ARM_TRAINING_COMPLETE_SINGLE_SEED',
        candidate_id=candidate,campaign=reference(suite_path),learning_contract=_learning_contract(suite,candidate),
        initial_checkpoint=_initial_reference(suite,candidate),checkpoint=_checkpoint_for(suite,ledger,candidate,16),
        evaluation_reports=[],arm_updates=16,physical_shifts=512,training_shifts=512,
        initialization_comparison=reference(Path(arm['python_output'])/'initialization_comparison.json'),
        production_guarantee=False,multi_seed_claim=False,stable_policy_claim=False,
        independent_evaluation=False,additional_physical_evaluation_shifts=0,
        waiting_in_training_objective=True,production_in_training_objective=True)
    _same_or_save(Path(arm['python_output'])/'arm_complete.json',result)
    return result


def _trial(suite_path,suite,phases,ledger):
    import torch
    torch.set_num_threads(suite['cpu_threads'])
    require(suite['stage']=='formal' and suite['arm_order']==['s01_BG128'] and
            len(suite['arms'])==1 and len(phases)==16,
            'only the frozen single BG128 formal campaign is executable')
    for arm in suite['arms']:
        model,initial=_initial_arm(suite,suite_path,arm)
        _compare_initializations(suite,model,initial)
        del model
    for cfg,spec in zip(phases,suite['phase_configs']):
        arm,role,local,global_update=_phase_contract(cfg,suite)
        require(role=='train','this campaign has no separately authorized evaluation shifts')
        state=ledger.resume();commits=state['campaigns'][suite['campaign_id']]['commits']
        if str(global_update) in commits:
            _verify_committed(cfg,suite,suite_path,ledger);continue
        parent=_checkpoint_for(suite,ledger,cfg['candidate_id'],local-1)
        model=_load_model(parent,arm['learner_config'])
        require(model.update_index==local-1,'training parent is not previous local update')
        episodes,refs,_=_phase_data(cfg,spec['path'],suite,ledger,model)
        model,_=_train_phase(model,episodes,refs,suite=suite,suite_path=suite_path,cfg=cfg,ledger=ledger)
        del episodes,model
    results=[_finish_arm(suite_path,suite,phases,a,ledger) for a in suite['arms']]
    state=ledger.resume();commits=state['campaigns'][suite['campaign_id']]['commits']
    require(sorted(map(int,commits))==list(range(1,17)) and
            all(state['jobs'][str(j['run_index'])]['status']=='COMPLETE' for j in suite['jobs']),
            'BG128 training inventory incomplete')
    output=Path(suite['python_output'])
    result=dict(status='BUSINESS_GRAPH_SUMMARY_TRAINING_COMPLETE_SINGLE_SEED',
        campaign=reference(suite_path),physical_shifts=512,training_shifts=512,global_updates=16,
        arms=[reference(Path(a['python_output'])/'arm_complete.json') for a in suite['arms']],
        checkpoints=[r['checkpoint'] for r in results],production_guarantee=False,training_seed_count=1,
        independent_evaluation=False,additional_physical_evaluation_shifts=0,
        stable_policy_claim=False,waiting_in_training_objective=True,production_in_training_objective=True)
    _same_or_save(output/'suite_complete.json',result)
    write_progress(suite,status='COMPLETE',committed_updates=16,physical_shifts=512)
    return result


def run(suite_path,approval_path):
    from .budget import BudgetLedger,GLOBAL_PATH
    suite,phases,approval=preflight(suite_path,approval_path=approval_path)
    require(approval is not None, 'exact-suite budget/execution approval required')
    ledger=BudgetLedger(GLOBAL_PATH,suite,campaign_sha256=sha(suite_path),approval=approval,approval_sha256=sha(approval_path))
    output=Path(suite['python_output'])
    with _controller_lock(output):
        ledger.resume();stop=Path(suite['control_directory'])/'STOP.properties'
        require(not stop.exists(), 'preserved STOP requires recovery review; never clear automatically')
        try:return _trial(suite_path,suite,phases,ledger)
        except BaseException as error:
            if not stop.exists():
                stop.parent.mkdir(parents=True,exist_ok=True)
                _publish(stop,('reason='+repr(error).replace('\n',' ')+'\n').encode())
            raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    for name in ('preflight','run'):
        command=sub.add_parser(name);command.add_argument('--suite',required=True)
        command.add_argument('--approval',required=name=='run')
    args=parser.parse_args(argv)
    if args.command=='preflight':
        suite,phases,approval=preflight(args.suite,approval_path=args.approval)
        print(json.dumps(dict(status='PREFLIGHT_PASS_NO_LISTENER',suite=reference(args.suite),
            physical_shifts=suite['new_shift_cap'],phases=len(phases),execution_authorized=approval is not None,
            manual_anylogic_launches=1,arms=suite['arm_order'],scope='single_seed_fresh_on_policy_training_only_no_independent_evaluation'),ensure_ascii=False))
        return 0
    run(args.suite,args.approval);return 0


if __name__=='__main__':raise SystemExit(main())
