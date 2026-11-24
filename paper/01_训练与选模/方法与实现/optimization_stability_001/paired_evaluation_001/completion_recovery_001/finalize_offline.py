"""Finish scoring 100 already-COMPLETE slots with bounded episode memory.

No collector, RPC, simulator, optimizer step, ledger writer or retry is called.
The interrupted collector's original progress and log remain in audit evidence.
"""
import copy
import gc
import importlib.util
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[6]
sys.path.insert(0,str(ROOT))
HERE=Path(__file__).resolve().parent


def main():
    raise RuntimeError('DISABLED: sandbox PID invisibility was not process exit; host controller 634568 was confirmed alive. Never run this recovery against the live campaign.')
    import torch
    from baseline1_v1.evaluation import ReadOnlyTrainingGuard,training_state_fingerprint
    from grade_only_ppo_v2.runtime import _controller_lock,_actor_snapshot
    from business_graph_stability_evaluation_v1.gate import PLAN,REPORT,preflight,reference,check_sources
    from business_graph_stability_evaluation_v1 import runtime
    from business_graph_stability_evaluation_v1.metrics import build_report,write_reports
    from business_graph_stability_evaluation_v1.budget import GLOBAL_PATH,connection,_payload,_state
    from business_graph_stability_campaign_v1.plan_io import write,encoded
    from business_graph_stability_campaign_v1.health import _process_identity
    from business_graph_stability_campaign_v1.collection import load_admitted_episode
    from business_graph_summary_v1.evaluation import evaluate_episodes
    from single_critic_task_query_v1.evaluation import _queue_samples
    from single_critic_task_query_v1.policy import replay_policy
    from business_graph_summary_v1.models import digest
    from online_v1_1.operations import _publish
    suite_path=PLAN/'prepared_001/suite.json';suite,cfg,approval=preflight(suite_path)
    output=Path(suite['python_output']);report_dir=REPORT/'completion_recovery_001'
    launch=PLAN/'prepared_001/launch/launch_receipt.json';launched=json.loads(launch.read_bytes())
    process=_process_identity(launched['pid'])
    assert process.get('exists') is False or process.get('start_ticks')!=launched['process_start_ticks'], 'original controller still alive'
    assert not (output/'suite_complete.json').exists() and not (output/'evaluation_complete.json').exists()
    assert not (output/'control/STOP.properties').exists()
    refs_before=[reference(GLOBAL_PATH),*suite['parent_artifacts'].values(),suite['training_suite'],suite['training_completion']]
    refs_before += [value for value in suite['budget_state_before'].values() if isinstance(value,dict) and set(value)=={'path','sha256'}]
    core=ROOT.parents[1]/'python/episode_budget_ppo.py';refs_before.append(reference(core))
    with connection(GLOBAL_PATH,readonly=True) as con:
        entry=_payload(con.execute('SELECT payload,checksum FROM campaigns WHERE id=?',(suite['campaign_id'],)).fetchone())
        state=_state(con,suite,entry)
    assert len(state['jobs'])==100 and all(row['status']=='COMPLETE' for row in state['jobs'].values())
    assert state['campaigns'][suite['campaign_id']]['commits']=={}
    write(report_dir/'before.json',dict(status='ALL100_PHYSICAL_COMPLETE_REPORTS_NOT_PUBLISHED',
        source_progress=json.loads((output/'run_status.json').read_bytes()),
        original_controller=process,launch_receipt=reference(launch),controller_log=reference(PLAN/'prepared_001/launch/controller.log'),
        protected_references=refs_before,physical_shifts=100,new_physical_shifts=0,
        exit_cause='not_determined_from_available_log',recovery='offline_single_episode_scoring_only'))
    spec=importlib.util.spec_from_file_location('bounded_eval_aggregate',HERE/'aggregate_records.py')
    aggregate=importlib.util.module_from_spec(spec);spec.loader.exec_module(aggregate)
    assert reference(ROOT/'single_critic_task_query_v1/evaluation.py')['sha256']==aggregate.SOURCE_SHA256
    torch.set_num_threads(2)
    with _controller_lock(output):
        model=runtime._load_model(suite)
        assert (output/'actor.pt').is_file()
        actor_ref=_actor_snapshot(model,output/'actor.pt')
        actor_identity={k:actor_ref[k] for k in ('path','sha256')}
        portable_before=runtime._portable_state_fingerprint(model)
        before=training_state_fingerprint(model)
        fingerprints=dict(actor=digest(model.actor),critic=digest(model.critic))
        records=[];queues=[];entropies=[];refs=[]
        with ReadOnlyTrainingGuard(model):
            for ordinal,job in enumerate(suite['jobs']):
                row=state['jobs'][str(job['run_index'])];receipt=row['receipt']
                assert row['identity']==job and receipt['normal_completion'] and receipt['four_windows']==4
                assert receipt['phase_config']==suite['phase_config'] and receipt['references']['actor']==actor_identity
                for ref in receipt['references'].values():assert reference(ref['path'])==ref
                assert reference(row['dataset_ref']['path'])==row['dataset_ref']
                episode=load_admitted_episode(receipt)
                identity=episode['identity']
                assert identity['run_index']==job['run_index'] and identity['environment_seed']==job['experiment_seed']
                assert identity['actor_sampling_seed']==job['actor_sampling_seed'] and identity['policy_version']==model.policy_version
                assert identity['policy_sha256']==actor_ref['sha256'] and identity['split']=='validation'
                report=evaluate_episodes([episode],model,expected_complete_shifts=1,
                    checkpoint_provenance=suite['parent_artifacts']['checkpoint'])
                records.extend(report['per_episode'])
                queues.append(_queue_samples(episode['states'][:-1]))
                with torch.no_grad():
                    entropies.append(replay_policy(model.actor,episode['states'][:-1])['entropies'].cpu().numpy().astype(float))
                refs.append(row['dataset_ref'])
                write(report_dir/'per_shift'/f"run_{job['run_index']}.json",report['per_episode'][0])
                model._summary_cache.clear()
                del episode,report
                gc.collect()
                if (ordinal+1)%5==0:
                    print('OFFLINE_SCORED '+str(ordinal+1)+'/100',flush=True)
            report=aggregate.aggregate(records,queues,entropies,model,suite['parent_artifacts']['checkpoint'])
        assert training_state_fingerprint(model)==before and runtime._portable_state_fingerprint(model)==portable_before
        assert all(reference(ref['path'])==ref for ref in refs_before)
        check_sources(suite['source_pins'])
        recovery=dict(schema='bg128-paired-evaluation-offline-finalization-v1',status='PASS',
            suite=reference(suite_path),before_evidence=reference(report_dir/'before.json'),
            aggregate_source=reference(HERE/'aggregate_records.py'),implementation=reference(__file__),
            physical_shifts_already_complete=100,new_physical_shifts=0,collector_calls=0,rpc_starts=0,
            optimizer_updates=0,gradient_updates=0,source_refs=refs,
            model_state_before=portable_before,model_state_after=runtime._portable_state_fingerprint(model),
            protected_artifacts_and_ledgers_unchanged=True,reason='original controller exited before final reports; cause undetermined')
        write(report_dir/'recovery_receipt.json',recovery)
        recovery_ref=reference(report_dir/'recovery_receipt.json')
        from business_graph_stability_v1.contract import VERSION,ablation_contract
        report['representation']=dict(version=VERSION,graph_contract=ablation_contract(),
            metric_implementation='single_critic_task_query_v1.evaluation.evaluate_episodes',
            metric_definition_changed=False,model_provided=True)
        report.update(campaign=reference(suite_path),phase_config=suite['phase_config'],source_refs=refs,
            selection_review=suite['selection'],evaluation_scope='paired_seed_fixed_checkpoint_independent_revalidation',
            smoke_only=False,stable_policy_claim=False,checkpoint_selected_before_sampling=True,
            model_state_before=portable_before,model_state_after=runtime._portable_state_fingerprint(model),
            completion_recovery=recovery_ref)
        runtime._same_or_save(output/'evaluation_complete.json',report)
        reports=write_reports(build_report(report,expected_complete_shifts=100),report,output/'reports')
        reports=dict(reports,evaluation=reference(output/'evaluation_complete.json'))
        binding=dict(campaign=reference(suite_path),phase_config=suite['phase_config'],
            checkpoint=suite['parent_artifacts']['checkpoint'],actor=actor_ref,source_refs=refs)
        receipt=dict(schema='business-graph-stability-fixed-checkpoint-independent-evaluation-receipt-v1',**binding,
            reports=reports,parent_artifacts_before=suite['parent_artifacts'],parent_artifacts_after=suite['parent_artifacts'],
            model_state_before=portable_before,model_state_after=runtime._portable_state_fingerprint(model),
            physical_shifts=100,checkpoint_update=13,optimizer_updates=0,training_updates=0,completion_recovery=recovery_ref)
        runtime._same_or_save(output/'evaluation_receipt.json',receipt)
        reports['receipt']=reference(output/'evaluation_receipt.json')
        complete=dict(schema='business-graph-stability-fixed-checkpoint-independent-evaluation-complete-v1',
            campaign=reference(suite_path),campaign_id=suite['campaign_id'],
            parent_artifacts_before=suite['parent_artifacts'],parent_artifacts_after=suite['parent_artifacts'],
            fingerprints=fingerprints,checkpoint_update=13,physical_shifts=100,selection_review=suite['selection'],
            training_shifts=0,training_updates=0,optimizer_updates=0,gradient_updates=0,manual_anylogic_launches=1,
            source_refs=refs,reports=reports,evaluation_scope='paired_seed_fixed_checkpoint_independent_revalidation',
            completion_recovery=recovery_ref)
        runtime._same_or_save(output/'suite_complete.json',complete)
        runtime._status(suite,'COMPLETE',completed_shifts=100,phase_id=cfg['phase_id'],completion_recovery=recovery_ref)
        print(json.dumps(dict(status='COMPLETE_OFFLINE_NO_NEW_SHIFTS',headline_metrics=report['headline_metrics'],reports=reports)),flush=True)


if __name__=='__main__':main()
