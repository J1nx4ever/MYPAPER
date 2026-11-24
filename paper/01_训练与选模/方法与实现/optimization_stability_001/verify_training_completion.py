"""Read-only completion and artifact audit; no fitting, collector or listener."""
from collections import Counter
import csv
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from business_graph_stability_campaign_v1.plan_io import PLAN,REPORT,reference,load_suite,write
from business_graph_stability_campaign_v1.gate import check_sources,check_jobs,PROTECTED_SHA
from business_graph_stability_campaign_v1.budget import GLOBAL_PATH,connection,_payload,_state

CACHE={}
def verified(ref):
    path=ref['path']
    if path not in CACHE:CACHE[path]=reference(path)
    assert CACHE[path]==ref, 'artifact bytes changed: '+path
    return Path(path)


def main():
    import h5py
    import torch
    from business_graph_stability_v1.learner import _load_checkpoint as load_new
    from business_graph_summary_v1.learner import _load_checkpoint as load_old
    from business_graph_stability_campaign_v1.pairing import training_source
    from business_graph_stability_evaluation_v1.budget import budget_snapshot
    suite_path=PLAN/'prepared_001/suite.json';suite,phases=load_suite(suite_path)
    check_sources(suite['source_pins']);check_jobs(suite)
    output=Path(suite['python_output']);arm=Path(suite['arms'][0]['python_output'])
    complete=json.loads((output/'suite_complete.json').read_bytes())
    assert complete['campaign']==reference(suite_path) and complete['global_updates']==16
    assert complete['physical_shifts']==512 and not (output/'control/STOP.properties').exists()
    before=reference(GLOBAL_PATH)
    with connection(GLOBAL_PATH,readonly=True) as con:
        entry=_payload(con.execute('SELECT payload,checksum FROM campaigns WHERE id=?',(suite['campaign_id'],)).fetchone())
        state=_state(con,suite,entry)
    rows=state['jobs'];commits=state['campaigns'][suite['campaign_id']]['commits']
    assert Counter(row['status'] for row in rows.values())=={'COMPLETE':512}
    assert sorted(map(int,commits))==list(range(1,17))
    source=suite['pairing']['common_initial_checkpoint']
    old,_=load_old(verified(source));new,_=load_new(arm/'initial.pt')
    init_checks=dict(actor_exact=all(torch.equal(new['actor'][k],v) for k,v in old['actor'].items()),
        critic_exact=all(torch.equal(new['critic'][k],v) for k,v in old['critic'].items()),
        generator_exact=torch.equal(new['generator'],old['generator']),
        sampling_generator_exact=torch.equal(new['sampling_generator'],old['sampling_generator']),
        adam_empty=not new['actor_optimizer']['state'] and not new['critic_optimizer']['state'],
        untouched_u0=new['update_index']==0 and not new['consumed_batches'] and not new['consumed_episodes'],
        source_binding=new['runtime_state']['common_initial_checkpoint']==source)
    assert all(init_checks.values())
    old_arm=Path(training_source()['arms'][0]['python_output'])
    batches=[];transitions=0
    for update,cfg in enumerate(phases,1):
        folder=arm/f'updates/{update:04d}'
        prepared=json.loads((folder/'prepared.json').read_bytes())
        checkpoint=verified(prepared['checkpoint'])
        assert commits[str(update)]['checkpoint_ref']==prepared['checkpoint']
        manifest=json.loads(Path(str(checkpoint)+'.manifest.json').read_bytes())
        seal=json.loads(Path(str(checkpoint)+'.seal.json').read_bytes())
        assert manifest['sha256']==seal['sha256']==prepared['checkpoint']['sha256']
        assert reference(str(checkpoint)+'.manifest.json')['sha256']==seal['manifest_sha256']
        assert manifest['update_index']==update
        verified(prepared['training_targets'])
        restored=json.loads((folder/'checkpoint_restore_verification.json').read_bytes())
        assert restored['checkpoint']==prepared['checkpoint'] and all(restored['checks'].values())
        summary_path=folder/'training_summary.json';summary=json.loads(summary_path.read_bytes())
        assert summary['transaction']==prepared['intent']
        summary_update=summary['update'];assert summary_update['update_index']==update
        assert summary_update['episodes']==32 and len(summary_update['epoch_diagnostics'])==4
        count=0
        for job in cfg['jobs']:
            row=rows[str(job['run_index'])];receipt=row['receipt']
            assert row['identity']==job and row['committed_update']==update
            assert receipt['normal_completion'] and receipt['run_index']==job['run_index']
            assert receipt['experiment_seed']==job['experiment_seed'] and receipt['four_windows']
            for ref in receipt['references'].values():verified(ref)
            verified(row['dataset_ref'])
            with h5py.File(receipt['references']['episode']['path'],'r') as h:
                identity=json.loads(h.attrs['identity_json'])
                assert identity==receipt['identity'] and identity['run_index']==job['run_index']
                assert identity['environment_seed']==job['experiment_seed'] and identity['actor_sampling_seed']==job['actor_sampling_seed']
                n=len(h['transitions/action']);assert len(h['states'])==n+1 and n==receipt['transitions']
                count+=n
        assert count==summary_update['transitions'];transitions+=count
        old_summary=json.loads((old_arm/f'updates/{update:04d}/training_summary.json').read_bytes())['update']
        metrics=summary_update['metrics'];old_metrics=old_summary['metrics']
        batches.append(dict(update=update,sampled_policy_update=update-1,episodes=32,transitions=count,
            sampled_cost=metrics['combined_cost_mean'],historical_same_seed_sampled_cost=old_metrics['combined_cost_mean'],
            post_update_kl=metrics['approx_kl'],historical_post_update_kl=old_metrics['approx_kl'],
            post_update_clip_fraction=metrics['clip_fraction'],historical_post_update_clip_fraction=old_metrics['clip_fraction'],
            critic_training_mc_mse=metrics['critic_mc_mse'],critic_training_ev=metrics['explained_variance'],
            epoch_diagnostics=summary_update['epoch_diagnostics'],checkpoint=prepared['checkpoint']))
    assert reference(GLOBAL_PATH)==before
    snapshot=budget_snapshot()
    checks=dict(complete512=True,commits16=True,no_unknown_or_claimed=True,all_referenced_bytes_verified=True,
        all_checkpoint_restore_receipts_passed=True,same_initial_checkpoint=True,paired_seeds=True,
        original_source_pins_unchanged=True,all_epochs_have_diagnostics=True,training_ledger_unchanged=True,
        protected_core_unchanged=reference(ROOT.parents[1]/'python/episode_budget_ppo.py')['sha256']==PROTECTED_SHA)
    assert all(checks.values())
    directory=REPORT/'training_completion_001'
    result=dict(status='PASS_TRAINING_COMPLETE_ARTIFACT_AUDIT',checks=checks,suite=reference(suite_path),
        completion=reference(output/'suite_complete.json'),training_budget=before,initial_checkpoint_checks=init_checks,
        physical_shifts=512,training_updates=16,decision_transitions=transitions,
        primary_checkpoint=batches[12]['checkpoint'],archived_final_checkpoint=batches[15]['checkpoint'],
        policy_selection='preregistered_u13_unchanged',merged_reserved_shifts=snapshot['maximum_total_shifts'],
        audit_scope='completion/provenance/seals/identity/recorded diagnostics; not an independent policy evaluation or fresh MC reconstruction',
        verified_unique_files=len(CACHE),new_physical_shifts=0,new_optimizer_steps=0,batches=batches)
    write(directory/'audit.json',result)
    columns=['sampled_policy_update','update','episodes','transitions','sampled_cost','historical_same_seed_sampled_cost',
             'post_update_kl','historical_post_update_kl','post_update_clip_fraction','historical_post_update_clip_fraction',
             'critic_training_mc_mse','critic_training_ev']
    with (directory/'training_comparison.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        writer.writerows({k:row[k] for k in columns} for row in batches)
    print(json.dumps(dict(status=result['status'],physical_shifts=512,training_updates=16,decision_transitions=transitions,
        merged_reserved_shifts=snapshot['maximum_total_shifts'],primary_checkpoint=result['primary_checkpoint'],audit=reference(directory/'audit.json'))))


if __name__=='__main__':main()
