"""Fixed-state policy movement across frozen 3e-4 checkpoints, no updates."""
import h5py
import numpy as np
import torch
from common import *
from baseline1_v1.evaluation import ReadOnlyTrainingGuard, training_state_fingerprint
from business_graph_summary_v1.learner import JointCostPPO
from business_graph_summary_v1.models import pack_summary_prefixes
from business_graph_summary_v1.targets import build_composite_targets
from business_graph_summary_storage_v1.episode import _read_table
from grade_event_memory_v1.events import Event,build_memory

def policy_metrics(p,q):
    """Exact categorical KL on unchanged legal support, unlike sampled PPO KL."""
    mask=p>0
    term=np.zeros_like(p);term[mask]=p[mask]*(np.log(p[mask])-np.log(q[mask]))
    kl=term.sum(-1);tv=.5*np.abs(p-q).sum(-1)
    return dict(exact_categorical_kl_mean=float(kl.mean()),exact_categorical_kl_p95=float(np.quantile(kl,.95)),
        total_variation_mean=float(tv.mean()),total_variation_p95=float(np.quantile(tv,.95)),
        argmax_changed_fraction=float(np.mean(p.argmax(-1)!=q.argmax(-1))),
        max_action_probability_change=float(np.abs(p-q).max()))

def main():
    torch.set_num_threads(1)
    suite,audit=sources()['3e-4'];bank=[];targets=[];meta=[];pins=[]
    for j in suite['jobs']:
        a=audit[j['run_index']];p=Path(a['hdf5']['path']);assert ref(p)==a['hdf5']
        for key in ('manifest','seal'):
            assert ref(a[key]['path'])==a[key]
        pins.append(ref(p))
        with h5py.File(p) as h:
            states=h['states'][...];ev=json.loads(h.attrs['evaluation_json'])
            memory=build_memory(_read_table(h,'material_events',Event))
            target=build_composite_targets(memory,wait_deltas=h['transitions/wait_delta_seconds'][...],
                ore_tons=ev['ore_tons'],waste_tons=ev['waste_tons'],total_wait_seconds=ev['total_wait_seconds'])
            ds=[e for e in memory.events if e.kind=='DECISION']
            assert len(ds)==len(target.mc_returns)==len(h['transitions/action'])
            for w in range(4):
                ids=[i for i,d in enumerate(ds) if w*7200<=d.time<(w+1)*7200]
                for quantile in (.25,.75):
                    i=ids[min(len(ids)-1,int(len(ids)*quantile))]
                    bank.append(states[i]);targets.append(target.mc_returns[i]);meta.append(dict(
                        run_index=j['run_index'],window=w,decision_sequence=i,time=ds[i].time,rank_quantile=quantile))
    raw=np.stack(bank);target=np.array(targets);assert len(raw)==800
    arm=PROJECT/'runs/business-graph-summary-formal-seed1-20261118-001/arms/s01_BG128'
    frozen=read(OUT.parent/'trained_analysis_001/training_reward_convergence_gradient.json')
    batch_rows=[]
    for r in frozen['batches']:
        p=arm/f"updates/{r['phase']:04}/training_summary.json";u=read(p)['update']
        assert 'minibatch_kl' not in u and set(u['mean'])=={'actor_grad_norm','actor_loss','critic_grad_norm','critic_mc_mse'}
        assert u['metrics']['approx_kl']==r['post_update_approx_kl']
        batch_rows.append({k:r[k] for k in ('phase','behavior_checkpoint_update','updated_checkpoint','transitions',
            'cost_mean','cost_se','complete_task_rate','critic_before_mse','critic_before_ev','mc_target_variance',
            'critic_after_in_sample_mse','critic_after_in_sample_ev','post_update_approx_kl','post_update_clip_fraction',
            'mean_recorded_actor_grad_norm_preclip','mean_recorded_critic_grad_norm_preclip')})
        pins.append(ref(p))
    probabilities={};values={};checkpoint_pins=[];rows=[];chunks=None;critic_rows=[]
    for n in range(10,17):
        cp=arm/f'updates/{n:04}/checkpoint.pt'
        model=JointCostPPO.from_checkpoint(cp,expected_config=suite['learner_config'],restore_global_rng=False)
        assert model.update_index==n and model.config['ppo']['actor_lr']==3e-4
        model.actor.eval();model.critic.eval();before=training_state_fingerprint(model)
        checkpoint_pins.append(ref(cp))
        if chunks is None:
            chunks=[(torch.from_numpy(raw[i:i+32].copy()),pack_summary_prefixes(raw[i:i+32],feature_config=model.actor.feature_config))
                    for i in range(0,len(raw),32)]
        pp=[];vv=[]
        with ReadOnlyTrainingGuard(model),torch.no_grad():
            for x,prefix in chunks:
                pp.append(model.actor(x,prefix).softmax(-1).numpy().astype(float))
                vv.append(model.critic(x,prefix).numpy().astype(float))
        assert before==training_state_fingerprint(model)
        probabilities[n]=np.concatenate(pp);values[n]=np.concatenate(vv);v=values[n]
        critic_rows.append(dict(checkpoint=n,mc_target_variance=float(np.var(target)),mse=float(np.mean((target-v)**2)),
            explained_variance=float(1-np.var(target-v)/np.var(target)),scope='800 sampled old-u13-trajectory return labels; off-policy value fit for other checkpoints'))
        for i,m in enumerate(meta):
            rows.append(dict(**m,checkpoint=n,mc_return=float(target[i]),critic_value=float(v[i]),
                **{f'p{a}':float(probabilities[n][i,a]) for a in range(4)}))
        print(f'fixed-state checkpoint u{n} complete',flush=True)
    changes=[]
    for before,after in [(n,n+1) for n in range(10,16)]+[(13,16)]:
        for window in [None,0,1,2,3]:
            idx=np.array([window is None or m['window']==window for m in meta])
            changes.append(dict(before=before,after=after,window='all' if window is None else window,
                observations=int(idx.sum()),**policy_metrics(probabilities[before][idx],probabilities[after][idx])))
    save('late_updates.json',dict(schema='bg128-frozen-late-checkpoint-state-bank-v1',
        state_bank=dict(source_checkpoint_behavior=13,source_lr='3e-4',shifts=100,states=len(raw),
            selection='For every old independent-evaluation shift and each window, decision ranks 25% and 75%; no outcome filter.',
            independent_performance_of_other_checkpoints=False),
        policy_changes=changes,critic_fixed_return_diagnostics=critic_rows,original_training_batches=batch_rows,
        historical_metric_availability=dict(minibatch_KL=False,minibatch_clip_fraction=False,
            epoch_KL=False,post_update_full_batch_sampled_approx_KL=True,
            post_update_full_batch_clip_fraction=True,mean_preclip_gradient_norm=True,
            note='The original 3e-4 summaries retain aggregate gradients and final full-batch metrics. Missing minibatch/epoch metrics are not reconstructed as historical observations.'),
        independent_performance_status='u13 only has the selected 100-shift independent evaluation; later checkpoints were not evaluated on these seeds.',
        limitations=['Exact categorical state-bank KL differs from sampled on-policy PPO approx KL.',
            'Return labels are realized under u13; Critic EV at other checkpoints is an off-policy diagnostic, not own-policy accuracy.',
            'Training batches use different seeds and changing return distributions; their cost changes do not prove an update caused regression.',
            'Policy movement and aggregate KL/clip do not establish a change in independent task performance.',
            'Later checkpoint independent performance and true minibatch trajectories remain open questions.'],
        physical_shifts=0,optimizer_updates=0,gradient_updates=0,model_states_unchanged=True,
        checkpoint_pins=checkpoint_pins,source_pins=pins+[ref(Path(__file__)),ref(OUT.parent/'trained_analysis_001/training_reward_convergence_gradient.json')],
        protected_ppo_sha=sha(CORE)))
    csv_save('late_state_outputs.csv',rows);csv_save('late_policy_changes.csv',changes)
    csv_save('original_training_batches.csv',batch_rows)
    assert sha(CORE)==CORE_SHA
    print(json.dumps(dict(states=800,checkpoints=7,physical_shifts=0,optimizer_updates=0)))

if __name__=='__main__':main()
