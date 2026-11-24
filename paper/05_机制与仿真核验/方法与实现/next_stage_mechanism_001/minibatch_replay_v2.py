"""Isolated original-update reconstruction; never save models or touch ledgers.

Only detach/no-grad observers are installed on THIS in-memory instance.
The original numerical learner source is not edited. A diagnostic is accepted
only after matching the original sealed endpoint (models, Adam, RNG, targets).
"""
from collections import defaultdict,deque
import time
import h5py
import numpy as np
import torch
from common import *
from business_graph_summary_v1.learner import (JointCostPPO,_load_checkpoint,
    _episode_key,_episode_fingerprint)
from business_graph_summary_storage_v1.episode import read_episode_hdf5

def equal_tree(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a,b)
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal_tree(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(equal_tree(x,y) for x,y in zip(a,b))
    return a==b

def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def main():
    cfgpath=PLAN/'minibatch_replay_config_v2.json';cfg=read(cfgpath)
    assert cfg['replay_implementation']==ref(Path(__file__))
    assert cfg['protected_core']==ref(CORE) and sha(CORE)==CORE_SHA
    torch.set_num_threads(cfg['cpu_threads']);out=Path(cfg['python_output']);out.mkdir(parents=True,exist_ok=True)
    assert not list(out.iterdir()),'fresh offline diagnostic output required'
    write_json(out/'started.json',dict(config=ref(cfgpath),new_physical_shifts=0,new_recording_id=None))
    all_rows=[];all_epochs=[];accepted=[]
    for item in cfg['updates']:
        n=item['update'];start=time.monotonic()
        for key in ('before','after','summary','actor'):assert ref(item[key]['path'])==item[key]
        saved,_=_load_checkpoint(item['after']['path']);summary=read(item['summary']['path'])
        batch=summary['transaction']['batch_id'];receipt=saved['consumed_batches'][batch]
        sources_by_key={r['episode_key']:r for r in receipt['sources']};episodes=[];pins=[]
        for source in summary['transaction']['sources']:
            assert ref(source['path'])==source
            path=Path(str(source['path']).removesuffix('.manifest.json'))
            def validate(e):
                r=sources_by_key[_episode_key(e)]
                assert r['identity']==e['identity'] and r['episode_sha256']==_episode_fingerprint(e)
            episodes.append(read_episode_hdf5(path,validator=validate));pins.append(ref(path))
        assert [e['identity']['run_index'] for e in episodes]==summary['transaction']['run_indices']
        learner=JointCostPPO.from_checkpoint(item['before']['path'],restore_global_rng=False)
        learner.bind_actor_snapshot(item['actor']['path'])
        assert learner.config['ppo']['actor_lr']==3e-4 and learner.update_index==n-1
        # Reject a numerical-environment mismatch BEFORE any optimizer step.
        # This is a deterministic read of original frozen labels/values; the
        # unmodified update below will revalidate and refreeze them itself.
        learner._freeze_targets(episodes)
        preflight=dict(frozen_targets=equal_tree(learner.last_targets,saved['last_targets']),
                       target_hash=learner.last_target_hash==saved['last_target_hash'])
        write_json(out/f'preflight_{n:04}.json',dict(checks=preflight,cpu_threads=torch.get_num_threads()))
        assert all(preflight.values()),preflight
        total=sum(len(e['actions']) for e in episodes);per_epoch=math_ceil(total,32)
        rows=[];epochs=[];ctx={'active':None,'queue':deque(),'chunks':[],'step':0,'critic':[]}
        original_distribution=learner.actor.distribution
        def observed_distribution(states,prefixes=None):
            distribution=original_distribution(states,prefixes)
            if ctx['active'] is not None:
                ids=ctx['queue'].popleft();assert len(ids)==len(states)
                actions,logs=ctx['active']
                with torch.no_grad():
                    delta=distribution.log_prob(actions[ids])-logs[ids];ratio=delta.exp()
                    ctx['chunks'].append(dict(kl=((ratio-1)-delta).detach().numpy().astype(float),
                        clipped=((ratio-1).abs()>.2).detach().numpy(),delta=delta.detach().numpy().astype(float)))
            return distribution
        learner.actor.distribution=observed_distribution
        original_actor_grad=learner._actor_grad;original_critic_grad=learner._critic_grad
        def groups(mapping,ids):
            result=defaultdict(list)
            for i in ids.tolist():result[mapping[i][0]].append(i)
            return deque(result.values())
        def actor_grad(e,mapping,ids,actions,logs,advantages):
            ctx['active']=(actions,logs);ctx['queue']=groups(mapping,ids);ctx['chunks']=[]
            try: result=original_actor_grad(e,mapping,ids,actions,logs,advantages)
            finally:ctx['active']=None
            assert not ctx['queue'];ctx['step']+=1
            kl=np.concatenate([r['kl'] for r in ctx['chunks']]);clip=np.concatenate([r['clipped'] for r in ctx['chunks']])
            delta=np.concatenate([r['delta'] for r in ctx['chunks']]);assert len(kl)==len(ids)
            rows.append(dict(update=n,epoch=(ctx['step']-1)//per_epoch+1,step=ctx['step'],minibatch_size=len(ids),
                sampled_reverse_approx_kl_before_actor_step=float(kl.mean()),
                clip_fraction_before_actor_step=float(clip.mean()),
                sampled_abs_log_ratio_max_before_actor_step=float(np.abs(delta).max())))
            return result
        learner._actor_grad=actor_grad
        def critic_hook(module,args,output):
            if ctx['critic_active'] is not None:
                ids=ctx['queue'].popleft()
                assert len(ids)==len(output)
                ctx['critic'].append((output.detach().numpy().astype(float),ctx['critic_active'][ids].detach().numpy().astype(float)))
        ctx['critic_active']=None
        hook=learner.critic.register_forward_hook(critic_hook)
        def critic_grad(e,mapping,ids,target):
            ctx['critic_active']=target;ctx['queue']=groups(mapping,ids);ctx['critic']=[]
            try: result=original_critic_grad(e,mapping,ids,target)
            finally:ctx['critic_active']=None
            assert not ctx['queue'];v=np.concatenate([v for v,t in ctx['critic']]);t=np.concatenate([t for v,t in ctx['critic']])
            variance=float(np.var(t));rows[-1].update(critic_mse_before_critic_step=float(np.mean((v-t)**2)),
                critic_target_variance=variance,critic_ev_before_critic_step=None if variance<=1e-12 else float(1-np.var(t-v)/variance))
            return result
        learner._critic_grad=critic_grad
        def progress(p):
            if p['status']=='epoch_end':
                m=learner._metrics(episodes)
                epochs.append(dict(update=n,epoch=p['completed_epochs'],**m))
                print(json.dumps(dict(update=n,epoch=p['completed_epochs'],elapsed=round(time.monotonic()-start,1),
                    full_batch_KL=m['approx_kl'],clip=m['clip_fraction'],EV=m['explained_variance'])),flush=True)
        learner.progress_callback=progress
        result=learner.update(episodes,batch_id=batch);hook.remove()
        checks=dict(actor=equal_tree(learner.actor.state_dict(),saved['actor']),
            critic=equal_tree(learner.critic.state_dict(),saved['critic']),
            actor_adam=equal_tree(learner.actor_optimizer.state_dict(),saved['actor_optimizer']),
            critic_adam=equal_tree(learner.critic_optimizer.state_dict(),saved['critic_optimizer']),
            private_shuffle_rng=torch.equal(learner.generator.get_state(),saved['generator']),
            frozen_targets=equal_tree(learner.last_targets,saved['last_targets']),
            target_hash=learner.last_target_hash==saved['last_target_hash'],
            update_receipt=result==summary['update'],
            source_checkpoint_files_unchanged=all(ref(item[k]['path'])==item[k] for k in ('before','after','summary','actor')),
            protected_core_unchanged=sha(CORE)==CORE_SHA)
        status='PASS_EXACT_RECONSTRUCTION' if all(checks.values()) else 'REJECT_DIVERGED_RECONSTRUCTION'
        record=dict(update=n,status=status,checks=checks,offline_actor_optimizer_steps=result['actor_optimizer_steps'],
            offline_critic_optimizer_steps=result['critic_optimizer_steps'],elapsed_seconds=time.monotonic()-start,
            source_pins=pins,observed_minibatches=rows,epoch_full_batch_metrics=epochs)
        write_json(out/f'update_{n:04}.json',record)
        assert all(checks.values()),checks
        all_rows.extend(rows);all_epochs.extend(epochs);accepted.append({k:v for k,v in record.items() if k not in ('observed_minibatches','epoch_full_batch_metrics','source_pins')})
    write_json(out/'complete.json',dict(status='PASS_EXACT_RECONSTRUCTION',config=ref(cfgpath),updates=accepted,
        new_physical_shifts=0,official_training_updates=0,official_checkpoint_writes=0,ledger_writes=0,
        offline_actor_optimizer_steps=sum(r['offline_actor_optimizer_steps'] for r in accepted),
        offline_critic_optimizer_steps=sum(r['offline_critic_optimizer_steps'] for r in accepted),
        interpretation='Numerically reconstructed original update path, not an original-time log or a new trained model.',
        metric_timing='Minibatch Actor KL/clip before its Adam step against frozen behavior logs; Critic EV before its Adam step. Epoch diagnostics evaluate the full original batch at fixed epoch-end weights.'))
    # These detailed rows are explicitly necessary for the requested update
    # diagnosis; no transition/RPC/full observation logging is enabled.
    for name,data in [('minibatches.csv',all_rows),('epochs.csv',all_epochs)]:
        with (out/name).open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    print(json.dumps(dict(status='PASS_EXACT_RECONSTRUCTION',updates=len(accepted),minibatches=len(all_rows))),flush=True)

def math_ceil(n,d):return (n+d-1)//d

if __name__=='__main__':main()
