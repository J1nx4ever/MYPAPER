"""Read sealed BG128 training evidence; diagnose gradients without stepping."""
from pathlib import Path
import argparse, hashlib, json, math, sqlite3, sys, time
import numpy as np
import h5py
ROOT=Path('/home/jinx/Faith/experiments/lagrangian_ppo_v1')
sys.path.insert(0,str(ROOT))
from business_graph_summary_campaign_v1.plan_io import load_suite, reference
from business_graph_summary_campaign_v1.gate import check_sources, PROTECTED_SHA
from business_graph_summary_campaign_v1.budget import connection, _payload, _state
from business_graph_summary_storage_v1.episode import _read_table
from grade_event_memory_v1.events import Event, build_memory
from business_graph_summary_v1.targets import build_composite_targets, grade_cost, cost_gae
from python.hdf5_arrays import read_array_bundle

CID='business-graph-summary-formal-seed1-20261118-001'
RUN=ROOT/'runs'/CID;ARM=RUN/'arms/s01_BG128'
PLAN=ROOT/'change_plans/token_attention_001/business_graph_summary_009/trained_analysis_001'
REPORT=ROOT/'reports/token_attention_001/business_graph_summary_009/trained_analysis_001'
NAMES=('grade','production','wait','old_value_baseline')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def checked(ref):assert reference(ref['path'])==ref;return Path(ref['path'])
def vector(v):return np.asarray(v,dtype=np.float64)
def gae(c,v):
    n=len(c)
    return vector(cost_gae(c,v,terminated=[False]*(n-1)+[True],truncated=[False]*n,gae_lambda=.95).advantages)
def cost_suffix(c):
    return gae(c,np.zeros(len(c)+1))
def setup():
    suite,_=load_suite(ROOT/'change_plans/token_attention_001/business_graph_summary_009/formal_training_001/prepared_001/suite.json')
    check_sources(suite['source_pins'])
    with connection(ROOT/'runs/business-graph-summary-budget-v1.sqlite',readonly=True) as con:
        entry=_payload(con.execute('select payload,checksum from campaigns where id=?',(CID,)).fetchone())
        state=_state(con,suite,entry)
    assert len(state['jobs'])==512 and len(state['campaigns'][CID]['commits'])==16
    return suite,state['jobs']
def load_batch(n,slots,*,states=False):
    folder=ARM/f'updates/{n:04}';summary=json.loads((folder/'training_summary.json').read_bytes())
    arr,meta=read_array_bundle(folder/'training_targets.h5')
    eps=[];parts={k:[] for k in NAMES};cursor=0;window_costs=[]
    for rid in summary['transaction']['run_indices']:
        receipt=slots[str(rid)]['receipt'];p=checked(receipt['references']['episode'])
        manifest=json.loads(checked(receipt['references']['episode_manifest']).read_bytes())
        seal=json.loads(checked(receipt['references']['episode_seal']).read_bytes())
        assert seal['hdf5_sha256']==manifest['hdf5_sha256']==sha(p)
        assert seal['manifest_sha256']==sha(str(p)+'.manifest.json')
        with h5py.File(p,'r') as h:
            identity=json.loads(h.attrs['identity_json']);assert identity==receipt['identity']
            memory=build_memory(_read_table(h,'material_events',Event))
            ev=json.loads(h.attrs['evaluation_json'])
            wd=h['transitions/wait_delta_seconds'][...]
            targets=build_composite_targets(memory,wait_deltas=wd,ore_tons=ev['ore_tons'],waste_tons=ev['waste_tons'],total_wait_seconds=ev['total_wait_seconds'])
            count=len(targets.step_costs);old=arr['old_values'][cursor:cursor+count]
            values=np.r_[old,0.]
            assert np.array_equal(arr['next_values'][cursor:cursor+count],values[1:])
            assert np.allclose(arr['costs'][cursor:cursor+count],targets.step_costs,rtol=0,atol=1e-14)
            assert np.allclose(arr['mc_returns'][cursor:cursor+count],targets.mc_returns,rtol=0,atol=1e-13)
            for key,costs in [('grade',targets.grade_costs),('production',targets.production_costs),('wait',targets.wait_costs)]:parts[key].extend(cost_suffix(costs))
            parts['old_value_baseline'].extend(gae(np.zeros(count),values))
            row=dict(run_index=rid,identity=identity,targets=targets,evaluation=ev,memory=memory,wait_deltas=wd,count=count,offset=cursor)
            if states:row.update(states=h['states'][...],actions=h['transitions/action'][...],old_log_probs=h['transitions/old_log_prob'][...])
            eps.append(row);cursor+=count
            window_costs.extend(grade_cost(s.grade if s.grade_valid else None,empty=not s.grade_valid) for s in memory.settlements)
    assert cursor==len(arr['costs'])
    parts={k:vector(v) for k,v in parts.items()};rawsum=sum(parts.values());std=arr['advantages'].std()+1e-8
    normalized={k:(v-v.mean())/std for k,v in parts.items()}
    assert np.allclose(rawsum,arr['advantages'],rtol=1e-11,atol=1e-11)
    assert np.allclose(sum(normalized.values()),arr['actor_advantages'],rtol=1e-10,atol=1e-10)
    closure=dict(raw_gae_max_abs_error=float(np.max(np.abs(rawsum-arr['advantages']))),normalized_advantage_max_abs_error=float(np.max(np.abs(sum(normalized.values())-arr['actor_advantages']))))
    return summary,arr,eps,parts,normalized,window_costs,closure

def aggregate(suite,slots):
    from business_graph_summary_v1.learner import _load_checkpoint
    previous,_=_load_checkpoint(ARM/'initial.pt')
    rows=[];all_windows=[]
    for n in range(1,17):
        summary,arr,eps,parts,normalized,wc,closure=load_batch(n,slots)
        u=summary['update'];cs={k:np.array([sum(getattr(e['targets'],k+'_costs')) for e in eps]) for k in NAMES[:3]}
        cost=sum(cs.values());assert np.isclose(cost.mean(),u['metrics']['combined_cost_mean'])
        current,_=_load_checkpoint(ARM/f'updates/{n:04}/checkpoint.pt')
        deltas={}
        for role in ('actor','critic'):
            floats=[(current[role][k].double()-v.double()).reshape(-1) for k,v in previous[role].items() if v.is_floating_point()]
            d=np.concatenate([t.numpy() for t in floats]);base=np.concatenate([v.double().reshape(-1).numpy() for v in previous[role].values() if v.is_floating_point()])
            deltas[role]=dict(net_parameter_delta_l2=float(np.linalg.norm(d)),relative_to_before_parameter_l2=float(np.linalg.norm(d)/np.linalg.norm(base)),max_abs_parameter_delta=float(np.max(np.abs(d))))
        previous=current
        grades=[s.grade for e in eps for s in e['memory'].settlements if s.grade_valid]
        comp={k:dict(total_cost=float(v.sum()),mean_cost_per_shift=float(v.mean()),fraction_of_batch_cost=float(v.sum()/cost.sum()),positive_shift_count=int(np.count_nonzero(v>1e-12)),direct_gae_nonzero_fraction=float(np.mean(np.abs(parts[k])>1e-12)),direct_raw_gae_l2=float(np.linalg.norm(parts[k])),joint_standardized_component_l2=float(np.linalg.norm(normalized[k]))) for k,v in cs.items()}
        step_stats={}
        for key in NAMES[:3]:
            v=np.concatenate([vector(getattr(e['targets'],key+'_costs')) for e in eps])
            step_stats[key]=dict(positive_transition_count=int(np.count_nonzero(v>1e-12)),positive_transition_fraction=float(np.mean(v>1e-12)),max_step_cost=float(v.max()))
        joint_compliant=[];task=[];ore=[];waste=[]
        for e in eps:
            win_ok=all(s.grade_valid and .26-1e-12<=s.grade<=.28+1e-12 for s in e['memory'].settlements)
            ev=e['evaluation'];om=ev['ore_tons']>=16000;wm=ev['waste_tons']>=4000
            joint_compliant.append(win_ok);ore.append(om);waste.append(wm);task.append(win_ok and om and wm)
        target=arr['mc_returns'];old=arr['old_values'];m=u['metrics']
        rows.append(dict(phase=n,behavior_checkpoint_update=n-1,updated_checkpoint=n,run_indices=summary['transaction']['run_indices'],transitions=len(target),cost_mean=float(cost.mean()),cost_sd=float(cost.std(ddof=1)),cost_se=float(cost.std(ddof=1)/math.sqrt(32)),derived_reward_mean=float(-cost.mean()),cost_components=comp,step_cost_sparsity=step_stats,zero_grade_cost_window_count=int(sum(v<=1e-12 for v in wc)),positive_grade_cost_window_count=int(sum(v>1e-12 for v in wc)),grade_cost_saturated_window_count=int(sum(v>=2-1e-12 for v in wc)),window_count=128,grade_rmse=float(np.sqrt(np.mean((vector(grades)-.27)**2))),joint_four_window_compliance_rate=float(np.mean(joint_compliant)),ore_target_met_rate=float(np.mean(ore)),waste_target_met_rate=float(np.mean(waste)),complete_task_rate=float(np.mean(task)),fleet_wait_seconds_mean=float(np.mean([e['evaluation']['total_wait_seconds'] for e in eps])),critic_before_mse=float(np.mean((target-old)**2)),critic_before_ev=float(1-np.var(target-old)/np.var(target)),critic_after_in_sample_mse=m['critic_mc_mse'],critic_after_in_sample_ev=m['explained_variance'],post_update_entropy=m['entropy'],post_update_approx_kl=m['approx_kl'],post_update_clip_fraction=m['clip_fraction'],mean_recorded_actor_grad_norm_preclip=u['mean']['actor_grad_norm'],mean_recorded_critic_grad_norm_preclip=u['mean']['critic_grad_norm'],actor_loss_mean=u['mean']['actor_loss'],critic_minibatch_mse_mean=u['mean']['critic_mc_mse'],optimizer_steps_per_actor=u['actor_optimizer_steps'],optimizer_steps_per_critic=u['critic_optimizer_steps'],actual_net_parameter_changes=deltas,joint_raw_advantage_mean=float(arr['advantages'].mean()),joint_raw_advantage_std=float(arr['advantages'].std()),mc_target_variance=float(np.var(target)),target_closure=closure,training_summary=reference(ARM/f'updates/{n:04}/training_summary.json'),training_targets=reference(ARM/f'updates/{n:04}/training_targets.h5')))
        all_windows.extend(wc)
        print('AGGREGATE phase='+str(n),flush=True)
    return rows

def grad_diagnostic(n,suite,slots):
    import torch
    from business_graph_summary_v1.learner import JointCostPPO
    from business_graph_summary_v1.models import digest
    from baseline1_v1.evaluation import ReadOnlyTrainingGuard, training_state_fingerprint
    checkpoint=ARM/'initial.pt' if n==1 else ARM/f'updates/{n-1:04}/checkpoint.pt'
    model=JointCostPPO.from_checkpoint(checkpoint,expected_config=suite['arms'][0]['learner_config'],expected_sha256=sha(checkpoint),restore_global_rng=False)
    summary,arr,eps,parts,normalized,wc,closure=load_batch(n,slots,states=True)
    assert model.policy_version==summary['update']['policy_before']
    named=list(model.actor.named_parameters());params=[p for _,p in named];sizes=[p.numel() for p in params];length=sum(sizes)
    vectors={k:np.zeros(length,dtype=np.float64) for k in (*NAMES,'entropy','joint_policy','joint_with_entropy')}
    N=len(arr['costs']);maxlog=0.;maxratio=0.;original_loss=0.;ratio_clip_count=0
    before=training_state_fingerprint(model)
    with ReadOnlyTrainingGuard(model):
        for ei,e in enumerate(eps):
            states=e['states'][:-1]
            for start in range(0,e['count'],16):
                ids=list(range(start,min(start+16,e['count'])));gids=np.array(ids)+e['offset']
                prefix=model._prefix(states,ids)
                dist=model.actor.distribution(torch.tensor(states[ids].copy()),prefix)
                actions=torch.tensor(e['actions'][ids],dtype=torch.int64)
                logs=dist.log_prob(actions);oldlogs=torch.tensor(e['old_log_probs'][ids],dtype=torch.float32)
                ratio=(logs-oldlogs).exp();maxlog=max(maxlog,float((logs-oldlogs).detach().abs().max()));maxratio=max(maxratio,float((ratio-1).detach().abs().max()))
                ratio_clip_count+=int(((ratio-1).abs()>.2).sum())
                losses={k:(ratio*torch.tensor(normalized[k][gids],dtype=torch.float32)).sum()/N for k in NAMES}
                losses['entropy']=-.01*dist.entropy().sum()/N
                adv=torch.tensor(arr['actor_advantages'][gids],dtype=torch.float32)
                losses['joint_policy']=torch.maximum(ratio*adv,ratio.clamp(.8,1.2)*adv).sum()/N
                losses['joint_with_entropy']=losses['joint_policy']+losses['entropy']
                original_loss+=float(losses['joint_with_entropy'].detach())
                for li,(key,loss) in enumerate(losses.items()):
                    grads=torch.autograd.grad(loss,params,retain_graph=li<len(losses)-1,allow_unused=True)
                    vectors[key]+=np.concatenate([np.zeros(sz) if g is None else g.detach().reshape(-1).double().numpy() for sz,g in zip(sizes,grads)])
            if ei%8==7:print('GRADIENT phase='+str(n)+' episodes='+str(ei+1)+'/32',flush=True)
    assert training_state_fingerprint(model)==before
    joint=vectors['joint_policy'];combined=vectors['joint_with_entropy'];summed=sum(vectors[k] for k in NAMES)
    rel=lambda x,y:float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-30))
    closure.update(gradient_direct_plus_baseline_relative_error=rel(summed,joint),gradient_plus_entropy_relative_error=rel(summed+vectors['entropy'],combined),ratio_max_abs_deviation_from_one=maxratio,old_log_probability_max_abs_error=maxlog,clipped_transition_count=ratio_clip_count)
    assert closure['gradient_direct_plus_baseline_relative_error']<2e-5 and closure['gradient_plus_entropy_relative_error']<2e-5 and ratio_clip_count==0
    def cosine(a,b):
        den=np.linalg.norm(a)*np.linalg.norm(b)
        return None if den<1e-28 else float(np.dot(a,b)/den)
    measures={}
    groups=[]
    for name,p in named:
        if name.startswith('state_encoder.graph_layers.'):group='.'.join(name.split('.')[:3])
        elif name.startswith('state_encoder.token_encoder.'):group='state_encoder.token_encoder'
        else:group=name.split('.')[0]
        groups.extend([group]*p.numel())
    groups=np.array(groups);group_names=sorted(set(groups))
    for key,v in vectors.items():
        den=float(np.dot(joint,joint))
        measures[key]=dict(l2_norm=float(np.linalg.norm(v)),dot_with_joint_policy=float(np.dot(v,joint)),cosine_with_joint_policy=cosine(v,joint),projection_on_joint_policy=float(np.dot(v,joint)/den) if den>0 else None,cosine_with_actual_joint_loss=cosine(v,combined),module_gradient_squared_norm_fraction={g:float(np.dot(v[groups==g],v[groups==g])/np.dot(v,v)) if np.dot(v,v)>0 else None for g in group_names})
    matrix={a:{b:cosine(vectors[a],vectors[b]) for b in vectors} for a in vectors}
    rawdirect=sum(vectors[k] for k in NAMES[:3])
    result=dict(phase=n,behavior_checkpoint_update=n-1,checkpoint=reference(checkpoint),policy_digest=model.policy_version,episodes=32,transitions=N,parameter_count=length,method='full_original_32_shift_batch_at_frozen_pre_update_parameters',original_joint_loss_at_frozen_policy=original_loss,closure=closure,gradients=measures,pairwise_cosine=matrix,loss_weighted_entropy_coefficient=.01,cancellation=dict(norm_joint_policy_over_sum_component_norms=float(np.linalg.norm(joint)/sum(np.linalg.norm(vectors[k]) for k in NAMES)),norm_direct_cost_sum=float(np.linalg.norm(rawdirect)),cosine_direct_cost_sum_with_baseline=cosine(rawdirect,vectors['old_value_baseline'])),model_training_state_unchanged=True,optimizer_steps=0,physical_shifts=0,actual_adam_cost_component_attribution_available=False)
    dump(REPORT/f'gradient_phase_{n:04}.json',result)
    return result

def pillow_plots(report):
    """Deterministic scientific plots when optional Matplotlib is unavailable."""
    from PIL import Image, ImageDraw, ImageFont
    font='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    def f(size):return ImageFont.truetype(font,size)
    colors=('#3465a4','#d97706','#248b60','#9b4ab2','#a83232','#555555')
    def panel(draw,box,title,series,*,log=False,ylim=None,xlabel='Behavior checkpoint update'):
        left,top,right,bottom=box;l=left+80;t=top+62;r=right-28;b=bottom-90
        draw.text((left+18,top+10),title,font=f(21),fill='black')
        allx=[x for _,xs,ys in series for x in xs];ally=[y for _,xs,ys in series for y in ys if y is not None and (not log or y>0)]
        xmin=min(allx);xmax=max(allx);ymin,ymax=ylim or ((min(ally),max(ally)) if log else (min(0,min(ally)),max(ally)*1.08))
        if log:ymin=10**math.floor(math.log10(ymin));ymax=10**math.ceil(math.log10(ymax))
        if ymax<=ymin:ymax=ymin+1
        def xp(x):return l+(x-xmin)/max(xmax-xmin,1)*(r-l)
        def yp(y):
            q=(math.log10(y)-math.log10(ymin))/(math.log10(ymax)-math.log10(ymin)) if log else (y-ymin)/(ymax-ymin)
            return b-q*(b-t)
        for i in range(5):
            frac=i/4;v=10**(math.log10(ymin)+frac*(math.log10(ymax)-math.log10(ymin))) if log else ymin+frac*(ymax-ymin);y=yp(v)
            draw.line((l,y,r,y),fill='#dddddd',width=1);label=f'{v:.3g}';draw.text((l-12-draw.textlength(label,font=f(15)),y-9),label,font=f(15),fill='#333333')
        draw.line((l,t,l,b,r,b),fill='black',width=2)
        ticks=sorted(set([xmin,xmax]+[v for v in allx if v%3==0]))
        for v in ticks:draw.text((xp(v)-8,b+8),str(v),font=f(15),fill='#333333')
        draw.text(((l+r)/2-115,b+34),xlabel,font=f(16),fill='#333333')
        for si,(label,xs,ys) in enumerate(series):
            color=colors[si%len(colors)];points=[(xp(x),yp(y)) for x,y in zip(xs,ys) if y is not None and (not log or y>0)]
            if len(points)>1:draw.line(points,fill=color,width=3)
            for px,py in points:draw.ellipse((px-4,py-4,px+4,py+4),fill=color)
            ly=top+36+si*19
            draw.line((r-260,ly+7,r-242,ly+7),fill=color,width=3);draw.text((r-236,ly-1),label,font=f(13),fill=color)
    rows=report['batches'];x=[r['behavior_checkpoint_update'] for r in rows]
    image=Image.new('RGB',(1800,1500),'white');d=ImageDraw.Draw(image)
    boxes=[(0,0,900,500),(900,0,1800,500),(0,500,900,1000),(900,500,1800,1000),(0,1000,900,1500),(900,1000,1800,1500)]
    panel(d,boxes[0],'Joint cost; actual behavior before each update',[('Mean cost',x,[r['cost_mean'] for r in rows]),('Negative training reward',x,[r['derived_reward_mean'] for r in rows])])
    panel(d,boxes[1],'Realized objective components; not gradient shares',[(k,x,[r['cost_components'][k]['mean_cost_per_shift'] for r in rows]) for k in NAMES[:3]])
    panel(d,boxes[2],'Grade center RMSE (fraction, lower is better)',[('Four-window grade RMSE',x,[r['grade_rmse'] for r in rows])])
    panel(d,boxes[3],'Critic MC MSE; changing on-policy targets',[('Before update / fresh batch',x,[r['critic_before_mse'] for r in rows]),('After update / same batch',x,[r['critic_after_in_sample_mse'] for r in rows])],log=True)
    panel(d,boxes[4],'Recorded mean preclip minibatch gradient norms',[('Actor',x,[r['mean_recorded_actor_grad_norm_preclip'] for r in rows]),('Critic',x,[r['mean_recorded_critic_grad_norm_preclip'] for r in rows]),('Clip threshold',x,[.5]*len(x))],log=True)
    panel(d,boxes[5],'Post-update full-batch PPO diagnostics',[('Approx KL',x,[r['post_update_approx_kl'] for r in rows]),('Clip fraction',x,[r['post_update_clip_fraction'] for r in rows])])
    image.save(REPORT/'training_reward_convergence.png',dpi=(180,180))
    ds=report.get('gradient_diagnostics',[])
    if ds:
        xx=[r['behavior_checkpoint_update'] for r in ds];keys=(*NAMES,'entropy','joint_policy')
        image=Image.new('RGB',(1800,650),'white');d=ImageDraw.Draw(image)
        panel(d,(0,0,900,650),'Frozen full-batch gradient norms; zero optimizer steps',[(k,xx,[r['gradients'][k]['l2_norm'] for r in ds]) for k in keys],log=True)
        panel(d,(900,0,1800,650),'Cosine with joint policy gradient; not Adam attribution',[(k,xx,[r['gradients'][k]['cosine_with_joint_policy'] for r in ds]) for k in (*NAMES,'entropy')],ylim=(-1.05,1.05))
        image.save(REPORT/'gradient_contributions.png',dpi=(180,180))

def plots(report):
    try:import matplotlib
    except ModuleNotFoundError:
        pillow_plots(report);return
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=report['batches'];x=np.array([r['behavior_checkpoint_update'] for r in rows])
    fig,ax=plt.subplots(3,2,figsize=(13,11),constrained_layout=True)
    a=ax[0,0];a.errorbar(x,[r['cost_mean'] for r in rows],yerr=[1.96*r['cost_se'] for r in rows],marker='o',capsize=3,label='Cost mean; descriptive +/-1.96 SE');a.axvline(13,color='black',ls='--',alpha=.5);a.set(ylabel='Joint cost (lower is better)',title='Behavior policy: update k sampled in batch k+1');a.legend(fontsize=8)
    a=ax[0,1];a.stackplot(x,*[[r['cost_components'][k]['mean_cost_per_shift'] for r in rows] for k in NAMES[:3]],labels=['Grade','Production shortfall','Fleet wait'],alpha=.8);a.set(ylabel='Mean joint cost',title='Cost components (not gradient shares)');a.legend(fontsize=8)
    a=ax[1,0];a.plot(x,[r['grade_rmse'] for r in rows],'o-',label='Grade RMSE');a.axhline(.004,color='gray',ls=':',label='Deadband half width (not RMSE constraint)');a.set(ylabel='Grade RMSE',title='Center accuracy and realized constraints');b=a.twinx();b.plot(x,[r['complete_task_rate'] for r in rows],'s--',color='darkorange',label='Complete task');b.set(ylabel='Complete-task rate',ylim=(0,1.05));a.legend(loc='upper right',fontsize=8);b.legend(loc='center right',fontsize=8)
    a=ax[1,1];a.semilogy(x,[r['critic_before_mse'] for r in rows],'o-',label='Pre-update Critic / next on-policy batch');a.semilogy(x,[r['critic_after_in_sample_mse'] for r in rows],'s--',label='Post-update Critic / same batch fit');a.set(ylabel='MC MSE (log scale)',title='Critic fit, with changing target distribution');a.legend(fontsize=8)
    a=ax[2,0];a.semilogy(x,[r['mean_recorded_actor_grad_norm_preclip'] for r in rows],'o-',label='Actor mean preclip grad norm');a.semilogy(x,[r['mean_recorded_critic_grad_norm_preclip'] for r in rows],'s-',label='Critic mean preclip grad norm');a.axhline(.5,color='black',ls='--',label='Clip threshold');a.set(ylabel='Gradient L2 (log scale)',title='Recorded minibatch totals; no component attribution');a.legend(fontsize=8)
    a=ax[2,1];a.plot(x,[r['post_update_approx_kl'] for r in rows],'o-',label='Approx KL');a.plot(x,[r['post_update_clip_fraction'] for r in rows],'s-',label='Clip fraction');a.set(ylabel='Update diagnostic',title='Full-batch replay after 4 PPO epochs');a.legend(fontsize=8)
    for a in ax.flat:a.set_xlabel('Behavior checkpoint update');a.grid(alpha=.2)
    fig.savefig(REPORT/'training_reward_convergence.png',dpi=180);plt.close(fig)
    ds=report.get('gradient_diagnostics',[])
    if ds:
        fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
        labels=['u'+str(d['behavior_checkpoint_update']) for d in ds];xx=np.arange(len(ds));keys=(*NAMES,'entropy','joint_policy')
        for i,k in enumerate(keys):axes[0].bar(xx+(i-2.5)*.12,[d['gradients'][k]['l2_norm'] for d in ds],width=.12,label=k)
        axes[0].set(xticks=xx,xticklabels=labels,yscale='log',ylabel='Gradient L2 (log scale)',title='Frozen pre-update full-batch diagnostic');axes[0].legend(fontsize=7)
        for k in (*NAMES,'entropy'):axes[1].plot(labels,[d['gradients'][k]['cosine_with_joint_policy'] if d['gradients'][k]['cosine_with_joint_policy'] is not None else np.nan for d in ds],'o-',label=k)
        axes[1].axhline(0,color='black',lw=.7);axes[1].set(ylim=(-1.05,1.05),ylabel='Cosine with joint policy gradient',title='Direction and cancellation; not Adam contribution');axes[1].legend(fontsize=7)
        for a in axes:a.grid(alpha=.2)
        fig.savefig(REPORT/'gradient_contributions.png',dpi=180);plt.close(fig)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--aggregate-only',action='store_true');args=parser.parse_args()
    REPORT.mkdir(parents=True,exist_ok=True);suite,slots=setup();core=ROOT.parents[1]/'python/episode_budget_ppo.py'
    before_core=sha(core);source_refs={k:v for k,v in suite['source_pins'].items()}
    rows=aggregate(suite,slots)
    result=dict(schema='bg128-training-reward-convergence-gradient-analysis-v1',status='AGGREGATE_COMPLETE' if args.aggregate_only else 'RUNNING_GRADIENT_DIAGNOSTICS',training_campaign=CID,analysis_script=reference(__file__),physical_training_shifts=512,updates=16,training_seed_count=1,batches=rows,gradient_diagnostics=[],selection_rule_unchanged=True,selected_checkpoint_update=13,protected_ppo_sha_before=before_core,physical_shifts_performed=0,optimizer_updates_performed=0,distinctions=dict(cost_share='ratio of scalar realized objective components; not gradients',direct_policy_gradient='diagnostic at original pre-update weights and original frozen joint standardized GAE',adam_actual_update='only checkpoint-to-checkpoint total net parameter delta observed; no per-cost attribution',recorded_gradient_norm='mean over all minibatches and epochs of clip_grad_norm_ returned preclip norm',critic_mse='before uses following on-policy batch; after is same-batch training fit',raw_rpc_reward='legacy RPC evidence is validated by training admission; analysis uses negative frozen rebuilt joint cost, not legacy raw RPC reward'))
    dump(REPORT/'training_reward_convergence_gradient.json',result);plots(result)
    if not args.aggregate_only:
        import torch
        torch.set_num_threads(2)
        for n in (1,7,11,14,16):
            result['gradient_diagnostics'].append(grad_diagnostic(n,suite,slots));dump(REPORT/'training_reward_convergence_gradient.json',result)
        result['status']='PASS_OFFLINE_ANALYSIS_NO_UPDATES'
        plots(result)
    assert sha(core)==before_core==PROTECTED_SHA
    assert all(sha(k)==v for k,v in source_refs.items())
    result['protected_ppo_sha_after']=sha(core);result['all_pinned_training_sources_unchanged']=True
    dump(REPORT/'training_reward_convergence_gradient.json',result)
    print(json.dumps(dict(status=result['status'],report=reference(REPORT/'training_reward_convergence_gradient.json'))),flush=True)
if __name__=='__main__':main()
