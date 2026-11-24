"""Expand observation-consistent A/B/C and four-combination readout probes.

Historical states are real; their branches are explicitly exogenous and have
not been reached by running the simulator. Only the frozen 3e-4/u13 is used.
"""
import heapq
import math
import h5py
import numpy as np
import torch
from common import *
from baseline1_v1.evaluation import ReadOnlyTrainingGuard, training_state_fingerprint
from business_graph_summary_v1.learner import JointCostPPO
from business_graph_summary_v1.models import pack_summary_prefixes
from business_graph_summary_v1.graph import build_graph

builder=module('branch_contract',PLAN.parent/'physical_intervention_001/build_scenarios.py')
prior=module('trained_readout',PLAN.parent/'trained_intervention_001/analyze.py')

def branch_a(base):
    trucks,loads,_,_=builder.blocks(base)
    for tid,t in enumerate(trucks):
        lid=builder.onehot(t[26:30])
        if t[0] or t[17] or builder.onehot(t[4:16])!=4 or lid is None: continue
        l=loads[lid];q=float(t[20])*193
        if not (l[1] and q>0 and t[24]==0 and t[25]==1 and t[23]>0): continue
        if not (builder.ids(l[38:47])==[tid] and l[15]==l[16]==1): continue
        if builder.ids(l[20:38]) or builder.onehot(t[45:61]) is not None: continue
        start=float(t[86])*30000;block=float(l[10])*500
        if block<=0 or abs(start-float(l[8])*30000)>.006: continue
        if int(start//block)!=int((start+q-1e-6)//block): continue
        if int(round(float(l[12])*60))-1!=int(start//block): continue
        if abs(float(t[23])-float(l[13]))>1e-7: continue
        raw=base.copy();ts,ls,_,_=builder.blocks(raw);t=ts[tid];l=ls[lid];grade=float(t[23])
        l[5]=float(l[5])-q/30000;l[6]=float(l[6])-q/30000;l[8]=float(l[8])+q/30000
        l[15:17]=0;l[38:47]=0;t[4:16]=0;t[15]=1
        t[17]=1;t[18]=1;t[19]=t[20];t[20]=0;t[21]=1;t[22]=grade;t[23:30]=0
        t[30:32]=(1,0);t[82:89]=0;builder.refresh_action_sidecar(raw)
        return raw,dict(truck=tid,load=lid,tons=q,grade=grade,
            boundary='completed known ore parcel before next travel-state entry; timestamp unchanged')
    return None,'no eligible single known-parcel loading service without queue'

def route_tail(roads,start,target,forbidden):
    heap=[(0.,start,[])];seen=set()
    while heap:
        cost,node,path=heapq.heappop(heap)
        if node==target: return path
        if node in seen: continue
        seen.add(node)
        for rid,(a,b) in enumerate(builder.ROAD_ENDPOINTS):
            if a==node and rid!=forbidden and roads[rid,4]==0:
                heapq.heappush(heap,(cost+float(roads[rid,2]),b,path+[rid]))
    return None

def branch_b(base):
    trucks,loads,_,roads=builder.blocks(base)
    for tid,t in enumerate(trucks):
        old=builder.onehot(t[26:30]);seg=builder.onehot(t[45:61]);q=float(t[20])*193
        if t[0] or t[17] or builder.onehot(t[4:16])!=2 or old is None or seg is None or q<=0: continue
        if not loads[old,1] or float(loads[old,17])*9<1-1e-6: continue
        if any(tid in builder.ids(l[20:47]) for l in loads): continue
        for new in range(4):
            if new==old or not loads[new,0] or not loads[new,1] or float(loads[new,7])*30000<q: continue
            tail=route_tail(roads,builder.ROAD_ENDPOINTS[seg][1],5+new,seg)
            if tail is None: continue
            route=[seg]+tail
            if len(set(route))!=len(route): continue
            raw=base.copy();ts,ls,_,_=builder.blocks(raw);t=ts[tid]
            t[26:30]=0;t[26+new]=1;t[63:79]=0
            for rank,r in enumerate(route,1):t[63+r]=rank/16
            t[62]=float(t[61])+sum(float(roads[r,2]) for r in tail)
            for lid,sign in ((old,-1),(new,1)):
                ls[lid,6]=float(ls[lid,6])+sign*q/30000
                ls[lid,7]=max(0.,float(ls[lid,5])-float(ls[lid,6]))
                ls[lid,17]=float(ls[lid,17])+sign/9
                if abs(float(ls[lid,17]))<1e-7:ls[lid,17]=0
            builder.refresh_action_sidecar(raw)
            return raw,dict(truck=tid,old_load=old,new_load=new,reserved_tons=q,route=route,
                boundary='unstarted reservation redirect; same current segment/position and material cursors')
    return None,'no eligible unstarted ore reservation with an open alternative road tail'

def context(raw):
    t,l,d,_=builder.blocks(raw)
    return np.array([raw[0],raw[3],raw[4],raw[5],d[0,5],
        sum(float(x[6]) for x in l if x[1]),sum(float(x[19]) for x in t if x[17] and x[18]),
        sum(float(x[16]) for x in l),sum(float(x[17]) for x in l),raw[1304]],dtype=float)

CONTEXT_NAMES=['shift_progress','window_remaining_7200','window_q_5000','window_m_5000','fifo_q_500',
    'ore_reservations_30000','loaded_ore_q_193','load_service_count','incoming_load_count_9','crusher_rate_2500']

def training_support():
    arm=PROJECT/'runs/business-graph-summary-formal-seed1-20261118-001/arms/s01_BG128'
    data=[];pins=[];count=0
    for n in range(1,17):
        p=arm/f'updates/{n:04}/training_summary.json';summary=read(p);pins.append(ref(p))
        for r in summary['transaction']['sources']:
            assert ref(r['path'])==r
            m=Path(r['path']);hpath=Path(str(m).removesuffix('.manifest.json'))
            manifest=read(m);seal=read(str(hpath)+'.seal.json')
            assert sha(hpath)==manifest['hdf5_sha256']==seal['hdf5_sha256']
            assert sha(m)==seal['manifest_sha256'];pins.append(ref(m));pins.append(ref(hpath))
            with h5py.File(hpath) as h:
                states=h['states'][...]
                # Include decision observations, exclude the terminal next state.
                N=len(h['transitions/action']);assert len(states) in (N,N+1)
                data.extend(context(raw) for raw in states[:N]);count+=1
    assert count==512
    return np.stack(data),pins

def forward(model,raw):
    x=torch.from_numpy(raw[None].copy());g=build_graph(x)
    prefix=pack_summary_prefixes(raw[None].copy(),feature_config=model.feature_config)
    y,d=model(x,prefix,diagnostic=True)
    assert torch.equal(y,model(x,prefix))
    return x,g,y,d

def main():
    torch.set_num_threads(1)
    src=sources();suite=src['3e-4'][0];cp=suite['parent_artifacts']['checkpoint']
    learner=JointCostPPO.from_checkpoint(cp['path'],expected_config=suite['learner_config'],expected_sha256=cp['sha256'],restore_global_rng=False)
    assert learner.update_index==13;learner.actor.eval();learner.critic.eval()
    before=training_state_fingerprint(learner)
    train,pins=training_support();states=read(OUT/'state_selection.json')['states']
    probes=[];eligibility=[];support_rows=[];branches=[]
    with ReadOnlyTrainingGuard(learner),torch.no_grad():
        for k,s in enumerate(states):
            base=np.asarray(s['raw'],dtype=np.float32);identity={n:s[n] for n in ('lr','run_index','window','state_index','time')}
            checks=builder.audit(base,base)
            assert all(r['passed'] for r in checks),(identity,[r for r in checks if not r['passed']])
            scenes={'BASE':base}
            for scene,fn in (('A',branch_a),('B',branch_b),('C',builder.scenario_c)):
                raw,definition=fn(base)
                eligibility.append(dict(**identity,scene=scene,eligible=raw is not None,reason='' if raw is not None else definition))
                if raw is None:continue
                audit=builder.audit(raw,base);assert all(r['passed'] for r in audit),(identity,scene,[r for r in audit if not r['passed']])
                assert np.array_equal(raw[19:27],base[19:27]),'Keep mask/quantity candidate sidecar fixed for the Graph/C128 decomposition'
                scenes[scene]=raw
                branches.append(dict(**identity,scene=scene,definition=definition,raw_sha256=builder.raw_sha(raw),
                    changed_indices=np.flatnonzero(raw!=base).tolist(),audit_checks=len(audit),all_audit_checks_passed=True))
            for scene,raw in scenes.items():
                ctx=context(raw)
                for dim,name in enumerate(CONTEXT_NAMES):
                    values=train[:,dim];v=float(ctx[dim])
                    support_rows.append(dict(**identity,scene=scene,feature=name,value=v,
                        training_min=float(values.min()),training_max=float(values.max()),
                        training_percentile=float(100*np.mean(values<=v)),
                        outside_observed_marginal_range=bool(v<values.min()-1e-6 or v>values.max()+1e-6)))
            for role,model in (('actor',learner.actor),('critic',learner.critic)):
                outputs={scene:forward(model,raw) for scene,raw in scenes.items()}
                x0,g0,y0,d0=outputs['BASE']
                for scene in scenes.keys()-{'BASE'}:
                    x1,g1,y1,d1=outputs[scene]
                    def out(x,g,d,c):
                        z=prior.readout(model,role,x,g,d['h2'],c)
                        return prior.numpy(z['probabilities' if role=='actor' else 'value']).reshape(-1).astype(float)
                    f00=out(x0,g0,d0,d0['current_summary']);f11=out(x1,g1,d1,d1['current_summary'])
                    f10=out(x1,g1,d1,d0['current_summary']);f01=out(x0,g0,d0,d1['current_summary'])
                    ordinary0=prior.numpy(y0.softmax(-1) if role=='actor' else y0).reshape(-1)
                    ordinary1=prior.numpy(y1.softmax(-1) if role=='actor' else y1).reshape(-1)
                    assert np.array_equal(f00,ordinary0.astype(float)) and np.array_equal(f11,ordinary1.astype(float))
                    gc=.5*((f10-f00)+(f11-f01));cc=.5*((f01-f00)+(f11-f10));full=f11-f00
                    assert np.max(np.abs(gc+cc-full))<1e-12
                    for a in range(len(full)):
                        probes.append(dict(**identity,role=role,scene=scene,action=a if role=='actor' else -1,
                            base=float(f00[a]),branch=float(f11[a]),full_delta=float(full[a]),
                            graph_shapley=float(gc[a]),c128_shapley=float(cc[a]),interaction=float(f11[a]-f10[a]-f01[a]+f00[a]),
                            opposite=bool(gc[a]*cc[a]<0 and min(abs(gc[a]),abs(cc[a]))>1e-6),
                            retained_fraction=abs(float(full[a]))/(abs(float(gc[a]))+abs(float(cc[a]))) if abs(gc[a])+abs(cc[a])>0 else 1.,
                            legal_actions=int(base[19:23].sum()),
                            graph_scope='GNN representations + deterministic graph quantities; raw mask/tons held fixed'))
            if (k+1)%20==0:print(f'expanded states {k+1}/{len(states)}',flush=True)
    assert before==training_state_fingerprint(learner);assert sha(CORE)==CORE_SHA
    aggregates=[]
    for role in ('actor','critic'):
        for scene in ('A','B','C'):
            rows=[r for r in probes if r['role']==role and r['scene']==scene]
            meaningful=[r for r in rows if abs(r['graph_shapley'])+abs(r['c128_shapley'])>1e-6]
            aggregates.append(dict(role=role,scene=scene,outputs=len(rows),meaningful_outputs=len(meaningful),
                opposing_outputs=sum(r['opposite'] for r in meaningful),
                mean_abs_full_delta=float(np.mean([abs(r['full_delta']) for r in rows])) if rows else None,
                median_retained_fraction=float(np.median([r['retained_fraction'] for r in meaningful])) if meaningful else None))
    save('interventions.json',dict(schema='bg128-expanded-real-state-exogenous-probes-v1',
        model=cp,model_lr='3e-4',checkpoint_update=13,base_states=len(states),
        state_cohort='failure-conditioned ten seed/window pairs, both real policy trajectories',
        training_decisions_scanned=len(train),training_shifts_scanned=512,aggregates=aggregates,
        eligible={s:sum(r['eligible'] for r in eligibility if r['scene']==s) for s in ('A','B','C')},
        model_state_unchanged=True,physical_shifts=0,optimizer_updates=0,
        limitations=['A completes a known current parcel at an internal event boundary; it is not an executed decision.',
            'B redirects a non-ego unstarted reservation; this is not an exposed PPO action.',
            'C halves crusher rate; all training rates are 2500 tph, so C is parameter extrapolation.',
            'Graph/C cross-combinations are neural mechanism probes, not physically reachable states.',
            'Opposing output contributions do not identify a faulty branch or demonstrate a worse dispatch.',
            'Marginal training ranges/percentiles do not establish joint support, reachability, or sufficient training coverage.',
            '1e-4 trajectory states are off-policy inputs to the frozen 3e-4 model.'],
        sources=[ref(OUT/'state_selection.json'),ref(Path(__file__)),ref(prior.__file__),ref(builder.__file__)],
        training_source_pins=pins,protected_ppo_sha=sha(CORE)))
    csv_save('intervention_eligibility.csv',eligibility);csv_save('readout_decomposition.csv',probes)
    csv_save('training_support.csv',support_rows);save('branches.json',branches)
    print(json.dumps(dict(states=len(states),branches=len(branches),aggregates=aggregates)))

if __name__=='__main__':main()
