"""Checkpoint13 raw A/B/C interventions plus separately labelled mechanism probes.

Only the four primary branches are physically consistent raw observations.
Message blocking and mixed graph/C128 readouts are diagnostic counterfactuals.
No optimizer, RPC, simulation or budget API is invoked.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
from datetime import datetime
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

PROJECT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT))
import h5py
import numpy as np
import torch

from baseline1_v1.evaluation import ReadOnlyTrainingGuard, training_state_fingerprint
from business_graph_summary_v1.models import digest, pack_summary_prefixes, TYPE_SLICES
from business_graph_summary_v1.learner import JointCostPPO
from business_graph_summary_v1.c_features import build_features, FEATURE_NAMES
from business_graph_summary_v1.graph import (build_graph, NODE_NAMES, GROUP_NAMES,
    RELATION_NAMES, EDGE_ATTRIBUTE_NAMES, QUANTITY_NAMES)

PLAN = Path(__file__).resolve().parent
OUT = PROJECT/'reports/token_attention_001/business_graph_summary_009/trained_intervention_001'
OLD = OUT.parent/'physical_intervention_001'
OLD_PLAN = PLAN.parent/'physical_intervention_001'
EVAL_PLAN = PLAN.parent/'checkpoint_evaluation_001/prepared_001'
SCENES = ('BASE', 'A', 'B', 'C')
CORE = PROJECT.parents[1]/'python/episode_budget_ppo.py'
CORE_SHA = '8e4cfa31c622f5e33756e304eea3f7526547db72ae93f7d9f00808845d61ef96'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def ref(path): return dict(path=str(Path(path).resolve()), sha256=sha(path))
def read(path): return json.loads(Path(path).read_bytes())
def save(name, value):
    data = (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2)+'\n').encode()
    with (OUT/name).open('xb') as stream: stream.write(data)


def csv_save(name, rows):
    assert rows, name
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('x', encoding='utf-8-sig', newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def module(name, path):
    spec=importlib.util.spec_from_file_location(name,path);result=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def numpy(value): return value.detach().cpu().numpy().copy()
def l2(value): return float(np.linalg.norm(np.asarray(value,dtype=np.float64)))
def centered(value): return value-value.mean(-1,keepdims=True)


def readout(model, role, raw, graph, h, current):
    """Exact formal readout with explicit graph and C128 interface arguments.

    Cross-scene arguments are allowed ONLY in the separately marked probes.
    The primary result is always obtained from the official complete forward.
    """
    ego=int(graph.ego_ids[0]);dump=graph.candidate_dump_ids
    pools=torch.stack([h[:,section].mean(1) for section in TYPE_SLICES],1)
    qpool=torch.stack([graph.quantities[:,section].sum(1) for section in TYPE_SLICES],1)
    common=torch.cat((pools.flatten(1),h[:,ego],h[:,18],qpool.flatten(1)),-1)
    if role=='critic':
        value=model.fusion(torch.cat((common,model.summary_encoder(current)),-1)).squeeze(-1)
        return dict(value=value, raw_context=common)
    task=model.summary_task_encoder(current[:,:88])
    action=model.summary_action_encoder(current[:,88:].reshape(-1,4,10))
    context=model.context_encoder(torch.cat((common,task),-1))
    candidates_in=torch.cat((h[:,ego:ego+1].expand(-1,4,-1),h[:,9:13],h[torch.arange(1)[:,None],dump],
        graph.quantities[:,ego:ego+1].expand(-1,4,-1),graph.quantities[:,9:13],
        graph.quantities[torch.arange(1)[:,None],dump],raw[:,19:23,None],raw[:,23:27,None],action),-1)
    candidates=model.candidate_encoder(candidates_in)
    logits=model.score_candidates(candidates,context)
    masked=logits.masked_fill(~raw[:,19:23].bool(),-torch.inf)
    return dict(logits=logits,probabilities=masked.softmax(-1),candidates=candidates,
                candidate_input=candidates_in,raw_context=common)


def filtered(graph, name, layer):
    src,dst=graph.edge_index
    keep=torch.ones(len(graph.edge_type),dtype=torch.bool)
    if name.startswith('no_') and name.endswith('_both'):
        group=name[3:-5]
        keep &= False if group=='all_messages' else graph.edge_group!=GROUP_NAMES.index(group)
    elif name=='block_Truck1_to_Dump0_L1' and layer==1:
        keep &= ~((src==1)&(dst==13)&(graph.edge_type==RELATION_NAMES.index('loaded_dump')))
    elif name=='block_Truck8_to_targetLoad_L1' and layer==1:
        keep &= ~((src==8)&(graph.edge_type==RELATION_NAMES.index('reserved_load')))
    elif name in ('block_Dump0_to_oreLoads_L1','block_Dump0_to_oreLoads_L2') and layer==int(name[-1]):
        keep &= ~((src==13)&(graph.edge_type==RELATION_NAMES.index('compatible_material_reverse')))
    return replace(graph,edge_index=graph.edge_index[:,keep],edge_type=graph.edge_type[keep],
        edge_group=graph.edge_group[keep],edge_attr=graph.edge_attr[keep]).validate()


def main():
    assert sys.executable=='/home/jinx/miniconda3/envs/anylogic-env/bin/python'
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    assert sha(CORE)==CORE_SHA
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'summary.json').exists(), 'preserve completed evidence; use a fresh report directory'
    builder=module('prior_physical_builder',OLD_PLAN/'build_scenarios.py')
    prior=module('prior_physical_analyzer',OLD_PLAN/'analyze_interventions.py')
    oracle=module('prior_physical_reviewer',OLD_PLAN/'review_interventions.py')
    manifest=read(OLD/'manifest.json');bundle=read(OLD/'scenarios.json')
    historical_refs=list(manifest['artifacts'].values())+manifest['script_references']
    assert all(sha(r['path'])==r['sha256'] for r in historical_refs)
    base=np.asarray(bundle['base']['raw'],dtype=np.float32)
    assert sha(bundle['base']['source_hdf5'])==bundle['base']['source_sha256']
    with h5py.File(bundle['base']['source_hdf5'],'r') as stream:
        assert np.array_equal(base,stream['states'][bundle['base']['state_index']])
    raw={'BASE':base};physical_checks={'BASE':builder.audit(base,base)}
    for scene,fn in (('A',builder.scenario_a),('B',builder.scenario_b),('C',builder.scenario_c)):
        value,definition=fn(base)
        assert np.array_equal(value,np.asarray(bundle['scenarios'][scene]['raw'],dtype=np.float32))
        assert definition==bundle['scenarios'][scene]['intervention']
        assert builder.raw_sha(value)==bundle['scenarios'][scene]['raw_sha256']
        physical_checks[scene]=builder.audit(value,base);raw[scene]=value
    assert all(r['passed'] for rows in physical_checks.values() for r in rows)
    assert np.flatnonzero(raw['C']!=base).tolist()==[1304] and raw['C'][1304]==base[1304]/2
    suite=read(EVAL_PLAN/'suite.json');parents=suite['parent_artifacts']
    assert all(ref(r['path'])==r for r in parents.values())
    learner=JointCostPPO.from_checkpoint(parents['checkpoint']['path'],expected_config=suite['learner_config'],
        expected_sha256=parents['checkpoint']['sha256'],restore_global_rng=False)
    assert learner.update_index==13 and learner.config['candidate_id']=='s01_BG128'
    learner.actor.eval();learner.critic.eval()
    before=training_state_fingerprint(learner)
    models={'actor':learner.actor,'critic':learner.critic}
    digests={role:digest(m) for role,m in models.items()}
    tensors={s:torch.tensor(r[None].copy()) for s,r in raw.items()}
    graphs={s:build_graph(x) for s,x in tensors.items()}
    arrays={};rows={name:[] for name in ('raw_changes','graph_edges','quantities','c128_features',
        'node_representations','representation_changes','actor_candidates','candidate_changes',
        'actor_logits','group_messages','incoming_paths','mechanism_probes','message_probe_nodes',
        'readout_combinations','readout_decomposition','token_input_changes')}
    schema=read(PROJECT/'python/observation_schema.json')
    detail={role:{} for role in models};outputs={role:{} for role in models}
    for scene in SCENES:
        g=graphs[scene];r=raw[scene]
        assert Counter(RELATION_NAMES[i] for i in g.edge_type.tolist())['road']==16
        assert np.max(np.abs(oracle.quantity_oracle(r)-numpy(g.raw_quantities[0])))<.0002
        rows['graph_edges'].extend(prior.edge_rows(g,scene))
        arrays[scene+'_raw']=r;arrays[scene+'_quantities']=numpy(g.quantities[0])
        arrays[scene+'_raw_quantities']=numpy(g.raw_quantities[0])
        for key in ('edge_index','edge_type','edge_group','edge_attr'):arrays[scene+'_'+key]=numpy(getattr(g,key))
        for node,name in enumerate(NODE_NAMES):
            rows['quantities'].append(dict(scene=scene,node=name,**dict(zip(QUANTITY_NAMES,numpy(g.raw_quantities[0,node]).tolist()))))
        if scene!='BASE':
            rows['raw_changes'].extend(dict(scene=scene,index=int(i),field=schema[i]['name'],before=float(base[i]),
                after=float(r[i]),delta=float(r[i])-float(base[i])) for i in np.flatnonzero(base!=r))
    with ReadOnlyTrainingGuard(learner),torch.no_grad():
        for role,m in models.items():
            for scene in SCENES:
                x=tensors[scene];g=graphs[scene]
                prefix=pack_summary_prefixes(raw[scene][None].copy(),feature_config=m.feature_config)
                result,d=m(x,prefix,diagnostic=True);ordinary=m(x,prefix)
                assert torch.equal(result,ordinary)
                direct=build_features(raw[scene],config=m.feature_config)
                assert np.array_equal(prefix.features[0],direct)
                if role=='actor':
                    arrays[scene+'_C128']=direct
                    for i,name in enumerate(FEATURE_NAMES):
                        rows['c128_features'].append(dict(scene=scene,index=i,feature=name,value=float(direct[i]),
                            base=float(arrays['BASE_C128'][i]),delta=float(direct[i]-arrays['BASE_C128'][i])))
                else:assert np.array_equal(direct,arrays[scene+'_C128'])
                detail[role][scene]=d;outputs[role][scene]=numpy(result[0] if role=='actor' else result)
                z,td=m.state_encoder.token_encoder(x,g,diagnostic=True)
                assert torch.equal(z,d['z0'])
                for key in ('site_base','site_input','site_geography','quantity_content','truck_local'):
                    arrays[f'{role}_{scene}_token_{key}']=numpy(td[key][0])
                for k,key in enumerate(('z0','h1','h2')):
                    arrays[f'{role}_{scene}_H{k}']=numpy(d[key][0])
                    for n,name in enumerate(NODE_NAMES):
                        value=arrays[f'{role}_{scene}_H{k}'][n]
                        rows['node_representations'].append(dict(role=role,scene=scene,layer=k,node=name,
                            **{f'h{i:02d}':float(v) for i,v in enumerate(value)}))
                        if scene!='BASE':rows['representation_changes'].append(dict(role=role,scene=scene,
                            layer=k,node=name,**prior.changes(arrays[f'{role}_BASE_H{k}'][n],value)))
                manual=readout(m,role,x,g,d['h2'],d['current_summary'])
                assert torch.equal(manual['logits' if role=='actor' else 'value'],d['unmasked_logits'] if role=='actor' else result)
                if role=='actor':
                    assert torch.equal(manual['candidate_input'],d['candidate_input'])
                    for key in ('candidate_input','candidates','context_input','context','task_summary','action_summary','unmasked_logits'):
                        arrays[f'actor_{scene}_{key}']=numpy(d[key][0])
                    p=numpy(result.softmax(-1)[0]);arrays[f'actor_{scene}_probs']=p
                    logit=numpy(d['unmasked_logits'][0]);cl=centered(logit)
                    arrays[f'actor_{scene}_centered_logits']=cl
                    for a in range(4):
                        rows['actor_candidates'].append(dict(scene=scene,action=a,load=f'Load{a}',
                            **{f'h{i:02d}':float(v) for i,v in enumerate(numpy(d['candidates'][0,a]))}))
                        rows['actor_logits'].append(dict(scene=scene,action=a,load=f'Load{a}',legal=bool(raw[scene][19+a]),
                            dispatch_tons=float(raw[scene][23+a]*193),unmasked_logit=float(logit[a]),centered_logit=float(cl[a]),
                            probability=float(p[a]),delta_logit=float(logit[a]-arrays['actor_BASE_unmasked_logits'][a]),
                            delta_centered_logit=float(cl[a]-arrays['actor_BASE_centered_logits'][a]),
                            delta_probability=float(p[a]-arrays['actor_BASE_probs'][a])))
                        if scene!='BASE':
                            for name,lo,hi in prior.CANDIDATE_BLOCKS:
                                rows['candidate_changes'].append(dict(scene=scene,action=a,block=name,
                                    **prior.changes(arrays['actor_BASE_candidate_input'][a,lo:hi],arrays[f'actor_{scene}_candidate_input'][a,lo:hi])))
                            rows['candidate_changes'].append(dict(scene=scene,action=a,block='candidates',
                                **prior.changes(arrays['actor_BASE_candidates'][a],arrays[f'actor_{scene}_candidates'][a])))
                else:arrays[f'critic_{scene}_value']=numpy(result)
                for layer in (1,2):
                    ld=d[f'layer{layer}'];arrays[f'{role}_{scene}_layer{layer}_group_messages']=numpy(ld['group_messages'][0])
                    contributions=[]
                    for gi,group in enumerate(GROUP_NAMES):
                        message=ld['group_messages'][:,:,gi]
                        weight=m.state_encoder.graph_layers[layer-1].output.weight
                        contribution=message@weight[:,gi*64:(gi+1)*64].T+ld['group_masks'][:,:,gi,None]*weight[:,256+gi]
                        contributions.append(contribution)
                        arrays[f'{role}_{scene}_layer{layer}_{group}_projected']=numpy(contribution[0])
                        for n,name in enumerate(NODE_NAMES):
                            rows['group_messages'].append(dict(role=role,scene=scene,layer=layer,node=name,group=group,
                                incoming_count=int(ld['group_counts'][0,n,gi]),message_norm=l2(numpy(message[0,n])),
                                projected_norm=l2(numpy(contribution[0,n])),
                                delta_message_l2=l2(numpy(message[0,n])-arrays[f'{role}_BASE_layer{layer}_group_messages'][n,gi]),
                                delta_projected_l2=l2(numpy(contribution[0,n])-arrays[f'{role}_BASE_layer{layer}_{group}_projected'][n])))
                    assert torch.allclose(sum(contributions),ld['projected'],rtol=2e-6,atol=2e-6)
        # Mixed readout interfaces are mechanism probes, not additional raw scenes.
        for role,m in models.items():
            for scene in SCENES[1:]:
                combos={}
                for gs,cs in (('BASE','BASE'),(scene,'BASE'),('BASE',scene),(scene,scene)):
                    out=readout(m,role,tensors[gs],graphs[gs],detail[role][gs]['h2'],detail[role][cs]['current_summary'])
                    key=('G1' if gs==scene else 'G0')+('C1' if cs==scene else 'C0')
                    combos[key]={k:numpy(v[0]) for k,v in out.items() if k in ('logits','probabilities','value')}
                    for target,value in combos[key].items():
                        for a,v in enumerate(np.atleast_1d(value)):
                            rows['readout_combinations'].append(dict(role=role,scene=scene,probe=key,
                                graph_source=gs,c128_source=cs,target=target,action=a,value=float(v),physical=False))
                for target in combos['G0C0']:
                    vv={key:np.atleast_1d(value[target]).astype(np.float64) for key,value in combos.items()}
                    if target=='logits':vv={key:centered(value) for key,value in vv.items()}
                    graph_part=((vv['G1C0']-vv['G0C0'])+(vv['G1C1']-vv['G0C1']))/2
                    c_part=((vv['G0C1']-vv['G0C0'])+(vv['G1C1']-vv['G1C0']))/2
                    total=vv['G1C1']-vv['G0C0'];assert np.allclose(graph_part+c_part,total,atol=1e-12)
                    for a in range(len(total)):
                        rows['readout_decomposition'].append(dict(role=role,scene=scene,
                            target='centered_logits' if target=='logits' else target,action=a,
                            total_change=float(total[a]),graph_interface_shapley=float(graph_part[a]),
                            C128_interface_shapley=float(c_part[a]),physical=False))
        # Edge deletion blocks only messages; graph quantities/C128/H0 stay full.
        probes=[f'no_{g}_both' for g in GROUP_NAMES]+['no_all_messages_both',
            'block_Truck1_to_Dump0_L1','block_Truck8_to_targetLoad_L1',
            'block_Dump0_to_oreLoads_L1','block_Dump0_to_oreLoads_L2']
        for role,m in models.items():
            for probe in probes:
                probe_results={};probe_h={}
                for scene in SCENES:
                    g=graphs[scene];h=m.state_encoder.token_encoder(tensors[scene],g)
                    assert torch.equal(h,detail[role][scene]['z0'])
                    for layer,network in enumerate(m.state_encoder.graph_layers,1):h=network(h,filtered(g,probe,layer))
                    probe_h[scene]=numpy(h[0]);o=readout(m,role,tensors[scene],g,h,detail[role][scene]['current_summary'])
                    probe_results[scene]={key:numpy(v[0]) for key,v in o.items() if key in ('logits','probabilities','value')}
                for scene in SCENES[1:]:
                    for n in list(range(9,15)):
                        full=arrays[f'{role}_{scene}_H2'][n]-arrays[f'{role}_BASE_H2'][n]
                        blocked=probe_h[scene][n]-probe_h['BASE'][n]
                        rows['message_probe_nodes'].append(dict(role=role,scene=scene,probe=probe,node=NODE_NAMES[n],
                            full_H2_delta_l2=l2(full),blocked_H2_delta_l2=l2(blocked),
                            difference_of_responses_l2=l2(full-blocked),physical=False))
                    for target in probe_results[scene]:
                        full0=arrays['actor_BASE_unmasked_logits' if target=='logits' else 'actor_BASE_probs'] if role=='actor' else arrays['critic_BASE_value']
                        full1=arrays[f'actor_{scene}_unmasked_logits' if target=='logits' else f'actor_{scene}_probs'] if role=='actor' else arrays[f'critic_{scene}_value']
                        a0=np.atleast_1d(probe_results['BASE'][target]);a1=np.atleast_1d(probe_results[scene][target])
                        if target=='logits':a0,a1,full0,full1=[centered(v) for v in (a0,a1,full0,full1)]
                        for a in range(len(a1)):
                            rows['mechanism_probes'].append(dict(role=role,scene=scene,probe=probe,
                                target='centered_logits' if target=='logits' else target,action=a,
                                full_delta=float(np.atleast_1d(full1-full0)[a]),probe_delta=float(a1[a]-a0[a]),
                                difference_of_deltas=float(np.atleast_1d(full1-full0)[a]-(a1[a]-a0[a])),physical=False))
    assert before==training_state_fingerprint(learner)
    assert digests=={role:digest(m) for role,m in models.items()}
    # Pure topology inventory: original entities and direction/attributes retained.
    for scene,source in (('A',1),('B',8),('C',13)):
        records=graphs[scene].records()
        for end in range(9,13):
            for first in [r for r in records if r['source']==source]:
                if first['destination']==end:
                    rows['incoming_paths'].append(dict(scene=scene,hops=1,source=NODE_NAMES[source],relay='',
                        destination=NODE_NAMES[end],relation1=first['relation'],group1=first['group'],relation2='',group2='',
                        attributes1=json.dumps(first['attributes']),attributes2=''))
                for second in [r for r in records if r['source']==first['destination'] and r['destination']==end]:
                    rows['incoming_paths'].append(dict(scene=scene,hops=2,source=NODE_NAMES[source],
                        relay=NODE_NAMES[first['destination']],destination=NODE_NAMES[end],relation1=first['relation'],
                        group1=first['group'],relation2=second['relation'],group2=second['group'],
                        attributes1=json.dumps(first['attributes']),attributes2=json.dumps(second['attributes'])))
    for role in models:
        for scene in SCENES[1:]:
            for key in ('site_base','site_input','site_geography','quantity_content','truck_local'):
                aa=arrays[f'{role}_BASE_token_{key}'];bb=arrays[f'{role}_{scene}_token_{key}']
                names=NODE_NAMES[9:15] if key.startswith('site') else NODE_NAMES if key=='quantity_content' else NODE_NAMES[:9]
                for n,name in enumerate(names):rows['token_input_changes'].append(dict(role=role,scene=scene,node=name,
                    token_part=key,delta_l2=l2(bb[n]-aa[n])))
    arrays['actor_digest_bytes']=np.frombuffer(bytes.fromhex(digests['actor']),dtype=np.uint8)
    np.savez_compressed(OUT/'analysis_tensors.npz',**arrays)
    for name,value in rows.items():csv_save(name+'.csv',value)
    assert all(ref(r['path'])==r for r in parents.values()) and sha(CORE)==CORE_SHA
    assert all(sha(r['path'])==r['sha256'] for r in historical_refs)
    graphinfo={s:dict(nodes=19,edges=len(g.edge_type),roads=int((g.edge_type==0).sum()),
        candidate_dump_ids=g.candidate_dump_ids[0].tolist(),ego_id=int(g.ego_ids[0])) for s,g in graphs.items()}
    result={s:dict(mask=raw[s][19:23].astype(int).tolist(),dispatch_tons=(raw[s][23:27]*193).tolist(),
        logits=arrays[f'actor_{s}_unmasked_logits'].tolist(),centered_logits=arrays[f'actor_{s}_centered_logits'].tolist(),
        probabilities=arrays[f'actor_{s}_probs'].tolist(),critic_value=float(arrays[f'critic_{s}_value'][0])) for s in SCENES}
    save('physical_checks.json',dict(checks=physical_checks,all_passed=True,
        quantity_oracle='previous independent reviewer quantity_oracle; 21 counts/sums per node',
        primary_scenarios_exact_old_raw=True))
    save('summary.json',dict(schema='bg128-trained-physical-interventions-v1',
        generated_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),checkpoint_update=13,
        parameter_status='trained formal BG128 checkpoint13; no random Actor/Critic used',
        parent_artifacts=parents,parameter_digests=digests,base={k:v for k,v in bundle['base'].items() if k!='raw'},
        intervention_definitions={s:bundle['scenarios'][s]['intervention'] for s in SCENES[1:]},
        graph=graphinfo,primary_results=result,
        frozen_inputs=dict(old_manifest=ref(OLD/'manifest.json'),old_scenarios=ref(OLD/'scenarios.json'),
            source_hdf5=ref(bundle['base']['source_hdf5'])),
        source_refs=[ref(p) for p in (OLD_PLAN/'build_scenarios.py',OLD_PLAN/'analyze_interventions.py',
            OLD_PLAN/'review_interventions.py',CORE)]+[ref(p) for p in sorted((PROJECT/'business_graph_summary_v1').glob('*.py'))],
        primary_method='same BASE; independent raw A/B/C; full official graph/quantities/Token/C128/Actor/Critic rebuild',
        mechanism_probes=dict(physical=False,message_inhibition=probes,
            retained='full original per-scene H0, quantities, C128 and direct candidate fields',
            readout_decomposition='two-interface Shapley over four Graph/C128 readout combinations; interactions split equally; not intrinsic feature importance'),
        execution=dict(training=False,optimizer_updates=0,gradient_updates=0,new_physical_shifts=0,
            rpc_started=False,anylogic_started=False,budget_registered=False,source_or_checkpoint_modified=False),
        historical_record_hashes_checked=len(historical_refs),csv_rows={k:len(v) for k,v in rows.items()},
        limits=['Historical BASE came from another policy, so this is a fixed-state intervention probe, not on-policy evaluation.',
            'A changes loading completion, service, reservation and inbound material jointly; no single-factor congestion attribution.',
            'C changes actual raw crusherRate, not busy; future processing is projected, not simulated.',
            'H0 already contains reference content and deterministic neighbor quantities; changed H0 is not solely self state.',
            'GAT group/edge blocking and cross-interface readouts are mechanism probes, not coherent physical states.',
            'Multiple paths and direct Dump reading can encode the same fact; counts/tons are not summed across neural views.',
            'Correct qualitative response in these three cases does not prove global monotonicity or improved operational return.']))
    print(json.dumps(dict(status='PASS',checkpoint_update=13,primary_results=result,
        optimizer_updates=0,new_physical_shifts=0),ensure_ascii=False))


if __name__=='__main__':main()
