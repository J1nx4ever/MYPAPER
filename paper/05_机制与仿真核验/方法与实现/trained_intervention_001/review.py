"""Independent official-forward replay, physical oracle and probe hook checks.

The primary analyzer's custom readout is not imported. Probe results are
reproduced by hooks on the unchanged official model's graph/summary interfaces.
"""
from pathlib import Path
import csv
from dataclasses import replace
import hashlib
import importlib.util
import json
import sys

PROJECT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(PROJECT))
import numpy as np
import torch
from baseline1_v1.evaluation import ReadOnlyTrainingGuard,training_state_fingerprint
from business_graph_summary_v1.learner import JointCostPPO
from business_graph_summary_v1.models import digest,pack_summary_prefixes
from business_graph_summary_v1.graph import build_graph,GROUP_NAMES,RELATION_NAMES

OUT=PROJECT/'reports/token_attention_001/business_graph_summary_009/trained_intervention_001'
OLD=OUT.parent/'physical_intervention_001'
PLAN=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def rows(n):
    with (OUT/n).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def arr(t):return t.detach().cpu().numpy()
def norm(x):return float(np.linalg.norm(np.asarray(x,dtype=np.float64)))


def graph_filter(g,probe,layer):
    # Separate predicate implementation for official GraphLayer pre-hooks.
    records=g.records();removed=[]
    for i,r in enumerate(records):
        block=False
        if probe.startswith('no_'):
            block=probe=='no_all_messages_both' or probe=='no_'+r['group']+'_both'
        if probe=='block_Truck1_to_Dump0_L1' and layer==1:
            block=r['source']==1 and r['destination']==13 and r['relation']=='loaded_dump'
        if probe=='block_Truck8_to_targetLoad_L1' and layer==1:
            block=r['source']==8 and r['relation']=='reserved_load'
        if probe=='block_Dump0_to_oreLoads_L'+str(layer):
            block=r['source']==13 and r['relation']=='compatible_material_reverse'
        removed.append(block)
    keep=~torch.tensor(removed,dtype=torch.bool)
    return replace(g,edge_index=g.edge_index[:,keep],edge_type=g.edge_type[keep],
        edge_group=g.edge_group[keep],edge_attr=g.edge_attr[keep]).validate()


def main():
    torch.set_num_threads(1)
    summary=read(OUT/'summary.json');bundle=read(OLD/'scenarios.json')
    data=np.load(OUT/'analysis_tensors.npz',allow_pickle=False)
    spec=importlib.util.spec_from_file_location('old_independent_physical_review',PLAN.parent/'physical_intervention_001/review_interventions.py')
    oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)
    checks={}
    def check(name,ok):checks[name]=bool(ok);assert checks[name],name
    check('checkpoint13_is_trained_formal',summary['checkpoint_update']==13 and 'trained formal' in summary['parameter_status'])
    parent=summary['parent_artifacts'];check('sealed_parent_bytes',all(sha(r['path'])==r['sha256'] for r in parent.values()))
    model=JointCostPPO.from_checkpoint(parent['checkpoint']['path'],expected_sha256=parent['checkpoint']['sha256'],restore_global_rng=False)
    model.actor.eval();model.critic.eval();before=training_state_fingerprint(model)
    check('formal_ckpt13_config',model.update_index==13 and model.config['candidate_id']=='s01_BG128' and model.config['source_kind']=='real_anylogic_rpc')
    old_summary=read(OLD/'summary.json')
    for role in ('actor','critic'):
        check(role+'_trained_not_random_parameters',digest(getattr(model,role))!=old_summary['roles'][role]['parameter_sha256_after'])
    raw={s:np.asarray(bundle['base']['raw'] if s=='BASE' else bundle['scenarios'][s]['raw'],dtype=np.float32) for s in ('BASE','A','B','C')}
    for key in data.files:check('finite_array_'+key,bool(np.isfinite(data[key]).all()))
    for scene,r in raw.items():
        check(scene+'_exact_frozen_raw',np.array_equal(r,data[scene+'_raw']))
        g=build_graph(torch.tensor(r[None]));check(scene+'_independent_quantity_oracle',np.max(np.abs(oracle.quantity_oracle(r)-arr(g.raw_quantities[0])))<.0002)
        for name,ok in oracle.physical_checks(r).items():check(scene+'_'+name,ok)
        check(scene+'_original_mask_tons',np.array_equal(r[19:27],raw['BASE'][19:27]))
    check('C_only_true_processing_rate_halved',np.flatnonzero(raw['C']!=raw['BASE']).tolist()==[1304] and raw['C'][1304]==raw['BASE'][1304]*.5)
    whites={'A':{155,162,164,166,167,168,169,170,172,173,177,231,232,233,1112,1113,1115,1122,1123,1145},
        'B':{1013,1014,1049,1058,1062,1063,1113,1114,1124,1160,1161,1171},'C':{1304}}
    for scene in 'ABC':check(scene+'_independent_raw_whitelist',set(np.flatnonzero(raw[scene]!=raw['BASE']))==whites[scene])
    probe_rows=rows('mechanism_probes.csv');node_rows=rows('message_probe_nodes.csv');combo_rows=rows('readout_combinations.csv')
    primary={role:{} for role in ('actor','critic')}
    with ReadOnlyTrainingGuard(model),torch.no_grad():
        for role in ('actor','critic'):
            network=getattr(model,role)
            for scene,r in raw.items():
                x=torch.tensor(r[None]);prefix=pack_summary_prefixes(r[None].copy(),feature_config=network.feature_config)
                output,d=network(x,prefix,diagnostic=True);primary[role][scene]=(output,d)
                check(role+'_'+scene+'_ordinary_exact',torch.equal(output,network(x,prefix)))
                for k,key in enumerate(('z0','h1','h2')):check(role+'_'+scene+'_H'+str(k)+'_exact',np.array_equal(arr(d[key][0]),data[f'{role}_{scene}_H{k}']))
                if role=='actor':
                    check(scene+'_actual_actor_probabilities',np.array_equal(arr(output.softmax(-1)[0]),data[f'actor_{scene}_probs']))
                    check(scene+'_actual_actor_candidates',np.array_equal(arr(d['candidates'][0]),data[f'actor_{scene}_candidates']))
                else:check(scene+'_actual_critic_value',np.array_equal(arr(output),data[f'critic_{scene}_value']))
            for probe in summary['mechanism_probes']['message_inhibition']:
                result={}
                handles=[layer.register_forward_pre_hook(lambda module,args,n=n:
                    (args[0],graph_filter(args[1],probe,n))) for n,layer in enumerate(network.state_encoder.graph_layers,1)]
                try:
                    for scene,r in raw.items():
                        prefix=pack_summary_prefixes(r[None].copy(),feature_config=network.feature_config)
                        out,d=network(torch.tensor(r[None]),prefix,diagnostic=True)
                        check(role+'_'+scene+'_'+probe+'_H0_unchanged',np.array_equal(arr(d['z0'][0]),data[f'{role}_{scene}_H0']))
                        result[scene]=(out,d)
                finally:
                    for h in handles:h.remove()
                for scene in 'ABC':
                    for row in [r for r in node_rows if r['role']==role and r['scene']==scene and r['probe']==probe]:
                        node=__import__('business_graph_summary_v1.graph',fromlist=['NODE_NAMES']).NODE_NAMES.index(row['node'])
                        delta=arr(result[scene][1]['h2'][0,node])-arr(result['BASE'][1]['h2'][0,node])
                        check(role+scene+probe+row['node']+'_blocked_H2_official',abs(norm(delta)-float(row['blocked_H2_delta_l2']))<1e-9)
                    for row in [r for r in probe_rows if r['role']==role and r['scene']==scene and r['probe']==probe]:
                        a=int(row['action']);target=row['target']
                        if target=='value':v0=arr(result['BASE'][0]).reshape(-1);v1=arr(result[scene][0]).reshape(-1)
                        elif target=='probabilities':v0=arr(result['BASE'][0].softmax(-1)[0]);v1=arr(result[scene][0].softmax(-1)[0])
                        else:
                            v0=arr(result['BASE'][1]['unmasked_logits'][0]);v1=arr(result[scene][1]['unmasked_logits'][0]);v0=v0-v0.mean();v1=v1-v1.mean()
                        check(role+scene+probe+target+str(a)+'_official_readout',abs(float(v1[a]-v0[a])-float(row['probe_delta']))<1e-6)
            # Independent mixed-C check through official summary encoder hooks.
            for scene in 'ABC':
                for gs,cs in (('BASE','BASE'),(scene,'BASE'),('BASE',scene),(scene,scene)):
                    current=torch.tensor(data[cs+'_C128'][None].copy())
                    if role=='actor':
                        handles=[network.summary_task_encoder.register_forward_pre_hook(lambda module,args:(current[:,:88],)),
                            network.summary_action_encoder.register_forward_pre_hook(lambda module,args:(current[:,88:].reshape(-1,4,10),))]
                    else:handles=[network.summary_encoder.register_forward_pre_hook(lambda module,args:(current,))]
                    try:
                        prefix=pack_summary_prefixes(raw[gs][None].copy(),feature_config=network.feature_config)
                        out=network(torch.tensor(raw[gs][None]),prefix)
                    finally:
                        for h in handles:h.remove()
                    key=('G1' if gs==scene else 'G0')+('C1' if cs==scene else 'C0')
                    for row in [r for r in combo_rows if r['role']==role and r['scene']==scene and r['probe']==key]:
                        value=out.softmax(-1) if row['target']=='probabilities' else out
                        check(role+scene+key+row['target']+row['action']+'_official_mixed_C',abs(float(value.reshape(-1)[int(row['action'])])-float(row['value']))<1e-7)
    check('training_state_parameters_optimizers_rng_unchanged',before==training_state_fingerprint(model))
    check('trained_digests_exact',summary['parameter_digests']=={role:digest(getattr(model,role)) for role in ('actor','critic')})
    for row in rows('readout_decomposition.csv'):
        check('additive_decomposition_'+row['role']+row['scene']+row['target']+row['action'],
            abs(float(row['total_change'])-float(row['graph_interface_shapley'])-float(row['C128_interface_shapley']))<1e-12)
    check('no_optimization_physical_shift_or_budget',summary['execution']==dict(training=False,optimizer_updates=0,gradient_updates=0,
        new_physical_shifts=0,rpc_started=False,anylogic_started=False,budget_registered=False,source_or_checkpoint_modified=False))
    check('frozen_sources_unchanged',all(sha(r['path'])==r['sha256'] for r in summary['source_refs']))
    check('parent_checkpoint_artifacts_unchanged',all(sha(r['path'])==r['sha256'] for r in parent.values()))
    report=dict(schema='bg128-trained-interventions-independent-review-v1',status='PASS',checks=checks,
        total_checks=len(checks),method='independent physical oracle, official complete primary forward, official graph/summary interface hook replay of mechanism probes; primary manual readout is not imported',
        source_refs=[dict(path=str((OUT/n).resolve()),sha256=sha(OUT/n)) for n in ('summary.json','analysis_tensors.npz')],
        optimizer_updates=0,new_physical_shifts=0)
    with (OUT/'independent_review.json').open('x') as stream:json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status='PASS',checks=len(checks),checkpoint_update=13),ensure_ascii=False))


if __name__=='__main__':main()
