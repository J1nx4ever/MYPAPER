"""Read-only BG009 graph construction on 32 historical shifts; no model forward.

All graph records, C128 and quantities are produced by the formal implementation.
CSV physical features are raw schema values, not learned 64D embeddings.  A
post-hoc pair selection is for visualization only and never changes either input.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT))
import h5py
import numpy as np
import torch
from business_graph_summary_v1.graph import (build_graph, NODE_NAMES, GROUP_NAMES,
    RELATION_NAMES, EDGE_FEATURE_NAMES, QUANTITY_NAMES, QUANTITY_SCALES,
    ROAD_ENDPOINTS, PHYSICAL_NODE_TO_TOKEN)
from business_graph_summary_v1.contract import ablation_contract, feature_config_for
from business_graph_summary_v1.c_features import build_features, FEATURE_NAMES, feature_contract
from python.hdf5_storage import read_dataset

DEFAULT_SOURCE = PROJECT / 'runs/single-critic-task-query-formal-seed1-20261117-002/arms/s01_TQ128/train_u0016/episodes'
DEFAULT_OUTPUT = PROJECT / 'reports/token_attention_001/business_graph_summary_009/historical_snapshot_001'
SCHEMA = PROJECT / 'change_plans/token_attention_001/business_graph_summary_009/schema.json'
RAW_SCHEMA = PROJECT / 'python/observation_schema.json'
CORE = PROJECT.parents[1] / 'python/episode_budget_ppo.py'
CORE_SHA = '8e4cfa31c622f5e33756e304eea3f7526547db72ae93f7d9f00808845d61ef96'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')

def write_csv(path, fields, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)

def onehot(x):
    return int(np.argmax(x)) if np.sum(x) > 0 else None

def wire(x):
    return int(np.rint(float(x)*9))-1

def stable_records(graph):
    """Preserve parallel relations and service instances, never merge node pairs."""
    records = graph.records()
    seen = Counter()
    for rec in records:
        a = rec['attributes']
        slot = str(int(round(a['position_9']*9))) if a['position_valid'] else '-'
        base = f"{rec['relation']}|{rec['source']}|{rec['destination']}|slot={slot}"
        n = seen[base]; seen[base] += 1
        rec['stable_key'] = base+f'|instance={n}'
        rec['source_name'] = NODE_NAMES[rec['source']]
        rec['destination_name'] = NODE_NAMES[rec['destination']]
    return records

def independent_quantities(raw):
    """Direct source-field oracle, deliberately independent of graph records."""
    q = np.zeros((19,21), dtype=np.float64)
    qi = {name:i for i,name in enumerate(QUANTITY_NAMES)}
    def inc(n,k,v=1): q[n,qi[k]] += v
    trucks=raw[27:1107].reshape(9,120); loads=raw[1107:1295].reshape(4,47)
    dumps=raw[1295:1391].reshape(2,48)
    ego=onehot(trucks[:,0]); segments=[onehot(t[45:61]) for t in trucks]
    occupancy=Counter(s for s in segments if s is not None)
    for rid,(a,z) in enumerate(ROAD_ENDPOINTS):
        src,dst=PHYSICAL_NODE_TO_TOKEN[a],PHYSICAL_NODE_TO_TOKEN[z]
        inc(src,'outgoing_road_count');inc(dst,'incoming_road_count')
        inc(src,'outgoing_segment_truck_count',occupancy[rid]);inc(dst,'incoming_segment_truck_count',occupancy[rid])
    for tid,t in enumerate(trucks):
        node=onehot(t[32:41]);lt=onehot(t[26:30]);dt=onehot(t[30:32])
        if segments[tid] is None and node is not None:inc(PHYSICAL_NODE_TO_TOKEN[node],'at_node_truck_count')
        if tid!=ego and t[17]==0 and t[20]>0 and lt is not None:
            inc(9+lt,'reserved_truck_count');inc(9+lt,'reserved_tons',float(t[20])*193)
        if t[17]==1 and dt is not None:
            n=13+dt; tons=float(t[19])*193
            inc(n,'inbound_loaded_truck_count');inc(n,'inbound_loaded_tons',tons)
            if t[21]==1:
                inc(n,'inbound_known_grade_truck_count');inc(n,'inbound_known_grade_tons',tons)
                inc(n,'inbound_known_metal_tons',tons*float(t[22]))
    for sites,start,col,rel in ((loads,9,20,'queue'),(dumps,13,12,'queue'),
        (loads,9,29,'seize'),(dumps,13,21,'seize'),(dumps,13,30,'capacity_wait'),
        (loads,9,38,'service'),(dumps,13,39,'service')):
        for sid,slots in enumerate(sites[:,col:col+9]):
            ids=[wire(v) for v in slots if wire(v)>=0]
            inc(start+sid,rel+'_unique_count',len(set(ids)));inc(start+sid,rel+'_instance_count',len(ids))
    for did,v in enumerate(dumps[:,11]):
        if wire(v)>=0:inc(13+did,'admission_count')
    return q

def node_states(raw):
    nodes=[]; trucks=raw[27:1107].reshape(9,120)
    for tid,t in enumerate(trucks):
        nodes.append(dict(id=tid,name=NODE_NAMES[tid],kind='Truck',ego=bool(t[0]),
            phase_id=onehot(t[4:16]),loaded=bool(t[17]),ore=bool(t[18]),
            payload_tons=float(t[19])*193,reserved_tons=float(t[20])*193,
            grade_known=bool(t[21]),payload_grade=float(t[22]) if t[21] else None,
            load_target=onehot(t[26:30]),dump_target=onehot(t[30:32]),
            geographic_node=onehot(t[32:41]),segment=onehot(t[45:61]),
            remaining_segment_m=float(t[61])*10000,initial_predecessor=wire(t[81]) if wire(t[81])>=0 else None))
    for lid,t in enumerate(raw[1107:1295].reshape(4,47)):
        nodes.append(dict(id=9+lid,name=NODE_NAMES[9+lid],kind='Load',ore=bool(t[1]),
            grade_mean=float(t[2]),grade_std=float(t[3]),
            queue_ids=[wire(v) for v in t[20:29] if wire(v)>=0],
            seize_ids=[wire(v) for v in t[29:38] if wire(v)>=0],
            service_ids=[wire(v) for v in t[38:47] if wire(v)>=0]))
    for did,t in enumerate(raw[1295:1391].reshape(2,48)):
        nodes.append(dict(id=13+did,name=NODE_NAMES[13+did],kind='Dump',ore=bool(t[1]),
            queue_ids=[wire(v) for v in t[12:21] if wire(v)>=0],
            seize_ids=[wire(v) for v in t[21:30] if wire(v)>=0],
            capacity_wait_ids=[wire(v) for v in t[30:39] if wire(v)>=0],
            service_ids=[wire(v) for v in t[39:48] if wire(v)>=0],
            admission_id=wire(t[11]) if wire(t[11])>=0 else None))
    for k in (2,3,4): nodes.append(dict(id=15+k-2,name=f'Junction{k}',kind='Junction',geographic_node=k))
    nodes.append(dict(id=18,name='Global',kind='Global',window_index=int(round(float(raw[2])*4)),
        shift_elapsed_seconds=float(raw[0])*28800,terminal=bool(raw[17])))
    return nodes

def select_pair(paths):
    # Post-hoc display selection across two observed states only. Never used as model input.
    candidates=[]
    for path in paths:
        with h5py.File(path,'r') as h: x=h['states'][:]
        t=x[:,27:1107].reshape(-1,9,120)
        for i in range(len(x)-1):
            dt=float(x[i+1,0]-x[i,0])*28800
            if x[i,17] or x[i+1,17] or not(.01<dt<200) or not(.20<float(x[i,0])<.75): continue
            ids=np.flatnonzero((t[i,:,17]==0)&(t[i+1,:,17]==1))
            ore=ids[t[i+1,ids,18]==1]
            if len(ore):
                score=100*len(ore)+10*len(ids)+float(t[i+1,ids,19].sum())
                candidates.append((score,-int(path.parent.name[4:]),-i,path,i,ids.tolist()))
    if not candidates:raise ValueError('no adjacent valid loading transition found; no pair imputed')
    chosen=max(candidates,key=lambda z:z[:3]);path,i,ids=chosen[3:]
    rows,episodes,_,_=read_dataset(path)
    try:
        r0,r1=rows[i],rows[i+1]
        if r0['episode_id']!=r1['episode_id'] or rows.state_index[i+1]!=rows.state_index[i]+1:
            raise ValueError('selected rows not adjacent in one episode')
        manifest=json.loads((path.parent/'manifest.json').read_bytes())
        selection=dict(source_hdf5=str(path),source_sha256=sha(path),
            run_index=manifest['provenance']['run_index'],seed=manifest['provenance']['seed'],
            episode_id=r0['episode_id'],recording_id=manifest['provenance']['recording_id'],
            selection_rule='maximum ore loading transitions among adjacent valid decisions in middle 20%-75% of shift; dt in (0.01,200)s; then earliest run/index',
            posthoc_selection_for_display_only=True,loaded_transition_truck_ids=ids,
            adjacent_original_state_indices=True,adjacent_valid_decisions=True,
            ego_changes=r0['truck_id']!=r1['truck_id'],
            T0=dict(state_index=int(rows.state_index[i]),transition_index=i,time_seconds=r0['action_time_seconds'],ego_id=r0['truck_id']),
            T1=dict(state_index=int(rows.state_index[i+1]),transition_index=i+1,time_seconds=r1['action_time_seconds'],ego_id=r1['truck_id']))
        for label,idx in (('T0',i),('T1',i+1)):
            # Each action snapshot after the initial state was emitted by the previous settlement.
            info=rows[idx-1]['settlement_info'] if idx>0 else episodes[0]['start_info']
            selection[label]['decision_sequence']=info.get('decision_sequence')
            selection[label]['request_id']=info.get('request_id')
            selection[label]['settlement_snapshot_time_seconds']=info.get('sim_time_seconds')
        return selection,np.stack([r0['observation'],r1['observation']]).astype(np.float32)
    finally:rows.close()

def snapshot_json(raw, meta):
    graph=build_graph(raw[None]);records=stable_records(graph)
    return dict(**meta,raw=raw.tolist(),edges=records,
        quantities=graph.quantities[0].tolist(),raw_quantities=graph.raw_quantities[0].tolist(),
        independent_raw_quantities=independent_quantities(raw).tolist(),
        candidate_dump_ids=graph.candidate_dump_ids[0].tolist(),
        candidate_load_ids=[9,10,11,12],candidate_mask=raw[19:23].tolist(),
        candidate_tons=(raw[23:27].astype(np.float64)*193).tolist(),
        candidate_indices_are_readout_only=True,
        c128=build_features(raw,config=feature_config_for()).tolist(),nodes=node_states(raw))

def diff_records(a,b):
    maps=[{e['stable_key']:e for e in x} for x in (a,b)]
    result=[]
    for key in sorted(set(maps[0])|set(maps[1])):
        x,y=(m.get(key) for m in maps)
        changed=[n for n in EDGE_FEATURE_NAMES if x and y and x['attributes'][n]!=y['attributes'][n]]
        status='added' if x is None else 'removed' if y is None else 'changed' if changed else 'unchanged'
        for rec in (x,y):
            if rec is not None:rec['diff_status']=status;rec['changed_attributes']=changed
        result.append(dict(stable_key=key,status=status,changed_attributes=changed,T0=x,T1=y))
    return result

def output_csvs(out,snapshots,raw_schema,schema):
    features=[];counts=[];paths=[];summary={};crows=[]
    for label in ('T0','T1'):
        snap=snapshots[label];raw=np.array(snap['raw'],dtype=np.float32); records=snap['edges']
        fields=['snapshot','record_index','stable_key','source','source_name','destination','destination_name','relation','group','diff_status','changed_attributes',*EDGE_FEATURE_NAMES]
        write_csv(out/f'graph_edges_{label}.csv',fields,[dict(snapshot=label,record_index=i,
            **{k:v for k,v in e.items() if k not in ('attributes','changed_attributes')},**e['attributes'],changed_attributes=';'.join(e['changed_attributes'])) for i,e in enumerate(records)])
        # Exact schema-owned raw fields; shared FIFO belongs to crusher Dump0.
        ranges=[(n,27+120*n,27+120*(n+1)) for n in range(9)]+[(9+n,1107+47*n,1107+47*(n+1)) for n in range(4)]+[(13+n,1295+48*n,1295+48*(n+1)) for n in range(2)]+[(13,1391,1439),(18,0,19)]
        for nid,a,z in ranges:
            for j in range(a,z):
                entry=raw_schema[j]
                features.append(dict(snapshot=label,node_id=nid,node_name=NODE_NAMES[nid],feature=entry['name'],source_index=j,
                    raw_schema_value=float(raw[j]),provenance='original_observation',formula=entry.get('formula',''),meaning=entry.get('meaning',''),learned_feature=False))
        # Raw road attributes are incident edge inputs, not 16 learned node tokens.
        roads=raw[1439:1519].reshape(16,5)
        for n,k in enumerate((2,3,4),15):
            incoming=[i for i,p in enumerate(ROAD_ENDPOINTS) if p[1]==k];outgoing=[i for i,p in enumerate(ROAD_ENDPOINTS) if p[0]==k]
            for name,value,formula in (
                ('incoming_road_count',len(incoming),'count ROAD_ENDPOINTS destination'),
                ('outgoing_road_count',len(outgoing),'count ROAD_ENDPOINTS source'),
                ('incoming_length_sum_m',float(roads[incoming,2].astype(np.float64).sum()*10000),'sum incident original road.length *10000'),
                ('outgoing_length_sum_m',float(roads[outgoing,2].astype(np.float64).sum()*10000),'sum incident original road.length *10000')):
                features.append(dict(snapshot=label,node_id=n,node_name=NODE_NAMES[n],feature=name,source_index='incident_road_rows',raw_schema_value=value,
                    provenance='deterministic_junction_description_not_learned64',formula=formula,meaning='No standalone raw Junction block; formal encoder combines incident road/vehicle content',learned_feature=False))
        for nid in range(19):
            for j,name in enumerate(QUANTITY_NAMES):
                counts.append(dict(snapshot=label,node_id=nid,node_name=NODE_NAMES[nid],quantity_index=j,quantity=name,
                    raw=float(snap['raw_quantities'][nid][j]),scaled=float(snap['quantities'][nid][j]),divide_raw_by=QUANTITY_SCALES[j],
                    independently_rebuilt_raw=snap['independent_raw_quantities'][nid][j],source_rule=schema['quantities'][j]['source']))
        incoming=defaultdict(list)
        for e in records:incoming[e['destination']].append(e)
        target_summaries=[]
        for target in range(9,15):
            direct=incoming[target];n2=0;truckdump=[]
            for e in direct:
                paths.append(path_row(label,target,[e]))
                for first in incoming[e['source']]:
                    paths.append(path_row(label,target,[first,e]));n2+=1
                    if first['source']<9 and e['source'] in (13,14) and target in range(9,13) and first['relation']=='loaded_dump' and e['relation']=='compatible_material_reverse':
                        truckdump.append([first['source'],e['source'],target])
            target_summaries.append(dict(target=NODE_NAMES[target],direct_record_count=len(direct),exact_two_edge_walk_count=n2,
                truck_dump_load_paths=[[NODE_NAMES[v] for v in p] for p in truckdump],
                missing_loaded_dump_trucks=[n['name'] for n in snap['nodes'][:9] if n['loaded'] and n['dump_target'] is None],
                missing_historical_load_origin=[n['name'] for n in snap['nodes'][:9] if n['loaded'] and n['load_target'] is None],
                no_candidate_edges=True))
        summary[label]=target_summaries
    for j,name in enumerate(FEATURE_NAMES):
        a=snapshots['T0']['c128'][j];b=snapshots['T1']['c128'][j]
        crows.append(dict(index=j,name=name,partition='task_common_88' if j<88 else f'candidate_Load{(j-88)//10}',T0=a,T1=b,delta=b-a,changed=a!=b))
    write_csv(out/'node_features.csv',['snapshot','node_id','node_name','feature','source_index','raw_schema_value','provenance','formula','meaning','learned_feature'],features)
    write_csv(out/'count_features.csv',['snapshot','node_id','node_name','quantity_index','quantity','raw','scaled','divide_raw_by','independently_rebuilt_raw','source_rule'],counts)
    write_csv(out/'c128_diff.csv',['index','name','partition','T0','T1','delta','changed'],crows)
    write_csv(out/'two_hop_paths.csv',['snapshot','target_id','target_name','hops','nodes','source_entities','relations','groups','edge_keys','edge_attributes','edge_diff_status','return_to_source','scope'],paths)
    write_json(out/'path_summary.json',dict(scope='direct inbound records and every exact two-edge directed walk, including return walks; no learned forward',targets=summary,
        missing_relations=[dict(snapshot=label,source=n['name'],relation='historical_load_origin',reason='loaded truck loadTarget cleared; never inferred')
            for label in ('T0','T1') for n in snapshots[label]['nodes'][:9] if n['loaded'] and n['load_target'] is None],
        candidate_edges='absent by design; readout binding only; not missing factual commitments'))

def path_row(label,target,edges):
    nodes=[edges[0]['source']]+[e['destination'] for e in edges]
    return dict(snapshot=label,target_id=target,target_name=NODE_NAMES[target],hops=len(edges),
        nodes=' -> '.join(NODE_NAMES[n] for n in nodes),source_entities=';'.join(NODE_NAMES[n] for n in nodes),
        relations=' -> '.join(e['relation'] for e in edges),groups=' -> '.join(e['group'] for e in edges),
        edge_keys=';'.join(e['stable_key'] for e in edges),edge_attributes=json.dumps([e['attributes'] for e in edges],sort_keys=True),
        edge_diff_status=';'.join(e['diff_status'] for e in edges),return_to_source=nodes[0]==nodes[-1],scope='topology only; no model forward')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=DEFAULT_SOURCE);p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT);p.add_argument('--chunk',type=int,default=16);args=p.parse_args()
    if Path(sys.executable).resolve()!=Path('/home/jinx/miniconda3/envs/anylogic-env/bin/python').resolve():raise RuntimeError('anylogic-env required')
    torch.set_num_threads(2)
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    if (out/'selection.json').exists():raise RuntimeError('use a fresh output directory; previous audit is immutable')
    source_files=sorted(args.source.glob('run_*/samples.h5'))
    sources=[dict(path=str(path),sha256=sha(path)) for f in source_files for path in (f,f.parent/'manifest.json')]
    code_files=sorted((PROJECT/'business_graph_summary_v1').glob('*.py'))+[SCHEMA,RAW_SCHEMA,Path(__file__)]
    code_pins=[dict(path=str(f),sha256=sha(f)) for f in code_files]
    missing=[]
    if len(source_files)!=32:missing.append(f'expected32 HDF5, got {len(source_files)}')
    if sha(CORE)!=CORE_SHA:missing.append('protected PPO SHA differs')
    schema=json.loads(SCHEMA.read_bytes());raw_schema=json.loads(RAW_SCHEMA.read_bytes())
    if schema['graph_contract']!=ablation_contract():missing.append('frozen schema/formal graph contract mismatch')
    inspections=[]
    for f in source_files:
        with h5py.File(f,'r') as h:
            required=['states','state_index','episode_index','action_info_reference','transitions','labels/mc_reward','labels/mc_grade','episode_ledger','window_ledger']
            absent=[k for k in required if k not in h]
            if absent:missing.append(dict(path=str(f),missing=absent));continue
            if h['states'].shape[1:]!=(1519,) or h['states'].dtype!=np.dtype('float32'):missing.append(dict(path=str(f),error='states must float32 Nx1519'))
            inspections.append(dict(path=str(f),states=list(h['states'].shape),transitions=int(h.attrs['transitions']),complete=bool(h.attrs.get('complete',False)),schema_version=str(h.attrs['schema_version'])))
    preflight=dict(scope='read historical32 shifts only; zero model instantiation/forward/optimizer/RPC/AnyLogic/budget',
        expected_shifts=32,available_shifts=len(source_files),missing_fields=missing,
        complete_input=not missing,schema_path=str(SCHEMA),schema_canonical_location='change_plans/.../business_graph_summary_009/schema.json; no package/schema.json file',
        node_count=19,road_relation_records_per_graph=16,total_relation_vocabulary=25,
        vocabulary_note='25 includes road; 24 other directed relation types; a snapshot need not contain all25',
        actor_critic_inputs='same deterministic raw/graph/C inputs; independent learned encoders are not instantiated',
        C128_feature_contract=feature_contract(feature_config_for()),inspections=inspections,code_pins=code_pins)
    write_json(out/'preflight.json',preflight)
    if missing:raise RuntimeError('missing required input/schema: '+repr(missing))
    selection,pair=select_pair(source_files);write_json(out/'selection.json',selection)
    snaps=dict(selection=selection,**{label:snapshot_json(pair[i],selection[label]) for i,label in enumerate(('T0','T1'))})
    snaps['diff']=diff_records(snaps['T0']['edges'],snaps['T1']['edges'])
    snaps['diff_policy']='exact original float32 edge attributes; stable relation/source/destination/ordered-slot/duplicate-instance key; no tolerance; no node-pair collapse'
    write_json(out/'snapshots.json',snaps)
    print('SNAPSHOTS_READY '+str(out/'snapshots.json'),flush=True)
    output_csvs(out,snaps,raw_schema,schema)
    audits=[];totals=Counter();started=time.monotonic()
    for f in source_files:
        rows,episodes,_,_=read_dataset(f)
        with h5py.File(f,'r') as h:x=h['states'][:]
        valid=np.flatnonzero((x[:,17]==0)&(x[:,19:23].sum(-1)>0))
        counts=Counter();maxerr=0.;duplicate_instances=0;unknown_loaded=0;repeated_memberships=0
        times=[]
        for idx in range(len(rows)):
            row=rows[idx];times.append(row['action_time_seconds'])
            if not np.array_equal(row['observation'],x[int(rows.state_index[idx])]):raise ValueError('decoded/raw observation mismatch')
            if row['episode_id']!=selection['episode_id'] and len(episodes)!=1:raise ValueError('unexpected episode inventory')
        if len(episodes)!=1 or np.any(np.diff(times)<0):raise ValueError('non-monotone/split shift')
        for offset in range(0,len(valid),args.chunk):
            indices=valid[offset:offset+args.chunk];batch=x[indices].copy();g=build_graph(batch)
            for local,index in enumerate(indices):
                records=g.records(local);counts.update(e['relation'] for e in records)
                if sum(e['relation']=='road' for e in records)!=16:raise ValueError('road row count differs')
                q=independent_quantities(batch[local]);err=float(np.max(np.abs(q-g.raw_quantities[local].numpy())))
                maxerr=max(maxerr,err)
                if err>0.001:raise ValueError('quantity oracle mismatch '+str(err))
                dup=Counter((e['source'],e['destination'],e['relation'],json.dumps(e['attributes'],sort_keys=True)) for e in records)
                duplicate_instances+=sum(v-1 for v in dup.values() if v>1)
                trucks=batch[local,27:1107].reshape(9,120)
                unknown_loaded+=sum(t[17]==1 and onehot(t[30:32]) is None for t in trucks)
                # Relation overlap is reported, never added into reserved/inbound tonnage.
                memberships=defaultdict(set)
                for e in records:
                    if e['group']=='service':
                        memberships[(min(e['source'],e['destination']),max(e['source'],e['destination']))].add(e['relation'].removesuffix('_reverse'))
                repeated_memberships+=sum(len(v)>1 for v in memberships.values())
        entry=dict(path=str(f),run_index=int(f.parent.name[4:]),states=len(x),transitions=len(rows),valid_decisions=len(valid),
            terminal_states=int(np.sum(x[:,17]==1)),time_monotone=True,same_single_episode=True,
            finite_original_observations=bool(np.isfinite(x).all()),all_valid_graphs_constructed=True,
            observed_relations=dict(counts),max_raw_quantity_oracle_absolute_error=maxerr,
            exact_duplicate_relation_instances=duplicate_instances,multigroup_service_pairs=repeated_memberships,
            loaded_without_dump_target_instances=int(unknown_loaded),candidate_edges=0,candidate_virtual_tons=0)
        audits.append(entry);totals.update(counts);rows.close()
        print(f'GRAPH_AUDIT {len(audits)}/32 decisions={len(valid)} max_quantity_error={maxerr:.8g}',flush=True)
    # Selected candidate readout changes cannot affect any factual graph tensor.
    candidate_invariance=[]
    for raw in pair:
        changed=raw.copy();changed[23:27]*=.5
        a,b=build_graph(raw[None]),build_graph(changed[None])
        candidate_invariance.append(all(torch.equal(getattr(a,k),getattr(b,k)) for k in ('edge_index','edge_type','edge_group','edge_attr','quantities','candidate_dump_ids','ego_ids')))
    for pin in sources+code_pins:
        if sha(pin['path'])!=pin['sha256']:raise ValueError('read-only source changed: '+pin['path'])
    if sha(CORE)!=CORE_SHA:raise ValueError('protected core changed')
    audit=dict(shifts=len(audits),valid_decisions=sum(a['valid_decisions'] for a in audits),
        transitions=sum(a['transitions'] for a in audits),all_input_complete=True,all_graphs_constructed=True,
        observed_relation_counts=dict(totals),relation_vocabulary_size=25,checks=dict(
            formal_schema_exact=True,node_order_19=True,road_order16=True,all_quantity_oracle_checks=True,
            candidate_tons_perturbation_leaves_factual_graph_unchanged=all(candidate_invariance),
            original_history_bytes_unchanged=True,formal_source_bytes_unchanged=True,protected_core_unchanged=True),
        count_note='Repeated service instances/multigroup memberships are retained as distinct facts; commitment tonnes count each qualifying Truck once only.',
        missing_note='Unknown targets remain absent. Historical loaded origin is not recoverable from cleared loadTarget. Junction neural64 is intentionally not evaluated.',
        source_pins=sources,code_pins=code_pins,protected_core_sha256=CORE_SHA,per_shift=audits,
        model_instantiations=0,model_forwards=0,optimizer_updates=0,new_physical_shifts=0,rpc_started=False,budget_registered=False,
        elapsed_seconds=time.monotonic()-started)
    write_json(out/'32shift_build_audit.json',audit)
    print('COMPLETE '+str(out),flush=True)

if __name__=='__main__':main()
