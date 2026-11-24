"""Build observation-consistent exogenous branches of authenticated historical T1.

This script does not simulate events, train, call RPC, or write any historical
artifact.  A is a completed-load boundary before the next travel state starts;
B transfers a not-yet-serviced reservation while keeping its current road;
C changes current crusher capacity, not previously processed material.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT))
import h5py
import numpy as np
from relation_gat_ego_v1.adapter import validate_states

INPUT = PROJECT / 'reports/token_attention_001/business_graph_summary_009/historical_snapshot_001'
OUT = PROJECT / 'reports/token_attention_001/business_graph_summary_009/physical_intervention_001'
ALP = PROJECT / 'anylogic/3d/DRL-PPO-baseline1/DRL-PPO-baseline1.alp'
SCHEMA = PROJECT / 'python/observation_schema.json'
CORE = PROJECT.parents[1] / 'python/episode_budget_ppo.py'
CORE_SHA = '8e4cfa31c622f5e33756e304eea3f7526547db72ae93f7d9f00808845d61ef96'
SEED = 2026111801
ROAD_ENDPOINTS = ((0,2),(5,3),(6,4),(7,4),(8,3),(1,3),(2,0),(2,3),
                  (3,5),(3,8),(3,1),(3,2),(3,4),(4,6),(4,7),(4,3))
TOL_TONS = .006  # float32 wire values normalized by 30000 t.


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def raw_sha(raw):
    return hashlib.sha256(np.ascontiguousarray(raw, dtype=np.float32).tobytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n')


def onehot(v):
    return int(v.argmax()) if v.sum() else None


def ids(v):
    return [int(round(float(a)*9))-1 for a in v if a > 0]


def blocks(raw):
    return (raw[27:1107].reshape(9,120), raw[1107:1295].reshape(4,47),
            raw[1295:1391].reshape(2,48), raw[1439:].reshape(16,5))


def authenticate():
    obj = json.loads((INPUT / 'snapshots.json').read_text())
    selection = obj['selection']
    assert sha(selection['source_hdf5']) == selection['source_sha256']
    with h5py.File(selection['source_hdf5'], 'r') as h:
        raw = h['states'][selection['T1']['state_index']]
    assert raw.dtype == np.float32 and raw.shape == (1519,)
    assert np.array_equal(raw, np.asarray(obj['T1']['raw'], dtype=np.float32))
    assert sha(CORE) == CORE_SHA
    # Exact Java snippets pin the semantics used below, not an ALP modification.
    code = ALP.read_text()
    snippets = {
        'reservation_only': 'reservedTons += tons;\nincomingTrucks++;',
        'arrival_removes_incoming': 'incomingTrucks--;',
        'completed_source': 'remainingTons = Math.max(0.0, remainingTons - tons);',
        'completed_reservation': 'reservedTons = Math.max(0.0, reservedTons - tons);',
        'completed_production': 'producedTons += tons;',
        'load_completion': 'plannedPayloadTons = 0.0;\nisLoaded = true;',
        'loading_gate': 'boolean loading=!isLoaded && "LOADING".equals(executionPhase);',
        'phase_other': 'int phase=11;',
        'known_grade_gate': 'w.add(p+"gradeKnown",isLoaded?1:0);',
        'allocation_at_service_start': 't.obsLoadMaterialStart = extractionCursorTons;\nt.payloadGrade = takeMaterialGrade(t.plannedPayloadTons);',
        'unloading_buffer_reservation': 'crusherReservedTons += t.payloadTons;',
        'rate_observation': 'w.add(p+"crusherRate",acceptsOre?crusherRateTph/2500.0:0.0);',
        'no_repeated_road': 'Route repeats a segment; change encoding',
        'motion_reset': 'obsMotionSamples.clear();',
    }
    for name, snippet in snippets.items():
        assert snippet in code, 'Java source contract differs: ' + name
    return raw.copy(), selection, {name: True for name in snippets}


def refresh_action_sidecar(raw):
    trucks, loads, _, _ = blocks(raw)
    ego = int(trucks[:,0].argmax())
    capacity = float(trucks[ego,16])*193
    waste_committed = float(raw[9])*4000
    waste_committed += sum(float(l[6])*30000 for l in loads if l[1] == 0)
    waste_committed += sum(float(t[19])*193 for t in trucks if t[17] == 1 and t[18] == 0)
    for a, l in enumerate(loads):
        available = max(0., float(l[5]-l[6])*30000) if l[0] else 0.
        q = min(capacity, available) if not raw[17] else 0.
        if not l[1] and waste_committed >= 6000-1e-6:
            q = 0.
        raw[19+a] = float(q > 1e-6)
        raw[23+a] = q/193 if q > 1e-6 else 0.


def scenario_a(base):
    raw = base.copy(); trucks, loads, _, _ = blocks(raw)
    tid, lid = 1, 0; t, l = trucks[tid], loads[lid]
    assert t[0] == 0 and t[17] == 0 and onehot(t[4:16]) == 4
    assert onehot(t[26:30]) == lid and t[20] > 0
    assert t[24] == 0 and t[25] == 1 and t[23] > 0
    assert ids(l[38:47]) == [tid] and l[15] == l[16] == 1
    assert not ids(l[20:38]) and onehot(t[45:61]) is None
    q, grade = float(t[20])*193, float(t[23])
    start, block_tons = float(t[86])*30000, float(l[10])*500
    assert abs(start-float(l[8])*30000) < TOL_TONS
    assert int(start//block_tons) == int((start+q-1e-6)//block_tons)
    assert int(round(float(l[12])*60))-1 == int(start//block_tons)
    assert abs(grade-float(l[13])) < 1e-7
    # Complete the already allocated parcel; no further material allocation.
    l[5] = float(l[5])-q/30000
    l[6] = float(l[6])-q/30000
    # (remaining-q) - (reserved-q) equals the original available amount.
    # Preserve that already serialized value instead of creating a one-ULP
    # nuisance change by subtracting two separately rounded float32 fields.
    l[8] = float(l[8])+q/30000
    l[15:17] = 0; l[38:47] = 0
    t[4:16] = 0; t[4+11] = 1
    t[17] = 1; t[18] = 1; t[19] = t[20]; t[20] = 0
    t[21] = 1; t[22] = grade; t[23:30] = 0
    t[30:32] = (1,0)
    t[82:89] = 0  # no waiting/loading/dumping in completed-load boundary.
    refresh_action_sidecar(raw)
    return raw, dict(
        label='A_complete_one_known_load_at_current_site', subject_truck_id=tid,
        old_load=lid, new_load=None, dump_id=0, payload_tons=q,
        payload_grade=grade, grade_source='T1 truck.1.loadingGradeMean; inferred=1 and std=0',
        observation_boundary='after onLoadingFinished; before next travel-state entry',
        phase_id_before=4, phase_id_after=11, physically_on_road=False,
        statement='One existing vehicle becomes loaded with Dump0 destination; fleet stays nine.',
        exogenous_setting='Force completion of the current known parcel at unchanged timestamp; no simulation event is executed.',
        constraints=['No new unknown grade is drawn or obtained from a future row.',
            'Load0 cursor was advanced when service began; it is not advanced again.',
            'Crusher buffer reservation starts at unloading service, not loading completion; it remains zero.',
            'Phase11 is the schema OTHER result while executionPhase remains LOADING and isLoaded becomes true.',
            'The pending next statechart event is not executed; road and motion history remain unchanged.'],
        limitation='Observation-level completed-load boundary, not proof that a production RPC decision can pause at this intermediate event.')


def scenario_b(base):
    raw = base.copy(); trucks, loads, _, roads = blocks(raw)
    tid, old, new = 8, 0, 1; t = trucks[tid]
    assert t[0] == 0 and t[17] == 0 and onehot(t[4:16]) == 2
    assert onehot(t[26:30]) == old and onehot(t[45:61]) == 7
    assert loads[old,1] == loads[new,1] == 1 and t[20] > 0
    assert all(tid not in ids(l[20:47]) for l in loads)
    q = float(t[20])*193
    assert float(loads[new,7])*30000 >= q and float(loads[old,17])*9 >= 1-1e-6
    t[26:30] = 0; t[26+new] = 1
    route = [7,12,13]
    t[63:79] = 0
    for i,r in enumerate(route,1): t[63+r] = i/16
    t[62] = float(t[61]) + sum(float(roads[r,2]) for r in route[1:])
    for lid, sign in ((old,-1),(new,1)):
        loads[lid,6] = float(loads[lid,6]) + sign*q/30000
        loads[lid,7] = max(0.,float(loads[lid,5])-float(loads[lid,6]))
        loads[lid,17] = float(loads[lid,17]) + sign/9
        if abs(float(loads[lid,17])) < 1e-7: loads[lid,17] = 0
    refresh_action_sidecar(raw)
    return raw, dict(
        label='B_transfer_unstarted_reservation_Load0_to_Load1', subject_truck_id=tid,
        old_load=old, new_load=new, payload_tons=0., reserved_tons=q,
        route_before=[7,8], route_after=route, current_segment=7,
        statement='Transfer the existing 193 t reservation and future road tail; preserve current road position.',
        exogenous_setting='Atomic reassignment of an empty truck that has not arrived or entered loading service.',
        constraints=['Both targets are ore; no waste commitment or material-type change.',
            'ReserveLoad changes reservedTons/incomingTrucks, not extractionCursorTons.',
            'No source or target FIFO/service list includes Truck8, so no list edit is required.',
            'Current segment7 and past movement remain unchanged; only the not-yet-travelled route tail changes.',
            'Stored FIFO, cumulative production, source material cursors and known grades remain unchanged.'],
        limitation='The current simulator does not expose this reassignment as a normal PPO action; this is an exogenous raw-observation branch, not an executed policy transition.')


def scenario_c(base):
    raw = base.copy(); assert raw[1304] == 1
    raw[1304] = .5
    return raw, dict(
        label='C_Dump0_instant_processing_derating', dump_id=0,
        crusher_rate_tph_before=2500., crusher_rate_tph_after=1250.,
        statement='Reduce current Dump0 crusher processing rate by 50% at the same time.',
        exogenous_setting='Positive current processing-capacity parameter change; offline only.',
        constraints=['No past processed tons/metal, FIFO, buffer, reservation, queue or truck state is changed.',
            'The future processing projection in C128 must be recomputed from this modified observation.'],
        limitation='Current runtime configuration remains 2500 tph; this is not a deployed parameter change or a historical observation.')


def audit(raw, base):
    validate_states(raw[None])
    trucks, loads, dumps, roads = blocks(raw)
    bt, bl, bd, _ = blocks(base)
    results = []
    def check(name, passed, **details):
        results.append(dict(name=name,passed=bool(passed),**details))
    def close(name, observed, expected, tol=TOL_TONS):
        check(name,abs(observed-expected)<=tol,observed=float(observed),expected=float(expected),tolerance=tol)
    check('raw_schema_validation',True)
    check('fleet_is_exactly_nine',trucks.shape==(9,120))
    check('same_ego',np.array_equal(trucks[:,0],bt[:,0]))
    check('same_time_and_accumulated_global_history',np.array_equal(raw[:19],base[:19]))
    check('same_motion_and_free_speed_history',np.array_equal(trucks[:,89:],bt[:,89:]))
    check('same_initial_dispatch_history',np.array_equal(trucks[:,80:82],bt[:,80:82]))
    check('same_fifo',np.array_equal(raw[1391:1439],base[1391:1439]))
    check('same_geographic_positions',np.array_equal(trucks[:,32:43],bt[:,32:43]))
    check('road_configuration_unchanged',np.array_equal(roads,blocks(base)[3]))
    for i,l in enumerate(loads):
        reserved=sum(float(t[20])*193 for t in trucks if onehot(t[26:30])==i)
        close(f'Load{i}_reservations_match_trucks',float(l[6])*30000,reserved)
        close(f'Load{i}_initial_equals_remaining_plus_produced',float(l[4])*30000,float(l[5]+l[8])*30000)
        close(f'Load{i}_available_equals_remaining_minus_reserved',float(l[7])*30000,
              max(0.,float(l[5]-l[6])*30000) if l[0] else 0.)
        check(f'Load{i}_nonnegative_and_reserved_within_remaining',
              bool((l[[4,5,6,7,8]] >= -1e-7).all() and l[6]<=l[5]+1e-7))
        service=ids(l[38:47])
        close(f'Load{i}_service_count',float(l[16]),len(service),tol=1e-6)
        close(f'Load{i}_busy_matches_service_count_for_this_fixture',float(l[15]),len(service),tol=1e-6)
        active_tons=sum(float(trucks[t,20])*193 for t in service)
        close(f'Load{i}_cursor_equals_produced_plus_allocated_service',float(l[9])*30000,
              float(l[8])*30000+active_tons)
        incoming=sum(onehot(t[4:16])==2 and onehot(t[26:30])==i for t in trucks)
        close(f'Load{i}_incoming_matches_to_load',float(l[17])*9,incoming,tol=2e-6)
        for tid in service:
            check(f'Load{i}_service_Truck{tid}_coherent',trucks[tid,17]==0 and
                  onehot(trucks[tid,26:30])==i and onehot(trucks[tid,4:16])==4)
    for tid,t in enumerate(trucks):
        lt,dt,seg=onehot(t[26:30]),onehot(t[30:32]),onehot(t[45:61])
        if t[17]:
            check(f'Truck{tid}_loaded_fields',t[19]>0 and t[20]==0 and t[21]==1
                  and lt is None and dt==(0 if t[18] else 1) and 0<=t[22]<=1)
        else:
            check(f'Truck{tid}_unloaded_fields',t[19]==0 and t[21]==0 and t[22]==0 and dt is None)
        ranks=np.rint(t[63:79]*16).astype(int)
        route=sorted(np.flatnonzero(ranks).tolist(),key=lambda r:ranks[r])
        if seg is None:
            check(f'Truck{tid}_no_segment_route',not route and t[61]==t[62]==t[79]==0)
        else:
            check(f'Truck{tid}_route_order',bool(route and route[0]==seg and
                  [ranks[r] for r in route]==list(range(1,len(route)+1))))
            check(f'Truck{tid}_route_contiguous',all(ROAD_ENDPOINTS[a][1]==ROAD_ENDPOINTS[z][0]
                  for a,z in zip(route,route[1:])))
            close(f'Truck{tid}_route_remaining_m',float(t[62])*10000,
                  (float(t[61])+sum(float(roads[r,2]) for r in route[1:]))*10000,tol=.003)
            check(f'Truck{tid}_segment_distance_in_bounds',0<=t[61]<=roads[seg,2]+1e-7)
            destination=(5+lt) if lt is not None else dt
            check(f'Truck{tid}_route_reaches_current_target',destination is not None and
                  ROAD_ENDPOINTS[route[-1]][1]==destination)
    for did,d in enumerate(dumps):
        if d[1]:
            close(f'Dump{did}_buffer_capacity',float(d[8])*500,float(d[5]+d[6]+d[7])*500)
            unload_tons=sum(float(trucks[t,19])*193 for t in ids(d[39:48]))
            close(f'Dump{did}_buffer_reserved_only_current_unloading',float(d[6])*500,unload_tons)
            check(f'Dump{did}_positive_crusher_rate',d[9]>0)
        close(f'Dump{did}_service_count',float(d[4])*3,len(ids(d[39:48])),tol=1e-6)
    ore_produced=sum(float(l[8])*30000 for l in loads if l[1])
    ore_accounted=float(raw[7])*20000+float(dumps[0,5])*500+sum(float(t[19])*193 for t in trucks if t[17] and t[18])
    close('ore_mass_produced_equals_processed_stored_and_loaded',ore_produced,ore_accounted,tol=.015)
    refreshed=raw.copy(); refresh_action_sidecar(refreshed)
    check('action_mask_and_tons_recomputed',np.array_equal(raw[19:27],refreshed[19:27]))
    check('all_checks_passed',all(r['passed'] for r in results))
    return results


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    base, selection, java_checks=authenticate()
    raw_schema=json.loads(SCHEMA.read_text())
    metadata=dict(schema='bg128-physical-observation-interventions-v1',seed=SEED,
        experiment_type='offline exogenous current-observation branches',
        training=False,optimizer_updates=0,rpc_started=False,anylogic_started=False,new_shifts=0,
        source_references=[dict(path=str(p),sha256=sha(p)) for p in
                           (SCHEMA,ALP,CORE,INPUT/'snapshots.json',INPUT/'selection.json')],
        base=dict(label='T1',**selection['T1'],source_hdf5=selection['source_hdf5'],
                  source_sha256=selection['source_sha256'],raw=base.tolist(),raw_sha256=raw_sha(base)),
        scenarios={})
    checks={'scope':'observable-field constraints; not a simulator reachability proof',
            'java_semantic_snippet_checks':java_checks,'base':audit(base,base),'scenarios':{}}
    changes=[]
    for name,builder in (('A',scenario_a),('B',scenario_b),('C',scenario_c)):
        raw, intervention=builder(base)
        scenario_checks=audit(raw,base)
        changed=np.flatnonzero(raw!=base).tolist()
        metadata['scenarios'][name]=dict(label=intervention['label'],raw=raw.tolist(),
            raw_sha256=raw_sha(raw),intervention=intervention,changed_indices=changed,
            changed_field_count=len(changed),all_physical_checks_passed=all(x['passed'] for x in scenario_checks))
        checks['scenarios'][name]=scenario_checks
        for j in changed:
            f=raw_schema[j]
            changes.append(dict(scenario=name,index=j,name=f['name'],before=float(base[j]),
                after=float(raw[j]),delta=float(raw[j])-float(base[j]),
                formula=f.get('formula',''),meaning=f.get('meaning','')))
    checks['protected_core_sha256_after']=sha(CORE)
    checks['all_passed']=all(x['passed'] for x in checks['base']) and all(
        x['passed'] for rows in checks['scenarios'].values() for x in rows)
    assert checks['protected_core_sha256_after']==CORE_SHA
    write_json(OUT/'scenarios.json',metadata)
    write_json(OUT/'physical_checks.json',checks)
    with (OUT/'raw_changes.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(changes[0]));w.writeheader();w.writerows(changes)
    print(json.dumps(dict(base='T1',changed_fields={k:v['changed_field_count'] for k,v in metadata['scenarios'].items()},
                         all_passed=checks['all_passed'],output=str(OUT)),ensure_ascii=False))
    if not checks['all_passed']:
        failures={k:[x for x in rows if not x['passed']] for k,rows in
                  [('base',checks['base']),*checks['scenarios'].items()]}
        raise AssertionError(json.dumps(failures,ensure_ascii=False))


if __name__=='__main__':main()
