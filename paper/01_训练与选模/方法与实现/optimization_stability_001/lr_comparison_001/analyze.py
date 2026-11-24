"""Paired same-u0/same-seed LR comparison; immutable physical tapes only.

Never starts a collector, simulator, listener, optimizer or second controller.
Primary bootstrap resamples complete shift pairs, retaining their four windows.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT))
import h5py
import numpy as np

from business_graph_stability_campaign_v1.plan_io import reference,write
from business_graph_summary_storage_v1.episode import _read_table
from grade_event_memory_v1.events import Event,build_memory
from business_graph_summary_v1.targets import build_composite_targets
from single_critic_task_query_v1.evaluation import _physical,_aggregate_critic_metrics

PLAN=Path(__file__).resolve().parent
REPORT=ROOT/'reports/token_attention_001/business_graph_summary_009/optimization_stability_001/lr_comparison_001'
OLD=ROOT/'runs/business-graph-summary-best-checkpoint-eval100-20261118-001'
NEW=ROOT/'runs/business-graph-stability-paired-eval100-20261118-001'
OLD_SUITE=ROOT/'change_plans/token_attention_001/business_graph_summary_009/checkpoint_evaluation_001/prepared_001/suite.json'
NEW_SUITE=PLAN.parent/'paired_evaluation_001/prepared_001/suite.json'
BOOTSTRAPS=20000
SEED=2026111803


def read(path):return json.loads(Path(path).read_bytes())


def load_physical():
    from business_graph_stability_evaluation_v1.gate import check_jobs,check_sources
    from business_graph_stability_evaluation_v1.budget import connection,_payload,_state
    suite=read(NEW_SUITE);old_suite=read(OLD_SUITE)
    check_jobs(suite);check_sources(suite['source_pins'])
    assert suite['checkpoint_update']==old_suite['checkpoint_update']==13
    assert suite['sampling']==old_suite['sampling']=='stochastic_private_rng'
    before=reference(suite['evaluation_budget_path'])
    with connection(suite['evaluation_budget_path'],readonly=True) as con:
        entry=_payload(con.execute('SELECT payload,checksum FROM campaigns WHERE id=?',(suite['campaign_id'],)).fetchone())
        state=_state(con,suite,entry)
    assert Counter(r['status'] for r in state['jobs'].values())=={'COMPLETE':100}
    old_report=read(OLD/'evaluation_complete.json')
    old_records=old_report['per_episode'];records=[];sources=[]
    for ordinal,(job,old_job) in enumerate(zip(suite['jobs'],old_suite['jobs'])):
        assert all(job[k]==old_job[k] for k in ('experiment_seed','actor_sampling_seed','actor_seed'))
        assert job['run_index']!=old_job['run_index'] and job['recording_id']!=old_job['recording_id']
        row=state['jobs'][str(job['run_index'])];receipt=row['receipt'];refs=receipt['references']
        assert row['identity']==job and receipt['normal_completion'] and receipt['four_windows']==4
        assert refs['config']==suite['phase_config']
        for key in ('episode','episode_manifest','episode_seal','journal'):
            assert reference(refs[key]['path'])==refs[key]
        manifest=read(refs['episode_manifest']['path']);seal=read(refs['episode_seal']['path'])
        assert manifest['hdf5_sha256']==seal['hdf5_sha256']==refs['episode']['sha256']
        assert seal['manifest_sha256']==refs['episode_manifest']['sha256']
        with h5py.File(refs['episode']['path'],'r') as h:
            identity=json.loads(h.attrs['identity_json'])
            assert identity==receipt['identity'] and identity['environment_seed']==job['experiment_seed']
            assert identity['actor_sampling_seed']==job['actor_sampling_seed'] and identity['split']=='validation'
            memory=build_memory(_read_table(h,'material_events',Event))
            evaluation=json.loads(h.attrs['evaluation_json']);wait=h['transitions/wait_delta_seconds'][...]
            targets=build_composite_targets(memory,wait_deltas=wait,ore_tons=evaluation['ore_tons'],
                waste_tons=evaluation['waste_tons'],total_wait_seconds=evaluation['total_wait_seconds'])
            episode=dict(identity=identity,memory=memory,evaluation=evaluation,targets=targets,
                wait_deltas=wait,states=h['states'][...],actions=h['transitions/action'][...],
                composite_contract=json.loads(h.attrs['composite_contract_json']))
            physical=_physical(episode)
            records.append(dict(identity=identity,physical=physical,transitions=len(episode['actions'])))
        assert old_records[ordinal]['identity']['environment_seed']==identity['environment_seed']
        assert old_records[ordinal]['identity']['actor_sampling_seed']==identity['actor_sampling_seed']
        sources.append(dict(run_index=job['run_index'],references=refs))
    assert reference(suite['evaluation_budget_path'])==before
    return old_records,records,dict(old_suite=reference(OLD_SUITE),new_suite=reference(NEW_SUITE),
        old_evaluation=reference(OLD/'evaluation_complete.json'),new_physical_sources=sources,
        sidecar=before,paired_count=100,sidecar_state='COMPLETE100',new_optimizer_steps=0,new_physical_shifts=0)


def rmse_components(records):
    sse=[];counts=[]
    for row in records:
        values=[(w['grade']-.27)**2 for w in row['physical']['windows'] if w['grade'] is not None]
        sse.append(math.fsum(values));counts.append(len(values))
    return np.asarray(sse),np.asarray(counts)


def paired_statistics(old,new):
    assert len(old)==len(new)==100
    rng=np.random.default_rng(SEED);ix=rng.integers(0,100,size=(BOOTSTRAPS,100))
    results={}
    keys=['complete_task','all_four_windows_compliant','ore_target_met','waste_target_met',
          'total_wait_seconds','combined_cost','grade_cost','production_cost','waiting_cost','ore_tons','waste_tons']
    for key in keys:
        a=np.asarray([r['physical'][key] for r in old],dtype=float)
        b=np.asarray([r['physical'][key] for r in new],dtype=float)
        delta=b-a;draw=delta[ix].mean(1)
        item=dict(old=float(a.mean()),new=float(b.mean()),new_minus_old=float(delta.mean()),
            paired_bootstrap_95ci=np.quantile(draw,[.025,.975]).tolist(),
            pairs_new_lower=int((delta<0).sum()),pairs_new_higher=int((delta>0).sum()),ties=int((delta==0).sum()))
        if key in keys[:4]:
            old_only=int(((a==1)&(b==0)).sum());new_only=int(((a==0)&(b==1)).sum());discord=old_only+new_only
            p=min(1.,2*sum(math.comb(discord,k) for k in range(min(old_only,new_only)+1))/2**discord) if discord else 1.
            item.update(paired_outcomes=dict(both_success=int(((a==1)&(b==1)).sum()),old_success_new_failure=old_only,
                old_failure_new_success=new_only,both_failure=int(((a==0)&(b==0)).sum())),exact_mcnemar_two_sided_p=p)
        results[key]=item
    a,na=rmse_components(old);b,nb=rmse_components(new)
    draw=np.sqrt(b[ix].sum(1)/nb[ix].sum(1))-np.sqrt(a[ix].sum(1)/na[ix].sum(1))
    results['grade_rmse']=dict(old=float(np.sqrt(a.sum()/na.sum())),new=float(np.sqrt(b.sum()/nb.sum())),
        new_minus_old=float(np.sqrt(b.sum()/nb.sum())-np.sqrt(a.sum()/na.sum())),
        paired_bootstrap_95ci=np.quantile(draw,[.025,.975]).tolist(),old_valid_windows=int(na.sum()),new_valid_windows=int(nb.sum()))
    ac=np.asarray([sum(w['compliant'] for w in r['physical']['windows']) for r in old])/4
    bc=np.asarray([sum(w['compliant'] for w in r['physical']['windows']) for r in new])/4
    results['window_compliance']=dict(old=float(ac.mean()),new=float(bc.mean()),new_minus_old=float((bc-ac).mean()),
        paired_bootstrap_95ci=np.quantile((bc-ac)[ix].mean(1),[.025,.975]).tolist())
    windows=[]
    for w in range(4):
        x=np.array([r['physical']['windows'][w]['grade'] for r in old],dtype=float)
        y=np.array([r['physical']['windows'][w]['grade'] for r in new],dtype=float)
        windows.append(dict(window=w,old_rmse=float(np.sqrt(np.mean((x-.27)**2))),
            new_rmse=float(np.sqrt(np.mean((y-.27)**2))),
            old_compliant=sum(r['physical']['windows'][w]['compliant'] for r in old),
            new_compliant=sum(r['physical']['windows'][w]['compliant'] for r in new)))
    return results,windows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--physical-only',action='store_true');args=parser.parse_args()
    old,new,provenance=load_physical();results,windows=paired_statistics(old,new)
    result=dict(schema='bg128-paired-actor-lr-comparison-v1',old_actor_lr=3e-4,new_actor_lr=1e-4,
        initialization='same_sealed_original_common_u0',checkpoint_update=13,paired_environment_and_sampling_seeds=True,
        shifts_per_arm=100,bootstrap=dict(repetitions=BOOTSTRAPS,seed=SEED,unit='complete_shift_pair_with_all_four_windows',
            interpretation='test_scenario_uncertainty_conditional_on_one_training_seed; no multiplicity adjustment'),
        physical_metrics=results,per_window=windows,provenance=provenance,
        old_failed_runs=[r['identity']['run_index'] for r in old if not r['physical']['complete_task']],
        new_failed_runs=[r['identity']['run_index'] for r in new if not r['physical']['complete_task']],
        new_physical_shifts=0,optimizer_steps=0)
    if not args.physical_only:
        p=NEW/'suite_complete.json';assert p.is_file(),'official evaluator still finishing; do not treat collection as final metrics'
        completed=read(p);report=read(NEW/'evaluation_complete.json')
        assert all(completed[k]==0 for k in ('optimizer_updates','training_updates','gradient_updates'))
        assert completed['parent_artifacts_before']==completed['parent_artifacts_after']
        assert report['training_state_unchanged'] and report['n_shifts']==100
        for cached,final in zip(new,report['per_episode']):
            assert cached['identity']==final['identity'] and cached['physical']==final['physical']
        old_report=read(OLD/'evaluation_complete.json')
        result['critic']=dict(old=old_report['critic_mc'],new=report['critic_mc'])
        result['official_new_completion']=reference(p)
        result['official_new_summary']=reference(NEW/'reports/summary.json')
        result['status']='FINAL_PAIRED_COMPARISON'
    else:result['status']='PHYSICAL_COMPARISON_OFFICIAL_VALUE_SCORING_PENDING'
    name='physical_comparison.json' if args.physical_only else 'comparison.json'
    write(REPORT/name,result)
    rows=[]
    for ordinal,(a,b) in enumerate(zip(old,new)):
        row=dict(ordinal=ordinal,environment_seed=a['identity']['environment_seed'],actor_sampling_seed=a['identity']['actor_sampling_seed'],
            old_run=a['identity']['run_index'],new_run=b['identity']['run_index'])
        for key in ('complete_task','combined_cost','grade_cost','production_cost','total_wait_seconds','ore_tons','waste_tons'):
            row['old_'+key]=a['physical'][key];row['new_'+key]=b['physical'][key]
            row['delta_'+key]=float(b['physical'][key])-float(a['physical'][key])
        rows.append(row)
    filename='physical_pairs.csv' if args.physical_only else 'pairs.csv'
    with (REPORT/filename).open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(status=result['status'],metrics=results,per_window=windows,
        old_failed_runs=result['old_failed_runs'],new_failed_runs=result['new_failed_runs']),ensure_ascii=False))


if __name__=='__main__':main()
