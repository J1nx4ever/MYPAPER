"""Paired, exit-anchored results from complete, sealed physical receipts."""
from collections import defaultdict
import argparse
import json
import sqlite3
from pathlib import Path
from statistics import fmean

from resilience_campaign import load_suite, ROOT, PLAN


def mean(xs):
    return fmean(xs) if xs else None


def metrics(rows):
    p = [r['physical'] for r in rows]
    eligible = [r['outage'] for r in rows if r['outage']['window_valid']]
    result = {k: mean([row[k] for row in p]) for k in (
        'all_four_windows_compliant','complete_task','ore_tons','waste_tons','ore_shortfall','waste_shortfall',
        'total_wait_seconds','grade_cost','production_cost','waiting_cost','combined_cost')}
    events = [r['outage'] for r in rows]
    result.update(shifts=len(rows), valid_post_exit_windows=len(eligible),
                  no_fault_shifts=sum(w['trigger_seconds'] < 0 for w in events),
                  still_retiring_at_end=sum(w['trigger_seconds'] >= 0 and w['exit_seconds'] < 0 for w in events),
                  late_exits=sum(w['exit_seconds'] > 21600 for w in events),
                  mean_trigger_to_exit_seconds=mean([w['exit_delay_seconds'] for w in events if w['exit_seconds'] >= 0]),
                  mean_ore_shortfall_tons=mean([max(0.,r['ore_target_tons']-r['ore_tons']) for r in p]),
                  mean_waste_shortfall_tons=mean([max(0.,r['waste_target_tons']-r['waste_tons']) for r in p]),
                  compliant_post_exit_windows=sum(w['window_compliant'] for w in eligible),
                  empty_post_exit_windows=sum(w['window_empty'] for w in eligible),
                  post_exit_rate=mean([w['window_compliant'] for w in eligible]))
    return result


def summarize(suite):
    groups = defaultdict(dict)
    per_stratum = defaultdict(list)
    for j in suite['jobs']:
        if j['role'] != 'eval':
            continue
        receipt=json.loads((Path(j['python_output'])/'receipt.json').read_bytes())
        for key in ('run_index','recording_id','experiment_seed','actor_sampling_seed','config_sha256','arm','condition','split','pair_ordinal'):
            if receipt[key]!=j[key]:
                raise ValueError('evaluation receipt mismatch: '+key)
        cell=(j['arm'],j['condition'])
        pairs=groups[(j['split'],j['pair_ordinal'])]
        if cell in pairs:
            raise ValueError('duplicate arm or condition in evaluation pair')
        pairs[cell]=receipt
        if j['condition']=='outage':
            per_stratum[(j['split'], j['arm'], j['starting_count'], j['fault_truck_id']//3,j['waste_site'])].append(receipt)
    summary={}
    for split in ('validation','test'):
        cases=[cells for (part,_),cells in groups.items() if part==split]
        if len(cases)!=suite['protocol'][split+'_slots'] or not all(len(case)==4 for case in cases):
            raise ValueError('incomplete evaluation cells')
        summaries={}
        for arm in ('static','dynamic'):
            h=[c[arm,'healthy'] for c in cases]
            f=[c[arm,'outage'] for c in cases]
            for i in range(len(f)):
                peers=list(cases[i].values())
                if len({(r['experiment_seed'],r['actor_sampling_seed']) for r in peers})!=1:
                    raise ValueError('unpaired physical seeds')
            summaries[arm]=dict(healthy=metrics(h),outage=metrics(f),
                paired_ore_loss_tons=fmean(a['physical']['ore_tons']-b['physical']['ore_tons'] for a,b in zip(h,f)),
                paired_waste_loss_tons=fmean(a['physical']['waste_tons']-b['physical']['waste_tons'] for a,b in zip(h,f)))
        common=[c for c in cases if all(c[a,'outage']['outage']['window_valid'] for a in ('dynamic','static'))]
        summaries['paired_valid_exit_windows']=len(common)
        summaries['paired_exit_rate_delta']=mean([int(c['dynamic','outage']['outage']['window_compliant'])
                -int(c['static','outage']['outage']['window_compliant']) for c in common])
        summary[split]=summaries
    val=summary['validation']
    margin=suite['grade_margin']
    grade_ok=all(val['dynamic'][c]['all_four_windows_compliant']+margin>=val['static'][c]['all_four_windows_compliant']
                 for c in ('healthy','outage'))
    delta=val['paired_exit_rate_delta']
    grade_ok=grade_ok and delta is not None and delta>=-margin
    def shortage(arm):
        p=val[arm]['outage']
        return p['ore_shortfall']+p['waste_shortfall']
    selection='dynamic' if grade_ok and shortage('dynamic')<shortage('static') else 'static'
    return dict(campaign_id=suite['campaign_id'],grade_margin=margin,validation_grade_gate=grade_ok,
                selection_from_validation=selection,results=summary,
                strata=[dict(split=key[0],arm=key[1],opening_count=key[2],truck_type=(777,785,789)[key[3]],
                             waste_site=key[4],**metrics(rows)) for key,rows in sorted(per_stratum.items())])


def make_report(suite):
    from resilience_runtime import record
    from resilience_budget import ledger_path
    from resilience_campaign import reference
    with sqlite3.connect(ledger_path(suite).as_uri()+'?mode=ro', uri=True) as db:
        rows = {run:(state,pin) for run,state,pin in db.execute('SELECT run,status,evidence FROM slots')}
    for job in suite['jobs']:
        state, pin = rows[job['run_index']]
        if state != 'COMPLETE' or json.loads(pin) != reference(Path(job['python_output'])/'receipt.json'):
            raise ValueError('incomplete or altered physical receipt for '+str(job['run_index']))
    result=summarize(suite)
    path=ROOT/'reports'/PLAN.parent.parent.name/PLAN.parent.name/PLAN.name/suite['campaign_id']/'evaluation.json'
    if path.exists():
        if json.loads(path.read_bytes())!=result:
            raise ValueError('archived evaluation differs')
    else:
        record(path,result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite',required=True)
    print(json.dumps(make_report(load_suite(parser.parse_args().suite)),ensure_ascii=False))
