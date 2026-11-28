"""Audit the physical shifts before comparing policies and outage severity."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import stdev
import sys

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent),str(HERE.parent/'same_seed_control_001')]
import outage_results as pilot
import control_results as paired
import batch_campaign as campaign
import batch_stats
from batch_campaign import require,witness
import collect_stats

START=14400
HOUR_END=18000


def audit_targets(_,cfg):
    scene=cfg['expected_runtime_scene']
    env=scene['exogenous_config']
    fleet=scene['opening_fleet']
    expected=collect_stats.scaled_task(collect_stats.REFERENCE_TARGETS,fleet)
    require(sum(r[1] for r in fleet)==9 and scene['layout_id']=='tree_A' and env['fault_truck_id']==-1
            and env['deterministic'] is False and env['endogenous_congestion'] is False,'expected nominal A')
    require(all(cfg['task'][k]==env[k]==v for k,v in expected.items()) and env['ore_target_tons_2h']==4000.
            and cfg['task']['window_seconds']==7200 and cfg['task']['grade_lower']==.26 and cfg['task']['grade_upper']==.28,
            'unexpected business rule or target')
    require(cfg['action_mode']==('deterministic' if cfg['strategy']=='flow' else 'argmax')
            and cfg['actor_sampling_seed'] is None,'policy evaluation settings differ')
    require(env['load_site_outage']==(cfg['outage'] if cfg['duration_minutes'] else False)
            and scene['load_sites'][0][:5]==[0,5,True,320000,10000], 'unexpected high-grade site')
    return dict(active_trucks=9,ore_target_tons_2h=4000.,**expected)


assess=witness.bind(collect_stats.assess_one,opening_targets=audit_targets)


def audit_flow(cfg,job,physical,policy,decisions):
    folder=Path(job['python_output'])
    seal=json.loads((folder/'flow_diagnostics_seal.json').read_bytes())
    require(seal['native_manifest']==physical['manifest'] and seal['inference_manifest']==policy['inference_manifest']
            and seal['run_index']==cfg['run_index'] and seal['recording_id']==cfg['recording_id']
            and seal['config_sha256']==job['config_sha256'] and seal['policy_digest']==cfg['checkpoint_sha256']
            and Path(seal['diagnostic']['path'])==folder/'flow_diagnostics.json','flow receipts differ')
    info=json.loads(pilot.audit_file(seal['diagnostic']))
    expected=[('total',len(decisions)),('before_4h',sum(d['sim_time_seconds']<START for d in decisions))]
    if cfg['duration_minutes']:
        end=START+cfg['duration_minutes']*60
        expected.append(('during_outage',sum(START<=d['sim_time_seconds']<end for d in decisions)))
    for phase,count in expected:
        require(info['phases'].get(phase,{}).get('decisions',0)==count,'unaccounted flow decisions')
    return dict(receipt=witness.reference(folder/'flow_diagnostics_seal.json'),phases=info['phases'])


def digest_records(obj):
    return sha256(witness.canonical(obj)).hexdigest()


def audit_case(label):
    cfg,_,job=campaign.load_case(label)
    p,policy=assess((label,cfg,job,False))
    require(p['physical_status']=='COMPLETE' and policy['status']=='VERIFIED' and p.get('material_balance_verified') is True,
            f'{label} physical or inference validation failed {p.get("reason")} {policy}')
    m=json.loads(pilot.audit_file(p['manifest']))
    require(m['native_strategy']==cfg['strategy'] and m['native_policy_id']==cfg['policy_id']
            and m['native_policy_update']==cfg['policy_update'] and m['native_duration_minutes']==cfg['duration_minutes']
            and cfg['checkpoint_sha256']==m['native_checkpoint_sha256']==policy['checkpoint_sha256']
            and m['invalid_dispatches']==m['fallback_dispatches']==0, f'{label} wrong policy binding')
    events=pilot.native_events(p['manifest'])
    decisions=pilot.read_decisions(policy,p['manifest'])
    minutes=cfg['duration_minutes']
    end=START+minutes*60
    trace,jobs=pilot.site_timeline(events,0,START,end)
    deficit=0
    during=[d for d in decisions if START<=d['sim_time_seconds']<end]
    if minutes:
        native=pilot.sidecar_audit(cfg,p,jobs)
        require(all(d['action_mask'][0]==0 and d['action']!=0 for d in during)
                and trace['service_starts_during']==trace['service_completions_during']==0
                and all(d['action_mask'][0]==1 for d in decisions if d['sim_time_seconds']>=end),
                f'{label} a high-grade service or action continued during closure, or failed to reopen')
        cursor=native['stop_before']['event_cursor']
        committed=[j for j in jobs if j['dispatch_seq']<cursor<=j.get('complete_seq',math.inf)]
        require(len(committed)==native['committed_trucks_at_stop'] and
                all(j.get('complete',-1)>=end for j in committed),f'{label} missing previous site commitment')
        deficit=native['paused_in_service_count']
    else:
        require(m['control_site_id']==0 and m['control_available_decisions']==len(decisions)
                and m['control_available_window_decisions']==sum(START<=d['sim_time_seconds']<16200 for d in decisions)
                and m['control_available_60min_decisions']==sum(START<=d['sim_time_seconds']<HOUR_END for d in decisions)
                and all(d['action_mask'][0]==1 for d in decisions) and 'outage_evidence_sha256' not in m,
                f'{label} normal high-grade site unavailable')
    prior_decisions=paired.pre_decisions(decisions)
    prior_events=paired.pre_events(events)
    waiting=trace['total_wait_seconds']
    require(math.isclose(sum(paired.wait_by_phase(events)),waiting,abs_tol=1e-6,rel_tol=1e-9), 'fleet wait replay differs')
    windows=p['windows']
    grades=[100*w['grade'] if w['ore_tons']>0 else None for w in windows]
    require(len(windows)==4,'four closed windows required')
    obs=dict(case=label,strategy=cfg['strategy'],duration_minutes=minutes,slot=cfg['slot'],
             environment_seed=cfg['environment_seed'],run_index=cfg['run_index'],joint=int(p['task_status']=='MET'),
             four=p['compliant_windows']==4,ore_met=p['ore_target_met'],waste_met=p['waste_target_met'],
             window_pass=[w['ore_tons']>0 and w['cost']==0 for w in windows],window_grade_percent=grades,
             grade=stdev(grades) if all(x is not None for x in grades) else None,wait=waiting/3600,
             ore_t=p['ore_tons'],waste_t=p['waste_tons'],decisions=policy['decision_count'],
             paused_in_service=deficit,high_tons_4_to_5=math.fsum(j['tons'] for j in jobs if START<=j.get('complete',math.inf)<HOUR_END),
             prefix=dict(decisions=digest_records(prior_decisions),business_events=digest_records(prior_events),
                         first_two_window_material=[[w['ore_tons'],w['metal_tons']] for w in windows[:2]]))
    if cfg['strategy']=='flow':
        obs['flow']=audit_flow(cfg,job,p,policy,decisions)
    return dict(case=label,physical=p,inference_audit=policy,metrics=obs)


def collect():
    require(len(campaign.case_labels())==270, 'expected 30 complete nine-way slots')
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(audit_case,campaign.case_labels(),chunksize=3))
    metrics=[r['metrics'] for r in rows]
    require(len({r['physical']['recording_id'] for r in rows})==len({r['run_index'] for r in metrics})==270,'duplicate physical identity')
    groups=defaultdict(list)
    for m in metrics:
        groups[m['strategy'],m['slot']].append(m)
    mismatches=defaultdict(list)
    for (s,slot),g in groups.items():
        if not all(item['prefix']==g[0]['prefix'] for item in g):
            mismatches[s].append(slot)
    require(not mismatches['full'] and not mismatches['flat'], 'neural policy prefix differs before outage')
    descriptive,contrasts=batch_stats.summarize(metrics)
    flow={}
    for duration in (0,30,60):
        phases=defaultdict(Counter)
        for m in metrics:
            if m['strategy']=='flow' and m['duration_minutes']==duration:
                for phase,counts in m['flow']['phases'].items():
                    phases[phase].update(counts)
        flow[str(duration)]={phase:dict(c) for phase,c in phases.items()}
    report=dict(descriptive=descriptive,contrasts=contrasts,flow_diagnostics=flow,prefix_mismatch_slots=dict(mismatches),
                new_physical_shifts=len(rows),environment_seeds=30,verified_actions=sum(m['decisions'] for m in metrics),
                native_invalid=0,native_fallback=0,protocol=witness.reference(HERE/'protocol.json'))
    with campaign.settlement_session() as settle:
        for r in rows:
            _,suite,job=campaign.load_case(r['case'])
            r['ledger_status']=settle(suite,job,r['physical'],collect=True)
            require(r['ledger_status']=='COMPLETE', 'incomplete physical settlement')
            pilot.freeze(Path(job['python_output'])/'policy_verification.json',r['inference_audit'])
    report['budget']=campaign.verify_budget(rows)
    pilot.freeze(campaign.REPORTS/'cases.json',rows)
    pilot.freeze(campaign.REPORTS/'summary.json',report)
    from batch_report import write_report
    write_report(campaign.REPORTS,report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--case',help='read-only audit of one completed shift')
    mode.add_argument('--collect',action='store_true')
    args=parser.parse_args()
    result=audit_case(args.case)['metrics'] if args.case else collect()['descriptive']
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))
