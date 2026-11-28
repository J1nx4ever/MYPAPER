"""Verify the single standard A-map Full u32 shift with a physical high-grade loader outage."""
import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
import a_outage_campaign as campaign
from a_outage_campaign import require, witness
import collect_stats
from collect_trial import iter_event_bundle
from resilience_runtime import atomic, record

EPS=1.e-6


def freeze(path, value):
    if path.exists():
        require(json.loads(path.read_bytes())==value, f'existing sealed evidence differs at {path}')
    else:
        record(path,value)


def audit_file(pin):
    raw = Path(pin['path']).read_bytes()
    require(sha256(raw).hexdigest() == pin['sha256'], f"evidence changed at {pin['path']}")
    return raw


def read_decisions(policy, manifest_pin):
    seal = json.loads(audit_file(policy['inference_manifest']))
    path = Path(policy['inference_manifest']['path']).with_name('inference.jsonl')
    rows = [json.loads(line) for line in audit_file(dict(path=path,sha256=seal['inference_sha256'])).splitlines()]
    manifest = json.loads(audit_file(manifest_pin))
    require(len(rows)==policy['decision_count']==seal['decision_count']==manifest['native_policy_decisions']
            and manifest['complete'] and seal['complete']
            and manifest['native_checkpoint_sha256']==policy['checkpoint_sha256']==seal['checkpoint_sha256']
            and all(seal[k]==manifest[k] for k in ('recording_id','config_sha256','run_index')),
            'inference and native seals differ')
    return rows


def native_events(pin):
    meta=json.loads(audit_file(pin))
    return list(iter_event_bundle(Path(pin['path']).parent,meta))


def phase(t,start,end):
    return 'pre' if t < start else ('during' if t < end else 'post')


def site_timeline(events, site, start, end):
    # All workflow steps for the same physical truck stay linked to its dispatch.
    jobs=[]
    pending={}
    actions={k:[0,0,0,0] for k in ('pre','during','post')}
    for e in events:
        t,kind,data=e['sim_time_seconds'],e['event_type'],e['data']
        truck=data.get('truck_id')
        if kind=='ACTION_APPLIED':
            target=data['action']
            actions[phase(t,start,end)][target]+=1
            require(truck not in pending, 're-dispatch before completing the committed high-grade load')
            if target==site:
                job=dict(truck_id=truck,decision=data['decision_sequence'],dispatch=t,dispatch_seq=e["event_seq"])
                jobs.append(job)
                pending[truck]=job
            continue
        if data.get('site_id')!=site:
            continue
        if kind in ('WAIT_ENTER','WAIT_EXIT') and data['resource']=='load':
            key = 'arrival' if kind=='WAIT_ENTER' else 'begin'
            previous = 'dispatch' if kind=='WAIT_ENTER' else 'arrival'
        elif kind=='LOAD_COMPLETE':
            key,previous='complete','begin'
            require(data['material']=='ore','non-ore at high-grade site')
        else:
            continue
        require(truck in pending and previous in pending[truck] and key not in pending[truck]
                and t>=pending[truck][previous], f'{kind} differs from the high-grade load commitment')
        pending[truck][key]=t
        pending[truck][key+'_seq']=e['event_seq']
        if key=='complete':
            pending[truck].update(tons=data['tons'],metal_tons=data['metal_tons'],payload_id=data['payload_id'])
            del pending[truck]
    def during(j,field):
        return start <= j.get(field, math.inf) < end
    completed=[j for j in jobs if during(j,'complete')]
    completions=[j['complete'] for j in jobs if end <= j.get('complete',-math.inf)]
    starts=[j['begin'] for j in jobs if end <= j.get('begin',-math.inf)]
    dispatches=[j['dispatch'] for j in jobs if j['dispatch']>=end]
    require(events[-1]['event_type']=='EPISODE_END', 'missing complete physical shift')
    out=dict(dispatch_by_phase=actions,
             arrivals_during=sum(during(j,'arrival') for j in jobs),
             service_starts_during=sum(during(j,'begin') for j in jobs),
             service_completions_during=len(completed),high_tons_loaded_during=math.fsum(j['tons'] for j in completed),
             high_metal_loaded_during=math.fsum(j['metal_tons'] for j in completed),
             first_high_dispatch_after_restart_seconds=min(dispatches)-end if dispatches else None,
             first_high_start_after_restart_seconds=min(starts)-end if starts else None,
             first_high_completion_after_restart_seconds=min(completions)-end if completions else None,
             total_wait_seconds=events[-1]['data']['cumulative_wait_seconds'])
    return out,jobs


def physical_snapshot(jobs,snapshot):
    cursor=snapshot['event_cursor']
    stages={'incoming':[],'queued':[],'loading':[]}
    for j in jobs:
        if j['dispatch_seq']>=cursor or j.get('complete_seq',math.inf)<cursor:
            continue
        stage=('incoming' if j.get('arrival_seq',math.inf)>=cursor else
               'queued' if j.get('begin_seq',math.inf)>=cursor else 'loading')
        stages[stage].append(j['truck_id'])
    return {k:sorted(v) for k,v in stages.items()}


def sidecar_audit(cfg, physical, jobs):
    manifest=json.loads(audit_file(physical['manifest']))
    path=Path(physical['manifest']['path']).with_name('outage_evidence.json')
    require('outage_evidence_sha256' in manifest and path.is_file(),
            'native sidecar or its final seal is missing; recovery requires originally recorded bytes and manifest SHA')
    pin=dict(path=str(path),sha256=manifest['outage_evidence_sha256'])
    raw=json.loads(audit_file(pin))
    site,start,end=(cfg['outage'][k] for k in ('site_id','start_seconds','end_seconds'))
    require(raw['schema_version']==1 and raw['site_id']==site and raw['start_seconds']==start
            and raw['end_seconds']==end and raw['run_index']==cfg['run_index']
            and raw['environment_seed']==cfg['environment_seed'] and raw['recording_id']==cfg['recording_id']
            and raw['config_sha256']==physical['config_sha256'] and raw['checkpoint_sha256']==cfg['checkpoint_sha256'],
            'outage sidecar identity or planned times differ')
    marks=[raw[key] for key in ('stop_before','stop_after','resume_before','resume_after')]
    require([v['time_seconds'] for v in marks]==[start,start,end,end]
            and [v['available'] for v in marks]==[True,False,False,True]
            and [v['resource_capacity'] for v in marks]==[1,0,0,1],
            'resource capacity or availability does not follow stop/restart')
    cursors=[s['event_cursor'] for s in marks]
    require(cursors==sorted(cursors) and 0<cursors[0]<=cursors[-1]<=manifest['event_count'],'invalid event cursor')
    for snap in marks:
        ledger=physical_snapshot(jobs,snap)
        actual=dict(incoming=sorted(snap['inbound_ids']),queued=sorted(snap['queue_ids']+snap['seize_ids']),
                    loading=sorted(x['truck_id'] for x in snap['service']))
        require(actual==ledger, f'actual native queue or committed trucks differ at {snap["time_seconds"]}: {actual} vs {ledger}')
    for field in ('reserved_tons','remaining_tons','produced_tons','produced_metal_tons','extraction_cursor_tons'):
        require(max(s[field] for s in marks[:3])-min(s[field] for s in marks[:3])<=EPS,
                f'high-grade {field} changed while closed')
        # Resuming can admit a waiting truck and move the extraction cursor at the same timestamp.
        if field!='extraction_cursor_tons':
            require(abs(marks[-2][field]-marks[-1][field])<=EPS, f'restart changed {field}')
    suspended=[{a['truck_id']:a for a in s['service']} for s in marks]
    require(set(suspended[0])==set(suspended[1])==set(suspended[2]) and set(suspended[0])<=set(suspended[3]),
            'service occupancy changed while paused')
    for truck,s in suspended[0].items():
        require([r[truck]['suspended'] for r in suspended]==[False,True,True,False],
                'in-progress native service did not suspend and resume')
        remaining=s['remaining_seconds']
        require(all(abs(r[truck]['remaining_seconds']-remaining)<1e-5 for r in suspended[1:]),
                'remaining native service time advanced while stopped')
        j=next(j for j in jobs if j['truck_id']==truck and j['begin_seq']<cursors[0]<=j.get('complete_seq',math.inf))
        require(math.isclose(j['complete'],end+remaining,abs_tol=1e-5,rel_tol=0),
                'preempted truck did not complete its original remaining service')
    return dict(evidence=pin,stop_before=marks[0],stop_after=marks[1],resume_before=marks[2],resume_after=marks[3],
                paused_in_service_count=len(suspended[0]),
                stopped_for_seconds=end-start,committed_trucks_at_stop=sum(map(len,physical_snapshot(jobs,marks[0]).values())))


def a_targets(_, cfg):
    scene=cfg['expected_runtime_scene']
    native=scene['exogenous_config']
    targets=collect_stats.opening_targets('A',cfg)
    require(native['load_site_outage']==cfg['outage'] and scene['layout_id']=='tree_A'
            and sum(r[1] for r in scene['opening_fleet'])==targets['active_trucks']==9,
            'native A fleet, topology or fault configuration differs')
    site=cfg['outage']['site_id']
    ore=[r for r in scene['load_sites'] if r[2]]
    high=max(ore,key=lambda r:r[3])
    require(sum(r[3]==high[3] for r in ore)==1 and high[:5]==[site,5,True,320000,10000],
            'outage did not select the highest grade point in A')
    return targets


assess=witness.bind(collect_stats.assess_one,opening_targets=a_targets)


def one(label):
    cfg,_,job=campaign.load_case(label)
    physical,policy=assess((label,cfg,job,False))
    require(physical['physical_status']=='COMPLETE' and physical.get('material_balance_verified') is True
            and policy['status']=='VERIFIED' and physical['invalid_dispatches']==physical['fallback_dispatches']==0,
            f"{label} physical or policy evidence incomplete: {physical.get('reason')} {policy.get('reason')}")
    require(cfg['checkpoint_sha256']==policy['checkpoint_sha256']==campaign.CHECKPOINT_SHA
            and cfg['policy_update']==32 and cfg['paired_baseline'] is None,
            'expected a single full u32 evaluation')
    manifest=json.loads(audit_file(physical['manifest']))
    require(manifest['native_policy_update']==32,'unexpected native policy update')
    decisions=read_decisions(policy,physical['manifest'])
    events=native_events(physical['manifest'])
    site,start,end=(cfg['outage'][k] for k in ('site_id','start_seconds','end_seconds'))
    trace,jobs=site_timeline(events,site,start,end)
    native=sidecar_audit(cfg,physical,jobs)
    during=[r for r in decisions if start<=r['sim_time_seconds']<end]
    post=[r for r in decisions if r['sim_time_seconds']>=end]
    require(all(r['action_mask'][site]==0 and r['action']!=site for r in during),
            'the closed site remained eligible or was dispatched')
    require(all(r['action_mask'][site]==1 for r in post), 'the reopened site failed to restore eligibility')
    require(trace['service_starts_during']==trace['service_completions_during']==0
            and trace['dispatch_by_phase']['during'][site]==0,'load credit, dispatch or service continued while closed')
    committed=[j for j in jobs if j['dispatch_seq']<native['stop_before']['event_cursor']<=j.get('complete_seq',math.inf)]
    require(len(committed)==native['committed_trucks_at_stop'] and all(j.get('complete',-1)>=end for j in committed),
            'an existing high-grade commitment failed to resume and finish during the shift')
    trace.update(at_stop=physical_snapshot(jobs,native['stop_before']),
                 before_restart=physical_snapshot(jobs,native['resume_before']),
                 observed_closed_masks=len(during),observed_reopened_masks=len(post),
                 preserved_commitments=[dict(truck_id=j['truck_id'],dispatch_sequence=j['decision'],
                     tons=j['tons'],completion_seconds=j['complete']) for j in committed])
    return dict(case=label,physical=physical,inference_audit=policy,trace=trace,suspension=native)


def summary(row,cfg):
    result=row['physical']
    outcome=dict(status=result['task_status'],ore_tons=result['ore_tons'],waste_tons=result['waste_tons'],
                 ore_pass=result['ore_target_met'],waste_pass=result['waste_target_met'],
                 windows=[dict(index=w['window_index'],ore_tons=w['ore_tons'],metal_tons=w['metal_tons'],
                               grade_percent=100*w['grade'] if w['ore_tons']>0 else None,
                               compliant=w['ore_tons']>0 and w['cost']==0) for w in result['windows']])
    return dict(physical_shifts=1,map=cfg['layout_id'],checkpoint_sha256=cfg['checkpoint_sha256'],
                policy_id=cfg['policy_id'],run_index=cfg['run_index'],recording_id=cfg['recording_id'],
                environment_seed=cfg['environment_seed'],outage=cfg['outage'],exposure=row['trace'],
                service=row['suspension'],outcomes=outcome,targets=result['opening_targets'],
                verified_actions=row['inference_audit']['decision_count'],
                mass_balance_error_tons=result['fifo_balance_error_tons'],invalid_dispatches=result['invalid_dispatches'],
                fallback_dispatches=result['fallback_dispatches'],matched_control=None)


def markdown(report):
    r=report['outcomes']
    trace=report['exposure']
    service=report['service']
    f=report['outage']
    t=report['targets']
    outcome='达标' if r['status']=='MET' else '未达标'
    lines=['# A 图高品位装载点短暂停工单班验证','',
           f"Full BG128 s02/u32 在标准 A 图按贪心规则派车，九车出勤。"
           f"装载点 {f['site_id']}（节点 5，ShovelOreHigh1，中心品位 32%）"
           f"在第 {f['start_seconds']/3600:g} 至 {f['end_seconds']/3600:g} 小时停工。",'',
           f"**本班联合考核{outcome}。** 已破碎矿石 {r['ore_tons']:,.2f} t，目标 {t['ore_target_tons']:,.2f} t。"
           f"卸达废石 {r['waste_tons']:,.2f} t，目标范围 {t['waste_target_tons']:,.2f}–{t['waste_upper_tons']:,.2f} t。",'',
           '| 班次时间 | 破碎矿石（t） | 破碎品位（%） | 满足 26%–28% |','|---|---:|---:|---|']
    for w in r['windows']:
        i=w['index']
        grade='无出矿' if w['grade_percent'] is None else f"{w['grade_percent']:.3f}"
        lines.append(f"| {2*i}–{2*i+2} h | {w['ore_tons']:,.2f} | {grade} | {'是' if w['compliant'] else '否'} |")
    a=trace['at_stop']
    lines+=['','## 现场停工与复工','',
            f"触发时有 {len(a['incoming'])} 辆车在途，{len(a['queued'])} 辆排队，{len(a['loading'])} 辆正在装载。"
            f"停工期间进场 {trace['arrivals_during']} 辆，策略收到 {trace['observed_closed_masks']} 次派车请求，"
            f"高品位点均被屏蔽。该点开装 {trace['service_starts_during']} 次、完成 {trace['service_completions_during']} 次。",'',
            f"复工后核对到 {trace['observed_reopened_masks']} 次决策可重新选择该点；"
            f"停工瞬间承诺的 {service['committed_trucks_at_stop']} 笔装运任务均跟踪至复工后完成。"]
    if trace['first_high_dispatch_after_restart_seconds'] is not None:
        lines.append(f"复工后首次向该点派车发生在 {trace['first_high_dispatch_after_restart_seconds']:.2f} 秒后。")
    if trace['first_high_completion_after_restart_seconds'] is not None:
        lines.append(f"复工后第一笔该点矿石在 {trace['first_high_completion_after_restart_seconds']:.2f} 秒后完成装载。")
    lines+=['',f"全班 {report['verified_actions']} 次派车均与原生事件吻合。矿石及金属数量守恒均通过核验。"
            f"结果只描述此种子的一班物理运行。",'','[逐班记录](cases.json) · [完整统计](summary.json)','']
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collect',action='store_true',help='seal physical receipt, merged ledger and report')
    args=parser.parse_args()
    labels=campaign.case_labels()
    require(tuple(labels)==('O_001',),'exactly one physical slot expected')
    row=one(labels[0])
    cfg,suite,job=campaign.load_case(labels[0])
    report=summary(row,cfg)
    if args.collect:
        with campaign.settlement_session() as settle:
            row['ledger_status']=settle(suite,job,row['physical'],collect=True)
        require(row['ledger_status']=='COMPLETE','outage ledger must seal COMPLETE')
        freeze(Path(job['python_output'])/'policy_verification.json',row['inference_audit'])
        report['budget']=campaign.verify_budget([row])
        freeze(campaign.REPORTS/'cases.json',[row])
        freeze(campaign.REPORTS/'summary.json',report)
        atomic(campaign.REPORTS/'summary.md',markdown(report).encode(),replace=True)
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
