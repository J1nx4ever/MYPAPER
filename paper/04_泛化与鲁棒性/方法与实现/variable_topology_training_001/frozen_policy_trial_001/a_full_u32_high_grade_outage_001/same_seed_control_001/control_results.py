"""Replay the native no-outage control and its independently sealed A-map outage counterpart."""
import argparse
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import outage_results as original
import a_outage_campaign as stopped
import collect_stats
from a_outage_campaign import require, witness
from resilience_runtime import atomic

START, END, HORIZON = 14400, 16200, 28800
INTERVALS = ((0, START), (START, END), (END, HORIZON))


def signed(a, b, tolerance=1e-9):
    value = a-b
    return 0. if abs(value) <= tolerance else value


def control_shift(campaign):
    label, = campaign.case_labels()
    cfg, suite, job = campaign.load_case(label)
    require(cfg['outage'] is None and cfg['expected_runtime_scene']['exogenous_config']['load_site_outage'] is False,
            'control scenario must have no outage')
    assess = witness.bind(collect_stats.assess_one,
                          opening_targets=lambda _, cfg: collect_stats.opening_targets('A', cfg))
    physical, policy = assess((label, cfg, job, False))
    require(physical['physical_status']=='COMPLETE' and policy['status']=='VERIFIED'
            and physical.get('material_balance_verified') is True and physical['invalid_dispatches']==physical['fallback_dispatches']==0,
            f"control verification failed: {physical.get('reason')} {policy.get('reason')}")
    manifest = json.loads(original.audit_file(physical['manifest']))
    decisions = original.read_decisions(policy, physical['manifest'])
    during = [r for r in decisions if START <= r['sim_time_seconds'] < END]
    require(manifest['native_policy_update']==32 and manifest['control_site_id']==0
            and manifest['native_checkpoint_sha256']==stopped.CHECKPOINT_SHA
            and manifest['control_available_decisions']==len(decisions)
            and manifest['control_available_window_decisions']==len(during)
            and 'outage_evidence_sha256' not in manifest and all(r['action_mask'][0]==1 for r in decisions),
            'the high-grade loader was unavailable in the nominal control')
    return cfg, suite, job, dict(case=label,physical=physical,inference_audit=policy)


def frozen_outage(cfg):
    row = original.one('O_001')
    old, suite, job = stopped.load_case('O_001')
    pin = cfg['paired_outage']
    require(stopped.ledger_entry(suite,job)==('COMPLETE',pin['receipt'])
            and witness.reference(suite['suite_path'])==pin['config']
            and row['physical']['manifest']==pin['manifest']
            and row['suspension']['evidence']==pin['outage_evidence']
            and json.loads(original.audit_file(pin['policy_verification']))==row['inference_audit'],
            'outage comparison is not bound to its original verified run')
    receipt = json.loads(original.audit_file(pin['receipt']))
    require(all(receipt[k]==v for k,v in row['physical'].items()), 'replayed outage differs from original receipt')
    require(old['run_index']!=cfg['run_index'] and old['recording_id']!=cfg['recording_id']
            and old['event_directory']!=cfg['event_directory']
            and old['python_output_directory']!=cfg['python_output_directory'], 'physical run identities overlap')
    fields = ('layout_id','layout_sha256','task','task_sha256','policy_id','policy_update','actor_digest',
              'checkpoint_sha256','action_mode','actor_sampling_seed','slot_permutation_seed','environment_seed')
    require(all(old[k]==cfg[k] for k in fields) and cfg['environment_seed']==92953747099209
            and cfg['checkpoint_sha256']==stopped.CHECKPOINT_SHA and cfg['layout_id']=='tree_A'
            and cfg['action_mode']=='argmax' and cfg['actor_sampling_seed'] is None,
            'model, randomness or task differs between the two cases')
    def scene(c):
        x=dict(c['expected_runtime_scene'])
        x.pop('model_sha256')
        exogenous=dict(x['exogenous_config'])
        exogenous.pop('load_site_outage')
        x['exogenous_config']=exogenous
        return x
    require(scene(old)==scene(cfg) and old['outage']==stopped.OUTAGE, 'physical scenes differ beyond intervention')
    return row


def pre_decisions(rows):
    keys = ('sim_time_seconds','truck_id','action_mask','action','decision_sequence','dispatchable_mask',
            'in_system_mask','observation_sha256')
    return [[r[k] for k in keys] for r in rows if r['sim_time_seconds']<START]


def pre_events(events):
    result=[]
    for event in events:
        t,kind=event['sim_time_seconds'],event['event_type']
        if t >= START or kind in ('EPISODE_START','FIFO_PROCESS'):
            continue
        data=dict(event['data'])
        if 'request_id' in data:
            require(data.pop('request_id')==f"{event['episode_id']}-{data['decision_sequence']}", 'unexpected native request identity')
        result.append([t,kind,data])
    return result


def wait_by_phase(events):
    waits={}
    totals=[0.,0.,0.]
    for event in events:
        kind,t,data=event['event_type'],event['sim_time_seconds'],event['data']
        if kind=='WAIT_ENTER':
            waits[(data['truck_id'],data['resource'],data['site_id'])]=t
        elif kind in ('WAIT_EXIT','EPISODE_END'):
            if kind=='WAIT_EXIT':
                starts=[waits.pop((data['truck_id'],data['resource'],data['site_id']))]
            else:
                starts=list(waits.values())
            for start in starts:
                for i,(lo,hi) in enumerate(INTERVALS):
                    totals[i]+=max(0.,min(t,hi)-max(start,lo))
    return totals


def measure(row):
    p=row['physical']
    policy=row['inference_audit']
    events=original.native_events(p['manifest'])
    decisions=original.read_decisions(policy,p['manifest'])
    trace,jobs=original.site_timeline(events,0,START,END)
    wait=wait_by_phase(events)
    total_wait=trace['total_wait_seconds']
    require(math.isclose(sum(wait),total_wait,rel_tol=1e-9,abs_tol=1e-6), 'split fleet wait differs from end total')
    window_grades=[100*w['grade'] if w['ore_tons']>0 else None for w in p['windows']]
    committed=[dict(truck_id=j['truck_id'],decision=j['decision'],dispatch_seconds=j['dispatch'],
                    arrival_seconds=j.get('arrival'),start_loading_seconds=j.get('begin'),
                    complete_loading_seconds=j.get('complete'),tons=j.get('tons'))
               for j in jobs if j['dispatch']<START<=j.get('complete',math.inf)]
    metrics=dict(run_index=p['run_index'],recording_id=p['recording_id'],manifest=p['manifest'],
                 checkpoint_sha256=policy['checkpoint_sha256'],seed=p['environment_seed'],
                 verified_decisions=policy['decision_count'],joint=p['task_status'],ore_met=p['ore_target_met'],
                 waste_met=p['waste_target_met'],windows_passed=p['compliant_windows'],ore_tons=p['ore_tons'],
                 metal_tons=p['metal_tons'],waste_tons=p['waste_tons'],wait_fleet_seconds=total_wait,
                 wait_fleet_seconds_by_phase=wait,window_grade_percent=window_grades,
                 window_ore_tons=[w['ore_tons'] for w in p['windows']],
                 window_metal_tons=[w['metal_tons'] for w in p['windows']],
                 window_compliant=[w['cost']==0 and w['ore_tons']>0 for w in p['windows']],
                 high_site=trace,committed_high_site_trips=committed,
                 invalid=p['invalid_dispatches'],fallback=p['fallback_dispatches'],
                 material_verified=p['material_balance_verified'])
    return metrics,pre_decisions(decisions),pre_events(events)


def compare(cfg,control,outage):
    c,cd,ce=measure(control)
    o,od,oe=measure(outage)
    require(cd and ce and cd==od and ce==oe, 'dispatch sequence, observation or ordinary physical event diverged before outage')
    require(all(abs(c[k][i]-o[k][i])<1e-6 for k in ('window_ore_tons','window_metal_tons') for i in range(2)),
            'pre-outage material totals differ')
    require(abs(c['wait_fleet_seconds_by_phase'][0]-o['wait_fleet_seconds_by_phase'][0])<1e-6,
            'pre-outage fleet wait differs')
    def sha(items):
        return sha256(json.dumps(items,separators=(',',':'),sort_keys=True,allow_nan=False).encode()).hexdigest()
    diff={k:signed(o[k],c[k],1e-6) for k in ('ore_tons','waste_tons','metal_tons','wait_fleet_seconds')}
    diff['window_grade_pp']=[None if a is None or b is None else signed(a,b)
                             for a,b in zip(o['window_grade_percent'],c['window_grade_percent'],strict=True)]
    diff['window_ore_tons']=[signed(a,b,1e-6) for a,b in zip(o['window_ore_tons'],c['window_ore_tons'],strict=True)]
    diff['wait_fleet_seconds_by_phase']=[signed(a,b,1e-6) for a,b in zip(o['wait_fleet_seconds_by_phase'],c['wait_fleet_seconds_by_phase'],strict=True)]
    trips=[]
    for former, current in zip(c['committed_high_site_trips'],o['committed_high_site_trips'],strict=True):
        require(all(former[k]==current[k] for k in ('truck_id','decision','dispatch_seconds'))
                and all(x[k] is not None for x in (former,current) for k in ('arrival_seconds','start_loading_seconds',
                        'complete_loading_seconds','tons')), 'committed trip changed or did not complete')
        trips.append(dict(truck_id=former['truck_id'],prior_decision=former['decision'],
                          wait_delta_seconds=signed(current['start_loading_seconds']-current['arrival_seconds'],
                                                    former['start_loading_seconds']-former['arrival_seconds']),
                          completion_delay_seconds=signed(current['complete_loading_seconds'],former['complete_loading_seconds']),
                          control=former,outage=current))
    return dict(environment_seed=cfg['environment_seed'],policy_id=cfg['policy_id'],new_physical_runs=1,
                matched_pairs=1,shift_seconds=HORIZON,scheduled_outage_seconds=[START,END],
                common_prefix=dict(before_seconds=START,identical_decisions=len(cd),decisions_sha256=sha(cd),
                                   identical_business_events=len(ce),business_events_sha256=sha(ce),
                                   business_events_by_type=dict(Counter(e[1] for e in ce)),first_two_windows_material_matches=True),
                control=c,outage=o,outage_minus_control=diff,
                committed_high_site_trips=trips,stop_evidence=outage['suspension']['evidence'])


def format_report(report):
    c,o,d=report['control'],report['outage'],report['outage_minus_control']
    prefix=report['common_prefix']
    lines=['# A 图最高品位点停工的同种子无停工对照','',
           f"无停工对照 run {c['run_index']} 与原停工班 run {o['run_index']} 共用环境种子 {report['environment_seed']}。"
           "两班使用冻结的 BG128 Full s02/u32、九车和相同的贪心派车规则。原班的 0 号点在 4:00–4:30 停工，对照班全程开放。",'',
           f"第 4 小时前，{prefix['identical_decisions']} 次派车的时刻、车辆、动作、动作掩码和观测哈希完全一致，"
           f"{prefix['identical_business_events']} 条业务事件及前两窗的矿石、金属量亦一致。",'',
           '| 全班指标 | 无停工对照 | 停工班 | 停工班减对照 |','|---|---:|---:|---:|',
           f"| 已破碎矿石（t） | {c['ore_tons']:,.2f} | {o['ore_tons']:,.2f} | {d['ore_tons']:+,.2f} |",
           f"| 已卸废石（t） | {c['waste_tons']:,.2f} | {o['waste_tons']:,.2f} | {d['waste_tons']:+,.2f} |",
           f"| 车队累计等待（车辆·小时） | {c['wait_fleet_seconds']/3600:.3f} | {o['wait_fleet_seconds']/3600:.3f} | {d['wait_fleet_seconds']/3600:+.3f} |",
           f"| 合规的两小时窗 | {c['windows_passed']}/4 | {o['windows_passed']}/4 | {o['windows_passed']-c['windows_passed']:+d} |",
           f"| 整班联合达标 | {'是' if c['joint']=='MET' else '否'} | {'是' if o['joint']=='MET' else '否'} |  |",'',
           '| 班内窗口 | 对照品位（%） | 停工班品位（%） | 差值（百分点） |','|---|---:|---:|---:|']
    for i,(a,b,x) in enumerate(zip(c['window_grade_percent'],o['window_grade_percent'],d['window_grade_pp'],strict=True)):
        cells=['无出矿' if y is None else f'{y:.3f}' for y in (a,b)]
        diff='无对应值' if x is None else f'{x:+.3f}'
        lines.append(f"| {2*i}–{2*i+2} h | {cells[0]} | {cells[1]} | {diff} |")
    ct,ot=c['high_site'],o['high_site']
    lines+=['','## 停工的半小时','',
            '| 指标 | 无停工对照 | 停工班 |','|---|---:|---:|',
            f"| 四个装载点派车次数（点 0、1、2、3） | {'、'.join(map(str,ct['dispatch_by_phase']['during']))} | {'、'.join(map(str,ot['dispatch_by_phase']['during']))} |",
            f"| 0 号点开装次数 | {ct['service_starts_during']} | {ot['service_starts_during']} |",
            f"| 0 号点完装次数 | {ct['service_completions_during']} | {ot['service_completions_during']} |",
            f"| 0 号点完装矿石（t） | {ct['high_tons_loaded_during']:.2f} | {ot['high_tons_loaded_during']:.2f} |",
            f"| 车队累计等待（车辆·分钟） | {c['wait_fleet_seconds_by_phase'][1]/60:.2f} | {o['wait_fleet_seconds_by_phase'][1]/60:.2f} |",'',
            f"4:30–8:00 的车队累计等待为对照 {c['wait_fleet_seconds_by_phase'][2]/3600:.3f}、停工班 {o['wait_fleet_seconds_by_phase'][2]/3600:.3f} 车辆·小时。"
            "等待量按各车实际排队时间求和。该结果描述一个配对种子。",'',
            f"对照班 {c['verified_decisions']} 次、停工班 {o['verified_decisions']} 次派车均与原生动作一致，两班的矿石与金属守恒检查均通过。",'']
    for trip in report['committed_high_site_trips']:
        a,b=trip['control'],trip['outage']
        def shift_time(t):
            hours, rest=divmod(t,3600)
            minutes,seconds=divmod(rest,60)
            return f'{int(hours)}:{int(minutes):02d}:{seconds:05.2f}'
        lines.extend([f"停工前已派往 0 号点的 {trip['truck_id']} 号车，在对照班于 {shift_time(a['arrival_seconds'])} 到场、"
                      f"{shift_time(a['start_loading_seconds'])} 开装、{shift_time(a['complete_loading_seconds'])} 完装 {a['tons']:.1f} t。"
                      f"停工班对应的时刻为 {shift_time(b['arrival_seconds'])}、{shift_time(b['start_loading_seconds'])}、"
                      f"{shift_time(b['complete_loading_seconds'])}，装载量为 {b['tons']:.1f} t。"
                      f"该车的排队时间差为 {trip['wait_delta_seconds']/60:+.2f} 分钟，完装时间差为 {trip['completion_delay_seconds']/60:+.2f} 分钟。",''])
    lines.extend(['[配对完整统计](summary.json) · [对照班回放](cases.json)',''])
    return '\n'.join(lines)


def main():
    import control_campaign as campaign
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collect',action='store_true',help='seal the control receipt and publish the paired report')
    args=parser.parse_args()
    cfg,suite,job,control=control_shift(campaign)
    outage=frozen_outage(cfg)
    report=compare(cfg,control,outage)
    if args.collect:
        with campaign.settlement_session() as settle:
            control['ledger_status']=settle(suite,job,control['physical'],collect=True)
        require(control['ledger_status']=='COMPLETE','control ledger did not seal COMPLETE')
        original.freeze(Path(job['python_output'])/'policy_verification.json',control['inference_audit'])
        report['budget']=campaign.verify_budget([control])
        original.freeze(campaign.REPORTS/'cases.json',[control])
        original.freeze(campaign.REPORTS/'summary.json',report)
        atomic(campaign.REPORTS/'summary.md',format_report(report).encode(),replace=True)
    print(json.dumps(report,ensure_ascii=False,allow_nan=False),flush=True)


if __name__=='__main__':
    main()
