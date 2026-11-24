"""Read-only multi-episode physical/reward/mask audit; no collection or learner.

Strict wire/event replay is a prerequisite. The second calculation below uses
raw wait intervals, FIFO batch grades and physical processing times, not adapter
outputs or replay summaries, as its arithmetic inputs.
"""
import argparse
from collections import deque
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .acceptance_replay import replay_recording
from .constrained_batch import mc_labels
from .constrained_ppo import gae


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def boundary_target_coverage(events, mode):
    """Describe observed B targets after strict wire/physical replay succeeds.

    Boundary times and zero overlap are exact. Window contributions use the
    physical intervals of the linked FIFO batch, never its declared window or
    the time at which a deferred processing/window event was written. This
    helper supplies coverage evidence; it does not replace strict replay.
    """
    require(mode in (1, 2, 3), 'unsupported boundary coverage mode')
    target_events = [e for e in events if e['data'].get('truck_id') == 0]

    def exact(kind, second):
        return [e for e in target_events if e['event_type'] == kind
                and e['sim_time_seconds'] == second]

    def summary(event):
        fields = ('truck_id', 'payload_id', 'batch_id', 'material', 'operation',
                  'tons', 'metal_tons', 'interval_start_seconds',
                  'interval_end_seconds', 'window_index')
        return dict(event_seq=event['event_seq'], event_type=event['event_type'],
                    sim_time_seconds=event['sim_time_seconds'],
                    **{k: event['data'][k] for k in fields if k in event['data']})

    load = exact('LOAD_COMPLETE', 7200.)
    unload = exact('UNLOAD_COMPLETE', 14400.)
    cutoff = exact('CUTOFF_EXCLUDED', 28800.)
    coverage = dict(mode=mode,
        load_at_7200=[e['event_seq'] for e in load],
        unload_at_14400=[e['event_seq'] for e in unload],
        cutoff_load=[e['event_seq'] for e in cutoff if e['data'].get('operation') == 'loading'],
        cutoff_unload=[e['event_seq'] for e in cutoff if e['data'].get('operation') == 'unloading'],
        # Keep wrong-time, wrong-payload and wrong-operation observations even
        # when no target can be linked, so a miss remains reviewable.
        observed_target_events=[summary(e) for e in target_events if e['event_type'] in
            ('LOAD_COMPLETE', 'UNLOAD_COMPLETE', 'FIFO_ENQUEUE', 'CUTOFF_EXCLUDED')])

    if mode == 1:
        payload = load[0]['data'].get('payload_id') if len(load) == 1 else None
        valid_payload = isinstance(payload, str) and bool(payload)
        enqueues = [e for e in events if e['event_type'] == 'FIFO_ENQUEUE'
                    and valid_payload and e['data'].get('payload_id') == payload]
        enqueue = enqueues[0] if len(enqueues) == 1 else None
        batch = enqueue['data'].get('batch_id') if enqueue else None
        valid_batch = isinstance(batch, str) and bool(batch)
        processes = [e for e in events if e['event_type'] == 'FIFO_PROCESS'
                     and valid_batch and e['data'].get('batch_id') == batch]
        processing, intervals_valid = [], True
        for event in processes:
            d = event['data']
            values = [d.get(k) for k in ('interval_start_seconds', 'interval_end_seconds', 'tons', 'metal_tons')]
            numeric = all(type(v) in (int, float) and math.isfinite(v) for v in values)
            start, end, tons, metal = values
            valid = numeric and start < end and tons > 0 and metal >= 0
            intervals_valid &= valid
            row = summary(event)
            row['valid_physical_interval'] = valid
            if valid:
                for name, lower, upper in (('second_window', 7200., 14400.),
                                           ('third_window', 14400., 21600.)):
                    overlap = max(0., min(end, upper)-max(start, lower))
                    row[name+'_seconds'] = overlap
                    row[name+'_tons'] = tons*(overlap/(end-start))
                    row[name+'_metal_tons'] = metal*(overlap/(end-start))
            processing.append(row)

        def integral(name, field):
            return math.fsum(p.get(name+'_'+field, 0.) for p in processing)

        second = {k: integral('second_window', k) for k in ('seconds', 'tons', 'metal_tons')}
        third = {k: integral('third_window', k) for k in ('seconds', 'tons', 'metal_tons')}
        checks = dict(
            unique_load_at_7200=len(load) == 1,
            unique_unload_at_14400=len(unload) == 1,
            same_ore_payload=bool(valid_payload and len(unload) == 1
                and load[0]['data'].get('material') == 'ore'
                and unload[0]['data'].get('material') == 'ore'
                and unload[0]['data'].get('payload_id') == payload),
            unique_payload_fifo_enqueue_at_14400=bool(enqueue is not None and valid_batch
                and enqueue['data'].get('truck_id') == 0 and enqueue['sim_time_seconds'] == 14400.),
            load_enqueue_unload_sequence=bool(len(load) == len(unload) == 1 and enqueue is not None
                and load[0]['event_seq'] < enqueue['event_seq'] < unload[0]['event_seq']),
            valid_target_process_intervals=bool(valid_batch and intervals_valid),
            second_window_contribution_zero=bool(valid_batch and intervals_valid
                and all(v == 0. for v in second.values())),
            third_window_positive_processing=bool(valid_batch and intervals_valid
                and third['seconds'] > 0. and third['tons'] > 0.))
        coverage.update(target_payload_id=payload, target_batch_id=batch,
            target_payload_enqueues=[summary(e) for e in enqueues],
            target_batch_processing=processing,
            target_second_window_contribution=second,
            target_third_window_contribution=third)
    else:
        completions = [e for e in target_events if e['sim_time_seconds'] == 28800.
                       and e['event_type'] in ('LOAD_COMPLETE', 'UNLOAD_COMPLETE')]
        enqueues = exact('FIFO_ENQUEUE', 28800.)
        checks = dict(
            target_cutoff_exclusion_observed=bool(coverage['cutoff_load' if mode == 2 else 'cutoff_unload']),
            no_target_completion_at_cutoff=not completions,
            no_target_fifo_enqueue_at_cutoff=not enqueues)
        coverage.update(target_completions_at_cutoff=[summary(e) for e in completions],
                        target_fifo_enqueues_at_cutoff=[summary(e) for e in enqueues])
    coverage.update(checks=checks, miss_reasons=[name for name, hit in checks.items() if not hit],
                    scenario_targets_hit=all(checks.values()))
    return coverage


def independent_episode(rows, events, reward, physical):
    checks = []
    def check(name, actual, expected, absolute=0., relative=0., save=True):
        ok = math.isclose(actual, expected, abs_tol=absolute, rel_tol=relative)
        record = dict(name=name, actual=actual, expected=expected, residual=actual-expected,
                      abs_tol=absolute, rel_tol=relative, passed=ok)
        if save:
            checks.append(record)
        require(ok, json.dumps(record))
    mass = reward['mass_balance_epsilon']
    active, intervals, fifo, grades = {}, [], deque(), {}
    q, metal = [[] for _ in range(4)], [[] for _ in range(4)]
    enqueued, processed, dropped, waste = [], [], [], []
    windows, last_end = [], 0.
    for event in events:
        kind, d, now = event['event_type'], event['data'], event['sim_time_seconds']
        if kind == 'EPISODE_START':
            require(d['initial_fifo'] == d['initial_waits'] == d['initial_payloads'] == [], 'nonempty initial physical state')
        elif kind == 'WAIT_ENTER':
            key = (d['truck_id'], d['resource'], d['site_id'])
            require(not any(k[0] == key[0] for k in active), 'overlapping waiting')
            check('wait_enter_seconds', d['engine_queue_start_seconds'], now, 1e-7, 1e-10)
            active[key] = now
        elif kind == 'WAIT_EXIT':
            key = (d['truck_id'], d['resource'], d['site_id'])
            start = active.pop(key)
            check('wait_exit_duration', d['engine_wait_seconds'], now-start, 1e-7, 1e-10)
            intervals.append((start, now))
        elif kind == 'FIFO_ENQUEUE':
            require(d['batch_id'] not in grades and d['tons'] > 0, 'duplicate/empty batch')
            grades[d['batch_id']] = d['metal_tons']/d['tons']
            fifo.append(dict(id=d['batch_id'], remaining=d['tons'], entered=now))
            enqueued.append(d['tons'])
        elif kind == 'FIFO_PROCESS':
            require(fifo and fifo[0]['id'] == d['batch_id'], 'FIFO head mismatch')
            start, end, j = d['interval_start_seconds'], d['interval_end_seconds'], d['window_index']
            require(j == len(windows) and j*7200 <= start < end <= (j+1)*7200, 'physical window interval mismatch')
            check('continuous_fifo_time', start, max(last_end, fifo[0]['entered']), 1e-7, 1e-10, False)
            tons = (end-start)*physical['crusher_rate_tph']/3600.
            m = tons*grades[d['batch_id']]
            check('time_rate_tons', d['tons'], tons, mass, save=False)
            check('batch_grade_metal', d['metal_tons'], m, mass, save=False)
            require(d['tons'] <= fifo[0]['remaining'] + mass, 'batch overconsumed')
            fifo[0]['remaining'] -= d['tons']
            q[j].append(tons); metal[j].append(m); processed.append(d['tons'])
            last_end = end
        elif kind == 'FIFO_DROP':
            require(fifo and fifo[0]['id'] == d['batch_id'], 'FIFO drop head mismatch')
            remain = fifo.popleft()['remaining']
            require(-1e-12 <= remain <= 1e-9, 'non-numerical FIFO drop')
            check('fifo_drop', d['tons'], max(0., remain), 1e-12, save=False)
            dropped.append(d['tons'])
        elif kind == 'UNLOAD_COMPLETE' and d['material'] == 'waste':
            waste.append(d['tons'])
        elif kind == 'WINDOW_CLOSE':
            j = d['window_index']
            require(j == len(windows), 'window sequence mismatch')
            tons, m = math.fsum(q[j]), math.fsum(metal[j])
            grade = m/tons if tons > 0 else 0.
            cost = int(tons == 0 or grade < reward['grade_lower']-reward['grade_epsilon']
                       or grade > reward['grade_upper']+reward['grade_epsilon'])
            for key, actual, expected, eps in [('tons',d['tons'],tons,mass), ('metal',d['metal_tons'],m,mass),
                    ('grade',d['grade'],grade,1e-12), ('cost',d['cost'],cost,0.)]:
                check(f'window_{j}_{key}', actual, expected, eps)
            windows.append(dict(index=j, tons=tons, metal_tons=m, grade=grade, cost=cost))
    require(len(windows) == 4, 'four physical windows required')
    intervals.extend((start,28800.) for start in active.values())
    ore, waste_tons = math.fsum(w['tons'] for w in windows), math.fsum(waste)
    check('fifo_mass', math.fsum(enqueued), math.fsum(processed)+math.fsum(dropped)+math.fsum(b['remaining'] for b in fifo), mass)
    final = rows[-1]['next_info']['reward_components']
    check('terminal_ore', final['ore_tons_shift'], ore, mass)
    check('terminal_waste', final['waste_tons_shift'], waste_tons, mass)
    closed, output = 0, []
    for index, row in enumerate(rows):
        a, n = row['action_info'], row['next_info']
        begin, end = a['sim_time_seconds'], n['sim_time_seconds']
        terminal = index == len(rows)-1
        require(row['terminated'] == terminal and (end == 28800 if terminal else end < 28800), 'false terminal')
        delta = math.fsum(max(0.,min(y,end)-max(x,begin)) for x,y in intervals)
        cumulative = math.fsum(max(0.,min(y,end)-x) for x,y in intervals)
        c = n['reward_components']
        check(f'{index}:wait_delta', c['delta_wait_vehicle_seconds'], delta, 1e-7, 1e-10)
        check(f'{index}:wait_cumulative', c['wait_vehicle_seconds'], cumulative, 1e-7, 1e-10)
        penalties = (-reward['ore_weight']*max(0.,1-ore/16000), -reward['waste_weight']*max(0.,1-waste_tons/4000)) if terminal else (0.,0.)
        expected_r = -reward['kappa']*delta + sum(penalties)
        for key, expected in [('wait',-reward['kappa']*delta),('production',penalties[0]),('waste',penalties[1])]:
            check(f'{index}:{key}', c[key], expected, 1e-9, 1e-10)
        for key in ('formal_reward','raw_reward','env_reward'):
            check(f'{index}:{key}', row[key], expected_r, 1e-9, 1e-10)
        count = min(4,int(end//7200))
        settled = list(range(closed,count))
        require([w['window_index'] for w in n['closed_windows']] == settled, 'missing/duplicate window in interval')
        valid, bootstrap = [j >= closed for j in range(4)], [j >= count for j in range(4)]
        cost = [float(windows[j]['cost']) if j in settled else 0. for j in range(4)]
        expected = dict(grade_valid=valid, grade_bootstrap=bootstrap, grade_trace=bootstrap,
                        next_grade_valid=bootstrap, grade_costs=cost, reward_bootstrap=not terminal, reward_trace=not terminal)
        for key,value in expected.items():
            require(row[key] == value, f'{index}:{key} mismatch: {row[key]} != {value}')
        output.append(dict(step=index,start=begin,end=end,settled_windows=settled,expected=expected,
                           independent_wait=delta,expected_reward=expected_r,actual_reward=row['formal_reward']))
        closed = count
    waiting = math.fsum(y-x for x,y in intervals)
    total = -reward['kappa']*waiting-reward['ore_weight']*max(0.,1-ore/16000)-reward['waste_weight']*max(0.,1-waste_tons/4000)
    check('whole_shift_reward', math.fsum(r['formal_reward'] for r in rows), total, 1e-8, 1e-10)
    mc = [math.fsum(r['formal_reward'] for r in rows[t:]) for t in range(len(rows))]
    nonterminal_multi = [r['step'] for r in output if len(r['settled_windows']) > 1 and r['end'] < 28800]
    return dict(episode_id=rows[0]['episode_id'],transitions=len(rows),windows=windows,checks=checks,
                rows=output,wait_vehicle_seconds=waiting,ore_tons=ore,waste_tons=waste_tons,total_reward=total,
                independent_reward_mc=mc,window_costs=[w['cost'] for w in windows],
                coverage=dict(empty_windows=[w['index'] for w in windows if w['tons']==0],
                    nonterminal_multiwindow_steps=nonterminal_multi,
                    nonzero_ore_penalty=final['production'] < 0,
                    fallback_entries=sum(e['event_type']=='WAIT_ENTER' and e['data']['fallback'] for e in events),
                    cutoff_exclusions=sum(e['event_type']=='CUTOFF_EXCLUDED' for e in events)))


def audit_recording(recording, bundles, *, allow_synthetic=False, identity_reconciliation=None):
    recording, bundles = Path(recording), [Path(p) for p in bundles]
    inputs = [recording/n for n in ('acceptance_run_config.json','recording_manifest.json','rpc.jsonl','transitions.jsonl')]
    inputs += [p/n for p in bundles for n in ('events.jsonl','events_manifest.json')]
    if identity_reconciliation is not None:
        inputs.append(Path(identity_reconciliation))
    before = {str(p.resolve()):digest(p) for p in inputs}
    strict = replay_recording(recording,event_bundles=bundles,require_events=True,
                             allow_synthetic=allow_synthetic,
                             identity_reconciliation=identity_reconciliation)
    config = read(recording/'acceptance_run_config.json')
    all_rows = [r for r in lines(recording/'transitions.jsonl') if r['event']=='transition']
    event_map = {read(b/'events_manifest.json')['episode_id']:b for b in bundles}
    reports, bounds, cursor = [], [], 0
    for episode in strict['episodes']:
        eid = episode['episode_id']; bundle = event_map[eid]
        rows = [r for r in all_rows if r['episode_id']==eid]
        events = lines(bundle/'events.jsonl')
        result = independent_episode(rows,events,config['reward'],read(bundle/'events_manifest.json')['config'])
        if config.get('acceptance_scenario',{}).get('name')=='long_load_floor_v1':
            floor = config['acceptance_scenario']['expected_rpc']['config.acceptance.load_not_before_seconds']
            early = [e['event_seq'] for e in events if e['event_type'] in ('LOAD_COMPLETE','FIFO_ENQUEUE') and e['sim_time_seconds'] < floor]
            require(not early, f'long-load scene produced material before floor: {early}')
            upper = read(bundle/'events_manifest.json')['config']['crusher_rate_tph']*(28800-floor)/3600.
            require(result['ore_tons'] <= upper+config['reward']['mass_balance_epsilon'], 'long-load physical production bound exceeded')
            result['long_load_coverage'] = dict(first_two_empty=result['coverage']['empty_windows'][:2]==[0,1],
                exact_two_window_nonterminal_steps=[r['step'] for r in result['rows']
                    if r['start']<7200 and 14400<=r['end']<21600 and r['settled_windows']==[0,1]],
                ore_upper_bound=upper,nonzero_ore_penalty=result['coverage']['nonzero_ore_penalty'])
            result['scenario_targets_hit']=bool(result['long_load_coverage']['first_two_empty']
                and result['long_load_coverage']['exact_two_window_nonterminal_steps']
                and result['long_load_coverage']['nonzero_ore_penalty'])
        elif config.get('acceptance_scenario',{}).get('name')=='boundary_service_v1':
            mode=config['acceptance_scenario']['expected_rpc']['config.acceptance.boundary_mode']
            coverage=boundary_target_coverage(events,mode)
            result['boundary_coverage']=coverage
            result['scenario_targets_hit']=coverage['scenario_targets_hit']
        reports.append(result)
        bounds.append(dict(start=cursor,stop=cursor+len(rows),window_costs=result['window_costs']))
        cursor += len(rows)
    data = [dict(reward=r['formal_reward'],grade_valid=r['grade_valid']) for r in all_rows]
    yr, yg = mc_labels(data,bounds)
    expected = np.array([v for e in reports for v in e['independent_reward_mc']])
    require(np.allclose(yr,expected,atol=1e-8,rtol=1e-10),'MC suffix mismatch')
    for e,b in zip(reports,bounds):
        for i in range(b['start'],b['stop']):
            require(yg[i].tolist() == [c if m else 0. for c,m in zip(e['window_costs'],data[i]['grade_valid'])], 'MC grade mask mismatch')
    # Pure numerical diagnostic with V=0, gamma=lambda=1; no model/optimizer.
    signals = np.asarray([r['reward'] for r in data]); zero = np.zeros_like(signals)
    boot = np.asarray([r['reward_bootstrap'] for r in all_rows])
    trace = np.asarray([r['reward_trace'] for r in all_rows])
    limits = [(b['start'],b['stop']) for b in bounds]
    advantages,_ = gae(zero,zero,signals,boot,trace,np.ones_like(signals),limits,1.)
    require(np.allclose(advantages,expected,atol=1e-8,rtol=1e-10),'GAE episode suffix mismatch')
    grade_signal = np.asarray([r['grade_costs'] for r in all_rows])
    grade_zero = np.zeros_like(grade_signal)
    grade_valid = np.asarray([r['grade_valid'] for r in all_rows])
    grade_boot = np.asarray([r['grade_bootstrap'] for r in all_rows])
    grade_trace = np.asarray([r['grade_trace'] for r in all_rows])
    grade_adv,_ = gae(grade_zero,grade_zero,grade_signal,grade_boot,grade_trace,grade_valid,limits,1.)
    require(np.array_equal(grade_adv,yg),'grade GAE per-head boundary mismatch')
    isolation = None
    if len(bounds)>1:
        changed = copy.deepcopy(data); start = bounds[-1]['start']
        for row in changed[start:]: row['reward'] += 1000.
        shifted,_ = mc_labels(changed,bounds)
        perturbed = np.asarray([r['reward'] for r in changed])
        shifted_gae,_ = gae(zero,zero,perturbed,boot,trace,np.ones_like(signals),limits,1.)
        perturbed_grade = grade_signal.copy();perturbed_grade[start:] += 1000.
        shifted_grade,_ = gae(grade_zero,grade_zero,perturbed_grade,grade_boot,grade_trace,grade_valid,limits,1.)
        isolation = bool(np.array_equal(yr[:start],shifted[:start]) and np.array_equal(advantages[:start],shifted_gae[:start])
                         and np.array_equal(grade_adv[:start],shifted_grade[:start]))
        require(isolation,'cross-episode propagation')
    require(before == {str(p.resolve()):digest(p) for p in inputs},'inputs changed during audit')
    return dict(status='audited_complete_recording_not_stage1_certificate',source_kind=strict['scope'],
                episodes=reports,cross_episode_isolation=isolation,episode_bounds=limits,
                scenario_targets_hit=(all(e.get('scenario_targets_hit',False) for e in reports)
                    and (isolation is True if config.get('acceptance_scenario',{}).get('name')=='long_load_floor_v1' else True))
                    if config.get('acceptance_scenario') else None,
                source_sha256=before,inputs_unchanged=True,event_evidence_verified=True,
                identity_reconciliation_used=identity_reconciliation is not None,
                strict_identity_passed=identity_reconciliation is None,
                stage1_acceptance_passed=False,trainable_dataset=False,training_executed=False,
                limits=['Source reservation mutations still lack a full independent event journal.',
                        'Only observed scenarios are covered; synthetic input stays synthetic.',
                        'GAE uses diagnostic V=0/lambda=1, not formal training settings.'])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recording',type=Path,required=True)
    p.add_argument('--events',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--allow-synthetic',action='store_true')
    a = p.parse_args(argv)
    a.output.mkdir(parents=True,exist_ok=False)
    try:
        report = audit_recording(a.recording,a.events,allow_synthetic=a.allow_synthetic)
        (a.output/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    except Exception as exc:
        (a.output/'FAILED.json').write_text(json.dumps(dict(status='failed',error=str(exc),stage1_acceptance_passed=False),ensure_ascii=False,indent=2)+'\n')
        raise
    print(json.dumps(dict(episodes=len(report['episodes']),inputs_unchanged=True,cross_episode_isolation=report['cross_episode_isolation'])))


if __name__ == '__main__':
    main()
