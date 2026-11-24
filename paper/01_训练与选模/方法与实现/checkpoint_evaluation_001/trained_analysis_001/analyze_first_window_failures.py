"""Bounded read-only material-flow diagnosis; no model or alternate rollout."""
from pathlib import Path
import json
import hashlib
import csv
import math
from collections import Counter
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
import numpy as np
from grade_event_memory_v1.journal import read_journal
from python.constrained_batch import load_dataset

PLAN = ROOT / 'change_plans/token_attention_001/business_graph_summary_009/checkpoint_evaluation_001'
REPORT = ROOT / 'reports/token_attention_001/business_graph_summary_009/checkpoint_evaluation_001/trained_analysis_001'
RUNS = (9950012, 9950029)


def ref(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def read(path):
    return json.loads(Path(path).read_bytes())


def aggregate(events):
    q, m = math.fsum(e.tons for e in events), math.fsum(e.metal for e in events)
    bins = {name: math.fsum(e.tons for e in events if predicate(e.grade)) for name, predicate in
            [('below_026', lambda g: g < .26), ('in_026_028', lambda g: .26 <= g <= .28),
             ('above_028', lambda g: g > .28), ('below_center027', lambda g: g < .27)]}
    return dict(tons=q, metal_tons=m, grade=m/q if q else None, records=len(events),
                **bins, low_share=bins['below_026']/q if q else None)


def source_mix(events):
    total = math.fsum(e.tons for e in events)
    return {str(i): dict(aggregate([e for e in events if e.source_id == i]),
                        tons_share=math.fsum(e.tons for e in events if e.source_id == i)/total if total else 0)
            for i in range(4)}


def dist(values):
    values = np.array(values, dtype=float)
    return dict(n=len(values), mean=float(values.mean()), minimum=float(values.min()),
        p5=float(np.quantile(values,.05)), p25=float(np.quantile(values,.25)), median=float(np.median(values)),
        p75=float(np.quantile(values,.75)), p95=float(np.quantile(values,.95)), maximum=float(values.max()))


def fifo_at(events, seq):
    batches = {}
    for e in events:
        if e.seq > seq:
            break
        if e.kind == 'UNLOAD':
            batches[e.batch_id] = dict(batch_id=e.batch_id, source_id=e.source_id, truck_id=e.truck_id,
                unload_time=e.time, original_tons=e.tons, remaining_tons=e.tons, grade=e.grade)
        elif e.kind == 'PROCESS':
            batches[e.batch_id]['remaining_tons'] -= e.tons
    remaining = [dict(b, remaining_tons=max(0., b['remaining_tons'])) for b in batches.values()
                 if b['remaining_tons'] > 1e-6]
    q = math.fsum(b['remaining_tons'] for b in remaining)
    m = math.fsum(b['remaining_tons'] * b['grade'] for b in remaining)
    return dict(tons=q, metal_tons=m, grade=m/q if q else None, batches=remaining)


def main():
    suite = read(PLAN/'prepared_001/suite.json')
    output = Path(suite['python_output'])
    audit = read(ROOT/'reports/token_attention_001/business_graph_summary_009/checkpoint_evaluation_001/completion_independent_audit.json')
    prior = {r['run_index']: r for r in audit['per_run']}
    journals, cohort = {}, []
    window_csv = output/'reports/per_window.csv'
    with window_csv.open() as stream:
        windows = list(csv.DictReader(stream))
    for job in suite['jobs']:
        run = job['run_index']
        assert ref(job['material_journal']) == prior[run]['material_journal']
        journal = read_journal(job['material_journal'])
        journals[run] = journal
        process = [e for e in journal.events if e.kind == 'PROCESS' and e.window == 0]
        unload = [e for e in journal.events if e.kind == 'UNLOAD' and e.window == 0]
        close = next(e for e in journal.events if e.kind == 'WINDOW_CLOSE' and e.window == 0)
        p, u = aggregate(process), aggregate(unload)
        assert math.isclose(p['tons'], close.window_q, abs_tol=1e-6)
        assert math.isclose(p['metal_tons'], close.window_m, abs_tol=1e-6)
        w = next(w for w in windows if int(w['run_index']) == run and int(w['window']) == 0)
        assert math.isclose(p['grade'], float(w['grade']), abs_tol=1e-10)
        fifo = fifo_at(journal.events, close.seq)
        assert math.isclose(u['tons'] - p['tons'], fifo['tons'], abs_tol=1e-6)
        cohort.append(dict(run_index=run, processed=p, unloaded=u, process_by_load=source_mix(process),
            unload_by_load=source_mix(unload), end_fifo_tons=fifo['tons'], end_fifo_grade=fifo['grade'],
            compliant=.26-1e-12 <= p['grade'] <= .28+1e-12))
    schema = read(ROOT/'python/observation_schema.json')
    fields = schema['fields'] if isinstance(schema, dict) else schema
    names = {f['name']: f['index'] for f in fields}
    details = []
    for run in RUNS:
        job = next(j for j in suite['jobs'] if j['run_index'] == run)
        raw = output/'evaluation/episodes'/f'run_{run}'
        rows, episodes, *_ = load_dataset(raw, require_real=True)
        journal = journals[run]
        decision_seq = {e.decision_sequence: e.seq for e in journal.events if e.kind == 'DECISION'}
        actions = [dict(decision_sequence=i, time=float(r['action_time_seconds']), truck_id=int(r['truck_id']),
            load=int(r['action']), candidate_tons=float(r['observation'][23+int(r['action'])])*193.,
            journal_seq=decision_seq[i])
            for i, r in enumerate(rows) if r['action_time_seconds'] < 7200]
        state = rows[len(actions)-1]['observation']
        time = actions[-1]['time']
        loaded = []
        for tid in range(9):
            get = lambda field: float(state[names[f'truck.{tid}.{field}']])
            loaded_flag = get('loaded') > .5
            # The schema calls this Boolean simply `ore`.
            ore_name = next(f['name'].split('.',2)[2] for f in fields if f['name'].startswith(f'truck.{tid}.')
                            and 'payloadIsOre' in f['formula'])
            ore = get(ore_name) > .5
            if loaded_flag and ore:
                loaded.append(dict(truck_id=tid, tons=get('payloadTons')*193,
                    grade=get('payloadGrade') if get('gradeKnown') > .5 else None,
                    phase=int(np.argmax([get('phase.'+str(p)) for p in range(12)]))))
        reserves = {str(i): float(state[names[f'load.{i}.reservedTons']])*30000 for i in range(4)}
        initial = rows[0]['observation']
        load_config = {str(i): {k: float(initial[names[f'load.{i}.{k}']]) for k in
                        ('isOre','meanGrade','gradeHalfRange')} for i in range(4)}
        process = [e for e in journal.events if e.kind == 'PROCESS' and e.window == 0]
        unload = [e for e in journal.events if e.kind == 'UNLOAD' and e.window == 0]
        close = next(e for e in journal.events if e.kind == 'WINDOW_CLOSE' and e.window == 0)
        marker = next(e for e in journal.events if e.kind == 'DECISION' and e.decision_sequence == len(actions)-1)
        last_fifo = fifo_at(journal.events, marker.seq)
        terminal_fifo = fifo_at(journal.events, close.seq)
        timeline = []
        for cutoff in (1800,3600,5400,6300,6600,6900,7200):
            p = aggregate([e for e in process if e.time <= cutoff+1e-8])
            u = aggregate([e for e in unload if e.time <= cutoff+1e-8])
            timeline.append(dict(visible_by_seconds=cutoff, processed=p, unloaded=u,
                note='visibility-time prefix, no interpolation within aggregated PROCESS records'))
        joins = []
        for e in unload:
            candidates = [a for a in actions if a['truck_id'] == e.truck_id and a['journal_seq'] < e.seq]
            a = candidates[-1] if candidates else None
            joins.append(dict(batch_id=e.batch_id, truck_id=e.truck_id, source_id=e.source_id,
                unload_time=e.time, tons=e.tons, grade=e.grade,
                dispatch_time=a['time'] if a else None,
                dispatch_to_unload_seconds=e.time-a['time'] if a else None,
                latest_dispatch_source_matches=(a['load'] == e.source_id) if a else None))
        current = next(r for r in cohort if r['run_index'] == run)
        deficit = max(0., .26*current['processed']['tons']-current['processed']['metal_tons'])
        before_last = aggregate([e for e in process if e.seq <= marker.seq])
        after_last = aggregate([e for e in process if e.seq > marker.seq])
        late_marker = next(e for e in journal.events if e.kind == 'DECISION' and 6300 <= e.time < 7200)
        late_before = aggregate([e for e in process if e.seq <= late_marker.seq])
        late_after = aggregate([e for e in process if e.seq > late_marker.seq])
        last15 = aggregate([e for e in process if e.interval_start >= 6300-1e-8])
        max_observed_grade = max(e.grade for e in unload)
        action_counts = {str(i): sum(a['load'] == i for a in actions) for i in range(4)}
        action_tons = {str(i): math.fsum(a['candidate_tons'] for a in actions if a['load'] == i) for i in range(4)}
        details.append(dict(**current, first_window_rank_ascending=1+sum(r['processed']['grade'] < current['processed']['grade'] for r in cohort),
            low_processed_share_percentile=100*sum(r['processed']['low_share'] <= current['processed']['low_share'] for r in cohort)/100,
            end_fifo=terminal_fifo, load_configuration=load_config,
            first_window_actions=actions, action_counts=action_counts, dispatched_candidate_tons_approx=action_tons,
            last_pre_boundary_decision=dict(time=time, seconds_remaining=7200-time,
                action=actions[-1], observed_loaded_ore=loaded,
                observed_loaded_ore_tons=math.fsum(t['tons'] for t in loaded),
                load_reserved_tons_approx=reserves, fifo_at_this_decision=last_fifo,
                note='pre-action sampled snapshot, not an exact7200 truck/reservation census; loaded flag includes waiting/service as well as road transit'),
            first_window_timeline=timeline, last900_process_records_wholly_in_interval=last15,
            exact_late_decision_prefix=dict(time=late_marker.time, processed_before=late_before,
                processed_after_until_first_window_close=late_after,
                note='exact journal sequence barrier; uses only first-window physical processing records'),
            last_decision_to_window_close_accounting=dict(processed_before_last_decision=before_last,
                actual_processed_after_last_decision=after_last,
                nominal_remaining_processing_capacity_at_2500tph=(7200-time)*2500/3600,
                required_grade_for_observed_remaining_tonnage_to_finish026=(.26*current['processed']['tons']-before_last['metal_tons'])/after_last['tons'],
                no_future_unloads_included=True),
            last900_unloads=[j for j in joins if j['unload_time'] >= 6300],
            dispatch_to_unload_observed_seconds=dist([j['dispatch_to_unload_seconds'] for j in joins if j['latest_dispatch_source_matches'] is True]),
            dispatch_join_missing_or_source_mismatch=sum(j['latest_dispatch_source_matches'] is not True for j in joins),
            processed_metal_deficit_to_026=deficit,
            hindsight_extra_tons_at_best_observed_grade_to_026=deficit/(max_observed_grade-.26),
            best_observed_first_window_unload_grade=max_observed_grade,
            source_refs=dict(raw_manifest=ref(raw/'manifest.json'), raw_samples=ref(raw/'samples.h5'),
                material_journal=prior[run]['material_journal'])))
        rows.close()
    result = dict(schema='bg128-first-window-failure-flow-diagnosis-v1', status='PASS_READ_ONLY',
        scope=dict(checkpoint_update=13, shifts_compared=100, detailed_failure_shifts=list(RUNS),
            model_loads=0, new_simulations=0, parameter_updates=0),
        first_window_distribution=dict(grade=dist([r['processed']['grade'] for r in cohort]),
            processed_tons=dist([r['processed']['tons'] for r in cohort]),
            processed_low_grade_mass_share=dist([r['processed']['low_share'] for r in cohort]),
            end_fifo_tons=dist([r['end_fifo_tons'] for r in cohort]),
            compliant_shifts=sum(r['compliant'] for r in cohort),
            low_grade_threshold=.26, high_grade_threshold=.28,
            process_share_by_load={str(i): dist([r['process_by_load'][str(i)]['tons_share'] for r in cohort]) for i in range(4)}),
        failures=details, all100_first_windows=cohort,
        limitations=['Material journal UNLOAD/PROCESS records describe ore entering crusher FIFO; Load3 waste is not silently treated as missing ore.',
            'No LOAD_COMPLETE event; loading completion time, exact assignment-to-payload realized grade time, and true ETA are unavailable.',
            'Actions contain candidate reservation tons at dispatch (float32); cumulative dispatch flow is not outstanding stock and is not added to FIFO or loaded mass.',
            'Last pre-boundary snapshot is earlier than7200; exact truck/reservation census at7200 cannot be reconstructed from this sampling alone.',
            'Loaded ore cannot always be attributed to its original Load using current observation; no future UNLOAD used to relabel that snapshot.',
            'All-unloaded grade and best-observed-grade extra tons are hindsight accounting, not causal alternatives or feasibility estimates.',
            'PROCESS may be aggregated; intermediate time-cutoff prefixes use visibility time, not interpolation or assumed exact physical quantities at arbitrary cutoff.',
            'Only two failures; descriptive comparison does not establish policy/reward/representation causal attribution.'],
        references=dict(suite=ref(PLAN/'prepared_001/suite.json'),
            completion_audit=ref(ROOT/'reports/token_attention_001/business_graph_summary_009/checkpoint_evaluation_001/completion_independent_audit.json'),
            per_window=ref(window_csv), observation_schema=ref(ROOT/'python/observation_schema.json'), implementation=ref(__file__)))
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT/'first_window_failures.json').write_text(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],distribution=result['first_window_distribution'],
        failures=[{k:r[k] for k in ('run_index','processed','unloaded','end_fifo','first_window_rank_ascending','last_pre_boundary_decision','action_counts','processed_metal_deficit_to_026','hindsight_extra_tons_at_best_observed_grade_to_026')} for r in details]),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
