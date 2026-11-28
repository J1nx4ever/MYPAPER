"""Replay actual settled ore windows and trace unloaded batches through the crusher."""

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
from math import isclose, sqrt
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PROJECT = next(p for p in HERE.parents if p.name == 'lagrangian_ppo_v1')
sys.path.insert(0, str(PROJECT))
from grade_event_memory_v1.events import build_memory
from grade_event_memory_v1.journal import read_journal

OUT = PROJECT / 'reports' / HERE.relative_to(PROJECT / 'change_plans')
SUMMARY = OUT.parent / 'gat_weak_ablation_001/seed_extension_001/paired_evaluation_summary.json'
SUMMARY_SHA = '3d7483da4ac12919db37108a1d8992f4ec111faed00530ff12c33a024bdc9dcb'
EXAMPLE_RUN = 10050673


def require(condition, message):
    if not condition:
        raise ValueError(message)


def frozen(ref):
    data = Path(ref['path']).read_bytes()
    require(sha256(data).hexdigest() == ref['sha256'], f"frozen SHA differs at {ref['path']}")
    return json.loads(data)


def phase_result(item):
    seed, scenario, report_ref, checkpoint = item
    report_path = Path(report_ref['path'])
    report = frozen(report_ref)
    require(report['scenario_id'] == scenario and report['policy_arm'] == 'new'
            and report['checkpoint_provenance'] == checkpoint and report['training_state_unchanged']
            and report['evaluation_optimizer_updates'] == 0 and report['n_shifts'] == 100,
            'fixed-policy scope differs')
    counts = Counter()
    run_ids, seeds = [], []
    example = None
    for episode, mref in zip(report['per_episode'], report['source_refs'], strict=True):
        ident = episode['identity']
        require(frozen(mref)['identity'] == ident, 'material identity differs')
        run = ident['run_index']
        run_ids.append(run)
        seeds.append((ident['environment_seed'], ident['actor_sampling_seed']))
        receipt = json.loads((report_path.parent / 'episodes' / f'run_{run}' / 'material_receipt.json').read_bytes())
        src = receipt['references']['journal']
        require(receipt['identity'] == ident and receipt['references']['episode_manifest'] == mref
                and receipt['normal_completion'] and src['sha256'] == ident['journal_file_sha256'],
                'physical receipt differs from the sealed evaluation')
        journal = read_journal(src['path'])
        h = journal.header
        require(journal.file_sha256 == src['sha256'] and journal.logical_sha256 == ident['journal_logical_sha256']
                and h['run_index'] == run and h['seed'] == ident['environment_seed']
                and h['recording_id'] == ident['recording_id'] and h['config_sha256'] == ident['config_sha256']
                and h['source_kind'] == 'actual_anylogic' and h['target_grade'] == .27
                and h['grade_lower'] == .26 and h['grade_upper'] == .28, 'unexpected journal identity')
        tape = build_memory(journal.events)
        windows = episode['physical']['windows']
        require(len(tape.settlements) == len(windows) == 4, 'missing actual window')
        for settled, w in zip(tape.settlements, windows, strict=True):
            require(settled.window == w['window'] and bool(settled.grade_valid) != bool(w['empty'])
                    and isclose(settled.tons, w['tons'], abs_tol=1e-7, rel_tol=1e-10)
                    and isclose(settled.metal, w['metal_tons'], abs_tol=1e-7, rel_tol=1e-10),
                    'window or actual mass reconciliation failed')
            counts['windows'] += 1
            if settled.grade_valid:
                g = settled.metal / settled.tons
                require(isclose(g, w['grade'], abs_tol=1e-10), 'settled grade differs from report')
                counts['squared_grade_error'] += (g - h['target_grade'])**2
                counts['out_of_band_nonempty_windows'] += not (h['grade_lower']-1e-12 <= g <= h['grade_upper']+1e-12)
            else:
                counts['empty_windows'] += 1
            counts['noncompliant_windows'] += not w['compliant']
        counts['four_window_compliant_shifts'] += all(w['compliant'] for w in windows)
        unloaded = {e.batch_id: e for e in journal.events if e.kind == 'UNLOAD'}
        received = sum(e.tons for e in unloaded.values())
        local = Counter()
        batches_by_window = defaultdict(Counter)
        first_processed = {}
        for e in journal.events:
            if e.kind != 'PROCESS':
                continue
            u = unloaded[e.batch_id]
            local['actual_crushed_tons'] += e.tons
            batches_by_window[e.batch_id][e.window] += e.tons
            first_processed.setdefault(e.batch_id, e.interval_start)
            counts[f'unload_W{u.window+1}_crush_W{e.window+1}_tons'] += e.tons
            if e.window > u.window:
                local['previous_window_unload_tons_crushed'] += e.tons
                if u.grade > h['target_grade']:
                    local['previous_window_high_grade_tons_crushed'] += e.tons
        split = sum(u.grade > h['target_grade'] and batches_by_window[key][u.window] > 0
                    and any(w > u.window for w in batches_by_window[key]) for key, u in unloaded.items())
        residual = received - local['actual_crushed_tons']
        require(residual > -1e-6, 'negative end pocket')
        local.update(dict(shifts=1, actual_unloaded_tons=received, unprocessed_at_shift_end_tons=max(0., residual),
                          cross_window_shifts=int(local['previous_window_unload_tons_crushed'] > 0),
                          high_split_batches=int(split), shifts_with_high_split_batch=int(split > 0)))
        counts.update(local)
        if run == EXAMPLE_RUN:
            batch = 'batch-24'
            u = unloaded[batch]
            example = dict(seed=seed, scenario=scenario, run=run, journal=src, batch_id=batch,
                           source_id=u.source_id, truck_id=u.truck_id, unloaded_s=u.time, batch_tons=u.tons,
                           batch_grade_percent=100*u.grade, first_crushed_s=first_processed[batch],
                           actual_tons_by_window={f'W{k+1}': v for k, v in batches_by_window[batch].items()},
                           physical_window_grade_percent=[100*w['grade'] for w in windows],
                           unprocessed_at_shift_end_tons=residual)
    require(len(run_ids) == len(set(run_ids)) == len(set(seeds)) == 100, 'nonunique or missing evaluation slots')
    return seed, scenario, counts, example, run_ids, seeds


def stats(c):
    n = c['shifts']
    return dict(shifts=n, physical_windows=c['windows'], nonempty_windows=c['windows']-c['empty_windows'],
                empty_windows=c['empty_windows'], out_of_band_nonempty_windows=c['out_of_band_nonempty_windows'],
                noncompliant_windows=c['noncompliant_windows'], four_window_compliant_shifts=c['four_window_compliant_shifts'],
                four_window_compliance_rate=c['four_window_compliant_shifts']/n,
                actual_grade_rmse_pp=100*sqrt(c['squared_grade_error']/(c['windows']-c['empty_windows'])),
                actual_window_noncompliance_rate=c['noncompliant_windows']/c['windows'],
                cross_window_tons_total=c['previous_window_unload_tons_crushed'],
                cross_window_tons_per_shift=c['previous_window_unload_tons_crushed']/n,
                cross_window_fraction_of_processed=c['previous_window_unload_tons_crushed']/c['actual_crushed_tons'],
                high_grade_cross_window_tons_per_shift=c['previous_window_high_grade_tons_crushed']/n,
                unprocessed_at_shift_end_tons_per_shift=c['unprocessed_at_shift_end_tons']/n,
                cross_window_shifts=c['cross_window_shifts'], high_split_batches=c['high_split_batches'],
                shifts_with_high_split_batch=c['shifts_with_high_split_batch'],
                unloaded_tons_total=c['actual_unloaded_tons'], crushed_tons_total=c['actual_crushed_tons'],
                unload_to_crush_window_tons={k:v for k,v in sorted(c.items()) if k.startswith('unload_W')})


def main():
    source = dict(path=str(SUMMARY), sha256=SUMMARY_SHA)
    index = frozen(source)
    held = index['scopes']['held_out']
    jobs = []
    for seed, refs in index['sources'].items():
        complete = frozen(refs['full_completion'])
        checkpoint = complete['new_checkpoint']
        require(complete['status'] == 'COMPLETE' and complete['training_updates'] == 32
                and checkpoint == refs['full_checkpoint'], 'Full/u32 source differs')
        for phase in complete['phase_results']:
            ref = phase['report']
            path = Path(ref['path'])
            if phase['checkpoint'] == checkpoint and path.parent.name == 'new' and path.parents[1].name in held:
                jobs.append((seed, path.parents[1].name, ref, checkpoint))
    require(len(jobs) == 25 and len({(seed, sc) for seed,sc,*_ in jobs}) == 25, 'incomplete Full panel')
    per_seed, per_case, total = defaultdict(Counter), defaultdict(dict), Counter()
    all_runs, reference_slots, example = set(), None, None
    with ProcessPoolExecutor(max_workers=4) as pool:
        for seed, case, counts, event, runs, slots in pool.map(phase_result, jobs):
            if reference_slots is None:
                reference_slots = slots
            require(slots == reference_slots and not all_runs.intersection(runs), 'paired panel differs')
            all_runs.update(runs)
            per_case[seed][case] = stats(counts)
            per_seed[seed].update(counts)
            total.update(counts)
            if event:
                example = event
    require(len(all_runs) == 2500 and example is not None, 'missing runs or designated example')
    doc = dict(source_summary=source, policy='Full mixed u32, s01-s05',
               verified='sealed phase, material manifest, physical identities and binary digests; FIFO and four actual window settlements',
               rmse_definition='sqrt(mean((actual crushed metal / actual crushed tons - 0.27)**2)) across nonempty physical windows, then multiply by 100; noncompliant and empty windows are reported separately',
               cross_window_definition='PROCESS tons by batch with physical crush window later than its UNLOAD window; each processed ton counted once, independent of interval publication time',
               high_grade_definition='actual UNLOAD batch grade > 0.27',
               high_split_definition='positive crushed tons from one high-grade batch in both its own unloading window and a later window',
               residual_definition='actual unloaded ore minus actual crushed ore at shift end; excludes trucks in transit',
               total=stats(total), by_training_seed={seed:stats(c) for seed,c in per_seed.items()},
               by_seed_and_case=per_case, physical_split_example=example, evaluation_shift_count_added=0)
    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / 'material_replay.json'
    output.write_text(json.dumps(doc, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)+'\n')
    print(output)


if __name__ == '__main__':
    main()
