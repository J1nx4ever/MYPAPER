"""Compare factual high-grade deliveries in matched held_order_1 s04/s05 evaluation slots."""

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PROJECT = next(p for p in HERE.parents if p.name == 'lagrangian_ppo_v1')
sys.path.insert(0, str(PROJECT))
from grade_event_memory_v1.events import build_memory
from grade_event_memory_v1.journal import read_journal

OUT = PROJECT / 'reports' / HERE.relative_to(PROJECT / 'change_plans')
CASE = 'held_order_1'


def sealed(ref):
    data = Path(ref['path']).read_bytes()
    if hashlib.sha256(data).hexdigest() != ref['sha256']:
        raise ValueError(f"Source changed: {ref['path']}")
    return json.loads(data)


def report(seed):
    name = f'bg128-gat-depth-20261125-full-{seed}'
    root = PROJECT / 'runs' / name
    completed = json.loads((root / 'suite_complete.json').read_bytes())
    index = sealed(completed['comparison'])
    ref = index['reports'][CASE]
    data = sealed(ref)
    if data['n_shifts'] != 100 or data['checkpoint_provenance'] != completed['new_checkpoint']:
        raise ValueError('Unexpected frozen evaluation')
    return root / 'evaluation' / CASE / 'new' / 'episodes', ref, data['per_episode']


def batches(folder, episode):
    ident = episode['identity']
    run = ident['run_index']
    receipt = json.loads((folder / f'run_{run}' / 'material_receipt.json').read_bytes())
    ref = receipt['references']['journal']
    tape = read_journal(ref['path'])
    if (tape.file_sha256 != ref['sha256'] or tape.file_sha256 != ident['journal_file_sha256']
            or receipt['identity'] != ident or tape.header['run_index'] != run):
        raise ValueError('Journal identity differs')
    build_memory(tape.events)
    out, processing = {}, defaultdict(list)
    stock = 0.0
    for e in tape.events:
        if e.kind == 'UNLOAD':
            deadline = (e.window+1)*7200
            if e.window < 3 and 0 <= deadline-e.time <= 900 and e.grade > 0.28:
                out[e.batch_id] = dict(run=run, batch=e.batch_id, source=e.source_id, truck=e.truck_id,
                        window=e.window+1, boundary_s=deadline, unload_s=e.time, tons=e.tons,
                        grade=e.grade, fifo_ahead_t=max(0.0, stock))
            stock += e.tons
        elif e.kind == 'PROCESS':
            processing[e.batch_id].append(e)
            stock -= e.tons
    for bid, row in out.items():
        ps = processing[bid]
        row['first_process_s'] = ps[0].interval_start if ps else None
        by_window = [math.fsum(e.tons for e in ps if e.window==w) for w in range(4)]
        row['actual_tons_by_window'] = by_window
        row['by_close_fraction'] = by_window[row['window']-1]/row['tons']
    return list(out.values())


def main():
    left, lref, la = report('s04')
    right, rref, ra = report('s05')
    pairs = []
    for k, (a, b) in enumerate(zip(la, ra, strict=True)):
        seeds = [a['identity'][i] for i in ('environment_seed','actor_sampling_seed')]
        if seeds != [b['identity'][i] for i in ('environment_seed','actor_sampling_seed')]:
            raise ValueError('Unpaired evaluation RNG')
        ls, rs = batches(left, a), batches(right, b)
        candidates = defaultdict(list)
        for l in ls:
            for r in rs:
                dt, dg = abs(l['unload_s']-r['unload_s']), abs(l['grade']-r['grade'])
                if (l['window'] == r['window'] and l['source'] == r['source']
                    and abs(l['tons']-r['tons']) <= .01 and dt <= 60 and dg <= .001):
                    candidates[l['window']].append((dt, dg, l['batch'], r['batch'], l, r))
        for w, options in candidates.items():
            # At most one pair per boundary per slot. Selection does not use processing outcomes.
            dt, dg, _, _, l, r = min(options, key=lambda item: item[:4])
            fractions = sorted((l['by_close_fraction'], r['by_close_fraction']))
            kind = 'one_all_one_none' if fractions[0]<1e-8 and fractions[1]>1-1e-8 else (
                    'one_at_least_80pct_one_at_most_20pct' if fractions[0]<=0.2 and fractions[1]>=0.8 else 'other')
            pairs.append(dict(ordinal=k, environment_actor_seeds=seeds, window=w, unload_gap_s=dt,
                grade_gap=dg, result=kind, full_s04=l, full_s05=r))
    counts = Counter(p['result'] for p in pairs)
    counts_by_order = {'s04_earlier':0, 's05_earlier':0}
    contrasts = [p for p in pairs if p['result'] != 'other']
    for p in contrasts:
        counts_by_order['s04_earlier' if p['full_s04']['by_close_fraction'] >
                        p['full_s05']['by_close_fraction'] else 's05_earlier'] += 1
    examples = sorted(contrasts, key=lambda p:(p['result'] != 'one_all_one_none', p['unload_gap_s'], p['grade_gap']))
    result = dict(scenario=CASE, shifts_per_checkpoint=100, boundaries_screened=300, matched=len(pairs),
        selection='Grade above 28%, unloaded in 900s before first three closes. Same source, physical mass within 0.01t, '
                  'grade within 0.1 percentage point and unload time within 60s. Closest unload time first, '
                  'then closest grade and stable batch IDs, one pair per boundary and dual-seed slot.',
        outcome='Fraction of batch actually processed in its unloading window. Factual trajectories of two trained '
                'Full policies. Matched deliveries need not be the same trip and other traffic and supply can differ.',
        input_evaluation_reports=[lref,rref], counts=dict(counts), direction_in_contrasts=counts_by_order,
        illustrative_pair=examples[0] if examples else None, all_pairs=pairs, new_physical_shifts=0)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'matched_batches.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('all_pairs','input_evaluation_reports')},ensure_ascii=False))


if __name__ == '__main__':
    main()
