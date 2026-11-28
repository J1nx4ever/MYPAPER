"""Recompute paired effects from the five archived fixed-u32 comparisons."""

from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import t


HERE = Path(__file__).resolve().parent
PROJECT = next(p for p in HERE.parents if p.name == 'lagrangian_ppo_v1')
OUTPUT = PROJECT / 'reports' / HERE.relative_to(PROJECT / 'change_plans')
SUMMARY = OUTPUT.parent / 'paired_evaluation_summary.json'
FROZEN_SHA = '3d7483da4ac12919db37108a1d8992f4ec111faed00530ff12c33a024bdc9dcb'
PRIMARY = ('complete_task', 'total_wait_seconds')
SEEDS = ('s01', 's02', 's03', 's04', 's05')
PANELS = ['held_order_1', 'held_order_2', 'new_numeric', 'joint_shift', 'waste_swap', 'nominal', 'grade_swap']
BOOTSTRAP_SEED = 20261127
REPS = 50_000


def read_frozen(path, checksum):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != checksum:
        raise ValueError(f'Invalid source digest for {path}')
    return json.loads(data)


def load_panel(summary):
    metrics = list(summary['summaries']['held_out'])
    # Axes are matched training schedule, fixed scenario, paired RNG slot, algorithm, endpoint.
    panel = np.full((5, 7, 100, 2, len(metrics)), np.nan)
    rng_slots, runs = {}, set()
    for i, seed in enumerate(SEEDS):
        source = summary['sources'][seed]['paired_evaluation']
        comparison = read_frozen(source['path'], source['sha256'])
        allocated = set()
        for row in comparison['rows']:
            j, k = PANELS.index(row['scenario']), row['pair_ordinal']
            ids = [row['full_run'], row['ablation_run']]
            if not 0 <= k < 100 or (j, k) in allocated or len(set(ids)) != 2 or runs.intersection(ids):
                raise ValueError('Unexpected or duplicate evaluation slot')
            allocated.add((j, k))
            runs.update(ids)
            pair = row['environment_seed'], row['actor_sampling_seed']
            if rng_slots.setdefault(k, pair) != pair:
                raise ValueError('Environment or Actor seed differs across policies, scenarios or schedules')
            for a, arm in enumerate(('full', 'ablation')):
                panel[i, j, k, a] = [row[arm][name] for name in metrics]
        if len(allocated) != 700:
            raise ValueError(f'Incomplete panel for {seed}')
    if len(runs) != 7000 or len(set(rng_slots.values())) != 100:
        raise ValueError('Incorrect physical run or seed inventory')
    return metrics, panel


def paired_statistics(arm_means):
    if not np.isfinite(arm_means).all():
        raise ValueError('Nonfinite seed-level mean')
    d = arm_means[:, 1] - arm_means[:, 0]
    estimate = float(d.mean())
    se = d.std(ddof=1) / np.sqrt(len(d))
    half = t.ppf(0.975, df=len(d) - 1) * se
    signs = np.asarray(list(itertools.product((-1, 1), repeat=len(d))))
    flipped = np.abs((signs * d).mean(axis=1))
    equal = np.isclose(flipped, abs(estimate), rtol=1e-12, atol=1e-13)
    p = float(((flipped > abs(estimate)) | equal).mean())
    loo = [(d.sum() - x) / (len(d) - 1) for x in d]
    return dict(full=float(arm_means[:, 0].mean()), gat1=float(arm_means[:, 1].mean()), delta=estimate,
                seed_delta=d.tolist(), seed_arm_means=arm_means.tolist(),
                seed_sd_by_arm=arm_means.std(axis=0, ddof=1).tolist(), paired_t95=[estimate-half, estimate+half],
                exact_two_sided_sign_flip_p=p, leave_one_seed_out_delta=loo)


def conditional_intervals(slot_d, names):
    # The input contains one matched difference per dual-seed slot, averaged over fixed
    # scenarios and checkpoints. Redrawing the slot therefore preserves all repeated measures.
    if slot_d.shape != (100, len(names)) or not np.isfinite(slot_d).all():
        raise ValueError('Incomplete or unpaired bootstrap endpoints')
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    batches = []
    for start in range(0, REPS, 1000):
        which = rng.integers(100, size=(min(REPS-start, 1000), 100))
        batches.append(slot_d[which].mean(axis=1))
    limits = np.quantile(np.concatenate(batches), (0.025, 0.975), axis=0)
    return {name: ci.tolist() for name, ci in zip(names, limits.T, strict=True)}


def single_seed_ablations(summary):
    source = summary['sources']['s01']['paired_evaluation']
    reference = read_frozen(source['path'], source['sha256'])
    full = {(r['scenario'], r['pair_ordinal']): r for r in reference['rows']}
    reports = {
        'structural_ablation_001/flat-s01': 'c89d097933585ceebaa07cb7faf5e67c8b45d53337363a94218c47cb57cb97d8',
        'structural_ablation_001/noc128-s01': '511bd2b02be551f16344bdd1e7c96cb8da9f482acaf6f5b8486fa242581a297e',
        'gat_weak_ablation_001/nodump-s01': 'd5718dc4a3e6393d50a80764ff34380daa0d9c0e90434940d8851c22ac39436e',
    }
    names = (*PRIMARY, 'all_four_windows_compliant', 'ore_target_met', 'waste_target_met')
    output = {}
    for unit, checksum in reports.items():
        path = OUTPUT.parents[2] / unit / 'paired_evaluation.json'
        report = read_frozen(path, checksum)
        current = {(r['scenario'], r['pair_ordinal']): r for r in report['rows']}
        if len(report['rows']) != 700 or current.keys() != full.keys():
            raise ValueError(f'Evaluation slots differ for {unit}')
        for slot, row in current.items():
            if any(row[key] != full[slot][key] for key in ('full', 'full_run', 'environment_seed', 'actor_sampling_seed')):
                raise ValueError(f'Full reference or dual seed differs for {unit}/{slot}')
        differences = np.array([[
            [float(current[sc, k]['ablation'][name]) - float(current[sc, k]['full'][name]) for name in names]
            for k in range(100)] for sc in PANELS[:5]])
        output[unit] = dict(source=dict(path=str(path), sha256=checksum),
            held_out={name: report['scope']['held_out'][name] for name in names},
            fixed_checkpoint_slot_percentile95=conditional_intervals(differences.mean(axis=0), names))
    return output


def failure_counts(heldout, metrics):
    keys = ['ore_target_met', 'waste_target_met', 'all_four_windows_compliant', 'complete_task']
    values = heldout[..., [metrics.index(key) for key in keys]]
    if not np.isin(values, [0, 1]).all():
        raise ValueError('Incomplete binary outcomes')
    ore, waste, grade, success = np.moveaxis(values.astype(bool), -1, 0)
    if not ore.all() or not np.array_equal(ore & waste & grade, success):
        raise ValueError('Failure decomposition requires all ore targets met and the original task conjunction')
    failure = {}
    for arm, name in enumerate(('full', 'gat1')):
        w, g = waste[..., arm], grade[..., arm]
        failure[name] = dict(waste_only=int((~w & g).sum()), grade_only=int((w & ~g).sum()),
                             both=int((~w & ~g).sum()), complete=int((w & g).sum()))
    pairs = Counter(tuple(row) for row in success.reshape(-1, 2))
    return failure, {'both_pass': pairs[True, True], 'only_full_pass': pairs[True, False],
                     'only_gat1_pass': pairs[False, True], 'both_fail': pairs[False, False]}


def main():
    summary = read_frozen(SUMMARY, FROZEN_SHA)
    if summary['training_seed_schedules'] != list(SEEDS) or summary['scopes']['held_out'] != PANELS[:5]:
        raise ValueError('Unexpected held-out scenarios or training schedule')
    metrics, panel = load_panel(summary)
    groups = {'held_out': list(range(5)), **{s: [j] for j, s in enumerate(PANELS)}}
    results = {}
    for group, scenarios in groups.items():
        original = summary['summaries']['held_out'] if group == 'held_out' else summary['scenarios'][group]
        results[group] = {}
        for m, key in enumerate(metrics):
            v = panel[..., m][:, scenarios]
            finite_pairs = np.isfinite(v).all(axis=-1)
            valid = finite_pairs.sum(axis=(1, 2))
            if not valid.all():
                raise ValueError(f'No valid pairs for {group} {key}')
            means = np.where(finite_pairs[..., None], v, 0).sum(axis=(1, 2)) / valid[:, None]
            for i, seed in enumerate(SEEDS):
                ref = original[key]['by_training_seed'][seed]
                if valid[i] != ref['paired_n'] or not np.allclose(means[i], [ref['full'], ref['ablation']],
                                                                 rtol=1e-12, atol=1e-12):
                    raise ValueError(f'Aggregation differs from archived pair: {group}/{key}/{seed}')
            results[group][key] = dict(paired_n_by_schedule=valid.tolist(), **paired_statistics(means))
    p = {key: results['held_out'][key]['exact_two_sided_sign_flip_p'] for key in PRIMARY}
    a, b = sorted(PRIMARY, key=p.get)
    holm = {a: min(1.0, 2*p[a]), b: min(1.0, max(2*p[a], p[b]))}
    fail, discordance = failure_counts(panel[:, :5], metrics)
    differences = panel[:, :5, :, 1, :] - panel[:, :5, :, 0, :]
    primary_slots = differences.mean(axis=(0, 1))[:, [metrics.index(name) for name in PRIMARY]]
    report = dict(source_summary=dict(path=str(SUMMARY), sha256=FROZEN_SHA), source_pairs=summary['sources'],
                  training_schedules=SEEDS, unique_evaluation_rng_pairs=100, physical_evaluation_runs=7000,
                  direction='GAT1 minus Full', primary=PRIMARY, groups=results, primary_holm_p=holm,
                  fixed_checkpoint_slot_bootstrap=dict(seed=BOOTSTRAP_SEED, draws=REPS,
                      resampling_unit='pair_ordinal shared over all training schedules and scenarios',
                      percentile95=conditional_intervals(primary_slots, PRIMARY)),
                  held_out_failure_partition=fail, held_out_paired_success=discordance,
                  other_s01_ablations=single_seed_ablations(summary))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / 'paired_statistics.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
