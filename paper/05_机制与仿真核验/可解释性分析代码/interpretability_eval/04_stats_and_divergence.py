"""Calculate M1 relation attention, M4 fleet allocation, and M5 policy divergence."""

import argparse
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import kruskal, t as student_t

from _shared import GROUPS, edge_attention_heads, iter_rows, total_shovel_queue, write_json


ATTENTION_LAYERS = {'Transport': 2, 'Commitment': 2, 'Service': 1, 'Material': 1}
TRUCK_TYPES = ('Cat777', 'Cat785', 'Cat789')
TRAINING_SEEDS = ('s01', 's02', 's03')
BASELINES = ('Greedy', 'RHO')
REGIMES = ('Normal', 'Grade-Critical', 'Congestion')
GRADE_CHANGE_HORIZON_MIN = 20
TRAJECTORY_METRICS = ('grade_margin_pp', 'crusher_queue')
STATE_METRICS = ('grade_margin_pp', 'grade_expected_pp', 'crusher_queue', 'remaining_min')
RETAINED_FIELDS = ('seed', 'scenario', 'sim_time_min', 'decision_index', 'pair_id', 'dispatch_kind',
                   'swap_time_min', 'action', 'baseline_actions', 'truck_type', 'truck_payload_t', *STATE_METRICS)


def finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and np.isfinite(value)


def shift_key(row):
    return row['policy'], row.get('training_seed'), row['shift_id']


def group_shifts(rows):
    shifts = defaultdict(list)
    for row in rows:
        if not finite(row.get('sim_time_min')):
            raise ValueError(f"Missing or invalid sim_time_min for shift {row.get('shift_id')!r}")
        key = shift_key(row)
        compact = {field: row.get(field) for field in RETAINED_FIELDS}
        if 'grade_margin_plus20_pp' in row:
            compact['grade_margin_plus20_pp'] = row['grade_margin_plus20_pp']
            compact['grade_margin_plus20_method'] = row.get('grade_margin_plus20_method')
        if key[0] in ('BG128', 'Flat-PPO'):
            compact['chosen_shovel'] = chosen_shovel(row)
        if key[0] == 'BG128':
            conditions = regimes(row)
            compact['regimes'] = conditions
            if conditions:
                compact['factual_attention'] = factual_attention_groups(row)
        shifts[key].append(compact)
    for decisions in shifts.values():
        decisions.sort(key=lambda row: (row['sim_time_min'], row.get('decision_index') or 0))
    return shifts


def summarize(samples, keys):
    """Average within a shift, then across shifts, then across training seeds."""
    by_seed = defaultdict(list)
    for (condition, seed, _shift), (total, count) in samples.items():
        by_seed[condition, seed].append(total / count)
    by_condition = defaultdict(dict)
    for (condition, seed), means in by_seed.items():
        by_condition[condition][seed] = float(np.mean(means))

    results = []
    for condition, by_seed_mean in sorted(by_condition.items()):
        values = list(by_seed_mean.values())
        n = len(values)
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if n > 1 else None
        half_width = float(student_t.ppf(.975, n - 1) * std / np.sqrt(n)) if n > 1 else None
        results.append(dict(zip(keys, condition)) | {
            'mean': mean,
            'std': std,
            'ci95_low': mean - half_width if half_width is not None else None,
            'ci95_high': mean + half_width if half_width is not None else None,
            'n_seeds': sum(seed in TRAINING_SEEDS for seed in by_seed_mean),
            'n_groups': n,
            'n_shifts': sum(len(by_seed[condition, seed]) for seed in by_seed_mean),
            'seed_means': {(seed or 'unseeded'): value for seed, value in by_seed_mean.items()},
        })
    return results


def track(samples, condition, key, value):
    bucket = condition, key[1], key
    total, count = samples.get(bucket, (0.0, 0))
    samples[bucket] = total + value, count + 1


def chosen_shovel(row):
    if str(row.get('dispatch_kind')).upper() != 'LOAD':
        return None
    action = row.get('action')
    if not isinstance(action, int) or isinstance(action, bool):
        return None
    return next((site for site in row.get('shovels') or [] if site.get('action') == action), None)


def regimes(row):
    margin = row.get('grade_margin_pp')
    crusher = row.get('crusher_queue')
    flags = []
    if finite(margin) and margin > .15 and finite(crusher) and crusher <= 2:
        flags.append('Normal')
    if finite(margin) and margin <= .05:
        flags.append('Grade-Critical')
    total_queue = total_shovel_queue(row)
    if finite(total_queue) and total_queue >= 2:
        flags.append('Congestion')
    return flags


def factual_attention_groups(row):
    """Summarize factual directed edges, with a single head mean per edge."""
    heads = {layer: edge_attention_heads(row, layer) for layer in (1, 2)}
    means = {layer: weights.mean(axis=1) for layer, weights in heads.items() if weights is not None}
    measured = defaultdict(list)
    missing_layers = set()
    for index, edge in enumerate(row.get('edges') or []):
        group = str(edge.get('group')).title()
        if group not in GROUPS:
            raise ValueError(f'Unknown graph relation group {group!r}')
        layer = ATTENTION_LAYERS[group]
        if layer in means:
            measured[group].append(float(means[layer][index]))
        else:
            missing_layers.add(layer)
    return {group: (sum(values), len(values)) for group, values in measured.items()}, (
        bool(missing_layers) or not means
    )


def attention_analysis(shifts):
    attention, swapped = {}, {}
    regime_decisions = dict.fromkeys(REGIMES, 0)
    coverage = {(state, group): dict(state=state, group=group, layer=ATTENTION_LAYERS[group],
                                     n_edges=0, n_decisions=0, edge_sum=0.)
                for state in REGIMES for group in GROUPS}
    missing_attention = no_measured_edges = unrecorded_swap = 0
    for key, decisions in shifts.items():
        policy = key[0]
        if policy == 'BG128':
            for row in decisions:
                conditions = row['regimes']
                if not conditions:
                    continue
                measured, missing = row['factual_attention']
                missing_attention += missing
                no_measured_edges += not measured
                for state in conditions:
                    regime_decisions[state] += 1
                    for group, (edge_sum, count) in measured.items():
                        cell = coverage[state, group]
                        cell['n_edges'] += count
                        cell['edge_sum'] += edge_sum
                        cell['n_decisions'] += 1
                        bucket = (state, group), key[1], key
                        total, samples = attention.get(bucket, (0., 0))
                        attention[bucket] = total + edge_sum, samples + count

        if policy not in ('BG128', 'Flat-PPO') or str(decisions[0].get('scenario')).casefold() != 'swapped':
            continue
        times = {row['swap_time_min'] for row in decisions if finite(row.get('swap_time_min'))}
        if len(times) != 1:
            unrecorded_swap += 1
            continue
        swap_time = times.pop()
        denominator = 0
        high = 0
        post_swap = (row for row in decisions if row['sim_time_min'] >= swap_time)
        for step, row in enumerate(post_swap, start=1):
            if step > 30:
                break
            shovel = row['chosen_shovel']
            if shovel is not None and isinstance(shovel.get('is_high_grade'), bool):
                denominator += 1
                high += shovel['is_high_grade']
            if denominator:
                track(swapped, (policy, step), key, high / denominator)

    for cell in coverage.values():
        edge_sum = cell.pop('edge_sum')
        cell['pooled_edge_mean'] = edge_sum / cell['n_edges'] if cell['n_edges'] else None
    return {
        'attention': [dict(row, **{
            field: coverage[row['state'], row['group']][field]
            for field in ('layer', 'n_edges', 'n_decisions', 'pooled_edge_mean')
        }) for row in summarize(attention, ('state', 'group'))],
        'attention_layers': ATTENTION_LAYERS,
        'attention_coverage': list(coverage.values()),
        'regime_decisions': regime_decisions,
        'swapped': summarize(swapped, ('policy', 'step')),
        'missing_attention_decisions': missing_attention,
        'decisions_without_measured_edges': no_measured_edges,
        'swapped_shifts_without_unique_swap_time': unrecorded_swap,
        'regime_definition': {
            'Normal': 'grade_margin_pp > 0.15 and crusher_queue <= 2',
            'Grade-Critical': 'grade_margin_pp <= 0.05',
            'Congestion': 'total shovel queue (unique queue/Seize-wait trucks at each shovel) >= 2',
        },
        'regime_membership': 'A decision contributes to each regime whose thresholds it satisfies.',
        'attention_scope': 'Observed directed edges in the frozen Actor graph. Use layer 1 for Service and '
                           'Material message collection and layer 2 for Transport and Commitment. '
                           'Average heads per edge, then edges within each shift and group. The primary '
                           'mean weights shifts equally within each training seed and training seeds '
                           'equally; pooled_edge_mean gives the unweighted average over factual edges. '
                           'Attention is normalized within each destination and relation group, so '
                           'the edge means do not measure redistribution of a shared group budget.',
        'swapped_denominator': 'Known LOAD destinations cumulatively within the first 30 decisions at or after '
                               'the recorded swap time. Shifts without a recorded swap time are excluded.',
    }


def kruskal_wallis_tests(metrics):
    """Use one within-shift mean per vehicle-type group, with one test per outcome."""
    tests, shift_level = [], []
    for metric, samples in metrics.items():
        by_state = defaultdict(lambda: defaultdict(list))
        for ((state, truck_type), seed, key), (total, count) in samples.items():
            row = dict(metric=metric, state=state, truck_type=truck_type, value=total / count,
                       training_seed=seed, shift_id=key[2])
            by_state[state][truck_type].append(row)
            shift_level.append(row)
        for state, groups in sorted(by_state.items()):
            group_values = [[row['value'] for row in groups[t]] for t in TRUCK_TYPES]
            pooled = [value for values in group_values for value in values]
            statistic = p_value = reason = None
            if any(not values for values in group_values):
                reason = 'A vehicle type has no shift-level observations.'
            elif min(pooled) == max(pooled):
                reason = 'All shift-level values are identical.'
            else:
                result = kruskal(*group_values)
                statistic, p_value = float(result.statistic), float(result.pvalue)
            tests.append(dict(
                metric=metric, state=state, statistic=statistic, p_value=p_value, p_holm=None,
                n_shifts=len({(row['training_seed'], row['shift_id'])
                              for group in groups.values() for row in group}),
                n_by_type=dict(zip(TRUCK_TYPES, map(len, group_values))),
                method='Kruskal–Wallis', unavailable_reason=reason,
            ))
    holm_adjust(tests)
    return tests, shift_level


def holm_adjust(tests):
    eligible = sorted((row for row in tests if row['p_value'] is not None), key=lambda row: row['p_value'])
    previous = 0.0
    for index, row in enumerate(eligible):
        previous = min(1.0, max(previous, (len(eligible) - index) * row['p_value']))
        row['p_holm'] = previous


def fleet_analysis(shifts):
    assignments, distances, contributions = {}, {}, {}
    dispatch, grade_variance, missing_destinations = [], [], 0
    for key, decisions in shifts.items():
        if key[0] != 'BG128':
            continue
        grades = defaultdict(list)
        for row in decisions:
            truck_type = row.get('truck_type')
            if truck_type not in TRUCK_TYPES:
                continue
            shovel = row['chosen_shovel']
            if shovel is None:
                missing_destinations += str(row.get('dispatch_kind')).upper() == 'LOAD'
                continue
            high, margin = shovel.get('is_high_grade'), row.get('grade_margin_pp')
            if isinstance(high, bool) and finite(margin):
                if margin < .05:
                    track(assignments, ('Grade-Critical', truck_type), key, int(high))
                elif margin > .15:
                    track(assignments, ('Grade-Ample', truck_type), key, int(high))

            distance = shovel.get('haul_distance_km')
            distance = float(distance) if finite(distance) else None
            if distance is not None:
                track(distances, ('All', truck_type), key, distance)
            grade = shovel.get('grade_pp')
            grade = float(grade) if finite(grade) and shovel.get('is_waste') is False else None
            payload = row.get('truck_payload_t')
            contribution = grade * payload / 100 if grade is not None and finite(payload) else None
            if grade is not None:
                grades[truck_type].append(grade)
            if contribution is not None:
                track(contributions, ('All', truck_type), key, contribution)
            if distance is not None or grade is not None:
                dispatch.append(dict(truck_type=truck_type, distance_km=distance, grade_pp=grade,
                                     grade_contribution_t=contribution, training_seed=key[1], shift_id=key[2]))

        for truck_type, values in grades.items():
            if len(values) > 1:
                grade_variance.append(dict(training_seed=key[1], shift_id=key[2], truck_type=truck_type,
                                           grade_variance_pp2=float(np.var(values, ddof=1)), n_loads=len(values)))

    tests, shift_level = kruskal_wallis_tests(dict(
        high_grade_share=assignments, distance_km=distances, grade_contribution_t=contributions,
    ))
    return {
        'assignment': summarize(assignments, ('state', 'truck_type')),
        'dispatch': dispatch,
        'grade_variance': grade_variance,
        'tests': tests,
        'shift_level': shift_level,
        'missing_load_destinations': missing_destinations,
        'test_definition': 'Kruskal–Wallis across the three vehicle types. Each group contains one '
                           'decision mean per physical shift. Holm correction spans the calculable '
                           'metric-by-regime omnibus tests. Shift observations can recur across vehicle '
                           'types, and this rank test does not adjust for that dependence.',
        'distance_definition': 'Recorded shovel-to-unload haul route length in kilometres.',
        'grade_contribution_definition': 'Truck payload in tonnes times selected ore shovel grade as a mass '
                                         'fraction at dispatch. This is the implied metal mass at dispatch.',
    }


def matching_baseline(key, row, shifts_by_slot, baseline):
    scenario = str(row.get('scenario')).casefold()
    if row.get('seed') is None or row.get('scenario') is None:
        return None
    matches = shifts_by_slot.get((baseline, scenario, row['seed'], row.get('pair_id')), [])
    same_training_seed = [entry for entry in matches if entry[0][1] == key[1]]
    candidates = same_training_seed or matches
    return candidates[0][1] if len(candidates) == 1 else None


def previous_values(rows, times, target):
    if target < times[0] or target > times[-1]:
        return {}
    return rows[bisect_right(times, target) - 1]


def grade_after_20min(row, decisions, times):
    if 'grade_margin_plus20_pp' in row:
        value = row['grade_margin_plus20_pp']
        method = row.get('grade_margin_plus20_method') or 'recorded_20min_grade'
        return value, method

    target = row['sim_time_min'] + GRADE_CHANGE_HORIZON_MIN
    after = bisect_right(times, target)
    if after == 0 or target > times[-1]:
        return None, None
    left_grade = decisions[after - 1].get('grade_margin_pp')
    lower = times[after - 1]
    if not finite(left_grade):
        return None, None
    if target == lower:
        return left_grade, 'decision_at_target'
    right_grade = decisions[after].get('grade_margin_pp')
    if not finite(right_grade):
        return None, None
    fraction = (target - lower) / (times[after] - lower)
    return left_grade + fraction * (right_grade - left_grade), 'decision_grade_interpolation'


def divergence_analysis(shifts, step_min):
    rates, state, trajectory, change = {}, {}, {}, {}
    shifts_by_slot = defaultdict(list)
    counts = {baseline: dict(comparable=0, divergent=0, missing_baseline_action=0,
                             unpaired_divergent=0, paired_events=0) for baseline in BASELINES}
    coverage = {
        (baseline, agreement): dict(baseline=baseline, agreement=agreement, comparable=0,
                                    n_events=0, missing_start_grade=0, missing_end_grade=0,
                                    negative_events=0, target_methods={})
        for baseline in BASELINES for agreement in ('agree', 'divergent')
    }
    offsets = list(range(-15, 46, step_min))
    for key, decisions in shifts.items():
        if key[0] in BASELINES:
            row = decisions[0]
            shifts_by_slot[key[0], str(row.get('scenario')).casefold(), row.get('seed'), row.get('pair_id')].append((key, decisions))

    for key, decisions in shifts.items():
        if key[0] != 'BG128':
            continue
        times = [row['sim_time_min'] for row in decisions]
        for baseline in BASELINES:
            paired = matching_baseline(key, decisions[0], shifts_by_slot, baseline)
            paired_times = [row['sim_time_min'] for row in paired] if paired else []
            for row in decisions:
                action = row.get('action')
                alternative = (row.get('baseline_actions') or {}).get(baseline)
                if action is None or alternative is None:
                    counts[baseline]['missing_baseline_action'] += 1
                    continue
                different = action != alternative
                agreement = 'divergent' if different else 'agree'
                counts[baseline]['comparable'] += 1
                counts[baseline]['divergent'] += int(different)
                track(rates, (baseline,), key, int(different))
                for metric in STATE_METRICS:
                    value = row.get(metric)
                    if finite(value):
                        track(state, (baseline, agreement, metric), key, value)

                current_grade = row.get('grade_margin_pp')
                event_coverage = coverage[baseline, agreement]
                event_coverage['comparable'] += 1
                if not finite(current_grade):
                    event_coverage['missing_start_grade'] += 1
                else:
                    future_grade, method = grade_after_20min(row, decisions, times)
                    if finite(future_grade):
                        difference = future_grade - current_grade
                        track(change, (baseline, agreement), key, difference)
                        event_coverage['n_events'] += 1
                        event_coverage['negative_events'] += int(difference < 0)
                        methods = event_coverage['target_methods']
                        methods[method] = methods.get(method, 0) + 1
                    else:
                        event_coverage['missing_end_grade'] += 1
                if not different:
                    continue
                if paired is None:
                    counts[baseline]['unpaired_divergent'] += 1
                    continue
                counts[baseline]['paired_events'] += 1
                for offset in offsets:
                    target = row['sim_time_min'] + offset
                    bg = row if offset == 0 else previous_values(decisions, times, target)
                    control = previous_values(paired, paired_times, target)
                    for metric in TRAJECTORY_METRICS:
                        if finite(bg.get(metric)) and finite(control.get(metric)):
                            track(trajectory, (baseline, 'BG128', metric, offset), key, bg[metric])
                            track(trajectory, (baseline, baseline, metric, offset), key, control[metric])

    return {
        'rates': summarize(rates, ('baseline',)),
        'state': summarize(state, ('baseline', 'agreement', 'metric')),
        'trajectory': summarize(trajectory, ('baseline', 'policy', 'metric', 'offset_min')),
        'grade_change_20min': [
            dict(row, metric='grade_change_20min_pp',
                 n_events=coverage[row['baseline'], row['agreement']]['n_events'])
            for row in summarize(change, ('baseline', 'agreement'))
        ],
        'grade_change_20min_coverage': list(coverage.values()),
        'grade_change_20min_definition': 'Observed BG128 rolling grade margin at t+20 min minus the '
                                         'same shift’s measured rolling grade margin at t, in '
                                         'percentage points. The group is determined by the rule’s '
                                         'same-state action at t. Use the exact-horizon field when '
                                         'provided. Older inputs without that field use bracketing '
                                         'decision grades within the shift. This comparison is observational '
                                         'and uses the BG128 trajectory for both groups.',
        'counts': counts,
        'alignment': 'At each BG128 divergence time, compare the last logged state at or before the same '
                     'simulation time in both paired trajectories. Exclude time outside either trajectory '
                     'and keep only paired values for each metric. Event windows can overlap within a shift. '
                     'These are comparisons of complete policy trajectories.',
    }


def analyze(rows, step_min):
    shifts = group_shifts(rows)
    seen_seeds = {key[1] for key in shifts if key[0] == 'BG128'}
    return {
        'schema': 'bg128-interpretability-stats-v3',
        'input_rows': sum(len(decisions) for decisions in shifts.values()),
        'missing_bg128_training_seeds': [seed for seed in TRAINING_SEEDS if seed not in seen_seeds],
        'aggregation': 'Decision and event means within a shift, equal shift means within training seed, '
                       'and equal training seed means. Error uses sample standard deviation across observed '
                       'seed means. Unseeded baselines contribute one descriptive group.',
        'm1': attention_analysis(shifts),
        'm4': fleet_analysis(shifts),
        'm5': divergence_analysis(shifts, step_min),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='Decision rows exported by 01_collect_rollouts.py')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--event-step-min', type=int, default=1, choices=(1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60))
    args = parser.parse_args()
    write_json(args.output, analyze(iter_rows(args.input), args.event_step_min))


if __name__ == '__main__':
    main()
