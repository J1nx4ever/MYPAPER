"""Pair each fixed ablation u32 with the corresponding completed Full u32."""

from pathlib import Path
from statistics import fmean

from ablation_plan import REPORT, SCENARIOS, UNITS, VARIANTS, locations, parent
from business_graph_summary_campaign_v1.plan_io import load_reference, reference

HELD_OUT = SCENARIOS[2:]
GROUPS = dict(held_out=HELD_OUT, nominal=('nominal',), grade_swap=('grade_swap',))
METRICS = ('complete_task', 'total_wait_seconds', 'all_four_windows_compliant',
           'ore_target_met', 'waste_target_met', 'ore_tons', 'waste_tons',
           *(f'window_{window}_absolute_grade_error' for window in range(1, 5)),
           *(f'window_{window}_empty' for window in range(1, 5)))


def _physical(row):
    raw = row['physical']
    windows = raw['windows']
    parent.require([item['window'] for item in windows] == list(range(4)), 'four-window physical order differs')
    result = {key: raw[key] for key in METRICS[:7]}
    for i, window in enumerate(windows, 1):
        result[f'window_{i}_empty'] = window['empty']
        result[f'window_{i}_absolute_grade_error'] = None if window['empty'] else abs(window['grade'] - 0.27)
    return result


def _aggregate(rows):
    metrics = {}
    for key in METRICS:
        pairs = [(row['full'][key], row['ablation'][key]) for row in rows
                 if row['full'][key] is not None and row['ablation'][key] is not None]
        if pairs:
            metrics[key] = dict(paired_n=len(pairs), full=fmean(full for full, _ in pairs),
                                ablation=fmean(test for _, test in pairs),
                                delta=fmean(float(test) - float(full) for full, test in pairs))
        else:
            metrics[key] = dict(paired_n=0, full=None, ablation=None, delta=None)
    return metrics


def paired_report(suite, phases, results, checkpoint):
    rows = []
    for cfg, result, scenario in zip(phases, results, SCENARIOS, strict=True):
        full_ref = suite['baseline_reports'][scenario]
        full = load_reference(full_ref)
        current = load_reference(result['report'])
        parent.require(cfg['scenario_id'] == scenario == full['scenario_id'] == current['scenario_id']
                       and current['checkpoint_provenance'] == checkpoint
                       and full['checkpoint_provenance'] == suite['baseline_checkpoint']
                       and current['n_shifts'] == full['n_shifts'] == parent.EVAL_COUNT
                       and current['evaluation_optimizer_updates'] == full['evaluation_optimizer_updates'] == 0
                       and current['training_state_unchanged'] and full['training_state_unchanged'],
                       'fixed policy evaluation provenance differs')
        for job, a, b in zip(cfg['jobs'], full['per_episode'], current['per_episode'], strict=True):
            expected = (job['experiment_seed'], job['actor_sampling_seed'])
            old, new = a['identity'], b['identity']
            parent.require(all((i['environment_seed'], i['actor_sampling_seed']) == expected for i in (old, new))
                           and new['run_index'] == job['run_index'] and old['run_index'] != new['run_index'],
                           'paired evaluation slot changed')
            rows.append(dict(scenario=scenario, pair_ordinal=job['pair_ordinal'], environment_seed=expected[0],
                             actor_sampling_seed=expected[1], full_run=old['run_index'], ablation_run=new['run_index'],
                             full=_physical(a), ablation=_physical(b)))
    return dict(unit=suite['unit'], ablation=suite['ablation'], baseline_seed=suite['baseline_seed'],
                baseline_checkpoint=suite['baseline_checkpoint'], checkpoint=checkpoint, rows=rows,
                scenarios={name: _aggregate([r for r in rows if r['scenario'] == name]) for name in SCENARIOS},
                scope={name: _aggregate([r for r in rows if r['scenario'] in scenarios])
                       for name, scenarios in GROUPS.items()})


def complete_report(save):
    comparisons = {}
    for unit in UNITS:
        location = locations(unit)
        complete = load_reference(reference(Path(location['python_output']) / 'suite_complete.json'))
        parent.require(complete['status'] == 'COMPLETE' and complete['training_updates'] == parent.UPDATES,
                       'all ablation u32 runs must be complete')
        comparisons[unit] = complete['comparison']
    groups = {VARIANTS[unit.split('-')[0]]: load_reference(ref)['scope'] for unit, ref in comparisons.items()}
    output = REPORT / 'paired_evaluation_summary.json'
    save(output, dict(training_seed_schedule='s01', scopes=GROUPS, summaries=groups, source_reports=comparisons,
                      direction='ablation_minus_full', empty_window_rule='grade error uses mutually nonempty pairs'))
    return reference(output)
