"""Compare GAT1 and its matched Full policy across the three training seeds."""

from pathlib import Path
from statistics import fmean, stdev

from replication_plan import GAT1_S01_COMPLETE, REPORT, UNITS, historical_sources, locations, parent
from ablation_report import GROUPS, METRICS, SCENARIOS, paired_report
from business_graph_summary_campaign_v1.plan_io import load_reference, reference


def _panel(phases):
    return [(phase['scenario_id'], job['pair_ordinal'], job['experiment_seed'], job['actor_sampling_seed'])
            for phase in phases for job in phase['jobs']]


def _summary(reports, sections, names):
    result = {}
    for name in names:
        metrics = {}
        for metric in METRICS:
            values = {seed: report[sections][name][metric] for seed, report in reports.items()}
            valid = [item for item in values.values() if item['paired_n'] > 0]
            deltas = [item['delta'] for item in valid]
            metrics[metric] = dict(
                by_training_seed=values, training_seed_count=len(valid),
                full_mean=fmean(item['full'] for item in valid) if valid else None,
                gat1_mean=fmean(item['ablation'] for item in valid) if valid else None,
                delta_mean=fmean(deltas) if deltas else None,
                delta_sd=stdev(deltas) if len(deltas) >= 2 else None,
            )
        result[name] = metrics
    return result


def complete_report(save):
    """Write the equal-training-seed-weight summary once the replicas finish."""
    sources = historical_sources()
    require = parent.require
    panel = _panel(sources['s01']['evaluation'])
    require(len(panel) == len(SCENARIOS) * parent.EVAL_COUNT and len(panel) == len(set(panel)),
            'archived evaluation panel is incomplete')
    completions = {'s01': GAT1_S01_COMPLETE}
    for unit in UNITS:
        seed = unit.rsplit('-', 1)[-1]
        layout = locations(unit)
        ref = reference(Path(layout['python_output']) / 'suite_complete.json')
        require(load_reference(ref)['suite'] == reference(layout['suite_path']),
                'replica completion belongs to a different suite')
        completions[seed] = ref
    require(set(completions) == set(sources), 'replication does not cover the matched Full seeds')

    reports, references = {}, {}
    for seed, completion_ref in completions.items():
        completion = load_reference(completion_ref)
        checkpoint = completion['new_checkpoint']
        require(completion['status'] == 'COMPLETE' and completion['training_updates'] == parent.UPDATES
                and completion['training_shifts'] == parent.TRAINING_SHIFTS
                and completion['evaluation_shifts'] == len(panel)
                and completion['evaluation_optimizer_updates'] == 0
                and reference(checkpoint['path']) == checkpoint, 'invalid fixed-u32 completion')
        report = load_reference(completion['comparison'])
        source = sources[seed]
        rows = report['rows']
        observed = [(row['scenario'], row['pair_ordinal'], row['environment_seed'], row['actor_sampling_seed'])
                    for row in rows]
        full_runs = [job['run_index'] for phase in source['evaluation'] for job in phase['jobs']]
        require(report['unit'] == f'gat1-{seed}' and report['baseline_seed'] == seed
                and report['baseline_checkpoint'] == source['checkpoint'] and report['checkpoint'] == checkpoint
                and observed == _panel(source['evaluation']) == panel
                and [row['full_run'] for row in rows] == full_runs,
                'paired report differs from the matching Full policy or fixed evaluation panel')
        reports[seed] = report
        references[seed] = dict(completion=completion_ref, paired_evaluation=completion['comparison'],
                                gat1_checkpoint=checkpoint, full_checkpoint=source['checkpoint'],
                                reused_archive=(seed == 's01'))

    require(all(report['ablation'] == reports['s01']['ablation'] for report in reports.values()),
            'GAT1 architecture differs among training seeds')
    output = REPORT / 'paired_evaluation_summary.json'
    save(output, dict(
        schema='bg128-gat1-three-training-seeds-v1', architecture=reports['s01']['ablation'],
        training_seed_schedules=list(reports), evaluation_slots_per_checkpoint=len(panel),
        scopes=GROUPS, summaries=_summary(reports, 'scope', GROUPS),
        scenarios=_summary(reports, 'scenarios', SCENARIOS), sources=references,
        direction='gat1_minus_full',
        weighting='equal training-seed weights; evaluation slots are reused across training seeds',
        empty_window_rule='grade error uses mutually nonempty pairs within each training seed',
    ))
    return reference(output)
