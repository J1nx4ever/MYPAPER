"""Summarize five paired Full and GAT1 training schedules on the original evaluation panel."""

from pathlib import Path

from extension_plan import GOLDEN_EVAL_SHA, REPORT, historical_sources, locations, parent
from ablation_report import GROUPS, SCENARIOS, paired_report
from business_graph_summary_campaign_v1.plan_io import load_reference, load_suite, reference
from replication_plan import REPORT as PRIOR_REPORT
from replication_report import _panel, _summary
from weak_plan import eval_sha

PRIOR_SEEDS = ('s01', 's02', 's03')
NEW_SEEDS = ('s04', 's05')
PRIOR_SUMMARY = dict(path=str(PRIOR_REPORT / 'paired_evaluation_summary.json'),
                     sha256='57347fcba8e7522ea1d1cacfa7592867444bf4dcc82fc5b0853cc32f8fae43ee')


def _completion_ref(unit):
    return reference(Path(locations(unit)['python_output']) / 'suite_complete.json')


def _load_completed(completion_ref, unit, panel):
    completion = load_reference(completion_ref)
    checkpoint = completion['new_checkpoint']
    suite_ref = completion['suite']
    if unit.rsplit('-', 1)[1] in NEW_SEEDS:
        parent.require(suite_ref['path'] == locations(unit)['suite_path'],
                       f'{unit} completion belongs to a different suite')
    parent.require(completion['status'] == 'COMPLETE'
                   and completion['training_updates'] == parent.UPDATES
                   and completion['training_shifts'] == parent.TRAINING_SHIFTS
                   and completion['evaluation_shifts'] == len(panel)
                   and completion['evaluation_optimizer_updates'] == 0
                   and reference(checkpoint['path']) == checkpoint
                   and reference(suite_ref['path']) == suite_ref,
                   f'{unit} fixed-u32 completion differs')
    suite, phases = load_suite(suite_ref['path'])
    evaluation = phases[parent.UPDATES:]
    parent.require(suite['unit'] == unit and _panel(evaluation) == panel
                   and eval_sha(phases) == GOLDEN_EVAL_SHA,
                   f'{unit} fixed panel or suite identity differs')
    return completion, suite, evaluation


def _full_source(seed, panel):
    unit = f'full-{seed}'
    completion_ref = _completion_ref(unit)
    completion, suite, evaluation = _load_completed(completion_ref, unit, panel)
    checkpoint = completion['new_checkpoint']
    reports = {}
    for offset, (phase, result) in enumerate(zip(evaluation, completion['phase_results'], strict=True)):
        report_ref = result['report']
        report = load_reference(report_ref)
        scenario = phase['scenario_id']
        parent.require(report['scenario_id'] == scenario
                       and report['phase_config'] == suite['phase_configs'][parent.UPDATES + offset]
                       and report['checkpoint_provenance'] == result['checkpoint'] == checkpoint
                       and report['n_shifts'] == parent.EVAL_COUNT
                       and report['training_state_unchanged'] and report['evaluation_optimizer_updates'] == 0,
                       f'{unit} evaluation provenance differs in {scenario}')
        reports[scenario] = report_ref
    index = load_reference(completion['comparison'])
    parent.require(index == dict(schema='bg128-depth-extension-full-evaluation-v1', unit=unit,
                                 checkpoint=checkpoint, fixed_evaluation_slot_sha256=GOLDEN_EVAL_SHA,
                                 reports=reports), f'{unit} fixed-u32 evaluation index differs')
    return dict(suite=completion['suite'], completion=completion_ref, checkpoint=checkpoint,
                evaluation=evaluation, reports=reports)


def _paired_seed(seed, completion_ref, source, panel, architecture):
    unit = f'gat1-{seed}'
    completion, suite, evaluation = _load_completed(completion_ref, unit, panel)
    binding = dict(baseline_checkpoint=source['checkpoint'], baseline_completion=source['completion'],
                   baseline_reports=source['reports'])
    parent.require(suite['baseline_seed'] == seed, f'{unit} matched Full seed differs')
    if seed in NEW_SEEDS:
        parent.require(suite['matched_full_suite_path'] == source['suite']['path']
                       and suite['baseline_checkpoint'] is None and suite['baseline_completion'] is None
                       and suite['baseline_reports'] == {}, f'{unit} prepared Full binding differs')
        suite.update(binding)
    else:
        parent.require(all(suite[key] == value for key, value in binding.items()),
                       f'{unit} archived Full binding differs')
    checkpoint = completion['new_checkpoint']
    paired_ref = completion['comparison']
    report = load_reference(paired_ref)
    recomputed = paired_report(suite, evaluation, completion['phase_results'], checkpoint)
    full_runs = [job['run_index'] for phase in source['evaluation'] for job in phase['jobs']]
    parent.require(report == recomputed and report['ablation'] == architecture
                   and [row['full_run'] for row in report['rows']] == full_runs,
                   f'{unit} paired outcomes, Full run IDs, or GAT1 architecture differ')
    pair = dict(completion=completion_ref, paired_evaluation=paired_ref, gat1_checkpoint=checkpoint,
                full_checkpoint=source['checkpoint'], full_completion=source['completion'])
    return report, pair


def complete_report(save):
    """Seal the equal-training-seed-weight evaluation once all five pairs are complete."""
    previous = load_reference(PRIOR_SUMMARY)
    parent.require(previous['schema'] == 'bg128-gat1-three-training-seeds-v1'
                   and tuple(previous['training_seed_schedules']) == PRIOR_SEEDS
                   and previous['evaluation_slots_per_checkpoint'] == len(SCENARIOS) * parent.EVAL_COUNT,
                   'frozen three-seed summary inventory differs')
    full = historical_sources()
    panel = _panel(full['s01']['evaluation'])
    parent.require(len(panel) == len(set(panel)) == len(SCENARIOS) * parent.EVAL_COUNT
                   and eval_sha(full['s01']['evaluation']) == GOLDEN_EVAL_SHA,
                   'original evaluation panel differs')
    reports, references, run_indexes = {}, {}, []
    for seed in (*PRIOR_SEEDS, *NEW_SEEDS):
        if seed in PRIOR_SEEDS:
            anchor = previous['sources'][seed]
            full_source = full[seed]
            completion_ref = anchor['completion']
        else:
            full_source = _full_source(seed, panel)
            completion_ref = _completion_ref(f'gat1-{seed}')
        report, pair = _paired_seed(seed, completion_ref, full_source, panel, previous['architecture'])
        if seed in PRIOR_SEEDS:
            parent.require(all(pair[key] == anchor[key] for key in
                               ('completion', 'paired_evaluation', 'gat1_checkpoint', 'full_checkpoint')),
                           f'{seed} differs from the frozen three-seed source')
        reports[seed] = report
        references[seed] = pair
        run_indexes.extend(row[arm] for row in report['rows'] for arm in ('full_run', 'ablation_run'))
    parent.require(len(run_indexes) == len(set(run_indexes)) == 2 * len(reports) * len(panel),
                   'physical evaluation identities overlap')
    output = REPORT / 'paired_evaluation_summary.json'
    save(output, dict(
        schema='bg128-gat1-five-training-seeds-v1', architecture=previous['architecture'],
        training_seed_schedules=list(reports), evaluation_slots_per_checkpoint=len(panel),
        fixed_evaluation_slot_sha256=GOLDEN_EVAL_SHA, prior_three_seed_summary=PRIOR_SUMMARY,
        scopes=GROUPS, summaries=_summary(reports, 'scope', GROUPS),
        scenarios=_summary(reports, 'scenarios', SCENARIOS), sources=references,
        direction='gat1_minus_full',
        weighting='equal training-seed weights; evaluation slots are reused across training seeds',
        empty_window_rule='grade error uses mutually nonempty pairs within each training seed',
    ))
    return reference(output)
