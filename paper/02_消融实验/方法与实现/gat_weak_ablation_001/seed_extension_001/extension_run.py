"""Prepare, verify or run the paired s04/s05 BG128 depth extension."""

from contextlib import contextmanager
from pathlib import Path

from extension_plan import (ALP_RUNS, CAP, ENTRY, EVAL, GOLDEN_EVAL_SHA, PREPARED, TRAIN,
                            UNITS, bind, load_package, locations, parent, prepare)
from extension_budget import ExtensionLedger, LOCKS, assert_available, authority_path, authorization, budget_snapshot
from extension_report import complete_report
import ablation_run as runner
from ablation_report import paired_report
from business_graph_summary_campaign_v1.plan_io import load_reference, reference
from weak_runtime import runtime_scope as gat1_runtime


@contextmanager
def runtime_scope(suite):
    if suite['unit'].startswith('gat1-'):
        with gat1_runtime(suite) as runtime:
            yield runtime
    else:
        import mixed_runtime

        yield mixed_runtime


def _evaluation_reports(phases, results, checkpoint):
    reports = {}
    for cfg, item in zip(phases, results, strict=True):
        report_ref = item['report']
        report = load_reference(report_ref)
        scenario = cfg['scenario_id']
        parent.require(scenario == report['scenario_id'] and scenario not in reports
                       and report['checkpoint_provenance'] == item['checkpoint'] == checkpoint
                       and report['n_shifts'] == parent.EVAL_COUNT
                       and report['evaluation_optimizer_updates'] == 0
                       and report['training_state_unchanged'], 'fixed Full evaluation report differs')
        for job, row in zip(cfg['jobs'], report['per_episode'], strict=True):
            identity = row['identity']
            parent.require((identity['run_index'], identity['environment_seed'], identity['actor_sampling_seed'])
                           == (job['run_index'], job['experiment_seed'], job['actor_sampling_seed']),
                           'fixed Full evaluation slot differs')
        reports[scenario] = report_ref
    parent.require(tuple(reports) == tuple(parent.EVAL_CASES), 'fixed Full evaluation inventory differs')
    return reports


def _evaluation_report(suite, phases, results, checkpoint):
    seed = suite['baseline_seed']
    if suite['unit'] == f'full-{seed}':
        return dict(schema='bg128-depth-extension-full-evaluation-v1', unit=suite['unit'], checkpoint=checkpoint,
                    fixed_evaluation_slot_sha256=GOLDEN_EVAL_SHA,
                    reports=_evaluation_reports(phases, results, checkpoint))

    layout = locations(f'full-{seed}')
    full_suite, full_phases = load_package(f'full-{seed}')
    parent.require(suite['matched_full_suite_path'] == full_suite['suite_path'], 'matched Full path differs')
    completion = load_reference(reference(Path(layout['python_output']) / 'suite_complete.json'))
    parent.require(completion['status'] == 'COMPLETE' and completion['suite'] == reference(full_suite['suite_path'])
                   and completion['training_shifts'] == TRAIN and completion['training_updates'] == parent.UPDATES
                   and completion['evaluation_shifts'] == EVAL and completion['evaluation_optimizer_updates'] == 0,
                   'matched Full completion differs')
    full_checkpoint = completion['new_checkpoint']
    parent.require(reference(full_checkpoint['path']) == full_checkpoint, 'matched Full checkpoint bytes differ')
    reports = _evaluation_reports(full_phases[parent.UPDATES:], completion['phase_results'], full_checkpoint)
    full_index = load_reference(completion['comparison'])
    parent.require(full_index['unit'] == full_suite['unit'] and full_index['checkpoint'] == full_checkpoint
                   and full_index['reports'] == reports, 'matched Full evaluation index differs')
    matched = dict(suite, baseline_checkpoint=full_checkpoint, baseline_reports=reports)
    return paired_report(matched, phases, results, checkpoint)


_lane = bind(runner._lane, LOCKS=LOCKS)
_intent = bind(runner._intent, ENTRY=ENTRY, authority_path=authority_path)
_start = bind(runner._start, UNITS=UNITS, locations=locations, _lane=_lane, _intent=_intent,
              assert_available=assert_available, budget_snapshot=budget_snapshot,
              authority_path=authority_path, authorization=authorization)
_controller = bind(runner.controller, UNITS=UNITS, _lane=_lane, _intent=_intent, AblationLedger=ExtensionLedger,
                   runtime_scope=runtime_scope, complete_report=complete_report, paired_report=_evaluation_report)
_base_check = bind(runner._check, PREPARED=PREPARED, ALP_RUNS=ALP_RUNS, UNITS=UNITS, load_package=load_package,
                   runtime_scope=runtime_scope, assert_available=assert_available, budget_snapshot=budget_snapshot)


def _check():
    launch = load_reference(reference(PREPARED / 'launch_identity.json'))
    parent.require(launch['fixed_evaluation_slot_sha256'] == GOLDEN_EVAL_SHA
                   and launch['training_shifts'] == TRAIN * len(UNITS)
                   and launch['evaluation_shifts'] == EVAL * len(UNITS), 'combined phase totals or panel differs')
    return dict(_base_check(), fixed_evaluation_slot_sha256=GOLDEN_EVAL_SHA)


if __name__ == '__main__':
    run = bind(runner.main, UNITS=UNITS, locations=locations, load_package=load_package, prepare=prepare,
               _check=_check, _start=_start, controller=_controller, AblationLedger=ExtensionLedger)
    run()
