"""Prepare and run the sealed two-arm weak GAT ablation from AnyLogic."""

import sys

from weak_plan import (ENTRY, GRADE_ARCHIVE, GOLDEN_EVAL_SHA, PREPARED, UNITS, bind, load_package,
                       locations, parent, prepare)
import ablation_run as runner
from business_graph_summary_campaign_v1.plan_io import encoded, load_reference, reference
from weak_budget import LOCKS, WeakLedger, assert_available, authorization, authority_path, budget_snapshot
from weak_report import complete_report, grade_replay, paired_report
from weak_runtime import runtime_scope

_lane = bind(runner._lane, LOCKS=LOCKS)
_intent = bind(runner._intent, ENTRY=ENTRY, authority_path=authority_path)
_start = bind(runner._start, UNITS=UNITS, locations=locations, _lane=_lane, _intent=_intent,
              assert_available=assert_available, budget_snapshot=budget_snapshot, authority_path=authority_path,
              authorization=authorization)
_controller = bind(runner.controller, UNITS=UNITS, _lane=_lane, _intent=_intent, AblationLedger=WeakLedger,
                   runtime_scope=runtime_scope, complete_report=complete_report, paired_report=paired_report)
_check_package = bind(runner._check, PREPARED=PREPARED, UNITS=UNITS, load_package=load_package,
                      runtime_scope=runtime_scope, assert_available=assert_available, budget_snapshot=budget_snapshot)


def _check():
    launch = load_reference(reference(PREPARED / 'launch_identity.json'))
    parent.require(launch['fixed_evaluation_slot_sha256'] == GOLDEN_EVAL_SHA
                   and load_reference(GRADE_ARCHIVE)['analyzed_decisions'] == 400, 'offline replay scope differs')
    result = _check_package()
    result['fixed_evaluation_slot_sha256'] = GOLDEN_EVAL_SHA
    return result


def _run_controller(suite, phases, lane_fd):
    _controller(suite, phases, lane_fd)
    if suite['unit'] == UNITS[-1]:
        sys.stdout.write(encoded(dict(grade_replay=grade_replay())).decode())


def main():
    if sys.argv[1:] == ['--grade-replay']:
        sys.stdout.write(encoded(grade_replay()).decode())
        return
    run = bind(runner.main, UNITS=UNITS, locations=locations, load_package=load_package, prepare=prepare,
               _check=_check, _start=_start, controller=_run_controller, AblationLedger=WeakLedger)
    run()


if __name__ == '__main__':
    main()
