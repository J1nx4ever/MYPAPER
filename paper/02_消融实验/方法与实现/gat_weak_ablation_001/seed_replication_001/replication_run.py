"""Prepare and execute the two GAT1 mixed-seed replications through AnyLogic."""

from replication_plan import (ENTRY, GAT1_S01_COMPLETE, GOLDEN_EVAL_SHA, PREPARED, UNITS, bind,
                              load_package, locations, parent, prepare)
import ablation_run as runner
from business_graph_summary_campaign_v1.plan_io import load_reference, reference
from replication_budget import (LOCKS, ReplicationLedger, assert_available, authorization, authority_path,
                                budget_snapshot)
from replication_report import complete_report, paired_report
from weak_runtime import runtime_scope

_lane = bind(runner._lane, LOCKS=LOCKS)
_intent = bind(runner._intent, ENTRY=ENTRY, authority_path=authority_path)
_start = bind(runner._start, UNITS=UNITS, locations=locations, _lane=_lane, _intent=_intent,
              assert_available=assert_available, budget_snapshot=budget_snapshot, authority_path=authority_path,
              authorization=authorization)
_controller = bind(runner.controller, UNITS=UNITS, _lane=_lane, _intent=_intent, AblationLedger=ReplicationLedger,
                   runtime_scope=runtime_scope, complete_report=complete_report, paired_report=paired_report)
_check_package = bind(runner._check, PREPARED=PREPARED, UNITS=UNITS, load_package=load_package,
                      runtime_scope=runtime_scope, assert_available=assert_available, budget_snapshot=budget_snapshot)


def _check():
    launch = load_reference(reference(PREPARED / 'launch_identity.json'))
    anchor = load_reference(GAT1_S01_COMPLETE)
    parent.require(launch['fixed_evaluation_slot_sha256'] == GOLDEN_EVAL_SHA
                   and anchor['status'] == 'COMPLETE' and anchor['evaluation_optimizer_updates'] == 0,
                   'fixed evaluation scope differs')
    result = _check_package()
    result['fixed_evaluation_slot_sha256'] = GOLDEN_EVAL_SHA
    return result


def main():
    run = bind(runner.main, UNITS=UNITS, locations=locations, load_package=load_package, prepare=prepare,
               _check=_check, _start=_start, controller=_controller, AblationLedger=ReplicationLedger)
    run()


if __name__ == '__main__':
    main()
