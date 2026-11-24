"""One serial AnyLogic invocation and six read-only evaluations of BG128 u32."""
from collections import Counter
from datetime import datetime, timezone

from gen_common import (
    APPROVAL, CAMPAIGN, CONTROL, CORE, COUNT, LAUNCH, OUTPUT, REPORT, SUITE, TOTAL,
    authorization_for, check_sources, checked, load_package, read, reference, require,
)
from business_graph_summary_campaign_v1.plan_io import encoded
from grade_only_ppo_v2.runtime import _controller_lock
from online_v1_1.operations import _publish
from gen_budget import GenLedger
from gen_evaluation import evaluate_phase, write_comparison


def status(value, **fields):
    info = dict(schema='bg128-u32-generalization-progress-v1', campaign_id=CAMPAIGN, status=value,
                updated_at_utc=datetime.now(timezone.utc).isoformat(), total_shifts=TOTAL, **fields)
    _publish(OUTPUT / 'run_status.json', encoded(info), replace=True)


def run(suite, ledger):
    import torch

    torch.set_num_threads(suite['cpu_threads'])
    ledger.register()
    results = []
    for phase in suite['phases']:
        check_sources(suite)
        scenario = phase['scenario_id']
        status('EVALUATING_SCENARIO', scenario_id=scenario, completed_preceding_shifts=COUNT * len(results))
        results.append(evaluate_phase(suite, phase, ledger))
        status('SCENARIO_COMPLETE', scenario_id=scenario, completed_shifts=COUNT * len(results))
    state = ledger.snapshot()
    counts = Counter(row['status'] for row in state['jobs'].values())
    require(counts == {'COMPLETE': TOTAL} and state['campaigns'][CAMPAIGN]['commits'] == {}
            and all(row['physical_shifts'] == COUNT and
                    row['training_updates'] == row['optimizer_updates'] == row['gradient_updates'] == 0
                    for row in results), 'generalization completion counts or fixed-policy scope differs')
    check_sources(suite)
    for ref in suite['parent_artifacts'].values():
        checked(ref)
    comparison = write_comparison(suite, results, REPORT)
    _publish(OUTPUT / 'suite_complete.json', encoded(dict(
        status='COMPLETE', campaign=reference(SUITE), checkpoint=suite['parent_artifacts']['checkpoint'],
        physical_shifts=TOTAL, training_updates=0, optimizer_updates=0, gradient_updates=0,
        results=results, comparisons=comparison, ledger_status_counts=dict(counts), protected_core=reference(CORE))))
    status('COMPLETE', completed_shifts=TOTAL)


def main():
    suite, _ = load_package()
    intent = read(LAUNCH / 'launch_intent.json')
    require(read(APPROVAL) == authorization_for(suite)
            and intent['suite'] == reference(SUITE) and intent['authorization'] == reference(APPROVAL)
            and intent['shifts'] == TOTAL, 'AnyLogic launch authority differs')
    ledger = GenLedger(suite, suite_path=SUITE, approval_path=APPROVAL)
    with _controller_lock(OUTPUT):
        stop = CONTROL / 'STOP.properties'
        require(not stop.exists() and not ledger.path.exists(), 'previous execution requires review')
        try:
            run(suite, ledger)
        except BaseException as error:
            CONTROL.mkdir(parents=True, exist_ok=True)
            if not stop.exists():
                _publish(stop, ('reason=' + repr(error).replace('\n', ' ') + '\n').encode())
            status('STOP', reason=repr(error))
            raise


if __name__ == '__main__':
    main()
