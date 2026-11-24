"""Resume the original generalization suite with strict online-shaped replay."""
from recovery_common import (
    CONTROL_DIR, DESCRIPTOR, LAUNCH_DIR, OUTPUT, REMAINING, preflight, read, reference, require,
)
from business_graph_summary_campaign_v1.plan_io import encoded
from exact_admission import exact_online_admission
from grade_only_ppo_v2.runtime import _controller_lock
from online_v1_1.operations import _publish
from run_generalization import run, status


def main():
    data, suite, _, ledger = preflight()
    intent = read(LAUNCH_DIR / 'launch_intent.json')
    require(intent['recovery'] == reference(DESCRIPTOR) and intent['remaining_shifts'] == REMAINING
            and intent['additional_shifts'] == 0 and intent['original_suite'] == data['original_suite'],
            'recovery launch identity differs')
    recovered_suite = dict(suite, control_directory=str(CONTROL_DIR))
    with _controller_lock(OUTPUT), exact_online_admission():
        stop = CONTROL_DIR / 'STOP.properties'
        require(not stop.exists(), 'previous recovery STOP must be reviewed')
        try:
            run(recovered_suite, ledger)
        except BaseException as error:
            stop.parent.mkdir(parents=True, exist_ok=True)
            if not stop.exists():
                _publish(stop, ('reason=' + repr(error).replace('\n', ' ') + '\n').encode())
            status('STOP', reason=repr(error), recovery=reference(DESCRIPTOR))
            raise


if __name__ == '__main__':
    main()
