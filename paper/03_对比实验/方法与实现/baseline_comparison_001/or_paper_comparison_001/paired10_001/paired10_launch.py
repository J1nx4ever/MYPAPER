"""Run the twenty-shift MILP evaluation from its matching AnyLogic roster."""

from paired10_campaign import (
    CAMPAIGN, CORE, COUNT, EXECUTION, LAUNCHER, LEDGER, LOCKS, METHOD, OUTPUT,
    POLICIES, PYTHON, REPORT, ROOT, SCHEMA, SUITE, CONTROL, Ledger, check_fresh, load_suite,
    preceding_snapshot,
)
import grade_launch as serial
from paired10_runtime import make_policy, run_shift
from paired10_statistics import aggregate
from weak_plan import bind


LANE = LOCKS / f"port-{EXECUTION['port']}.lock"
AUTHORITY = OUTPUT / 'launch' / 'run_authority.json'
INTENT = OUTPUT / 'launch' / 'intent.json'
STATUS = OUTPUT / 'run_status.json'
STOP = CONTROL / 'STOP.properties'


status = bind(serial.status, STATUS=STATUS, CAMPAIGN=CAMPAIGN, COUNT=COUNT, SUITE=SUITE)
_set_status = bind(serial._set_status, STATUS=STATUS, CAMPAIGN=CAMPAIGN, COUNT=COUNT)
_controller_command = bind(serial._controller_command, PYTHON=PYTHON, LAUNCHER=LAUNCHER, SUITE=SUITE)
_check_launch_card = bind(serial._preflight, OUTPUT=OUTPUT, REPORT=REPORT, SUITE=SUITE,
                          check_fresh=check_fresh, preceding_snapshot=preceding_snapshot, CAMPAIGN=CAMPAIGN)


def _preflight(suite):
    result = _check_launch_card(suite)
    for policy in POLICIES:
        make_policy(policy, suite['method'])
    return dict(result, imported_policies=list(POLICIES))


_from_anylogic = bind(serial._from_anylogic, _preflight=_preflight, LANE=LANE, EXECUTION=EXECUTION,
                      AUTHORITY=AUTHORITY, OUTPUT=OUTPUT, SCHEMA=SCHEMA, CAMPAIGN=CAMPAIGN, COUNT=COUNT,
                      INTENT=INTENT, ROOT=ROOT, _controller_command=_controller_command)
_controller = bind(serial._controller, LANE=LANE, AUTHORITY=AUTHORITY, INTENT=INTENT, LEDGER=LEDGER,
                   _controller_command=_controller_command, Ledger=Ledger, _set_status=_set_status,
                   _run_shift=run_shift, COUNT=COUNT, REPORT=REPORT, METHOD=METHOD, CORE=CORE, SCHEMA=SCHEMA,
                   CAMPAIGN=CAMPAIGN, OUTPUT=OUTPUT, SUITE=SUITE, STOP=STOP, _aggregate=aggregate)

main = bind(serial.main, SUITE=SUITE, status=status, load_suite=load_suite, _preflight=_preflight,
            _from_anylogic=_from_anylogic, _controller=_controller)


if __name__ == '__main__':
    main()
