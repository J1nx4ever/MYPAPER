"""One explicitly authorized controller launch and read-only compact status.

There is no shell command argument, arbitrary module, network target, automatic
budget increase, automatic resume, or automatic STOP/UNKNOWN clearing.  Budget
registration remains in the fixed runtime after all frozen gate checks.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess

from .health import inspect, _process_identity


ROOT = Path(__file__).resolve().parents[1]
PYTHON = '/home/jinx/miniconda3/envs/anylogic-env/bin/python'


def _encoded(value):
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False).encode() + b'\n'


def _reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _exclusive_json(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(_encoded(document))
        stream.flush()
        os.fsync(stream.fileno())


@contextmanager
def _launch_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'launch.lock').open('a+b') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def authorization(suite, suite_reference, user_instruction):
    """Render a scoped record only for a caller's explicit execution instruction."""
    if not isinstance(user_instruction, str) or not user_instruction.strip():
        raise ValueError('explicit nonempty user execution instruction required')
    return dict(decision='approved', approved_by='user', execution_authorized=True,
                ack_frozen_bg128_single_seed=True,
                execution_profile=suite['execution_profile'], pair_id=suite['pair_id'],
                source_user_instruction=user_instruction.strip(), campaign_id=suite['campaign_id'],
                campaign_sha256=suite_reference['sha256'], max_new_shifts=suite['new_shift_cap'],
                legacy_reserved_shifts=suite['legacy_reserved_shifts'], active_h2_reserved_shifts=0,accounting='merged_all_ledgers_no_h2_future_reservation',
                maximum_total_after_registration=suite['maximum_total_if_approved'],
                approved_at_utc=datetime.now(timezone.utc).isoformat(),
                approval_scope='one_frozen_campaign_budget_and_execution',
                scope_interpretation='User requested the next registered improvement: Actor LR 1e-4; frozen existing formal_32ep_v1 protocol means 16 updates x 32 fresh AnyLogic shifts = 512. No extra smoke or independent evaluation is included.',
                note='One explicit authorization covers this frozen experiment; phase/update/status checks do not ask again.')


def start(suite_path, *, authorize=False, user_instruction=None, approval_path=None,
          launch_directory=None, spawn=None):
    """Gate first, then detach exactly the fixed local runtime; never click ALP.

    ``--authorize`` must represent an actual user instruction for this concrete
    frozen experiment.  Merely asking to prepare or accelerate training does
    not authorize another physical shift. Existing approved records are reused
    after validation; pending records are never promoted or overwritten.
    """
    from .gate import preflight, require_prior_complete
    suite_path = Path(suite_path).resolve()
    ref = _reference(suite_path)
    suite, _, _ = preflight(suite_path)  # no listener/budget/output side effects
    if _reference(suite_path) != ref:
        raise ValueError('suite bytes changed during preflight')
    # Read-only preparation may run independently. Actual collection requires
    # current executable engineering evidence and the frozen BG128 contract.
    require_prior_complete(suite)
    plan = suite_path.parent
    approval_path = Path(approval_path or plan / 'approval.json').resolve()
    launch_directory = Path(launch_directory or plan / 'launch').resolve()
    output = Path(suite['python_output']).resolve()
    # A launch record must not itself consume the as-yet unused run identity.
    if launch_directory == output or output in launch_directory.parents:
        raise ValueError('launch directory must be outside physical run output')
    if approval_path == output or output in approval_path.parents:
        raise ValueError('approval record must be outside physical run output')
    if not approval_path.is_file() and not authorize:
        raise ValueError('one explicit frozen-campaign budget/execution authorization required')
    if not approval_path.is_file() and not (isinstance(user_instruction, str) and user_instruction.strip()):
        raise ValueError('--authorize also requires the exact --user-instruction')
    snapshot = inspect(suite_path, launch_directory=launch_directory)
    if snapshot['status'] == 'COMPLETE':
        return dict(status='ALREADY_COMPLETE', suite=ref, new_processes=0)
    if snapshot['status'] in ('STOP_REQUIRES_REVIEW', 'UNKNOWN_REQUIRES_RECONCILIATION'):
        raise ValueError(snapshot['status'] + '; never clear or retry automatically')
    if snapshot['physical_shifts']['claimed']:
        raise ValueError('outstanding physical claim requires recovery review; no automatic resubmission')
    with _launch_lock(launch_directory):
        for name in ('launch_receipt.json', 'launch_intent.json'):
            if (launch_directory / name).exists():
                raise ValueError('existing launch record; use status, never launch twice or auto-resume')
        if not approval_path.exists():
            _exclusive_json(approval_path, authorization(suite, ref, user_instruction))
        # Checks the structured record, current sources/schema/budget and every
        # frozen phase before any child or real run directory is created.
        observed, _, checked = preflight(suite_path, approval_path=approval_path)
        if observed != suite or checked is None or _reference(suite_path) != ref:
            raise ValueError('authorized suite changed or approval missing before launch')
        snapshot = inspect(suite_path, launch_directory=launch_directory)
        if snapshot['status'] in ('STOP_REQUIRES_REVIEW', 'UNKNOWN_REQUIRES_RECONCILIATION') or snapshot['physical_shifts']['claimed']:
            raise ValueError('runtime recovery state changed during launch checks')
        command = [PYTHON, '-B', '-m', 'business_graph_stability_campaign_v1.runtime', 'run',
                   '--suite', str(suite_path), '--approval', str(approval_path)]
        intent = dict(schema='business-graph-stability-campaign-launch-intent-v1', suite=ref,
                      approval=_reference(approval_path), command=command,
                      cwd=str(ROOT), started_at_utc=datetime.now(timezone.utc).isoformat())
        _exclusive_json(launch_directory / 'launch_intent.json', intent)
        environment = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
        with (launch_directory / 'controller.log').open('ab', buffering=0) as log:
            child = (spawn or subprocess.Popen)(command, cwd=str(ROOT), env=environment,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, close_fds=True)
        identity = _process_identity(child.pid)
        receipt = dict(intent, schema='business-graph-stability-campaign-launch-receipt-v1', pid=child.pid,
                       process_start_ticks=identity.get('start_ticks'), controller_log=str(launch_directory / 'controller.log'))
        _exclusive_json(launch_directory / 'launch_receipt.json', receipt)
    return dict(status='CONTROLLER_STARTED_WAIT_FOR_READY', suite=ref, pid=child.pid,
                approval=str(approval_path), controller_log=receipt['controller_log'],
                manual_anylogic_launches_required=1, new_controller_processes=1,
                instruction='Wait for LISTENING and the first ready barrier, then click AnyLogic Run once.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    launch = commands.add_parser('start')
    launch.add_argument('--suite', required=True)
    launch.add_argument('--approval')
    launch.add_argument('--launch-directory')
    launch.add_argument('--authorize', action='store_true')
    launch.add_argument('--user-instruction')
    status = commands.add_parser('status')
    status.add_argument('--suite', required=True)
    status.add_argument('--launch-directory')
    status.add_argument('--pid', type=int, help='Explicit opt-in to validated /proc CPU inspection')
    args = parser.parse_args(argv)
    if args.command == 'start':
        result = start(args.suite, authorize=args.authorize, user_instruction=args.user_instruction,
                       approval_path=args.approval, launch_directory=args.launch_directory)
    else:
        result = inspect(args.suite, launch_directory=args.launch_directory, pid=args.pid)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
