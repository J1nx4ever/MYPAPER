"""Start or attach this audited BG128 recovery without reserving another shift."""
import argparse
import json
from pathlib import Path
import socket
import sys

ROOT = Path('/home/jinx/Faith/experiments/lagrangian_ppo_v1')
sys.path.insert(0, str(ROOT))


def check_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind(('127.0.0.1', 50051))
        except OSError as error:
            raise ValueError('RPC port 50051 occupied; inspect existing controller') from error


def trigger(recovery_path, run_index, environment_seed, recording_id, *, clicked=False,
            starter=None, port_check=None):
    from business_graph_summary_campaign_v1.plan_io import reference
    from business_graph_summary_recovery_v1.gate import preflight
    from business_graph_summary_recovery_v1.launch import start
    path = Path(recovery_path).resolve()
    if Path.cwd().resolve() != ROOT or path.parent != Path(__file__).resolve().parent:
        raise ValueError('fixed BG128 recovery bridge working directory/package required')
    descriptor = json.loads(path.read_bytes())
    if descriptor['bridge'] != reference(__file__):
        raise ValueError('recovery bridge byte SHA changed')
    first = descriptor['remaining_jobs'][0]
    if (first['ordinal'], run_index, environment_seed, recording_id) != (
            1, first['run_index'], first['experiment_seed'], first['recording_id']):
        raise ValueError('only original ordinal 1 can start this audited recovery')
    preflight(path, require_reconciled=True)
    if not clicked:
        return dict(status='RECOVERY_ONE_CLICK_CHECK_ONLY', remaining_physical_shifts=511,
                    new_shift_cap=0, listener_started=False, budget_registered=False)
    receipt_path = path.parent / 'launch/launch_receipt.json'
    if receipt_path.is_file():
        from business_graph_summary_campaign_v1.health import _process_identity
        receipt = json.loads(receipt_path.read_bytes())
        observed = _process_identity(receipt['pid'])
        if (receipt.get('recovery') != reference(path) or
                observed.get('start_ticks') != receipt.get('process_start_ticks') or
                observed.get('state') == 'Z' or
                'business_graph_summary_recovery_v1.runtime' not in observed.get('command', []) or
                str(path) not in observed.get('command', [])):
            raise ValueError('preserved recovery launch is not this exact live controller; do not resubmit')
        return dict(status='ATTACHED_TO_EXACT_EXISTING_RECOVERY_CONTROLLER', recovery=reference(path),
                    pid=receipt['pid'], new_controller_processes=0,
                    new_shift_cap=0, manual_anylogic_launches_required=0)
    (port_check or check_port)()
    result = (starter or start)(path)
    if result['status'] != 'RECOVERY_CONTROLLER_STARTED_WAIT_FOR_READY':
        raise ValueError(str(result))
    return dict(result, manual_anylogic_launches_required=0, new_shift_cap=0,
                instruction='This Run started the remaining 511 original jobs; do not click Run again.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recovery', required=True)
    parser.add_argument('--run-index', required=True, type=int)
    parser.add_argument('--environment-seed', required=True, type=int)
    parser.add_argument('--recording-id', required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--from-anylogic-run', action='store_true')
    args = parser.parse_args()
    print(json.dumps(trigger(args.recovery, args.run_index, args.environment_seed,
                             args.recording_id, clicked=args.from_anylogic_run),
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
