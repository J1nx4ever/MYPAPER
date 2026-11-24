"""Start or attach the exact frozen single-seed BG128 controller from AnyLogic."""
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


def trigger(suite_path, run_index, environment_seed, recording_id, *, clicked=False,
            starter=None, port_check=None):
    from business_graph_stability_campaign_v1.plan_io import PLAN, reference, load_suite
    from business_graph_stability_campaign_v1.gate import preflight
    from business_graph_stability_campaign_v1.launch import start
    path = Path(suite_path).resolve()
    if Path.cwd().resolve() != ROOT or path.parent.parent != PLAN:
        raise ValueError('fixed BG128 bridge working directory/package required')
    suite, _ = load_suite(path)
    if suite['one_click_entry'] != reference(__file__):
        raise ValueError('bridge byte SHA changed')
    first = suite['jobs'][0]
    if (run_index, environment_seed, recording_id) != (
            first['run_index'], first['experiment_seed'], first['recording_id']):
        raise ValueError('only the first frozen AnyLogic job can start this campaign')
    preflight(path)
    if not clicked:
        return dict(status='ONE_CLICK_CHECK_ONLY', physical_shifts=suite['new_shift_cap'],
                    listener_started=False, budget_registered=False)
    receipt_path = path.parent / 'launch/launch_receipt.json'
    if receipt_path.is_file():
        from business_graph_stability_campaign_v1.health import _process_identity
        receipt = json.loads(receipt_path.read_bytes())
        observed = _process_identity(receipt['pid'])
        preflight(path, approval_path=path.parent / 'approval.json')
        if (receipt.get('suite') != reference(path) or
                observed.get('start_ticks') != receipt.get('process_start_ticks') or
                observed.get('state') == 'Z' or
                'business_graph_stability_campaign_v1.runtime' not in observed.get('command', []) or
                str(path) not in observed.get('command', [])):
            raise ValueError('preserved launch is not this exact live controller; do not resubmit')
        return dict(status='ATTACHED_TO_EXACT_EXISTING_CONTROLLER', suite=reference(path),
                    pid=receipt['pid'], new_controller_processes=0,
                    manual_anylogic_launches_required=0)
    (port_check or check_port)()
    instruction = (
        'User instructions: 按照最新的改进顺序下一步进行；用和这次相同的种子进行训练和推理；'
        '同时应该从原始共同初始化 Checkpoint 重新开始。 '
        f'User clicked AnyLogic Run for disclosed BG128 campaign {suite["campaign_id"]}, '
        'one initialization seed, exactly 512 training shifts in 16 batches of 32. '
        'Architecture fixed at two GAT layers, four relation groups and original C128. '
        'Actor LR is 1e-4; other PPO coefficients unchanged. Actor and scalar Critic use the same fixed h=.004, production8/8, wait=.2 joint cost. '
        'The first formal batch checks the complete RPC/collection/update path and belongs to these 512 shifts. '
        'No separate smoke, physical evaluation, replay training or additional campaign.')
    result = (starter or start)(path, authorize=True, user_instruction=instruction)
    if result['status'] != 'CONTROLLER_STARTED_WAIT_FOR_READY':
        raise ValueError(str(result))
    return dict(result, manual_anylogic_launches_required=0,
                instruction='This Run started the campaign; do not click Run again.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', required=True)
    parser.add_argument('--run-index', required=True, type=int)
    parser.add_argument('--environment-seed', required=True, type=int)
    parser.add_argument('--recording-id', required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--from-anylogic-run', action='store_true')
    args = parser.parse_args()
    print(json.dumps(trigger(args.suite, args.run_index, args.environment_seed,
                             args.recording_id, clicked=args.from_anylogic_run),
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
