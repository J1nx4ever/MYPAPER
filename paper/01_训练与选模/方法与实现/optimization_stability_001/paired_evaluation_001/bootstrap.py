"""AnyLogic one-click bridge for the explicitly requested100 independent shifts."""
import argparse
import json
from pathlib import Path
import socket
import sys

ROOT=Path('/home/jinx/Faith/experiments/lagrangian_ppo_v1')
sys.path.insert(0,str(ROOT))


def check_port():
    with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as probe:
        try:probe.bind(('127.0.0.1',50051))
        except OSError as error:raise ValueError('RPC50051 occupied; inspect existing controller') from error


def trigger(suite_path,run_index,environment_seed,recording_id,*,clicked=False,starter=None,port_check=None):
    from business_graph_stability_evaluation_v1.gate import preflight,PLAN,reference
    from business_graph_stability_evaluation_v1.launch import start
    path=Path(suite_path).resolve()
    if Path.cwd().resolve()!=ROOT or path.parent.parent!=PLAN:
        raise ValueError('fixed independent-evaluation package and working directory required')
    suite,_,approval=preflight(path)
    if suite['one_click_entry']!=reference(__file__):raise ValueError('bridge bytes changed')
    first=suite['jobs'][0]
    if (run_index,environment_seed,recording_id)!=(first['run_index'],first['experiment_seed'],first['recording_id']):
        raise ValueError('only the first frozen evaluation job can start this controller')
    if not clicked:return dict(status='EVALUATION_CHECK_ONLY',physical_shifts=100,listener_started=False)
    if approval is None:raise ValueError('frozen user100-shift authorization missing')
    receipt=path.parent/'launch/launch_receipt.json'
    if receipt.exists():
        from business_graph_stability_campaign_v1.health import _process_identity
        old=json.loads(receipt.read_bytes());current=_process_identity(old['pid'])
        if (old['suite']!=reference(path) or current.get('start_ticks')!=old.get('process_start_ticks') or
            current.get('state')=='Z' or 'business_graph_stability_evaluation_v1.runtime' not in current.get('command',[]) or
            str(path) not in current.get('command',[])):
            raise ValueError('preserved evaluation controller is not live; inspect, never resubmit')
        return dict(status='ATTACHED_TO_EXACT_EVALUATION_CONTROLLER',pid=old['pid'])
    (port_check or check_port)()
    return (starter or start)(path)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite',required=True)
    parser.add_argument('--run-index',required=True,type=int)
    parser.add_argument('--environment-seed',required=True,type=int)
    parser.add_argument('--recording-id',required=True)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check',action='store_true');mode.add_argument('--from-anylogic-run',action='store_true')
    args=parser.parse_args()
    print(json.dumps(trigger(args.suite,args.run_index,args.environment_seed,args.recording_id,clicked=args.from_anylogic_run)))
