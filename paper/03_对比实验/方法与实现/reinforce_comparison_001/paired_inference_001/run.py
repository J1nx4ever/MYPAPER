"""Prepare, verify or explicitly start the frozen S02 paired inference."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from change_plans.token_attention_001.business_graph_summary_009.reinforce_comparison_001.paired_inference_001 import (
    budget, prepare as plan, runtime
)
from business_graph_summary_campaign_v1.launch import _exclusive_json


def verify():
    suite, phases = plan.load_package()
    result = runtime.verify(suite, phases)
    if plan.LEDGER.exists():
        result['registered_shifts'] = len(budget.InferenceLedger(suite, phases).snapshot()['jobs'])
    else:
        result['budget'] = budget.assert_available(suite,budget.preceding_snapshot())
        result['registered_shifts'] = 0
    return result


def start(authorize):
    import fcntl
    import socket

    suite, phases = plan.load_package()
    runtime.verify(suite, phases)
    plan.require(plan.LEDGER.is_file() or authorize, 'first physical start requires --authorize-start')
    lane = budget.LOCKS / f"port-{suite['rpc_port']}.lock"
    lane.parent.mkdir(parents=True, exist_ok=True)
    with lane.open('a+b') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with socket.socket() as probe:
            probe.settimeout(.2)
            plan.require(probe.connect_ex(('127.0.0.1', suite['rpc_port'])) != 0, 'RPC port is already serving another campaign')
        if not plan.LEDGER.is_file():
            path = budget.authority_path(suite)
            if not path.exists():
                _exclusive_json(path, budget.authorization(suite))
            budget.InferenceLedger(suite, phases).register()
        return runtime.run(suite, phases, budget.InferenceLedger(suite, phases))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    commands.add_parser('prepare')
    commands.add_parser('verify')
    commands.add_parser('run').add_argument('--authorize-start', action='store_true')
    args = parser.parse_args()
    value = plan.prepare() if args.action == 'prepare' else verify() if args.action == 'verify' else start(args.authorize_start)
    print(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
