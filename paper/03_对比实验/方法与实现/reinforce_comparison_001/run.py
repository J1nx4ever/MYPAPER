"""Prepare, verify or explicitly start the externally controlled BG128 comparison."""

import argparse
import fcntl
import json
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from change_plans.token_attention_001.business_graph_summary_009.reinforce_comparison_001 import budget
from change_plans.token_attention_001.business_graph_summary_009.reinforce_comparison_001 import prepare as plan
from business_graph_summary_campaign_v1.launch import _exclusive_json
from business_graph_reinforce_v1 import runtime


def verified():
    suite, phases = plan.load_package()
    technical = runtime.verify(suite, phases)
    if plan.LEDGER.is_file():
        registered = budget.ReinforceLedger(suite, phases).snapshot()
        technical['registered_shifts'] = len(registered['jobs'])
    else:
        technical['budget'] = budget.assert_available(suite, budget.budget_snapshot())
        technical['registered_shifts'] = 0
    return technical


def run(*, authorize_start):
    suite, phases = plan.load_package()
    runtime.verify(suite, phases)
    authority = budget.authority_path(suite)
    plan.original.require(plan.LEDGER.is_file() or authorize_start,
                          'first start requires --authorize-start after reviewing launch_identity.json')
    lane = budget.LOCKS / f"port-{suite['rpc_port']}.lock"
    lane.parent.mkdir(parents=True, exist_ok=True)
    with lane.open('a+b') as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"RPC port {suite['rpc_port']} is reserved by another campaign") from error
        with socket.socket() as probe:
            probe.settimeout(.2)
            plan.original.require(probe.connect_ex(('127.0.0.1', suite['rpc_port'])) != 0,
                                  'configured RPC port already has a listener')
        if not plan.LEDGER.is_file():
            if not authority.exists():
                _exclusive_json(authority, budget.authorization(suite))
            budget.ReinforceLedger(suite, phases).register()
        ledger = budget.ReinforceLedger(suite, phases)
        return runtime.run(suite, phases, ledger)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    sub.add_parser('prepare', help='publish a fresh offline package only')
    sub.add_parser('verify', help='check the package, code and identifiers without starting training')
    execute = sub.add_parser('run', help='run the externally controlled training campaign')
    execute.add_argument('--authorize-start', action='store_true', help='authorize the initial 1,024-shift registration')
    args = parser.parse_args()
    if args.action == 'prepare':
        result = plan.prepare()
    elif args.action == 'verify':
        result = verified()
    else:
        result = run(authorize_start=args.authorize_start)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
