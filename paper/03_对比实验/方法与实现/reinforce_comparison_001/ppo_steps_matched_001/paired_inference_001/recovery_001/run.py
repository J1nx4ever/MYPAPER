"""Prepare, verify, or run the interrupted evaluation's newly authorized suffix."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[7]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from change_plans.token_attention_001.business_graph_summary_009.reinforce_comparison_001.paired_inference_001 import run as base
from change_plans.token_attention_001.business_graph_summary_009.reinforce_comparison_001.ppo_steps_matched_001.paired_inference_001.recovery_001 import budget, prepare as plan
from weak_plan import bind


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('prepare')
    commands.add_parser('verify')
    commands.add_parser('run').add_argument('--authorize-start', action='store_true')
    args = parser.parse_args()
    if args.command == 'prepare':
        result = plan.prepare()
    else:
        from change_plans.token_attention_001.business_graph_summary_009.reinforce_comparison_001.ppo_steps_matched_001.paired_inference_001.recovery_001 import runtime
        if args.command == 'verify':
            result = bind(base.verify, plan=plan, budget=budget, runtime=runtime)()
        else:
            result = bind(base.start, plan=plan, budget=budget, runtime=runtime)(args.authorize_start)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
