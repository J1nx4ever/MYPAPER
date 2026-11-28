"""Prepare, verify or dispatch the 700-shift S02 u32 matched evaluation."""
import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[6]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from change_plans.token_attention_001.business_graph_summary_009.reinforce_comparison_001.paired_inference_001 import run as prior
from change_plans.token_attention_001.business_graph_summary_009.reinforce_comparison_001.ppo_steps_matched_001.paired_inference_001 import (
    budget, prepare as plan, runtime,
)
from weak_plan import bind


verify = bind(prior.verify, plan=plan, budget=budget, runtime=runtime)
start = bind(prior.start, plan=plan, budget=budget, runtime=runtime)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('prepare')
    commands.add_parser('verify')
    commands.add_parser('run').add_argument('--authorize-start', action='store_true')
    args = parser.parse_args()
    result = {'prepare': plan.prepare, 'verify': verify,
              'run': lambda: start(args.authorize_start)}[args.command]()
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
