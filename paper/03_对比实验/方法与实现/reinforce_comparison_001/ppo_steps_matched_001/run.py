"""Prepare, verify or explicitly launch the matched S02 training."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from change_plans.token_attention_001.business_graph_summary_009.reinforce_comparison_001 import run as original
from change_plans.token_attention_001.business_graph_summary_009.reinforce_comparison_001.ppo_steps_matched_001 import (
    budget, prepare as plan, runtime
)
from weak_plan import bind


verify = bind(original.verified, plan=plan, budget=budget, runtime=runtime)
run = bind(original.run, plan=plan, budget=budget, runtime=runtime)
main = bind(original.main, plan=plan, verified=verify, run=run)


if __name__ == '__main__':
    main()
