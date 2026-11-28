"""Freeze evaluation of the completed, PPO-step-matched S02 policy."""
from pathlib import Path
from types import SimpleNamespace

from ...paired_inference_001 import prepare as prior
from .. import prepare as training
from business_graph_summary_campaign_v1.plan_io import load_reference, reference
from weak_plan import bind


PLAN = Path(__file__).resolve().parent
ROOT = training.ROOT
CAMPAIGN = 'bg128-reinforce-ppo-steps-s02-u32-paired700-20261128-001'
FIRST_RUN = 10096001
COUNT, PER_CASE, RPC_PORT = prior.COUNT, prior.PER_CASE, prior.RPC_PORT
PREPARED = PLAN / 'prepared_001'
SUITE = PREPARED / 'suite.json'
OUTPUT = ROOT / 'runs' / CAMPAIGN
MATERIAL = OUTPUT.with_name(CAMPAIGN + '_materials')
CONTROL = OUTPUT / 'control'
REPORT = ROOT / 'reports' / PLAN.relative_to(ROOT / 'change_plans')
LEDGER = OUTPUT / 'budget.sqlite'
COMPLETION = dict(path=str(training.OUTPUT / 'suite_complete.json'),
                  sha256='f35fa611f55edae32a876a8475e407e39b4db7b554556d9310ba78d2758a8b9b')
PAIR_KEYS = prior.PAIR_KEYS
original, require = prior.original, prior.require
historical, verify_physical_starts = prior.historical, prior.verify_physical_starts


# The previous evaluation planner's baseline is Full S02 PPO. Its only training
# fields vary here so the physical scenario and pair schedule stay identical.
_documents = bind(prior.documents, training=SimpleNamespace(SUITE=training.SUITE,
                  COMPLETION=prior.training.COMPLETION, PROTECTED_CORE=training.parent.PROTECTED_CORE),
                  CAMPAIGN=CAMPAIGN, FIRST_RUN=FIRST_RUN, PREPARED=PREPARED, SUITE=SUITE,
                  OUTPUT=OUTPUT, CONTROL=CONTROL, MATERIAL=MATERIAL, REPORT=REPORT,
                  LEDGER=LEDGER, COMPLETION=COMPLETION)


def documents():
    sealed = load_reference(COMPLETION)
    parent = load_reference(sealed['suite'])
    plan = parent['reinforce']['actor_steps_by_update']
    require(sealed['actor_steps_by_update'] == plan and
            sealed['actor_optimizer_steps'] == sum(plan) == 23612 and len(plan) == 32 and
            sealed['critic_optimizer_steps'] == sealed['evaluation_shifts'] == 0 and
            not (training.CONTROL / 'STOP.properties').exists(),
            'completed training does not match the frozen 23,612-step plan')
    fixed, phases = _documents()
    fixed['schema'] = 'bg128-full-s02-step-matched-reinforce-u32-fixed-paired-v1'
    fixed['source_actor_optimizer_steps'] = 23612
    return fixed, phases


def source_pins():
    pins = training.source_pins()
    files = [PLAN / name for name in ('prepare.py', 'budget.py', 'run.py', 'runtime.py', 'prepare_java.py')]
    files += [prior.PLAN / name for name in ('prepare.py', 'budget.py', 'run.py', 'runtime.py',
                                             'paired_metrics.py', 'prepare_java.py')]
    pins.update({str(path): reference(path)['sha256'] for path in files})
    return dict(sorted(pins.items()))


load_package = bind(prior.load_package, documents=documents, SUITE=SUITE,
                    PREPARED=PREPARED, source_pins=source_pins)
prepare = bind(prior.prepare, __package__=__package__, CAMPAIGN=CAMPAIGN, FIRST_RUN=FIRST_RUN,
               PREPARED=PREPARED, SUITE=SUITE, OUTPUT=OUTPUT, MATERIAL=MATERIAL,
               CONTROL=CONTROL, REPORT=REPORT, LEDGER=LEDGER, documents=documents,
               load_package=load_package, source_pins=source_pins)
