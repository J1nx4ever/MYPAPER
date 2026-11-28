"""Register this training only against the combined S02 training and inference history."""

from .. import budget as base
from ..paired_inference_001 import budget as preceding
from . import prepare as plan
from weak_plan import bind


LOCKS = base.LOCKS
authority_path = base.authority_path
rows = bind(base._rows, plan=plan, _prior_rows=bind(base._prior_rows, SUITE=plan.SUITE))
assert_available = bind(base.assert_available, plan=plan, _rows=rows)


def authorization(suite):
    return dict(base.authorization(suite), schema='bg128-ppo-steps-matched-reinforce-authority-v1',
                actor_optimizer_steps=suite['actor_optimizer_steps_total'])


class ReinforceLedger(base.ReinforceLedger):
    __init__ = bind(base.ReinforceLedger.__init__, FORMAT='bg128-ppo-steps-matched-reinforce-ledger-v1',
                    authorization=authorization, plan=plan)
    register = bind(base.ReinforceLedger.register, assert_available=assert_available, predecessor=preceding)
    snapshot = bind(base.ReinforceLedger.snapshot, plan=plan)


budget_snapshot = bind(base.budget_snapshot, predecessor=preceding, plan=plan,
                       assert_available=assert_available, ReinforceLedger=ReinforceLedger)
