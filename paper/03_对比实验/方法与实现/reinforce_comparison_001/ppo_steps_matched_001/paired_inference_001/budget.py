"""Register the new evaluation against the complete historical budget."""
from . import prepare as plan
from .. import budget as training
from ...paired_inference_001 import budget as prior
from business_graph_summary_campaign_v1.launch import _launch_lock
from business_graph_summary_campaign_v1.plan_io import load_reference, reference
from weak_plan import bind


LOCKS = prior.LOCKS
authority_path = prior.authority_path
preceding_snapshot = training.budget_snapshot
rows_for_collision = bind(prior.rows_for_collision, plan=plan,
                          _prior_rows=bind(prior.rows_for_collision.__globals__['_prior_rows'], SUITE=plan.SUITE))
_assert_available = bind(prior.assert_available, plan=plan, rows_for_collision=rows_for_collision)


def assert_available(suite, history):
    campaign = history['campaigns'][plan.training.CAMPAIGN]
    completed = load_reference(suite['training_completion'])
    commits = campaign['commits']
    plan.require(len(commits) == completed['training_updates'] == 32 and
                 commits['32']['checkpoint_ref'] == suite['frozen_checkpoint'] and
                 all(history['jobs'][str(job['run_index'])]['status'] == 'COMPLETE'
                     for job in campaign['campaign']['jobs']),
                 'training budget or u32 commit is incomplete')
    return _assert_available(suite, history)


_authority = bind(prior.authorization, plan=plan)


def authorization(suite):
    return dict(_authority(suite), schema='bg128-step-matched-reinforce-u32-evaluation-authority-v1',
                checkpoint=suite['frozen_checkpoint'], training_completion=suite['training_completion'])


class InferenceLedger(prior.InferenceLedger):
    __init__ = bind(prior.InferenceLedger.__init__, plan=plan, authorization=authorization)
    snapshot = bind(prior.InferenceLedger.snapshot, plan=plan)
    register = bind(prior.InferenceLedger.register, assert_available=assert_available,
                    preceding_snapshot=preceding_snapshot)


budget_snapshot = bind(prior.budget_snapshot, plan=plan, preceding_snapshot=preceding_snapshot,
                       assert_available=assert_available, InferenceLedger=InferenceLedger)
