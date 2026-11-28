"""Count optimizer steps and training row visits in the two completed S02 runs."""
from pathlib import Path
import sys

FOLDER = Path(__file__).resolve().parent
ROOT = FOLDER.parents[4]
sys.path.insert(0, str(ROOT))
from business_graph_summary_campaign_v1.plan_io import load_reference, reference
from business_graph_summary_evaluation_v1.metrics import _json, _publish
from change_plans.token_attention_001.business_graph_summary_009.reinforce_comparison_001 import prepare as plan
from signal_audit import COMPLETION


def main():
    full = load_reference(plan.COMPLETION)
    rf = load_reference(COMPLETION)
    params = load_reference(full['suite'])['learner_config']['ppo']
    assert params['epochs'] == 4 and params['minibatch_size'] == 32
    rows = []
    for update, (checkpoint, summary_ref) in enumerate(zip(full['training_checkpoints'], rf['updates'], strict=True), 1):
        ppo_path = Path(checkpoint['path']).parent / 'training_summary.json'
        a = load_reference(reference(ppo_path))['update']
        b = load_reference(summary_ref)['reinforce']
        m = (a['transitions'] + params['minibatch_size'] - 1) // params['minibatch_size']
        assert a['epochs'] == params['epochs'] and a['actor_optimizer_steps'] == a['critic_optimizer_steps'] == m * 4
        assert b['actor_optimizer_steps'] == 1 and a['episodes'] == b['episodes'] == 32
        rows.append(dict(batch=update, ppo_decisions=a['transitions'], reinforce_decisions=b['transitions'],
                         ppo_minibatches_per_epoch=m, ppo_actor_steps=a['actor_optimizer_steps'],
                         ppo_critic_steps=a['critic_optimizer_steps'], reinforce_actor_steps=b['actor_optimizer_steps'],
                         reinforce_critic_steps=0, ppo_actor_row_visits=4*a['transitions'],
                         reinforce_actor_row_visits=b['transitions']))
    assert len(rows) == 32
    totals = {key: sum(row[key] for row in rows) for key in rows[0] if key not in ('batch','ppo_minibatches_per_epoch')}
    result = dict(ppo_source=plan.COMPLETION, reinforce_source=COMPLETION,
                  shifts_per_arm=1024, batches_per_arm=len(rows), totals=totals, batches=rows,
                  actor_step_ratio=totals['ppo_actor_steps']/totals['reinforce_actor_steps'],
                  actor_row_visit_ratio=totals['ppo_actor_row_visits']/totals['reinforce_actor_row_visits'])
    directory = ROOT / 'reports' / FOLDER.relative_to(ROOT / 'change_plans')
    _publish(directory/'actor_budget.json', _json(result))
    print(totals)


if __name__ == '__main__':
    main()
