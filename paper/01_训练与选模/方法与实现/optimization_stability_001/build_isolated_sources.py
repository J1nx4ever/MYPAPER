"""One-time, exclusive source derivation; never launch or write a run/ledger.

Keep the original BG009 implementation and its sealed source pins intact.
The generated source is checked by AST and bitwise synthetic comparisons.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
PLAN = Path(__file__).resolve().parent
REPORT = ROOT / 'reports/token_attention_001/business_graph_summary_009/optimization_stability_001'


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(value)


def replace(text, before, after):
    assert text.count(before) == 1, before
    return text.replace(before, after)


def main():
    assert ROOT.name == 'lagrangian_ppo_v1'
    numerical = ROOT / 'business_graph_stability_v1'
    campaign = ROOT / 'business_graph_stability_campaign_v1'
    assert not numerical.exists() and not campaign.exists()
    preserved = list((ROOT/'business_graph_summary_v1').glob('*.py'))
    preserved += list((ROOT/'business_graph_summary_campaign_v1').glob('*.py'))
    preserved += list((ROOT/'business_graph_summary_evaluation_v1').glob('*.py'))
    preserved += list((ROOT/'runs').glob('*budget*'))
    preserved += list((ROOT/'anylogic').glob('*.alp'))
    preserved += [ROOT.parents[1]/'python/episode_budget_ppo.py']
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in preserved if p.is_file()}
    put(REPORT/'preserved_before.json', json.dumps(before, indent=2, sort_keys=True)+'\n')
    put(numerical/'__init__.py', '"""Isolated BG128 Actor-LR stability trial; original models and targets reused."""\n')
    for name in ('targets','c_features','models','features_cache','contract','policy','behavior','evaluation','storage'):
        put(numerical/(name+'.py'), f'"""Unchanged frozen BG009 implementation."""\nfrom business_graph_summary_v1.{name} import *\n')
    put(numerical/'admission.py', (ROOT/'business_graph_summary_v1/admission.py').read_text())
    source = (ROOT/'business_graph_summary_v1/learner.py').read_text()
    source = source.replace("'business-graph-summary-ppo-v1'", "'business-graph-stability-ppo-v1'")
    source = source.replace("'business-graph-summary-ppo-checkpoint-v1'", "'business-graph-stability-ppo-checkpoint-v1'")
    source = source.replace("'business-graph-summary-actor-v1'", "'business-graph-stability-actor-v1'")
    source = replace(source, 'actor_lr=3e-4,', 'actor_lr=1e-4,')
    source = replace(source, '    # Include inherited implementations, not only the local re-export modules.',
        "    sources += sorted(p for p in (project/'business_graph_summary_v1').glob('*.py')\n"
        "                      if not p.name.startswith('test_'))\n"
        '    # Include inherited implementations, not only the local re-export modules.')
    source = replace(source, "        for epoch in range(p['epochs']):", "        epoch_diagnostics = []\n        for epoch in range(p['epochs']):\n            epoch_loss_start = len(losses)")
    source = replace(source, "            progress('epoch_end',epoch+1)",
        "            epoch_diagnostics.append(self._epoch_diagnostics(\n"
        "                episodes, old_logs, losses[epoch_loss_start:], epoch+1))\n"
        "            progress('epoch_end',epoch+1)")
    source = replace(source, '                    actor_optimizer_steps=len(losses),critic_optimizer_steps=len(losses),',
        '                    actor_optimizer_steps=len(losses),critic_optimizer_steps=len(losses),\n'
        '                    epoch_diagnostics=epoch_diagnostics,')
    source = replace(source, '    def _metrics(self, episodes, *, old_logs=None):', '''    def _epoch_diagnostics(self, episodes, old_logs, losses, epoch):
        """Four compact summaries per update; no RNG draws or optimizer changes.

        KL and clip are post-epoch full-batch replay against the frozen behavior,
        whereas gradient distributions are actual preclip minibatch norms.
        """
        metrics = self._metrics(episodes, old_logs=old_logs)
        gradients = {}
        for role in ('actor', 'critic'):
            norms = np.asarray([row[role+'_grad_norm'] for row in losses], dtype=np.float64)
            gradients[role] = dict(count=len(norms), mean=float(norms.mean()),
                min=float(norms.min()), p50=float(np.quantile(norms,.5)),
                p90=float(np.quantile(norms,.9)), p95=float(np.quantile(norms,.95)),
                max=float(norms.max()), clip_hits=int((norms>self.config['ppo']['max_grad_norm']).sum()),
                clip_fraction=float((norms>self.config['ppo']['max_grad_norm']).mean()))
        return dict(epoch=epoch, replay=metrics, preclip_gradient_norms=gradients,
                    behavior_reference='frozen_batch_old_log_probs',
                    optimizer_control='diagnostic_only_fixed_four_epochs')

    def _metrics(self, episodes, *, old_logs=None):''')
    put(numerical/'learner.py', source)
    for path in (ROOT/'business_graph_summary_campaign_v1').glob('*.py'):
        if path.name.startswith('test_'):
            continue
        value = path.read_text().replace('business_graph_summary_campaign_v1', 'business_graph_stability_campaign_v1')
        value = value.replace('business_graph_summary_v1', 'business_graph_stability_v1')
        value = value.replace('formal_training_001', 'optimization_stability_001')
        value = value.replace('business-graph-summary', 'business-graph-stability')
        value = value.replace('BUSINESS_GRAPH_SUMMARY', 'BUSINESS_GRAPH_STABILITY')
        if path.name == 'gate.py':
            value = replace(value, "('business_graph_stability_v1','business_graph_stability_campaign_v1')",
                "('business_graph_summary_v1','business_graph_summary_campaign_v1',\n"
                "                    'business_graph_summary_evaluation_v1',\n"
                "                    'business_graph_stability_v1','business_graph_stability_campaign_v1')")
            value = replace(value, "for name in ('bootstrap.py','prepare_training.py'):",
                "for name in ('bootstrap.py','prepare_training.py','protocol.json'):")
        if path.name == 'budget.py':
            start = value.index('def preceding_snapshot():')
            end = value.index('def check_collisions(', start)
            value = value[:start]+'''def preceding_snapshot():
    # Latest entry includes BG009 training and its 100-shift fixed evaluation.
    from business_graph_summary_evaluation_v1.budget import budget_snapshot
    return budget_snapshot()

def preceding_reference():
    from business_graph_summary_evaluation_v1.budget import preceding_reference as inherited, GLOBAL_PATH as eval_path
    state = inherited()
    state['bg128_independent_extension_before'] = reference(eval_path) if eval_path.is_file() else None
    return state

'''+value[end:]
            value = replace(value, '                # Check identities/seeds again under the writer lock.',
                "                require(preceding_reference() == suite['budget_state_before'], 'preceding budget changed during registration')\n"
                '                check_collisions(suite, preceding_snapshot())\n'
                '                # Check identities/seeds again under the writer lock.')
        if path.name == 'launch.py':
            value = value.replace('User requested one random-seed formal training;',
                'User requested the next registered improvement: Actor LR 1e-4;')
        put(campaign/path.name, value)
    old_plan = ROOT/'change_plans/token_attention_001/business_graph_summary_009/formal_training_001'
    for name in ('bootstrap.py','prepare_training.py'):
        value = (old_plan/name).read_text()
        value = value.replace('business_graph_summary_campaign_v1','business_graph_stability_campaign_v1')
        value = value.replace('business_graph_summary_v1','business_graph_stability_v1')
        value = value.replace('business-graph-summary-formal-seed1-20261118-001',
                              'business-graph-stability-lr1e4-seed1-20261118-001')
        value = value.replace('RUN_BASE = 9900001','RUN_BASE = 9960001')
        value = value.replace("    require(init == 75597143230906, 'initialization identity differs from disclosed seed')\n", '')
        value = value.replace('Actor and scalar Critic use the same fixed',
                              'Actor LR is 1e-4; other PPO coefficients unchanged. Actor and scalar Critic use the same fixed')
        put(PLAN/name, value)


if __name__ == '__main__':
    main()
