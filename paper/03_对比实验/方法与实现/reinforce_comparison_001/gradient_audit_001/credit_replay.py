"""Compare credit assignment on identical logged actions at three frozen policies."""
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import torch

FOLDER = Path(__file__).resolve().parent
ROOT = FOLDER.parents[4]
sys.path.insert(0, str(ROOT))
from business_graph_summary_campaign_v1.plan_io import load_reference, reference
from business_graph_summary_evaluation_v1.metrics import _json, _publish
from business_graph_summary_storage_v1.episode import Event, _read_table, build_memory
from business_graph_summary_v1.admission import frozen_actor
from business_graph_summary_v1.policy import replay_policy
from business_graph_summary_v1.targets import build_composite_targets
from causal_update import advantage_weights, next_baseline, shift_loss
from signal_audit import COMPLETION


def targets_from_file(f):
    physical = json.loads(f.attrs['evaluation_json'])
    return build_composite_targets(
        build_memory(_read_table(f, 'material_events', Event)),
        wait_deltas=f['transitions/wait_delta_seconds'][:],
        **{key: physical[key] for key in ('ore_tons', 'waste_tons', 'total_wait_seconds')})


def score_rows(actor, f):
    """Exact analytical score w.r.t. the final shared linear scorer's 128 weights."""
    features = []
    def capture(_layer, inputs):
        features.append(inputs[0].detach().double())
    actions = f['transitions/action'][:]
    handle = actor.scorer[-1].register_forward_pre_hook(capture)
    try:
        with torch.no_grad():
            replay = replay_policy(actor, f['states'][:-1], actions=actions)
    finally:
        handle.remove()
    p = replay['probabilities'].double()
    h = torch.cat(features)
    assert torch.allclose(replay['log_probs'].double(), torch.tensor(f['transitions/old_log_prob'][:]),
                          atol=1e-5, rtol=0)
    return (h[torch.arange(len(h)), actions] - (p.unsqueeze(-1) * h).sum(dim=1)).numpy()


def dispersion(gradients):
    vectors = np.stack(gradients)
    average = vectors.mean(axis=0)
    return dict(shifts=len(vectors), mean_gradient_norm=float(np.linalg.norm(average)),
                sample_covariance_trace=float(np.square(vectors - average).sum() / (len(vectors) - 1)))


def main():
    completed = load_reference(COMPLETION)
    baseline = np.zeros(5)
    comparison = {}
    torch.set_num_threads(1)
    for update, summary_ref in enumerate(completed['updates'], 1):
        summary = load_reference(summary_ref)
        intent, metrics = summary['intent'], summary['reinforce']
        selected = update in (1, 16, 32)
        if selected:
            assert reference(intent['actor']['path']) == intent['actor']
            actor, digest = frozen_actor(intent['actor']['path'])
            assert digest == metrics['rollout_actor_digest']
        contributions = {name: [] for name in ('full_shift', 'remaining_same_scalar', 'remaining_phase_ema')}
        batch = []
        for source in intent['sources']:
            receipt = load_reference(source['receipt'])
            path = receipt['references']['episode']
            assert reference(path['path']) == path and receipt['references']['episode_manifest'] == source['dataset']
            with h5py.File(path['path'], 'r') as f:
                targets = targets_from_file(f)
                batch.append(targets)
                if not selected:
                    continue
                scores = score_rows(actor, f)
            scalar = metrics['baseline_before']
            weights = dict(full_shift=np.full(len(scores), -targets.mc_returns[0] - scalar),
                           remaining_same_scalar=advantage_weights(targets, np.repeat(scalar, 5)),
                           remaining_phase_ema=advantage_weights(targets, baseline))
            for method, w in weights.items():
                contributions[method].append(w @ scores)
            # The proposed loss must supply exactly the declared coefficients to autograd.
            logp = torch.zeros(len(scores), dtype=torch.float64, requires_grad=True)
            shift_loss(logp, weights['remaining_phase_ema'], 32).backward()
            np.testing.assert_allclose(logp.grad.numpy(), -weights['remaining_phase_ema']/32, rtol=0, atol=0)
        assert len(batch) == 32
        np.testing.assert_allclose(-np.mean([x.mc_returns[0] for x in batch]), metrics['mean_return'],
                                   rtol=0, atol=1e-10)
        if selected:
            stats = {name: dispersion(values) for name, values in contributions.items()}
            comparison[str(update)] = dict(baseline_before=baseline.tolist(), estimates=stats, actor=intent['actor'])
        baseline = next_baseline(batch, baseline)
    report = dict(completion=COMPLETION, batches=comparison, physical_shifts_started=0,
                  policy_weights_updated=0, sampled_batches=[1, 16, 32],
                  scope='Across-shift sample dispersion of the 128 final shared scorer weights. '
                        'Each estimate uses the original on-policy actions at the same frozen Actor. '
                        'This is an offline comparison, not a new policy performance estimate.')
    folder = ROOT / 'reports' / FOLDER.relative_to(ROOT / 'change_plans')
    _publish(folder / 'credit_replay.json', _json(report))
    print(json.dumps(comparison))


if __name__ == '__main__':
    main()
