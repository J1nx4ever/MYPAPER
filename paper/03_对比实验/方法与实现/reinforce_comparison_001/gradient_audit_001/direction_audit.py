"""Check the saved Adam inputs against an independent score objective on sealed shifts."""
from pathlib import Path
import json
import math
import sys

import h5py
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT))
from business_graph_summary_campaign_v1.plan_io import load_reference, reference
from business_graph_summary_v1.learner import _load_checkpoint
from business_graph_summary_v1.models import actor_from_config, digest
from single_critic_task_query_v1.features_cache import SummaryFeatureCache

RUN = ROOT / 'runs/bg128-reinforce-full-shift-20261127-s02'
ARM = RUN / 'arms/s01_BG128'
REPORT = ROOT / 'reports' / HERE.relative_to(ROOT / 'change_plans') / 'direction_audit.json'


def frozen_inputs(intent, baseline, feature_config):
    """Read the physical measurements directly, independently of the learner's step costs."""
    chunks, returns = [], []
    source_count = len(intent['sources'])
    assert source_count == 32
    for index, source in enumerate(intent['sources']):
        receipt = load_reference(source['receipt'])
        refs = receipt['references']
        identity = receipt['identity']
        manifest = load_reference(source['dataset'])
        assert source['dataset'] == refs['episode_manifest'] and manifest['identity'] == identity
        assert manifest['hdf5_sha256'] == refs['episode']['sha256']
        assert reference(refs['episode']['path']) == refs['episode'] and receipt['normal_completion']
        assert identity['run_index'] == source['run_index'] and identity['policy_sha256'] == intent['actor']['sha256']
        with h5py.File(refs['episode']['path'], 'r') as file:
            physical = json.loads(file.attrs['evaluation_json'])
            cost = json.loads(file.attrs['composite_contract_json'])
            events = file['material_events']
            closed = events['kind'][:] == b'WINDOW_CLOSE'
            assert events['window'][:][closed].tolist() == [0, 1, 2, 3]
            window_cost = math.fsum(
                min(2., max(abs(float(grade) - cost['grade_center']) - cost['grade_deadband'], 0.) / cost['grade_scale'])
                if valid else cost['empty_window_cost']
                for grade, valid in zip(events['window_grade'][:][closed], events['window_grade_valid'][:][closed], strict=True))
            shift_cost = (window_cost + cost['ore_shortfall_weight'] * max(0., 1 - physical['ore_tons'] / cost['ore_target_tons'])
                          + cost['waste_shortfall_weight'] * max(0., 1 - physical['waste_tons'] / cost['waste_target_tons'])
                          + cost['wait_weight'] * physical['total_wait_seconds'] / cost['wait_reference_seconds'])
            states = file['states'][:-1]
            actions = torch.as_tensor(file['transitions/action'][:], dtype=torch.long)
            online = torch.as_tensor(file['transitions/old_log_prob'][:], dtype=torch.float64)
        returns.append(-shift_cost)
        coefficient = (-shift_cost - baseline) / source_count
        cache = SummaryFeatureCache(states, feature_config=feature_config)
        for start in range(0, len(actions), 32):
            end = min(start + 32, len(actions))
            ids = list(range(start, end))
            chunks.append((index, torch.from_numpy(states[start:end]), actions[start:end], online[start:end],
                           cache.prefix(ids), coefficient))
    return chunks, returns


def flat(items):
    return torch.cat([item.detach().flatten().double() for item in items])


def angle(a, b):
    return float(torch.dot(a, b) / (torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)))


def saved_gradients(actor, previous, current):
    """Invert Adam's first-moment recurrence to read the gradient actually delivered."""
    group, = current['optimizer']['param_groups']
    beta = group['betas'][0]
    inputs, moves = [], []
    for pid, (name, p) in zip(group['params'], actor.named_parameters(), strict=True):
        new = current['optimizer']['state'].get(pid)
        old = previous['optimizer']['state'].get(pid)
        old_step = int(old['step']) if old is not None else 0
        if new is None or int(new['step']) == old_step:
            grad = torch.zeros_like(p)
        else:
            assert int(new['step']) == old_step + 1
            old_m = old['exp_avg'].double() if old is not None else 0.
            grad = (new['exp_avg'].double() - beta * old_m) / (1 - beta)
        inputs.append(grad)
        moves.append(current['actor'][name].double() - previous['actor'][name].double())
    return flat(inputs), flat(moves)


def independent_objective(actor, chunks, *, gradient=False):
    """Reward-ascent score with chosen masked logits; no production update function."""
    scores, halves = [], []
    log_error = 0.
    for parity in (0, 1):
        actor.zero_grad(set_to_none=True)
        for i, states, actions, old_logp, prefix, coefficient in chunks:
            if i % 2 != parity:
                continue
            with torch.set_grad_enabled(gradient):
                logits = actor(states, prefix).double()
                logp = logits.gather(1, actions[:, None]).squeeze(1) - logits.logsumexp(dim=1)
                assert torch.isfinite(logp).all()
                value = coefficient * logp.sum()
                if gradient:
                    value.backward()
            log_error = max(log_error, float((logp.detach() - old_logp).abs().max()))
            scores.append(float(value.detach()))
        if gradient:
            halves.append(flat(p.grad if p.grad is not None else torch.zeros_like(p) for p in actor.parameters()))
    return math.fsum(scores), log_error, halves


def audit(update, completion):
    prepared = load_reference(reference(ARM / 'updates' / f'{update:04}' / 'prepared.json'))
    assert prepared['training_summary'] == completion['updates'][update - 1]
    summary = load_reference(prepared['training_summary'])
    intent, metrics = prepared['intent'], summary['reinforce']
    assert intent == summary['intent']
    before, _ = _load_checkpoint(intent['parent_checkpoint']['path'])
    after, _ = _load_checkpoint(prepared['checkpoint']['path'])
    for ref in (intent['actor'], intent['parent_checkpoint'], prepared['checkpoint']):
        assert ref == reference(ref['path'])
    actor = actor_from_config(before['actor_config']).eval()
    actor.load_state_dict(before['actor'])
    assert digest(actor) == metrics['rollout_actor_digest'] == before['actor_digest']
    assert metrics['baseline_before'] == before['baseline']
    chunks, rewards = frozen_inputs(intent, before['baseline'], actor.feature_config)
    input_g, move = saved_gradients(actor, before, after)
    base, log_error, halves = independent_objective(actor, chunks, gradient=True)
    reward_g = halves[0] + halves[1]
    relative_error = (input_g + reward_g).norm().item() / input_g.norm().item()
    print(f'u{update}: saved gradient vs independent negative gradient rel={relative_error:.6g}, '
          f'cosine={angle(input_g, reward_g):.8g}', flush=True)
    # Alpha is a fraction of the actual archived Adam parameter displacement.
    def score_at(alpha):
        for name, p in actor.named_parameters():
            origin, target = before['actor'][name].double(), after['actor'][name].double()
            with torch.no_grad():
                p.copy_(origin + alpha * (target - origin))
        return independent_objective(actor, chunks)[0]
    epsilon = 0.0625
    fd = (score_at(epsilon) - score_at(-epsilon)) / (2 * epsilon)
    actual_gain = score_at(1.) - base
    predicted = float(torch.dot(move, reward_g))
    assert log_error < 1e-4 and math.isclose(base, -metrics['policy_loss'], abs_tol=0.0002)
    assert math.isclose(np.mean(rewards), metrics['mean_return'], abs_tol=1e-9)
    assert math.isclose(input_g.norm().item(), metrics['actor_grad_norm'], rel_tol=2e-5)
    assert relative_error < 2e-4, 'archived Adam input differs from the negative of the reward gradient'
    return dict(update=update, shift_count=len(rewards), decisions=sum(len(c[2]) for c in chunks),
                training_summary=prepared['training_summary'], parent=intent['parent_checkpoint'], checkpoint=prepared['checkpoint'],
                baseline_before=before['baseline'], mean_reward=float(np.mean(rewards)), actor_sha=intent['actor']['sha256'],
                max_abs_replayed_chosen_logp_minus_online=log_error, independent_reward_score=base,
                saved_optimizer_gradient_norm=input_g.norm().item(), independent_reward_gradient_norm=reward_g.norm().item(),
                loss_gradient_vs_reward_ascent_cosine=angle(input_g, reward_g), relative_norm_of_gradient_sum=relative_error,
                reward_gradient_dot_actual_adam_step=predicted, central_difference_along_actual_step=fd,
                relative_error_vs_directional_autograd=abs(fd-predicted)/max(abs(predicted), 1e-12),
                central_difference_fraction_of_step=epsilon, same_batch_reward_score_gain_after_actual_step=actual_gain,
                interleaved_half_batch_reward_gradient_cosine=angle(halves[0], halves[1]))


if __name__ == '__main__':
    torch.set_num_threads(2)
    completion_ref = reference(RUN / 'suite_complete.json')
    completion = load_reference(completion_ref)
    assert completion['status'] == 'COMPLETE' and completion['training_updates'] == 32
    rows = []
    for update in (1, 32):
        item = audit(update, completion)
        rows.append(item)
        print(json.dumps(item, allow_nan=False), flush=True)
    report = dict(schema='bg128-reinforce-archived-gradient-direction-v1', source=completion_ref,
                  batches=rows, new_physical_shifts=0, optimizer_steps_executed=0)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
