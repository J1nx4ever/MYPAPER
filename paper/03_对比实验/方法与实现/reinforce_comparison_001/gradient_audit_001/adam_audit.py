"""Read the sealed S02 optimizer and actor history to audit every realized update."""
from pathlib import Path
import json
import math
import statistics
import sys

import h5py
import torch
from torch.nn.functional import cosine_similarity

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

from business_graph_summary_v1.models import actor_from_config
from business_graph_summary_v1.policy import replay_policy
from business_graph_summary_v1.targets import grade_cost
from business_graph_summary_campaign_v1.plan_io import load_reference, reference


BASE = ROOT/'runs/bg128-reinforce-full-shift-20261127-s02'
ARM = BASE / 'arms/s01_BG128'
REPORT = ROOT/'reports' / Path(__file__).resolve().parent.relative_to(ROOT/'change_plans') / 'adam_audit.json'


def load(path):
    signed = json.loads(path.with_suffix('.pt.manifest.json').read_text())
    seal = json.loads(path.with_suffix('.pt.seal.json').read_text())
    assert reference(path)['sha256'] == signed['sha256'] == seal['sha256']
    assert reference(path.with_suffix('.pt.manifest.json'))['sha256'] == seal['manifest_sha256']
    return torch.load(path, map_location='cpu', weights_only=False)


def physical_cost(hdf):
    contract = json.loads(hdf.attrs['composite_contract_json'])
    terminal = json.loads(hdf.attrs['evaluation_json'])
    events = hdf['material_events']
    close = events['kind'][:] == b'WINDOW_CLOSE'
    assert list(events['window'][:][close]) == [0, 1, 2, 3]
    values = zip(events['window_grade'][:][close], events['window_grade_valid'][:][close], strict=True)
    grade = math.fsum(grade_cost(float(g)) if valid else grade_cost(None, empty=True) for g, valid in values)
    production = sum(contract[f'{material}_shortfall_weight'] * max(0., 1 - terminal[f'{material}_tons']
                     / contract[f'{material}_target_tons']) for material in ('ore', 'waste'))
    return grade + production + contract['wait_weight'] * terminal['total_wait_seconds'] / contract['wait_reference_seconds']


@torch.no_grad()
def verify_head(previous, current, intent, inferred, update):
    """Analytic derivative of the last scorer weight using the masked softmax identity."""
    actor = actor_from_config(previous['actor_config']).eval()
    actor.load_state_dict(previous['actor'])
    width = actor.scorer[-1].in_features
    analytic = torch.zeros(width, dtype=torch.float64)
    max_logp_error, decisions = 0., 0
    stored_costs = []
    sources = intent['sources']
    for source in sources:
        receipt = load_reference(source['receipt'])
        manifest = load_reference(source['dataset'])
        ref = receipt['references']['episode']
        assert reference(ref['path']) == ref and ref['sha256'] == manifest['hdf5_sha256']
        assert receipt['identity']['policy_version'] == previous['actor_digest']
        with h5py.File(ref['path'], 'r') as hdf:
            states = hdf['states'][:-1]
            actions = hdf['transitions/action'][:]
            logs = torch.tensor(hdf['transitions/old_log_prob'][:], dtype=torch.float64)
            cost = physical_cost(hdf)
        stored_costs.append(cost)
        features = []
        def collect_input(module, args):
            features.append(args[0].double())
        hook = actor.scorer[-1].register_forward_pre_hook(collect_input)
        try:
            replay = replay_policy(actor, states, actions=actions, chunk_size=16)
        finally:
            hook.remove()
        h = torch.cat(features)
        probs = replay['probabilities'].double()
        score = (h[torch.arange(len(actions)), torch.tensor(actions).long()] - (probs[..., None] * h).sum(1)).sum(0)
        analytic += (cost + previous['baseline']) * score / len(sources)
        max_logp_error = max(max_logp_error, (replay['log_probs'].double() - logs).abs().max().item())
        decisions += len(actions)
    cos = cosine_similarity(analytic, inferred, dim=0).item()
    relative = ((analytic - inferred).norm() / inferred.norm()).item()
    assert cos > 0.99999 and relative < 1e-4 and max_logp_error < 1e-5
    return dict(update=update, shifts=len(sources), decisions=decisions, head_dimensions=width, mean_cost=statistics.mean(stored_costs),
                analytic_vs_inferred_cosine=cos, relative_gradient_error=relative, max_replay_abs_logp_error=max_logp_error)


def audit():
    torch.set_num_threads(1)
    previous = load(ARM/'initial.pt')
    complete = json.loads((BASE/'suite_complete.json').read_text())
    assert complete['status'] == 'COMPLETE' and complete['training_updates'] == 32
    names = [name for name, _ in actor_from_config(previous['actor_config']).named_parameters()]
    rows, head_checks = [], []
    for t in range(1, 33):
        folder = ARM/'updates'/f'{t:04d}'
        current = load(folder/'checkpoint.pt')
        metadata = json.loads((folder/'training_summary.json').read_text())['reinforce']
        group, = current['optimizer']['param_groups']
        assert len(names) == len(group['params']) == len(current['optimizer']['state']) == 71
        assert group['betas'] == (.9, .999) and group['eps'] == 1e-5 and group['lr'] == 3e-4
        assert not group['amsgrad'] and not group.get('decoupled_weight_decay', False)
        assert current['update_index'] == t and metadata['actor_optimizer_steps'] == 1
        assert metadata['rollout_actor_digest'] == previous['actor_digest'] and metadata['actor_digest'] == current['actor_digest']
        assert not group['maximize'] and group['weight_decay'] == 0
        beta, _ = group['betas']
        n2 = step2 = dot = 0.
        for idx, name in zip(group['params'], names, strict=True):
            state = current['optimizer']['state'][idx]
            assert state['step'].item() == t
            old_m = previous['optimizer']['state'][idx]['exp_avg'].double() if t > 1 else 0.
            g = (state['exp_avg'].double() - beta * old_m) / (1.-beta)
            delta = current['actor'][name].double() - previous['actor'][name].double()
            a, b = g.reshape(-1), delta.reshape(-1)
            dot += (a @ b).item()
            n2 += (a @ a).item()
            step2 += (b @ b).item()
            if t in (1, 15, 32) and name == 'scorer.2.weight':
                intent = json.loads((folder/'intent.json').read_bytes())
                head_checks.append(verify_head(previous, current, intent, a, t))
        norm = math.sqrt(n2)
        recorded = metadata['actor_grad_norm']
        rows.append(dict(update=t, recorded_grad_norm=recorded, inferred_grad_norm=norm,
                         grad_norm_relative_error=abs(norm-recorded)/recorded,
                         step_norm=math.sqrt(step2), loss_gradient_dot_step=dot,
                         cosine_loss_descent_and_step=-dot/(norm*math.sqrt(step2))))
        previous = current
    cos = [r['cosine_loss_descent_and_step'] for r in rows]
    d = dict(steps=len(rows), negative_dot_steps=sum(r['loss_gradient_dot_step'] < 0 for r in rows),
             min_descent_cosine=min(cos), median_descent_cosine=statistics.median(cos), max_descent_cosine=max(cos),
             max_grad_norm_relative_error=max(r['grad_norm_relative_error'] for r in rows))
    assert len(head_checks) == 3 and d['max_grad_norm_relative_error'] < 1e-6
    return dict(summary=d, optimizer=dict(betas=[.9, .999], eps=1e-5, lr=3e-4, parameter_tensors=71),
                physical_shifts_launched=0, training_completion=reference(BASE/'suite_complete.json'),
                audit_source=reference(__file__), analytic_head_checks=head_checks, per_update=rows)


if __name__ == '__main__':
    result = audit()
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ('summary', 'analytic_head_checks')}))
