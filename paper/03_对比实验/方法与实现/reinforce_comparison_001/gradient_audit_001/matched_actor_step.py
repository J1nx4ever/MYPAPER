"""Exercise exactly one full-batch S02 PPO Actor step on the sealed historical u0 data."""
import json
import math
from pathlib import Path
import sys

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

from business_graph_summary_campaign_v1.plan_io import load_reference, reference
from business_graph_summary_v1.learner import JointCostPPO, _target_hash
from business_graph_summary_v1.models import digest
from python.hdf5_arrays import read_array_bundle
from change_plans.token_attention_001.business_graph_summary_009.reinforce_comparison_001.prepare import COMPLETION


def main():
    torch.set_num_threads(1)
    complete = load_reference(COMPLETION)
    source = complete['initial_checkpoint']
    folder = Path(complete['training_checkpoints'][0]['path']).parent
    prepared = json.loads((folder/'prepared.json').read_bytes())
    transaction = prepared['intent']
    old = json.loads((folder/'training_summary.json').read_bytes())
    assert transaction == old['transaction'] and transaction['parent_checkpoint'] == source
    assert prepared['checkpoint'] == complete['training_checkpoints'][0]
    target_ref = prepared['training_targets']
    assert target_ref == reference(target_ref['path'])
    targets, metadata = read_array_bundle(target_ref['path'])
    assert metadata['transaction'] == transaction and _target_hash(targets) == old['update']['target_sha256']
    model = JointCostPPO.from_checkpoint(source['path'], expected_sha256=source['sha256'])
    before = model.policy_version
    assert model.update_index == 0 and before == old['update']['policy_before']
    critic_before = digest(model.critic)
    episodes, actions, logs, mapping = [], [], [], []
    for index, (ref, run) in enumerate(zip(transaction['sources'], transaction['run_indices'], strict=True)):
        manifest = load_reference(ref)
        hdf = Path(ref['path']).with_suffix('').with_suffix('')
        assert reference(hdf)['sha256'] == manifest['hdf5_sha256']
        with h5py.File(hdf, 'r') as f:
            identity = json.loads(f.attrs['identity_json'])
            assert identity == manifest['identity'] and identity['policy_version'] == before and identity['run_index'] == run
            states, action, log = f['states'][:], f['transitions/action'][:], f['transitions/old_log_prob'][:]
            assert len(states) == len(action) + 1 and len(log) == len(action)
        episodes.append(dict(states=states))
        actions.append(action)
        logs.append(log)
        mapping.extend((index, local) for local in range(len(action)))
    n = len(mapping)
    assert len(episodes) == 32 and n == old['update']['transitions'] == 5844
    assert targets['actor_advantages'].shape == (n,)
    raw_advantages = targets['advantages']
    np.testing.assert_allclose(targets['actor_advantages'], (raw_advantages-raw_advantages.mean()) / (raw_advantages.std()+1e-8))
    p = model.config['ppo']
    actions = torch.tensor(np.concatenate(actions), dtype=torch.int64)
    raw_logs = torch.tensor(np.concatenate(logs), dtype=torch.float32)
    advantages = torch.tensor(targets['actor_advantages'], dtype=torch.float32)
    model.actor_optimizer.zero_grad(set_to_none=True)
    # Passing all decision IDs gives _actor_grad a single 5,844-row denominator.
    # Internally it calls backward once per episode and releases each graph.
    loss = model._actor_grad(episodes, mapping, torch.arange(n), actions, raw_logs, advantages)
    preclip = torch.nn.utils.clip_grad_norm_(model.actor.parameters(), p['max_grad_norm'], error_if_nonfinite=True)
    assert all(param.grad is not None for param in model.actor.parameters()) and math.isfinite(loss)
    assert not model.actor_optimizer.state
    model.actor_optimizer.step()
    assert all(st['step'].item() == 1 for st in model.actor_optimizer.state.values())
    after = model.policy_version
    assert len(model.actor_optimizer.state) == len(list(model.actor.parameters()))
    assert before != after and digest(model.critic) == critic_before and not model.critic_optimizer.state
    result = dict(source_checkpoint=source, training_targets=target_ref, audit_source=reference(__file__),
                  shifts=32, decisions=n, original_batch_actor_steps=old['update']['actor_optimizer_steps'],
                  matched_actor_steps=1, critic_steps_in_this_actor_only_probe=0, physical_shifts_launched=0,
                  saved_models=0, actor_digest_before=before, actor_digest_after_in_memory=after,
                  cost_gae_lambda=p['gae_lambda'], probability_ratio_clip=p['clip_range'],
                  advantage_processing=p['advantage_processing'], entropy_coefficient=p['entropy_coef'],
                  gradient_norm_before_clip=float(preclip), max_gradient_norm=p['max_grad_norm'],
                  actor_lr=p['actor_lr'], adam_eps=p['adam_eps'], actor_loss=loss)
    out = ROOT/'reports'/Path(__file__).resolve().parent.relative_to(ROOT/'change_plans')/'matched_actor_step.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
