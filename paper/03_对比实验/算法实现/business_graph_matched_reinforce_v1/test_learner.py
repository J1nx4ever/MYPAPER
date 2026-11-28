"""Run real BG128 on small sealed shifts and compare against the explicit clipped objective."""
import math

import pytest
import torch

from business_graph_summary_v1.learner import default_config
from business_graph_summary_v1.test_learner import assert_tree_equal, synthetic_batch
from business_graph_matched_reinforce_v1.learner import MatchedReinforce, _batches


def test_partitions_keep_reference_tail_and_adapt_smaller_or_larger_trajectories():
    for n, buckets, expected in ((65, 3, [32, 32, 1]), (61, 3, [21, 20, 20]), (68, 2, [34, 34])):
        order = torch.randperm(n)
        chunks = _batches(order, 32, buckets)
        assert list(map(len, chunks)) == expected
        assert torch.equal(torch.cat(chunks), order)


def test_cost_sign_fixed_behavior_and_resume(tmp_path):
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        check_cost_sign_and_resume(tmp_path)
    finally:
        torch.set_num_threads(original_threads)


def check_cost_sign_and_resume(tmp_path):
    config = default_config(source_kind='synthetic', seed=373, execution_profile='synthetic_test_v1')
    opts = dict(actor_config=config, actor_steps_by_update=[8, 4])
    model, reference = MatchedReinforce(**opts), MatchedReinforce(**opts)
    batch = synthetic_batch(model)
    opts_ppo = config['ppo']
    actions = torch.cat([torch.tensor(e['actions'], dtype=torch.long) for e in batch])
    logs = torch.cat([torch.tensor(e['old_log_probs'], dtype=torch.float32) for e in batch])
    costs = torch.cat([torch.full((len(e['actions']),), math.fsum(e['targets'].step_costs)) for e in batch])
    shuffle = torch.Generator().manual_seed(config['seed'])
    for _ in range(4):
        for ids in torch.tensor_split(torch.randperm(len(logs), generator=shuffle), 2):
            reference.actor_optimizer.zero_grad()
            offset = 0
            for ep in batch:
                chosen = ids[(ids >= offset) & (ids < offset + len(ep['actions']))]
                local = (chosen - offset).tolist()
                offset += len(ep['actions'])
                if not local:
                    continue
                states = ep['states'][:-1]
                policy = reference.actor.distribution(torch.tensor(states[local]), reference._prefix(states, local))
                ratio = (policy.log_prob(actions[chosen]) - logs[chosen]).exp()
                reward_adv = -costs[chosen]
                surrogate = torch.minimum(reward_adv * ratio, reward_adv * ratio.clamp(.8, 1.2))
                (-(surrogate.sum()+opts_ppo['entropy_coef']*policy.entropy().sum()) / len(ids)).backward()
            torch.nn.utils.clip_grad_norm_(reference.actor.parameters(), opts_ppo['max_grad_norm'])
            reference.actor_optimizer.step()
    result = model.update(batch, batch_id='batch_1')
    assert result['actor_optimizer_steps'] == 8 and result['optimizer_decision_visits'] == 4 * len(logs)
    assert result['first_step_max_abs_log_ratio'] < 1e-5 and result['post_first_step_max_abs_log_ratio'] > 1e-5
    assert torch.equal(shuffle.get_state(), model.generator.get_state())
    for actual, expected in zip(model.actor.parameters(), reference.actor.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)
    checkpoint = tmp_path/'matched.pt'
    seal = model.save_checkpoint(checkpoint)
    restored = MatchedReinforce.from_checkpoint(checkpoint, expected_sha256=seal['sha256'], expected_settings=model.settings)
    second = synthetic_batch(model, recording='second')
    assert model.update(second, batch_id='batch_2') == restored.update(second, batch_id='batch_2')
    assert_tree_equal(model.actor_optimizer.state_dict(), restored.actor_optimizer.state_dict())
    assert model.consumed_batches['batch_2']['cumulative_actor_optimizer_steps'] == 12
    assert model.consumed_batches['batch_2']['baseline_before'] == result['baseline_after']
    with pytest.raises(ValueError, match='exhausted'):
        model.update(second)
