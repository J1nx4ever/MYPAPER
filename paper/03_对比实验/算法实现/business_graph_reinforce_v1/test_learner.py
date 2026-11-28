"""Offline checks with the real BG128 Actor and complete synthetic shifts."""

from copy import deepcopy
from dataclasses import replace
import math

import numpy as np
import pytest
import torch

from business_graph_summary_v1.admission import frozen_actor
from business_graph_summary_v1.learner import default_config
from business_graph_summary_v1.policy import replay_policy
from business_graph_summary_v1.targets import build_composite_targets
from business_graph_summary_v1.test_learner import assert_tree_equal, synthetic_batch
from grade_event_memory_v1.events import build_memory

from . import learner as reinforce


@pytest.fixture(autouse=True)
def one_thread():
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(original)


def new_learner():
    return reinforce.FullShiftReinforce(actor_config=default_config(
        source_kind='synthetic', seed=373, execution_profile='synthetic_test_v1'))


def shorten(episode, actor):
    """Remove a dispatch from the same complete material and waiting tape."""
    events = []
    for event in episode['memory'].events:
        if event.request_id == 'q1':
            continue
        if event.kind in ('DECISION', 'END') and event.decision_sequence > 1:
            event = replace(event, decision_sequence=event.decision_sequence - 1)
        events.append(event)
    episode['memory'] = build_memory(events)
    episode['states'] = episode['states'][[0, 2, 3]]
    episode['actions'] = episode['actions'][[0, 2]]
    episode['wait_deltas'] = np.array([3., 1447.])
    episode['evaluation']['total_wait_seconds'] = 1450.
    episode['targets'] = build_composite_targets(episode['memory'], wait_deltas=episode['wait_deltas'],
                                                 **episode['evaluation'])
    with torch.no_grad():
        episode['old_log_probs'] = replay_policy(actor, episode['states'][:-1], actions=episode['actions'])[
            'log_probs'].numpy().astype(np.float64)


def test_equation_19_matches_direct_objective_with_unequal_episode_lengths(monkeypatch):
    monkeypatch.setattr(reinforce, 'REPLAY_CHUNK', 2)
    model, direct = new_learner(), new_learner()
    assert model.policy_version == direct.policy_version
    episodes = synthetic_batch(model)
    shorten(episodes[1], model.actor)

    shift_returns = [-math.fsum(row['targets'].step_costs) for row in episodes]
    terms = []
    for episode, reward in zip(episodes, shift_returns, strict=True):
        replay = replay_policy(direct.actor, episode['states'][:-1], actions=episode['actions'], chunk_size=2)
        terms.append(-reward * replay['log_probs'].double().sum())
    expected_loss = torch.stack(terms).mean()
    expected_loss.backward()
    direct.actor_optimizer.step()

    result = model.update(episodes, batch_id='first')
    assert result['policy_loss'] == pytest.approx(float(expected_loss.detach()), abs=1e-8)
    assert result['mean_return'] == pytest.approx(np.mean(shift_returns))
    assert result['mean_combined_cost'] == -result['mean_return']
    assert result['baseline_before'] == 0
    assert result['baseline_after'] == result['mean_return'] / 2
    assert result['transitions'] == 5
    assert result['actor_optimizer_steps'] == 1
    assert result['actor_grad_norm'] > 0
    assert not hasattr(model, 'critic')
    for actual, expected in zip(model.actor.parameters(), direct.actor.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=4e-8)
        state = model.actor_optimizer.state[actual]
        assert state['step'].item() == 1
        torch.testing.assert_close(state['exp_avg'], direct.actor_optimizer.state[expected]['exp_avg'],
                                   rtol=1e-5, atol=4e-8)


def test_checkpoint_continuation_and_actor_reader(tmp_path):
    model = new_learner()
    original_digest = model.policy_version
    first_result = model.update(synthetic_batch(model, recording='first'))
    assert model.policy_version != original_digest
    model.runtime_state.update(campaign={'name': 'offline-reinforce'}, update=1)
    artifact = model.save_actor(tmp_path / 'actor.pt')
    frozen, version = frozen_actor(artifact['path'], allow_small_synthetic=True)
    assert not any(parameter.requires_grad for parameter in frozen.parameters())
    assert version == model.policy_version
    checkpoint = model.save_checkpoint(tmp_path / 'checkpoint.pt')

    restored = reinforce.FullShiftReinforce.from_checkpoint(checkpoint['path'],
        expected_actor_config=model.config, expected_settings=model.settings, expected_sha256=checkpoint['sha256'])
    assert restored.baseline == first_result['baseline_after']
    assert restored.update_index == 1
    assert restored.runtime_state == model.runtime_state
    assert_tree_equal(model.actor_optimizer.state_dict(), restored.actor_optimizer.state_dict())
    restored.bind_actor_snapshot(artifact['path'])
    assert restored.policy_version == version

    batch = synthetic_batch(model, recording='second')
    direct, replayed = model.update(batch), restored.update(batch)
    assert direct == replayed
    assert direct['baseline_before'] == first_result['baseline_after']
    assert direct['baseline_after'] == (direct['baseline_before'] + direct['mean_return']) / 2
    assert_tree_equal(model.actor.state_dict(), restored.actor.state_dict())
    assert_tree_equal(model.actor_optimizer.state_dict(), restored.actor_optimizer.state_dict())
    assert len(restored.consumed_batches) == 2
    assert len(restored.consumed_episodes) == 4

    with pytest.raises(ValueError, match='expected byte SHA'):
        reinforce.FullShiftReinforce.from_checkpoint(checkpoint['path'], expected_sha256='b' * 64)
    with pytest.raises(ValueError, match='settings differ from campaign'):
        reinforce.FullShiftReinforce.from_checkpoint(checkpoint['path'],
            expected_settings=dict(model.settings, baseline_decay=.25))


def test_reject_invalid_or_consumed_trajectories_before_updating():
    model = new_learner()
    original = model.policy_version
    batch = synthetic_batch(model)
    invalid = deepcopy(batch)
    invalid[1]['old_log_probs'][0] += .1

    with pytest.raises(ValueError, match='configured shift count'):
        model.update(batch[:1])
    with pytest.raises(ValueError, match='old action probabilities'):
        model.update(invalid)
    with pytest.raises(ValueError, match='duplicate complete shift'):
        model.update([batch[0], batch[0]])
    assert model.policy_version == original
    assert model.update_index == 0
    assert model.baseline == 0
    assert not model.actor_optimizer.state

    model.update(batch)
    after_update = model.policy_version
    with pytest.raises(ValueError, match='stale/off-policy'):
        model.update(batch)
    with pytest.raises(ValueError, match='already consumed'):
        model.update(synthetic_batch(model))
    assert model.policy_version == after_update
    assert model.update_index == 1
