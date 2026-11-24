"""Regression checks for retirement, causal action replay, and scaled terminal labels."""
import numpy as np
import pytest
import torch

from fleet_policy import (FleetLearner, SummaryFeatureCache, episode_node_order, models,
                          pack_states, scene, shift_targets)
from business_graph_summary_v1.test_learner import episode_on_production_map
from business_graph_summary_v1.test_support import business_state


def episode(model, ordinal):
    historical = episode_on_production_map()
    raw = historical['states'].copy()
    raw[:, 11:13] *= .8
    closures = []
    tape = historical['memory']
    for w in tape.settlements:
        step = next(i for i,q in enumerate(tape.queries) if q.current_seq < w.seq < q.next_seq)
        closures.append(dict(window=w.window, step=step, tons=w.tons, metal_tons=w.metal))
    # Synthetic complete trial: retire an in-system truck after the first action.
    non_egos = np.flatnonzero(raw[:-1, 27:1107:120].sum(0) == 0)
    retired_id = int(non_egos[-1])
    d = [511] + [511 ^ (1 << retired_id)]*(len(raw)-1)
    p = [511]*len(raw)
    gen = torch.Generator().manual_seed(1000 + ordinal)
    decisions = [model.act(x, dispatchable, present, gen)
                 for x,dispatchable,present in zip(raw[:-1], d[:-1], p[:-1])]
    return dict(states=raw, actions=np.asarray([a for a,_,_ in decisions]),
                old_log_probs=np.asarray([v for _,v,_ in decisions]),
                dispatchable_masks=d, in_system_masks=p, wait_deltas=historical['wait_deltas'],
                evaluation=historical['evaluation'], window_closures=closures,
                policy_version=model.policy_version,
                identity=dict(run_index=ordinal, recording_id=f'synthetic-{ordinal}',
                              environment_seed=ordinal, actor_sampling_seed=1000+ordinal),
                rpc_config={'config.target_ore_tons_shift': 12800., 'config.target_waste_tons_shift': 3200.,
                            'config.mass_balance_epsilon': 1.e-5})


def test_inbound_retiring_truck_is_physical_until_exit():
    torch.set_num_threads(2)
    raw = business_state()
    model = FleetLearner(19)
    mask = 511 ^ (1 << 3)
    x = torch.from_numpy(pack_states(raw[None], [mask], [511]))
    _, _, _, _, graph = scene(model.actor.state_encoder, model.actor.status_projection, x)
    from business_graph_summary_v1.graph import QUANTITY_NAMES
    assert float(graph.raw_quantities[0,13,QUANTITY_NAMES.index('inbound_loaded_tons')]) == pytest.approx(140.)
    with pytest.raises(ValueError, match='active physical fields'):
        pack_states(raw[None], [mask], [mask])
    parked = raw.copy()
    row = parked[27:1107].reshape(9, 120)[3]
    row[4:16] = 0
    row[15] = 1
    row[17:41] = 0
    row[43:] = 0
    x = torch.from_numpy(pack_states(parked[None], [mask], [mask]))
    _, _, h, _, graph = scene(model.actor.state_encoder, model.actor.status_projection, x)
    assert not (graph.edge_index == 3).any()
    assert torch.count_nonzero(h[:, 3]) == 0
    assert float(graph.raw_quantities[0,13,QUANTITY_NAMES.index('inbound_loaded_tons')]) == 0.


def test_scaled_targets_and_full_ppo_step(tmp_path):
    torch.set_num_threads(2)
    model = FleetLearner(19)
    group = [episode(model, i) for i in range(32)]
    e = group[0]
    assert shift_targets(e).production_costs[-1] == pytest.approx(8*(1-40/12800)+8)
    forged = dict(e, rpc_config={**e['rpc_config'], 'config.target_ore_tons_shift':16000.})
    with pytest.raises(ValueError, match='START goals'):
        shift_targets(forged)
    old = model.policy_version
    summary = model.update(group, batch_id='synthetic-test-32')
    assert model.policy_version != old
    assert summary['optimizer_steps'] > 0 and summary['update_index'] == 1
    assert torch.count_nonzero(model.actor.status_projection.weight)
    assert torch.count_nonzero(model.critic.status_projection.weight)
    path = tmp_path/'checkpoint.pt'
    model.save(path)
    restored = FleetLearner.load(path)
    assert restored.policy_version == model.policy_version
    assert models.digest(restored.critic) == models.digest(model.critic)
    assert restored.update_index == 1
    assert torch.equal(restored.generator.get_state(), model.generator.get_state())
    with pytest.raises(ValueError, match='fresh batch'):
        restored.update(group, batch_id='synthetic-test-32')
    branch = restored.fork(137)
    assert branch.policy_version == model.policy_version
    assert branch.update_index == 0 and not branch.actor_optimizer.state


@torch.no_grad()
def test_permutation_preserves_topology_actions_and_retirement():
    torch.set_num_threads(2)
    model = FleetLearner.warm_start(109)
    raw = business_state()
    # Truck 3 is loaded/retiring. Empty slot 8 was never part of this roster.
    opening = 511 ^ (1 << 8)
    eligible = opening ^ (1 << 3)
    truck = raw[27:1107].reshape(9, 120)[8]
    truck[4:16] = 0
    truck[15] = 1
    truck[17:41] = 0
    truck[43:] = 0
    order = episode_node_order(17265, opening)
    assert order == episode_node_order(17265, opening)
    assert order != episode_node_order(17266, opening)
    assert order[8] == 8 and order[:8] != tuple(range(8)) and order[9:13] != tuple(range(9,13))
    original = torch.from_numpy(pack_states(raw[None], [eligible], [opening]))
    moved = torch.from_numpy(pack_states(raw[None], [eligible], [opening], node_order=order))
    def frame(x):
        return scene(model.actor.state_encoder, model.actor.status_projection, x)
    context, ego, h, quantity, graph = frame(original)
    other_context, other_ego, other_h, other_quantity, other_graph = frame(moved)
    physical_id = torch.tensor(order)
    assert torch.equal(physical_id[other_graph.edge_index], graph.edge_index)
    assert torch.equal(physical_id[other_graph.ego_ids], graph.ego_ids)
    assert torch.equal(other_graph.edge_attr, graph.edge_attr)
    assert torch.equal(other_graph.edge_type, graph.edge_type)
    assert torch.equal(other_quantity, quantity[:, order])
    assert torch.equal(other_graph.candidate_dump_ids, graph.candidate_dump_ids[:, physical_id[9:13]-9])
    torch.testing.assert_close(other_h, h[:, order], atol=1.e-6, rtol=0.)
    torch.testing.assert_close(other_context, context, atol=1.e-6, rtol=0.)
    torch.testing.assert_close(other_ego, ego, atol=1.e-6, rtol=0.)
    features = SummaryFeatureCache(raw[None], feature_config=model.actor.feature_config).prefix()
    torch.testing.assert_close(model.actor(original, features), model.actor(moved, features), atol=1.e-6, rtol=0.)
    torch.testing.assert_close(model.critic(original, features), model.critic(moved, features), atol=1.e-6, rtol=0.)
