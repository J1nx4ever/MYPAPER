"""Nine-slot BG128 with causal fleet state and the parent's cost-PPO estimators."""
import math
import sys
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from random import Random

import numpy as np
import torch
from torch import nn

PLAN = Path(__file__).resolve().parent
PROJECT = PLAN.parents[3]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from business_graph_summary_v1 import learner as bg
from business_graph_summary_v1 import models
from business_graph_summary_v1.graph import build_graph
from single_critic_task_query_v1.features_cache import validate_sequence, SummaryFeatureCache
from single_critic_task_query_v1.targets import CompositeTargets, grade_cost

PARENT = PROJECT / 'runs/bg128-continuous-mixed-20261120-s02/arms/s01_BG128/updates/0032/checkpoint.pt'
PARENT_SHA = '67b7b2c7104079836ccef8f5747e7aebb4e9e5a92dcaac67c94e42412f104f41'
SCHEMA = 'bg128-fleet-resilience-v2'
RAW = 1519
NODES = 19


def require(condition, message):
    if not condition:
        raise ValueError(message)


@lru_cache(maxsize=128)
def episode_node_order(actor_seed, opening_mask):
    """Return physical node IDs in their graph-slot order, fixed through END."""
    require(0 <= actor_seed < 2**64 and 0 <= opening_mask <= 511 and
            opening_mask.bit_count() >= 6, 'invalid node-permutation seed or opening roster')
    rng = Random(f'bg128-fleet-node-order-v1/{actor_seed}/{opening_mask}')
    active = [i for i in range(9) if opening_mask & (1 << i)]
    order = list(range(NODES))
    for slot, truck in zip(active, rng.sample(active, len(active))):
        order[slot] = truck
    order[9:13] = rng.sample(range(9, 13), 4)
    return tuple(order)


def pack_states(states, dispatchable_masks, in_system_masks, *, node_order=None):
    raw = np.asarray(states)
    require(raw.ndim == 2 and raw.shape[1] == RAW and raw.dtype == np.float32 and np.isfinite(raw).all(),
            'expected finite physical float32 observations')
    channels = []
    for values in (dispatchable_masks, in_system_masks):
        mask = np.asarray(values)
        require(mask.shape == (len(raw),) and mask.dtype.kind in 'ui' and
                ((0 <= mask) & (mask <= 511)).all(), 'expected one nine-bit integer per observation')
        channels.append((mask.astype(np.int64)[:, None] >> np.arange(9)) & 1)
    eligible, present = channels
    require((eligible <= present).all() and (eligible.sum(1) >= 6).all(), 'invalid fleet states')
    trucks = raw[:, 27:1107].reshape(-1, 9, 120)
    require(not np.any(trucks[:, :, 0] * (1 - eligible)), 'ineligible decision vehicle')
    parked = trucks[present == 0]
    require(not (parked[:, 0].any() or parked[:, 4:15].any() or parked[:, 17:41].any() or
                 parked[:, 43:].any()) and (parked[:, 15] == 1).all(), 'exited truck has active physical fields')
    loads = raw[:, 1107:1295].reshape(-1, 4, 47)
    dumps = raw[:, 1295:1391].reshape(-1, 2, 48)
    refs = np.concatenate((trucks[:, :, 81], loads[:, :, 20:].reshape(len(raw), -1),
                           dumps[:, :, 11:].reshape(len(raw), -1)), axis=1)
    ids = np.rint(9 * refs).astype(np.int64) - 1
    require(((ids >= -1) & (ids < 9)).all(), 'invalid truck reference')
    require(not np.any((ids >= 0) & (np.take_along_axis(present, ids.clip(0, 8), 1) == 0)),
            'exited truck still referenced by scene')
    order = np.asarray(tuple(range(NODES)) if node_order is None else node_order)
    require(order.shape == (NODES,) and order.dtype.kind in 'iu' and
            np.array_equal(np.sort(order[:9]), np.arange(9)) and
            np.array_equal(np.sort(order[9:13]), np.arange(9, 13)) and
            np.array_equal(order[13:], np.arange(13, NODES)), 'invalid typed node permutation')
    order = np.broadcast_to(order, (len(raw), NODES))
    return np.concatenate((raw, eligible, present, order), axis=1).astype(np.float32)


def scene(encoder, status_projection, packed):
    raw = packed[:, :RAW].contiguous()
    eligible = packed[:, RAW:RAW+9]
    present = packed[:, RAW+9:RAW+18]
    order = packed[:, RAW+18:].long()
    rows = torch.arange(len(raw))
    inverse = order.argsort(1)
    status = torch.stack((eligible-1, present-1), dim=-1)
    graph = build_graph(raw)
    tokens = encoder.token_encoder(raw, graph)
    # Centering removes the new input exactly for an all-available fleet. The
    # inherited affine weights keep their original shape and GEMM operation.
    trucks = (tokens[:, :9] + status_projection(status)) * present.unsqueeze(-1)
    global_token = tokens[:, 18:] + status_projection(status.mean(1, keepdim=True))
    h = torch.cat((trucks, tokens[:, 9:18], global_token), dim=1)
    # Keep geography, edge types/attributes and edge traversal order unchanged.
    # A graph slot maps to a physical node via order; inverse maps every edge
    # endpoint, including sides of the batched graph, into that graph's slots.
    h = h[rows[:, None], order]
    mask = present.gather(1, order[:, :9]).unsqueeze(-1)
    sample, physical = graph.edge_index // NODES, graph.edge_index % NODES
    graph = replace(graph, edge_index=sample * NODES + inverse[sample, physical],
                    quantities=graph.quantities[rows[:, None], order],
                    candidate_dump_ids=graph.candidate_dump_ids.gather(1, order[:, 9:13]-9),
                    ego_ids=inverse[rows, graph.ego_ids])
    for layer in encoder.graph_layers:
        h = layer(h, graph)
        h = torch.cat((h[:, :9] * mask, h[:, 9:]), dim=1)
    # pack_states rejects edges/commitments to absent vehicles before building
    # the graph. Their residual self features cannot enter the pooled vector.
    # Restore physical reduction order solely for float32 reproducibility.
    physical_h = h[rows[:, None], inverse]
    active_mean = (physical_h[:, :9] * present.unsqueeze(-1)).sum(1) / present.sum(1, keepdim=True)
    truck_pool = torch.where(present.bool().all(1, keepdim=True), physical_h[:, :9].mean(1), active_mean)
    pools = torch.stack([truck_pool] + [physical_h[:, s].mean(1) for s in models.TYPE_SLICES[1:]], 1)
    ego = h[rows, graph.ego_ids]
    quantities = graph.quantities
    physical_quantities = quantities[rows[:, None], inverse]
    counts = torch.stack([physical_quantities[:, s].sum(1) for s in models.TYPE_SLICES], 1)
    context = torch.cat((pools.flatten(1), ego, h[:, 18], counts.flatten(1)), dim=-1)
    return context, ego, h, quantities, graph


class FleetActor(models.Actor):
    def forward(self, packed, prefixes):
        raw = packed[:, :RAW].contiguous()
        current = validate_sequence(prefixes, raw, self.feature_config)
        context, ego, h, quantities, graph = scene(self.state_encoder, self.status_projection, packed)
        task = self.summary_task_encoder(current[:, :88])
        action = self.summary_action_encoder(current[:, 88:].reshape(-1, 4, 10))
        context = self.context_encoder(torch.cat((context, task), -1))
        rows = torch.arange(len(h))
        loads = packed[:, RAW+18+9:RAW+18+13].long()-9
        mask = raw[:, 19:23].gather(1, loads)
        dump_indices = graph.candidate_dump_ids
        candidate = torch.cat((ego[:, None].expand(-1, 4, -1), h[:, 9:13], h[rows[:, None], dump_indices],
                               quantities[rows, graph.ego_ids][:, None].expand(-1, 4, -1),
                               quantities[:, 9:13], quantities[rows[:, None], dump_indices],
                               mask.unsqueeze(-1), raw[:, 23:27].gather(1, loads).unsqueeze(-1),
                               action[rows[:, None], loads]), -1)
        scores = self.score_candidates(self.candidate_encoder(candidate), context)
        return scores.masked_fill(~mask.bool(), -torch.inf).gather(1, loads.argsort(1))


class FleetCritic(models.MaterialCostCritic):
    def forward(self, packed, prefixes):
        require(not bool(packed[:, 17].any()), 'critic queries require decisions. END bootstrap is zero')
        current = validate_sequence(prefixes, packed[:, :RAW].contiguous(), self.feature_config)
        context, *_ = scene(self.state_encoder, self.status_projection, packed)
        return self.fusion(torch.cat((context, self.summary_encoder(current)), -1)).squeeze(-1)


def _nonnegative(value, name):
    require(type(value) is not bool and math.isfinite(float(value)) and float(value) >= 0, name)
    return float(value)


def shift_targets(episode):
    n = len(episode['actions'])
    require(n > 0, 'empty shift')
    cfg, ev = episode['rpc_config'], episode['evaluation']
    ore_target = _nonnegative(cfg['config.target_ore_tons_shift'], 'invalid START ore target')
    waste_target = _nonnegative(cfg['config.target_waste_tons_shift'], 'invalid START waste target')
    require(min(ore_target, waste_target) > 0, 'positive shift targets required')
    ore = _nonnegative(ev['ore_tons'], 'invalid processed ore')
    waste = _nonnegative(ev['waste_tons'], 'invalid discharged waste')
    observed = np.asarray(episode['states'])
    require(np.allclose(observed[:, 11], np.float32(ore_target/20000.), rtol=0, atol=1e-6) and
            np.allclose(observed[:, 12], np.float32(waste_target/4000.), rtol=0, atol=1e-6),
            'observation goals disagree with frozen START goals')
    windows = episode['window_closures']
    require([w['window'] for w in windows] == list(range(4)), 'four ordered window closures required')
    steps = [w['step'] for w in windows]
    require(all(isinstance(i, (int, np.integer)) and 0 <= i < n for i in steps) and steps == sorted(steps),
            'window closure outside ordered decisions')
    costs = np.zeros(n)
    closures = np.zeros(n, dtype=int)
    for w in windows:
        q = _nonnegative(w['tons'], 'invalid window ore')
        metal = _nonnegative(w['metal_tons'], 'invalid window metal')
        require(metal <= q, 'window metal exceeds ore')
        costs[w['step']] += grade_cost(metal/q if q else None, empty=(q == 0))
        closures[w['step']] += 1
    require(math.isclose(math.fsum(w['tons'] for w in windows), ore, rel_tol=0,
                         abs_tol=cfg['config.mass_balance_epsilon']), 'processed ore disagrees with four windows')
    prod = np.zeros(n)
    prod[-1] = 8 * max(0., 1-ore/ore_target) + 8 * max(0., 1-waste/waste_target)
    wait = np.asarray(episode['wait_deltas'], dtype=np.float64)
    require(wait.shape == (n,) and np.isfinite(wait).all() and (wait >= 0).all() and
            math.isclose(math.fsum(wait), ev['total_wait_seconds'], rel_tol=1e-9, abs_tol=1e-6),
            'waiting increments disagree with END')
    wait = wait * .2 / 28800.
    steps_cost = costs + prod + wait
    return CompositeTargets(tuple(steps_cost), tuple(steps_cost[::-1].cumsum()[::-1]),
                            tuple(closures.cumsum()-closures), 0., tuple(costs), tuple(prod), tuple(wait))


class FleetLearner(bg.JointCostPPO):
    """Reuse BG128 GAE, bounded C128 cache, clipped cost PPO and MC value loss."""

    def __init__(self, seed):
        super().__init__(bg.default_config(seed=seed))
        args = dict(feature_config=self.config['feature_config'], pair_seed=seed)
        with torch.random.fork_rng(devices=[]):
            self.actor = FleetActor(**args)
            self.critic = FleetCritic(**args)
            for model in (self.actor, self.critic):
                model.status_projection = nn.Linear(2, 64, bias=False)
                nn.init.zeros_(model.status_projection.weight)
        ppo = self.config['ppo']
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=ppo['actor_lr'], eps=ppo['adam_eps'])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=ppo['critic_lr'], eps=ppo['adam_eps'])

    @classmethod
    def warm_start(cls, seed):
        require(bg._file_sha(PARENT) == PARENT_SHA, 'parent checkpoint SHA differs')
        parent, manifest = bg._load_checkpoint(PARENT)
        require(parent['update_index'] == manifest['update_index'] == 32, 'parent update differs')
        learner = cls(seed)
        require(parent['config'] == {**learner.config, 'seed': parent['config']['seed']},
                'unexpected parent training or feature configuration')
        for name in ('actor', 'critic'):
            model = getattr(learner, name)
            migrated = dict(parent[name])
            migrated['status_projection.weight'] = torch.zeros_like(model.status_projection.weight)
            model.load_state_dict(migrated, strict=True)
        return learner

    def fork(self, seed):
        branch = type(self)(seed)
        branch.actor.load_state_dict(self.actor.state_dict())
        branch.critic.load_state_dict(self.critic.state_dict())
        return branch

    def _prefix(self, states, ids):
        return self._summary_cache.get(np.ascontiguousarray(states[:, :RAW])).prefix(ids)

    @torch.no_grad()
    def act(self, state, dispatchable_mask, in_system_mask, generator, *, opening_mask=511):
        raw = np.asarray(state)[None, :]
        require(in_system_mask & opening_mask == in_system_mask, 'observed truck outside opening roster')
        order = episode_node_order(generator.initial_seed(), opening_mask)
        packed = pack_states(raw, [dispatchable_mask], [in_system_mask], node_order=order)
        prefix = SummaryFeatureCache(raw, feature_config=self.actor.feature_config).prefix()
        inputs = torch.from_numpy(packed)
        dist = self.actor.distribution(inputs, prefix)
        chosen = torch.multinomial(dist.probs[0], 1, generator=generator)
        return int(chosen.item()), float(dist.log_prob(chosen).item()), float(self.critic(inputs, prefix).item())

    def _prepare_episode(self, episode):
        require(episode['policy_version'] == self.policy_version, 'stale on-policy episode')
        eligibility = np.asarray(episode['dispatchable_masks'])
        presence = np.asarray(episode['in_system_masks'])
        order = episode_node_order(episode['identity']['actor_sampling_seed'], int(eligibility[0]))
        states = pack_states(episode['states'], eligibility, presence, node_order=order)
        require(eligibility[0] == presence[0] and
                ((eligibility[1:] & eligibility[:-1]) == eligibility[1:]).all() and
                ((presence[1:] & presence[:-1]) == presence[1:]).all() and
                (int(eligibility[0]) ^ int(eligibility[-1])).bit_count() <= 1,
                'fleet lifecycle disagrees with one-way single-truck withdrawal')
        require(states[-1, 17] == 1 and not states[:-1, 17].any(), 'one true END required')
        n = len(states)-1
        actions, logs = np.asarray(episode['actions']), np.asarray(episode['old_log_probs'])
        require(actions.shape == logs.shape == (n,) and n > 0 and actions.dtype.kind in 'iu' and
                ((actions >= 0) & (actions < 4)).all() and np.isfinite(logs).all() and
                (states[np.arange(n), 19+actions] == 1).all(), 'inconsistent sampled action or probability')
        targets = shift_targets(episode)
        admitted = dict(episode, states=states, actions=actions, old_log_probs=logs, targets=targets)
        with torch.no_grad():
            replay, _ = self._policy_rows(admitted)
        if not np.allclose(replay.numpy(), logs, rtol=1e-6, atol=1e-6):
            raise ValueError('stored fleet masks/states fail action likelihood replay')
        return admitted

    def update(self, episodes, *, batch_id):
        p = self.config['ppo']
        require(not self._failed_update and isinstance(batch_id, str) and batch_id and
                batch_id not in self.consumed_batches and len(episodes) == p['episodes_per_batch'],
                'update requires a fresh batch of 32 completed shifts')
        self._summary_cache.clear()
        episodes = tuple(self._prepare_episode(e) for e in episodes)
        keys = [bg._episode_key(e) for e in episodes]
        runs = [e['identity']['run_index'] for e in episodes]
        require(len(set(keys)) == len(set(runs)) == len(episodes) and
                not set(keys).intersection(self.consumed_episodes), 'duplicate on-policy episode')
        t = self._freeze_targets(episodes)
        mapping = [(ei, i) for ei, e in enumerate(episodes) for i in range(len(e['actions']))]
        actions = torch.from_numpy(np.concatenate([e['actions'] for e in episodes]).astype(np.int64))
        logs = torch.tensor(np.concatenate([e['old_log_probs'] for e in episodes]), dtype=torch.float32)
        advantages = torch.tensor(t['actor_advantages'], dtype=torch.float32)
        mc = torch.tensor(t['mc_returns'], dtype=torch.float32)
        self._failed_update = True
        losses = []
        for _ in range(p['epochs']):
            for ids in torch.randperm(len(mapping), generator=self.generator).split(p['minibatch_size']):
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss = self._actor_grad(episodes, mapping, ids, actions, logs, advantages)
                actor_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), p['max_grad_norm'], error_if_nonfinite=True)
                self.actor_optimizer.step()
                self.critic_optimizer.zero_grad(set_to_none=True)
                critic_loss = self._critic_grad(episodes, mapping, ids, mc)
                critic_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), p['max_grad_norm'], error_if_nonfinite=True)
                self.critic_optimizer.step()
                losses.append((actor_loss, critic_loss, float(actor_norm), float(critic_norm)))
        metrics = self._metrics(episodes, old_logs=logs)
        require(all(torch.isfinite(v).all() for m in (self.actor, self.critic) for v in m.parameters()),
                'nonfinite parameters after update')
        self.update_index += 1
        self.consumed_batches[batch_id] = self.update_index
        self.consumed_episodes.update(dict.fromkeys(keys, batch_id))
        self._failed_update = False
        strata = {}
        for episode in episodes:
            initial = int(episode['dispatchable_masks'][0])
            last = int(episode['dispatchable_masks'][-1])
            outgoing = initial & ~last
            fault_type = ('none' if outgoing == 0 else
                          ('777', '785', '789')[(outgoing.bit_length()-1)//3])
            is_ore = episode['states'][0, 1107:1295].reshape(4, 47)[:, 1]
            waste_sites = np.flatnonzero(is_ore == 0)
            site_label = '-'.join(str(index) for index in waste_sites)
            label = f'{initial.bit_count()}_trucks/{fault_type}/waste_{site_label}'
            row = strata.setdefault(label, dict(shifts=0, decisions=0, grade_cost_sum=0.,
                                                production_cost_sum=0., waiting_cost_sum=0.))
            row['shifts'] += 1
            row['decisions'] += len(episode['actions'])
            for key, costs in (('grade_cost_sum', 'grade_costs'), ('production_cost_sum', 'production_costs'),
                               ('waiting_cost_sum', 'wait_costs')):
                row[key] += math.fsum(getattr(episode['targets'], costs))
        return dict(batch_id=batch_id, update_index=self.update_index, decisions=len(mapping), episodes=len(episodes),
                    optimizer_steps=len(losses), metrics=metrics, strata=strata,
                    mean=dict(zip(('actor_loss', 'critic_mc_mse', 'actor_grad_norm', 'critic_grad_norm'),
                                  np.mean(losses, axis=0).tolist())))

    def save(self, path):
        require(not self._failed_update, 'cannot save a partial optimizer update')
        data = dict(schema=SCHEMA, parent_sha256=PARENT_SHA, seed=self.config['seed'], update_index=self.update_index,
                    generator=self.generator.get_state(), consumed_batches=self.consumed_batches,
                    consumed_episodes=self.consumed_episodes)
        for name in ('actor', 'critic', 'actor_optimizer', 'critic_optimizer'):
            data[name] = getattr(self, name).state_dict()
        return bg._atomic_torch(path, data, metadata=dict(schema=SCHEMA, update_index=self.update_index,
                               actor_digest=self.policy_version, parent_sha256=PARENT_SHA))

    @classmethod
    def load(cls, path):
        data, manifest = bg._load_checkpoint(path)
        require(data['schema'] == manifest['schema'] == SCHEMA and
                data['parent_sha256'] == manifest['parent_sha256'] == PARENT_SHA, 'checkpoint lineage differs')
        result = cls(data['seed'])
        for name in ('actor', 'critic', 'actor_optimizer', 'critic_optimizer'):
            getattr(result, name).load_state_dict(data[name])
        result.update_index = data['update_index']
        result.consumed_batches = data['consumed_batches']
        result.consumed_episodes = data['consumed_episodes']
        result.generator.set_state(data['generator'])
        require(manifest['actor_digest'] == result.policy_version and
                len(result.consumed_batches) == result.update_index == manifest['update_index'], 'restore differs')
        return result
