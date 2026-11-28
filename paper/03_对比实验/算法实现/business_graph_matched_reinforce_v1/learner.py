"""Terminal-return Actor with the reference PPO Actor's exact per-batch step budget."""
from collections import defaultdict
from copy import deepcopy
import math
from pathlib import Path

import numpy as np
import torch

from business_graph_reinforce_v1 import learner as terminal
from business_graph_summary_v1 import learner as bg

SETTINGS_SCHEMA = 'bg128-terminal-advantage-ppo-actor-step-budget-v1'
CHECKPOINT_SCHEMA = 'bg128-terminal-advantage-ppo-actor-step-budget-checkpoint-v1'


def _implementation():
    return dict(source=bg._file_sha(Path(__file__)), terminal=terminal._implementation())


def _batches(permutation, nominal_size, count):
    """Keep PPO's original partition whenever it gives the prescribed count."""
    if math.ceil(len(permutation) / nominal_size) == count:
        return permutation.split(nominal_size)
    return torch.tensor_split(permutation, count)


class MatchedReinforce(terminal.FullShiftReinforce):
    def __init__(self, actor=None, *, actor_steps_by_update, actor_config=None, actor_lr=3e-4,
                 baseline_decay=.5, episodes_per_batch=None, adam_eps=1e-5):
        super().__init__(actor, actor_config=actor_config, actor_lr=actor_lr, baseline_decay=baseline_decay,
                         episodes_per_batch=episodes_per_batch, adam_eps=adam_eps)
        p = self.config['ppo']
        terminal._require(isinstance(actor_steps_by_update, (list, tuple)) and actor_steps_by_update and
                          all(type(v) is int and v > 0 and v % p['epochs'] == 0 for v in actor_steps_by_update),
                          'positive whole-epoch Actor step schedule required')
        terminal._require(p['actor_lr'] == actor_lr and p['adam_eps'] == adam_eps,
                          'Actor Adam must match the reference PPO config')
        self.settings.update(schema=SETTINGS_SCHEMA, actor_steps_by_update=list(actor_steps_by_update))
        self.settings_sha256 = bg._hash(self.settings)
        self.generator = torch.Generator().manual_seed(self.config['seed'])

    def _loss_and_ratio(self, episodes, mapping, ids, actions, old_logs, advantages):
        # Intercept the existing PPO Actor forward outputs only for detached diagnostics.
        # Its gradient, clipping objective and weighting execute in the original function.
        grouped = defaultdict(list)
        for idx in ids.tolist():
            grouped[mapping[idx][0]].append(idx)
        groups = iter(grouped.values())
        max_abs, outside = 0., 0

        def record(module, inputs, logits):
            nonlocal max_abs, outside
            take = next(groups)
            with torch.no_grad():
                policy = torch.distributions.Categorical(logits=logits.detach())
                delta = policy.log_prob(actions[take]) - old_logs[take]
                max_abs = max(max_abs, float(delta.abs().max()))
                outside += int(((delta.exp() - 1).abs() > self.config['ppo']['clip_range']).sum())

        hook = self.actor.register_forward_hook(record)
        try:
            loss = bg.JointCostPPO._actor_grad(self, episodes, mapping, ids, actions, old_logs, advantages)
        finally:
            hook.remove()
        terminal._require(next(groups, None) is None, 'PPO minibatch did not evaluate every shift group')
        return loss, max_abs, outside

    def _update(self, episodes, *, batch_id):
        schedule = self.settings['actor_steps_by_update']
        index = self.update_index
        terminal._require(index < len(schedule), 'matched Actor step schedule is exhausted')
        episodes = bg.JointCostPPO._validate_episodes(self, episodes, training=True)
        keys = [bg._episode_key(episode) for episode in episodes]
        batch_id = bg._hash(keys) if batch_id is None else batch_id
        terminal._require(type(batch_id) is str and batch_id and batch_id not in self.consumed_batches and
                          len({e['identity']['run_index'] for e in episodes}) == len(episodes),
                          'invalid, duplicate or consumed matched update')
        actions = torch.tensor(np.concatenate([ep['actions'] for ep in episodes]), dtype=torch.int64)
        logs = torch.tensor(np.concatenate([ep['old_log_probs'] for ep in episodes]), dtype=torch.float32)
        returns = [-math.fsum(ep['targets'].step_costs) for ep in episodes]
        mean_return = math.fsum(returns) / len(returns)
        baseline = self.baseline
        # JointCostPPO minimizes its cost surrogate, so cost advantage is -(R-baseline).
        cost_adv = torch.repeat_interleave(torch.tensor([baseline-r for r in returns], dtype=torch.float32),
                                           torch.tensor([len(ep['actions']) for ep in episodes]))
        mapping = [(ei, local) for ei, ep in enumerate(episodes) for local in range(len(ep['actions']))]
        p = self.config['ppo']
        count = schedule[index] // p['epochs']
        n = len(mapping)
        terminal._require(count <= n and torch.isfinite(cost_adv).all().item(),
                          'impossible nonempty minibatch partition or nonfinite terminal return')
        start_digest = self.policy_version
        steps, clipped_rows = 0, 0
        first_ratio, later_ratio = 0., 0.
        losses, norms, sizes = [], [], []
        # Any exception after this point requires the last sealed checkpoint.
        self._failed_update = True
        for _ in range(p['epochs']):
            perm = torch.randperm(n, generator=self.generator)
            for ids in _batches(perm, p['minibatch_size'], count):
                self.actor_optimizer.zero_grad(set_to_none=True)
                loss, log_ratio, clipped = self._loss_and_ratio(episodes, mapping, ids, actions, logs, cost_adv)
                norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), p['max_grad_norm'], error_if_nonfinite=True)
                self.actor_optimizer.step()
                if steps == 0:
                    first_ratio = log_ratio
                else:
                    later_ratio = max(later_ratio, log_ratio)
                steps += 1
                clipped_rows += clipped
                sizes.append(len(ids))
                losses.append(loss)
                norms.append(float(norm))
        terminal._require(steps == schedule[index] and sum(sizes) == n*p['epochs'] and
                          bg._finite_tree(self.actor.state_dict()) and bg._finite_tree(self.actor_optimizer.state_dict()),
                          'matched epoch/step count or final optimizer state differs')
        decay = self.settings['baseline_decay']
        after = decay*baseline + (1-decay)*mean_return
        cumulative = sum(schedule[:index+1])
        receipt = dict(batch_id=batch_id, update_index=index+1, episode_keys=keys, episodes=len(episodes),
                       transitions=n, epochs=p['epochs'], actor_optimizer_steps=steps,
                       cumulative_actor_optimizer_steps=cumulative, minibatch_evaluations=steps,
                       optimizer_decision_visits=sum(sizes), min_minibatch_size=min(sizes), max_minibatch_size=max(sizes),
                       baseline_before=baseline, baseline_after=after, mean_return=mean_return,
                       mean_combined_cost=-mean_return, policy_loss=math.fsum(losses)/steps,
                       actor_grad_norm=math.fsum(norms)/steps, clip_fraction=clipped_rows/sum(sizes),
                       first_step_max_abs_log_ratio=first_ratio, post_first_step_max_abs_log_ratio=later_ratio,
                       max_abs_log_ratio=max(first_ratio, later_ratio), rollout_actor_digest=start_digest,
                       actor_digest=self.policy_version, settings_sha256=self.settings_sha256)
        self.update_index = index + 1
        self.baseline = after
        self.consumed_batches[batch_id] = receipt
        self.consumed_episodes.update(dict.fromkeys(keys, batch_id))
        self._actor_artifact = None
        self._failed_update = False
        return deepcopy(receipt)

    def save_checkpoint(self, path):
        self._check_settings()
        terminal._require(not self._failed_update and bg._finite_tree(self.actor.state_dict()) and
                          bg._finite_tree(self.actor_optimizer.state_dict()), 'incomplete Actor step transaction')
        bg._canonical(self.runtime_state)
        metadata = dict(schema=CHECKPOINT_SCHEMA, actor_config_sha256=self.config_sha256,
                        settings_sha256=self.settings_sha256, actor_digest=self.policy_version, update_index=self.update_index)
        value = dict(**metadata, actor_config=self.config, settings=self.settings, actor=self.actor.state_dict(),
                     optimizer=self.actor_optimizer.state_dict(), baseline=self.baseline, generator=self.generator.get_state(),
                     consumed_batches=self.consumed_batches, consumed_episodes=self.consumed_episodes,
                     runtime_state=self.runtime_state, implementation=_implementation())
        return bg._atomic_torch(path, value, metadata=metadata)

    @classmethod
    def from_checkpoint(cls, path, *, expected_actor_config=None, expected_sha256=None, expected_settings=None):
        if expected_sha256 is not None:
            terminal._require(bg._file_sha(path) == expected_sha256, 'checkpoint expected byte SHA differs')
        saved, manifest = bg._load_checkpoint(path)
        terminal._require(saved['schema'] == manifest['schema'] == CHECKPOINT_SCHEMA and
                          saved['implementation'] == _implementation(), 'matched checkpoint schema or source differs')
        kwargs = dict(saved['settings'])
        terminal._require(kwargs.pop('schema') == SETTINGS_SCHEMA, 'invalid matched settings schema')
        model = cls(actor_config=saved['actor_config'], **kwargs)
        for key, expected in [('actor_config_sha256', model.config_sha256), ('settings_sha256', model.settings_sha256)]:
            terminal._require(manifest.get(key) == saved[key] == expected, 'checkpoint configuration digest differs')
        terminal._require((expected_actor_config is None or model.config == bg.validate_config(expected_actor_config)) and
                          (expected_settings is None or model.settings == expected_settings), 'checkpoint settings differ from campaign')
        bg._validate_model_state(saved['actor'], model.actor)
        bg._validate_optimizer(saved['optimizer'], model.actor_optimizer.state_dict(), list(model.actor.parameters()))
        model.actor.load_state_dict(saved['actor'])
        model.actor_optimizer.load_state_dict(saved['optimizer'])
        terminal._require(model.policy_version == saved['actor_digest'] == manifest['actor_digest'], 'Actor fingerprint differs')
        model._restore_history(saved, manifest)
        return model

    def _restore_history(self, saved, manifest):
        batches = saved['consumed_batches']
        idx = saved['update_index']
        schedule = self.settings['actor_steps_by_update']
        terminal._require(type(idx) is int and 0 <= idx <= len(schedule) and idx == len(batches) == manifest['update_index'],
                          'matched update inventory differs')
        decay, before, episodes, prior, cumulative = self.settings['baseline_decay'], 0., {}, None, 0
        expected_rng = torch.Generator().manual_seed(self.config['seed'])
        for index, (batch_id, rec) in enumerate(batches.items()):
            keys = rec['episode_keys']
            cumulative += schedule[index]
            after = decay*before + (1-decay)*rec['mean_return']
            terminal._require(rec['update_index'] == index+1 and rec['batch_id'] == batch_id and
                              rec['actor_optimizer_steps'] == schedule[index] and
                              rec['cumulative_actor_optimizer_steps'] == cumulative and rec['epochs'] == self.config['ppo']['epochs'] and
                              rec['baseline_before'] == before and rec['baseline_after'] == after and
                              rec['settings_sha256'] == self.settings_sha256 and bg._finite_tree(rec) and
                              len(keys) == len(set(keys)) == self.settings['episodes_per_batch'] and
                              episodes.keys().isdisjoint(keys) and (prior is None or prior == rec['rollout_actor_digest']),
                              'matched checkpoint consumption history differs')
            for _ in range(rec['epochs']):
                torch.randperm(rec['transitions'], generator=expected_rng)
            before, prior = after, rec['actor_digest']
            episodes.update(dict.fromkeys(keys, batch_id))
        state = self.actor_optimizer.state
        terminal._require((idx == 0 and not state) or (len(state) > 0 and
                          all(0 < st['step'].item() <= cumulative for st in state.values()) and
                          max(st['step'].item() for st in state.values()) == cumulative), 'Adam step history differs')
        terminal._require(episodes == saved['consumed_episodes'] and before == saved['baseline'] and
                          (prior is None or prior == self.policy_version) and
                          torch.equal(expected_rng.get_state(), saved['generator']), 'checkpoint returns, Actor or shuffle RNG differ')
        self.generator.set_state(saved['generator'])
        self.consumed_batches, self.consumed_episodes = batches, episodes
        self.baseline, self.update_index = before, idx
        bg._canonical(saved['runtime_state'])
        self.runtime_state = saved['runtime_state']
