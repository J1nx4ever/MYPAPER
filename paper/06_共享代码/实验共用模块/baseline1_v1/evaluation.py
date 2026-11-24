"""Single-Critic read-only scoring of already collected complete shifts.

This module starts no collector, RPC service or simulator. The caller owns
authorization, physical budget/receipts, checkpoint pinning and held-out seeds.
Only compact per-shift physical/score statistics leave this evaluator.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from contextlib import AbstractContextManager

import numpy as np
import torch

from python.constrained_ppo import digest
from .reward import CONTRACT, SingleCriticTrainingBatch, training_batch, validate_training_batch

FORMAT = 'baseline1-independent-evaluation-v1'


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _tree_hash(value):
    result = hashlib.sha256()

    def add(item):
        if isinstance(item, torch.Tensor):
            item = item.detach().cpu().contiguous()
            result.update(str((str(item.dtype), tuple(item.shape))).encode())
            result.update(item.numpy().tobytes())
        elif isinstance(item, np.ndarray):
            result.update(str((str(item.dtype), item.shape, item.flags.writeable)).encode())
            result.update(item.tobytes())
        elif isinstance(item, torch.Generator):
            add(item.get_state())
        elif isinstance(item, np.random.Generator):
            add(item.bit_generator.state)
        elif isinstance(item, random.Random):
            add(item.getstate())
        elif isinstance(item, dict):
            for key in sorted(item, key=repr):
                add(key)
                add(item[key])
        elif isinstance(item, (list, tuple)):
            result.update(type(item).__name__.encode())
            for part in item:
                add(part)
        elif isinstance(item, set):
            add(sorted(item, key=repr))
        else:
            result.update((type(item).__name__ + ':' + repr(item)).encode())

    add(value)
    return result.hexdigest()


def _module_state(module):
    return dict(parameters=module.state_dict(),
                modes=[(name, submodule.training) for name, submodule in module.named_modules()],
                flags=[(name, parameter.requires_grad, parameter.grad)
                       for name, parameter in module.named_parameters()])


def training_state_fingerprint(model):
    """Cover actual single-value learner, Adam moments, counters and RNGs."""
    if model is None:
        return None
    _require(hasattr(model, 'actor') and hasattr(model, 'critic'), 'single-Critic model required')
    _require(not hasattr(model, 'critics') and not hasattr(model, 'lambdas'),
             'dual-Critic or multiplier model cannot enter Baseline 1 evaluation')
    state = dict(actor=_module_state(model.actor), critic=_module_state(model.critic))
    for name in ('actor_optimizer', 'critic_optimizer'):
        optimizer = getattr(model, name)
        state[name] = optimizer.state_dict()
        state[name + '_parameter_references'] = [[id(p) for p in group['params']]
                                                for group in optimizer.param_groups]
    # Include new identity/consumption/failed-update fields automatically instead
    # of silently guarding an older subset as the scalar learner evolves.
    handled = {'actor', 'critic', 'actor_optimizer', 'critic_optimizer'}
    for name, value in vars(model).items():
        if name not in handled and not callable(value):
            state[name] = value
    return _tree_hash(state)


def environment_fingerprint(environment):
    if environment is None:
        return None
    if isinstance(environment, dict):
        return _tree_hash(environment)
    if hasattr(environment, 'evaluation_snapshot'):
        return _tree_hash(environment.evaluation_snapshot())
    state = {}
    for name in ('_pending', '_episode_id', '_last_observation', 'episodes_started',
                 'total_episodes', '_np_random', '_np_random_seed'):
        if hasattr(environment, name):
            state[name] = getattr(environment, name)
    for name in ('action_space', 'observation_space'):
        space = getattr(environment, name, None)
        if space is not None:
            state[name] = {key: value for key, value in vars(space).items()
                           if key.startswith('_np_random')}
    if hasattr(environment, 'runner'):
        state['runner'] = environment_fingerprint(environment.runner)
    bridge = getattr(environment, 'bridge', None)
    if bridge is not None:
        fields = ('_expected_episode_metadata', '_started_episode_count',
                  '_last_sequence_by_episode', '_next_window_index_by_episode',
                  '_seen_experiment_run_indices', '_seen_experiment_seeds',
                  '_metadata_by_episode', '_request_bytes', '_closed', '_fatal_error')
        state['bridge'] = {name: getattr(bridge, name) for name in fields if hasattr(bridge, name)}
        queue = getattr(bridge, '_incoming', None)
        if queue is not None:
            with queue.mutex:
                state['incoming'] = [(id(ticket), getattr(getattr(ticket, 'request', None), 'request_id', None))
                                     for ticket in queue.queue]
    _require(bool(state), 'training environment needs a known lifecycle or evaluation_snapshot')
    return _tree_hash(state)


class ReadOnlyTrainingGuard(AbstractContextManager):
    """Restore global streams on every exit; reject mutation, never undo it silently."""
    def __init__(self, model=None, environment=None):
        self.model, self.environment = model, environment

    def __enter__(self):
        pending = (self.environment.get('_pending', self.environment.get('pending'))
                   if isinstance(self.environment, dict) else getattr(self.environment, '_pending', None))
        _require(pending is None, 'evaluation cannot abandon a pending training callback')
        self.torch_rng = torch.get_rng_state().clone()
        self.numpy_rng = copy.deepcopy(np.random.get_state())
        self.python_rng = random.getstate()
        self.cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        self.threads = torch.get_num_threads()
        self.model_before = training_state_fingerprint(self.model)
        self.environment_before = environment_fingerprint(self.environment)
        return self

    def __exit__(self, kind, value, traceback):
        try:
            unchanged = (training_state_fingerprint(self.model) == self.model_before and
                         environment_fingerprint(self.environment) == self.environment_before)
        finally:
            torch.set_rng_state(self.torch_rng)
            np.random.set_state(self.numpy_rng)
            random.setstate(self.python_rng)
            if self.cuda_rng is not None:
                torch.cuda.set_rng_state_all(self.cuda_rng)
            torch.set_num_threads(self.threads)
        _require(unchanged, 'evaluation changed protected training or environment state')
        return False


def mc_returns(rewards, bounds):
    """Undiscounted complete-shift return; reset carry at every real terminal."""
    rewards = np.asarray(rewards, dtype=np.float64)
    _require(rewards.ndim == 1 and np.isfinite(rewards).all(), 'invalid evaluation rewards')
    result = np.empty_like(rewards)
    cursor = 0
    for start, stop in bounds:
        _require(start == cursor and start < stop <= len(rewards), 'MC episode boundary gap/overlap')
        carry = 0.
        for index in range(stop - 1, start - 1, -1):
            carry += float(rewards[index])
            result[index] = carry
        cursor = stop
    _require(cursor == len(rewards) and cursor > 0 and np.isfinite(result).all(),
             'incomplete or nonfinite evaluation MC')
    return result


def _score_stats(targets, predictions):
    residual = targets - predictions
    return dict(n_states=len(targets), target_sum=float(np.sum(targets)),
                target_squared_sum=float(targets @ targets),
                error_sum=float(np.sum(residual)), error_squared_sum=float(residual @ residual))


def _regression(records, *, shift_equal):
    mass = len(records) if shift_equal else sum(row['n_states'] for row in records)
    mean = lambda key: math.fsum(row[key] / row['n_states'] if shift_equal else row[key]
                                for row in records) / mass
    target_mean, error_mean = mean('target_sum'), mean('error_sum')
    target_var = max(0., mean('target_squared_sum') - target_mean ** 2)
    error_var = max(0., mean('error_squared_sum') - error_mean ** 2)
    # Cancellation of constant targets may leave insignificant positive roundoff.
    tolerance = np.finfo(np.float64).eps * max(1., abs(mean('target_squared_sum'))) * 16
    constant = target_var <= tolerance
    return dict(n_shifts=len(records), n_states=sum(row['n_states'] for row in records),
                weighting='complete_shift_equal' if shift_equal else 'state_equal',
                mse=mean('error_squared_sum'), bias=-error_mean,
                explained_variance=None if constant else 1. - error_var / target_var,
                explained_variance_reason='constant_target' if constant else None,
                target_variance=target_var, residual_variance=error_var)


def _distribution(values):
    value = np.asarray(values, dtype=np.float64)
    _require(value.ndim == 1 and len(value) > 0 and np.isfinite(value).all(), 'invalid metric distribution')
    return dict(n=len(value), mean=float(value.mean()),
                sd=float(value.std(ddof=1)) if len(value) > 1 else None,
                min=float(value.min()), p50=float(np.quantile(value, .5)),
                p95=float(np.quantile(value, .95)), max=float(value.max()))


def summarize_episodes(episodes):
    """Aggregate disjoint same-policy score summaries, without observation snapshots."""
    _require(bool(episodes), 'empty independent evaluation')
    for key in ('episode_id', 'experiment_seed', 'experiment_run_index'):
        _require(len({row[key] for row in episodes}) == len(episodes), 'duplicate evaluated ' + key)
    _require(len({row['policy_version'] for row in episodes}) == 1, 'mixed evaluated Actors')
    _require(len({row['beta'] for row in episodes}) == 1, 'mixed evaluation beta')
    _require(len({row['critic_version'] for row in episodes}) == 1, 'mixed evaluated Critics')
    _require(len({(row['checkpoint_update'], row['cumulative_training_shifts']) for row in episodes}) == 1,
             'mixed evaluation checkpoint counters')
    physical = [row['production'] for row in episodes]
    costs = np.asarray([row['window_costs'] for row in physical], dtype=np.int64)
    _require(costs.shape == (len(episodes), 4) and np.isin(costs, [0, 1]).all(), 'invalid four-window costs')
    production = dict(n_shifts=len(episodes), window_violation_counts=costs.sum(axis=0).tolist(),
        window_violation_rates=costs.mean(axis=0).tolist(),
        ore_target_met_rate=float(np.mean([row['ore_target_met'] for row in physical])),
        waste_target_met_rate=float(np.mean([row['waste_target_met'] for row in physical])),
        both_targets_met_rate=float(np.mean([row['both_targets_met'] for row in physical])),
        joint_four_window_compliance_rate=float(np.mean([row['all_four_windows_compliant'] for row in physical])),
        overall_success_rate=float(np.mean([row['overall_success'] for row in physical])),
        empty_window_counts=np.asarray([row['empty_windows'] for row in physical], dtype=int).sum(axis=0).tolist(),
        vehicle_utilization=None, vehicle_utilization_reason='physical_busy_time_summary_not_available',
        queue_statistics=None, queue_statistics_reason='physical_time_weighted_queue_summary_not_available')
    for key in ('Q_ore', 'Q_waste', 'W_vehicle_seconds', 'mean_vehicle_wait_seconds'):
        production[key] = _distribution([row[key] for row in physical])
    production['epsilon_compatibility_disagreement_shifts'] = sum(
        row['ore_target_met'] != row['epsilon_ore_target_met'] or
        row['waste_target_met'] != row['epsilon_waste_target_met'] for row in physical)
    ledgers = {key: _distribution([row['reward_ledgers'][key] for row in episodes])
               for key in ('wait_reward', 'grade_cost', 'grade_penalty', 'production_cost',
                           'production_reward', 'total_reward', 'q_o', 'q_w', 'ledger_residual')}
    stats = [row['value_statistics'] for row in episodes]
    return dict(reward=dict(micro=_regression(stats, shift_equal=False),
                            shift_macro=_regression(stats, shift_equal=True)),
                production=production, reward_ledgers=ledgers,
                policy_entropy=dict(shift_macro=_distribution([row['policy_entropy_mean'] for row in episodes]),
                    state_mean=math.fsum(row['policy_entropy_sum'] for row in episodes) /
                               sum(row['n_states'] for row in episodes)),
                critic_mc_loss=dict(kind='evaluation_mse_against_complete_shift_mc',
                                    state_mean=_regression(stats, shift_equal=False)['mse'],
                                    optimizer_loss=None,
                                    optimizer_loss_reason='evaluation_performs_no_optimizer_updates'))


def evaluate_batch(batch, model, *, checkpoint_provenance=None, training_environment=None,
                   expected_complete_shifts=None):
    """Validate a raw/Baseline view and score frozen Actor/scalar-Critic copies.

    This is not authorization to collect, nor a checkpoint-byte verifier. A
    physical runner must pin its checkpoint and guard the collection lifecycle.
    Invalid/incomplete batches raise; they are never silently dropped from n.
    """
    with ReadOnlyTrainingGuard(model, training_environment):
        if hasattr(model, '_assert_healthy'):
            model._assert_healthy()
        beta = model.config['beta']
        _require(model.config['reward_contract'] == CONTRACT, 'evaluation model reward contract differs')
        view = batch if isinstance(batch, SingleCriticTrainingBatch) else training_batch(batch, beta=beta)
        rewards = validate_training_batch(view, beta=beta, contract=CONTRACT)
        source = getattr(batch, 'source_kind', None)
        _require(source in ('synthetic_unit_test', 'real_anylogic_rpc'), 'explicit evaluation source kind required')
        _require(view.policy_version == digest(model.actor), 'evaluation trajectory belongs to another Actor')
        count = len(view.episodes)
        if expected_complete_shifts is not None:
            _require(type(expected_complete_shifts) is int and expected_complete_shifts > 0,
                     'explicit positive evaluation shift count required')
            _require(count == expected_complete_shifts, 'independent evaluation shift count differs')
        actor, critic = copy.deepcopy(model.actor).cpu().eval(), copy.deepcopy(model.critic).cpu().eval()
        for module in (actor, critic):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        before = (digest(actor), digest(critic))
        x = torch.as_tensor(np.asarray(view.observations).copy(), dtype=torch.float32)
        with torch.no_grad():
            predictions = critic(x).detach().cpu().double().numpy()
            entropy = actor.distribution(x).entropy().detach().cpu().double().numpy()
        _require(predictions.shape == (len(rewards),) and np.isfinite(predictions).all(),
                 'single-Critic evaluation output must be finite scalar per state')
        _require(entropy.shape == predictions.shape and np.isfinite(entropy).all(), 'invalid policy entropy')
        _require(before == (digest(actor), digest(critic)), 'evaluation copies changed')
        targets = mc_returns(rewards, view.bounds)
        audits = {row['episode_id']: row for row in view.reward_provenance}
        _require(len(audits) == count, 'evaluation reward audit episode identity differs')
        episodes = []
        for episode in view.episodes:
            start, stop = episode['start'], episode['stop']
            terminal = view.rows[stop - 1]['settlement_info']['reward_components']
            ore, waste, wait, trucks = [float(terminal[key]) for key in
                ('ore_tons_shift', 'waste_tons_shift', 'wait_vehicle_seconds', 'config.num_trucks')]
            _require(all(math.isfinite(value) and value >= 0 for value in (ore, waste, wait)) and
                     math.isfinite(trucks) and trucks > 0, 'invalid evaluation physical totals')
            costs = [int(value) for value in episode['window_costs']]
            epsilon = float(terminal['config.tonnage_epsilon'])
            ore_met, waste_met = ore >= 16000., waste >= 4000.
            physical = dict(Q_ore=ore, Q_waste=waste, W_vehicle_seconds=wait, num_trucks=trucks,
                mean_vehicle_wait_seconds=wait / trucks, window_costs=costs,
                windows=copy.deepcopy(episode['windows']),
                empty_windows=[window['ore_tons'] <= 0 for window in episode['windows']],
                ore_target_met=ore_met, waste_target_met=waste_met,
                both_targets_met=ore_met and waste_met, all_four_windows_compliant=not any(costs),
                overall_success=ore_met and waste_met and not any(costs),
                epsilon_ore_target_met=ore >= 16000. - epsilon,
                epsilon_waste_target_met=waste >= 4000. - epsilon)
            audit = audits[episode['episode_id']]
            ledgers = {key: float(audit[key]) for key in ('wait_reward', 'grade_cost', 'grade_penalty',
                'production_cost', 'production_reward', 'total_reward', 'q_o', 'q_w', 'ledger_residual')}
            total = math.fsum(float(value) for value in rewards[start:stop])
            _require(math.isclose(total, ledgers['total_reward'], rel_tol=1e-10, abs_tol=1e-8),
                     'evaluation return differs from three-ledger audit')
            episodes.append(dict(episode_id=episode['episode_id'],
                experiment_seed=int(episode['experiment_seed']),
                experiment_run_index=int(episode['experiment_run_index']),
                scene_group=episode['scene_group'], policy_version=view.policy_version, beta=float(beta),
                critic_version=before[1], checkpoint_update=model.update_index,
                cumulative_training_shifts=model.completed_episodes,
                n_states=stop-start, production=physical, reward_ledgers=ledgers,
                value_statistics=_score_stats(targets[start:stop], predictions[start:stop]),
                policy_entropy_mean=float(entropy[start:stop].mean()),
                policy_entropy_sum=float(entropy[start:stop].sum())))
        result = dict(schema=FORMAT, reward_contract=CONTRACT, beta=float(beta), source_kind=source,
            role='independent_evaluation', update_index=model.update_index,
            cumulative_training_shifts=model.completed_episodes,
            checkpoint_provenance=copy.deepcopy(checkpoint_provenance),
            fingerprints=dict(actor=before[0], critic=before[1]),
            n_shifts=count, n_states=len(rewards),
            denominators=dict(requested_complete_shifts=expected_complete_shifts,
                valid_complete_shifts=count, invalid_shifts=0,
                scope='validated_complete_batch_only; physical_attempt_counts_owned_by_runner'),
            statistical_unit='complete_shift; conditional_on_frozen_actor',
            metrics=summarize_episodes(episodes), per_episode=episodes,
            training_state_unchanged=True, gradient_updates=0, optimizer_updates=0)
        json.dumps(result, allow_nan=False)
    return result
