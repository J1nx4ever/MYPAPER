"""Complete synthetic RPC-ledger batches, never real simulation data.

These fixtures exercise the production storage/target/update interfaces.  They
carry no physical-performance claim and must never authorize real training.
"""
from __future__ import annotations

import copy
import hashlib

import numpy as np
import torch

from python.constrained_adapter import RewardCostAdapter
from python.constrained_batch import EpisodeBatchBuffer, TrainingBatch
from python.constrained_ppo import digest
from python.observation_contract import FIELD_NAMES, OBSERVATION_SCHEMA_ID
from python.reward_config import configuration_from_components
from python.reward_contract import REWARD_CONTRACT_ID
from python.scenario_contract import from_components


def observation(time=0.0, *, terminal=False, truck=0, variant=0):
    """Use the existing 1519D physical schema without widening any field."""
    value = np.zeros(1519, dtype=np.float32)
    value[0] = time / 28800.0
    value[1] = 1.0 - value[0]
    value[2] = min(4, int(time // 7200)) / 4.0
    value[17] = terminal
    if not terminal:
        value[19:23] = [1, 0, 1, 1] if variant else [1, 1, 1, 1]
        value[27 + truck * 120] = 1
    return value


def components():
    value = {
        'config.production_reward_version': 2.0,
        'config.wait_kappa': 4e-5, 'config.shift_seconds': 28800.0,
        'config.window_seconds': 7200.0, 'config.target_ore_tons_shift': 16000.0,
        'config.target_waste_tons_shift': 4000.0, 'config.w_production': 4.0,
        'config.waste_shortfall_weight': 4.0, 'config.grade_lower': .26,
        'config.grade_upper': .28, 'config.target_grade': .27,
        'config.grade_epsilon': 1e-12, 'config.tonnage_epsilon': 1e-6,
        'config.mass_balance_epsilon': 1e-5, 'config.num_trucks': 9.0,
        'config.waste_task_limit_enabled': 1.0, 'config.waste_task_limit_version': 1.0,
        'config.waste_task_upper_tons': 4193.0, 'config.waste_task_overrun_tons': 193.0,
        'wait': 0.0, 'production': 0.0, 'waste': 0.0,
        'grade_action': 0.0, 'grade_2h': 0.0, 'grade_shift': 0.0,
        'delta_wait_vehicle_seconds': 0.0, 'wait_vehicle_seconds': 0.0,
    }
    schema_sha = hashlib.sha256(('\n'.join(FIELD_NAMES) + '\n').encode()).hexdigest()
    value.update({f'config.obs.schema_word_{i}': float(int(schema_sha[i * 8:i * 8 + 8], 16))
                  for i in range(8)})
    return value


def window(index, grade):
    tons = 3500.0
    return dict(window_index=index, window_start_seconds=index * 7200.0,
                window_end_seconds=(index + 1) * 7200.0, ore_tons=tons,
                metal_tons=tons * grade, grade=grade, grade_gap=.27 - grade,
                grade_reward_raw=0.0, grade_reward_weighted=0.0,
                violation_upper=bool(grade > .28 + 1e-12),
                violation_lower=bool(grade < .26 - 1e-12))


def info(time, sequence, *, episode, seed, windows=(), delta=0.0,
         cumulative=0.0, elapsed=0.0, terminal=False, previous_action=-1,
         shift_grade=.27, mask=None):
    value = components()
    value.update(wait=-delta / 25000.0, delta_wait_vehicle_seconds=delta,
                 wait_vehicle_seconds=cumulative)
    if terminal:
        value.update(production=-.5, waste=-.5, shift_grade=shift_grade,
                     ore_tons_shift=14000.0, waste_tons_shift=3500.0,
                     crusher_received_tons=14000.0, crusher_buffer_tons=0.0,
                     mined_ore_tons=14000.0, mined_waste_tons=3500.0,
                     carried_ore_tons=0.0, carried_waste_tons=0.0,
                     load_sites_reserved_tons=0.0, assigned_not_loaded_tons=0.0)
    return dict(episode_id=episode, experiment_run_index=seed,
                experiment_seed=seed, truck_id=-1 if terminal else 0,
                decision_sequence=sequence, previous_action=previous_action,
                request_id=f'{episode}-{sequence}', sim_time_seconds=float(time),
                sim_time_delta_seconds=float(elapsed), reward_contract_id=REWARD_CONTRACT_ID,
                observation_schema_id=OBSERVATION_SCHEMA_ID, reward_components=value,
                closed_windows=list(windows),
                transition_reward_raw=value['wait'] + value['production'] + value['waste'],
                action_mask=[0] * 4 if terminal else list(mask or [1] * 4))


def make_batch(actor, seed=8123, batch_id='synthetic-v11', reward_version='v1.1'):
    """Return two complete shifts with true Actor log probabilities.

    Each Grade head has one positive and one negative shift.  Shift one covers
    exact 7200-second boundaries; shift two crosses two windows in one callback.
    The immutable raw rows retain the old RPC ledger; only the versioned batch
    reward view adds the frozen V1.1 terminal center penalty.
    """
    policy = digest(actor)
    buffer = EpisodeBatchBuffer(expected_episodes=2, source_kind='synthetic_unit_test')
    schedules = (
        ((100, 7200, 14400, 14500, 21600, 21700, 28800), (.292, .274, .252, .274)),
        ((100, 14500, 14600, 21600, 21700, 28800), (.268, .248, .268, .288)),
    )
    # Sampling consumes a local fork, preserving the caller's global RNG state.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for ordinal, (times, grades) in enumerate(schedules):
            episode_seed = seed * 10 + ordinal
            episode = f'{batch_id}-complete-shift-{ordinal}'
            scene = f'synthetic-scene-{ordinal}'
            group = f'synthetic-independent-shift-{ordinal}'
            initial = observation(variant=ordinal)
            start_info = info(0.0, 0, episode=episode, seed=episode_seed,
                              mask=initial[19:23].astype(int).tolist())
            scenario = from_components(start_info['reward_components'])
            adapter = RewardCostAdapter(configuration_from_components(start_info['reward_components']))
            adapter.start(start_info)
            start = len(buffer.rows)
            obs, before, closed, cumulative = initial.copy(), 0.0, 0, 0.0
            for sequence, time in enumerate(times, 1):
                with torch.no_grad():
                    distribution = actor.distribution(torch.tensor(obs)[None, :])
                    action_tensor = distribution.sample()
                    action = int(action_tensor.item())
                    log_probability = float(distribution.log_prob(action_tensor).item())
                total = min(4, int(time // 7200))
                summaries = [window(j, grades[j]) for j in range(closed, total)]
                closed = total
                terminal = time == 28800
                delta = (300.0 + 50 * ordinal) if sequence == 1 else (180.0 if terminal else 0.0)
                cumulative += delta
                nxt = observation(time, terminal=terminal, variant=(sequence + ordinal) % 2)
                settlement = info(time, sequence, episode=episode, seed=episode_seed,
                                  windows=summaries, delta=delta, cumulative=cumulative,
                                  elapsed=time - before, terminal=terminal, previous_action=action,
                                  shift_grade=sum(grades) / 4.0,
                                  mask=nxt[19:23].astype(int).tolist())
                adapted = adapter.transition(settlement, terminated=terminal, truncated=False)
                buffer.append(dict(**adapted, observation=obs, next_observation=nxt,
                                   truck_id=0, next_truck_id=settlement['truck_id'],
                                   episode_id=episode, experiment_seed=episode_seed,
                                   scene_id=scene, scene_group=group, action=action,
                                   mask=obs[19:23].astype(int).tolist(), old_log_prob=log_probability,
                                   policy_version=policy, settlement_info=copy.deepcopy(settlement),
                                   action_time_seconds=before, scenario_fingerprint=scenario.fingerprint))
                obs, before = nxt, float(time)
            buffer.finish_episode(start, start_info=start_info, initial_observation=initial,
                                  job=dict(scene_id=scene, scene_group=group),
                                  scenario_fingerprint=scenario.fingerprint)
    buffer.seal()
    batch = TrainingBatch.from_buffer(buffer)
    batch.batch_id = batch_id
    if reward_version == 'rpc_v2':
        return batch
    if reward_version != 'v1.1':
        raise ValueError('unknown synthetic reward version')
    from .reward import training_batch
    return training_batch(batch)
