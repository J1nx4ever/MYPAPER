"""Validate and annotate RPC transitions. Never replace or add to RPC reward."""
import math
import numpy as np
from .reward_config import configuration_from_components, finite
from .reward_contract import (REWARD_CONTRACT_ID, validate_reward_accounting, validate_window_summary)
from .observation_contract import OBSERVATION_SCHEMA_ID

ADAPTER_CONTRACT_ID = 'dispatch-four-window-transitions-v2'


class SamplingError(RuntimeError):
    pass


class TruncatedShift(SamplingError):
    pass


def require(condition, message):
    if not condition:
        raise SamplingError(message)


class RewardCostAdapter:
    def __init__(self, config):
        self.config, self.active = config, False

    def check(self, info, terminal):
        require(info['reward_contract_id'] == REWARD_CONTRACT_ID, 'reward contract changed')
        require(info['observation_schema_id'] == OBSERVATION_SCHEMA_ID, 'observation schema changed')
        require(configuration_from_components(info['reward_components'], terminal=terminal) == self.config,
                'immutable RPC configuration changed')

    def start(self, info):
        require(not self.active, 'cannot abandon an incomplete episode')
        self.check(info, False)
        require(info['decision_sequence'] == 0 and info['previous_action'] == -1 and
                info['sim_time_delta_seconds'] == 0 and not info['closed_windows'] and
                info['transition_reward_raw'] == 0, 'invalid episode start')
        require(info['sim_time_seconds'] >= 0 and info['sim_time_seconds'] < 7200, 'invalid episode prefix')
        require(finite(info['reward_components'], 'wait_vehicle_seconds') == 0 and
                finite(info['reward_components'], 'delta_wait_vehicle_seconds') == 0, 'unaccounted prefix waiting')
        self.identity = (info['episode_id'], info['experiment_run_index'], info['experiment_seed'])
        self.sequence, self.time = 0, info['sim_time_seconds']
        self.wait, self.total_reward, self.windows, self.active = 0., 0., [], True

    def transition(self, info, *, terminated, truncated):
        require(self.active, 'transition without active episode')
        if truncated:
            raise TruncatedShift('sampling truncation/abnormal interruption cannot be a normal shift')
        self.check(info, terminated)
        require((info['episode_id'], info['experiment_run_index'], info['experiment_seed']) == self.identity,
                'episode identity changed')
        require(info['decision_sequence'] == self.sequence + 1, 'noncontiguous decision sequence')
        now = finite(info, 'sim_time_seconds')
        require(now >= self.time and math.isclose(now-self.time, info['sim_time_delta_seconds'], rel_tol=0, abs_tol=1e-6),
                'transition time mismatch')
        require((terminated and now == 28800) or (not terminated and now < 28800), 'false shift terminal')
        c = info['reward_components']
        cumulative, delta = finite(c, 'wait_vehicle_seconds'), finite(c, 'delta_wait_vehicle_seconds')
        require(cumulative >= self.wait and delta >= 0, 'cumulative waiting regressed')
        require(math.isclose(cumulative-self.wait, delta, rel_tol=1e-10, abs_tol=1e-7), 'waiting conservation failed')
        # m is evaluated at the action state, BEFORE consuming settlement events.
        valid = np.array([j >= len(self.windows) for j in range(4)], dtype=bool)
        bootstrap = valid.copy()
        costs = np.zeros(4)
        for w in info['closed_windows']:
            j = len(self.windows)
            require(j < 4 and w['window_index'] == j and w['window_end_seconds'] <= now,
                    'missing/duplicate/out-of-order window')
            costs[j] = validate_window_summary(w, config=self.config)
            bootstrap[j] = False
            self.windows.append(dict(w))
        require(len(self.windows) == min(4, int(now//7200)), 'a crossed window settlement is missing')
        if terminated:
            require(len(self.windows) == 4, 'terminal requires four settlements')
            bootstrap[:] = False
            self._material(c)
        reward = float(info['transition_reward_raw'])
        validate_reward_accounting(reward, c, info['closed_windows'], config=self.config, terminal=terminated)
        self.total_reward += reward
        if terminated:
            expected = (-self.config.get('wait_kappa')*cumulative
                        -self.config.get('w_production')*max(0., 1-c['ore_tons_shift']/16000)
                        -self.config.get('waste_shortfall_weight')*max(0., 1-c['waste_tons_shift']/4000))
            require(math.isclose(self.total_reward, expected, rel_tol=1e-10, abs_tol=1e-8), 'full-shift reward ledger failed')
            self.active = False
        self.wait, self.time, self.sequence = cumulative, now, self.sequence+1
        return dict(reward=reward, grade_costs=costs.tolist(), grade_valid=valid.tolist(),
                    grade_bootstrap=bootstrap.tolist(), grade_trace=bootstrap.tolist(),
                    reward_bootstrap=not terminated, reward_trace=not terminated,
                    next_grade_valid=[j >= len(self.windows) for j in range(4)],
                    terminated=bool(terminated), truncated=False, abnormal_interruption=False,
                    sampler_truncated=False)

    def _material(self, c):
        keys = ('ore_tons_shift','waste_tons_shift','crusher_received_tons','crusher_buffer_tons',
                'mined_ore_tons','mined_waste_tons','carried_ore_tons','carried_waste_tons',
                'load_sites_reserved_tons','assigned_not_loaded_tons')
        p = {k: finite(c, k) for k in keys}
        require(all(v >= 0 for v in p.values()), 'negative terminal material')
        residuals = (sum(w['ore_tons'] for w in self.windows)-p['ore_tons_shift'],
                     p['crusher_received_tons']-p['ore_tons_shift']-p['crusher_buffer_tons'],
                     p['mined_ore_tons']-p['crusher_received_tons']-p['carried_ore_tons'],
                     p['mined_waste_tons']-p['waste_tons_shift']-p['carried_waste_tons'],
                     p['load_sites_reserved_tons']-p['assigned_not_loaded_tons'])
        require(all(abs(r) <= self.config.get('mass_balance_epsilon') for r in residuals), 'terminal material conservation failed')
        shift_grade=finite(c,'shift_grade')
        require(0<=shift_grade<=1,'invalid shift grade')
        require(abs(sum(w['metal_tons'] for w in self.windows)-p['ore_tons_shift']*shift_grade)
                <=self.config.get('mass_balance_epsilon'),'window/shift metal conservation failed')
