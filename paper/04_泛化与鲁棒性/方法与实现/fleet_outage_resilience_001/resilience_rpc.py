"""Single-shift on-policy gRPC collection. Only the 32 live training shifts need raw states."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT))
from python.dispatch_bridge import observation_array
from python.generated import dispatch_pb2 as wire
from python.observation_contract import FIELD_NAMES
from python.reward_config import RewardConfiguration, partition_components, require
from python.reward_contract import (REWARD_CONTRACT_ID, validate_window_summary,
                                    window_summary_to_dict)
from python.runtime_scenario import PREFIX, validate_runtime_fields
from python.scenario_contract import from_components as waste_scenario
from single_critic_task_query_v1.targets import grade_cost
from task_query_cmdp_v1.runtime import _near

SCHEMA_HASH = hashlib.sha256(('\n'.join(FIELD_NAMES) + '\n').encode()).hexdigest()
WINDOW_SECONDS = 7200.
HORIZON = 28800.


def verify_config(components, job, terminal):
    actual, _, _ = partition_components(components, terminal=terminal)
    require(actual == job['expected_rpc_config'], 'live RPC configuration differs from frozen shift')
    config = RewardConfiguration(tuple(sorted(actual.items())))
    require(config.get('num_trucks') == 9 and config.get('shift_seconds') == HORIZON
            and config.get('window_seconds') == WINDOW_SECONDS, 'wrong observation or settlement horizon')
    require(config.get('fleet.outage_version') == 2
            and config.get('fleet.disabled_mask') == 511 ^ job['initial_mask']
            and config.get('fleet.active_count') == job['initial_mask'].bit_count(), 'incorrect opening fleet')
    require(config.get('target_ore_tons_shift') == job['ore_target'] > 0.
            and config.get('target_waste_tons_shift') == job['waste_target'] > 0., 'opening target mismatch')
    for i in range(8):
        require(config.get(f'obs.schema_word_{i}') == int(SCHEMA_HASH[i*8:(i+1)*8], 16),
                'incorrect observation schema fingerprint')
    require(waste_scenario(components) is not None, 'missing waste dispatch scenario')
    validate_runtime_fields(components, {k:v for k,v in actual.items() if k.startswith(PREFIX)})
    return config


def check_fleet(job, request, c, state):
    now = request.sim_time_seconds
    initial = job['initial_mask']
    tid = job['fault_truck_id']
    planned = job['fault_trigger_seconds']
    trig, exited = c['fleet.trigger_seconds'], c['fleet.exit_seconds']
    if tid < 0 or now < planned:
        require(trig == exited == -1., 'fault visible before scheduled event')
    else:
        require(math.isclose(trig, planned, rel_tol=0., abs_tol=1e-7)
                and (exited == -1. or trig <= exited <= now), 'fault did not occur on the frozen clock')
    flag = 0 if tid < 0 else 1 << tid
    disp = initial ^ (flag if trig >= 0. else 0)
    physical = initial ^ (flag if exited >= 0. else 0)
    require(c['fleet.dispatchable_mask'] == disp and c['fleet.in_system_mask'] == physical,
            'live availability differs from trigger/exit')
    if request.action_required:
        require(0 <= request.truck_id < 9 and disp & (1 << request.truck_id), 'dispatch for ineligible truck')
    trucks = state[27:1107].reshape(9, 120)
    disabled = [i for i in range(9) if not physical & (1 << i)]
    parked = trucks[disabled]
    require(not parked[:, 0].any() and not parked[:, 4:15].any() and (parked[:, 15] == 1).all()
            and not parked[:, 17:41].any() and not parked[:, 43:].any(), 'retired slot physically active')
    loads = state[1107:1295].reshape(4, 47)
    dumps = state[1295:1391].reshape(2, 48)
    refs = np.concatenate([trucks[:, 81], loads[:, 20:].ravel(), dumps[:, 11:].ravel()])
    require(not np.isin(np.rint(refs * 9), np.asarray(disabled) + 1).any(), 'retired vehicle still referenced')
    return disp, physical


def check_snapshot(job, state, c):
    """Reconcile actual wire tons; include retiring trucks carrying ore or waste."""
    trucks = state[27:1107].reshape(9, 120)
    loads = state[1107:1295].reshape(4, 47)
    waste = loads[:, 1] == 0
    carried = trucks[(trucks[:, 17] == 1) & (trucks[:, 18] == 0), 19]
    ore_carried = trucks[(trucks[:, 17] == 1) & (trucks[:, 18] == 1), 19]
    source_ore = loads[~waste, 8]
    require(waste.sum() == 1, 'shift requires one waste load station')
    for idx, scale, target in [(11, 20000, job['ore_target']), (12, 4000, job['waste_target'])]:
        _near(float(state[idx])*scale, target, [(state[idx], scale)], 'observation/START target')
    committed = c['audit.waste_task.committed_tons']
    delivered = c['audit.waste_task.delivered_tons']
    _near(delivered, float(state[9])*4000, [(state[9], 4000)], 'delivered waste')
    _near(committed, delivered + carried.sum(dtype=np.float64)*193 + loads[waste,6].sum(dtype=np.float64)*30000,
          [(carried,193), (loads[waste,6],30000)], 'committed waste includes all physical cargo')
    _near(source_ore.sum(dtype=np.float64)*30000, float(state[7])*20000 + float(state[1300])*500
          + ore_carried.sum(dtype=np.float64)*193, [(source_ore,30000), (state[7],20000),
          (state[1300],500), (ore_carried,193)], 'ore loaded/crushed/FIFO/inbound reconciliation')
    stop = c['config.waste_task_dispatch_stop_tons']
    closed = c['audit.waste_task.assignment_closed']
    require(closed == float(committed >= stop - 1e-6) and
            (not closed or not state[19:23][waste].any()), 'waste dispatch stop differs from commitments')
    require(committed <= c['config.waste_task_upper_tons'] + 1e-6, 'waste overshoot')


def check_reward(request, c, config, terminal):
    delta = c['delta_wait_vehicle_seconds']
    require(delta >= 0. and c['wait_vehicle_seconds'] >= 0., 'negative fleet waiting')
    expected = {'wait': -config.get('wait_kappa')*delta, 'grade_action':0., 'grade_2h':0., 'grade_shift':0.}
    for key, tons, target, weight in [('production','ore_tons_shift','target_ore_tons_shift','w_production'),
                                      ('waste','waste_tons_shift','target_waste_tons_shift','waste_shortfall_weight')]:
        expected[key] = 0.
        if terminal:
            require(c[tons] >= 0., 'negative shift production')
            expected[key] = -config.get(weight)*max(0., 1.-c[tons]/config.get(target))
    for key, value in expected.items():
        require(math.isclose(c[key], value, rel_tol=1e-10, abs_tol=1e-9), f'RPC reward mismatch: {key}')
    require(math.isclose(request.transition_reward, math.fsum(expected.values()), rel_tol=1e-10,
                         abs_tol=1e-9), 'RPC combined reward mismatch')


class Shift:
    def __init__(self, job, learner, verify_terminal):
        self.job, self.model, self.verify_terminal = job, learner, verify_terminal
        self.version = learner.policy_version
        self.generator = torch.Generator().manual_seed(job['actor_sampling_seed'])
        self.states, self.actions, self.logs, self.wait = [], [], [], []
        self.dispatchable, self.in_system, self.closures, self.windows = [], [], [], []
        self.previous_time = self.previous_wait = 0.
        self.last_exit = -1.
        self.actual_config = None
        self.episode = self.receipt = None
        self.processed = {}

    def accept(self, request):
        r = request
        j = self.job
        require(r.experiment_run_index == j['run_index'] and r.experiment_seed == j['experiment_seed'],
                'physical run/seed does not match ready job')
        require(r.reward_contract_id == REWARD_CONTRACT_ID and r.request_id and
                r.episode_id == f"{REWARD_CONTRACT_ID}-run-{j['run_index']}-seed-{j['experiment_seed']}",
                'invalid callback identity')
        digest = hashlib.sha256(r.SerializeToString(deterministic=True)).digest()
        previous = self.processed.get(r.request_id)
        if previous is not None:
            require(previous[0] == digest, 'request identity reused with different bytes')
            return previous[1]
        n = len(self.actions)
        terminal = r.kind == wire.EPISODE_END
        now = float(r.sim_time_seconds)
        require(self.receipt is None and r.decision_sequence == n and not r.truncated and
                r.kind == (wire.EPISODE_START if n == 0 else wire.EPISODE_END if terminal else wire.DECISION)
                and r.action_required == (not terminal) and r.terminated == terminal,
                'invalid callback sequence or lifecycle')
        require(r.previous_action == (self.actions[-1] if n else -1), 'action echo mismatch')
        require(math.isfinite(now) and 0 <= self.previous_time <= now <= HORIZON and
                (now == HORIZON) == terminal and math.isclose(r.sim_time_delta_seconds,
                now-self.previous_time if n else 0., rel_tol=0., abs_tol=1e-6), 'invalid simulation time')
        c = dict(r.reward_components)
        cfg = verify_config(c, j, terminal)
        if self.actual_config is None:
            self.actual_config = cfg.to_dict()
        state = observation_array(r)
        d, p = check_fleet(j, r, c, state)
        if self.last_exit >= 0.:
            require(self.last_exit == c['fleet.exit_seconds'], 'retirement time changed')
        self.last_exit = c['fleet.exit_seconds']
        check_snapshot(j, state, c)
        check_reward(r, c, cfg, terminal)
        delta = c['delta_wait_vehicle_seconds']
        require(math.isclose(c['wait_vehicle_seconds'], self.previous_wait + delta, rel_tol=1e-9,
                             abs_tol=1e-6), 'cumulative waiting mismatch')
        if n:
            self.wait.append(delta)
        else:
            require(not r.closed_windows and delta == 0. and r.transition_reward == 0., 'invalid first callback')
        for w in r.closed_windows:
            values = window_summary_to_dict(w)
            validate_window_summary(values, config=cfg)
            require(w.window_index == len(self.windows) < 4 and w.window_end_seconds <= now,
                    'missing, future or duplicate settlement')
            grade = w.grade if w.ore_tons > 0. else None
            self.windows.append(dict(window=w.window_index, tons=w.ore_tons, metal_tons=w.metal_tons,
                                     grade=grade, empty=grade is None, compliant=grade is not None and
                                     cfg.get('grade_lower') <= grade <= cfg.get('grade_upper'),
                                     continuous_cost=grade_cost(grade, empty=grade is None)))
            self.closures.append(dict(window=w.window_index, step=n-1, tons=w.ore_tons, metal_tons=w.metal_tons))
        self.previous_time, self.previous_wait = now, c['wait_vehicle_seconds']
        self.states.append(state)
        self.dispatchable.append(d)
        self.in_system.append(p)
        response = wire.DispatchResponse(has_action=not terminal, episode_id=r.episode_id,
                                         decision_sequence=r.decision_sequence)
        if not terminal:
            action, log_prob, value = self.model.act(state, d, p, self.generator, opening_mask=j['initial_mask'])
            require(0 <= action < 4 and state[19+action] == 1. and math.isfinite(log_prob)
                    and math.isfinite(value), 'learner supplied an invalid action')
            self.actions.append(action)
            self.logs.append(log_prob)
            response.action, response.value = action, value
        else:
            self.finish(c)
        self.processed[r.request_id] = (digest, response)
        return response

    def finish(self, c):
        j = self.job
        require(len(self.windows) == 4 and self.actions, 'incomplete shift at END')
        outage = self.verify_terminal(j, c)
        ore = c['ore_tons_shift']
        waste = c['waste_tons_shift']
        for key, name in [('tons', 'ore'), ('metal_tons', 'metal')]:
            total = math.fsum(w[key] for w in self.windows)
            require(math.isclose(total, c['fleet.processed_' + name + '_tons'], rel_tol=1e-10, abs_tol=1e-5),
                    'four-window series disagrees with crusher totals')
        require(math.isclose(ore, c['fleet.processed_ore_tons'], rel_tol=1e-10, abs_tol=1e-5),
                'shift production disagrees with crusher total')
        ore_missing = max(0., 1.-ore/j['ore_target'])
        waste_missing = max(0., 1.-waste/j['waste_target'])
        grade = math.fsum(w['continuous_cost'] for w in self.windows)
        waiting = 0.2 * self.previous_wait / HORIZON
        production = 8. * (ore_missing + waste_missing)
        ok = all(w['compliant'] for w in self.windows)
        physical = dict(ore_tons=ore, waste_tons=waste, total_wait_seconds=self.previous_wait,
                        ore_target_tons=j['ore_target'], waste_target_tons=j['waste_target'],
                        ore_shortfall=ore_missing, waste_shortfall=waste_missing,
                        ore_target_met=ore >= j['ore_target'], waste_target_met=waste >= j['waste_target'],
                        grade_cost=grade, production_cost=production, waiting_cost=waiting,
                        combined_cost=grade+production+waiting, all_four_windows_compliant=ok,
                        complete_task=ok and ore>=j['ore_target'] and waste>=j['waste_target'], windows=self.windows)
        self.episode = dict(states=np.stack(self.states), actions=np.asarray(self.actions, dtype=np.int64),
                            dispatchable_masks=np.asarray(self.dispatchable,dtype=np.int64),
                            in_system_masks=np.asarray(self.in_system,dtype=np.int64),
                            wait_deltas=np.asarray(self.wait), old_log_probs=np.asarray(self.logs),
                            rpc_config=self.actual_config, policy_version=self.version, evaluation=physical,
                            identity=dict(recording_id=j['recording_id'], run_index=j['run_index'],
                                          environment_seed=j['experiment_seed'], actor_sampling_seed=j['actor_sampling_seed']),
                            window_closures=self.closures)
        self.receipt = {k: j[k] for k in ('run_index','recording_id','experiment_seed','actor_sampling_seed',
                        'config_sha256','phase','arm','role','update','split','pair_ordinal','condition') if k in j}
        self.receipt.update(physical=physical, outage=outage, decisions=len(self.actions), policy_version=self.version,
                            actual_rpc_config=self.actual_config, terminal_time_seconds=self.previous_time)
