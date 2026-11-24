"""Collect same-contract online MILP shifts with one acknowledgement per action."""

from collections import Counter
from hashlib import sha256
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from milp_smoke_campaign import (
    CAMPAIGN, CONTROL, EXECUTION, METHOD, SCHEMA, _checked_reference, _publish_json, _require, load_reference,
    reference,
)
from baseline1_v1.collection import wait_for_java_start
from python.constrained_adapter import RewardCostAdapter
from python.dispatch_bridge import DispatchCallbackBridge
from python.dispatch_env import DispatchEnv
from python.dispatch_service import create_server
from python.episode_runner import ParameterVariationEpisodeRunner
from python.reward_config import configuration_from_components
from python.runtime_scenario import PREFIX, fingerprint
from traditional_dispatch_v1 import runtime as shared
from traditional_dispatch_v2.runtime import _verify_anchor

def make_policy(kind, method=METHOD):
    if kind == 'rolling_quota':
        from rolling_quota_policy import RollingQuotaPolicy

        return RollingQuotaPolicy(planning_ms=method['planning_ms'], **method['policy_parameters'][kind])
    if kind == 'trip_fifo':
        from trip_fifo_policy import TripFifoPolicy

        return TripFifoPolicy(planning_ms=method['planning_ms'], **method['policy_parameters'][kind])
    raise ValueError(f'unknown MILP policy {kind!r}')


def _collect(env, job, method, policy, *, anchor=None):
    observation, info = env.reset()
    configuration = {key: value for key, value in info['reward_components'].items() if key.startswith('config.')}
    scene = {key: value for key, value in configuration.items() if key.startswith(PREFIX)}
    _require(fingerprint(scene) == job['runtime_scene_sha256'], 'actual scenario digest differs')
    _require((info['experiment_run_index'], info['experiment_seed']) == (job['run_index'], job['experiment_seed']),
             'START run/seed differs from frozen identity')
    material = np.array([scene[f'{PREFIX}load.{site}.{field}'] for site in range(4)
                         for field in shared.MATERIAL_RPC_FIELDS], dtype=np.float32)
    shared._check_rpc(observation, info, job, material, terminal=False)
    rewards = RewardCostAdapter(configuration_from_components(info['reward_components']))
    rewards.start(info)
    initial = dict(observation_sha256=sha256(observation.tobytes()).hexdigest(),
                   mask=info['action_mask'], truck_id=info['truck_id'], sim_time_seconds=info['sim_time_seconds'])
    _verify_anchor(dict(experiment_seed=info['experiment_seed'], initial_state=initial,
                        actual_rpc_configuration=configuration), anchor)
    counts, fallbacks, solver_status = Counter(), Counter(), Counter()
    timed_out = 0
    max_gap = None
    dispatches = [0] * 4
    policy_ms, response_ms = [], []
    while True:
        mask = info['action_mask']
        ticket = env._pending
        _require(ticket is not None, 'active decision has no transport ticket')
        started = perf_counter_ns()
        choice = policy.choose(observation, mask)
        policy_ms.append((perf_counter_ns() - started) / 1e6)
        action = choice.action
        _require(type(action) is int and 0 <= action < len(mask) and bool(mask[action]),
                 'MILP strategy selected an action outside the live mask')
        _require(choice.kind in ('single_legal', 'planned', 'fallback'), 'unknown MILP outcome')
        if choice.kind == 'single_legal':
            _require(sum(mask) == 1, 'single legal shortcut requires exactly one allowed action')
        if choice.kind == 'fallback':
            _require(bool(choice.fallback_reason), 'fallback requires a reason')
            fallbacks[choice.fallback_reason] += 1
        observation, _, terminated, truncated, info = env.step(action)
        response_ms.append(ticket.response.latency_us / 1000)
        shared._check_rpc(observation, info, job, material, terminal=terminated)
        rewards.transition(info, terminated=terminated, truncated=truncated)
        _require(info['previous_action'] == action, 'AnyLogic did not attest the issued action')
        policy.commit(choice)
        counts[choice.kind] += 1
        dispatches[action] += 1
        timed_out += int(choice.timed_out)
        if choice.lp_status:
            solver_status[choice.lp_status] += 1
        if choice.mip_gap is not None:
            _require(np.isfinite(choice.mip_gap) and choice.mip_gap >= 0, 'invalid MILP gap')
            max_gap = max(float(choice.mip_gap), max_gap or 0.0)
        if terminated:
            break

    decisions = sum(dispatches)
    _require(decisions > 0 and decisions == info['decision_sequence'] == rewards.sequence == sum(counts.values()),
             'issued, attested and terminal counts differ')
    summary = policy.summary()
    target = method['response_target_ms']
    over_target = sum(elapsed > target for elapsed in response_ms)
    engineering = dict(decisions=decisions, dispatches_by_site=dispatches, action_counts=dict(counts),
                       fallback_reasons=dict(fallbacks), fallback_rate=counts['fallback'] / decisions,
                       policy_latency_ms=shared._percentiles(policy_ms),
                       rpc_receive_to_response_ms=shared._percentiles(response_ms),
                       rpc_response_target_ms=target, rpc_responses_over_target=over_target,
                       solver_status_counts=dict(solver_status), solver_time_limit_requests=timed_out,
                       max_mip_gap=max_gap, solver=summary)
    if job['policy_kind'] == 'trip_fifo':
        acknowledged = summary['committed_actions'] == dispatches
    else:
        acknowledged = summary['committed_actions'] == decisions
    acceptance = dict(rpc_and_physical_completion=True, met_response_target=over_target == 0,
                      milp_incumbent_used=summary['mip_accepted'] > 0, actions_acknowledged=acknowledged)
    if job['policy_kind'] == 'trip_fifo':
        acceptance['joint_trip_plan_used'] = summary['joint_plan_commits'] > 0
    acceptance['passed'] = all(acceptance.values())
    return dict(initial_state=initial, actual_rpc_configuration=configuration,
                physical=shared._physical(rewards, info), engineering=engineering,
                engineering_acceptance=acceptance, episode_id=info['episode_id'],
                terminal_request_id=info['request_id'], terminal_time_seconds=info['sim_time_seconds'])


def run_shift(suite, ledger, job):
    run = job['run_index']
    startup = dict(campaign_id=CAMPAIGN, control_directory=str(CONTROL),
                   master_plan_sha256=suite['master_plan']['sha256'])
    output = Path(job['python_output']) / 'evaluation_receipt.json'
    claim = CONTROL / 'java_claims' / f'run_{run}.once.claim'
    ready = CONTROL / 'ready' / f'run_{run}.properties'
    _require(not any(path.exists() for path in (output, claim, ready, CONTROL / 'STOP.properties')),
             'run identity has existing evidence')
    anchor = None
    if job['ordinal'] != 0:
        first = suite['jobs'][0]
        _, slots = ledger.snapshot()
        slot = slots[str(first['run_index'])]
        _require(slot['status'] == 'COMPLETE' and slot['identity'] == first,
                 'first paired shift has no settled receipt')
        anchor = load_reference(slot['evidence'])
    policy = make_policy(job['policy_kind'], suite['method'])
    bridge = DispatchCallbackBridge(expected_episode_metadata=[(run, job['experiment_seed'])])
    timeout = EXECUTION['callback_timeout_seconds']
    env = DispatchEnv(bridge, ParameterVariationEpisodeRunner(1), timeout)
    server = create_server(bridge, host=EXECUTION['host'], port=EXECUTION['port'])
    claimed = False
    try:
        server.start()
        ledger.transition(job, 'PLANNED', 'CLAIMED')
        claimed = True
        shared._publish_ready(startup, job)
        wait_for_java_start(startup, job, job['phase_config']['sha256'], timeout)
        result = _collect(env, job, suite['method'], policy, anchor=anchor)
        _checked_reference(suite['protected_core'])
        receipt = dict(result, schema=SCHEMA + '-receipt', campaign_id=CAMPAIGN, status='COMPLETE',
                       normal_completion=True, completed_shifts=1, optimizer_updates=0,
                       java_start_claim=reference(claim),
                       **{key: job[key] for key in ('run_index', 'experiment_seed', 'recording_id',
                                                    'actor_sampling_seed', 'policy_kind', 'scene_id', 'phase_config')})
        pinned = _publish_json(output, receipt)
        ledger.transition(job, 'CLAIMED', 'COMPLETE', pinned)
        claimed = False
        return receipt, pinned
    except BaseException as error:
        if claimed:
            ledger.transition(job, 'CLAIMED', 'UNKNOWN', dict(reason=repr(error)))
        raise
    finally:
        env.close()
        server.stop(grace=0).wait()
