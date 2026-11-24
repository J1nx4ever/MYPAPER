"""One physical AnyLogic validation shift for the DISPATCH-inspired controller."""

from collections import Counter
import hashlib
import math
from pathlib import Path
import sys
import time

import numpy as np

from baseline1_v1.collection import wait_for_java_start
from business_graph_summary_campaign_v1.plan_io import load_reference, reference
from python.constrained_adapter import RewardCostAdapter
from python.dispatch_bridge import DispatchCallbackBridge
from python.dispatch_env import DispatchEnv
from python.dispatch_service import create_server
from python.episode_runner import ParameterVariationEpisodeRunner
from python.reward_config import configuration_from_components
from python.runtime_scenario import PREFIX, fingerprint
from traditional_dispatch_v1 import runtime as smoke


SUITE_SCHEMA = 'bg128-dispatch-inspired-validation-v1'
PHASE_SCHEMA = 'bg128-dispatch-inspired-validation-phase-v1'
RECEIPT_SCHEMA = 'bg128-dispatch-inspired-validation-receipt-v1'
REPORT_FILENAME = 'single_shift_validation.json'
POLICY = 'dispatch_inspired'
FLOW_FIELDS = ('flow_shortfall_ore_tons', 'flow_shortfall_waste_tons', 'flow_grade_slack_tons')


def _phase(suite, job):
    phase = load_reference(job['phase_config'])
    identity = {key: value for key, value in job.items() if key != 'phase_config'}
    smoke._require(phase['schema'] == PHASE_SCHEMA and phase['campaign_id'] == suite['campaign_id']
                   and phase['job'] == identity and phase['policy_kind'] == POLICY
                   and phase['method'] == suite['method'] and phase['execution'] == suite['execution'],
                   'phase configuration and frozen physical identity differ')
    return phase


def _verify_artifacts(suite):
    for key in ('master_plan', 'protected_core'):
        smoke._checked_reference(suite[key])


def _make_policy(method):
    from dispatch_policy import DispatchPolicy

    return DispatchPolicy(planning_ms=method['planning_ms'], replan_seconds=method['replan_seconds'],
                          match_trucks=method['match_trucks'])


def _count_choice(decision, mask, match_trucks, counts, fallbacks, flow_results):
    smoke._require(decision.kind in ('single_legal', 'planned', 'fallback'), 'unknown strategy outcome')
    smoke._require(type(decision.replanned) is bool and type(decision.timed_out) is bool,
                   'planning states must be explicit boolean')
    smoke._require(type(decision.matched_trucks) is int and 0 <= decision.matched_trucks <= match_trucks
                   and type(decision.assignments_scored) is int and decision.assignments_scored >= 0,
                   'invalid matching counters')
    counts[decision.kind] += 1
    if decision.kind == 'single_legal':
        smoke._require(sum(mask) == 1, 'single-action shortcut requires exactly one legal action')
    if decision.kind == 'fallback':
        smoke._require(bool(decision.fallback_reason), 'strategy fallback requires a reason')
        fallbacks[decision.fallback_reason] += 1
    if decision.lp_status is not None:
        smoke._require(decision.lp_status in ('optimal', 'feasible_limit', 'failed'), 'unknown flow-plan solver status')
        counts[f'lp_{decision.lp_status}'] += 1
    if decision.replanned:
        smoke._require(decision.lp_status in ('optimal', 'feasible_limit'), 'accepted flow plan is missing a feasible status')
        flow = {key: getattr(decision, key) for key in FLOW_FIELDS}
        smoke._require(all(value is not None and math.isfinite(value) and value >= 0 for value in flow.values()),
                       'accepted flow plan lacks finite nonnegative slack evidence')
        flow_results.append(flow)
        counts['flow_plan_successes'] += 1
    counts['timed_out'] += int(decision.timed_out)
    counts['assignments_scored'] += decision.assignments_scored
    matched = decision.kind == 'planned' and decision.matched_trucks > 1 and decision.assignments_scored > 0
    counts['live_multitruck_matches'] += int(matched)
    counts['max_matched_trucks'] = max(counts['max_matched_trucks'], decision.matched_trucks)
    counts[f'matched_{decision.matched_trucks}_trucks_requests'] += 1
    return matched


def _collect(env, job, method, policy):
    observation, info = env.reset()
    configuration = {key: value for key, value in info['reward_components'].items()
                     if key.startswith('config.')}
    scene = {key: value for key, value in configuration.items() if key.startswith(PREFIX)}
    smoke._require(fingerprint(scene) == job['runtime_scene_sha256'], 'actual scene digest differs')
    smoke._require((info['experiment_run_index'], info['experiment_seed'])
                   == (job['run_index'], job['experiment_seed']), 'START run/seed differs from the frozen job')
    material = np.array([scene[f'{PREFIX}load.{site}.{field}'] for site in range(4)
                         for field in smoke.MATERIAL_RPC_FIELDS], dtype=np.float32)
    smoke._check_rpc(observation, info, job, material, terminal=False)
    rewards = RewardCostAdapter(configuration_from_components(info['reward_components']))
    rewards.start(info)
    initial = dict(observation_sha256=hashlib.sha256(observation.tobytes()).hexdigest(),
                   mask=info['action_mask'], truck_id=info['truck_id'],
                   sim_time_seconds=info['sim_time_seconds'])

    counts, fallbacks = Counter(), Counter()
    dispatches = [0] * 4
    policy_ms, response_ms, flow_results = [], [], []
    witness = None
    while True:
        mask = info['action_mask']
        ticket = env._pending
        smoke._require(ticket is not None, 'active decision has no transport ticket')
        started = time.perf_counter_ns()
        decision = policy.choose(observation, mask)
        policy_ms.append((time.perf_counter_ns() - started) / 1e6)
        action = decision.action
        smoke._require(type(action) is int and 0 <= action < len(mask) and bool(mask[action]),
                       'strategy selected an action outside the live mask')
        matched = _count_choice(decision, mask, method['match_trucks'], counts, fallbacks, flow_results)
        if matched and witness is None:
            witness = dict(decision_sequence=info['decision_sequence'], sim_time_seconds=info['sim_time_seconds'],
                           truck_id=info['truck_id'], action=action, mask=mask,
                           matched_trucks=decision.matched_trucks, assignments_scored=decision.assignments_scored)
        observation, _, terminated, truncated, info = env.step(action)
        response_ms.append(ticket.response.latency_us / 1000.)
        smoke._check_rpc(observation, info, job, material, terminal=terminated)
        rewards.transition(info, terminated=terminated, truncated=truncated)
        smoke._require(info['previous_action'] == action, 'next callback did not attest the action that was just applied')
        policy.commit(decision)
        dispatches[action] += 1
        if terminated:
            break

    decisions = sum(dispatches)
    smoke._require(decisions > 0 and decisions == info['decision_sequence'] == rewards.sequence
                   and decisions == sum(counts[key] for key in ('single_legal', 'planned', 'fallback')),
                   'decision, live action and terminal counts differ')
    target = method['response_target_ms']
    over_target = sum(elapsed > target for elapsed in response_ms)
    engineering = dict(decisions=decisions, dispatches_by_site=dispatches, action_counts=dict(counts),
                       fallback_reasons=dict(fallbacks), fallback_rate=counts['fallback'] / decisions,
                       policy_latency_ms=smoke._percentiles(policy_ms),
                       rpc_receive_to_response_ms=smoke._percentiles(response_ms),
                       rpc_response_target_ms=target, rpc_responses_over_target=over_target,
                       first_live_multitruck_match=witness)
    if flow_results:
        engineering['accepted_flow_plan_slack'] = {
            key: dict(n=len(flow_results), mean=math.fsum(plan[key] for plan in flow_results) / len(flow_results),
                      max=max(plan[key] for plan in flow_results)) for key in FLOW_FIELDS}
    acceptance = dict(rpc_and_physical_completion=True, flow_plan_observed=counts['flow_plan_successes'] > 0,
                      live_multitruck_match_observed=witness is not None, met_response_target=over_target == 0)
    acceptance['passed'] = all(acceptance.values())
    return dict(initial_state=initial, actual_rpc_configuration=configuration,
                physical=smoke._physical(rewards, info), engineering=engineering,
                engineering_acceptance=acceptance, episode_id=info['episode_id'],
                terminal_request_id=info['request_id'], terminal_time_seconds=info['sim_time_seconds'])


def run(suite, ledger, *, progress=None):
    """Consume one fresh admitted job and return its sealed report reference."""
    smoke._require(suite['schema'] == SUITE_SCHEMA and len(suite['jobs']) == 1,
                   'one frozen DISPATCH-inspired physical validation shift is required')
    job = suite['jobs'][0]
    smoke._require(job['policy_kind'] == POLICY and job['actor_seed'] == job['actor_sampling_seed'] == 0,
                   'the deterministic validation policy cannot consume an Actor seed')
    phase = _phase(suite, job)
    _verify_artifacts(suite)
    output = Path(job['python_output']) / 'evaluation_receipt.json'
    control = Path(suite['control_directory'])
    ready = control / 'ready' / f"run_{job['run_index']}.properties"
    claim = control / 'java_claims' / f"run_{job['run_index']}.once.claim"
    report_path = Path(suite['report_directory']) / REPORT_FILENAME
    smoke._require(not any(path.exists() for path in (output, claim, ready, report_path, control / 'STOP.properties')),
                   'validation has existing physical evidence; do not reuse the run identity')
    frozen_slot = ledger.snapshot()['slots'][str(job['run_index'])]
    smoke._require(frozen_slot['status'] == 'PLANNED' and frozen_slot['identity'] == job,
                   'validation requires the exact unclaimed physical job')
    startup = dict(control_directory=str(control), campaign_id=suite['campaign_id'],
                   master_plan_sha256=suite['master_plan']['sha256'])
    settings = suite['execution']
    policy = _make_policy(phase['method'])
    bridge = DispatchCallbackBridge(expected_episode_metadata=[(job['run_index'], job['experiment_seed'])])
    env = DispatchEnv(bridge, ParameterVariationEpisodeRunner(1), settings['callback_timeout_seconds'])
    server = create_server(bridge, host=settings['host'], port=settings['port'])
    claimed = False
    try:
        server.start()
        ledger.claim(job)
        claimed = True
        smoke._publish_ready(startup, job)
        wait_for_java_start(startup, job, job['phase_config']['sha256'], settings['callback_timeout_seconds'])
        result = _collect(env, job, phase['method'], policy)
        server.stop(grace=2).wait()
        _verify_artifacts(suite)
        fields = ('run_index', 'experiment_seed', 'actor_sampling_seed', 'recording_id', 'policy_kind', 'phase_config')
        receipt = dict(result, **{key: job[key] for key in fields}, schema=RECEIPT_SCHEMA,
                       status='COMPLETE', normal_completion=True, optimizer_updates=0, completed_shifts=1,
                       java_start_claim=reference(claim))
        ref = smoke._publish_json(output, receipt)
        ledger.complete(job, ref)
        claimed = False
    except BaseException as error:
        if claimed:
            try:
                ledger.mark_unknown(job, repr(error))
            except Exception as ledger_error:
                print(f'ledger failure: {ledger_error!r}', file=sys.stderr, flush=True)
        raise
    finally:
        env.close()
        server.stop(grace=0).wait()

    report = dict(schema='bg128-dispatch-inspired-validation-report-v1', status='COMPLETE',
                  campaign_id=suite['campaign_id'], completed_shifts=1, optimizer_updates=0,
                  method=phase['method'], receipts=[ref], shifts=[receipt],
                  engineering_acceptance=receipt['engineering_acceptance'])
    published = smoke._publish_json(report_path, report)
    print(f"COMPLETE run_index={job['run_index']} policy={POLICY}", flush=True)
    if progress is not None:
        progress(1, job)
    return published
