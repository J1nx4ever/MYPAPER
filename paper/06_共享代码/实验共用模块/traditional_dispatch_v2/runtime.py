"""Low-frequency shared runtime for the five dispatch policies."""

from collections import Counter
import hashlib
import math
from pathlib import Path
import sys
from time import perf_counter_ns

import numpy as np
import torch

from baseline1_v1.collection import wait_for_java_start
from business_graph_summary_campaign_v1.plan_io import load_reference, reference
from business_graph_summary_v1.models import digest
from python.constrained_adapter import RewardCostAdapter
from python.dispatch_bridge import DispatchCallbackBridge
from python.dispatch_env import DispatchEnv
from python.dispatch_service import create_server
from python.episode_runner import ParameterVariationEpisodeRunner
from python.reward_config import configuration_from_components
from python.runtime_scenario import PREFIX, fingerprint
from traditional_dispatch_v1 import runtime as shared


POLICIES = ('unload_greedy', 'fifo_greedy', 'rolling', 'dispatch_inspired', 'bg128')
METHOD_DEFAULTS = dict(planning_ms=800, response_target_ms=1000, depth=3, match_trucks=3, replan_seconds=900)
SUITE_SCHEMA = 'bg128-traditional-audit-v2'
PHASE_SCHEMA = 'bg128-traditional-audit-phase-v2'
RECEIPT_SCHEMA = 'bg128-traditional-audit-receipt-v2'
REPORT_FILENAME = 'single_shift_validation.json'
FLOW_FIELDS = ('flow_shortfall_ore_tons', 'flow_shortfall_waste_tons', 'flow_grade_slack_tons')
NATIVE_FALLBACK_FIELDS = ('config.eval.fallback_enabled', 'config.eval.fallback_used')


def _verify_anchor(record, anchor):
    if anchor is not None:
        keys = ('experiment_seed', 'initial_state', 'actual_rpc_configuration')
        shared._require(all(record[key] == anchor[key] for key in keys),
                        'environment seed, initial observation or actual RPC configuration differs within the pair')


def _make_dispatch(method):
    from .dispatch_policy import DispatchPolicy

    return DispatchPolicy(planning_ms=method['planning_ms'], replan_seconds=method['replan_seconds'],
                          match_trucks=method['match_trucks'])


def _native_fallback_telemetry(info):
    components = info['reward_components']
    return all(field in components for field in NATIVE_FALLBACK_FIELDS)


def _choose(policy, observation, mask, method):
    from .decision import choose

    options = {key: method[key] for key in ('planning_ms', 'depth', 'rollout_limit') if key in method}
    return choose(policy, observation, mask, **options)


def _count_dispatch(choice, mask, method, counts, fallbacks, flow_plans):
    shared._require(choice.kind in ('single_legal', 'planned', 'fallback'), 'unknown strategy outcome')
    shared._require(type(choice.replanned) is bool and type(choice.timed_out) is bool,
                    'planning states must be explicit boolean')
    shared._require(type(choice.matched_trucks) is int and 0 <= choice.matched_trucks <= method['match_trucks']
                    and type(choice.assignments_scored) is int and choice.assignments_scored >= 0,
                    'invalid matching counters')
    counts[choice.kind] += 1
    if choice.kind == 'single_legal':
        shared._require(sum(mask) == 1, 'single-action shortcut requires exactly one legal action')
    if choice.kind == 'fallback':
        shared._require(bool(choice.fallback_reason), 'strategy fallback requires a reason')
        fallbacks[choice.fallback_reason] += 1
    if choice.lp_status is not None:
        shared._require(choice.lp_status in ('optimal', 'feasible_limit', 'failed'), 'unknown LP status')
        counts[f'lp_{choice.lp_status}'] += 1
    if choice.replanned:
        shared._require(choice.lp_status in ('optimal', 'feasible_limit'), 'replan lacks feasible solver status')
        flow = {key: getattr(choice, key) for key in FLOW_FIELDS}
        shared._require(all(value is not None and math.isfinite(value) and value >= 0 for value in flow.values()),
                        'replan lacks finite nonnegative slack evidence')
        flow_plans.append(flow)
        counts['flow_plan_successes'] += 1
    counts['timed_out'] += int(choice.timed_out)
    counts['assignments_scored'] += choice.assignments_scored
    matched = choice.kind == 'planned' and choice.matched_trucks > 1 and choice.assignments_scored > 0
    counts['live_multitruck_matches'] += int(matched)
    counts['max_matched_trucks'] = max(counts['max_matched_trucks'], choice.matched_trucks)
    counts[f'matched_{choice.matched_trucks}_trucks_requests'] += 1
    return matched


def _collect(env, job, method=None, anchor=None, actor=None):
    """Collect only verified shift/window aggregates from a live dispatch environment."""
    method = dict(METHOD_DEFAULTS, **(method or {}))
    kind = job['policy_kind']
    seed = job['actor_sampling_seed']
    shared._require(kind in POLICIES and ((actor is not None and seed > 0) if kind == 'bg128'
                                        else actor is None and seed == 0),
                    'policy selector and Actor sampling identity differ')
    dispatch = _make_dispatch(method) if kind == 'dispatch_inspired' else None
    generator = None
    if actor is not None:
        generator = torch.Generator(device='cpu').manual_seed(seed)
        rng_start = hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest()

    observation, info = env.reset()
    configuration = {key: value for key, value in info['reward_components'].items() if key.startswith('config.')}
    scene = {key: value for key, value in configuration.items() if key.startswith(PREFIX)}
    shared._require(fingerprint(scene) == job['runtime_scene_sha256'], 'actual scene digest differs')
    shared._require((info['experiment_run_index'], info['experiment_seed'])
                    == (job['run_index'], job['experiment_seed']), 'START run/seed differs from the frozen job')
    material = np.array([scene[f'{PREFIX}load.{site}.{field}'] for site in range(4)
                         for field in shared.MATERIAL_RPC_FIELDS], dtype=np.float32)
    shared._check_rpc(observation, info, job, material, terminal=False)
    rewards = RewardCostAdapter(configuration_from_components(info['reward_components']))
    rewards.start(info)
    initial = dict(observation_sha256=hashlib.sha256(observation.tobytes()).hexdigest(),
                   mask=info['action_mask'], truck_id=info['truck_id'], sim_time_seconds=info['sim_time_seconds'])
    start = dict(experiment_seed=job['experiment_seed'], initial_state=initial,
                 actual_rpc_configuration=configuration)
    _verify_anchor(start, anchor)

    counts, fallbacks = Counter(), Counter()
    dispatches = [0] * 4
    policy_ms, response_ms, improvements, flow_plans = [], [], [], []
    native_telemetry_callbacks = int(_native_fallback_telemetry(info))
    witness = None
    while True:
        mask = info['action_mask']
        ticket = env._pending
        shared._require(ticket is not None, 'active decision has no transport ticket')
        started = perf_counter_ns()
        if actor is not None:
            action = shared._actor_choice(actor, generator, observation)
        else:
            choice = dispatch.choose(observation, mask) if dispatch else _choose(kind, observation, mask, method)
            action = choice.action
        policy_ms.append((perf_counter_ns() - started) / 1e6)
        shared._require(type(action) is int and 0 <= action < len(mask) and bool(mask[action]),
                        'strategy selected an action outside the live mask')

        prior = info
        # The bridge checks previous_action against this response before accepting
        # the next callback, including a terminal callback for the last action.
        observation, _, terminated, truncated, info = env.step(action)
        response_ms.append(ticket.response.latency_us / 1000.)
        shared._check_rpc(observation, info, job, material, terminal=terminated)
        rewards.transition(info, terminated=terminated, truncated=truncated)
        native_telemetry_callbacks += int(_native_fallback_telemetry(info))
        if actor is not None:
            counts['sampled'] += 1
        elif dispatch:
            dispatch.commit(choice)
            if _count_dispatch(choice, mask, method, counts, fallbacks, flow_plans) and witness is None:
                witness = dict(decision_sequence=prior['decision_sequence'], sim_time_seconds=prior['sim_time_seconds'],
                               truck_id=prior['truck_id'], action=action, mask=mask,
                               matched_trucks=choice.matched_trucks, assignments_scored=choice.assignments_scored)
        else:
            improvement = shared._count_choice(choice, mask, counts, fallbacks)
            shared._require(type(choice.budget_exhausted) is bool, 'work limit state must be explicit boolean')
            counts['work_budget_exhausted'] += int(choice.budget_exhausted)
            if kind == 'rolling':
                counts['retained_multi_step_requests'] += int(choice.kind == 'planned' and not choice.timed_out
                                                               and choice.multi_step_plans > 0)
                counts['wall_guard_restored_incumbent'] += int(choice.kind == 'planned' and choice.timed_out)
            if improvement is not None:
                improvements.append(improvement)
        dispatches[action] += 1
        if terminated:
            break

    decisions = sum(dispatches)
    shared._require(decisions > 0 and decisions == info['decision_sequence'] == rewards.sequence
                    and decisions == sum(counts[key] for key in ('single_legal', 'planned', 'fallback', 'sampled')),
                    'decision, applied action and terminal counts differ')
    target = method['response_target_ms']
    over_target = sum(elapsed > target for elapsed in response_ms)
    engineering = dict(decisions=decisions, dispatches_by_site=dispatches, action_counts=dict(counts),
                       fallback_reasons=dict(fallbacks), fallback_rate=counts['fallback'] / decisions,
                       policy_latency_ms=shared._percentiles(policy_ms),
                       rpc_receive_to_response_ms=shared._percentiles(response_ms),
                       rpc_response_target_ms=target, rpc_responses_over_target=over_target,
                       native_fallback_telemetry=dict(total_callbacks=decisions + 1,
                                                      callbacks_with_both_flags=native_telemetry_callbacks))
    acceptance = dict(rpc_and_physical_completion=True, met_response_target=over_target == 0)
    if kind == 'dispatch_inspired':
        engineering['first_live_multitruck_match'] = witness
        acceptance.update(flow_plan_observed=bool(flow_plans), live_multitruck_match_observed=witness is not None)
    elif kind == 'rolling':
        acceptance['multi_step_trial_observed'] = counts['retained_multi_step_requests'] > 0
    elif kind != 'bg128':
        acceptance['candidate_comparison_observed'] = counts['planned'] > 0
    if flow_plans:
        engineering['accepted_flow_plan_slack'] = {
            key: dict(n=len(flow_plans), mean=math.fsum(plan[key] for plan in flow_plans) / len(flow_plans),
                      max=max(plan[key] for plan in flow_plans)) for key in FLOW_FIELDS}
    if improvements:
        engineering['nominal_cost_reduction'] = dict(n=len(improvements),
            mean=math.fsum(improvements) / len(improvements), max=max(improvements))
    if generator is not None:
        rng_end = hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest()
        engineering['actor_sampling_rng'] = dict(initial_sha256=rng_start, final_sha256=rng_end,
                                                  draws=counts['sampled'])
    acceptance['passed'] = all(acceptance.values())
    return dict(start, physical=shared._physical(rewards, info), engineering=engineering,
                engineering_acceptance=acceptance, episode_id=info['episode_id'],
                terminal_request_id=info['request_id'], terminal_time_seconds=info['sim_time_seconds'])


def _phase(suite, job):
    phase = load_reference(job['phase_config'])
    shared._require(phase['schema'] == PHASE_SCHEMA and phase['campaign_id'] == suite['campaign_id']
                    and phase['job'] == {key: value for key, value in job.items() if key != 'phase_config'}
                    and phase['policy_kind'] == job['policy_kind']
                    and phase['method'] == suite['method'] and phase['execution'] == suite['execution'],
                    'phase configuration and frozen physical identity differ')
    if job['policy_kind'] == 'bg128':
        refs = ('actor', 'checkpoint', 'checkpoint_manifest', 'checkpoint_seal')
        shared._require(all(phase[key] == suite[key] for key in refs), 'BG128 phase artifacts differ')
    return phase


def _run_shift(suite, startup, ledger, job, anchor, actor, actor_digest):
    phase = _phase(suite, job)
    output = Path(job['python_output']) / 'evaluation_receipt.json'
    control = Path(startup['control_directory'])
    claim = control / 'java_claims' / f"run_{job['run_index']}.once.claim"
    ready = control / 'ready' / f"run_{job['run_index']}.properties"
    shared._require(not any(path.exists() for path in (output, claim, ready)),
                    'physical evidence already exists; do not reuse this run identity')
    settings = suite['execution']
    bridge = DispatchCallbackBridge(expected_episode_metadata=[(job['run_index'], job['experiment_seed'])])
    env = DispatchEnv(bridge, ParameterVariationEpisodeRunner(1), settings['callback_timeout_seconds'])
    server = create_server(bridge, host=settings['host'], port=settings['port'])
    claimed = False
    try:
        server.start()
        ledger.claim(job)
        claimed = True
        shared._publish_ready(startup, job)
        wait_for_java_start(startup, job, job['phase_config']['sha256'], settings['callback_timeout_seconds'])
        result = _collect(env, job, phase['method'], anchor, actor)
        server.stop(grace=2).wait()
        shared._check_startup(suite)
        fields = ('run_index', 'recording_id', 'experiment_seed', 'actor_sampling_seed', 'policy_kind',
                  'scene_id', 'pair_ordinal', 'phase_config')
        receipt = dict(result, **{key: job[key] for key in fields}, schema=RECEIPT_SCHEMA, status='COMPLETE',
                       normal_completion=True, optimizer_updates=0, completed_shifts=1,
                       java_start_claim=reference(claim))
        if actor is not None:
            shared._require(digest(actor) == actor_digest and not actor.training
                            and all(not parameter.requires_grad for parameter in actor.parameters()),
                            'frozen Actor changed during independent inference')
            receipt.update(actor=suite['actor'], checkpoint=suite['checkpoint'], actor_digest=actor_digest)
        ref = shared._publish_json(output, receipt)
        ledger.complete(job, ref)
        return receipt, ref
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


def run(suite, ledger, *, progress=None):
    """Run the frozen one-slot audit and return its report reference."""
    jobs = suite['jobs']
    shared._require(suite['schema'] == SUITE_SCHEMA and tuple(job['policy_kind'] for job in jobs) == POLICIES
                    and {job['pair_ordinal'] for job in jobs} == {0}
                    and len({job['experiment_seed'] for job in jobs}) == 1,
                    'one frozen five-policy slot is required')
    control = Path(suite['control_directory'])
    report_path = Path(suite['report_directory']) / REPORT_FILENAME
    shared._require(not report_path.exists() and not (control / 'STOP.properties').exists(),
                    'audit output or STOP already exists; reconcile before proceeding')
    slots = ledger.snapshot()['slots']
    shared._require(all(slots[str(job['run_index'])]['status'] == 'PLANNED'
                        and slots[str(job['run_index'])]['identity'] == job for job in jobs),
                    'one fresh audit slot is required; never repeat a claimed physical job')
    shared._check_startup(suite)
    torch.set_num_threads(suite.get('cpu_threads', 2))
    actor, actor_digest = shared._load_actor(suite)
    startup = dict(control_directory=str(control), campaign_id=suite['campaign_id'],
                   master_plan_sha256=suite['master_plan']['sha256'])
    shifts, refs = [], []
    for job in jobs:
        anchor = shifts[0] if shifts else None
        receipt, ref = _run_shift(suite, startup, ledger, job, anchor,
                                  actor if job['policy_kind'] == 'bg128' else None, actor_digest)
        shifts.append(receipt)
        refs.append(ref)
        print(f"COMPLETE run_index={job['run_index']} policy={job['policy_kind']}", flush=True)
        if progress is not None:
            progress(len(shifts), job)
    report = dict(schema='bg128-traditional-audit-report-v2', status='COMPLETE',
                  campaign_id=suite['campaign_id'], method=suite['method'], completed_shifts=len(shifts),
                  optimizer_updates=0, receipts=refs, shifts=shifts,
                  engineering_acceptance={row['policy_kind']: row['engineering_acceptance'] for row in shifts})
    return shared._publish_json(report_path, report)
