"""Run four admitted AnyLogic shifts and seal only per-shift evidence."""

from collections import Counter
import hashlib
import math
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile
import time

import numpy as np

from baseline1_v1.collection import wait_for_java_start
from baseline1_v1.java_plan import COLUMNS
from business_graph_summary_campaign_v1.plan_io import encoded, load_reference, reference
from python.collect_offline import validate_runtime
from python.constrained_adapter import RewardCostAdapter
from python.dispatch_bridge import DispatchCallbackBridge
from python.dispatch_env import DispatchEnv
from python.dispatch_service import create_server
from python.episode_runner import ParameterVariationEpisodeRunner
from python.observation_contract import FIELD_NAMES
from python.reward_config import configuration_from_components
from python.runtime_scenario import PREFIX as SCENE_PREFIX, fingerprint
from single_critic_task_query_v1.targets import (
    ORE_TARGET_TONS, ORE_WEIGHT, WAIT_REFERENCE_SECONDS, WAIT_WEIGHT,
    WASTE_TARGET_TONS, WASTE_WEIGHT, grade_cost,
)
from task_query_cmdp_v1.runtime import validate_snapshot


POLICIES = ('unload_greedy', 'fifo_greedy', 'rolling', 'bg128')
INSTRUMENTATION = 'boundary-events-v1-native-rpc-runtime-scene-v1'
RECEIPT_SCHEMA = 'bg128-traditional-smoke-evaluation-receipt-v1'
MATERIAL_FIELDS = ('isOre', 'meanGrade', 'gradeHalfRange')
MATERIAL_RPC_FIELDS = ('isOre', 'grade', 'gradeHalfRange')
MATERIAL_INDICES = [FIELD_NAMES.index(f'load.{site}.{field}')
                    for site in range(4) for field in MATERIAL_FIELDS]
WAIT_FIELDS = ('breakdown_version', 'load_seconds', 'counted_load_seconds',
               'dump_total_seconds', 'dump_resource_queue_seconds',
               'dump_capacity_admission_seconds', 'load_reconciliation_residual_seconds',
               'dump_reconciliation_residual_seconds', 'total_reconciliation_residual_seconds',
               'unclassified_flagged_seconds', 'stage_errors', 'load_fallback_hits')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _publish_bytes(path, contents):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=f'.{path.name}.') as temporary:
        temporary.write(contents)
        temporary.flush()
        os.fsync(temporary.fileno())
        os.link(temporary.name, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return reference(path)


def _publish_json(path, value):
    return _publish_bytes(path, encoded(value))


def _checked_reference(ref):
    _require(reference(ref['path']) == ref, f"frozen bytes changed: {ref['path']}")


def _publish_ready(startup, job):
    config = job['phase_config']
    _checked_reference(config)
    control = Path(startup['control_directory'])
    _require(not (control / 'STOP.properties').exists(), 'campaign STOP is present')
    values = {key: str(job[key]) for key in COLUMNS}
    values.update(protocol='baseline1-one-click-ready-v1',
                  instrumentation_revision=INSTRUMENTATION,
                  campaign_id=startup['campaign_id'],
                  master_plan_sha256=startup['master_plan_sha256'],
                  config_sha256=config['sha256'], snapshot_sha256=config['sha256'],
                  snapshot_path=config['path'])
    path = control / 'ready' / f"run_{job['run_index']}.properties"
    _publish_bytes(path, ''.join(f'{key}={value}\n' for key, value in values.items()).encode('ascii'))


def _check_startup(suite):
    for key in ('master_plan', 'protected_core', 'actor', 'actor_manifest', 'checkpoint',
                'checkpoint_manifest', 'checkpoint_seal'):
        _checked_reference(suite[key])


def _load_actor(suite):
    """Bind the sealed inference artifact to the same frozen u32 checkpoint."""
    from business_graph_summary_v1.admission import frozen_actor

    checkpoint = load_reference(suite['checkpoint_manifest'])
    seal = load_reference(suite['checkpoint_seal'])
    _require(seal == dict(sha256=suite['checkpoint']['sha256'],
                          manifest_sha256=suite['checkpoint_manifest']['sha256'])
             and checkpoint['sha256'] == suite['checkpoint']['sha256']
             and checkpoint['schema'] == 'business-graph-summary-ppo-checkpoint-v1'
             and checkpoint['update_index'] == 32 and checkpoint['candidate_id'] == 's01_BG128',
             'u32 checkpoint and its original seal differ')
    actor, digest = frozen_actor(suite['actor']['path'])
    actor_manifest = load_reference(suite['actor_manifest'])
    shared = ('candidate_id', 'execution_profile', 'config_sha256', 'actor_digest', 'update_index')
    _require(all(actor_manifest[key] == checkpoint[key] for key in shared)
             and actor_manifest['sha256'] == suite['actor']['sha256']
             and suite['actor_manifest']['path'] == suite['actor']['path'] + '.manifest.json'
             and digest == checkpoint['actor_digest']
             and actor._artifact_config == suite['learner_config'],
             'frozen Actor does not belong to the specified frozen u32')
    return actor, digest


def _actor_choice(actor, generator, observation):
    import torch
    from single_critic_task_query_v1.features_cache import pack_summary_prefixes

    states = observation[None, :]
    summary = pack_summary_prefixes(states, feature_config=actor.feature_config)
    with torch.inference_mode():
        distribution = actor.distribution(torch.as_tensor(states.copy()), summary)
        return int(torch.multinomial(distribution.probs, 1, replacement=True,
                                     generator=generator).item())


def _check_rpc(observation, info, job, material, *, terminal):
    validate_runtime(info, job)
    components = info['reward_components']
    _require(components.get('config.material_event_memory_enabled') == 0.0
             and components.get('config.material_event_contract_version') == 1.0
             and components.get('config.wait_breakdown_enabled') == 1.0
             and components.get('config.wait_breakdown_version') == 1.0,
             'RPC low-frequency instrumentation differs')
    _require(not any(key.startswith('config.offline_native.') for key in components)
             and components.get('audit.p0.enabled', 0.0) == 0.0
             and components.get('config.eval.fallback_enabled', 0.0) == 0.0
             and components.get('config.eval.fallback_used', 0.0) == 0.0,
             'unexpected Java fallback, native collection or full recording')
    _require(terminal or not any(key.startswith('diag.wait.') for key in components),
             'wait breakdown is terminal-only')
    _require(np.array_equal(observation[MATERIAL_INDICES], material),
             'observation material attributes differ from the frozen scene')
    validate_snapshot(observation, components, terminal=terminal)


def _wait_breakdown(components):
    fields = {key.removeprefix('diag.wait.'): value for key, value in components.items()
              if key.startswith('diag.wait.')}
    _require(set(fields) == set(WAIT_FIELDS) and all(math.isfinite(v) for v in fields.values()),
             'missing, unknown or nonfinite terminal wait breakdown fields')
    _require(fields['breakdown_version'] == 1.0 and fields['stage_errors'] == 0.0
             and fields['load_fallback_hits'] == 0.0, 'invalid wait breakdown version or stage counter')
    load = fields['load_seconds']
    counted_load = fields['counted_load_seconds']
    dump = fields['dump_total_seconds']
    resource = fields['dump_resource_queue_seconds']
    capacity = fields['dump_capacity_admission_seconds']
    total = float(components['wait_vehicle_seconds'])
    _require(min(load, counted_load, dump, resource, capacity, total) >= 0.0,
             'negative cumulative wait in terminal RPC')
    tolerance = max(1e-6, 1e-9 * max(1.0, total))
    residuals = (load - counted_load, dump - resource - capacity, total - load - dump,
                 total - counted_load - resource - capacity)
    for field, actual in zip(WAIT_FIELDS[6:10], residuals, strict=True):
        _require(abs(actual) <= tolerance and abs(actual - fields[field]) <= tolerance,
                 f'waiting reconciliation differs: {field}')
    return fields


def _physical(adapter, terminal_info):
    components = terminal_info['reward_components']
    windows = []
    epsilon = adapter.config.get('grade_epsilon')
    lower = adapter.config.get('grade_lower')
    upper = adapter.config.get('grade_upper')
    for window in adapter.windows:
        tons = float(window['ore_tons'])
        metal = float(window['metal_tons'])
        grade = float(window['grade']) if tons > 0.0 else None
        compliant = grade is not None and lower - epsilon <= grade <= upper + epsilon
        windows.append(dict(window=window['window_index'], tons=tons, metal_tons=metal,
                            grade=grade, empty=grade is None, compliant=compliant,
                            continuous_cost=grade_cost(grade, empty=grade is None)))
    ore = float(components['ore_tons_shift'])
    waste = float(components['waste_tons_shift'])
    waiting = float(components['wait_vehicle_seconds'])
    ore_met = ore >= ORE_TARGET_TONS
    waste_met = waste >= WASTE_TARGET_TONS
    grade_met = all(window['compliant'] for window in windows)
    grade_total = math.fsum(window['continuous_cost'] for window in windows)
    production = (ORE_WEIGHT * max(0., 1. - ore / ORE_TARGET_TONS)
                  + WASTE_WEIGHT * max(0., 1. - waste / WASTE_TARGET_TONS))
    waiting_cost = WAIT_WEIGHT * waiting / WAIT_REFERENCE_SECONDS
    return dict(windows=windows, ore_tons=ore, waste_tons=waste, total_wait_seconds=waiting,
                wait_breakdown=_wait_breakdown(components), ore_target_met=ore_met,
                waste_target_met=waste_met, all_four_windows_compliant=grade_met,
                complete_task=ore_met and waste_met and grade_met,
                grade_cost=grade_total, production_cost=production, waiting_cost=waiting_cost,
                combined_cost=math.fsum((grade_total, production, waiting_cost)),
                legacy_rpc_reward=adapter.total_reward)


def _percentiles(values):
    return dict(zip(('p50', 'p95', 'p99', 'max'),
                    (*np.percentile(values, (50, 95, 99)).tolist(), max(values)), strict=True))


def _count_choice(result, mask, counts, fallbacks):
    _require(result.kind in ('single_legal', 'planned', 'fallback'), 'unknown decision outcome')
    counts[result.kind] += 1
    if result.kind == 'single_legal':
        _require(sum(mask) == 1, 'single-action shortcut admitted a multi-action request')
    if result.kind == 'planned':
        _require(result.completed_candidates == sum(mask), 'planned action lacks full baseline comparison')
    if result.kind == 'fallback':
        _require(bool(result.fallback_reason), 'fallback needs a specific cause')
        fallbacks[result.fallback_reason] += 1
    for name in ('completed_candidates', 'search_nodes', 'completed_plans', 'multi_step_plans',
                 'cross_window_candidates', 'invalid_branches'):
        value = getattr(result, name)
        _require(type(value) is int and value >= 0, f'invalid strategy count: {name}')
        counts[name] += value
    counts['timed_out'] += int(result.timed_out)
    counts['tree_complete'] += int(result.tree_complete)
    depth = result.max_scored_depth
    _require(type(depth) is int and 0 <= depth <= 3, 'invalid scored decision depth')
    counts[f'scored_depth_{depth}_requests'] += 1
    counts['max_scored_depth'] = max(counts['max_scored_depth'], depth)
    if result.selected_cost is not None:
        _require(result.initial_cost is not None and math.isfinite(result.initial_cost)
                 and math.isfinite(result.selected_cost)
                 and result.selected_cost <= result.initial_cost + 1e-12,
                 'selected nominal plan worsens the incumbent')
        counts['nominal_improvements'] += int(result.selected_cost < result.initial_cost - 1e-12)
        return result.initial_cost - result.selected_cost
    return None


def _collect(env, job, method, first_state, actor=None):
    import torch
    from .decision import choose

    obs, info = env.reset()
    rpc_config = {key: value for key, value in info['reward_components'].items()
                  if key.startswith('config.')}
    scene = {key: value for key, value in rpc_config.items() if key.startswith(SCENE_PREFIX)}
    _require(fingerprint(scene) == job['runtime_scene_sha256'], 'actual scene digest differs')
    material = np.array([scene[f'{SCENE_PREFIX}load.{site}.{field}']
                         for site in range(4) for field in MATERIAL_RPC_FIELDS], dtype=np.float32)
    _check_rpc(obs, info, job, material, terminal=False)
    adapter = RewardCostAdapter(configuration_from_components(info['reward_components']))
    adapter.start(info)
    initial_state = dict(observation_sha256=hashlib.sha256(obs.tobytes()).hexdigest(),
                         mask=info['action_mask'], truck_id=info['truck_id'],
                         sim_time_seconds=info['sim_time_seconds'])
    _require(first_state is None or first_state == initial_state, 'paired initial observation differs')
    counts, fallbacks = Counter(), Counter()
    actions = [0, 0, 0, 0]
    policy_ms, response_ms, improvements = [], [], []
    generator = None
    rng_start = None
    if actor is not None:
        generator = torch.Generator(device='cpu').manual_seed(job['actor_sampling_seed'])
        rng_start = hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest()
    while True:
        mask = info['action_mask']
        ticket = env._pending
        _require(ticket is not None, 'active decision has no transport ticket')
        started = time.perf_counter_ns()
        if actor is None:
            result = choose(job['policy_kind'], obs, mask, planning_ms=method['planning_ms'], depth=method['depth'])
            action = result.action
            improvement = _count_choice(result, mask, counts, fallbacks)
            if improvement is not None:
                improvements.append(improvement)
        else:
            action = _actor_choice(actor, generator, obs)
            counts['sampled'] += 1
        policy_ms.append((time.perf_counter_ns() - started) / 1e6)
        _require(type(action) is int and 0 <= action < len(mask) and bool(mask[action]),
                 'strategy returned an action outside the live mask')
        actions[action] += 1
        obs, _, terminated, truncated, info = env.step(action)
        response_ms.append(ticket.response.latency_us / 1000.)
        _check_rpc(obs, info, job, material, terminal=terminated)
        adapter.transition(info, terminated=terminated, truncated=truncated)
        if terminated:
            break
    decisions = sum(actions)
    _require(decisions > 0 and decisions == info['decision_sequence'] and decisions == adapter.sequence
             and decisions == sum(counts[k] for k in ('single_legal', 'planned', 'fallback', 'sampled')),
             'decision, action and terminal counters differ')
    if job['policy_kind'] == 'rolling':
        _require(counts['multi_step_plans'] > 0, 'rolling never completed a multi-step trial')
    if job['policy_kind'] in ('unload_greedy', 'fifo_greedy'):
        _require(counts['planned'] > 0, 'greedy never compared multiple legal actions')
    target = method['response_target_ms']
    engineering = dict(decisions=decisions, dispatches_by_site=actions, action_counts=dict(counts),
                       fallback_reasons=dict(fallbacks), fallback_rate=counts['fallback'] / decisions,
                       policy_latency_ms=_percentiles(policy_ms),
                       rpc_receive_to_response_ms=_percentiles(response_ms),
                       rpc_response_target_ms=target,
                       rpc_responses_over_target=sum(elapsed > target for elapsed in response_ms))
    if improvements:
        engineering['nominal_cost_reduction'] = dict(n=len(improvements),
                                                     mean=math.fsum(improvements) / len(improvements),
                                                     max=max(improvements))
    if generator is not None:
        rng_end = hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest()
        engineering['actor_sampling_rng'] = dict(initial_sha256=rng_start, final_sha256=rng_end,
                                                  draws=counts['sampled'])
    return dict(physical=_physical(adapter, info), engineering=engineering,
                initial_state=initial_state, actual_rpc_configuration=rpc_config,
                episode_id=info['episode_id'], terminal_request_id=info['request_id'],
                terminal_time_seconds=info['sim_time_seconds'])


def _run_shift(suite, startup, ledger, job, first_state, actor):
    phase = load_reference(job['phase_config'])
    _require(phase['schema'] == 'bg128-traditional-smoke-phase-v1'
             and phase['job'] == {key: value for key, value in job.items() if key != 'phase_config'}
             and phase['policy_kind'] == job['policy_kind'] and phase['execution'] == suite['execution'],
             'phase config and frozen job binding differ')
    if actor is not None:
        _require(all(phase[key] == suite[key] for key in ('actor', 'checkpoint', 'checkpoint_manifest', 'checkpoint_seal')),
                 'phase Actor or checkpoint references changed')
    settings = suite['execution']
    bridge = DispatchCallbackBridge(expected_episode_metadata=[(job['run_index'], job['experiment_seed'])])
    env = DispatchEnv(bridge, ParameterVariationEpisodeRunner(1), settings['callback_timeout_seconds'])
    server = create_server(bridge, host=settings['host'], port=settings['port'])
    claimed = False
    try:
        server.start()
        ledger.claim(job)
        claimed = True
        _publish_ready(startup, job)
        wait_for_java_start(startup, job, job['phase_config']['sha256'],
                            settings['callback_timeout_seconds'])
        result = _collect(env, job, phase['method'], first_state, actor)
        server.stop(grace=2).wait()
        claim = Path(startup['control_directory']) / 'java_claims' / f"run_{job['run_index']}.once.claim"
        result['java_start_claim'] = reference(claim)
        return result
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


def run(suite, ledger):
    """Run the frozen four-slot sequence. The launch entry owns ledger registration."""
    import torch
    from business_graph_summary_v1.models import digest

    control = Path(suite['control_directory'])
    startup = dict(control_directory=str(control), campaign_id=suite['campaign_id'],
                   master_plan_sha256=suite['master_plan']['sha256'])
    try:
        _require(tuple(job['policy_kind'] for job in suite['jobs']) == POLICIES,
                 'one frozen shift per strategy is required')
        _check_startup(suite)
        torch.set_num_threads(suite.get('cpu_threads', 2))
        actor, actor_digest = _load_actor(suite)
        receipts, first_state = [], None
        for job in suite['jobs']:
            result = _run_shift(suite, startup, ledger, job, first_state,
                                actor if job['policy_kind'] == 'bg128' else None)
            try:
                _check_startup(suite)
                _require(digest(actor) == actor_digest and not actor.training
                         and all(not p.requires_grad for p in actor.parameters()),
                         'frozen Actor changed during independent inference')
                first_state = result['initial_state']
                receipt = dict(result, schema=RECEIPT_SCHEMA, status='COMPLETE',
                               normal_completion=True, optimizer_updates=0, completed_shifts=1,
                               run_index=job['run_index'], experiment_seed=job['experiment_seed'],
                               actor_sampling_seed=job['actor_sampling_seed'], recording_id=job['recording_id'],
                               policy_kind=job['policy_kind'], phase_config=job['phase_config'])
                if job['policy_kind'] == 'bg128':
                    receipt.update(actor=suite['actor'], checkpoint=suite['checkpoint'], actor_digest=actor_digest)
                ref = _publish_json(Path(job['python_output']) / 'evaluation_receipt.json', receipt)
                ledger.complete(job, ref)
                receipts.append(ref)
                print(f"COMPLETE run_index={job['run_index']} policy={job['policy_kind']}", flush=True)
            except BaseException as error:
                try:
                    ledger.mark_unknown(job, repr(error))
                except Exception as ledger_error:
                    print(f'ledger failure: {ledger_error!r}', file=sys.stderr, flush=True)
                raise
        report = dict(schema='bg128-traditional-smoke-report-v1', status='COMPLETE',
                      campaign_id=suite['campaign_id'], completed_shifts=len(receipts), optimizer_updates=0,
                      receipts=receipts, shifts=[load_reference(ref) for ref in receipts])
        _publish_json(Path(suite['report_directory']) / 'single_shift_validation.json', report)
        return report
    except BaseException as error:
        for path, value in ((control / 'STOP.properties', b'status=STOP\n'),
                            (Path(suite['python_output']) / 'error.json', encoded(dict(error=repr(error))))):
            try:
                _publish_bytes(path, value)
            except FileExistsError:
                pass
            except OSError as publication_error:
                print(f'error evidence could not be written: {publication_error!r}', file=sys.stderr, flush=True)
        raise
