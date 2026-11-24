"""Run the frozen paired evaluation through the low-frequency AnyLogic contract."""

from collections import Counter
import hashlib
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

from baseline1_v1.collection import wait_for_java_start
from business_graph_summary_campaign_v1.plan_io import encoded, load_reference, reference
from business_graph_summary_v1.models import digest
from python.constrained_adapter import RewardCostAdapter
from python.dispatch_bridge import DispatchCallbackBridge
from python.dispatch_env import DispatchEnv
from python.dispatch_service import create_server
from python.episode_runner import ParameterVariationEpisodeRunner
from python.reward_config import configuration_from_components
from python.runtime_scenario import PREFIX, fingerprint
from traditional_dispatch_v1.decision import choose
from traditional_dispatch_v1 import runtime as smoke


SUITE_SCHEMA = 'bg128-traditional-paired100-v1'
PHASE_SCHEMA = 'bg128-traditional-paired100-phase-v1'
RECEIPT_SCHEMA = 'bg128-traditional-paired100-receipt-v1'
REPORT_FILENAME = 'paired_comparison.json'


def _pair_key(job):
    return job['scene_id'], job['pair_ordinal']


def _verify_anchor(record, anchor):
    if anchor is not None:
        fields = ('experiment_seed', 'initial_state', 'actual_rpc_configuration')
        smoke._require(all(record[key] == anchor[key] for key in fields),
                       'environment seed, initial observation or actual RPC configuration differs within the pair')


def _collect(env, job, method, anchor, actor):
    observation, info = env.reset()
    configuration = {key: value for key, value in info['reward_components'].items()
                     if key.startswith('config.')}
    scene = {key: value for key, value in configuration.items() if key.startswith(PREFIX)}
    smoke._require(fingerprint(scene) == job['runtime_scene_sha256'], 'actual scene digest differs')
    material = np.array([scene[f'{PREFIX}load.{site}.{field}'] for site in range(4)
                         for field in smoke.MATERIAL_RPC_FIELDS], dtype=np.float32)
    smoke._check_rpc(observation, info, job, material, terminal=False)
    rewards = RewardCostAdapter(configuration_from_components(info['reward_components']))
    rewards.start(info)
    initial = dict(observation_sha256=hashlib.sha256(observation.tobytes()).hexdigest(),
                   mask=info['action_mask'], truck_id=info['truck_id'],
                   sim_time_seconds=info['sim_time_seconds'])
    smoke._require((info['experiment_run_index'], info['experiment_seed'])
                   == (job['run_index'], job['experiment_seed']), 'START run/seed differs from the frozen pair')
    start = dict(initial_state=initial, actual_rpc_configuration=configuration,
                 experiment_seed=job['experiment_seed'])
    _verify_anchor(start, anchor)

    generator = None
    if actor is not None:
        generator = torch.Generator(device='cpu').manual_seed(job['actor_sampling_seed'])
        rng_start = hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest()

    counts, fallbacks = Counter(), Counter()
    dispatches = [0] * 4
    policy_ms, response_ms, improvements = [], [], []
    while True:
        mask = info['action_mask']
        ticket = env._pending
        smoke._require(ticket is not None, 'active decision has no transport ticket')
        started = time.perf_counter_ns()
        if actor is None:
            decision = choose(job['policy_kind'], observation, mask, planning_ms=method['planning_ms'],
                              depth=method['depth'])
            action = decision.action
            improvement = smoke._count_choice(decision, mask, counts, fallbacks)
            if improvement is not None:
                improvements.append(improvement)
        else:
            action = smoke._actor_choice(actor, generator, observation)
            counts['sampled'] += 1
        policy_ms.append((time.perf_counter_ns() - started) / 1e6)
        smoke._require(type(action) is int and 0 <= action < len(mask) and mask[action],
                       'strategy returned an action outside the live mask')
        dispatches[action] += 1
        observation, _, terminated, truncated, info = env.step(action)
        response_ms.append(ticket.response.latency_us / 1000.)
        smoke._check_rpc(observation, info, job, material, terminal=terminated)
        rewards.transition(info, terminated=terminated, truncated=truncated)
        if terminated:
            break

    decisions = sum(dispatches)
    smoke._require(decisions > 0 and decisions == info['decision_sequence'] == rewards.sequence
                   and decisions == sum(counts[k] for k in ('single_legal', 'planned', 'fallback', 'sampled')),
                   'decision, action and terminal counters differ')
    target = method['response_target_ms']
    engineering = dict(decisions=decisions, dispatches_by_site=dispatches, action_counts=dict(counts),
                       fallback_reasons=dict(fallbacks), fallback_rate=counts['fallback'] / decisions,
                       policy_latency_ms=smoke._percentiles(policy_ms),
                       rpc_receive_to_response_ms=smoke._percentiles(response_ms),
                       rpc_response_target_ms=target,
                       rpc_responses_over_target=sum(elapsed > target for elapsed in response_ms))
    if improvements:
        engineering['nominal_cost_reduction'] = dict(n=len(improvements),
            mean=math.fsum(improvements) / len(improvements), max=max(improvements))
    if generator is not None:
        rng_end = hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest()
        engineering['actor_sampling_rng'] = dict(initial_sha256=rng_start, final_sha256=rng_end,
                                                  draws=counts['sampled'])
    return dict(start, physical=smoke._physical(rewards, info), engineering=engineering,
                episode_id=info['episode_id'], terminal_request_id=info['request_id'],
                terminal_time_seconds=info['sim_time_seconds'])


def _phase(suite, job):
    phase = load_reference(job['phase_config'])
    identity = {key: value for key, value in job.items() if key != 'phase_config'}
    smoke._require(phase['schema'] == PHASE_SCHEMA and phase['campaign_id'] == suite['campaign_id']
                   and phase['job'] == identity and phase['policy_kind'] == job['policy_kind']
                   and phase['method'] == suite['method'] and phase['execution'] == suite['execution'],
                   'phase config and frozen job binding differ')
    if job['policy_kind'] == 'bg128':
        refs = ('actor', 'checkpoint', 'checkpoint_manifest', 'checkpoint_seal')
        smoke._require(all(phase[key] == suite[key] for key in refs), 'BG128 phase artifacts differ')
    return phase


def _receipt_path(job):
    return Path(job['python_output']) / 'evaluation_receipt.json'


def _claim_path(startup, job):
    return Path(startup['control_directory']) / 'java_claims' / f"run_{job['run_index']}.once.claim"


def _run_shift(suite, startup, ledger, job, anchor, actor, actor_digest):
    phase = _phase(suite, job)
    output = _receipt_path(job)
    claim = _claim_path(startup, job)
    ready = Path(startup['control_directory']) / 'ready' / f"run_{job['run_index']}.properties"
    smoke._require(not any(path.exists() for path in (output, claim, ready)),
                   'planned shift has existing physical evidence; reconcile before proceeding')
    settings = suite['execution']
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
        result = _collect(env, job, phase['method'], anchor, actor)
        server.stop(grace=2).wait()
        identity_fields = ('run_index', 'experiment_seed', 'actor_sampling_seed', 'recording_id',
                           'policy_kind', 'scene_id', 'pair_ordinal', 'phase_config')
        receipt = dict(result, **{key: job[key] for key in identity_fields}, schema=RECEIPT_SCHEMA,
                       status='COMPLETE', normal_completion=True, optimizer_updates=0, completed_shifts=1,
                       java_start_claim=reference(claim))
        if actor is not None:
            smoke._require(digest(actor) == actor_digest and not actor.training
                           and all(not parameter.requires_grad for parameter in actor.parameters()),
                           'frozen Actor changed during independent inference')
            receipt.update(actor=suite['actor'], checkpoint=suite['checkpoint'], actor_digest=actor_digest)
        ref = smoke._publish_json(output, receipt)
        ledger.complete(job, ref)
        return receipt
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


def _read_complete(suite, startup, job, slot, actor_digest):
    ref = slot['receipt']
    smoke._require(Path(ref['path']) == _receipt_path(job), 'completed receipt path differs')
    receipt = load_reference(ref)
    fields = ('run_index', 'experiment_seed', 'actor_sampling_seed', 'recording_id',
              'policy_kind', 'scene_id', 'pair_ordinal', 'phase_config')
    smoke._require(all(receipt[key] == job[key] for key in fields)
                   and receipt['schema'] == RECEIPT_SCHEMA and receipt['normal_completion'] is True
                   and receipt['status'] == 'COMPLETE' and receipt['completed_shifts'] == 1
                   and receipt['optimizer_updates'] == 0 and receipt['terminal_time_seconds'] == 28800,
                   'completed physical receipt differs from its frozen job')
    smoke._require(receipt['java_start_claim'] == reference(_claim_path(startup, job)),
                   'completed physical receipt has different Java start evidence')
    if job['policy_kind'] == 'bg128':
        smoke._require(receipt['actor'] == suite['actor'] and receipt['checkpoint'] == suite['checkpoint']
                       and receipt['actor_digest'] == actor_digest,
                       'completed BG128 receipt uses different frozen artifacts')
    return receipt


def run(suite, ledger, *, progress=None):
    """Evaluate a frozen suite, resuming only an authenticated complete prefix."""
    from paired_report import summarize

    jobs = suite['jobs']
    nslots = suite['evaluation_slots']
    order = tuple((job['pair_ordinal'], job['policy_kind']) for job in jobs)
    smoke._require(suite['schema'] == SUITE_SCHEMA and nslots == 100
                   and order == tuple((pair, policy) for pair in range(nslots) for policy in smoke.POLICIES)
                   and len(jobs) == suite['physical_shifts'], 'paired policy inventory or order differs')
    control = Path(suite['control_directory'])
    smoke._require(not (control / 'STOP.properties').exists(), 'campaign STOP requires reconciliation')
    startup = dict(control_directory=str(control), campaign_id=suite['campaign_id'],
                   master_plan_sha256=suite['master_plan']['sha256'])
    slots = ledger.snapshot()['slots']
    statuses = [slots[str(job['run_index'])]['status'] for job in jobs]
    completed = statuses.count('COMPLETE')
    smoke._require(statuses == ['COMPLETE'] * completed + ['PLANNED'] * (len(jobs) - completed),
                   'only a sealed complete prefix and previously unclaimed jobs may continue')
    smoke._check_startup(suite)
    torch.set_num_threads(suite.get('cpu_threads', 2))
    actor, actor_digest = smoke._load_actor(suite)

    anchors, receipts = {}, []
    for job in jobs:
        slot = slots[str(job['run_index'])]
        smoke._require(slot['identity'] == job, 'physical ledger slot differs from the frozen job')
        pair = _pair_key(job)
        anchor = anchors.get(pair)
        if slot['status'] == 'COMPLETE':
            receipt = _read_complete(suite, startup, job, slot, actor_digest)
            _verify_anchor(receipt, anchor)
        else:
            receipt = _run_shift(suite, startup, ledger, job, anchor,
                                 actor if job['policy_kind'] == 'bg128' else None, actor_digest)
            print(f"COMPLETE run_index={job['run_index']} pair={job['pair_ordinal']} "
                  f"policy={job['policy_kind']}", flush=True)
            if progress is not None:
                progress(len(receipts) + 1, job)
        anchors.setdefault(pair, receipt)
        receipts.append(receipt)

    smoke._check_startup(suite)
    report = summarize(receipts, suite=suite)
    path = Path(suite['report_directory']) / REPORT_FILENAME
    if path.exists():
        smoke._require(path.read_bytes() == encoded(report), 'existing paired report bytes differ')
        return reference(path)
    return smoke._publish_json(path, report)
