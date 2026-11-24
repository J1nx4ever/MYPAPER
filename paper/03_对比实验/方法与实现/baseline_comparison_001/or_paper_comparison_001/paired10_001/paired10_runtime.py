"""Collect the ten interleaved MILP pairs with per-slot START verification."""

from pathlib import Path

from paired10_campaign import (
    CAMPAIGN, CONTROL, EXECUTION, SCHEMA, _checked_reference, _require, load_reference, reference,
)
from baseline1_v1.collection import wait_for_java_start
from milp_smoke_runtime import _collect, make_policy
from python.dispatch_bridge import DispatchCallbackBridge
from python.dispatch_env import DispatchEnv
from python.dispatch_service import create_server
from python.episode_runner import ParameterVariationEpisodeRunner
from traditional_dispatch_v1.runtime import _publish_json, _publish_ready


def _paired_anchor(suite, ledger, job):
    if job['policy_kind'] == 'rolling_quota':
        return None, None
    first = suite['jobs'][job['ordinal'] - 1]
    _require(first['policy_kind'] == 'rolling_quota'
             and first['pair_ordinal'] == job['pair_ordinal']
             and first['experiment_seed'] == job['experiment_seed'],
             'paired quota identity differs from the current FIFO job')
    _, slots = ledger.snapshot()
    slot = slots[str(first['run_index'])]
    _require(slot['status'] == 'COMPLETE' and slot['identity'] == first,
             'paired quota shift has no settled receipt')
    pinned = slot['evidence']
    anchor = load_reference(pinned)
    _require(anchor['campaign_id'] == CAMPAIGN and anchor['status'] == 'COMPLETE'
             and anchor['run_index'] == first['run_index']
             and anchor['pair_ordinal'] == job['pair_ordinal']
             and anchor['phase_config'] == first['phase_config'],
             'paired quota receipt differs from the frozen slot')
    return anchor, pinned


def run_shift(suite, ledger, job):
    run = job['run_index']
    runner_config = dict(
        campaign_id=CAMPAIGN, control_directory=str(CONTROL),
        master_plan_sha256=suite['master_plan']['sha256'],
    )
    output = Path(job['python_output']) / 'evaluation_receipt.json'
    claim = CONTROL / 'java_claims' / f'run_{run}.once.claim'
    ready = CONTROL / 'ready' / f'run_{run}.properties'
    _require(not any(path.exists() for path in (output, claim, ready, CONTROL / 'STOP.properties')),
             'run identity already has physical evidence')
    anchor, anchor_ref = _paired_anchor(suite, ledger, job)
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
        _publish_ready(runner_config, job)
        wait_for_java_start(runner_config, job, job['phase_config']['sha256'], timeout)
        result = _collect(env, job, suite['method'], policy, anchor=anchor)
        _checked_reference(suite['protected_core'])
        fields = ('run_index', 'experiment_seed', 'recording_id', 'actor_sampling_seed',
                  'policy_kind', 'scene_id', 'phase_config', 'pair_ordinal')
        receipt = dict(
            result, schema=SCHEMA + '-receipt', campaign_id=CAMPAIGN, status='COMPLETE',
            normal_completion=True, completed_shifts=1, optimizer_updates=0,
            java_start_claim=reference(claim), paired_anchor=anchor_ref,
            **{key: job[key] for key in fields},
        )
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
