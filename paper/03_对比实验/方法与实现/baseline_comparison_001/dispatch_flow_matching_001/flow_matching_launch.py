"""Check or coordinate five physical shifts from the matching AnyLogic Run."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import socket
from statistics import mean
import subprocess

from flow_matching_campaign import (
    CAMPAIGN, CONTROL, CORE, COUNT, EXECUTION, LAUNCHER, LEDGER, LOCKS, METHOD, OUTPUT, POLICY,
    PYTHON, REPORT, ROOT, SCHEMA, SUITE, Ledger, _checked_reference, _publish_bytes, _publish_json,
    _require, check_fresh, encoded, load_reference, load_suite, reference, v3_ledger,
)
from online_v1_1.operations import _publish


LANE = LOCKS / f"port-{EXECUTION['port']}.lock"
AUTHORITY = OUTPUT / 'launch' / 'run_authority.json'
STATUS = OUTPUT / 'run_status.json'
STOP = CONTROL / 'STOP.properties'


def status():
    if STATUS.is_file():
        return json.loads(STATUS.read_bytes())
    return dict(campaign_id=CAMPAIGN, scheduled_shifts=COUNT, completed_shifts=0,
                status='PREPARED_NOT_STARTED' if SUITE.is_file() else 'NOT_PREPARED')


def _set_status(state, **details):
    payload = dict(status=state, campaign_id=CAMPAIGN, scheduled_shifts=COUNT,
                   updated_at_utc=datetime.now(timezone.utc).isoformat(), **details)
    _publish(STATUS, encoded(payload), replace=STATUS.exists())


def _preflight(suite):
    _require(not OUTPUT.exists() and not REPORT.exists(), 'existing activity requires reconciliation')
    identity = json.loads((SUITE.parent / 'launch_identity.json').read_bytes())
    _require(identity['suite'] == reference(SUITE) and identity['jobs'] == suite['jobs']
             and identity['java_helper'] == suite['java_helper'], 'launch card and frozen configuration differ')
    _checked_reference(identity['run_seed'])
    return dict(status='PREFLIGHT_PASS_NO_LISTENER', campaign_id=CAMPAIGN, suite=identity['suite'],
                launch_identity=reference(SUITE.parent / 'launch_identity.json'),
                budget=check_fresh(suite['jobs'], v3_ledger.budget_snapshot()))


def _from_anylogic(suite, identity):
    fields = ('run_index', 'experiment_seed', 'recording_id')
    _require(identity == tuple(suite['jobs'][0][key] for key in fields), 'AnyLogic job zero does not match the frozen roster')
    preflight = _preflight(suite)
    LANE.parent.mkdir(parents=True, exist_ok=True)
    with LANE.open('a+b') as lease:
        fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind((EXECUTION['host'], EXECUTION['port']))
        OUTPUT.mkdir()
        authority = _publish_json(AUTHORITY, dict(schema=SCHEMA + '-authority', suite=preflight['suite'],
            campaign_id=CAMPAIGN, maximum_new_shifts=COUNT, authorized_by_matching_anylogic_run=True,
            first_run=dict(zip(fields, identity, strict=True))))
        fd = lease.fileno()
        command = [PYTHON, '-B', str(LAUNCHER), '--suite', str(SUITE), '--controller', str(fd)]
        _publish_json(OUTPUT / 'launch' / 'intent.json', dict(authority=authority, command=command))
        log = OUTPUT / 'launch' / 'controller.log'
        with log.open('xb', buffering=0) as stream:
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream,
                stderr=subprocess.STDOUT, pass_fds=(fd,), start_new_session=True,
                env=dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1'))
        result = dict(status='CONTROLLER_STARTED', campaign_id=CAMPAIGN, pid=process.pid,
                      log=str(log), rpc_port=EXECUTION['port'])
        _publish_json(OUTPUT / 'launch' / 'receipt.json', result)
    return result


def _run_shift(suite, ledger, job):
    from baseline1_v1.collection import wait_for_java_start
    from change_plans.token_attention_001.business_graph_summary_009.baseline_comparison_001.dispatch_inspired_001.dispatch_runtime import _collect
    from flow_matching_policy import FlowMatchingPolicy
    from python.dispatch_bridge import DispatchCallbackBridge
    from python.dispatch_env import DispatchEnv
    from python.dispatch_service import create_server
    from python.episode_runner import ParameterVariationEpisodeRunner
    from traditional_dispatch_v1.runtime import _publish_ready

    run = job['run_index']
    startup = dict(campaign_id=CAMPAIGN, control_directory=str(CONTROL),
                   master_plan_sha256=suite['master_plan']['sha256'])
    output = Path(job['python_output']) / 'evaluation_receipt.json'
    claim = CONTROL / 'java_claims' / f'run_{run}.once.claim'
    ready = CONTROL / 'ready' / f'run_{run}.properties'
    _require(not any(path.exists() for path in (output, claim, ready, STOP)), 'run identity already has evidence')
    config = load_reference(job['phase_config'])
    method = config['method']
    policy = FlowMatchingPolicy(**{key: method[key] for key in
        ('planning_ms', 'replan_seconds', 'match_trucks', 'soft_replan_seconds')})
    bridge = DispatchCallbackBridge(expected_episode_metadata=[(run, job['experiment_seed'])])
    env = DispatchEnv(bridge, ParameterVariationEpisodeRunner(1), EXECUTION['callback_timeout_seconds'])
    server = create_server(bridge, host=EXECUTION['host'], port=EXECUTION['port'])
    claimed = False
    try:
        server.start()
        ledger.transition(job, 'PLANNED', 'CLAIMED')
        claimed = True
        _publish_ready(startup, job)
        wait_for_java_start(startup, job, job['phase_config']['sha256'], EXECUTION['callback_timeout_seconds'])
        result = _collect(env, job, method, policy)
        result['engineering']['route_matching'] = policy.summary()
        result['engineering_acceptance']['acknowledged_route_dispatches'] = (
            sum(result['engineering']['route_matching']['branch_counts'].values()) == result['engineering']['decisions'])
        gates = result['engineering_acceptance']
        gates['passed'] = all(value for key, value in gates.items() if key != 'passed')
        _checked_reference(suite['protected_core'])
        receipt = dict(result, schema=SCHEMA + '-receipt', campaign_id=CAMPAIGN, status='COMPLETE',
            normal_completion=True, completed_shifts=1, optimizer_updates=0, java_start_claim=reference(claim),
            **{key: job[key] for key in ('run_index', 'experiment_seed', 'actor_sampling_seed', 'recording_id',
                                       'policy_kind', 'scene_id', 'phase_config')})
        ref = _publish_json(output, receipt)
        ledger.transition(job, 'CLAIMED', 'COMPLETE', ref)
        return receipt, ref
    except BaseException as error:
        if claimed:
            ledger.transition(job, 'CLAIMED', 'UNKNOWN', dict(reason=repr(error)))
        raise
    finally:
        env.close()
        server.stop(grace=0).wait()


def _controller(suite, fd):
    _require(os.path.samestat(os.fstat(fd), LANE.stat()), 'inherited lock does not belong to the RPC lane')
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    authority = reference(AUTHORITY)
    intent = json.loads((OUTPUT / 'launch' / 'intent.json').read_bytes())
    _require(not LEDGER.exists() and intent == dict(authority=authority,
        command=[PYTHON, '-B', str(LAUNCHER), '--suite', str(SUITE), '--controller', str(fd)]),
        'launch intent or physical registration differs')
    shifts, references = [], []
    ledger = Ledger(suite)
    try:
        _set_status('STARTING', completed_shifts=0)
        ledger.register(authority)
        for job in suite['jobs']:
            _set_status('RUNNING', completed_shifts=len(shifts), current_run_index=job['run_index'])
            receipt, ref = _run_shift(suite, ledger, job)
            shifts.append(receipt)
            references.append(ref)
            print(f"COMPLETE run_index={job['run_index']} {len(shifts)}/{COUNT}", flush=True)
        _, slots = ledger.snapshot()
        _require(Counter(row['status'] for row in slots.values()) == {'COMPLETE': COUNT}
                 and [row['evidence'] for row in slots.values()] == references, 'physical ledger and receipts differ')
        _checked_reference(suite['protected_core'])
        physical = [shift['physical'] for shift in shifts]
        report = _publish_json(REPORT, dict(schema=SCHEMA + '-report', campaign_id=CAMPAIGN, status='COMPLETE',
            physical_shifts=COUNT, optimizer_updates=0, method=METHOD, protected_core=reference(CORE),
            complete_task_shifts=sum(row['complete_task'] for row in physical),
            engineering_passed_shifts=sum(row['engineering_acceptance']['passed'] for row in shifts),
            mean_combined_cost=mean(row['combined_cost'] for row in physical), receipts=references, shifts=shifts))
        result = dict(status='COMPLETE', campaign_id=CAMPAIGN, physical_shifts=COUNT, completed_shifts=COUNT,
                      training_shifts=0, optimizer_updates=0, suite=reference(SUITE), evaluation=report)
        _publish_json(OUTPUT / 'suite_complete.json', result)
        _set_status('COMPLETE', completed_shifts=COUNT, evaluation=report)
        return result
    except BaseException as error:
        if not STOP.exists():
            reason = str(error).replace('\n', ' ').replace('\r', ' ')
            _publish_bytes(STOP, ('reason=' + reason + '\n').encode('ascii', 'backslashreplace'))
        _set_status('STOP', completed_shifts=len(shifts), reason=repr(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, default=SUITE)
    parser.add_argument('--run-index', type=int)
    parser.add_argument('--environment-seed', type=int)
    parser.add_argument('--recording-id')
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--check', action='store_true')
    modes.add_argument('--status', action='store_true')
    modes.add_argument('--from-anylogic-run', action='store_true')
    modes.add_argument('--controller', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.status:
        result = status()
    else:
        suite = load_suite(args.suite)
        if args.check:
            result = _preflight(suite)
        elif args.from_anylogic_run:
            result = _from_anylogic(suite, (args.run_index, args.environment_seed, args.recording_id))
        else:
            result = _controller(suite, args.controller)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
