"""Prepare, check and run the one-click BG128 speed stress evaluation."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

from speed_plan import (
    AUTHORITY, CAMPAIGN, CONTROL, CORE, LAUNCH, LEDGER, MATERIALS, MIXED_SUITE, OUTPUT,
    PREPARED, ROOT, SLOTS, STD_KEY, SUITE, TOTAL, authorization_for, load_package,
    prepare, read, reference, require,
)
from business_graph_summary_campaign_v1.health import _process_identity
from business_graph_summary_campaign_v1.launch import _exclusive_json, _launch_lock
from business_graph_summary_campaign_v1.plan_io import encoded
from online_v1_1.operations import _publish
from speed_budget import SpeedLedger, check_collisions

PYTHON = '/home/jinx/miniconda3/envs/anylogic-env/bin/python'
ENTRY = Path(__file__).resolve()
INTENT = LAUNCH / 'launch_intent.json'
RECEIPT = LAUNCH / 'launch_receipt.json'
LOG = LAUNCH / 'controller.log'
STATUS = OUTPUT / 'run_status.json'
STOP = CONTROL / 'STOP.properties'


def _intent():
    return dict(
        suite=reference(SUITE), authority=reference(AUTHORITY),
        command=[PYTHON, '-B', str(ENTRY), '--controller'], max_new_shifts=TOTAL,
    )


def _status(value, **fields):
    _publish(STATUS, encoded(dict(
        campaign_id=CAMPAIGN, status=value, scheduled_shifts=TOTAL, training_updates=0,
        optimizer_updates=0, updated_at_utc=datetime.now(timezone.utc).isoformat(), **fields,
    )), replace=True)


def _load_frozen_model(suite):
    from business_graph_summary_v1.models import digest
    from mixed_runtime import _load_new

    checkpoint = suite['parent_artifacts']['checkpoint']
    parent = read(MIXED_SUITE)
    model = _load_new(parent, checkpoint, suite['checkpoint_update'])
    historical = read(suite['historical_nominal']['path'])
    fingerprints = dict(actor=digest(model.actor), critic=digest(model.critic))
    require(fingerprints == historical['fingerprints'] and
            model.config['seed'] == suite['actor_seed'] and
            model.config == suite['learner_config'],
            'loaded model differs from the mixed u32 fixed nominal evaluation')
    return model, fingerprints


def _check_rpc_contract(configs):
    """Check positive and cross-arm rejection using a sealed nominal START as a fixture."""
    from python.collect_offline import validate_runtime
    from speed_plan import PARENT_ROOT

    path = PARENT_ROOT / 'evaluation/nominal/new/episodes/run_10001125/episodes.json'
    nominal = read(path)[0]['start_info']
    fixtures = []
    for cfg in configs:
        job = cfg['jobs'][0]
        components = dict(nominal['reward_components'])
        require(components[STD_KEY] == .05, 'historical START speed setting differs')
        components[STD_KEY] = job['expected_rpc_config'][STD_KEY]
        fixture = dict(nominal, reward_components=components)
        validate_runtime(fixture, job)
        fixtures.append(fixture)
    for fixture, other in zip(fixtures, reversed(configs), strict=True):
        try:
            validate_runtime(fixture, other['jobs'][0])
        except ValueError as error:
            require(STD_KEY in str(error), 'wrong speed-mismatch error')
        else:
            raise ValueError('RPC accepted the other speed arm')


def offline_check(suite, configs):
    import torch
    from baseline1_v1.evaluation import ReadOnlyTrainingGuard
    from java_speed import verify_java

    if LEDGER.is_file():
        SpeedLedger(suite, configs, AUTHORITY).snapshot()
    else:
        check_collisions(suite)
    with ReadOnlyTrainingGuard():
        torch.set_num_threads(suite['cpu_threads'])
        _, fingerprints = _load_frozen_model(suite)
    _check_rpc_contract(configs)
    java = verify_java(PREPARED)
    return dict(status='PASS_OFFLINE_NO_LISTENER', suite=reference(SUITE), paired_slots=SLOTS,
                evaluation_shifts=TOTAL, optimizer_updates=0, fingerprints=fingerprints,
                java=java, nominal_rpc_fixture=True, cross_arm_rpc_rejected=True,
                protected_core=suite['protected_core'])


def start(suite, configs):
    with _launch_lock(LAUNCH):
        require(not any(path.exists() for path in (AUTHORITY, INTENT, RECEIPT, LOG, CONTROL, LEDGER, MATERIALS, STATUS))
                and all(path == LAUNCH for path in OUTPUT.iterdir()), 'previous launch artifacts require reconciliation')
        check_collisions(suite)
        execution = configs[0]['execution']
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind((execution['host'], execution['port']))
        _exclusive_json(AUTHORITY, authorization_for(suite))
        intent = _intent()
        _exclusive_json(INTENT, intent)
        with LOG.open('xb', buffering=0) as stream:
            child = subprocess.Popen(
                intent['command'], cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True,
                env=dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1'),
            )
        process = _process_identity(child.pid)
        _exclusive_json(RECEIPT, dict(suite=intent['suite'], pid=child.pid,
            process_start_ticks=process.get('start_ticks'), log=str(LOG)))
    return dict(status='SPEED_CONGESTION_CONTROLLER_STARTED', pid=child.pid, new_shift_limit=TOTAL)


def _run(suite, configs, ledger):
    import torch
    from baseline1_v1.evaluation import ReadOnlyTrainingGuard
    from business_graph_summary_campaign_v1.runtime import _same_or_save
    from business_graph_summary_evaluation_v1.metrics import build_report, write_reports
    from business_graph_summary_evaluation_v1.runtime import _portable_state_fingerprint
    from business_graph_summary_v1.evaluation import evaluate_episodes
    from grade_only_ppo_v2.runtime import _actor_snapshot
    from mixed_runtime import _collect, _exact_verification, _require_reference
    from recovery_002.storage_scope import bg128_storage
    from speed_report import congestion_evidence, publish_report
    from transport import per_job_transport

    torch.set_num_threads(suite['cpu_threads'])
    checkpoint = suite['parent_artifacts']['checkpoint']
    with ReadOnlyTrainingGuard():
        model, fingerprints = _load_frozen_model(suite)
        actor = _actor_snapshot(model, OUTPUT / 'frozen_actor/actor.pt')
        before = _portable_state_fingerprint(model)
        ledger.register()
        reports = []
        with bg128_storage(), per_job_transport(), ReadOnlyTrainingGuard(model):
            for cfg, phase_ref in zip(configs, suite['phase_configs'], strict=True):
                _status('COLLECTING', scenario_id=cfg['scenario_id'], phase_config=phase_ref)
                with _exact_verification(cfg['jobs']):
                    episodes, sources, actor_ref = _collect(
                        suite, cfg, phase_ref, checkpoint, ledger, model, actor_ref=actor,
                    )
                    report = evaluate_episodes(episodes, model, expected_complete_shifts=SLOTS,
                                               checkpoint_provenance=checkpoint)
                require(actor_ref == actor and report['fingerprints'] == fingerprints and
                        report['optimizer_updates'] == 0 and report['training_state_unchanged'] is True,
                        'fixed policy, Critic or zero-update invariant failed')
                report.update(
                    scenario_id=cfg['scenario_id'], phase_config=phase_ref, actor_artifact=actor_ref,
                    source_refs=sources, model_state_fingerprint=before,
                    congestion_evidence=congestion_evidence(episodes),
                )
                folder = Path(cfg['python_output'])
                _same_or_save(folder / 'evaluation_complete.json', report)
                summary = build_report(report, expected_complete_shifts=SLOTS)
                summary['scenario_id'] = cfg['scenario_id']
                write_reports(summary, report, Path(suite['report_directory']) / cfg['scenario_id'])
                reports.append(report)
        require(before == _portable_state_fingerprint(model)
                and reference(CORE) == suite['protected_core'], 'evaluation changed model state or PPO core bytes')
        for artifact in suite['parent_artifacts'].values():
            _require_reference(artifact)
        state = ledger.resume()
        require(Counter(row['status'] for row in state['jobs'].values()) == {'COMPLETE': TOTAL}
                and not state['campaigns'][CAMPAIGN]['commits'], 'physical ledger or update count differs')
        paired = publish_report(Path(suite['report_directory']), configs, reports)
        _same_or_save(OUTPUT / 'suite_complete.json', dict(
            status='COMPLETE', suite=reference(SUITE), checkpoint=checkpoint, actor=actor,
            fingerprints=fingerprints, evaluation_shifts=TOTAL, optimizer_updates=0,
            paired_report=paired,
        ))
        _status('COMPLETE', completed_shifts=TOTAL, paired_slots=SLOTS)


def controller():
    from grade_only_ppo_v2.runtime import _controller_lock

    with _controller_lock(OUTPUT):
        try:
            suite, configs = load_package()
            require(read(INTENT) == _intent() and read(AUTHORITY) == authorization_for(suite),
                    'launch intent or authority differs from the sealed package')
            require(not STOP.exists() and not LEDGER.exists(), 'previous execution requires reconciliation')
            _status('STARTING')
            _run(suite, configs, SpeedLedger(suite, configs, AUTHORITY))
        except BaseException as error:
            CONTROL.mkdir(parents=True, exist_ok=True)
            if not STOP.exists():
                reason = str(error).replace('\n', ' ').replace('\r', ' ')
                _publish(STOP, ('reason=' + reason + '\n').encode('ascii', 'backslashreplace'))
            _status('STOP', reason=repr(error))
            raise


def read_status(suite, configs):
    value = dict(campaign_id=CAMPAIGN, status='PREPARED_OFFLINE_ONLY', scheduled_shifts=TOTAL)
    if STATUS.is_file():
        value.update(read(STATUS))
    if LEDGER.is_file():
        state = SpeedLedger(suite, configs, AUTHORITY).snapshot()
        value['registered_status_counts'] = dict(Counter(row['status'] for row in state['jobs'].values()))
    progress = OUTPUT / 'training_progress.json'
    if progress.is_file():
        value['progress'] = read(progress)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=lambda value: Path(value).resolve(), default=SUITE)
    parser.add_argument('--run-index', type=int)
    parser.add_argument('--environment-seed', type=int)
    parser.add_argument('--recording-id')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--status', action='store_true')
    mode.add_argument('--from-anylogic-run', action='store_true')
    mode.add_argument('--controller', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    require(args.suite == SUITE, 'unexpected speed campaign suite path')
    if args.prepare:
        result = prepare()
    elif args.controller:
        controller()
        return
    else:
        suite, configs = load_package()
        if args.from_anylogic_run:
            first = suite['jobs'][0]
            require((args.run_index, args.environment_seed, args.recording_id)
                    == (first['run_index'], first['experiment_seed'], first['recording_id']),
                    'only the first frozen AnyLogic run can start this controller')
            result = start(suite, configs)
        elif args.check:
            result = offline_check(suite, configs)
        else:
            result = read_status(suite, configs)
    json.dump(result, sys.stdout, ensure_ascii=False, sort_keys=True)
    print()


if __name__ == '__main__':
    main()
