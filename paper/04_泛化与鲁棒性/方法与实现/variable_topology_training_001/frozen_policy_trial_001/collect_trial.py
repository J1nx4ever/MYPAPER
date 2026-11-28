"""Settle native trial shifts and reconcile every applied action with the frozen actor."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import sys

from prepare_trial import (CASES, HERE, PROJECT, admitted_witness, check_fresh, load_case,
                           require, witness)

from admit_layouts import check_targets
from python.hdf5_events import iter_event_bundle
from resilience_runtime import record


ERRORS = witness.EVIDENCE_ERRORS
register = witness.bind(witness.ledger.register, check_fresh=check_fresh)
settle = witness.bind(witness.settle, register=register)


def shift_outcomes(evidence, task):
    return dict(
        ore_tons=evidence['ore_tons'], waste_tons=evidence['waste_tons'],
        metal_tons=evidence['metal_tons'], windows=evidence['windows'],
        ore_target_met=evidence['ore_tons'] >= task['ore_target_tons'],
        waste_target_met=task['waste_target_tons'] <= evidence['waste_tons']
        <= task['waste_upper_tons'] + task['tonnage_epsilon'],
        compliant_windows=sum(
            row['ore_tons'] > 0 and task['grade_lower'] - task['grade_epsilon']
            <= row['metal_tons'] / row['ore_tons'] <= task['grade_upper'] + task['grade_epsilon']
            for row in evidence['windows']),
    )


def source_comparison(label, outcomes, task):
    _, source, baseline = admitted_witness(label)
    metrics = shift_outcomes(baseline, task)
    delta = {key: outcomes[key] - metrics[key]
             for key in ('ore_tons', 'waste_tons', 'metal_tons', 'compliant_windows')}
    delta['window_grade_percentage_points'] = [
        100 * (actual['grade'] - reference['grade']) if actual['ore_tons'] > 0 else None
        for actual, reference in zip(outcomes['windows'], metrics['windows'], strict=True)
    ]
    return dict(source_witness=source, source_outcomes=metrics, trial_minus_witness=delta,
                comparison='same physical scenario and deterministic settings, independent run and environment seed')


def assess(label, config, job, finalize_incomplete):
    task = config['task']
    base = {key: job[key] for key in ('layout_id', 'run_index', 'recording_id', 'config_sha256')}
    base.update(physical_status='NOT_STARTED', task_status='UNVERIFIED',
                environment_seed=job['experiment_seed'], manifest=None)
    event_dir = Path(job['event_directory'])
    try:
        identity = witness.native_manifest(event_dir, job)
    except ERRORS as error:
        return dict(base, physical_status='UNVERIFIED', reason=str(error)), None
    if identity is None:
        if event_dir.is_dir() and any(event_dir.iterdir()):
            return dict(base, physical_status='UNVERIFIED', reason='native events have no manifest'), None
        return dict(base, reason='no native event manifest'), None

    path, manifest, digest = identity
    base['manifest'] = {'path': str(path), 'sha256': digest}
    if not manifest['complete']:
        status = 'UNKNOWN' if finalize_incomplete else 'UNVERIFIED'
        return dict(base, physical_status=status, reason='native event manifest is incomplete'), None
    try:
        evidence = witness.replay_event_bundle(
            path.parent, expected_episode_id=manifest['episode_id'],
            expected_config_sha256=job['config_sha256'], expected_recording_id=job['recording_id'],
            expected_source_kind='actual_anylogic')
        require(evidence['run_index'] == job['run_index'] and evidence['seed'] == job['experiment_seed']
                and witness.reference(path) == base['manifest'], 'bound native event identity changed during replay')
    except ERRORS as error:
        return dict(base, physical_status='UNKNOWN', reason=f'native replay failed: {error}'), None

    outcomes = shift_outcomes(evidence, task)
    base.update(physical_status='COMPLETE', events_sha256=evidence['events_sha256'], **outcomes)
    native = evidence['config'].get('actual_runtime_scene')
    if native != config['expected_runtime_scene']:
        base['reason'] = 'runtime road, fleet, equipment, or model differs from the frozen trial'
        return base, None
    base.update(source_comparison(label, outcomes, task))
    try:
        check_targets(evidence, task)
    except ERRORS as error:
        base.update(task_status='NOT_MET', reason=str(error))
    else:
        base['task_status'] = 'MET'
    return base, (path, manifest, evidence)


def policy_audit(config, job, native):
    native_path, manifest, evidence = native
    source = Path(job['python_output']) / 'inference.jsonl'
    seal = source.with_name('inference_manifest.json')
    request_identity = dict(run_index=job['run_index'], recording_id=job['recording_id'],
                            config_sha256=job['config_sha256'],
                            checkpoint_sha256=config['checkpoint_sha256'])
    pin = witness.reference(seal)
    metadata = json.loads(seal.read_bytes())
    require(metadata['complete'] is True and metadata['source_kind'] == 'frozen_actor_http'
            and all(metadata[key] == value for key, value in request_identity.items()),
            'inference manifest does not bind the frozen run or actor')
    raw = source.read_bytes()
    digest = sha256(raw).hexdigest()
    require(raw.endswith(b'\n') and digest == metadata['inference_sha256'],
            'inference log bytes differ from seal')
    inferred = [json.loads(line) for line in raw.splitlines()]
    requests = [request for request in evidence['requests'] if not request['terminal']]
    native_count = manifest['native_policy_decisions']
    require(type(metadata['decision_count']) is int and type(native_count) is int
            and native_count == metadata['decision_count'] == len(requests) == len(inferred) > 0,
            'native and inference decision counts differ')
    require(manifest['native_checkpoint_sha256'] == config['checkpoint_sha256']
            and type(manifest['fallback_dispatches']) is int and manifest['fallback_dispatches'] == 0
            and type(manifest['invalid_dispatches']) is int and manifest['invalid_dispatches'] == 0,
            'native trial used another checkpoint, fallback, or invalid dispatch')

    arrivals = manifest['native_road_arrivals']
    require(isinstance(arrivals, dict) and bool(arrivals)
            and all(type(value) is int and value > 0 for value in arrivals.values()),
            'native trial has no valid road arrivals')
    active = sum(1 << row[0] for row in config['expected_runtime_scene']['opening_fleet'] if row[1])
    actions = [event for event in iter_event_bundle(native_path.parent, manifest)
               if event['event_type'] == 'ACTION_APPLIED']
    require(len(actions) == len(requests), 'native applied action coverage differs from requests')
    for step, (row, request, event) in enumerate(zip(inferred, requests, actions, strict=True)):
        data = event['data']
        mask = row['action_mask']
        require(all(row[key] == value for key, value in request_identity.items())
                and type(row['run_index']) is int
                and type(row['decision_sequence']) is int and row['decision_sequence'] == step
                and type(row['truck_id']) is int and type(data['truck_id']) is int
                and row['truck_id'] == data['truck_id'] and 0 <= row['truck_id'] < 9
                and row['dispatchable_mask'] == row['in_system_mask'] == active
                and (1 << row['truck_id']) & active
                and type(row['action']) is int and row['action'] == request['action'] == data['action']
                and isinstance(mask, list) and len(mask) == 4
                and all(type(flag) is int and flag in (0, 1) for flag in mask)
                and 0 <= row['action'] < 4 and mask[row['action']] == 1
                and request['decision_sequence'] == data['decision_sequence'] == step
                and request['request_id'] == data['request_id']
                and type(row['sim_time_seconds']) in (float, int) and math.isfinite(row['sim_time_seconds'])
                and row['sim_time_seconds'] == request['sim_time_seconds'] == event['sim_time_seconds']
                and re.fullmatch(r'[0-9a-f]{64}', row['observation_sha256']),
                f'inference disagrees with native applied dispatch at decision {step}')
        if 'request_id' in row:
            require(row['request_id'] == request['request_id'], f'inference request id differs at decision {step}')
    require(witness.reference(seal) == pin, 'inference manifest changed during read')
    return dict(status='VERIFIED', checkpoint_sha256=config['checkpoint_sha256'],
                policy_id=config['policy_id'], action_mode=config['action_mode'],
                decision_count=len(requests), inference_sha256=digest, inference_manifest=pin)


def verify_policy(config, job, native):
    if native is None:
        return dict(status='UNVERIFIED', reason='native scene and replay must verify first')
    if not (Path(job['python_output']) / 'inference_manifest.json').is_file():
        return dict(status='PENDING', reason='no sealed inference manifest')
    try:
        return policy_audit(config, job, native)
    except (ValueError, KeyError, TypeError, OSError) as error:
        return dict(status='UNVERIFIED', reason=str(error))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=CASES, help='inspect or settle one layout')
    parser.add_argument('--collect', action='store_true', help='settle actual native evidence and publish reports')
    parser.add_argument('--finalize-incomplete', action='store_true',
                        help='record UNKNOWN after the selected AnyLogic run has stopped incomplete')
    args = parser.parse_args(argv)
    if args.finalize_incomplete and not (args.collect and args.case):
        parser.error('--finalize-incomplete requires --collect and --case')
    cases = (args.case,) if args.case else tuple(CASES)
    results = []
    for label in cases:
        config, suite, job = load_case(label)
        physical, native = assess(label, config, job, args.finalize_incomplete)
        if args.collect and physical['physical_status'] in ('COMPLETE', 'UNKNOWN'):
            with witness.ledger._launch_lock(Path(suite['output']) / 'collector'):
                status = settle(suite, job, physical, collect=True)
        else:
            status = settle(suite, job, physical)
        if status == 'UNKNOWN':
            for key in ('ore_target_met', 'waste_target_met', 'compliant_windows',
                        'source_witness', 'source_outcomes', 'trial_minus_witness', 'comparison'):
                physical.pop(key, None)
        audit = verify_policy(config, job, native if status != 'UNKNOWN' else None)
        results.append(dict(physical, ledger_status=status, inference_audit=audit))
        if args.collect and status == 'COMPLETE':
            record(Path(job['python_output']) / 'policy_verification.json', audit, replace=True)
    if args.collect and any(row['ledger_status'] != 'UNREGISTERED' for row in results):
        suffix = f'_{args.case}' if args.case else ''
        folder = PROJECT / 'reports' / HERE.relative_to(PROJECT / 'change_plans')
        record(folder / f'trial_evaluation{suffix}.json', results, replace=True)
    print(json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if all(row['physical_status'] == 'COMPLETE' and row['inference_audit']['status'] == 'VERIFIED'
                    for row in results) else 2


if __name__ == '__main__':
    sys.exit(main())
