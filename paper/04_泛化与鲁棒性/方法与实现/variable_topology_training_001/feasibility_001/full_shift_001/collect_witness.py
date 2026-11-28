"""Check and settle native full-shift feasibility evidence without starting the simulator."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import sys


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[5]
GRAPH = HERE.parents[2]
sys.path[:0] = [str(HERE.parent), str(GRAPH / 'fleet_outage_resilience_001' / 'visual_acceptance_demo_001')]

from admit_layouts import (canonical, compile_layout, load_layouts, replay_event_bundle,
                           require, require_admission)
from demo_budget import bind, budget_snapshot, ledger, ledger_rows
from resilience_runtime import record


CAMPAIGN = 'site-layout-feasibility-witness-20261127-001'
CASES = {
    'A': ('tree_A', 10054031, 280117723616006),
    'B': ('loop_B', 10054032, 177596069752717),
}
EVIDENCE_ERRORS = (ValueError, KeyError, TypeError, OSError)
SEED_FIELDS = ('experiment_seed', 'environment_seed', 'actor_seed', 'actor_sampling_seed')


def reference(path):
    return {'path': str(path), 'sha256': sha256(path.read_bytes()).hexdigest()}


def inside_runs(value):
    path = Path(value)
    root = PROJECT / 'runs'
    resolved = path.resolve()
    require(path.is_absolute() and resolved.is_relative_to(root.resolve())
            and resolved != root.resolve(), 'output path must be inside project runs')
    return resolved


def case_config(label, layouts, task):
    path = HERE / 'configs' / f'{label}.json'
    ref = reference(path)
    require(path.with_suffix('.sha256').read_text().strip() == ref['sha256'],
            f'{label} configuration bytes differ from SHA256 sidecar')
    config = json.loads(path.read_bytes())
    require(reference(path) == ref, f'{label} configuration changed during read')
    layout_id, run, seed = CASES[label]
    recording = f'{CAMPAIGN}-{run}'
    require(config['purpose'] == 'topology_full_shift_feasibility'
            and config['layout_id'] == layout_id and config['recording_id'] == recording
            and type(config['run_index']) is int and config['run_index'] == run
            and type(config['environment_seed']) is int and config['environment_seed'] == seed,
            f'{label} configuration identity differs')
    layout = layouts[layout_id]
    require(config['layout_sha256'] == sha256(compile_layout(layout)[0]).hexdigest()
            and config['task_sha256'] == sha256(canonical(task)).hexdigest(),
            f'{label} configuration layout or task differs')
    for key in ('model_sha256', 'expected_runtime_scene_sha256'):
        require(re.fullmatch(r'[0-9a-f]{64}', config[key]), 'invalid ' + key)
    output = inside_runs(config['python_output_directory'])
    events = inside_runs(config['event_directory'])
    campaign = f'{CAMPAIGN}-{label}'
    job = dict(layout_id=layout_id, run_index=run, experiment_seed=seed, recording_id=recording,
               source_recording_id=recording, actor_seed=0, actor_sampling_seed=0,
               actor_sampling_seed_consumed=False, python_output=str(output),
               event_directory=str(events), config_sha256=ref['sha256'])
    # Each native experiment has one slot, so A and B can settle independently.
    suite = dict(campaign_id=campaign, suite_path=path, output=str(PROJECT / 'runs' / campaign), jobs=[job])
    return suite, job, layout


def native_manifest(path, job):
    if not path.is_dir():
        return None
    candidates = sorted(path.glob('*/events_manifest.json'))
    direct = path / 'events_manifest.json'
    if direct.is_file():
        candidates.insert(0, direct)
    require(len(candidates) <= 1, 'ambiguous event manifest under frozen event directory')
    if not candidates:
        return None
    source = candidates[0]
    raw = source.read_bytes()
    manifest = json.loads(raw)
    require(manifest['schema_version'] == 1 and manifest['record_kind'] == 'anylogic_boundary_events'
            and manifest['source_kind'] == 'actual_anylogic'
            and type(manifest['run_index']) is int and manifest['run_index'] == job['run_index']
            and type(manifest['seed']) is int and manifest['seed'] == job['experiment_seed']
            and manifest['recording_id'] == job['recording_id']
            and manifest['config_sha256'] == job['config_sha256'],
            'event manifest identity or native source differs')
    require(all(type(manifest[key]) is bool for key in ('complete', 'terminated', 'truncated'))
            and isinstance(manifest['episode_id'], str) and manifest['episode_id'],
            'event manifest lifecycle is invalid')
    return source, manifest, sha256(raw).hexdigest()


def assess(suite, job, layout, task, *, finalize_incomplete=False):
    result = {key: job[key] for key in ('layout_id', 'run_index', 'recording_id', 'config_sha256')}
    result.update(physical_status='NOT_STARTED', task_status='NOT_ADMITTED',
                  environment_seed=job['experiment_seed'], manifest=None, witness=None)
    event_dir = Path(job['event_directory'])
    try:
        bound = native_manifest(event_dir, job)
    except EVIDENCE_ERRORS as error:
        return dict(result, physical_status='UNVERIFIED', reason=str(error))
    if bound is None:
        if event_dir.is_dir() and any(event_dir.iterdir()):
            return dict(result, physical_status='UNVERIFIED', reason='native events have no manifest')
        return dict(result, reason='no native event manifest')

    source, manifest, digest = bound
    result['manifest'] = dict(path=str(source), sha256=digest)
    if not manifest['complete']:
        status = 'UNKNOWN' if finalize_incomplete else 'UNVERIFIED'
        return dict(result, physical_status=status, reason='native event manifest is incomplete')
    witness = dict(run_index=job['run_index'], recording_id=job['recording_id'],
                   environment_seed=job['experiment_seed'], config_path=str(suite['suite_path']),
                   config_sha256=job['config_sha256'], event_dir=str(source.parent),
                   event_manifest_sha256=digest)
    try:
        evidence = replay_event_bundle(
            source.parent, expected_episode_id=manifest['episode_id'],
            expected_config_sha256=job['config_sha256'], expected_recording_id=job['recording_id'],
            expected_source_kind='actual_anylogic')
        require(evidence['run_index'] == job['run_index'] and evidence['seed'] == job['experiment_seed']
                and reference(source) == result['manifest'], 'bound event identity changed during replay')
    except EVIDENCE_ERRORS as error:
        return dict(result, physical_status='UNKNOWN', reason=f'native replay failed: {error}')

    result.update(physical_status='COMPLETE', witness=witness, events_sha256=evidence['events_sha256'],
                  ore_tons=evidence['ore_tons'], waste_tons=evidence['waste_tons'],
                  metal_tons=evidence['metal_tons'], windows=evidence['windows'])
    try:
        result['certificate'] = require_admission(layout, task, witness)
    except EVIDENCE_ERRORS as error:
        result['reason'] = str(error)
    else:
        result['task_status'] = 'ADMITTED'
    return result


def prepared_rows(job):
    """Include stand-alone prepared identities, which lack a historical suite."""
    for path in sorted(GRAPH.rglob('launch_identity.json')):
        cases = json.loads(path.read_bytes()).get('cases', [])
        for row in cases.values() if isinstance(cases, dict) else cases:
            if (path.parent == HERE and row.get('recording_id') == job['recording_id']
                    and row.get('run_index') == job['run_index']
                    and row.get('environment_seed') == job['experiment_seed']):
                continue
            yield row


def check_fresh(suite):
    job, = suite['jobs']
    history = budget_snapshot()
    require(suite['campaign_id'] not in history['campaigns'], 'witness campaign is already registered')
    prior = bind(ledger.previous.paired10._prior_rows, SUITE=suite['suite_path'])(history)
    prior.extend(row['identity'] for row in history['jobs'].values() if isinstance(row.get('identity'), dict))
    prior.extend(prepared_rows(job))
    seed = job['experiment_seed']
    require(0 < seed <= ledger.JAVA_MASK, 'witness seed exceeds the Java seed width')
    seeds = set()
    for row in prior:
        require(job['run_index'] != row.get('run_index')
                and job['recording_id'] not in (row.get('recording_id'), row.get('source_recording_id')),
                'witness run or recording already used or prepared')
        for key in SEED_FIELDS:
            value = row.get(key)
            if type(value) is int and value > 0:
                require(value & ledger.JAVA_MASK != seed, 'witness seed already used or prepared')
                if key in ('experiment_seed', 'environment_seed'):
                    seeds.add(value)
    states = set(ledger.java_random_states(seed).values())
    require(len(states) == 5 and all(states.isdisjoint(ledger.java_random_states(v).values()) for v in seeds),
            'witness Java random stream overlaps history')


register = bind(ledger.register, check_fresh=check_fresh)


def ledger_entry(suite, job):
    path = ledger.ledger_path(suite)
    if not path.is_file():
        return 'UNREGISTERED', None
    name, pin, rows = ledger_rows(path)
    require(name == suite['campaign_id'] and pin == reference(suite['suite_path'])
            and set(rows) == {job['run_index']}, 'witness ledger differs from frozen configuration')
    identity, status, evidence = rows[job['run_index']]
    require(identity == ledger.identity(job), 'settled witness identity differs')
    return status, json.loads(evidence) if evidence else None


def settle(suite, job, result, *, collect=False):
    receipt = Path(job['python_output']) / 'receipt.json'
    status, pin = ledger_entry(suite, job)
    if status == 'COMPLETE':
        require(result['physical_status'] == status and reference(receipt) == pin,
                'sealed witness receipt or native events no longer verify')
        require(json.loads(receipt.read_bytes()) == receipt_payload(job, result),
                'replayed witness differs from sealed receipt')
        return status
    if status == 'UNKNOWN':
        result.update(physical_status='UNKNOWN', task_status='NOT_ADMITTED', witness=None,
                      reason=pin['reason'], manifest=pin.get('manifest', result['manifest']))
        for key in ('certificate', 'events_sha256', 'ore_tons', 'waste_tons', 'metal_tons', 'windows'):
            result.pop(key, None)
        return status
    if not collect or result['physical_status'] not in ('COMPLETE', 'UNKNOWN'):
        return status

    if status == 'UNREGISTERED':
        register(suite)
        status, _ = ledger_entry(suite, job)
    if status == 'PLANNED':
        ledger.claim(suite, job)
    if result['physical_status'] == 'COMPLETE':
        payload = receipt_payload(job, result)
        if receipt.is_file():
            require(json.loads(receipt.read_bytes()) == payload, 'existing witness receipt differs')
        else:
            record(receipt, payload)
        ledger.complete(suite, job, receipt)
    else:
        ledger.transition(suite, job, 'CLAIMED', 'UNKNOWN',
                          {'reason': result['reason'], 'manifest': result['manifest']})
    return result['physical_status']


def receipt_payload(job, result):
    return dict(result, **ledger.identity(job), actor_sampling_seed_consumed=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collect', action='store_true', help='settle manifests already produced by AnyLogic')
    parser.add_argument('--case', choices=CASES, help='inspect or settle just one layout')
    parser.add_argument('--finalize-incomplete', action='store_true',
                        help='record UNKNOWN after the selected AnyLogic run has stopped incomplete')
    args = parser.parse_args(argv)
    if args.finalize_incomplete and not (args.collect and args.case):
        parser.error('--finalize-incomplete requires --collect and --case')
    task = json.loads((HERE.parent / 'task.json').read_bytes())
    layouts = {layout['id']: layout for layout in load_layouts()}
    labels = (args.case,) if args.case else tuple(CASES)
    results = []
    for label in labels:
        suite, job, layout = case_config(label, layouts, task)
        result = assess(suite, job, layout, task, finalize_incomplete=args.finalize_incomplete)
        if args.collect and result['physical_status'] in ('COMPLETE', 'UNKNOWN'):
            with ledger._launch_lock(Path(suite['output']) / 'collector'):
                result['ledger_status'] = settle(suite, job, result, collect=True)
        else:
            result['ledger_status'] = settle(suite, job, result)
        results.append(result)

    if args.collect and any(row['ledger_status'] != 'UNREGISTERED' for row in results):
        suffix = f'_{args.case}' if args.case else ''
        report_dir = PROJECT / 'reports' / HERE.relative_to(PROJECT / 'change_plans')
        index = {row['layout_id']: row['witness'] for row in results
                 if row['ledger_status'] == 'COMPLETE' and row['witness'] is not None}
        record(report_dir / f'witness_index{suffix}.json', index, replace=True)
        record(report_dir / f'witness_collection{suffix}.json', results, replace=True)

    print(json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if all(row['task_status'] == 'ADMITTED' for row in results) else 2


if __name__ == '__main__':
    sys.exit(main())
