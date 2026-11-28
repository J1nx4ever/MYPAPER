"""Reject training candidates without independently replayed full-shift evidence."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[4]
sys.path[:0] = [str(PROJECT), str(HERE.parent)]

from compile_layouts import compile_layout, load_layouts
from python.acceptance_events import replay_event_bundle


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def load_bound_json(path, digest):
    raw = Path(path).read_bytes()
    require(sha256(raw).hexdigest() == digest, f'changed bytes: {path}')
    return json.loads(raw)


def check_targets(evidence, task):
    physical = evidence['config']
    for key in ('shift_seconds', 'window_seconds', 'grade_lower', 'grade_upper', 'grade_epsilon',
                'tonnage_epsilon', 'mass_balance_epsilon', 'crusher_rate_tph'):
        require(physical[key] == task[key], f'witness {key} differs from frozen task')
    windows = evidence['windows']
    require([row['window_index'] for row in windows] == list(range(4)), 'missing production windows')
    cap_per_window = task['crusher_rate_tph'] * task['window_seconds'] / 3600
    ton_tol = task['tonnage_epsilon']
    grade_tol = task['grade_epsilon']
    for row in windows:
        tons, metal = row['ore_tons'], row['metal_tons']
        require(0 < tons <= cap_per_window + ton_tol, 'empty or overcapacity crusher window')
        require(task['grade_lower'] - grade_tol <= metal / tons <= task['grade_upper'] + grade_tol,
                f"grade fails in window {row['window_index']}")
    mass_tol = task['mass_balance_epsilon']
    for field in ('ore_tons', 'metal_tons'):
        require(abs(math.fsum(row[field] for row in windows) - evidence[field]) <= mass_tol,
                f'shift/window mismatch: {field}')
    require(evidence['ore_tons'] >= task['ore_target_tons'], 'processed ore target unmet')
    require(task['waste_target_tons'] <= evidence['waste_tons'] <= task['waste_upper_tons'] + ton_tol,
            'delivered waste outside task range')
    require(evidence['fifo_dropped_tons'] <= mass_tol
            and evidence['fifo_dropped_metal_tons'] <= mass_tol, 'material discarded from crusher FIFO')


def check_runtime_capabilities(native, task):
    capacity = native['crusher_capacity_tons']
    require(type(capacity) in (int, float) and capacity == task['crusher_capacity_tons'],
            'runtime crusher capacity differs from frozen task')
    for key in ('loader_slots', 'dump_slots'):
        slots = native[key]
        require(type(slots) is list and all(type(count) is int for count in slots) and slots == task[key],
                f'runtime {key} differs from frozen task')


def require_admission(layout, task, witness):
    """Return a bound certificate, or fail closed before any training run is allocated."""
    scene, _ = compile_layout(layout)
    layout_sha = sha256(scene).hexdigest()
    task_sha = sha256(canonical(task)).hexdigest()
    require(witness is not None, 'no complete AnyLogic witness')
    snapshot = load_bound_json(witness['config_path'], witness['config_sha256'])
    require(snapshot['purpose'] == 'topology_full_shift_feasibility', 'wrong witness purpose')
    require(snapshot['layout_sha256'] == layout_sha and snapshot['task_sha256'] == task_sha,
            'frozen layout, roster, or workload mismatch')
    require(snapshot['recording_id'] == witness['recording_id'] and snapshot['run_index'] == witness['run_index']
            and snapshot['environment_seed'] == witness['environment_seed'], 'witness identity mismatch')
    event_dir = Path(witness['event_dir'])
    manifest_path = event_dir / 'events_manifest.json'
    manifest = load_bound_json(manifest_path, witness['event_manifest_sha256'])
    evidence = replay_event_bundle(
        event_dir, expected_config_sha256=witness['config_sha256'],
        expected_recording_id=witness['recording_id'], expected_source_kind='actual_anylogic')
    require(evidence['seed'] == witness['environment_seed'] and evidence['run_index'] == witness['run_index'],
            'runtime seed or run index mismatch')
    require(manifest == load_bound_json(manifest_path, witness['event_manifest_sha256']),
            'event manifest changed during proof verification')
    applied_dispatches = sum(
        not request['terminal'] and request['action'] is not None and 'action_event_seq' in request
        for request in evidence['requests']
    )
    require(applied_dispatches > 0, 'no applied nonterminal dispatch in event replay')

    physical = evidence['config']
    native = physical['actual_runtime_scene']
    required = {'layout_id', 'node_positions', 'directed_roads', 'opening_fleet', 'load_sites',
                'dump_sites', 'initial_inventory', 'exogenous_config', 'model_sha256',
                'crusher_capacity_tons', 'loader_slots', 'dump_slots'}
    require(isinstance(native, dict) and required <= native.keys(), 'actual startup scene incomplete')
    require(native['layout_id'] == layout['id'] and native['model_sha256'] == snapshot['model_sha256'],
            'runtime layout or model identity mismatch')
    check_runtime_capabilities(native, task)
    runtime_sha = sha256(canonical(native)).hexdigest()
    require(runtime_sha == snapshot['expected_runtime_scene_sha256'], 'actual road/fleet/equipment mismatch')
    fallbacks = manifest['fallback_dispatches']
    require(type(fallbacks) is int and fallbacks == 0, 'fallback dispatch cannot certify feasibility')
    invalid = manifest['invalid_dispatches']
    require(type(invalid) is int and invalid == 0, 'invalid dispatch cannot certify feasibility')
    # A naturally completed native event replay already checks unload/FIFO,
    # actual-action order, terminal flags, and cutoff timing.
    check_targets(evidence, task)
    return {'status': 'ADMITTED', 'layout_id': layout['id'], 'layout_sha256': layout_sha,
            'task_sha256': task_sha, 'runtime_scene_sha256': runtime_sha,
            'environment_seed': evidence['seed'], 'witness_run_index': evidence['run_index'],
            'witness_recording_id': evidence['recording_id'], 'config_sha256': witness['config_sha256'],
            'events_sha256': evidence['events_sha256'], 'event_manifest_sha256': witness['event_manifest_sha256'],
            'applied_dispatches': applied_dispatches,
            'ore_tons': evidence['ore_tons'], 'waste_tons': evidence['waste_tons']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--witness-index', type=Path, help='Frozen JSON mapping layout IDs to evidence bundles')
    args = parser.parse_args()
    task = json.loads((HERE / 'task.json').read_text())
    witnesses = json.loads(args.witness_index.read_text()) if args.witness_index else {}
    outcomes = []
    for layout in load_layouts():
        try:
            outcome = require_admission(layout, task, witnesses.get(layout['id']))
        except (ValueError, KeyError, TypeError, OSError) as error:
            outcome = {'status': 'NOT_ADMITTED', 'layout_id': layout['id'], 'reason': str(error)}
        outcomes.append(outcome)
    print(json.dumps(outcomes, indent=2, allow_nan=False))
    return 0 if all(row['status'] == 'ADMITTED' for row in outcomes) else 2


if __name__ == '__main__':
    sys.exit(main())
