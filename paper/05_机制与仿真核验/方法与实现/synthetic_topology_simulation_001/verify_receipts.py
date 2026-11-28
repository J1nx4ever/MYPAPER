"""Verify the actual vehicle receipts written by the two AnyLogic Simulations."""

import argparse
import csv
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'synthetic_topology_arrival_001'))
from run_validation import expected_at


def checked_close(actual, expected, tolerance, field):
    if not math.isfinite(actual) or abs(actual - expected) > tolerance:
        raise ValueError(f'{field}: observed {actual}, expected {expected}')


def position(row):
    return [float(row['x']), float(row['y'])]


def scene_config(case):
    raw = Path(case['config_path']).read_bytes()
    if sha256(raw).hexdigest() != case['config_sha256']:
        raise ValueError('Changed config bytes')
    config = dict(line.split('=', 1) for line in raw.decode('ascii').splitlines())
    points = {}
    for spec in config['nodes'].split(';'):
        name, x, y = spec.split(',')
        points[name] = [float(x), float(y)]
    return config, points


def verify(case, origin):
    config, points = scene_config(case)
    n_roads = len(config['roads'].split(';'))
    speed = float(config['speed'])
    sample = float(config['sample_seconds'])
    coord_tol = float(config['arrival_tolerance'])
    time_tol = 1e-6
    with origin.open(newline='') as source:
        rows = list(csv.DictReader(source))
    if len(rows) < 7 or rows[-1]['event'] != 'pass':
        raise ValueError('Missing final PASS receipt')
    metadata = {
        'recording_id': case['recording_id'], 'run_index': case['run_index'],
        'environment_seed': case['environment_seed'], 'config_sha256': case['config_sha256'],
        'layout': case['layout'], 'nodes': len(points), 'roads': n_roads,
    }
    for row in rows:
        for key, expected in metadata.items():
            if row[key] != str(expected):
                raise ValueError(f'{key} mismatch in native event log')
    legs = []
    order = []
    for leg in ('load', 'dump'):
        names = config[f'route_{leg}'].split(',')
        route = [points[node] for node in names]
        lengths = [math.dist(a, b) for a, b in zip(route, route[1:])]
        duration = sum(lengths) / speed
        events = [row for row in rows[:-1] if row['leg'] == leg]
        if len(events) < 3 or [row['event'] for row in events] != (
                ['depart'] + ['sample'] * (len(events) - 2) + ['arrival']):
            raise ValueError(f'Expected depart, samples and one native arrival for {leg}')
        order.extend(events)
        start, finish = float(events[0]['time_s']), float(events[-1]['time_s'])
        checked_close(finish - start, duration, time_tol, f'{leg} native duration')
        if legs:
            hold = start - legs[0]['arrival_time_s']
            planned = float(config['load_hold_seconds'])
            if not planned - time_tol <= hold <= planned + sample + time_tol:
                raise ValueError(f'Load-site wait was {hold:.9f} s')
        else:
            checked_close(start, 0.0, time_tol, 'load departure time')
        previous = start
        max_deviation = 0.0
        for row in events:
            timestamp = float(row['time_s'])
            if not previous - time_tol <= timestamp <= min(finish, previous + sample) + time_tol:
                raise ValueError(f'{leg} timestamps do not cover the route at {sample} s intervals')
            reference = expected_at(route, max(0.0, timestamp - start) * speed)
            deviation = math.dist(position(row), reference)
            checked_close(deviation, 0.0, coord_tol, f'{leg} sample at {timestamp:.4f} s')
            reported_target = [float(row['target_x']), float(row['target_y'])]
            checked_close(math.dist(reported_target, route[-1]), 0.0, coord_tol, f'{leg} recorded target')
            checked_close(float(row['error']), math.dist(position(row), route[-1]), coord_tol,
                          f'{leg} reported coordinate error')
            max_deviation = max(max_deviation, deviation)
            previous = timestamp
        target_error = math.dist(position(events[-1]), route[-1])
        checked_close(target_error, 0.0, coord_tol, f'{leg} coordinates at native arrival')
        legs.append({
            'leg': leg, 'route': names, 'target_xy': route[-1], 'arrival_xy': position(events[-1]),
            'arrival_time_s': finish, 'travel_seconds': finish - start, 'arrival_error': target_error,
            'max_route_deviation': max_deviation, 'coordinate_samples': len(events),
        })
    if rows[:-1] != order or rows[-1]['leg'] != 'dump':
        raise ValueError('Leg order does not match the two-stage trip')
    checked_close(float(rows[-1]['time_s']), legs[-1]['arrival_time_s'], time_tol, 'terminal receipt time')
    checked_close(math.dist(position(rows[-1]), points['dump']), 0.0, coord_tol, 'terminal receipt position')
    return {'status': 'PASS', 'source': 'native AnyLogic Simulation movement callbacks and tick samples',
            **metadata, 'event_file': str(origin), 'legs': legs}


def archive(path, contents):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open('x', encoding='utf-8') as file:
            file.write(contents)
    except FileExistsError:
        if path.read_text() != contents:
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layout', choices=('A', 'B'), help='Check only the named Simulation')
    args = parser.parse_args()
    manifest = json.loads((HERE / 'launch_identity.json').read_text())
    for case in manifest['cases']:
        if args.layout and args.layout != case['layout']:
            continue
        origin = Path(case['anylogic_events']) / 'events.csv'
        result = verify(case, origin)
        encoded = json.dumps(result, indent=2, allow_nan=False) + '\n'
        archive(Path(case['python_output']) / 'coordinate_checks.json', encoded)
        archive(Path(manifest['reports_directory']) / f"{case['layout']}_coordinate_checks.json", encoded)
        print(encoded, end='')


if __name__ == '__main__':
    main()
