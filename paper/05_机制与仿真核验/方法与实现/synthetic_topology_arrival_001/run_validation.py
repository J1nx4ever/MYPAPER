"""Build both maps in the AnyLogic SDK, sample their routes, and check coordinates."""

import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[5]
SDK = '8.9.0.202404161236'
DEFAULT_PLUGINS = '/home/jinx/apps/AnyLoigc/anylogic/plugins'


def classpath():
    plugins = Path(os.environ.get('ANYLOGIC_PLUGINS', DEFAULT_PLUGINS))
    bundles = ('engine', 'engine.editorapi', 'engine.datautil')
    jars = [plugins / f'com.anylogic.{name}_{SDK}' / f'com.anylogic.{name}.jar' for name in bundles]
    querydsl = plugins / f'com.anylogic.third_party_libraries_{SDK}' / 'lib/database/querydsl'
    jars.extend(querydsl / f'querydsl-{name}-5.0.0.jar' for name in ('sql', 'core'))
    if missing := [str(jar) for jar in jars if not jar.is_file()]:
        raise FileNotFoundError(', '.join(missing))
    return os.pathsep.join(map(str, jars))


def expected_at(points, distance):
    for start, end in zip(points, points[1:]):
        length = math.dist(start, end)
        if distance <= length:
            fraction = distance / length
            return [a + fraction * (b - a) for a, b in zip(start, end)]
        distance -= length
    return points[-1]


def validate(layout, config, raw_csv):
    rows = list(csv.DictReader(io.StringIO(raw_csv)))
    checked = []
    start_time = 0.0
    speed = config['speed_model_units_per_second']
    tolerance = config['arrival_tolerance_model_units']
    interval = config['sample_seconds']
    for stage in ('load', 'dump'):
        path = [layout['nodes'][node] for node in layout['expected_routes'][stage]]
        length = sum(math.dist(a, b) for a, b in zip(path, path[1:]))
        end_time = start_time + length / speed
        target = layout['nodes'][stage]
        leg = [row for row in rows if row['stage'] == stage]
        interior = range(1, math.ceil((end_time - start_time) / interval))
        times = [start_time, *(start_time + step * interval for step in interior), end_time]
        if [row['event'] for row in leg] != ['depart', *(['sample'] * len(interior)), 'arrival']:
            raise AssertionError(f"{layout['id']} {stage} is missing samples or events")
        max_route_error = 0.0
        for row, time in zip(leg, times):
            sample = [float(row['x']), float(row['y'])]
            route_error = math.dist(sample, expected_at(path, (time - start_time) * speed))
            reported_target = [float(row['target_x']), float(row['target_y'])]
            if (int(row['run_index']) != layout['run_index'] or abs(float(row['time_s']) - time) > 1e-8
                    or reported_target != target or not math.isfinite(route_error) or route_error > tolerance):
                raise AssertionError(f"{layout['id']} {stage} did not follow its expected route: {row}")
            max_route_error = max(max_route_error, route_error)
        end = [float(leg[-1]['x']), float(leg[-1]['y'])]
        error = math.dist(end, target)
        if error > tolerance:
            raise AssertionError(f"{layout['id']} {stage} did not reach {target}: got {end}")
        checked.append({
            'stage': stage, 'route': layout['expected_routes'][stage], 'path_length': length,
            'arrival_time_s': end_time, 'target_xy': target, 'arrival_xy': end, 'arrival_error': error,
            'max_path_sample_error': max_route_error, 'coordinate_samples': len(leg),
        })
        start_time = end_time
    if len(rows) != sum(item['coordinate_samples'] for item in checked):
        raise AssertionError(f"{layout['id']} has an unknown stage")
    return {'layout': layout['id'], 'run_index': layout['run_index'], 'recording_id': layout['recording_id'],
            'nodes': len(layout['nodes']), 'bidirectional_roads': len(layout['roads']), 'legs': checked}


def main():
    identity = json.loads((HERE / 'launch_identity.json').read_text())
    raw_config = (HERE / identity['config_file']).read_bytes()
    if hashlib.sha256(raw_config).hexdigest() != identity['config_sha256']:
        raise ValueError('campaign config differs from the recorded bytes')
    config = json.loads(raw_config)
    actual_cases = [{k: layout[k] for k in ('run_index', 'environment_seed', 'recording_id')}
                    | {'layout': layout['id']} for layout in config['layouts']]
    if actual_cases != identity['cases']:
        raise ValueError('campaign identities differ from the launch manifest')
    core = WORKSPACE / 'python/episode_budget_ppo.py'
    ppo_before = hashlib.sha256(core.read_bytes()).hexdigest()
    python_output = WORKSPACE / identity['python_output']
    event_output = WORKSPACE / identity['anylogic_events']
    if python_output.parent.exists():
        raise FileExistsError(python_output.parent)

    results = []
    logs = {}
    cp = classpath()
    with TemporaryDirectory(prefix='faith-topology-arrival-') as build:
        subprocess.run(['javac', '-cp', cp, '-d', build, str(HERE / 'TopologyArrivalProbe.java')], check=True)
        for layout in config['layouts']:
            nodes = ';'.join(f'{node},{x},{y}' for node, (x, y) in layout['nodes'].items())
            roads = ';'.join(','.join(edge) for edge in layout['roads'])
            params = [layout['run_index'], layout['environment_seed'], config['sample_seconds'],
                      config['speed_model_units_per_second'], config['stop_seconds'],
                      config['arrival_tolerance_model_units'], nodes, roads]
            result = subprocess.run(
                ['java', '-Djava.awt.headless=true', '-cp', build + os.pathsep + cp, 'TopologyArrivalProbe',
                 *map(str, params)], capture_output=True, text=True, timeout=20)
            if result.returncode != 0:
                raise RuntimeError(result.stderr + result.stdout)
            results.append(validate(layout, config, result.stdout))
            logs[layout['id']] = result

    ppo_after = hashlib.sha256(core.read_bytes()).hexdigest()
    if ppo_before != ppo_after:
        raise AssertionError('protected PPO core hash changed during the validation')
    summary = {'status': 'PASS', 'config_sha256': identity['config_sha256'],
               'method': identity['method'], 'anylogic_sdk': SDK, 'physical_shifts_started': 0,
               'native_simulation_clock_started': False, 'protected_ppo_sha256': ppo_after,
               'arrival_tolerance': config['arrival_tolerance_model_units'], 'layouts': results}
    python_output.mkdir(parents=True)
    event_output.mkdir()
    for layout_id, result in logs.items():
        (event_output / f'{layout_id}.csv').write_text(result.stdout)
        if result.stderr:
            (event_output / f'{layout_id}.stderr.txt').write_text(result.stderr)
    text = json.dumps(summary, indent=2, allow_nan=False) + '\n'
    (python_output / 'coordinate_checks.json').write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
