"""Audit native single-trip event receipts against frozen geometry and material balance."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
POS_TOL = 0.001
MASS_TOL = 1e-6


def require(value, message):
    if not value:
        raise ValueError(message)


def load_properties(path, expected_sha):
    data = Path(path).read_bytes()
    require(sha256(data).hexdigest() == expected_sha, f'{path} bytes do not match the manifest')
    return dict(line.split('=', 1) for line in data.decode('ascii').splitlines())


def within(a, b, tolerance, message):
    require(math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tolerance, message)


def verify(case, receipt):
    config = load_properties(case['config_path'], case['config_sha256'])
    scene = load_properties(case['scene_path'], case['scene_sha256'])
    expected_meta = {name: case[name] for name in
                     ('recording_id', 'run_index', 'environment_seed', 'config_sha256', 'layout_id')}
    expected_meta['layout_sha256'] = case['scene_sha256']
    events = [json.loads(line) for line in receipt.read_text(encoding='utf-8').splitlines()]
    require(len(events) >= 5 and events[0]['event'] == 'START' and events[-1]['event'] == 'PASS',
            'Missing START or terminal PASS receipt')
    last = 0.0
    for event in events:
        require(all(event.get(k) == v for k, v in expected_meta.items()), 'Event identity mismatch')
        now = event['time_s']
        require(math.isfinite(now) and last <= now <= float(config['stop_seconds']), 'Event time outside run')
        last = now

    slot = int(config['load_slot'])
    target, ore, grade, _ = scene[f'load.{slot}'].split(',')
    require(ore == '1', 'Slot must supply ore')
    active, _, capacity, start = scene[f"vehicle.{config['truck_id']}"].split(',')
    require(active == '1' and start == '0', 'Incorrect starting truck')
    roads = {}
    for key, spec in scene.items():
        if key.startswith('road.'):
            a, b, forward, reverse = spec.split(',')
            idx = int(key.split('.')[1])
            roads[a, b] = idx, float(forward)
            roads[b, a] = idx, float(reverse)

    def pos(wire_id):
        return tuple(map(float, scene[f'node.{wire_id}'].split(',')[1:]))

    segments = []
    for source, end, is_loaded in ((start, target, False), (target, start, True)):
        path = scene[f'route.{source}.{end}'].split(',')
        segments.extend((a, b, is_loaded) for a, b in zip(path, path[1:]))
    require(len(events) == 2 + 3 * len(segments), 'Unexpected number of segment receipts')
    for i, (a, b, loaded) in enumerate(segments):
        depart, middle, arrive = events[1 + 3 * i:4 + 3 * i]
        idx, limit = roads[a, b]
        require([row['event'] for row in (depart, middle, arrive)] == ['DEPART', 'MIDPOINT', 'ARRIVE'],
                'Missing native departure, sampled midpoint, or arrival')
        require(depart['time_s'] < middle['time_s'] < arrive['time_s'], 'Missing positive travel time')
        origin, target_xy = pos(a), pos(b)
        distance = math.dist(origin, target_xy)
        for row in (depart, middle, arrive):
            require((row['from'], row['to'], row['loaded'], row['native_road_index'])
                    == (int(a), int(b), loaded, idx), 'Native route or payload phase differs from config')
            within(row['payload_tons'], float(capacity) if loaded else 0.0, MASS_TOL, 'Invalid truck payload')
        within(math.dist((depart['x_m'], depart['y_m']), origin), 0.0, POS_TOL, 'Incorrect departure position')
        within(math.dist((arrive['x_m'], arrive['y_m']), target_xy), 0.0, POS_TOL, 'Incorrect arrival position')
        dx, dy = (target_xy[j] - origin[j] for j in (0, 1))
        x, y = middle['x_m'] - origin[0], middle['y_m'] - origin[1]
        progress = (dx*x + dy*y) / (distance * distance)
        require(0.5 - 1e-8 <= progress < 1, 'Sample is not in the interior of the intended road')
        within(abs(dx*y - dy*x) / distance, 0.0, POS_TOL, 'Sample is off the intended road')
        duration = distance / min(5.0 if loaded else 6.5, limit)
        within(arrive['time_s'] - depart['time_s'], duration, 0.005, 'Travel time differs from the controlled speed')

    final = events[-1]
    require(final['completed_trips'] == 1, 'Expected exactly one completed trip')
    within(final['loaded_tons'], float(capacity), MASS_TOL, 'Incorrect loaded tons')
    within(final['received_tons'], float(capacity), MASS_TOL, 'Incorrect received tons')
    within(final['received_metal_tons'], float(capacity) * float(grade), MASS_TOL, 'Incorrect received metal')
    within(final['crushed_tons'] + final['fifo_tons'], final['received_tons'], MASS_TOL, 'Crusher ore does not balance')
    within(final['crushed_metal_tons'] + final['fifo_metal_tons'], final['received_metal_tons'],
           MASS_TOL, 'Crusher metal does not balance')
    return {'status': 'PASS', **expected_meta, 'segments': len(segments), 'terminal': final,
            'source_receipt': str(receipt)}


def archive(path, contents):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open('x', encoding='utf-8') as output:
            output.write(contents)
    except FileExistsError:
        require(path.read_text(encoding='utf-8') == contents, 'Existing verified report differs')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--layout', choices=('tree_A', 'loop_B'))
    args = p.parse_args()
    manifest = json.loads((HERE / 'launch_identity.json').read_text())
    for case in manifest['cases']:
        if args.layout and args.layout != case['layout_id']:
            continue
        result = verify(case, Path(case['event_directory']) / 'events.jsonl')
        report = json.dumps(result, indent=2, allow_nan=False) + '\n'
        archive(Path(case['python_output']) / 'trip_check.json', report)
        archive(Path(manifest['reports_directory']) / f"{case['layout_id']}_trip_check.json", report)
        print(report, end='')


if __name__ == '__main__':
    main()
