"""Check nominal joint transport, loader, dump and crusher workload."""

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

from scipy.optimize import linprog

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from compile_layouts import CAPACITIES, compile_layout, load_layouts


@dataclass(frozen=True)
class Flow:
    truck_type: int
    origin: int
    load: int
    dump: int
    payload: float
    grade: float
    cycle: float
    loading: float
    dumping: float


def screen(layout, task):
    raw, edges = compile_layout(layout)
    properties = dict(row.split('=', 1) for row in raw.decode('ascii').splitlines())
    roads = {(a, b): (length, speed) for a, b, length, speed in edges}
    duration = task['shift_seconds']

    def travel(a, b, maximum):
        route = tuple(map(int, properties[f'route.{a}.{b}'].split(',')))
        return sum(length / min(maximum, speed)
                   for length, speed in (roads[x, y] for x, y in zip(route, route[1:])))

    flows = []
    for k, payload in enumerate(CAPACITIES):
        for origin in (0, 1):
            for slot, (ore, grade, _) in enumerate(layout['loads']):
                dump = 0 if ore else 1
                bucket = min(30, (13.5 * 2.2 if slot < 3 else 15.0 * 2.0) * .90)
                load_time = 60 * (.70 + .45 * math.ceil((payload - 1e-9) / bucket))
                dump_time = 60 * ((1.8, 2.0, 2.3) if ore else (1.5, 1.7, 2.0))[k]
                cycle = travel(origin, 5 + slot, 6.5) + load_time + travel(5 + slot, dump, 5.0) + dump_time
                flows.append(Flow(k, origin, slot, dump, payload, grade, cycle, load_time, dump_time))

    counts = Counter(slot // 3 for slot in layout['active_trucks'])
    origins = Counter((slot // 3, int(slot < 3)) for slot in layout['active_trucks'])
    rows, bounds = [], []

    def add(weights, bound, target=0.0):
        rows.append([*(weights(flow) for flow in flows), target])
        bounds.append(bound)

    for k in range(3):
        add(lambda f: f.cycle if f.truck_type == k else 0.0, counts[k] * duration)
        for d in (0, 1):
            add(lambda f: int(f.origin == d) - int(f.dump == d) if f.truck_type == k else 0.0, origins[k, d])
    for slot, capacity in enumerate(task['loader_slots']):
        add(lambda f: f.loading if f.load == slot else 0.0, capacity * duration)
        add(lambda f: f.payload if f.load == slot else 0.0, 30000)
    for d, capacity in enumerate(task['dump_slots']):
        add(lambda f: f.dumping if f.dump == d else 0.0, capacity * duration)
    add(lambda f: f.payload if f.dump == 0 else 0.0, task['crusher_rate_tph'] * duration / 3600)
    add(lambda f: f.payload if f.dump == 1 else 0.0, task['waste_upper_tons'])
    add(lambda f: -f.payload if f.dump == 0 else 0.0, 0.0, task['ore_target_tons'])
    add(lambda f: -f.payload if f.dump == 1 else 0.0, 0.0, task['waste_target_tons'])
    add(lambda f: f.payload * (f.grade - task['grade_upper']) if f.dump == 0 else 0.0, 0.0)
    add(lambda f: f.payload * (task['grade_lower'] - f.grade) if f.dump == 0 else 0.0, 0.0)

    fit = linprog([0] * len(flows) + [-1], A_ub=rows, b_ub=bounds, bounds=(0, None), method='highs')
    if not fit.success:
        raise RuntimeError(fit.message)
    scale = float(fit.x[-1])
    return {'layout_id': layout['id'], 'layout_sha256': sha256(raw).hexdigest(),
            'vehicles_by_type': [counts[k] for k in range(3)],
            'nominal_joint_target_multiple': scale, 'nominal_screen_pass': scale >= 1 - 1e-8,
            'model_scope': 'fractional full-load trips, no queue, nominal speeds, whole-shift mean grade'}


if __name__ == '__main__':
    task = json.loads((HERE / 'task.json').read_text())
    print(json.dumps([screen(layout, task) for layout in load_layouts()], indent=2, allow_nan=False))
