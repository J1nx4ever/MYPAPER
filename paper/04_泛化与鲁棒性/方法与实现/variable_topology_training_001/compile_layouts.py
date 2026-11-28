"""Compile synthetic shift layouts. No run IDs are allocated by this tool."""

import argparse
from hashlib import sha256
from heapq import heappop, heappush
import json
import math
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
SCHEMA = 'dispatch.topology-scene.v1'
NAMES = ('Crusher', 'dumpNode', 'intersection1', 'intersection2', 'intersection3',
         'ShovelOreHigh1', 'ShovelOreHigh2', 'ShovelOreLow', 'ShovelWaste')
STATIONS = {0, 1, 5, 6, 7, 8}
CAPACITIES = (91.7, 147.0, 193.0)


def shortest_path(edges, source, target, *, time_weighted):
    pending = [(0.0, (source,))]
    visited = set()
    while pending:
        cost, path = heappop(pending)
        node = path[-1]
        if node == target:
            return cost, path
        if node in visited:
            continue
        visited.add(node)
        for a, b, length, speed in edges:
            if a == node and b not in visited:
                weight = length / speed if time_weighted else length
                heappush(pending, (cost + weight, (*path, b)))
    return math.inf, ()


def compile_layout(layout):
    name = layout['id']
    if not re.fullmatch(r'[A-Za-z0-9_]+', name):
        raise ValueError('layout ID must be an ASCII identifier')
    nodes = {int(i): tuple(xy) for i, xy in sorted(layout['nodes_m'].items())}
    if not STATIONS <= nodes.keys() <= set(range(len(NAMES))):
        raise ValueError('the layout must keep four load sites and two dump sites in their business slots')
    if len(nodes) != len(layout['nodes_m']) or len(set(nodes.values())) != len(nodes) or any(
            len(xy) != 2 or not all(math.isfinite(x) for x in xy) for xy in nodes.values()):
        raise ValueError('each model node requires a unique finite coordinate pair')
    roads = layout['roads']
    if not 1 <= len(roads) <= 8:
        raise ValueError('the training prototype has at most 16 directed road slots')
    edges = []
    used = set()
    for a, b, outbound, inbound in roads:
        pair = frozenset((a, b))
        if a not in nodes or b not in nodes or a == b or pair in used:
            raise ValueError('missing endpoints, self-loop, or parallel road')
        used.add(pair)
        length = math.dist(nodes[a], nodes[b])
        if length > 10000 or not all(math.isfinite(s) and 0 < s <= 10 for s in (outbound, inbound)):
            raise ValueError('road length or speed outside the existing observation scale')
        edges.extend(((a, b, length, outbound), (b, a, length, inbound)))
    # Production can choose a speed-weighted OD, while native movement between
    # adjacent nodes must still select the exact road declared for that segment.
    for rid in range(0, len(edges), 2):
        a, b, direct, _ = edges[rid]
        alternate, _ = shortest_path(edges[:rid] + edges[rid + 2:], a, b, time_weighted=False)
        if alternate <= direct + 1e-6:
            raise ValueError(f'road {a}->{b} is not the unique native shortest path between its endpoints')
    if any(not shortest_path(edges, 0, node, time_weighted=False)[1] for node in nodes):
        raise ValueError('the layout contains an unreachable node')

    active = set(layout['active_trucks'])
    if not active <= set(range(9)) or not 6 <= len(active) <= 9 or len(active) != len(layout['active_trucks']):
        raise ValueError('the scene must select six to nine unique truck slots')
    loads = layout['loads']
    if len(loads) != 4 or sum(row[0] == 1 for row in loads) != 3:
        raise ValueError('the current task requires three ore load sites and one waste load site')
    for ore, grade, half_range in loads:
        if ore not in (0, 1) or not 0 <= half_range <= min(grade, 1 - grade):
            raise ValueError('invalid load grade interval')
        if not ore and (grade != 0 or half_range != 0):
            raise ValueError('waste requires zero grade in this prototype')

    props = {'format': SCHEMA, 'layout_id': name}
    for node, xy in nodes.items():
        props[f'node.{node}'] = ','.join(map(str, (NAMES[node], *xy)))
    for i, road in enumerate(roads):
        props[f'road.{i}'] = ','.join(map(str, road))
    for truck in range(9):
        category = truck // 3
        start = 1 if truck < 3 else 0
        props[f'vehicle.{truck}'] = f'{int(truck in active)},{category},{CAPACITIES[category]},{start}'
    for slot, (ore, grade, half_range) in enumerate(loads):
        props[f'load.{slot}'] = f'{slot + 5},{ore},{grade},{half_range}'
    props['dump.0'] = '0,ore'
    props['dump.1'] = '1,waste'
    # Every empty truck may be dispatched from either dump, and every load
    # must have a path to its declared material destination.
    for a, b in [(dump, ld) for dump in (0, 1) for ld in range(5, 9)] + [
            (5 + i, 0 if row[0] else 1) for i, row in enumerate(loads)]:
        _, path = shortest_path(edges, a, b, time_weighted=True)
        props[f'route.{a}.{b}'] = ','.join(map(str, path))
    payload = ''.join(f'{key}={value}\n' for key, value in props.items()).encode('ascii')
    return payload, edges


def load_layouts():
    layouts = json.loads((HERE / 'layouts.json').read_text())['layouts']
    if len({row['id'] for row in layouts}) != len(layouts):
        raise ValueError('duplicate layout IDs')
    return layouts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    compiled = [(layout['id'], *compile_layout(layout)) for layout in load_layouts()]
    args.output.mkdir(parents=True, exist_ok=True)
    for name, payload, edges in compiled:
        filename = args.output / f'{name}.properties'
        with filename.open('xb') as stream:
            stream.write(payload)
        print(f'{name}, {len(edges)} directed roads, {sha256(payload).hexdigest()}, {filename}')


if __name__ == '__main__':
    main()
