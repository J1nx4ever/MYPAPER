"""Offline whole-shift workload relaxation for the six-truck outage scenario."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse.csgraph import shortest_path


ROOT = Path(__file__).resolve().parents[4]
RUN = ROOT / 'runs/bg128-full-s02-u32-fleet6-zero-shot10-20261124-001'
ACTIVE = (0, 1, 3, 4, 6, 7)
SITE_NODES = (5, 6, 7, 8)
EMPTY_KPH = (26, 24, 22)
LOADED_KPH = (20, 18, 16)
UNLOAD_MINUTES = ((1.8, 2.0, 2.3), (1.5, 1.7, 2.0))
PREFIX = 'config.runtime_scene.'


def read_configuration():
    seal = json.loads((RUN / 'suite_complete.json').read_bytes())
    report = Path(seal['evaluation']['path']).read_bytes()
    sha = hashlib.sha256(report).hexdigest()
    if seal['status'] != 'COMPLETE' or sha != seal['evaluation']['sha256']:
        raise ValueError('Six-truck evaluation seal differs')
    configurations = [row['actual_rpc_configuration'] for row in json.loads(report)['shifts']]
    cfg = configurations[0]
    if (not all(row == cfg for row in configurations)
            or cfg['config.fleet.disabled_mask'] != 292
            or cfg['config.fleet.active_count'] != len(ACTIVE)):
        raise ValueError('RPC configuration differs from the fixed outage scenario')
    return cfg, sha


def route_times(cfg, truck_type, *, loaded, speed_factor):
    kph = LOADED_KPH if loaded else EMPTY_KPH
    base_key = 'loadedBase_mps' if loaded else 'emptyBase_mps'
    graph = np.full((9, 9), np.inf)
    for road_id in range(int(cfg[PREFIX + 'segmentMap_count'])):
        road = PREFIX + f'road.{road_id}.'
        start, end = (int(cfg[road + field]) for field in ('from_node_index', 'to_node_index'))
        speed = min(cfg[road + 'speedLimit_mps'],
                    cfg[road + base_key] * kph[truck_type] / kph[0] * speed_factor)
        graph[start, end] = cfg[road + 'length_scaled_f32'] * 10000 / speed / 60
    return shortest_path(graph)


def solve_relaxation(cfg, speed_factor, service_factor):
    """Keep route state and shift workload, aggregate grade over all four windows."""
    t_shift = cfg['config.shift_seconds'] / 60
    payloads = np.array([cfg[PREFIX + f'truck.{tid}.capacity_t'] for tid in ACTIVE])
    trips = np.arange(len(ACTIVE) * 2 * len(SITE_NODES)).reshape(len(ACTIVE), 2, len(SITE_NODES))
    end_at_waste = np.arange(trips.size, trips.size + len(ACTIVE))
    processed = np.arange(end_at_waste[-1] + 1, end_at_waste[-1] + 4)
    n_variables = processed[-1] + 1
    trip_time = np.empty(trips.shape)
    load_time = np.empty(trips.shape)
    unload_time = np.empty(trips.shape)
    delivered = np.zeros((len(SITE_NODES), n_variables))
    for i, (tid, payload) in enumerate(zip(ACTIVE, payloads, strict=True)):
        truck_type = tid // 3
        empty = route_times(cfg, truck_type, loaded=False, speed_factor=speed_factor)
        loaded = route_times(cfg, truck_type, loaded=True, speed_factor=speed_factor)
        for j, node in enumerate(SITE_NODES):
            site = PREFIX + f'load.{j}.'
            dump = int(not cfg[site + 'isOre'])
            bucket = min(30.0, cfg[site + 'bucket'] * cfg[site + 'density'] * 0.90)
            full_buckets = math.ceil((payload - 1e-9) / bucket)
            load_time[i, :, j] = (0.70 + cfg[site + 'singleServeTime'] * full_buckets) * service_factor
            unload_time[i, :, j] = UNLOAD_MINUTES[dump][truck_type] * service_factor
            trip_time[i, :, j] = (empty[:2, node] + load_time[i, :, j]
                                   + loaded[node, dump] + unload_time[i, :, j])
            delivered[j, trips[i, :, j]] = payload

    eye = np.eye(n_variables)
    a_ub, b_ub, a_eq, b_eq = [], [], [], []
    for i, tid in enumerate(ACTIVE):
        a_ub.append(trip_time[i].ravel() @ eye[trips[i].ravel()])
        b_ub.append(t_shift)
        # Transitions from the crusher to the waste dump add a unit of flow there.
        a_eq.append(eye[trips[i, 0, 3]] - eye[trips[i, 1, :3]].sum(axis=0)
                    - eye[end_at_waste[i]])
        b_eq.append(-int(tid < 3))
    for j in range(len(SITE_NODES)):
        a_ub.append(load_time[:, :, j].ravel() @ eye[trips[:, :, j].ravel()])
        b_ub.append(t_shift * cfg[PREFIX + f'load.{j}.shovelCapacity'])
    for dump, site_slice in enumerate((slice(0, 3), slice(3, 4))):
        a_ub.append(unload_time[:, :, site_slice].ravel() @ eye[trips[:, :, site_slice].ravel()])
        b_ub.append(t_shift * cfg[PREFIX + f'dump.{dump}.serviceCapacity'])

    total_processed = eye[processed].sum(axis=0)
    grades = np.array([cfg[PREFIX + f'load.{j}.grade'] for j in range(3)])
    a_ub.extend(eye[processed] - delivered[:3])
    b_ub.extend([0.0] * 3)
    a_ub.extend((delivered[:3].sum(axis=0) - total_processed, total_processed,
                 (cfg['config.grade_lower'] - grades) @ eye[processed],
                 (grades - cfg['config.grade_upper']) @ eye[processed],
                 -delivered[3], delivered[3]))
    b_ub.extend((cfg[PREFIX + 'dump.0.crusherBufferCapacityTons'],
                 t_shift / 60 * cfg[PREFIX + 'dump.0.crusherRateTph'], 0.0, 0.0,
                 -cfg['config.target_waste_tons_shift'], cfg['config.waste_task_upper_tons']))
    bounds = [(0, None)] * trips.size + [(0, 1)] * len(ACTIVE) + [(0, None)] * len(processed)
    result = linprog(-total_processed, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq,
                     bounds=bounds, method='highs')
    if not result.success:
        raise RuntimeError(result.message)
    return {
        'speed_factor': speed_factor,
        'service_time_factor': service_factor,
        'ore_processed_upper_within_relaxation_t': -result.fun,
        'waste_delivered_t': float(delivered[3] @ result.x),
        'ore_processed_by_source_t': result.x[processed].tolist(),
        'trips_by_truck_and_source': result.x[trips].sum(axis=1).tolist(),
    }


if __name__ == '__main__':
    configuration, report_sha = read_configuration()
    print(json.dumps({
        'six_truck_evaluation_sha256': report_sha,
        'active_truck_ids': ACTIVE,
        'scope': 'fractional trips, whole-shift grade, static travel and services, no queues',
        'cases': [solve_relaxation(configuration, speed, service)
                  for speed, service in ((0.95, 1.0), (1.0, 1.0), (1.0, 0.9), (1.1, 0.9))],
    }, indent=2, allow_nan=False))
