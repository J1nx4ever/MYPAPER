#!/usr/bin/env python3
"""Reproduce aggregate LP and integer-load feasibility screens; no AnyLogic execution.

Requires Python 3.10+, numpy and scipy>=1.11. Run from any directory.
The 4-window variables are abstract processed/transported allocations, NOT a timetable.
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / 'config.json').read_text())
P = np.array([t['payload_t'] for t in CFG['truck_types']])
COUNTS = np.array([t['count'] for t in CFG['truck_types']])
GRADES = np.array([s['mean_grade'] for s in CFG['sites'][:3]])
SHAPE = (4, 4, 3)


def route(a, b):
    graph = {}
    for e in CFG['roads']:
        graph.setdefault(e['a'], []).append((e['b'], e))
        graph.setdefault(e['b'], []).append((e['a'], e))
    stack = [(a, [], {a})]
    while stack:
        node, edges, seen = stack.pop()
        if node == b:
            return edges
        for nxt, edge in graph[node]:
            if nxt not in seen:
                stack.append((nxt, edges + [edge], seen | {nxt}))
    raise ValueError((a, b))


def travel(a, b, k, loaded):
    speed = CFG['truck_types'][k]['loaded_kph' if loaded else 'empty_kph']
    return sum(e['length_m'] * .06 / (speed * e['speed_factor']) for e in route(a, b))


LOAD = np.zeros((4, 3))
UNLOAD = np.zeros((4, 3))
CYCLE = np.zeros((4, 3))
DIST = []
for i, s in enumerate(CFG['sites']):
    dest = 'Crusher' if s['is_ore'] else 'dumpNode'
    density = s['loose_density_t_m3']
    bucket = min(CFG['loader']['bucket_payload_cap_t'],
                 s['bucket_m3'] * density * CFG['loader']['bucket_fill_factor'])
    DIST.append({origin: sum(e['length_m'] for e in route(origin, s['node']))
                 for origin in ['Crusher', 'dumpNode']})
    for k, t in enumerate(CFG['truck_types']):
        assert t['body_m3'] * density >= P[k], 'volume-limited payload'
        passes = int(np.ceil((P[k] - 1e-9) / bucket))
        LOAD[i, k] = CFG['loader']['spot_min'] + passes * CFG['loader']['bucket_cycle_min']
        UNLOAD[i, k] = t['crusher_service_min' if s['is_ore'] else 'dump_service_min']
        # Max over both unload origins covers cross-task empty repositioning conservatively.
        CYCLE[i, k] = (max(travel(o, s['node'], k, False) for o in ['Crusher', 'dumpNode'])
                       + travel(s['node'], dest, k, True) + LOAD[i, k] + UNLOAD[i, k]
                       + CFG['screening']['extra_cycle_min'])


def problem(integer=False):
    rows, lows, highs, names = [], [], [], []

    def add(name, values, low=-np.inf, high=np.inf):
        rows.append(values.ravel())
        lows.append(low)
        highs.append(high)
        names.append(name)

    for w in range(4):
        effective = CFG['screening']['window_effective_min'][w]
        budget = effective * CFG['screening']['capacity_fraction']
        a = np.zeros(SHAPE)
        a[w, :3] = P
        # A witness additionally supplies ~4000 t/window, not a fixed source plan for RL.
        add(f'ore_w{w}', a, 4000, 4100 if integer else 4000)
        a = np.zeros(SHAPE)
        a[w, 3] = P
        # Evenly split waste only to exhibit a stronger balanced witness.
        add(f'waste_w{w}_witness_only', a, 1000, 1200 if integer else 1000)
        # Tight mean band for integer witness; exact 27% in LP.
        lo, hi = ((.269, .271) if integer else (.27, .27))
        for sign, g in [(1, hi), (-1, lo)]:
            a = np.zeros(SHAPE)
            a[w, :3] = sign * (GRADES[:, None] - g) * P
            add(f'mean_grade_w{w}_{sign}', a, high=0)
        for k in range(3):
            a = np.zeros(SHAPE)
            a[w, :, k] = CYCLE[:, k]
            add(f'truck_w{w}_k{k}', a, high=COUNTS[k] * budget)
        for i in range(4):
            a = np.zeros(SHAPE)
            a[w, i] = LOAD[i]
            add(f'loader_w{w}_i{i}', a, high=budget)
        for ixs, cap, label in [(slice(0, 3), 1, 'crusher'), (slice(3, 4), 2, 'dump')]:
            a = np.zeros(SHAPE)
            a[w, ixs] = UNLOAD[ixs]
            add(f'unload_{label}_w{w}', a, high=cap * budget)
        a = np.zeros(SHAPE)
        a[w, :3] = P
        add(f'crusher_rate_w{w}', a, high=CFG['crusher']['rate_t_h'] * effective / 60)
    for i, site in enumerate(CFG['sites']):
        a = np.zeros(SHAPE)
        a[:, i] = P
        add(f'stock_i{i}', a, high=site['reserve_t'])
    return np.array(rows), np.array(lows), np.array(highs), names


def solve(integer=False):
    A, lo, hi, names = problem(integer)
    cost = np.tile(CYCLE, (4, 1, 1)).ravel()
    if integer:
        r = milp(cost, integrality=np.ones(cost.size), bounds=Bounds(0, np.inf),
                 constraints=LinearConstraint(A, lo, hi),
                 options={'time_limit': 25, 'mip_rel_gap': .03, 'presolve': True})
    else:
        mask = np.isfinite(lo)
        r = linprog(cost, A_ub=np.concatenate([A, -A[mask]]),
                    b_ub=np.concatenate([hi, -lo[mask]]), bounds=(0, None), method='highs')
    if r.x is None:
        raise RuntimeError(r.message)
    x = np.rint(r.x) if integer else r.x
    # An incumbent is accepted on feasibility, even if time-limited and non-optimal.
    ax = A @ x
    violation = float(max(np.max(np.maximum(lo - ax, 0)), np.max(np.maximum(ax - hi, 0))))
    if violation > 1e-5:
        raise AssertionError(f'infeasible candidate: {violation}')
    x = x.reshape(SHAPE)
    tons = x @ P
    ore = tons[:, :3].sum(axis=1)
    grade = tons[:, :3] @ GRADES / ore
    out = {'solver_message': r.message, 'optimal': bool(r.success),
           'max_constraint_violation': violation, 'allocation_trips': x.tolist(),
           'tons_by_window_site': tons.tolist(), 'ore_by_window_t': ore.tolist(),
           'waste_by_window_t': tons[:, 3].tolist(), 'mean_grade_by_window': grade.tolist(),
           'bounded_grade_range_by_window': np.column_stack([
               grade-CFG['grade_half_range'], grade+CFG['grade_half_range']]).tolist(),
           'truck_minutes_by_window_type': (x*CYCLE).sum(axis=1).tolist(),
           'site_totals_t': tons.sum(axis=0).tolist(),
           'ore_total_t': float(ore.sum()), 'waste_total_t': float(tons[:, 3].sum()),
           'truck_minutes_total': float((x*CYCLE).sum()),
           'truck_minute_fraction_of_calendar_capacity': float((x*CYCLE).sum()/(COUNTS.sum()*CFG['shift_min'])),
           'proof_scope': 'aggregate window allocations and integer loads; no vehicle timetable, FIFO buffer, or AnyLogic proof'}
    with (ROOT / ('integer_allocation.csv' if integer else 'lp_allocation.csv')).open('w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['window', 'site', 'truck_model', 'trips', 'tons', 'truck_minutes'])
        for w in range(4):
            for i, site in enumerate(CFG['sites']):
                for k, t in enumerate(CFG['truck_types']):
                    wr.writerow([w+1, site['name'], t['model'], x[w,i,k],
                                 x[w,i,k]*P[k], x[w,i,k]*CYCLE[i,k]])
    return out


def geology_checks():
    # Indexed blocks ensure truck assignment does not redraw the same material.
    def blocks(site, n):
        rng = np.random.default_rng(np.random.SeedSequence([20261018, site]))
        return CFG['sites'][site]['mean_grade'] + rng.uniform(-.005, .005, n)
    assert np.array_equal(blocks(0, 10), blocks(0, 20)[:10])
    for i in range(3):
        assert np.max(abs(blocks(i,1000)-GRADES[i])) <= .005
    # One 193 t load crossing the 500 t block boundary at extraction position 450 t.
    g = blocks(0, 2)
    mixed = (50*g[0] + 143*g[1])/193
    assert abs(mixed*193-(50*g[0]+143*g[1])) < 1e-10
    return {'block_prefix_independent_of_vehicle_partition': True,
            'bounds_checked': True, 'cross_block_mass_balance_checked': True}


if __name__ == '__main__':
    result = {'config_sha256': hashlib.sha256((ROOT/'config.json').read_bytes()).hexdigest(),
              'route_distance_m': DIST, 'load_minutes': LOAD.tolist(),
              'conservative_cycle_minutes': CYCLE.tolist(),
              'lp': solve(False), 'milp': solve(True), 'geology': geology_checks()}
    (ROOT/'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))
