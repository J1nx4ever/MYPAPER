"""Verify path-flow decisions on sealed snapshots and historical action prefixes."""
import argparse
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
import json
from math import fsum, isclose, isfinite
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np

from flow_matching_campaign import METHOD
from flow_matching_policy import FlowMatchingPolicy
from traditional_dispatch_v1.runtime import _publish_json, _require
from change_plans.token_attention_001.business_graph_summary_009.baseline_comparison_001.dispatch_fifo_v3_001.review_offline import (
    ARCHIVE, sealed_frames,
)


def _policy():
    options = ('planning_ms', 'replan_seconds', 'match_trucks', 'soft_replan_seconds')
    return FlowMatchingPolicy(**{key: METHOD[key] for key in options})


def _choose(policy, observation, run, frame):
    mask = observation[19:23]
    original = observation.tobytes()
    started = perf_counter()
    choice = policy.choose(observation, mask)
    elapsed = (perf_counter() - started) * 1000
    _require(type(choice.action) is int and 0 <= choice.action < len(mask) and bool(mask[choice.action]),
             f'illegal action at {run}/{frame}')
    expected_tons = float(observation[23 + choice.action]) * 193.0
    _require(choice.route is not None and choice.route[2] == choice.action
             and isclose(choice.assigned_tons, expected_tons, rel_tol=1e-6, abs_tol=1e-6),
             f'confirmed route or offered payload differs at {run}/{frame}')
    _require(observation.tobytes() == original, f'observation modified at {run}/{frame}')
    record = asdict(choice)
    if record['need_time'] is not None and not isfinite(record['need_time']):
        record['need_time'] = None
    return choice, dict(record, elapsed_ms=elapsed, observation_sha256=sha256(original).hexdigest())


def _plan_record(plan):
    if plan is None:
        return None
    paths = []
    for route, slots in sorted(plan.routes.items()):
        truck_type, origin, site, dump = route
        _require(0 <= truck_type < 3 and 0 <= origin < 9 and 0 <= site < 4 and 0 <= dump < 2,
                 f'invalid plan route {route}')
        end = plan.now
        for slot in slots:
            _require(end <= slot.start < slot.stop <= 28800 and isfinite(slot.tons) and slot.tons > 0,
                     f'invalid slot for {route}: {slot}')
            end = slot.stop
        if slots:
            paths.append(dict(path=list(map(int, route)), periods=len(slots),
                              planned_tons=fsum(slot.tons for slot in slots)))
    windows = []
    for item in plan.grade_windows:
        q, m, grade = item.ore_tons, item.metal_tons, item.grade
        _require(isfinite(q) and isfinite(m) and 0 <= m <= q and 0 <= item.window < 4,
                 'invalid forecast mineral/metal accounting')
        _require((grade is None and q == 0) or (grade is not None and q > 0 and abs(grade - m / q) < 1e-9),
                 'forecast grade differs from its mineral and metal tons')
        windows.append(dict(window=int(item.window), ore_tons=float(q), metal_tons=float(m),
                            grade=None if grade is None else float(grade),
                            compliant=grade is not None and .26 <= grade <= .28))
    return dict(status=plan.status, planned_ore_tons=plan.ore_tons, planned_waste_tons=plan.waste_tons,
                ore_shortfall_tons=plan.ore_shortfall_tons, waste_shortfall_tons=plan.waste_shortfall_tons,
                grade_slack_tons=plan.grade_slack_tons, routes=paths, windows=windows,
                all_remaining_windows_compliant=bool(windows) and all(w['compliant'] for w in windows))


def _archived_paths(samples):
    series = {}
    for row in samples:
        run = row['run_index']
        if run in series:
            continue
        path = ARCHIVE / f'run_{run}.h5'
        _require(sha256(path.read_bytes()).hexdigest() == row['archive_sha256'], 'changed transition archive')
        with h5py.File(path) as archived:
            states = archived['states'][:]
            actions = archived['transitions/action'][:]
        _require(len(states) == len(actions) + 1, 'archive transition and snapshot lengths differ')
        series[run] = (states, actions)
    return series


def _review_prefix(run, states, actions, first, *, limit=8):
    policy = _policy()
    rows = []
    for frame in range(first, min(first + limit, len(actions))):
        observation = states[frame]
        actual = int(actions[frame])
        _require(0 <= actual < 4 and bool(observation[19 + actual]), 'archived action violates its archived mask')
        choice, detail = _choose(policy, observation, run, frame)
        same = actual == choice.action
        rows.append(dict(frame=frame, recorded_action=actual, reproduced_action=choice.action,
                         accepted_logged_action=same, kind=choice.kind, plan_version=choice.plan_version))
        if not same:
            break
        policy.commit(choice)
    return dict(run_index=run, first_frame=first, transitions_examined=len(rows),
                commits=sum(row['accepted_logged_action'] for row in rows), rows=rows)


def review():
    samples = []
    for run, frame, seal, observation in sealed_frames():
        policy = _policy()
        _, result = _choose(policy, observation, run, frame)
        samples.append(dict(run_index=run, frame=frame, archive_sha256=seal, **result,
                            plan=_plan_record(policy.plan)))
    _require(len(samples) == 24, 'expected 24 first/last frame witnesses across three archived runs')

    archives = _archived_paths(samples)
    prefixes = []
    for run, (states, actions) in archives.items():
        run_samples = [row for row in samples if row['run_index'] == run]
        for row in run_samples:
            row['recorded_action'] = int(actions[row['frame']])
        # Start where the cold recommendation agrees if possible; discontinue as
        # soon as either policy changes the physical trajectory.
        start = next((r['frame'] for r in run_samples if r['action'] == r['recorded_action']), 0)
        prefixes.append(_review_prefix(run, states, actions, start))

    times = [row['elapsed_ms'] for row in samples]
    plans = [row['plan'] for row in samples if row['plan'] is not None]
    planned_routes = [route for plan in plans for route in plan['routes']]
    guided = [row for row in samples if row['kind'] == 'planned' and row['need_time'] is not None]
    return dict(
        schema='bg128-dispatch-flow-matching-offline-v1',
        scope='sealed observations and contiguous prefixes while decisions agree with archived actions',
        physical_shifts_started=0,
        method=METHOD,
        frames=len(samples),
        distinct_observations=len({row['observation_sha256'] for row in samples}),
        decisions=dict(Counter(row['kind'] for row in samples)),
        matching_outcomes=dict(Counter(row['matching_kind'] for row in samples if row['matching_kind'])),
        fallback_reasons=dict(Counter(row['fallback_reason'] for row in samples if row['fallback_reason'])),
        plan_errors=dict(Counter(row['plan_error'] for row in samples if row['plan_error'])),
        planning_timeouts=sum(row['timed_out'] for row in samples),
        planned_frames=len(plans),
        plans_with_positive_path_flow=sum(bool(plan['routes']) for plan in plans),
        nominally_compliant_remaining_window_plans=sum(plan['all_remaining_windows_compliant'] for plan in plans),
        distinct_positive_flow_paths=len({tuple(row['path']) for row in planned_routes}),
        due_path_decisions=len(guided),
        matching_archived_actions=sum(row['action'] == row['recorded_action'] for row in samples),
        committed_prefix_actions=sum(row['commits'] for row in prefixes),
        cold_planner_latency_p95_ms=float(np.quantile(times, .95)),
        cold_planner_latency_max_ms=max(times),
        contiguous_prefixes=prefixes,
        rows=samples,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Exclusive JSON output path')
    args = parser.parse_args()
    result = review()
    if args.output:
        print(json.dumps(_publish_json(args.output, result), ensure_ascii=False))
    else:
        print(json.dumps(result, allow_nan=False, indent=2, ensure_ascii=False))
