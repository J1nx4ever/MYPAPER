"""Exercise the two online controllers on sealed decision-time snapshots."""

import argparse
from collections import Counter
from hashlib import sha256
from itertools import islice
import json
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np

from traditional_dispatch_v1.runtime import _publish_json
from traditional_dispatch_v2.decision import _cost, _tail_action
from traditional_dispatch_v2.nominal import NominalState

from .rolling_quota_policy import RollingQuotaPolicy
from .trip_fifo_policy import TripFifoPolicy


ROOT = Path(__file__).resolve().parents[5]
ARCHIVE = ROOT / 'runs/bg128-continuous-mixed-20261120-s02/evaluation/nominal/new/material'
CORE = ROOT.parents[1] / 'python/episode_budget_ppo.py'
RUNS = (10004025, 10004026, 10004027)
POLICIES = {'rolling_quota': RollingQuotaPolicy, 'trip_fifo': TripFifoPolicy}


def _load_archive(run):
    path = ARCHIVE / f'run_{run}.h5'
    actual_sha = sha256(path.read_bytes()).hexdigest()
    seal = json.loads(path.with_suffix('.h5.seal.json').read_bytes())
    if actual_sha != seal['hdf5_sha256']:
        raise ValueError(f'archive bytes differ from seal for {run}')
    with h5py.File(path, 'r') as data:
        observations = data['states'][:-1]
        actions = data['transitions/action'][:]
    if observations.shape != (len(actions), 1519) or observations.dtype != np.float32:
        raise ValueError(f'invalid archived physical observations for {run}')
    return observations, actions, actual_sha


def _sample_frames(observations):
    windows = np.rint(observations[:, 2] * 4).astype(int)
    for window in range(4):
        indices = np.flatnonzero(windows == window)
        if len(indices) < 2:
            raise ValueError(f'missing first/last decision pair for window {window}')
        yield int(indices[0])
        yield int(indices[-1])


def _choose(policy, observation):
    original = observation.tobytes()
    mask = observation[19:23].copy()
    started = perf_counter()
    result = policy.choose(observation, mask)
    elapsed_ms = (perf_counter() - started) * 1000
    if type(result.action) is not int or not 0 <= result.action < 4 or not mask[result.action]:
        raise ValueError('controller returned an illegal action on a sealed frame')
    if original != observation.tobytes():
        raise ValueError('controller mutated the archived observation')
    return result, elapsed_ms


def _forecast(observation, first_action):
    # Both policies get exactly one applied action here. The existing shared
    # deterministic tail completes the forecast to compare that action alone.
    state = NominalState.from_observation(observation)
    deadline = perf_counter() + 5.0

    def check():
        if perf_counter() >= deadline:
            raise TimeoutError('offline nominal continuation exceeded five seconds')

    state.apply_action(first_action)
    state.advance(check_deadline=check)
    while not state.finished:
        state.apply_action(_tail_action(state, check))
        state.advance(check_deadline=check)
    score = state.score_inputs()
    windows = []
    for window in range(score.first_window, 4):
        q, m = float(score.fifo_q[window]), float(score.fifo_m[window])
        grade = m / q if q > 0 else None
        windows.append(dict(window=window, ore_tons=q, metal_tons=m, grade=grade,
                            compliant=grade is not None and .26 - 1e-12 <= grade <= .28 + 1e-12))
    return dict(cost=float(_cost(score, unload=False)), ore_tons=float(score.ore_tons),
                waste_tons=float(score.waste_tons), future_wait_seconds=float(score.wait_seconds),
                windows=windows)


def _confirmed(policy, expected):
    summary = policy.summary()
    actions = summary['committed_actions']
    count = sum(actions) if isinstance(actions, list) else actions
    if count != expected:
        raise ValueError(f'expected {expected} acknowledged actions, got {count}')
    return summary


def _prefix(factory, observations, recorded, start):
    # A different action would alter the historical state that comes next.
    # Only deliver an ACK when the archived trajectory actually took our action.
    policy = factory()
    accepted = examined = 0
    for observation, actual in islice(zip(observations, recorded, strict=True), start, start + 8):
        decision, _ = _choose(policy, observation)
        examined += 1
        if decision.action != actual:
            break
        policy.commit(decision)
        accepted += 1
    return dict(first_frame=start, examined=examined, confirmed=accepted,
                summary=_confirmed(policy, accepted))


def _summarize(rows):
    times = [row['elapsed_ms'] for row in rows]
    gaps = [row['mip_gap'] for row in rows if row['mip_gap'] is not None]
    return dict(
        frames=len(rows),
        decision_kinds=dict(Counter(row['kind'] for row in rows)),
        solver_statuses=dict(Counter(row['solver_status'] for row in rows if row['solver_status'])),
        replans=sum(row['replanned'] for row in rows),
        validated_joint_plans=sum(row['decision_summary'].get('joint_plan_commits', 0) for row in rows),
        max_reported_mip_gap=max(gaps, default=None),
        time_limits=sum(row['timed_out'] for row in rows),
        fallback_reasons=dict(Counter(row['fallback_reason'] for row in rows if row['fallback_reason'])),
        p95_ms=float(np.quantile(times, .95)),
        max_ms=max(times),
        over_rpc_target_count=sum(value > 1000 for value in times),
        mean_nominal_cost=float(np.mean([row['nominal_first_action']['cost'] for row in rows])),
    )


def review():
    sources = [Path(__file__).with_name(name) for name in
               ('review_milp_offline.py', 'rolling_quota_policy.py', 'trip_fifo_policy.py')]
    sources.append(CORE)
    hashes = {path.name: sha256(path.read_bytes()).hexdigest() for path in sources}
    samples, prefixes = [], []
    for run in RUNS:
        observations, actual, archive_sha = _load_archive(run)
        sample_positions = tuple(_sample_frames(observations))
        run_samples = []
        for frame in sample_positions:
            observation = observations[frame]
            decisions = {}
            forecasts = {}
            order = tuple(POLICIES)
            if len(samples) % 2:
                order = order[::-1]
            for name in order:
                # Midshift frames have no Policy4 action ledger. Use the
                # diagnostic zero-ledger bootstrap solely for cold solver checks.
                options = {'bootstrap_replan': True} if name == 'rolling_quota' else {}
                policy = POLICIES[name](**options)
                decision, elapsed_ms = _choose(policy, observation)
                policy.commit(decision)  # Isolated simulated ACK at this frame only.
                if decision.action not in forecasts:
                    forecasts[decision.action] = _forecast(observation, decision.action)
                decisions[name] = dict(
                    action=decision.action, kind=decision.kind, elapsed_ms=elapsed_ms,
                    replanned=bool(decision.replanned), solver_status=decision.lp_status,
                    timed_out=bool(decision.timed_out), fallback_reason=decision.fallback_reason,
                    mip_gap=decision.mip_gap, nominal_first_action=forecasts[decision.action],
                    decision_summary=_confirmed(policy, 1),
                )
            row = dict(run_index=run, frame=frame, archive_sha256=archive_sha,
                       observation_sha256=sha256(observation.tobytes()).hexdigest(),
                       recorded_action=int(actual[frame]), decisions=decisions)
            run_samples.append(row)
            samples.append(row)
        for name, factory in POLICIES.items():
            first = next((row['frame'] for row in run_samples
                          if row['decisions'][name]['action'] == row['recorded_action']), 0)
            prefixes.append(dict(run_index=run, policy=name,
                                 **_prefix(factory, observations, actual, first)))
    if hashes != {path.name: sha256(path.read_bytes()).hexdigest() for path in sources}:
        raise ValueError('a reviewed source changed during verification')
    return dict(
        schema='or-paper-milps-sealed-frame-review-v1',
        scope='archived snapshots and nominal first-action forecasts, zero new physical shifts',
        quota_cold_snapshot_has_no_historical_ledger=True,
        protected_core_sha256=hashes.pop(CORE.name),
        policy_sources=hashes,
        physical_shifts_started=0,
        groups={name: _summarize([row['decisions'][name] for row in samples]) for name in POLICIES},
        differing_first_actions=sum(row['decisions']['rolling_quota']['action'] !=
                                    row['decisions']['trip_fifo']['action'] for row in samples),
        prefixes=prefixes,
        snapshots=samples,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New JSON evidence path under reports/')
    arguments = parser.parse_args()
    print(json.dumps(_publish_json(arguments.output, review()), ensure_ascii=False))
