"""Check the frozen FIFO candidate selection against sealed decision-time snapshots."""
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np

from v3_plan import METHOD, ROOT
from fifo_policy import FIFODispatchPolicy


ARCHIVE = ROOT / 'runs/bg128-continuous-mixed-20261120-s02/evaluation/nominal/new/material'
OPTS = ('planning_ms', 'replan_seconds', 'match_trucks', 'cohort_trial_limit')


def sealed_frames():
    """Take the first and last decision in every window of each sealed shift."""
    for run in (10004025, 10004026, 10004027):
        file = ARCHIVE / f'run_{run}.h5'
        digest = sha256(file.read_bytes()).hexdigest()
        seal = json.loads(file.with_suffix('.h5.seal.json').read_bytes())
        if digest != seal['hdf5_sha256']:
            raise ValueError(f'changed observation archive {file}')
        with h5py.File(file) as archived:
            states = archived['states'][:-1]
        windows = np.rint(states[:, 2] * 4).astype(int)
        for window in range(4):
            candidates = np.flatnonzero(windows == window)
            for index in (int(candidates[0]), int(candidates[-1])):
                yield run, index, digest, states[index]


def review():
    samples = []
    for run, frame, seal, observation in sealed_frames():
        mask = observation[19:23]
        # A cold solver at every sampled frame exercises the upper LP as well.
        policy = FIFODispatchPolicy(**{k: METHOD[k] for k in OPTS})
        started = perf_counter()
        choice = policy.choose(observation, mask)
        elapsed_ms = (perf_counter() - started) * 1000
        if not mask[choice.action]:
            raise ValueError(f'illegal action {choice.action} at {run}/{frame}')
        policy.commit(choice)
        samples.append(dict(run_index=run, frame=frame, archive_sha256=seal, elapsed_ms=elapsed_ms,
                            observation_sha256=sha256(observation.tobytes()).hexdigest(),
                            **asdict(choice)))
    times = [row['elapsed_ms'] for row in samples]
    return dict(scope='sealed historical observation snapshots, not new physical shifts',
                frames=len(samples), distinct_observations=len({r['observation_sha256'] for r in samples}),
                decisions=dict(Counter(row['kind'] for row in samples)),
                planning_timeouts=sum(row['timed_out'] for row in samples),
                cold_planner_latency_p95_ms=float(np.quantile(times, .95)),
                cold_planner_latency_max_ms=max(times), rows=samples)


if __name__ == '__main__':
    print(json.dumps(review(), allow_nan=False, indent=2))
