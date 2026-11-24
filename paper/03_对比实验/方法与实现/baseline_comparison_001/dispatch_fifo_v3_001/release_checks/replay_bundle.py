"""Replay bundled snapshots against independently sealed expected actions and costs."""
import hashlib
import json
import math
from pathlib import Path
import sys
from time import perf_counter


def require(condition, message):
    if not condition:
        raise ValueError(message)


def replay():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'manifest.json').read_bytes())
    for name, expected_sha in manifest['files'].items():
        require(hashlib.sha256((root / name).read_bytes()).hexdigest() == expected_sha,
                f'file digest mismatch for {name}')
    source = root / 'src'
    sys.path[:0] = [str(source / manifest['policy_directory']), str(source)]
    import numpy as np
    from fifo_policy import FIFODispatchPolicy

    imported = set()
    for module in list(sys.modules.values()):
        origin = getattr(getattr(module, '__spec__', None), 'origin', None)
        if isinstance(origin, str) and origin.endswith('.py') and Path(origin).is_relative_to(source):
            imported.add(Path(origin).relative_to(root).as_posix())
    expected_sources = {name for name in manifest['files'] if name.startswith('src/') and name.endswith('.py')}
    require(imported == expected_sources, 'decision imports differ from the bundled source inventory')
    states = np.load(root / 'observations.npy', allow_pickle=False)
    rows = json.loads((root / 'expected.json').read_bytes())
    require(states.shape == (24, 1519) and states.dtype == np.float32 and len(rows) == 24,
            'expected 24 sealed float32 observations of 1519 fields each')
    latencies = []
    for state, saved in zip(states, rows, strict=True):
        where = f"run {saved['run_index']} frame {saved['frame']}"
        require(hashlib.sha256(state.tobytes()).hexdigest() == saved['observation_sha256'],
                f'raw observation digest mismatch at {where}')
        policy = FIFODispatchPolicy(**manifest['policy_options'])
        started = perf_counter()
        choice = policy.choose(state, state[19:23])
        elapsed = 1000 * (perf_counter() - started)
        policy.commit(choice)
        require(not choice.timed_out and choice.kind == 'planned' and not choice.fifo_model_error
                and not choice.fifo_guidance_error and choice.fifo_candidates == choice.fifo_roots_completed,
                f'incomplete default-budget decision at {where}, {elapsed:.1f} ms ({choice})')
        actual = dict(action=choice.action, fifo_selected_actions=list(choice.fifo_selected_actions),
                      fifo_selected_truck_ids=list(choice.fifo_selected_truck_ids),
                      fifo_root_feasible=choice.fifo_root_feasible, fifo_selected_feasible=choice.fifo_selected_feasible,
                      fifo_roots_completed=choice.fifo_roots_completed, lp_status=choice.lp_status)
        for key, value in actual.items():
            require(value == saved[key], f'{where} {key} expected {saved[key]}, got {value}')
        for key in ('fifo_root_cost', 'fifo_selected_cost', 'fifo_selected_industrial_loss'):
            value = getattr(choice, key)
            require(math.isclose(value, saved[key], rel_tol=1e-10, abs_tol=1e-10),
                    f'{where} {key} expected {saved[key]}, got {value}')
        latencies.append(elapsed)
    return dict(status='PASS', project_source_files=len(imported), frames=len(rows),
                distinct_frames=len({row['observation_sha256'] for row in rows}),
                archives=len({row['archive_sha256'] for row in rows}), decisions_matched=len(rows),
                wall_guard_count=0, planning_ms=manifest['policy_options']['planning_ms'],
                latency_max_ms=max(latencies), latency_p95_ms=float(np.percentile(latencies, 95)))


if __name__ == '__main__':
    print(json.dumps(replay(), indent=2))
