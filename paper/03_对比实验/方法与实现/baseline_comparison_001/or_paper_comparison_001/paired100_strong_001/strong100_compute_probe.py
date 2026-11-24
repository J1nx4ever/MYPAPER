"""Measure CPU and wall time for the main three methods on the same sealed frames."""

import argparse
from collections import Counter
from hashlib import sha256
from math import fsum
import os
from pathlib import Path
import platform
import sys
from time import perf_counter_ns, process_time_ns

import numpy as np
import scipy
import torch

PLAN = Path(__file__).resolve().parent
ROOT = PLAN.parents[5]
for directory in (ROOT, PLAN.parent):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from milp_smoke_campaign import METHOD as quota_method
import grade_campaign
from change_plans.token_attention_001.business_graph_summary_009.baseline_comparison_001.dispatch_flow_matching_001.grade_control_001.grade_policy import (
    GradeControlPolicy,
)
from change_plans.token_attention_001.business_graph_summary_009.baseline_comparison_001.or_paper_comparison_001.review_milp_offline import (
    RUNS, _load_archive, _sample_frames,
)
from rolling_quota_policy import RollingQuotaPolicy
from traditional_dispatch_v1.runtime import _actor_choice, _load_actor, _publish_json, _require
from business_graph_summary_campaign_v1.plan_io import load_reference, reference


V2_SUITE = PLAN.parents[1] / 'paired_v2_100_001/prepared_001/suite.json'
METHODS = ('bg128', 'dispatch_grade', 'rolling_quota_replan', 'rolling_quota_tracking')


def _construct(name, actor, generator):
    if name == 'bg128':
        return lambda observation, mask: _actor_choice(actor, generator, observation)
    if name == 'dispatch_grade':
        settings = {key: grade_campaign.METHOD[key] for key in
                    ('planning_ms', 'replan_seconds', 'match_trucks', 'soft_replan_seconds')}
        return GradeControlPolicy(**settings).choose
    settings = dict(quota_method['policy_parameters']['rolling_quota'])
    settings.update(bootstrap_replan=name == 'rolling_quota_replan', rolling=name == 'rolling_quota_replan')
    return RollingQuotaPolicy(planning_ms=quota_method['planning_ms'], **settings).choose


def _timed(method, observation):
    mask = observation[19:23].copy()
    original = observation.tobytes()
    wall_start, cpu_start = perf_counter_ns(), process_time_ns()
    result = method(observation, mask)
    cpu_ms = (process_time_ns() - cpu_start) / 1e6
    wall_ms = (perf_counter_ns() - wall_start) / 1e6
    action = result if type(result) is int else result.action
    _require(type(action) is int and 0 <= action < len(mask) and bool(mask[action])
             and observation.tobytes() == original, 'probe returned an illegal action or altered an archived frame')
    return dict(action=action, cpu_ms=cpu_ms, wall_ms=wall_ms,
                solver_status=getattr(result, 'lp_status', None),
                outcome=getattr(result, 'kind', 'sampled'),
                replanned=bool(getattr(result, 'replanned', False)))


def _summarize(rows):
    result = {}
    for key in ('wall_ms', 'cpu_ms'):
        values = [row[key] for row in rows]
        result[f'{key}_p50'] = float(np.quantile(values, .50))
        result[f'{key}_p95'] = float(np.quantile(values, .95))
        result[f'{key}_max'] = max(values)
        result[f'{key}_total'] = fsum(values)
    return dict(result, frames=len(rows),
                outcomes=dict(Counter(row['outcome'] for row in rows)),
                solver_statuses=dict(Counter(row['solver_status'] for row in rows if row['solver_status'])),
                replans=sum(row['replanned'] for row in rows))


def review():
    suite_ref = reference(V2_SUITE)
    suite = load_reference(suite_ref)
    torch.set_num_threads(suite.get('cpu_threads', 2))
    actor, digest = _load_actor(suite)
    generator = torch.Generator(device='cpu').manual_seed(90221)
    sampled = []
    for run in RUNS:
        states, _, archive_sha = _load_archive(run)
        sampled.extend((run, index, archive_sha, states[index]) for index in _sample_frames(states))

    # One untimed call per method initializes its libraries. Every measured LP
    # or MILP starts from a fresh policy on that frame to expose replan cost.
    for name in METHODS:
        _timed(_construct(name, actor, generator), sampled[0][3])
    rows = []
    for ordinal, (run, index, archive_sha, observation) in enumerate(sampled):
        shift = ordinal % len(METHODS)
        order = METHODS[shift:] + METHODS[:shift]
        choices = {name: _timed(_construct(name, actor, generator), observation) for name in order}
        rows.append(dict(run_index=run, frame=index, source_hdf5_sha256=archive_sha,
                         observation_sha256=sha256(observation.tobytes()).hexdigest(), choices=choices))
    return dict(
        schema='bg128-or-strong-same-host-compute-v1',
        scope='same archived observations in one process; decision only; no RPC or simulation',
        physical_shifts_started=0,
        source_v2_suite=suite_ref,
        checkpoint=suite['actor'],
        checkpoint_digest=digest,
        method_scopes={
            'bg128': 'frozen s02/u32 feature preparation, inference and sampled action',
            'dispatch_grade': 'cold rolling LP, matching and FIFO grade correction',
            'rolling_quota_replan': 'forced isolated MILP replan with no prior ACK ledger, then Policy4',
            'rolling_quota_tracking': 'Policy4 tracking of original table with no replan on this frame',
        },
        execution=dict(
            cpu_model=next((line.split(':', 1)[1].strip() for line in
                            Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),
                           platform.machine()),
            affinity_cpu_count=len(os.sched_getaffinity(0)),
            numpy=np.__version__, scipy=scipy.__version__, torch=str(torch.__version__),
            torch_threads=torch.get_num_threads(),
            omp_num_threads=os.environ.get('OMP_NUM_THREADS'),
            openblas_num_threads=os.environ.get('OPENBLAS_NUM_THREADS'),
        ),
        groups={name: _summarize([row['choices'][name] for row in rows]) for name in METHODS},
        rows=rows,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    arguments = parser.parse_args()
    print(_publish_json(arguments.output, review()), flush=True)
