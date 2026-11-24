"""Run the frozen controls, checking the historical match before any dispatch."""

from collections import Counter
from functools import lru_cache
from hashlib import sha256
from math import isfinite
import os
from pathlib import Path
import platform
from time import perf_counter_ns, process_time_ns

import numpy
import scipy

from strong100_campaign import CAMPAIGN, CONTROL, EXECUTION, SCHEMA
from baseline1_v1.collection import wait_for_java_start
from business_graph_summary_campaign_v1.plan_io import load_reference, reference
from change_plans.token_attention_001.business_graph_summary_009.baseline_comparison_001.dispatch_flow_matching_001.grade_control_001.grade_policy import (
    GradeControlPolicy,
)
from change_plans.token_attention_001.business_graph_summary_009.baseline_comparison_001.dispatch_inspired_001.dispatch_runtime import (
    _collect as collect_dispatch,
)
from milp_smoke_runtime import _collect as collect_quota
from rolling_quota_policy import RollingQuotaPolicy
from python.dispatch_bridge import DispatchCallbackBridge
from python.dispatch_env import DispatchEnv
from python.dispatch_service import create_server
from python.episode_runner import ParameterVariationEpisodeRunner
from traditional_dispatch_v1.runtime import (
    _checked_reference, _percentiles, _publish_json, _publish_ready, _require,
)
from traditional_dispatch_v2.runtime import _verify_anchor


def make_policy(kind, method):
    if kind == 'dispatch_grade':
        return GradeControlPolicy(planning_ms=method['planning_ms'], **method['policy_parameters'][kind])
    if kind == 'rolling_quota':
        return RollingQuotaPolicy(planning_ms=method['planning_ms'], **method['policy_parameters'][kind])
    raise ValueError(f'unknown strong baseline {kind!r}')


def _read_text(path):
    try:
        return Path(path).read_text(encoding='utf-8').strip()
    except OSError:
        return ''


@lru_cache(maxsize=1)
def process_context():
    cpu_model = next((line.partition(':')[2].strip()
                      for line in _read_text('/proc/cpuinfo').splitlines()
                      if line.partition(':')[0].strip() == 'model name'), None)
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None
    thread_settings = ('OMP_NUM_THREADS', 'OMP_DYNAMIC', 'MKL_NUM_THREADS', 'MKL_DYNAMIC',
                       'OPENBLAS_NUM_THREADS', 'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
    return dict(
        system=platform.system(), release=platform.release(), machine=platform.machine(),
        cpu_model=cpu_model or platform.processor() or None, logical_cpus=os.cpu_count(),
        process_affinity=affinity, cgroup_cpu_max=_read_text('/sys/fs/cgroup/cpu.max') or None,
        python_version=platform.python_version(), scipy_version=scipy.__version__,
        numpy_version=numpy.__version__,
        thread_environment={name: os.environ.get(name) for name in thread_settings},
    )


class TimedPolicy:
    """Measure the complete policy call, including native solver CPU time."""

    def __init__(self, policy):
        self.policy = policy
        self.choose_wall_ns = []
        self.choose_cpu_ns = []
        self.commit_wall_ns = 0
        self.commit_cpu_ns = 0
        self.committed = 0
        self.solver_status = Counter()
        self.solver_time_limits = 0
        self.solver_accepts = 0
        self.max_mip_gap = None

    def choose(self, observation, mask):
        cpu_start = process_time_ns()
        wall_start = perf_counter_ns()
        choice = self.policy.choose(observation, mask)
        wall_elapsed = perf_counter_ns() - wall_start
        cpu_elapsed = process_time_ns() - cpu_start
        self.choose_wall_ns.append(wall_elapsed)
        self.choose_cpu_ns.append(cpu_elapsed)
        return choice

    def commit(self, choice):
        cpu_start = process_time_ns()
        wall_start = perf_counter_ns()
        self.policy.commit(choice)
        self.commit_wall_ns += perf_counter_ns() - wall_start
        self.commit_cpu_ns += process_time_ns() - cpu_start
        self.committed += 1
        if choice.lp_status is not None:
            self.solver_status[choice.lp_status] += 1
        self.solver_time_limits += int(choice.timed_out)
        self.solver_accepts += int(choice.replanned)
        gap = getattr(choice, 'mip_gap', None)
        if gap is not None:
            _require(isfinite(gap) and gap >= 0, 'invalid relative MILP gap')
            self.max_mip_gap = max(float(gap), self.max_mip_gap or 0.0)

    def summary(self):
        return self.policy.summary()

    def compute(self, decisions):
        _require(decisions > 0 and len(self.choose_wall_ns) == len(self.choose_cpu_ns) == self.committed == decisions,
                 'timed policy calls and acknowledged decisions differ')
        wall_total = sum(self.choose_wall_ns)
        cpu_total = sum(self.choose_cpu_ns)
        return dict(
            policy_wall_seconds=wall_total / 1e9, policy_cpu_seconds=cpu_total / 1e9,
            policy_wall_ms_per_decision=wall_total / (decisions * 1e6),
            policy_cpu_ms_per_decision=cpu_total / (decisions * 1e6),
            policy_cpu_latency_ms=_percentiles([elapsed / 1e6 for elapsed in self.choose_cpu_ns]),
            policy_commit_wall_seconds=self.commit_wall_ns / 1e9,
            policy_commit_cpu_seconds=self.commit_cpu_ns / 1e9,
            process_context=process_context(),
        )


class _HistoricalStartEnv(DispatchEnv):
    def __init__(self, bridge, runner, timeout, anchor):
        super().__init__(bridge, runner, timeout)
        self.anchor = anchor

    def reset(self, *, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        record = dict(
            experiment_seed=info['experiment_seed'],
            initial_state=dict(
                observation_sha256=sha256(observation.tobytes()).hexdigest(),
                mask=info['action_mask'], truck_id=info['truck_id'],
                sim_time_seconds=info['sim_time_seconds'],
            ),
            actual_rpc_configuration={key: value for key, value in info['reward_components'].items()
                                      if key.startswith('config.')},
        )
        _verify_anchor(record, self.anchor)
        return observation, info


def _anchor(job):
    anchor = load_reference(job['anchor_receipt'])
    _require(anchor['policy_kind'] == 'bg128' and anchor['status'] == 'COMPLETE'
             and anchor['normal_completion'] is True and anchor['optimizer_updates'] == 0
             and anchor['actor_sampling_seed'] > 0
             and anchor['run_index'] == job['anchor_run_index']
             and anchor['pair_ordinal'] == job['pair_ordinal']
             and anchor['experiment_seed'] == job['experiment_seed']
             and anchor['scene_id'] == job['scene_id'],
             'historical BG128 evaluation receipt differs from the frozen matched slot')
    return anchor


def _collect(env, job, method, policy):
    if job['policy_kind'] == 'dispatch_grade':
        dispatch_method = dict(method, **method['policy_parameters']['dispatch_grade'])
        result = collect_dispatch(env, job, dispatch_method, policy)
        matching = policy.summary()
        result['engineering']['route_matching'] = matching
        gates = result['engineering_acceptance']
        gates['acknowledged_route_dispatches'] = (
            sum(matching['branch_counts'].values()) == result['engineering']['decisions'])
        gates['passed'] = all(value for key, value in gates.items() if key != 'passed')
    else:
        result = collect_quota(env, job, method, policy)
    engineering = result['engineering']
    engineering.update(
        compute=policy.compute(engineering['decisions']),
        solver_status_counts=dict(policy.solver_status),
        solver_attempt_requests=sum(policy.solver_status.values()),
        solver_time_limit_requests=policy.solver_time_limits,
        accepted_solver_requests=policy.solver_accepts,
        max_mip_gap=policy.max_mip_gap,
    )
    return result


def run_shift(suite, ledger, job):
    run = job['run_index']
    startup = dict(campaign_id=CAMPAIGN, control_directory=str(CONTROL),
                   master_plan_sha256=suite['master_plan']['sha256'])
    output = Path(job['python_output']) / 'evaluation_receipt.json'
    claim = CONTROL / 'java_claims' / f'run_{run}.once.claim'
    ready = CONTROL / 'ready' / f'run_{run}.properties'
    _require(not any(path.exists() for path in (output, claim, ready, CONTROL / 'STOP.properties')),
             'run identity already has physical evidence')
    _require(job['actor_seed'] == job['actor_sampling_seed'] == 0, 'deterministic baseline has an Actor seed')
    method = suite['method']
    policy = TimedPolicy(make_policy(job['policy_kind'], method))
    bridge = DispatchCallbackBridge(expected_episode_metadata=[(run, job['experiment_seed'])])
    timeout = EXECUTION['callback_timeout_seconds']
    env = _HistoricalStartEnv(bridge, ParameterVariationEpisodeRunner(1), timeout, _anchor(job))
    server = create_server(bridge, host=EXECUTION['host'], port=EXECUTION['port'])
    claimed = False
    try:
        server.start()
        ledger.transition(job, 'PLANNED', 'CLAIMED')
        claimed = True
        _publish_ready(startup, job)
        wait_for_java_start(startup, job, job['phase_config']['sha256'], timeout)
        result = _collect(env, job, method, policy)
        _checked_reference(suite['protected_core'])
        fields = ('run_index', 'experiment_seed', 'recording_id', 'actor_sampling_seed', 'policy_kind',
                  'scene_id', 'phase_config', 'pair_ordinal', 'anchor_receipt', 'anchor_run_index')
        receipt = dict(result, schema=SCHEMA + '-receipt', campaign_id=CAMPAIGN, status='COMPLETE',
                       normal_completion=True, completed_shifts=1, optimizer_updates=0,
                       java_start_claim=reference(claim), **{key: job[key] for key in fields})
        pinned = _publish_json(output, receipt)
        ledger.transition(job, 'CLAIMED', 'COMPLETE', pinned)
        claimed = False
        return receipt, pinned
    except BaseException as error:
        if claimed:
            ledger.transition(job, 'CLAIMED', 'UNKNOWN', dict(reason=repr(error)))
        raise
    finally:
        env.close()
        server.stop(grace=0).wait()
