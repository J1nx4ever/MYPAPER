"""Projection, matching and collector tests without a simulator or network."""
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from single_critic_task_query_v1.test_scenario_bridge import InProcessDispatch
from traditional_dispatch_v2 import dispatch_policy as v2_dispatch
from traditional_dispatch_v2.test_runtime import _fixture_job
from traditional_dispatch_v2.test_flow_plan import fast_snapshot
from python.dispatch_bridge import DispatchCallbackBridge, ProtocolError
from python.dispatch_env import DispatchEnv

from fifo_policy import FIFOChoice
import v3_flow as flow
import v3_matching as matching
import v3_runtime as runtime


def test_committed_backlog_exact_integral_straddles_arrival_and_depletion():
    replay = SimpleNamespace(origin=0., initial_stock=3., _crusher_rate=1., ore_arrivals=[(4., 3.)])
    launch = SimpleNamespace(departure=(0., 10.), route=SimpleNamespace(q=1., dump_end=0.))
    # The busy plateaus start processing at t=3 and t=7. On the gaps,
    # each prospective truck starts immediately; triangular overlaps split
    # the contribution at each quality-window boundary.
    fractions = flow._ore_fractions(launch, [(0., 4.), (4., 8.), (8., 12.)], flow._backlog_segments(replay), 1.)
    assert fractions == pytest.approx([.35, .4, .25])


def test_planning_matrix_projects_real_candidates_and_checks_clipped_rows():
    plan = flow.solve_flow_plan(fast_snapshot())
    assert plan.status == 'optimal' and len(plan.periods) == 4
    assert all(0 <= row.predicted_ore_tons <= 5000.002 for row in plan.periods)
    raw = SimpleNamespace(status=0, x=np.array([-.001, 1.]))
    with pytest.raises(flow.FlowPlanError, match='infeasible_after_nonnegative_clip'):
        flow._clipped_feasible(raw, np.array([[1., 1.]]), np.array([.9975]))


def test_coverage_shortage_is_charged_for_entire_unmatched_part_and_inverse_has_no_early_time():
    early = v2_dispatch._Target(0., 10., 90.)
    late = v2_dispatch._Target(20., 30., 20.)
    assert matching.need_time((early, late), 90.) == (20., 2.)
    assert 9.99 < matching.need_time((early, late), 89.999999)[0] <= 10.
    root = v2_dispatch._Option(0, 100., 0., 0., 10., 4.)
    cohort = [v2_dispatch._TruckOptions(1, (root,), 0.)]
    original = [[0.]] * 4
    empty = [[], [], [], []]
    assert matching.loss([root], cohort, [(early,), (), (), ()], [0.] * 4, original, empty) == 10.
    # At equal arrival time a visible truck with larger ID goes after the
    # newly projected lower-ID truck, consistent with the nominal event heap.
    visible = [[(0., 1, 2, 1., None)], [], [], []]
    assert matching.loss([root], cohort, [(early,), (), (), ()], [0.] * 4, original, visible) == 10.


def test_small_roundoff_does_not_create_second_assignment_but_real_partial_load_remains():
    option = v2_dispatch._Option(0, 100., 0., 0., 20., 10.)
    load = SimpleNamespace(ore=True, service=lambda q: q / 10.)
    assert matching.reserve((option, option), [100.001, 0., 0., 0.], [load], 0.) is None
    root, following = matching.reserve((option, option), [125., 0., 0., 0.], [load], 0.)
    assert root.tons == 100. and following.tons == 25. and following.loading_seconds == 2.5


class _Policy:
    def __init__(self, *, compared=True, model_error=False):
        self.calls = []
        self.compared = compared
        self.model_error = model_error

    def choose(self, _observation, _mask):
        self.calls.append('choose')
        return FIFOChoice(0, 'planned', replanned=True, lp_status='optimal', matched_trucks=2 if self.compared else 1,
                          assignments_scored=4, flow_shortfall_ore_tons=0., flow_shortfall_waste_tons=0.,
                          flow_grade_slack_tons=0., fifo_candidates=4, fifo_roots_completed=4,
                          fifo_trials_nominated=2, fifo_trials_completed=1, fifo_trials_compared=int(self.compared),
                          fifo_model_error='nominal_event_clock_stalled' if self.model_error else None,
                          timed_out=not self.compared and not self.model_error,
                          fifo_guard_restored_root=not self.compared and not self.model_error,
                          fifo_selected_actions=(0, 1), fifo_selected_truck_ids=(0, 1),
                          fifo_root_cost=1., fifo_selected_cost=1., fifo_root_feasible=True,
                          fifo_selected_feasible=True)

    def commit(self, _choice):
        self.calls.append('commit')


def _run_rpc(*, wrong_anchor=False, wrong_echo=None, compared=True, model_error=False):
    job, requests = _fixture_job(runtime.POLICY, wrong_echo=wrong_echo)
    bridge = DispatchCallbackBridge(expected_episode_metadata=[(801, 701)])
    env = DispatchEnv(bridge, callback_timeout=2.)
    producer = InProcessDispatch(bridge, requests)
    root = requests[0]
    sha = hashlib.sha256(np.asarray(root.obs.values, dtype=np.float32).tobytes()).hexdigest()
    anchor = dict(experiment_seed=701, actual_rpc_configuration=job['expected_rpc_config'],
                  initial_state=dict(observation_sha256=sha if not wrong_anchor else 'wrong',
                                     mask=[True] * 4, truck_id=0, sim_time_seconds=0.))
    policy = _Policy(compared=compared, model_error=model_error)
    producer.start()
    try:
        if wrong_anchor:
            with pytest.raises(ValueError, match='differs within the pair'):
                runtime._collect(env, job, {'match_trucks': 3, 'response_target_ms': 1000}, anchor, policy)
            assert not policy.calls
            return
        if wrong_echo is not None:
            with pytest.raises(ProtocolError, match='previous_action disagrees'):
                runtime._collect(env, job, {'match_trucks': 3, 'response_target_ms': 1000}, anchor, policy)
            assert policy.calls == ['choose', 'commit', 'choose']
            return
        row = runtime._collect(env, job, {'match_trucks': 3, 'response_target_ms': 1000}, anchor, policy)
        acceptance = row['engineering_acceptance']
        assert acceptance['live_multitruck_match_observed'] is compared
        assert acceptance['fifo_executable_comparison_observed'] is compared
        assert acceptance['fifo_models_valid'] is not model_error
        assert acceptance['passed'] is (compared and not model_error)
        assert row['engineering']['first_completed_fifo_trial']['request_ordinal'] == 0
        assert bool(row['engineering']['first_compared_fifo_trial']) is compared
        assert row['engineering']['action_counts']['live_multitruck_matches'] == 2 * int(compared)
        assert bool(row['engineering']['fifo_planning_errors']) is model_error
        assert row['engineering']['action_counts']['fifo_trials_completed'] == 2
        assert row['engineering']['decisions'] == 2 and len(row['physical']['windows']) == 4
        assert policy.calls == ['choose', 'commit', 'choose', 'commit']
    finally:
        env.close()
        producer.join()


def test_real_bridge_start_anchor_and_terminal_echo():
    _run_rpc()
    _run_rpc(wrong_anchor=True)
    _run_rpc(wrong_echo=2)


def test_restored_or_model_aborted_trials_cannot_pass_multitruck_acceptance():
    _run_rpc(compared=False)
    _run_rpc(compared=False, model_error=True)
