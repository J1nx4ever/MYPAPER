from pathlib import Path
from types import SimpleNamespace

import h5py
import pytest

from flow_matching_plan import FlowPlanError
from flow_matching_policy import FlowMatchingPolicy, _Offer, _need, _reserve, _truck_offers


@pytest.fixture
def opening_frame():
    project = Path(__file__).resolve().parents[5]
    archive = project / "runs/bg128-continuous-mixed-20261120-s02/evaluation/nominal/new/material/run_10004025.h5"
    with h5py.File(archive) as sealed:
        return sealed["states"][0]


def test_demand_inverts_dispatch_time_and_preserves_due_time():
    first = SimpleNamespace(start=0.0, stop=1200.0, tons=300.0)
    empty = SimpleNamespace(start=1200.0, stop=1800.0, tons=0.0)
    last = SimpleNamespace(start=1800.0, stop=2100.0, tons=75.0)
    assert _need((first, empty, last), 150.0) == (600.0, first)
    assert _need((first, empty, last), 300.0) == (1800.0, last)
    assert _need((first, empty, last), 375.0) == (float("inf"), None)


@pytest.mark.parametrize(("solver_status", "receipt_status"),
                         [("optimal", "optimal"), ("feasible_stage_one", "feasible_limit")])
def test_opening_match_waits_for_ack_and_tracks_actual_route(opening_frame, solver_status, receipt_status):
    def fixed_plan(observation, state, *, deadline):
        truck_types = tuple(int(row[1:4].argmax()) for row in observation[27:1107].reshape(9, 120))
        route = _truck_offers(state, truck_types)[2].route
        return SimpleNamespace(
            now=state.now, routes={route: (SimpleNamespace(start=state.now, stop=900.0, tons=300.0),)},
            status=solver_status, ore_shortfall_tons=0.0, waste_shortfall_tons=0.0, grade_slack_tons=(),
            grade_windows=(), accepts_extra=lambda *args, **kwargs: False,
        )

    policy = FlowMatchingPolicy(solve_plan=fixed_plan)
    choice = policy.choose(opening_frame, opening_frame[19:23])
    assert choice.matching_kind == "need_time"
    assert choice.action == 2
    assert choice.replanned
    assert choice.lp_status == receipt_status
    with pytest.raises(ValueError, match="pending"):
        policy.summary()
    policy.commit(choice)
    policy.commit(choice)
    report = policy.summary()
    assert report["branch_counts"] == {"need_time": 1}
    assert report["accepted_solver_outcomes"] == {solver_status: 1}
    assert len(report["route_periods"]) == 1
    assert report["route_periods"][0]["confirmed_tons"] == choice.assigned_tons


def test_failed_plan_returns_masked_safety_action_and_reason(opening_frame):
    def failing_plan(observation, state, *, deadline):
        raise FlowPlanError("test planner failure")

    policy = FlowMatchingPolicy(solve_plan=failing_plan)
    choice = policy.choose(opening_frame, opening_frame[19:23])
    assert choice.kind == "fallback"
    assert choice.fallback_reason == "flow_plan_unavailable"
    assert choice.plan_error == "test planner failure"
    assert opening_frame[19 + choice.action]
    policy.commit(choice)


def test_waste_reservation_checks_the_live_action_before_future_assignment():
    state = SimpleNamespace(current_truck=3, _available_tons=[10_000.0] * 4,
                            _waste_committed=lambda: 5900.0)
    live = _Offer(3, (1, 1, 3, 1), 800.0, 147.0, 60.0, 0.0)
    future = _Offer(0, (0, 1, 3, 1), 880.0, 91.7, 30.0, 0.0)
    assert _reserve(live, state, ())
    assert not _reserve(live, state, (future,))
