"""Analytical and synthetic tests for the upper-stage haulage LP."""

import numpy as np
import pytest

from grade_commitment_ppo_v1.test_features import fifo_state, put_truck, synthetic_state
from .flow_plan import FlowPlanError, _CommitmentReplay, _mean_overlap, solve_flow_plan
from .nominal import NominalState, SHIFT_END


def fast_snapshot(*, fleet_ready=True, **kwargs):
    state = synthetic_state(**kwargs)
    state[1439:].reshape(16, 5)[:, 2] = .00065
    if fleet_ready:
        trucks = state[27:1107].reshape(9, 120)
        trucks[:, 4:16] = 0
        trucks[:, 4] = 1
    return state


def test_resource_time_overlap_for_uniform_launches_and_crossing_window():
    assert _mean_overlap((0, 10), (5, 8), 2, 4) == pytest.approx(.6)
    assert _mean_overlap((0, 10), (0, 20), 2, 4) == 2
    assert _mean_overlap((0, 10), (15, 20), 2, 4) == 0


def test_initial_plan_has_positive_flows_and_targets_in_physical_units():
    observation = fast_snapshot()
    nominal = NominalState.from_observation(observation)
    before = observation.tobytes()
    plan = solve_flow_plan(observation, nominal)
    assert plan.status == 'optimal'
    assert [period.window for period in plan.periods] == [0, 1, 2, 3]
    assert plan.committed_ore_tons == plan.committed_waste_tons == 0
    assert plan.ore_shortfall_tons == plan.waste_shortfall_tons == 0
    assert plan.predicted_ore_tons >= 16000 - 1e-5
    assert plan.predicted_waste_tons >= 4000 - 1e-5
    assert plan.predicted_ore_tons == pytest.approx(sum(p.predicted_ore_tons for p in plan.periods))
    for period in plan.periods:
        tons, rates = np.asarray(period.tons_by_type_site), np.asarray(period.rate_tph_by_type_site)
        assert tons.shape == rates.shape == (3, 4)
        assert np.allclose(rates * (period.end - period.start) / 3600, tons)
        assert period.predicted_ore_tons <= 5000 + 1e-5
        assert .266 - 1e-8 <= period.predicted_metal_tons / period.predicted_ore_tons <= .274 + 1e-8
    assert 4000 <= sum(np.asarray(p.tons_by_type_site)[:, 3].sum() for p in plan.periods) <= 6000
    assert observation.tobytes() == before and nominal.current_truck == 0


def test_known_fifo_is_credited_once_and_late_new_trips_remain_shortfalls():
    observation = fifo_state(fast_snapshot(window_index=3, remaining_seconds=72), [(100, .27)])
    observation[4], observation[5], observation[7] = 200 / 5000, 54 / 5000, 15000 / 20000
    observation[9] = 3900 / 4000
    plan = solve_flow_plan(observation)
    period, = plan.periods
    assert sum(map(sum, period.tons_by_type_site)) == 0
    assert plan.committed_ore_tons == plan.predicted_ore_tons == pytest.approx(15050, abs=.002)
    assert plan.ore_shortfall_tons == pytest.approx(950, abs=.002)
    assert plan.committed_waste_tons == plan.predicted_waste_tons == pytest.approx(3900, abs=.002)
    assert plan.waste_shortfall_tons == pytest.approx(100, abs=.002)
    assert period.predicted_ore_tons == pytest.approx(250, abs=.002)
    assert period.predicted_metal_tons == pytest.approx(67.5, abs=.002)


def test_existing_waste_commitment_consumes_stop_budget_without_recounting_production():
    observation = fast_snapshot()
    observation[9] = 5990 / 4000
    observation = put_truck(observation, tid=1, phase=5, q=100, ore=False)
    plan = solve_flow_plan(observation)
    assert plan.committed_waste_tons == plan.predicted_waste_tons == pytest.approx(6090, abs=.002)
    assert plan.waste_shortfall_tons == 0
    assert all(np.asarray(p.tons_by_type_site)[:, 3].sum() == 0 for p in plan.periods)


def test_infeasible_current_grade_has_a_high_side_slack_and_later_windows_can_be_filled():
    observation = fast_snapshot(remaining_seconds=30)
    observation[4], observation[5], observation[7] = 500 / 5000, 175 / 5000, 500 / 20000
    plan = solve_flow_plan(observation)
    first = plan.periods[0]
    assert first.predicted_ore_tons == pytest.approx(500)
    assert plan.grade_slack_tons[0] == pytest.approx((0, 38), abs=.002)
    assert plan.periods[1].predicted_ore_tons > 0


def test_positive_feed_is_planned_even_if_shift_production_is_already_met():
    observation = fast_snapshot(window_index=3)
    observation[7], observation[9] = 16000 / 20000, 4000 / 4000
    plan = solve_flow_plan(observation)
    period, = plan.periods
    assert sum(map(sum, period.tons_by_type_site)) > 90.0
    assert period.predicted_ore_tons == pytest.approx(91.7, abs=.002)
    assert period.grade_window_supply_shortfall_tons == 0


def test_parked_and_unreleased_initial_trucks_cannot_contribute_fleet_time():
    observation = fast_snapshot(fleet_ready=False)
    rows = observation[27:1107].reshape(9, 120)
    # The first type-1 truck waits on parked truck 2. A genuine predecessor
    # chain of the root and then truck 6 makes two type-2 trucks actionable.
    for tid, previous in ((3, 2), (6, 0), (7, 6)):
        rows[tid, 4:16] = 0
        rows[tid, 5] = 1
        rows[tid, 81] = (previous + 1) / 9
    projected = _CommitmentReplay.from_observation(observation)
    projected._move(check_deadline=None, dispatch=False)
    releases = projected.dispatch_release_times()
    assert [i for i, timestamp in enumerate(releases) if timestamp == 0] == [0, 6, 7]
    plan = solve_flow_plan(observation)
    assert all(sum(period.tons_by_type_site[1]) == 0 for period in plan.periods)


def test_committed_truck_first_becomes_available_when_its_own_unload_finishes():
    observation = fast_snapshot(fleet_ready=False)
    observation = put_truck(observation, tid=3, phase=5, q=100, ore=False)
    projected = _CommitmentReplay.from_observation(observation)
    projected._move(check_deadline=None, dispatch=False)
    release = projected.dispatch_release_times()[3]
    assert release == pytest.approx(200 + 102, abs=.001)
    assert projected.dispatch_release_times()[4] == SHIFT_END
    plan = solve_flow_plan(observation)
    support = plan.periods[0].dispatch_start_by_type_site
    assert support[0] == (0.0,) * 4
    assert support[1] == (release,) * 4
    assert support[2] == (None,) * 4
    assert plan.periods[1].dispatch_start_by_type_site[1] == (7200.0,) * 4


def test_final_window_crusher_cannot_use_capacity_before_first_possible_new_ore():
    observation = fast_snapshot(window_index=3, remaining_seconds=900)
    world = NominalState.from_observation(observation)
    first_unload = min(world.candidate_timing(action).unload_complete for action in (0, 1, 2))
    causal_capacity = (SHIFT_END - first_unload) * 2500 / 3600
    plan = solve_flow_plan(observation, world)
    assert causal_capacity == pytest.approx(384.79, abs=.01)
    assert plan.predicted_ore_tons == pytest.approx(causal_capacity, abs=.002)


def test_last_partial_load_does_not_wait_for_a_full_truck_loading_pass():
    observation = fast_snapshot(window_index=3, remaining_seconds=259)
    observation[19:27] = 0
    observation[19], observation[23] = 1, 31 / 193
    sites = observation[1107:1295].reshape(4, 47)
    sites[:, 7] = 0
    sites[0, 2], sites[0, 7] = .27, 31 / 30000
    plan = solve_flow_plan(observation)
    assert 0 < plan.predicted_ore_tons < 31
    assert sum(row[0] for row in plan.periods[0].tons_by_type_site) <= 31.0 + .002


def test_unusable_observation_is_an_explicit_error():
    observation = fast_snapshot()
    observation[1302] = .1  # FIFO, reservations and free capacity cannot all be correct.
    with pytest.raises(FlowPlanError, match='crusher_capacity_inconsistent'):
        solve_flow_plan(observation)
