"""Analytical checks for the shared deterministic rollout."""
import math

import pytest

from grade_commitment_ppo_v1.test_features import fifo_state, put_truck, synthetic_state

from .nominal import NominalModelError, NominalState, SHIFT_END


def short_roads(snapshot):
    # All edges have length 6.5 m. The nominal empty and loaded speeds
    # therefore cost 1.0 and 1.3 seconds per edge, respectively.
    roads = snapshot[1439:].reshape(16, 5)
    roads[:, 2] = .00065
    return snapshot


def only_root_load(snapshot, action):
    tons = snapshot[23 + action] * 193.0
    snapshot[19:27] = 0
    snapshot[19 + action] = 1
    snapshot[23 + action] = tons / 193.0
    for aid, row in enumerate(snapshot[1107:1295].reshape(4, 47)):
        row[7] = tons / 30000.0 if aid == action else 0.0
    return snapshot


def put_startup_truck(snapshot, truck, predecessor):
    row = snapshot[27 + 120 * truck:147 + 120 * truck]
    row[4:16] = 0
    row[5] = 1
    row[81] = (predecessor + 1) / 9


def test_root_candidate_uses_exact_mask_and_its_fifo_batch_crosses_window():
    s = only_root_load(short_roads(synthetic_state(remaining_seconds=370)), 0)
    s[4], s[5], s[7] = 100 / 5000, 27 / 5000, 100 / 20000
    raw = s.tobytes()
    initial = NominalState.from_observation(s)
    candidate, = initial.candidates()
    assert candidate.action == 0 and candidate.tons == pytest.approx(91.7)
    timing = initial.candidate_timing(0)
    assert timing.wait_seconds == 0 and initial.now < timing.unload_complete < 7200
    branch = initial.clone()
    branch.apply_action(0)
    branch.advance()
    assert branch.finished and branch.now == SHIFT_END
    score = branch.score_inputs()
    first_processed = (7200 - timing.unload_complete) * 2500 / 3600
    assert score.root_cross_window
    assert score.fifo_q == pytest.approx((100 + first_processed, 91.7 - first_processed, 0, 0))
    assert score.unload_q == pytest.approx((191.7, 0, 0, 0))
    assert sum(score.fifo_m) == pytest.approx(27 + 91.7 * .35)
    assert sum(score.fifo_m) == pytest.approx(sum(score.unload_m))
    assert score.ore_tons == pytest.approx(191.7)
    assert initial.current_truck == 0 and initial.candidates() == (candidate,)
    assert s.tobytes() == raw


def test_fifo_capacity_reservation_and_waiters_are_counted_once():
    s = only_root_load(short_roads(synthetic_state()), 3)
    s = fifo_state(s, [(300, .3), (120, .2)])
    s = put_truck(s, tid=1, phase=9, q=80, grade=.35)
    s = put_truck(s, tid=2, phase=7, q=100, grade=.4)
    s[1297:1303] = [2 / 3, 2 / 3, 1 / 3, .84, .16, 0.0]
    s[1295 + 30] = 3 / 9
    s[1295 + 39] = 2 / 9
    world = NominalState.from_observation(s)
    q, m, ore, waste = world.tail_context()
    assert (q, m, ore, waste) == pytest.approx((420, 114, 600, 0))

    world.apply_action(3)
    assert world.tail_context()[3] == pytest.approx(91.7)
    world.advance()
    result = world.score_inputs()
    assert world.finished
    assert result.wait_seconds == pytest.approx(100 * 3600 / 2500)
    assert result.fifo_q == pytest.approx((600, 0, 0, 0))
    assert result.fifo_m == pytest.approx((182, 0, 0, 0))
    assert result.unload_q == pytest.approx(result.fifo_q)
    assert result.waste_tons == pytest.approx(91.7)
    assert not result.root_cross_window


def test_current_shovel_owner_delays_same_site_but_not_other_site():
    s = short_roads(synthetic_state())
    s = put_truck(s, tid=1, phase=4, q=91.7, loaded=False, grade=.2, target=0)
    s[1107 + 15] = 1
    s[1107 + 38] = 2 / 9
    s[1297] = 2 / 3
    world = NominalState.from_observation(s)
    same = world.candidate_timing(0)
    other = world.candidate_timing(1)
    assert same.wait_seconds == pytest.approx(228.0, abs=.001)
    assert other.wait_seconds == 0
    assert same.unload_complete > other.unload_complete


def test_virtual_predecessor_release_and_waste_commitments_change_next_mask():
    s = short_roads(synthetic_state())
    put_startup_truck(s, 1, predecessor=0)
    put_startup_truck(s, 2, predecessor=1)
    s[9] = 5960.0 / 4000
    world = NominalState.from_observation(s)
    world.apply_action(3)
    world.advance()
    assert (world.now, world.current_truck) == (0.0, 1)
    assert tuple(c.action for c in world.candidates()) == (0, 1, 2)
    world.apply_action(2)
    world.advance()
    assert (world.now, world.current_truck) == (0.0, 2)


def test_shift_end_keeps_unfinished_load_out_of_production():
    s = only_root_load(short_roads(synthetic_state(window_index=3, remaining_seconds=60)), 3)
    world = NominalState.from_observation(fifo_state(s, [(100, .3)]))
    assert math.isinf(world.candidate_timing(3).unload_complete)
    world.apply_action(3)
    world.advance()
    score = world.score_inputs()
    assert world.finished and score.first_window == 3
    assert score.fifo_q == pytest.approx((0, 0, 0, 60 * 2500 / 3600))
    assert score.unload_q == pytest.approx((0, 0, 0, 100))
    assert score.waste_tons == 0


def test_zero_mass_fifo_is_removed_privately_and_unknown_grade_fails():
    s = fifo_state(synthetic_state(), [(0, .35), (40, .25)])
    original = s.tobytes()
    assert NominalState.from_observation(s).tail_context()[:2] == pytest.approx((40, 10))
    assert s.tobytes() == original
    s = put_truck(s, tid=1, phase=5, q=100)
    s[27 + 120 + 21] = 0
    with pytest.raises(NominalModelError, match='gradeKnown'):
        NominalState.from_observation(s)


def test_deadline_inside_both_forward_paths_without_mutating_original():
    world = NominalState.from_observation(synthetic_state())

    class Deadline(RuntimeError):
        pass

    def stop():
        raise Deadline

    with pytest.raises(Deadline):
        world.candidate_timing(0, check_deadline=stop)
    assert world.current_truck == 0
    branch = world.clone()
    branch.apply_action(0)
    with pytest.raises(Deadline):
        branch.advance(check_deadline=stop)
    assert branch.now == world.now == 0.0
