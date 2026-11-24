"""Physical boundary fixtures for the validator used by the online MILPs."""

import pytest

from grade_commitment_ppo_v1.test_features import put_truck, synthetic_state
from traditional_dispatch_v1.test_nominal import only_root_load, short_roads
from traditional_dispatch_v2.nominal import NominalState


def test_parallel_berths_enter_fifo_by_unload_completion_across_window():
    snapshot = only_root_load(short_roads(synthetic_state(remaining_seconds=100)), 3)
    snapshot = put_truck(snapshot, tid=1, phase=9, q=100, grade=.20)
    snapshot = put_truck(snapshot, tid=2, phase=9, q=100, grade=.40)
    snapshot[27 + 120 + 88] = 120 / 600
    snapshot[27 + 2 * 120 + 88] = 40 / 600
    ore_dump = snapshot[1295:1343]
    ore_dump[2:4] = [1, 2 / 3]
    ore_dump[6:8] = [.4, .6]
    ore_dump[39:41] = [2 / 9, 3 / 9]

    state = NominalState.from_observation(snapshot)
    state.apply_action(3)
    while not state.finished:
        state.advance()
        if not state.finished:
            state.apply_action(3)
    result = state.score_inputs()

    before_boundary = 60 * 2500 / 3600
    assert result.fifo_q[:2] == pytest.approx((before_boundary, 200 - before_boundary), abs=1e-4)
    assert result.fifo_m[:2] == pytest.approx((.40 * before_boundary, 60 - .40 * before_boundary), abs=1e-4)
    assert result.unload_q[:2] == pytest.approx((100, 100), abs=1e-4)
    assert result.ore_tons == pytest.approx(200, abs=1e-4)
