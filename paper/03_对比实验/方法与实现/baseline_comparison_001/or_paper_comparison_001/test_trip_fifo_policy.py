"""Regression checks for the appointment MILP and the live-action contract."""

from time import perf_counter

import numpy as np
import pytest

from grade_commitment_ppo_v1.test_features import synthetic_state
from traditional_dispatch_v1.nominal import ScoreInputs
from traditional_dispatch_v1.test_nominal import short_roads

from trip_fifo_policy import (
    TripFifoPolicy, _Appointment, _Rows, _crusher_slots, _fifo_conflicts, _projection, _resource_rows, _solve,
)


def _option(truck, action, *, berth, unload, crush_start, crush_end):
    return _Appointment(truck, action, 0, 100., .3, 0., 0., 10., 20.,
                        berth, berth, unload, crush_start, crush_end)


def test_milp_selects_joint_feasible_fifo_appointments():
    # The cheapest local choice for the second truck overtakes the first batch.
    # Waiting until the next appointment preserves FIFO and the one crusher.
    alternatives = (
        _option(0, 0, berth=20, unload=30, crush_start=50, crush_end=150),
        _option(1, 1, berth=30, unload=40, crush_start=40, crush_end=140),
        _option(1, 1, berth=30, unload=40, crush_start=150, crush_end=250),
    )
    rows = _Rows(3)
    rows.add((0,), (1.,), low=1., high=1.)
    rows.add((1, 2), (1., 1.), low=1., high=1.)
    _fifo_conflicts(rows, alternatives, lambda: None)
    _resource_rows(rows, (np.array([50., 40., 150.]), np.array([150., 140., 250.])), 1, ())
    options, _, stages, _ = _solve(alternatives, rows, np.array([0., 0., 1.]),
                                    np.array([0., 0., 1.]), perf_counter() + 3.)
    assert stages == 2
    assert options == (alternatives[0], alternatives[2])


def test_crusher_reserves_existing_fifo_batch_and_splits_tons_by_processing():
    matches = tuple(_crusher_slots(7100., 100., 1., ((7000., 7000., 7150.),), 90.))
    assert matches[0] == (7150., 7250.)
    alternative = _option(0, 0, berth=7090., unload=7100., crush_start=7150., crush_end=7250.)
    empty = ScoreInputs((0.,) * 4, (0.,) * 4, (0.,) * 4, (0.,) * 4, 0., 0., 0., 0)
    q, m, *_ = _projection((alternative,), empty, empty, empty, 1.)
    assert q[:, 0] == pytest.approx([50., 50., 0., 0.])
    assert m[:, 0] == pytest.approx([15., 15., 0., 0.])
    assert tuple(_crusher_slots(7100., 100., 1., ((7150., 7150., 7300.),), 90.)) == ()


def test_expired_trip_chooser_keeps_legal_root_and_confirms_once():
    observation = short_roads(synthetic_state())
    previous = observation.tobytes()
    policy = TripFifoPolicy(planning_ms=1e-6)
    choice = policy.choose(observation, observation[19:23])
    assert choice.kind == 'fallback' and choice.timed_out
    assert observation[19 + choice.action] == 1
    assert observation.tobytes() == previous
    assert sum(policy.summary()['committed_actions']) == 0
    with pytest.raises(ValueError, match='was not confirmed'):
        policy.choose(observation, observation[19:23])
    policy.commit(choice)
    assert sum(policy.summary()['committed_actions']) == 1
    with pytest.raises(ValueError, match='unconfirmed choice'):
        policy.commit(choice)
