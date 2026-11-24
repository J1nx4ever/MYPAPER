"""Focused regressions for shared rollout and actual-action selection."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from traditional_dispatch_v1.test_decision import _NominalFixture
from traditional_dispatch_v1.nominal import _Load as FrozenLoad
from .decision import _PlanningDeadline, choose
from .dispatch_policy import _loss, _need_time, _Option, _reserve, _Target, _targets, _TruckOptions
from .nominal import _ObservedLoad


def _two_legal():
    obs = np.zeros(1519, dtype=np.float32)
    obs[19:21] = 1
    return obs, obs[19:23]


@patch('traditional_dispatch_v2.decision.NominalState.from_observation', side_effect=lambda _: _NominalFixture())
def test_objective_twins_and_rolling_budget(_factory):
    obs, mask = _two_legal()
    unload = choose('unload_greedy', obs, mask)
    fifo = choose('fifo_greedy', obs, mask)
    rolling = choose('rolling', obs, mask)
    bounded = choose('rolling', obs, mask, rollout_limit=2)
    assert (unload.action, fifo.action, rolling.action) == (1, 0, 1)
    assert rolling.initial_cost == fifo.selected_cost
    assert rolling.selected_cost < fifo.selected_cost
    assert (unload.completed_candidates, fifo.completed_candidates, bounded.completed_candidates) == (2, 2, 2)
    assert bounded.search_nodes == 2 and bounded.budget_exhausted and not bounded.timed_out
    assert bounded == choose('rolling', obs, mask, rollout_limit=2)


@patch('traditional_dispatch_v2.decision.NominalState.from_observation', side_effect=lambda _: _NominalFixture())
def test_wall_cutoff_discards_machine_dependent_continuation(_factory):
    obs, mask = _two_legal()

    def partially_scored(result, *_):
        result.action, result.selected_cost = 1, 0.0
        raise _PlanningDeadline

    with patch('traditional_dispatch_v2.decision._search', side_effect=partially_scored):
        rolling = choose('rolling', obs, mask)
    fifo = choose('fifo_greedy', obs, mask)
    assert rolling.timed_out and not rolling.budget_exhausted
    assert (rolling.action, rolling.selected_cost) == (fifo.action, fifo.selected_cost)


def test_bucket_rounding_does_not_add_a_loading_pass():
    bucket = float(np.float32(10 / 30)) * 30
    load = _ObservedLoad(2, True, True, .27, 1, bucket, .45)
    frozen = FrozenLoad(2, True, True, .27, 1, bucket, .45)
    for q in (30, 60, 90, 120, 150, 180):
        decoded = float(np.float32(q / 193)) * 193
        assert load.service(decoded) == pytest.approx(42 + 27 * (q // 10))
    decoded = float(np.float32(60 / 193)) * 193
    assert frozen.service(decoded) > load.service(decoded)
    assert load.service(60.0001) == pytest.approx(42 + 27 * 7)


def test_dispatch_due_uses_release_time_not_shovel_start():
    option = _Option(0, 100., 600., 600., 1000., 100.)
    truck = _TruckOptions(0, (option,), 600.)
    targets = ((_Target(0., 1000., 1000.),), (), (), ())
    assigned = [0.] * 4
    bookings = [[] for _ in range(4)]
    assert _loss((option,), (truck,), targets, assigned, [[0.], [0.], [0.], [0.]], bookings) == 0.0
    assert _loss((option,), (truck,), targets, assigned, [[700.], [0.], [0.], [0.]], bookings) == 10.0
    late = replace(option, arrival=625.)
    assert _loss((late,), (truck,), targets, assigned, [[0.], [0.], [0.], [0.]], bookings) == 25.0


def test_targets_integrate_later_truck_class_release():
    period = SimpleNamespace(end=100.,
                             tons_by_type_site=((100., 0., 0., 0.), (250., 0., 0., 0.)),
                             dispatch_start_by_type_site=((0., None, None, None), (50., None, None, None)))
    first, *other_sites = _targets(SimpleNamespace(periods=(period,)))
    assert first == (_Target(0., 50., 50.), _Target(50., 100., 300.))
    assert _need_time(first, 110.) == (60., 6.)
    assert other_sites == [(), (), ()]


def test_virtual_reservations_allow_partial_payload_and_enforce_waste_stop():
    ore = _ObservedLoad(2, True, True, .27, 1, 10., .45)
    waste = replace(ore, ore=False)
    options = (_Option(0, 90., 10., 10., 400., ore.service(90.)),) * 2
    result = _reserve(options, [120., 500., 0., 0.], (ore, waste), 5970.)
    assert [route.tons for route in result] == [90., 30.]
    assert result[1].loading_seconds == ore.service(30.)
    waste_options = (replace(options[0], site=1),) * 2
    assert _reserve(waste_options[:1], [120., 500., 0., 0.], (ore, waste), 5970.)
    assert _reserve(waste_options, [120., 500., 0., 0.], (ore, waste), 5970.) is None
    assert _reserve(options, [90., 500., 0., 0.], (ore, waste), 5970.) is None
