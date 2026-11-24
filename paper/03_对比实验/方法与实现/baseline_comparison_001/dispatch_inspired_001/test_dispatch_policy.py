from itertools import product

import numpy as np
import pytest

from dispatch_policy import (
    DispatchPolicy, _Option, _Target, _TruckOptions, _fits_commitments, _loss, _need_time,
)


def test_inverse_target_tonnage_handles_multiple_periods():
    targets = (_Target(0, 10, 0), _Target(10, 20, 100), _Target(20, 30, 200))
    assert _need_time(targets, 0) == (10, 10)
    assert _need_time(targets, 140) == (22, 20)
    assert _need_time(targets, 300)[1] == 0


def _option(site, travel):
    return _Option(site, 100., travel, travel, travel + 35, 40.)


def test_anticipating_a_second_truck_can_change_the_current_action():
    current = _TruckOptions(1, (_option(0, 10), _option(1, 12)), 10)
    later = _TruckOptions(2, (_option(0, 2), _option(1, 100)), 2)
    targets = [(_Target(0, 100, 1000),)] * 4
    state = (targets, [0.0]*4, [[0.0] for _ in range(4)], [[] for _ in range(4)])
    one = min(current.choices, key=lambda option: _loss((option,), (current,), *state))
    two = min(product(current.choices, later.choices),
              key=lambda choices: _loss(choices, (current, later), *state))
    assert one.site == 0
    assert tuple(option.site for option in two) == (1, 0)


def test_futures_respect_supply_and_waste_commitments():
    waste = _option(3, 10)
    ore = _option(0, 10)
    ore_sites = True, True, True, False
    assert _fits_commitments((waste, ore), [150]*4, ore_sites, 5950)
    assert not _fits_commitments((waste, waste), [500]*4, ore_sites, 5950)
    assert not _fits_commitments((ore, ore), [150]*4, ore_sites, 5950)


def test_only_acknowledged_dispatch_advances_the_ledger():
    obs = np.zeros(1519, dtype=np.float32)
    obs[19:23] = [0, 1, 0, 0]
    obs[24] = .5
    policy = DispatchPolicy()
    policy.plan = object()
    choice = policy.choose(obs, obs[19:23])
    assert choice.kind == 'single_legal'
    assert policy._assigned == [0.] * 4
    with pytest.raises(ValueError):
        policy.choose(obs, obs[19:23])
    policy.commit(choice)
    assert policy._assigned[1] == 96.5
    with pytest.raises(ValueError):
        policy.commit(choice)
