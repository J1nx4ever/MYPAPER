"""Deterministic, offline cross-checks of the actual baseline selection routines."""
from collections import defaultdict
import hashlib
from itertools import product
import json
import math
from pathlib import Path
from random import Random
import sys
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from traditional_dispatch_v2 import decision, dispatch_policy as dispatch


class TwoChoiceTree:
    """Controlled terminal scores and deterministic candidate waiting times."""
    window_index = 0
    actions = (1, 3)

    def __init__(self, grades, waits, local, path=()):
        self.grades, self.waits, self.local, self.path = grades, waits, local, path

    @property
    def now(self):
        return 100. * len(self.path)

    @property
    def finished(self):
        return len(self.path) == 3

    def clone(self):
        return type(self)(self.grades, self.waits, self.local, self.path)

    def candidates(self):
        return tuple(SimpleNamespace(action=a, ore=True, tons=100., grade=.27) for a in self.actions)

    def tail_context(self):
        return 1000., 270., 16000., 4000.

    def candidate_timing(self, action, **_):
        return SimpleNamespace(wait_seconds=self.local[self.path+(action,)], unload_complete=1000.)

    def apply_action(self, action):
        self.path += (action,)

    def advance(self, **_):
        pass

    def score_inputs(self):
        unload, fifo = self.grades[self.path]
        return SimpleNamespace(unload_q=[100.]*4, fifo_q=[100.]*4, unload_m=[100*unload, 27., 27., 27.],
                               fifo_m=[100*fifo, 27., 27., 27.], first_window=0, ore_tons=16000.,
                               waste_tons=4000., wait_seconds=self.waits[self.path], root_cross_window=False)


def check_trees():
    actions = TwoChoiceTree.actions
    obs = np.zeros(1519, dtype=np.float32)
    obs[20] = obs[22] = 1
    leaves = list(product(actions, repeat=3))
    for seed in range(300):
        rng = Random(seed)
        grades = {p: (rng.uniform(.23, .31), rng.uniform(.23, .31)) for p in leaves}
        waits = {p: rng.randint(0, 15000) for p in leaves}
        local = {p+(a,): rng.randint(0, 25) for d in (0, 1, 2) for p in product(actions, repeat=d) for a in actions}
        tree = TwoChoiceTree(grades, waits, local)

        def cost(path, mode):
            deviation = max(0., abs(grades[path][mode]-.27)-.004)
            return min(2., deviation/.006) + .2*waits[path]/28800

        def tail(action):
            path = (action,)
            while len(path) < 3:
                path += (min(actions, key=lambda a: (local[path+(a,)], a)),)
            return path

        with patch.object(decision.NominalState, 'from_observation', return_value=tree):
            for i, policy in enumerate(('unload_greedy', 'fifo_greedy')):
                selected = decision.choose(policy, obs, obs[19:23])
                expected = min(actions, key=lambda a: (cost(tail(a), i), a))
                assert selected.action == expected and abs(selected.selected_cost - cost(tail(expected), i)) < 1e-11
            rolling = decision.choose('rolling', obs, obs[19:23])
            optimal = min(cost(p, 1) for p in leaves)
            assert rolling.tree_complete and not rolling.timed_out and abs(rolling.selected_cost-optimal) < 1e-11
            assert any(p[0] == rolling.action and abs(cost(p, 1)-optimal) < 1e-11 for p in leaves)
    return dict(trees=300, leaves=2400, action_and_cost_comparisons=900)


def independent_loss(options, trucks, targets, servers, bookings):
    assigned = defaultdict(float)
    extra = [[] for _ in servers]
    cost = 0.
    for o, truck in zip(options, trucks, strict=True):
        before = 0.
        due_penalty = o.tons
        for interval in targets[o.site]:
            if assigned[o.site] < before + interval.tons - 1e-6:
                fraction = (assigned[o.site]-before)/interval.tons
                due = interval.start + (interval.stop-interval.start)*fraction
                deviation = abs((o.arrival-o.empty_seconds)-due)
                due_penalty = min(o.tons, deviation*interval.tons/(interval.stop-interval.start))
                break
            before += interval.tons
        assigned[o.site] += o.tons
        cost += due_penalty + o.tons/o.cycle_seconds*(o.empty_seconds-truck.shortest_empty)
        extra[o.site].append((o.arrival, 2, truck.truck_id, o.loading_seconds, o))
    for site, occupied in enumerate(servers):
        free = list(occupied)
        for arrival, _, _, duration, o in sorted(bookings[site] + extra[site], key=lambda x: x[:3]):
            index = min(range(len(free)), key=free.__getitem__)
            start = max(free[index], arrival)
            free[index] = start + duration
            if o is not None:
                cost += o.tons/o.cycle_seconds*(start-arrival)
    return cost


def check_dispatch():
    root = ROOT/'runs/bg128-continuous-mixed-20261120-s02/evaluation/nominal/new/material'
    frames = allocations = 0
    for run in (10004025, 10004026, 10004027):
        path = root/f'run_{run}.h5'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == json.loads(path.with_suffix('.h5.seal.json').read_text())['hdf5_sha256']
        with h5py.File(path) as source:
            observations = source['states'][:-1:4]
        for obs in observations:
            policy = dispatch.DispatchPolicy()
            decision = policy.choose(obs, obs[19:23])
            assert decision.kind == 'planned' and decision.lp_status == 'optimal' and not decision.timed_out
            state = dispatch.NominalState.from_observation(obs)
            trucks = dispatch._ready_trucks(state, 3, lambda: None)
            targets = dispatch._targets(policy.plan)
            for j, site in enumerate(targets):
                mass = sum(row[j] for p in policy.plan.periods for row in p.tons_by_type_site)
                assert math.isclose(sum(t.tons for t in site), mass, abs_tol=1e-5)
            servers, bookings = dispatch._shovel_calendars(state)
            for n in range(len(trucks), 0, -1):
                costs = []
                for options in product(*(tr.choices for tr in trucks[:n])):
                    obs_trips = dispatch._reserve(options, state._available_tons, state._loads, state._waste_committed())
                    if obs_trips is None:
                        continue
                    reference = independent_loss(obs_trips, trucks[:n], targets, servers, bookings)
                    score = dispatch._loss(obs_trips, trucks[:n], targets, [0.]*4, servers, bookings)
                    assert math.isclose(reference, score, rel_tol=0, abs_tol=1e-8)
                    costs.append((reference, options[0].site))
                    allocations += 1
                if costs:
                    optimum = min(costs)[0]
                    assert n == decision.matched_trucks and any(
                        action == decision.action and abs(score-optimum) < 1e-8 for score, action in costs)
                    break
            else:
                raise AssertionError('no feasible legal root in archive')
            frames += 1
    return dict(frames=frames, rescored_allocations=allocations)


if __name__ == '__main__':
    if not __debug__:
        raise RuntimeError('run the checks with Python assertions enabled')
    print(json.dumps(dict(greedy_and_rolling=check_trees(), dispatch_inspired=check_dispatch()), indent=2))
