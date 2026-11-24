import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from .decision import _PlanningDeadline, choose
from .nominal import NominalModelError


class _NominalFixture:
    """Small search tree with distinct unload, FIFO, and multi-step choices."""

    def __init__(self, actions=()):
        self.actions = actions
        self.window_index = 0

    @property
    def now(self):
        return 100.0 * len(self.actions)

    @property
    def finished(self):
        return len(self.actions) == 3

    def clone(self):
        return type(self)(self.actions)

    def candidates(self):
        return tuple(SimpleNamespace(action=i, ore=True, tons=100.0, grade=0.27) for i in (0, 1))

    def candidate_timing(self, action, *, check_deadline):
        check_deadline()
        return SimpleNamespace(wait_seconds=float(action), unload_complete=200.0 + action)

    def tail_context(self):
        return 1_000.0, 270.0, 16_000.0, 4_000.0

    def apply_action(self, action):
        self.actions += (action,)

    def advance(self, *, check_deadline):
        check_deadline()

    def score_inputs(self):
        first, second, _ = self.actions
        improved = first == second == 1
        return SimpleNamespace(first_window=0, fifo_q=(100, 100, 100, 100),
                               fifo_m=(27 if first == 0 or improved else 25, 27, 27, 27),
                               unload_q=(100, 100, 100, 100),
                               unload_m=(25 if first == 0 else 27, 27, 27, 27),
                               ore_tons=16_000, waste_tons=4_000, wait_seconds=0 if improved else 100,
                               root_cross_window=first == 1)


class DecisionTest(unittest.TestCase):
    def setUp(self):
        self.observation = np.zeros(1519, dtype=np.float32)
        self.mask = np.array([1, 1, 0, 0], dtype=np.int8)
        self.observation[19:23] = self.mask

    @patch("traditional_dispatch_v1.decision.NominalState.from_observation")
    def test_paired_policies_and_rolling_incumbent(self, factory):
        factory.side_effect = lambda obs: _NominalFixture()
        by_unload = choose("unload_greedy", self.observation, self.mask)
        by_fifo = choose("fifo_greedy", self.observation, self.mask)
        rolling = choose("rolling", self.observation, self.mask)
        self.assertEqual((by_unload.action, by_fifo.action, rolling.action), (1, 0, 1))
        self.assertEqual((by_unload.completed_candidates, by_fifo.completed_candidates), (2, 2))
        self.assertEqual((by_unload.cross_window_candidates, by_fifo.cross_window_candidates), (1, 1))
        self.assertEqual(rolling.initial_cost, by_fifo.selected_cost)
        self.assertLess(rolling.selected_cost, rolling.initial_cost)
        self.assertGreater(rolling.multi_step_plans, 0)
        self.assertEqual(rolling.max_scored_depth, 3)
        self.assertTrue(rolling.tree_complete)

    @patch("traditional_dispatch_v1.decision.NominalState.from_observation")
    def test_unavailable_model_uses_legal_fallback(self, factory):
        factory.side_effect = NominalModelError("model fixture unavailable")
        result = choose("rolling", self.observation, self.mask)
        self.assertTrue(self.mask[result.action])
        self.assertEqual(result.kind, "fallback")
        self.assertEqual(result.fallback_reason, "model fixture unavailable")

    @patch("traditional_dispatch_v1.decision.NominalState.from_observation")
    def test_single_legal_is_a_fast_path(self, factory):
        self.mask[:] = [0, 0, 0, 1]
        self.observation[19:23] = self.mask
        result = choose("fifo_greedy", self.observation, self.mask)
        self.assertEqual((result.kind, result.action), ("single_legal", 3))
        factory.assert_not_called()

    @patch("traditional_dispatch_v1.decision.perf_counter", side_effect=[0.0, 1.0])
    def test_deadline_with_no_complete_candidate_uses_fallback(self, _clock):
        result = choose("rolling", self.observation, self.mask, planning_ms=800)
        self.assertEqual(result.kind, "fallback")
        self.assertTrue(result.timed_out)
        self.assertEqual(result.fallback_reason, "incomplete_one_step_at_deadline")

    @patch("traditional_dispatch_v1.decision._search", side_effect=_PlanningDeadline)
    @patch("traditional_dispatch_v1.decision.NominalState.from_observation")
    def test_deadline_during_search_returns_best_complete_incumbent(self, factory, _search):
        factory.side_effect = lambda obs: _NominalFixture()
        result = choose("rolling", self.observation, self.mask)
        self.assertEqual((result.action, result.kind), (0, "planned"))
        self.assertTrue(result.timed_out)
        self.assertEqual(result.selected_cost, result.initial_cost)
        self.assertIsNone(result.fallback_reason)

    def test_mismatched_action_mask_rejected(self):
        with self.assertRaisesRegex(ValueError, "matching"):
            choose("fifo_greedy", self.observation, np.array([0, 1, 0, 0]))
