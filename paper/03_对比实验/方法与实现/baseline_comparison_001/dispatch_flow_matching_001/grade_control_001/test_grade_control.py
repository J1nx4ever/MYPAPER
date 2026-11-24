from collections import Counter
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import h5py
import pytest

from change_plans.token_attention_001.business_graph_summary_009.baseline_comparison_001.dispatch_flow_matching_001.grade_control_001 import grade_guard
from change_plans.token_attention_001.business_graph_summary_009.baseline_comparison_001.dispatch_flow_matching_001.grade_control_001.grade_policy import GradeControlPolicy
from change_plans.token_attention_001.business_graph_summary_009.baseline_comparison_001.dispatch_inspired_001.dispatch_runtime import _count_choice


def test_opening_plan_reserves_upper_grade_margin_and_acknowledges_once():
    b = Path(__file__).resolve().parents[6]
    archive = b / 'runs/bg128-continuous-mixed-20261120-s02/evaluation/nominal/new/material/run_10004025.h5'
    with h5py.File(archive) as stored:
        observation = stored['states'][0]
    mask = observation[19:23]
    policy = GradeControlPolicy(planning_ms=3000)
    choice = policy.choose(observation, mask)
    assert mask[choice.action] and choice.kind == 'planned'
    assert policy.plan.planning_stages == 3
    assert all(.266 - 1e-7 <= window.grade <= .274 + 1e-7 for window in policy.plan.grade_windows)
    assert policy.plan.ore_shortfall_tons < .002
    assert policy.plan.waste_shortfall_tons < .002
    _count_choice(choice, mask, 3, Counter(), Counter(), [])
    policy.commit(choice)
    policy.commit(choice)
    summary = policy.summary()
    assert sum(summary['branch_counts'].values()) == 1
    assert sum(row['confirmed_tons'] for row in summary['route_periods']) == pytest.approx(choice.assigned_tons)


@pytest.fixture
def choices():
    return SimpleNamespace(candidates=lambda: (
        SimpleNamespace(action=0, ore=True), SimpleNamespace(action=1, ore=True),
        SimpleNamespace(action=2, ore=True), SimpleNamespace(action=3, ore=False),
    ))


def test_whole_load_guard_corrects_high_grade_and_preserves_ore(choices, monkeypatch):
    assessments = {
        0: grade_guard.Assessment(0, 16000., ((4000., 1140.), (4000., 1080.))),
        1: grade_guard.Assessment(1, 16000., ((4000., 1090.), (4000., 1080.))),
        2: grade_guard.Assessment(2, 15980., ((3980., 1074.6), (4000., 1080.))),
    }
    monkeypatch.setattr(grade_guard, '_run_branch', lambda state, action, check: assessments[action])
    result = grade_guard.guard_action(choices, 0, deadline=perf_counter() + 1.)
    assert (result.action, result.reason) == (1, 'outer_grade_risk')
    assert grade_guard.guard_action(choices, 3, deadline=perf_counter() + 1.).action == 3


def test_guard_preserves_flow_when_alternative_moves_violation_to_next_window(choices, monkeypatch):
    assessments = {
        0: grade_guard.Assessment(0, 16000., ((4000., 1140.), (4000., 1080.))),
        1: grade_guard.Assessment(1, 16000., ((4000., 1080.), (4000., 1124.))),
        2: grade_guard.Assessment(2, 16000., ((4000., 1139.9), (4000., 1080.))),
    }
    monkeypatch.setattr(grade_guard, '_run_branch', lambda state, action, check: assessments[action])
    result = grade_guard.guard_action(choices, 0, deadline=perf_counter() + 1.)
    assert (result.action, result.reason) == (0, 'matching_retained')


def test_guard_retains_original_when_branch_comparison_hits_deadline(choices, monkeypatch):
    def expired(state, action, check):
        check()
        raise AssertionError('late candidate must not be scored')

    monkeypatch.setattr(grade_guard, '_run_branch', expired)
    result = grade_guard.guard_action(choices, 0, deadline=perf_counter() - 1.)
    assert (result.action, result.reason) == (0, 'projection_deadline')
