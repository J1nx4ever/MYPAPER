"""Check quota publication, acknowledged tracking and terminal plan accounting."""
from time import perf_counter

import numpy as np
import pytest

from grade_commitment_ppo_v1.test_features import synthetic_state
from traditional_dispatch_v1.test_nominal import only_root_load
from traditional_dispatch_v2.nominal import NominalState

from change_plans.token_attention_001.business_graph_summary_009.baseline_comparison_001.or_paper_comparison_001.rolling_quota_policy import (
    POLICY4_INITIAL_QUOTAS, QuotaPlan, RollingQuotaPolicy, _dispatch_periods,
    flow, plan_window, solve_quota,
)


def _choose_at(policy, seconds):
    frame = synthetic_state(remaining_seconds=7200 - seconds)
    return policy.choose(frame, frame[19:23])


def test_rolling_plan_is_added_to_confirmed_actions_and_failure_keeps_plan():
    counts = np.zeros((4, 3, 4), dtype=int)
    counts[0, 0, 3] = 1
    calls = []

    def planner(_observation, state, *, deadline):
        calls.append(state.now)
        if len(calls) == 2:
            raise flow.FlowPlanError('quota_milp_limit_without_incumbent')
        return QuotaPlan(counts, 'optimal', 0., 2, 0., 0., 0., 0., False, 0.)

    policy = RollingQuotaPolicy(solve_plan=planner)
    first = _choose_at(policy, 0)
    assert first.action == 0 and first.stage == 1 and not first.replanned
    assert not calls and not policy.assigned.any()
    assert np.array_equal(policy.quota, POLICY4_INITIAL_QUOTAS)
    with pytest.raises(RuntimeError, match='unacknowledged'):
        _choose_at(policy, 0)
    policy.commit(first)
    with pytest.raises(RuntimeError, match='acknowledgement'):
        policy.commit(first)

    second = _choose_at(policy, 901)
    assert second.action == 3 and second.replanned
    assert policy.quota[0, 0].tolist() == [1, 0, 0, 1]
    assert policy.assigned[0, 0].tolist() == [1, 0, 0, 0]
    policy.commit(second)

    third = _choose_at(policy, 1802)
    assert not third.replanned and third.lp_status == 'failed' and third.timed_out
    assert policy.quota[0, 0].tolist() == [1, 0, 0, 1]
    policy.commit(third)
    assert policy.summary()['mip_accepted'] == 1
    assert policy.summary()['committed_actions'] == 3


def test_dispatch_periods_break_at_policy4_lead_boundary():
    periods = _dispatch_periods(5820.)
    assert periods[0] == (5820., 6300.)
    assert plan_window(6299., 0) == 0
    assert plan_window(6300., 0) == 1
    assert periods[-1][1] == 28800.
    assert all(plan_window(start, int(start // 7200)) == plan_window(stop - 1.e-6, int((stop - 1.e-6) // 7200))
               for start, stop in periods)


def test_short_terminal_window_cannot_credit_unfinished_waste_trip():
    frame = only_root_load(synthetic_state(window_index=3, remaining_seconds=30.), 3)
    state = NominalState.from_observation(frame)
    result = solve_quota(frame, state, deadline=perf_counter() + 1.)
    assert result.counts.sum() == 0
    assert result.waste_shortfall_tons == pytest.approx(4000.)
    assert result.max_row_residual < 1.e-6
