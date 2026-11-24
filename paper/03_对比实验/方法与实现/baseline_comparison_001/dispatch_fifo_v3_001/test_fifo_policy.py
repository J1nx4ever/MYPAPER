"""Audit the FIFO gate, physical prefix checks, and bounded search."""

from pathlib import Path
from types import SimpleNamespace

import h5py
import pytest

from change_plans.token_attention_001.business_graph_summary_009.baseline_comparison_001.dispatch_fifo_v3_001 import (
    fifo_policy as v3,
)
from traditional_dispatch_v2.nominal import NominalState
from traditional_dispatch_v2.test_flow_plan import fast_snapshot


def _projection(action, cost, feasible, *, depth=1):
    options = tuple(SimpleNamespace(site=action) for _ in range(depth))
    trucks = tuple(SimpleNamespace(truck_id=idx) for idx in range(depth))
    return v3._Projection(cost, feasible, trucks, options)


def test_feasibility_gate_then_industrial_loss_else_frozen_fifo_cost(monkeypatch):
    costs = {0: 4., 1: 2., 2: 0.}
    observed_depths = []

    def match_loss(options, trucks, *_):
        assert len(options) == len(trucks)
        observed_depths.append(len(options))
        return costs[options[0].site]

    monkeypatch.setattr(v3, 'loss', match_loss)
    a = _projection(0, .4, True, depth=3)
    b = _projection(1, .5, True, depth=2)
    c = _projection(2, .1, False, depth=3)
    assert v3._best([a, b, c], (), lambda: None) == (b, 2.)
    assert observed_depths == [2, 2, 2]
    a = _projection(0, .2, False)
    b = _projection(1, .3, False)
    assert v3._best([a, b], (), lambda: None) == (a, 4.)
    assert v3._best([a, b], None, lambda: None) == (a, None)


def test_gate_only_considers_windows_from_observed_current_window():
    inputs = SimpleNamespace(first_window=2, fifo_q=(0, 0, 5000, 5000), fifo_m=(0, 0, 1300, 1400),
                             ore_tons=16000., waste_tons=4000.)
    assert v3._complete(inputs)
    inputs.fifo_m = (0, 0, 1299, 1400)
    assert not v3._complete(inputs)
    inputs.fifo_q, inputs.fifo_m = (0, 0, 5000, 0), (0, 0, 1350, 0)
    assert not v3._complete(inputs)


def test_prefix_must_match_physical_next_id_action_and_payload():
    obs = fast_snapshot(window_index=3)
    state = NominalState.from_observation(obs)
    root = state.current_truck
    tons = state.candidates()[0].tons
    with pytest.raises(v3._UnexecutablePrefix, match='next physically actionable truck'):
        v3._project(state, ((root + 1, 0, tons),), {}, 3, lambda: None)
    with pytest.raises(v3._UnexecutablePrefix, match='physical action mask'):
        v3._project(state, ((root, 4, tons),), {}, 3, lambda: None)
    with pytest.raises(v3._UnexecutablePrefix, match='payload differs'):
        v3._project(state, ((root, 0, tons - 1),), {}, 3, lambda: None)
    with pytest.raises(v3._UnexecutablePrefix, match='next physically actionable truck'):
        v3._project(state, ((root, 0, tons), (root, 0, None)), {}, 3, lambda: None)


def test_wall_deadline_restores_pretrial_winner(monkeypatch):
    obs = fast_snapshot(window_index=3)
    roots_called = []
    trials_called = []

    def project(_state, prefix, _cache, _match_trucks, _check):
        action = prefix[0][1]
        if len(prefix) == 1:
            roots_called.append(action)
            return _projection(action, 4 - action, False)
        trials_called.append(action)
        if len(trials_called) == 2:
            raise v3.industrial._Deadline
        return _projection(action, 0, True)

    def nominate(state, actions, *_):
        return [((state.current_truck, action, None), (8, action, None)) for action in actions], 20

    monkeypatch.setattr(v3, '_project', project)
    monkeypatch.setattr(v3, '_nominate', nominate)
    monkeypatch.setattr(v3.FIFODispatchPolicy, '_guide', lambda *args: ())
    monkeypatch.setattr(v3, 'loss', lambda *args: 0.)
    policy = v3.FIFODispatchPolicy()
    choice = policy.choose(obs, obs[19:23])
    assert roots_called == [0, 1, 2, 3] and trials_called == [0, 1]
    assert choice.action == 3 and choice.kind == 'planned'
    assert choice.fifo_roots_completed == choice.fifo_candidates == 4
    assert choice.fifo_trials_completed == 1
    assert choice.fifo_trials_compared == 0 and choice.matched_trucks == 1
    assert not choice.fifo_selected_from_cohort
    assert choice.fifo_guard_restored_root and choice.timed_out
    assert choice.fifo_selected_cost == choice.fifo_root_cost == 1
    policy.commit(choice)


@pytest.mark.parametrize('failure, expected_rejected, expected_error', [
    (v3._UnexecutablePrefix('truck changed'), 4, None),
    (v3.NominalModelError('nominal_event_clock_stalled'), 0, 'nominal_event_clock_stalled'),
])
def test_false_multitruck_witness_and_internal_model_failure(monkeypatch, failure, expected_rejected, expected_error):
    obs = fast_snapshot(window_index=3)

    def project(_state, prefix, *_):
        if len(prefix) > 1:
            raise failure
        return _projection(prefix[0][1], 1., True)

    def nominate(state, actions, *_):
        return [((state.current_truck, action, None), (8, action, None)) for action in actions], 20

    monkeypatch.setattr(v3, '_project', project)
    monkeypatch.setattr(v3, '_nominate', nominate)
    monkeypatch.setattr(v3.FIFODispatchPolicy, '_guide', lambda *args: ())
    monkeypatch.setattr(v3, 'loss', lambda *args: 0.)
    policy = v3.FIFODispatchPolicy()
    choice = policy.choose(obs, obs[19:23])
    assert choice.kind == 'planned' and choice.fifo_roots_completed == choice.fifo_candidates == 4
    assert choice.fifo_trials_nominated == 4 and choice.fifo_trials_rejected == expected_rejected
    assert choice.fifo_trials_completed == choice.fifo_trials_compared == 0
    assert choice.matched_trucks == 1 and not choice.fifo_selected_from_cohort
    assert choice.fifo_model_error == expected_error and choice.fifo_guidance_error is None
    policy.commit(choice)


def test_guidance_failure_uses_complete_root_result_and_keeps_reason():
    def fail(*args, **kwargs):
        raise v3.FlowPlanError('LP intentionally unavailable')

    obs = fast_snapshot(window_index=3, remaining_seconds=900)
    p = v3.FIFODispatchPolicy(solve_flow=fail)
    choice = p.choose(obs, obs[19:23])
    assert choice.kind == 'planned' and obs[19 + choice.action] == 1
    assert choice.fifo_roots_completed == choice.fifo_candidates == 4
    assert choice.fifo_selected_cost == choice.fifo_root_cost and choice.fifo_trials_completed == 0
    assert choice.lp_status == 'failed' and choice.fifo_guidance_error == 'LP intentionally unavailable'


def test_one_legal_action_is_forced_and_commit_uses_returned_identity():
    obs = fast_snapshot(window_index=3, remaining_seconds=900)
    obs[20:23] = 0
    obs[24:27] = 0
    p = v3.FIFODispatchPolicy()
    choice = p.choose(obs, obs[19:23])
    assert isinstance(choice, v3.FIFOChoice)
    assert choice.kind == 'single_legal' and choice.action == 0 and choice.fifo_candidates == 1
    p.commit(choice)


def test_sealed_first_frame_covers_every_root_and_finds_feasible_third_ranked_trial():
    root = Path(__file__).resolve().parents[5]
    path = (root / 'runs/bg128-u32-generalization-20261119-001/evaluation'
            / 'nominal/episodes/run_9991001/samples.h5')
    if not path.exists():
        pytest.skip('sealed nominal observation missing')
    with h5py.File(path) as archive:
        obs = archive['states'][0]
    before = obs.tobytes()
    p = v3.FIFODispatchPolicy()
    choice = p.choose(obs, obs[19:23])
    assert obs.tobytes() == before and obs[19 + choice.action] == 1
    assert choice.kind == 'planned' and not choice.timed_out and choice.replanned
    assert choice.fifo_roots_completed == choice.fifo_candidates == 4
    assert choice.matched_trucks == 3 and choice.assignments_scored == 68
    assert choice.fifo_trials_completed == 12 and choice.fifo_trials_rejected == 0
    assert choice.fifo_trials_compared == choice.fifo_trials_nominated == 12
    assert choice.fifo_selected_from_cohort and choice.fifo_model_error is None
    assert choice.fifo_selected_industrial_loss is not None and choice.fifo_selected_actions[0] == choice.action
    assert choice.fifo_selected_truck_ids == (3, 0, 1)
    assert not choice.fifo_root_feasible and choice.fifo_selected_feasible
    assert choice.fifo_selected_actions == (2, 0, 2)
    p.commit(choice)
