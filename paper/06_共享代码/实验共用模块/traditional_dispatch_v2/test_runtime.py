"""Exercise the common evaluator through the real in-process RPC bridge."""

import numpy as np
import pytest
import torch

from online_v1_1.synthetic import info, window
from python.collect_offline import NORMAL_FIXTURE
from python.dispatch_bridge import DispatchCallbackBridge, ProtocolError
from python.dispatch_env import DispatchEnv
from python.runtime_scenario import baseline_fields
from single_critic_task_query_v1.test_scenario_bridge import InProcessDispatch, dispatch_request
from task_query_cmdp_v1.test_runtime import snapshot

from . import decision, runtime
from .dispatch_policy import Choice


def _fixture_job(policy, *, wrong_echo=None, java_fallback=False):
    scene = baseline_fields()
    schedule = (0., 20., 28800.)
    requests = []
    for sequence, time in enumerate(schedule):
        terminal = time == 28800.
        delivery = 3500. if terminal else 0.
        raw = info(time, sequence, episode='paired-rpc-test', seed=701, terminal=terminal,
                   elapsed=time - schedule[max(0, sequence - 1)],
                   previous_action=-1 if sequence == 0 else int(sequence == wrong_echo),
                   windows=[window(index, .27) for index in range(4)] if terminal else [])
        raw['experiment_run_index'] = 801
        components = raw['reward_components']
        components.update(scene | NORMAL_FIXTURE | {
            'config.waste_task_limit_version': 2., 'config.waste_task_dispatch_stop_tons': 6000.,
            'config.waste_task_upper_tons': 6193., 'config.material_event_memory_enabled': 0.,
            'config.material_event_contract_version': 1., 'config.wait_breakdown_enabled': 1.,
            'config.wait_breakdown_version': 1., 'audit.waste_task.committed_tons': delivery,
            'audit.waste_task.delivered_tons': delivery, 'audit.waste_task.assignment_closed': 0.,
        })
        if terminal:
            components.update({f'diag.wait.{key}': float(key == 'breakdown_version')
                               for key in runtime.shared.WAIT_FIELDS})
        if java_fallback and sequence == 1:
            components['config.eval.fallback_used'] = 1.
        request = dispatch_request(raw, terminal=terminal)
        state = snapshot(delivery, terminal=terminal)
        state[:4] = [time / 28800., 1 - time / 28800., 1. if terminal else 0.,
                     0. if terminal else 1 - time / 7200.]
        if terminal:
            state[7] = 14000. / 20000.
            state[1107 + 8] = 14000. / 30000.
            state[1107 + 5] = state[1107 + 7] = 1. - 14000. / 30000.
        state[runtime.shared.MATERIAL_INDICES] = [scene[f'{runtime.PREFIX}load.{site}.{field}']
                    for site in range(4) for field in runtime.shared.MATERIAL_RPC_FIELDS]
        request.obs.values[:] = state
        requests.append(request)
    return dict(policy_kind=policy, run_index=801, experiment_seed=701,
                actor_sampling_seed=1971 if policy == 'bg128' else 0,
                runtime_scene_sha256=runtime.fingerprint(scene),
                expected_rpc_config={key: value for key, value in requests[0].reward_components.items()
                                     if key.startswith('config.')}), requests


def _collect(job, requests, *, actor=None, anchor=None):
    bridge = DispatchCallbackBridge(expected_episode_metadata=[(801, 701)])
    env = DispatchEnv(bridge, callback_timeout=2.)
    producer = InProcessDispatch(bridge, requests)
    producer.start()
    try:
        result = runtime._collect(env, job, {'rollout_limit': 20}, anchor=anchor, actor=actor)
    finally:
        env.close()
        producer.join()
    assert not producer.errors
    assert [response.has_action for response in producer.responses] == [True, True, False]
    return result


class RecordingDispatch:
    def __init__(self):
        self.calls = []

    def choose(self, _observation, _mask):
        first = not self.calls
        self.calls.append('choose')
        return Choice(0, 'planned', replanned=first, lp_status='optimal' if first else None,
                      matched_trucks=3, assignments_scored=4, flow_shortfall_ore_tons=1.,
                      flow_shortfall_waste_tons=2., flow_grade_slack_tons=3.)

    def commit(self, _choice):
        self.calls.append('commit')


@pytest.mark.parametrize('policy', runtime.POLICIES)
def test_common_collection_settles_four_true_windows_and_counts_applied_actions(monkeypatch, policy):
    dispatch = RecordingDispatch()
    monkeypatch.setattr(runtime, '_make_dispatch', lambda _method: dispatch)

    def choose(kind, _observation, mask, *, planning_ms, depth, rollout_limit):
        assert kind == policy
        assert (planning_ms, depth, rollout_limit) == (800, 3, 20)
        rolling = kind == 'rolling'
        return decision.Decision(0, 'planned', completed_candidates=sum(mask), budget_exhausted=rolling,
                                 max_scored_depth=3 if rolling else 1, multi_step_plans=int(rolling))

    def sample(_actor, generator, _observation):
        torch.rand((), generator=generator)
        return 0

    monkeypatch.setattr(decision, 'choose', choose)
    monkeypatch.setattr(runtime.shared, '_actor_choice', sample)
    job, requests = _fixture_job(policy)
    result = _collect(job, requests, actor=object() if policy == 'bg128' else None)
    assert [row['grade'] for row in result['physical']['windows']] == pytest.approx([.27] * 4)
    assert result['physical']['legacy_rpc_reward'] == -1.
    assert result['physical']['combined_cost'] == 2.
    assert not result['physical']['complete_task']
    assert result['terminal_time_seconds'] == 28800.
    assert 'passed' in result['engineering_acceptance']
    engineering = result['engineering']
    assert engineering['dispatches_by_site'] == [2, 0, 0, 0]
    assert engineering['native_fallback_telemetry'] == {'total_callbacks': 3, 'callbacks_with_both_flags': 0}
    if policy == 'dispatch_inspired':
        assert dispatch.calls == ['choose', 'commit', 'choose', 'commit']
        assert engineering['first_live_multitruck_match']['matched_trucks'] == 3
        assert engineering['accepted_flow_plan_slack']['flow_grade_slack_tons']['mean'] == 3.
    if policy == 'rolling':
        assert engineering['action_counts']['work_budget_exhausted'] == 2
        assert engineering['action_counts']['retained_multi_step_requests'] == 2
    if policy == 'bg128':
        rng = engineering['actor_sampling_rng']
        assert rng['draws'] == 2 and rng['initial_sha256'] != rng['final_sha256']


@pytest.mark.parametrize('callback,committed', [(1, 0), (2, 1)])
def test_bridge_requires_previous_action_echo_before_commit_including_end(monkeypatch, callback, committed):
    dispatch = RecordingDispatch()
    monkeypatch.setattr(runtime, '_make_dispatch', lambda _method: dispatch)
    job, requests = _fixture_job('dispatch_inspired', wrong_echo=callback)
    with pytest.raises(ProtocolError, match='previous_action disagrees'):
        _collect(job, requests)
    assert dispatch.calls == ['choose', 'commit'] * committed + ['choose']


def test_reject_explicit_java_fallback_before_committing(monkeypatch):
    dispatch = RecordingDispatch()
    monkeypatch.setattr(runtime, '_make_dispatch', lambda _method: dispatch)
    job, requests = _fixture_job('dispatch_inspired', java_fallback=True)
    with pytest.raises(ProtocolError, match='dispatch fallback was used'):
        _collect(job, requests)
    assert dispatch.calls == ['choose']


def test_retain_physical_result_when_rolling_wall_guard_restores_root(monkeypatch):
    def select(_policy, _observation, mask, **_options):
        return decision.Decision(0, 'planned', timed_out=True, completed_candidates=sum(mask),
                                 multi_step_plans=1, max_scored_depth=2, initial_cost=2., selected_cost=2.)

    monkeypatch.setattr(decision, 'choose', select)
    job, requests = _fixture_job('rolling')
    result = _collect(job, requests)
    assert result['engineering']['action_counts']['wall_guard_restored_incumbent'] == 2
    assert not result['engineering_acceptance']['multi_step_trial_observed']
    assert not result['engineering_acceptance']['passed']
    assert result['engineering_acceptance']['rpc_and_physical_completion']


def test_reject_a_pair_whose_initial_observation_differs(monkeypatch):
    monkeypatch.setattr(decision, 'choose', lambda *_args, **_options:
                        decision.Decision(0, 'fallback', fallback_reason='synthetic fallback'))
    job, requests = _fixture_job('fifo_greedy')
    anchor = _collect(job, requests)
    assert anchor['engineering']['fallback_rate'] == 1.
    assert anchor['engineering_acceptance']['rpc_and_physical_completion']
    job, requests = _fixture_job('fifo_greedy')
    requests[0].obs.values[23] = np.float32(.45)
    with pytest.raises(ValueError, match='initial observation'):
        _collect(job, requests, anchor=anchor)
