"""Check the six-active RPC contract and frozen Actor without opening a listener."""

from hashlib import sha256
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from business_graph_summary_v1.admission import frozen_actor
from business_graph_summary_v1.graph import build_graph
from python.dispatch_env import DispatchEnv
from single_critic_task_query_v1.features_cache import pack_summary_prefixes
from single_critic_task_query_v1.test_scenario_bridge import InProcessDispatch
from traditional_dispatch_v2.test_runtime import _fixture_job

import fleet_runtime as runtime


ROOT = Path(__file__).resolve().parents[4]
STATE_FILE = (ROOT / 'runs/bg128-u32-generalization-20261119-001/evaluation'
              / 'nominal/episodes/run_9991001/samples.h5')
ACTOR_FILE = (ROOT / 'runs/bg128-continuous-mixed-20261120-s02'
              / 'evaluation/nominal/new/actor.pt')
ACTOR_SHA256 = 'e1575fec566c647a882c63934ec57c908cef28d43f076431ef9103cc9b73e0a9'
STATE_SHA256 = '47c42489284bac90b374fc4218c9d77098a6fc3576bc0b92c3302637c5acc814'


def _park(state):
    projected = state.copy()
    trucks = projected[27:1107].reshape(9, 120)
    trucks[:, 1:4] = runtime._TYPES
    trucks[:, 16] = runtime._CAPACITY
    for truck in runtime.DISABLED:
        physical = trucks[truck, [1, 2, 3, 16, 41, 42]].copy()
        trucks[truck] = 0
        trucks[truck, [1, 2, 3, 16, 41, 42]] = physical
        trucks[truck, 15] = 1
    trucks[6, 81] = 5 / 9
    return projected


@pytest.fixture(scope='module')
def sealed_state():
    if not STATE_FILE.exists():
        pytest.skip('sealed nominal source snapshot is unavailable')
    with h5py.File(STATE_FILE) as archive:
        state = archive['states'][0]
    assert sha256(state.tobytes()).hexdigest() == STATE_SHA256
    return state


def test_frozen_actor_on_archived_nominal_and_counterfactual_six_active(sealed_state):
    if not ACTOR_FILE.exists():
        pytest.skip('sealed BG128 s02/u32 Actor artifact is unavailable')
    assert sha256(ACTOR_FILE.read_bytes()).hexdigest() == ACTOR_SHA256
    actor, digest = frozen_actor(ACTOR_FILE)
    assert digest == '189a2c571c4bcf0fbe978330d92dddb84f3b196b753f7ed93e533988d4c59abd'
    six = _park(sealed_state)
    runtime.validate_outage(six, runtime.OUTAGE_FIELDS, truck_id=3, terminal=False)
    samples = np.stack((sealed_state, six))
    summary = pack_summary_prefixes(samples, feature_config=actor.feature_config)
    with torch.inference_mode():
        probabilities = actor.distribution(torch.from_numpy(samples), summary).probs.numpy()
    assert probabilities.shape == (2, 4) and np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(1), np.ones(2), atol=1e-6)
    assert np.all(probabilities[samples[:, 19:23] == 0] == 0)
    assert not np.array_equal(probabilities[0], probabilities[1])
    graph = build_graph(six[None, :])
    assert graph.quantities.shape[:2] == (1, 19)
    assert not np.isin(graph.edge_index.numpy(), runtime.DISABLED).any()
    generator = torch.Generator(device='cpu').manual_seed(1971)
    action = runtime.shared._actor_choice(actor, generator, six)
    assert six[19 + action] == 1


@pytest.mark.parametrize('index,value', (
    (27 + 120*2, 1), (27 + 120*2 + 15, 0), (27 + 120*5 + 19, .1),
    (27 + 120*8 + 20, .1), (27 + 120*5 + 45, 1), (27 + 120*5 + 80, 1),
    (27 + 120*6 + 81, 6/9), (1107 + 20, 3/9), (1107 + 29, 3/9),
    (1107 + 38, 3/9), (1295 + 11, 9/9), (1295 + 12, 9/9),
    (1295 + 21, 9/9), (1295 + 30, 9/9), (1295 + 39, 9/9),
))
def test_rejects_disabled_activity_and_cross_references(sealed_state, index, value):
    state = _park(sealed_state)
    state[index] = value
    with pytest.raises(runtime.ProtocolError, match='disabled truck'):
        runtime.validate_outage(state, runtime.OUTAGE_FIELDS, truck_id=3, terminal=False)


def _requests():
    job, requests = _fixture_job('bg128')
    job['expected_rpc_config'].update(runtime.OUTAGE_FIELDS)
    for request in requests:
        request.reward_components.update(runtime.OUTAGE_FIELDS)
        request.obs.values[:] = _park(np.asarray(request.obs.values, dtype=np.float32))
    return job, requests


def _collect(job, requests, monkeypatch):
    monkeypatch.setattr(runtime.shared, '_actor_choice', lambda *_: 0)
    bridge = runtime.FleetOutageBridge(expected_episode_metadata=[(801, 701)])
    env = DispatchEnv(bridge, callback_timeout=2.)
    producer = InProcessDispatch(bridge, requests)
    producer.start()
    try:
        result = runtime._collect_with_outage(env, job, {'rollout_limit': 20}, None, object())
    finally:
        env.close()
        producer.join()
    assert not producer.errors
    return result


def test_callback_start_decision_end_and_shared_physical_settlement(monkeypatch):
    job, requests = _requests()
    result = _collect(job, requests, monkeypatch)
    outage = result['engineering']['fleet_outage']
    assert outage['validated_callbacks'] == 3
    assert sum(outage['active_dispatches'].values()) == 2
    assert set(outage['active_dispatches']) == {str(truck) for truck in runtime.ACTIVE}
    assert outage['disabled_ids'] == list(runtime.DISABLED)
    assert result['physical']['combined_cost'] == 2
    assert result['physical']['ore_tons'] == 14000
    assert result['physical']['waste_tons'] == 3500
    assert result['engineering_acceptance']['fleet_outage_validated']
    assert not result['engineering_acceptance']['all_six_active_trucks_dispatched']
    assert not result['engineering_acceptance']['passed']


@pytest.mark.parametrize('at,field,error', (
    (0, 'config', 'fleet outage configuration'),
    (1, 'position', 'parking coordinates'),
    (2, 'reference', 'disabled truck appears'),
))
def test_bad_outage_stops_the_waiting_python_bridge(monkeypatch, at, field, error):
    job, requests = _requests()
    bad = requests[at]
    if field == 'config':
        bad.reward_components['config.fleet.disabled_mask'] = 293
    elif field == 'position':
        bad.obs.values[27 + 120*2 + 41] = .125
    else:
        bad.obs.values[1295 + 39] = 3/9
    with pytest.raises(runtime.ProtocolError, match=error):
        _collect(job, requests, monkeypatch)
