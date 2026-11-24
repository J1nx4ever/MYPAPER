"""Actual dispatch service/bridge -> env -> buffer scenario-v2 regression.

Requests use the real protobuf, service handler and DispatchCallbackBridge.
The producer runs in process and never opens a socket or simulator. Fixtures
are based on online_v1_1.synthetic's complete physical/reward RPC records.
"""
import threading
import grpc
import numpy as np
import pytest
from online_v1_1.synthetic import info,observation,window
from python.generated import dispatch_pb2 as pb
from python.dispatch_bridge import DispatchCallbackBridge,ProtocolError
from python.dispatch_service import DispatchService
from python.dispatch_env import DispatchEnv
from python.constrained_batch import EpisodeBatchBuffer,collect_batch,load_dataset
from python.scenario_contract import from_components,metadata,to_components

V1_DICTIONARY = dict(waste_task_limit_enabled=True,waste_task_limit_version=1,
    waste_target_tons=4000.,waste_task_upper_tons=4193.,waste_task_overrun_tons=193.)
V1_FINGERPRINT = 'a958710695fcf499cf59607080402f4bf93dc4f195396056e7d0bae557c4302f'


def rpc_info(time,sequence,*,version=2,terminal=False):
    value = info(time,sequence,episode='synthetic-scenario-bridge',seed=701,
        elapsed=time if sequence == 1 else (time-20. if sequence == 2 else 0.),
        terminal=terminal,previous_action=-1 if sequence == 0 else 0,
        windows=[window(index,.27) for index in range(4)] if terminal else [])
    value['experiment_run_index'] = 801
    if version == 2:
        value['reward_components'].update({'config.waste_task_limit_version':2.,
            'config.waste_task_dispatch_stop_tons':6000.,'config.waste_task_upper_tons':6193.})
    return value


def dispatch_request(value,*,terminal=False):
    request = pb.DispatchRequest(truck_id=value['truck_id'],episode_id=value['episode_id'],
        request_id=value['request_id'],decision_sequence=value['decision_sequence'],
        kind=pb.EPISODE_END if terminal else (pb.EPISODE_START if value['decision_sequence'] == 0 else pb.DECISION),
        action_required=not terminal,previous_action=value['previous_action'],
        transition_reward=value['transition_reward_raw'],reward_components=value['reward_components'],
        sim_time_seconds=value['sim_time_seconds'],sim_time_delta_seconds=value['sim_time_delta_seconds'],
        terminated=terminal,truncated=False,experiment_run_index=value['experiment_run_index'],
        experiment_seed=value['experiment_seed'],reward_contract_id=value['reward_contract_id'],
        obs=pb.Observation(schema_id=value['observation_schema_id'],
            values=observation(value['sim_time_seconds'],terminal=terminal)))
    for summary in value['closed_windows']:
        request.closed_windows.add(**summary)
    return request


class RpcAbort(Exception):
    def __init__(self,code,details):
        super().__init__(details)
        self.code = code


class Context:
    def time_remaining(self):
        return 2.
    def abort(self,code,details):
        raise RpcAbort(code,details)


class InProcessDispatch:
    """Invoke the real gRPC handler without a server/listener."""
    def __init__(self,bridge,requests):
        self.service = DispatchService(bridge)
        self.requests = requests
        self.responses = []
        self.errors = []
        self.thread = threading.Thread(target=self.produce,daemon=True)
    def produce(self):
        try:
            for request in self.requests:
                self.responses.append(self.service.GetNextDestination(request,Context()))
        except BaseException as error:
            self.errors.append(error)
    def start(self):
        self.thread.start()
    def join(self):
        self.thread.join(timeout=3.)
        assert not self.thread.is_alive(),'synthetic dispatch producer leaked'


class FixedBehavior:
    policy_version = 'synthetic-scenario-bridge-policy'
    def sample(self,state,mask):
        assert mask[0] == 1 and state.shape == (1519,)
        return 0,-float(np.log(4.))
    def assert_frozen(self):
        return None


def collect_fixture(tmp_path,*,version=2,mutate_start=None,mutate_decision=None):
    start = rpc_info(0.,0,version=version)
    decision = rpc_info(20.,1,version=version)
    end = rpc_info(28800.,2,version=version,terminal=True)
    if mutate_start:
        mutate_start(start['reward_components'])
    if mutate_decision:
        mutate_decision(decision['reward_components'])
    requests = [dispatch_request(start),dispatch_request(decision),dispatch_request(end,terminal=True)]
    bridge = DispatchCallbackBridge(runtime_configuration=True,expected_episode_metadata=[(801,701)])
    producer = InProcessDispatch(bridge,requests)
    env = DispatchEnv(bridge,callback_timeout=2.)
    buffer = EpisodeBatchBuffer(tmp_path/'synthetic-buffer',expected_episodes=1,source_kind='synthetic_unit_test')
    settings = dict(kappa=4e-5,ore_weight=4.,waste_weight=4.,grade_lower=.26,grade_upper=.28,
        grade_epsilon=1e-12,tonnage_epsilon=1e-6,mass_balance_epsilon=1e-5)
    jobs = [dict(run_index=801,experiment_seed=701,scene_id='synthetic-scene',scene_group='synthetic-scene-group')]
    producer.start()
    try:
        collect_batch(env,FixedBehavior(),buffer,jobs,reward_settings=settings)
    except BaseException as error:
        error.fixture_bridge = bridge
        error.fixture_producer = producer
        error.fixture_buffer = buffer
        raise
    finally:
        env.close()
        producer.join()
    return bridge,producer,buffer,start


def test_v2_start_reaches_real_bridge_and_sealed_buffer_without_changing_4000_target(tmp_path):
    bridge,producer,buffer,start = collect_fixture(tmp_path)
    assert not producer.errors and len(producer.responses) == 3
    assert [response.has_action for response in producer.responses] == [True,True,False]
    assert buffer.complete and len(buffer.rows) == 2 and len(buffer.episodes) == 1
    stored = buffer.episodes[0]['start_info']['reward_components']
    assert stored == start['reward_components']
    assert stored['config.target_waste_tons_shift'] == 4000.
    assert stored['config.waste_task_dispatch_stop_tons'] == 6000.
    assert stored['config.waste_task_upper_tons'] == 6193.
    scenario = bridge.scenario_config
    assert scenario.waste_target_tons == 4000. and scenario.waste_task_dispatch_stop_tons == 6000.
    assert scenario.waste_task_upper_tons == 6193. and scenario.waste_task_overrun_tons == 193.
    assert scenario.scenario_id == 'dispatch-scenario-waste-task-v2'
    assert scenario.fingerprint != V1_FINGERPRINT
    assert from_components(to_components(scenario)) == scenario
    assert buffer.episodes[0]['scenario_fingerprint'] == scenario.fingerprint
    rows,episodes,_,_,manifest = load_dataset(buffer.output_dir,require_real=False)
    assert len(rows) == 2 and episodes[0]['scenario_fingerprint'] == scenario.fingerprint
    assert manifest['source_kind'] == 'synthetic_unit_test'


def test_v1_identity_dict_id_and_fingerprint_are_byte_compatible_through_bridge(tmp_path):
    bridge,producer,buffer,_ = collect_fixture(tmp_path,version=1)
    assert not producer.errors and buffer.complete
    scenario = bridge.scenario_config
    assert scenario.to_dict() == V1_DICTIONARY
    assert 'waste_task_dispatch_stop_tons' not in scenario.to_dict()
    assert scenario.scenario_id == 'dispatch-scenario-waste-task-v1'
    assert scenario.fingerprint == V1_FINGERPRINT
    assert metadata(scenario) == dict(scenario_id='dispatch-scenario-waste-task-v1',
        scenario_configuration=V1_DICTIONARY,scenario_fingerprint=V1_FINGERPRINT)
    assert from_components(to_components(scenario)) == scenario
    assert buffer.episodes[0]['scenario_fingerprint'] == V1_FINGERPRINT


@pytest.mark.parametrize('mutate',[
    lambda c:c.pop('config.waste_task_dispatch_stop_tons'),
    lambda c:c.pop('config.waste_task_limit_version'),
    lambda c:c.update({'config.waste_task_limit_version':3.}),
    lambda c:c.update({'config.waste_task_unknown':1.}),
    lambda c:c.update({'config.waste_task_upper_tons':4193.}),
    lambda c:c.update({'config.waste_task_dispatch_stop_tons':float('nan')}),
])
def test_invalid_v2_start_fails_closed_in_real_handler_before_buffer_append(tmp_path,mutate):
    with pytest.raises(ProtocolError) as caught:
        collect_fixture(tmp_path,mutate_start=mutate)
    error = caught.value
    assert not error.fixture_buffer.complete and not error.fixture_buffer.rows
    assert not error.fixture_buffer.episodes
    assert isinstance(error.fixture_producer.errors[0],RpcAbort)
    assert error.fixture_producer.errors[0].code is grpc.StatusCode.INVALID_ARGUMENT
    assert error.fixture_bridge.last_failure['request_id'] == 'synthetic-scenario-bridge-0'
    with pytest.raises(ProtocolError) as waiting:
        error.fixture_bridge.next_callback(timeout=.1)
    assert waiting.value is error
    assert (error.fixture_buffer.output_dir/'ABORTED.json').exists()
    assert not (error.fixture_buffer.output_dir/'manifest.json').exists()


@pytest.mark.parametrize('mutate',[
    lambda c:c.update({'config.waste_task_dispatch_stop_tons':6001.,'config.waste_task_upper_tons':6194.}),
    lambda c:c.update({'config.target_waste_tons_shift':4100.}),
    lambda c:(c.pop('config.waste_task_dispatch_stop_tons'),c.update({'config.waste_task_limit_version':1.,'config.waste_task_upper_tons':4193.})),
    lambda c:c.pop('config.waste_task_dispatch_stop_tons'),
])
def test_v2_drift_after_valid_start_reaches_waiting_sampler_and_never_seals(tmp_path,mutate):
    with pytest.raises(ProtocolError) as caught:
        collect_fixture(tmp_path,mutate_decision=mutate)
    error = caught.value
    assert error.fixture_bridge.scenario_config.waste_task_limit_version == 2
    assert error.fixture_bridge.scenario_config.waste_target_tons == 4000.
    assert len(error.fixture_producer.responses) == 1 and error.fixture_producer.responses[0].has_action
    assert isinstance(error.fixture_producer.errors[0],RpcAbort)
    assert error.fixture_producer.errors[0].code is grpc.StatusCode.INVALID_ARGUMENT
    assert not error.fixture_buffer.complete and not error.fixture_buffer.rows
    assert not (error.fixture_buffer.output_dir/'manifest.json').exists()
