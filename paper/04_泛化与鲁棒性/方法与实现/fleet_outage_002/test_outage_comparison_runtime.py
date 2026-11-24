"""Validate all four parked masks, paired-start checks, and 20 completed receipts."""

from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from business_graph_summary_v1.admission import frozen_actor
from change_plans.token_attention_001.business_graph_summary_009.fleet_outage_002 import fleet_runtime as runtime
from python.dispatch_env import DispatchEnv
from single_critic_task_query_v1.features_cache import pack_summary_prefixes
from single_critic_task_query_v1.test_scenario_bridge import InProcessDispatch
from traditional_dispatch_v2.test_runtime import _fixture_job

ROOT = Path(__file__).resolve().parents[4]
SAMPLES = (ROOT / 'runs/bg128-u32-generalization-20261119-001/evaluation'
           / 'nominal/episodes/run_9991001/samples.h5')
ACTOR = ROOT / 'runs/bg128-continuous-mixed-20261120-s02/evaluation/nominal/new/actor.pt'


def _park(state, mask, *, truck=None):
    state = state.copy()
    rows = state[27:1107].reshape(9, 120)
    rows[:, 1:4] = runtime.prior._TYPES
    rows[:, 16] = runtime.prior._CAPACITY
    rows[:, 0] = 0
    condition = runtime._condition(mask)
    predecessors = {}
    for index in condition['ACTIVE']:
        node = tuple(rows[index, 32:41])
        rows[index, 81] = predecessors.get(node, 0)
        predecessors[node] = (index + 1) / 9
    for index in condition['DISABLED']:
        rows[index, 4:16] = 0
        rows[index, 15] = 1
        rows[index, 17:41] = 0
        rows[index, 43:] = 0
    if truck is not None:
        rows[truck, 0] = rows[truck, 4] = 1
        rows[truck, 5:16] = 0
        state[23:27] = state[19:23] * rows[truck, 16]
    return state


@pytest.mark.parametrize('mask', runtime.MASKS)
def test_mask_on_full_in_process_transport(mask, monkeypatch):
    job, requests = _fixture_job('bg128')
    condition = runtime._condition(mask)
    job['expected_rpc_config'].update(condition['OUTAGE_FIELDS'])
    active = condition['ACTIVE']
    for index, request in enumerate(requests):
        truck = active[index] if request.action_required else None
        if truck is not None:
            request.truck_id = truck
        request.reward_components.update(condition['OUTAGE_FIELDS'])
        request.obs.values[:] = _park(np.asarray(request.obs.values, dtype=np.float32), mask, truck=truck)
    expected = runtime._non_fleet_configuration(requests[0].reward_components)
    bridge = runtime._bridge_for(mask, expected)(expected_episode_metadata=[(801, 701)])
    env = DispatchEnv(bridge, callback_timeout=2.)
    producer = InProcessDispatch(bridge, requests)
    monkeypatch.setattr(runtime.shared, '_actor_choice', lambda *_: 0)
    producer.start()
    try:
        result = runtime.prior._collect_with_outage(env, job, {'rollout_limit': 20}, None, object())
    finally:
        env.close()
        producer.join()
    assert not producer.errors
    assert result['engineering']['fleet_outage']['active_dispatches'] == {
        str(truck): int(truck in active[:2]) for truck in active}
    assert result['engineering']['fleet_outage']['disabled_ids'] == list(condition['DISABLED'])
    assert result['engineering']['fleet_outage']['validated_callbacks'] == 3
    assert result['physical']['combined_cost'] == 2
    assert runtime.prior.DISABLED == (2, 5, 8)


def test_extra_non_fleet_configuration_rejected_before_first_action():
    _, requests = _fixture_job('bg128')
    first = requests[0]
    first.reward_components.update(runtime._condition(448)['OUTAGE_FIELDS'])
    first.obs.values[:] = _park(np.asarray(first.obs.values, dtype=np.float32), 448, truck=first.truck_id)
    expected = runtime._non_fleet_configuration(first.reward_components)
    expected['config.diagnostic.unknown_to_the_standard_contract'] = 1.
    bridge = runtime._bridge_for(448, expected)(expected_episode_metadata=[(801, 701)])
    with pytest.raises(runtime.ProtocolError, match='outside fleet outage fields'):
        bridge.submit_callback(first, timeout=.1)
    with pytest.raises(runtime.ProtocolError, match='outside fleet outage fields'):
        bridge.next_callback(timeout=.1)


def test_frozen_actor_accepts_each_composition_on_sealed_nominal_start():
    if not (SAMPLES.exists() and ACTOR.exists()):
        pytest.skip('sealed observation or frozen Actor unavailable')
    with h5py.File(SAMPLES) as archive:
        nominal = archive['states'][0]
    states = []
    for mask in runtime.MASKS:
        active = runtime._condition(mask)['ACTIVE']
        truck = 3 if 3 in active else 6
        states.append(_park(nominal, mask, truck=truck))
    states = np.stack(states)
    actor, _ = frozen_actor(ACTOR)
    summary = pack_summary_prefixes(states, feature_config=actor.feature_config)
    with torch.inference_mode():
        result = actor.distribution(torch.from_numpy(states), summary).probs.numpy()
    assert result.shape == (4, 4) and np.isfinite(result).all()
    np.testing.assert_allclose(result.sum(1), np.ones(4), atol=1e-6)
    assert np.all(result[states[:, 19:23] == 0] == 0)
    assert len({tuple(row) for row in result}) == 4


def _suite(directory):
    scenarios = {
        scenario: dict(outage_version=1, disabled_mask=mask, active_count=6, physical_count=9,
                       disabled_truck_ids=list(runtime._condition(mask)['DISABLED']),
                       scene_id='nominal_fleet_outage_' + scenario, scenario_id=scenario)
        for scenario, mask in zip(runtime.SCENARIOS, runtime.MASKS, strict=True)
    }
    jobs = []
    for slot in range(5):
        for scenario, fleet in scenarios.items():
            index = len(jobs)
            jobs.append(dict(ordinal=index, pair_ordinal=slot, scene_id=fleet['scene_id'], fleet=fleet,
                             run_index=801 + index, recording_id=f'draft-{index}', policy_kind='bg128',
                             checkpoint_update=32, actor_seed=700, actor_sampling_seed_consumed=True,
                             experiment_seed=730 + slot, actor_sampling_seed=770 + slot,
                             phase_config=dict(path=f'phase-{index}'), runtime_scene_sha256='nominal-scene',
                             event_bundle=str(directory / f'events-{index}'),
                             python_output=str(directory / f'out-{index}'),
                             expected_rpc_config=runtime._condition(fleet['disabled_mask'])['OUTAGE_FIELDS']))
    return dict(schema=runtime.SUITE_SCHEMA, campaign_id='fixture', learning=None,
                evaluation_slots=5, physical_shifts=20, training_shifts=0, optimizer_updates=0,
                pairing=dict(slots=5, scenarios=list(runtime.SCENARIOS), policies=['bg128'],
                             share=['experiment_seed', 'actor_sampling_seed']),
                learner_config=dict(seed=700), fleet=scenarios, jobs=jobs,
                master_plan=dict(sha256='plan'), method={}, control_directory=str(directory / 'control'),
                report_directory=str(directory / 'reports'))


@pytest.mark.parametrize('field', ('experiment_seed', 'actor_sampling_seed', 'runtime_scene_sha256'))
def test_roster_rejects_unpaired_parameters(tmp_path, field):
    suite = _suite(tmp_path)
    suite['jobs'][1][field] = 'wrong'
    with pytest.raises(ValueError, match='same-slot seed or planned'):
        runtime._validate_suite(suite)


class _Ledger:
    def __init__(self, jobs):
        self.slots = {str(job['run_index']): dict(identity=job, status='PLANNED') for job in jobs}

    def snapshot(self):
        return {'slots': self.slots}


def _row(job, mask):
    condition = runtime._condition(mask)
    physical = dict(ore_tons=16000., waste_tons=4000., grade_cost=mask / 500., production_cost=0.,
                    waiting_cost=.1, combined_cost=mask / 500. + .1, total_wait_seconds=14400.,
                    complete_task=mask < 300, ore_target_met=True, waste_target_met=True,
                    windows=[dict(grade=.271, compliant=True) for _ in range(4)])
    engineering = dict(fleet_outage=dict(disabled_ids=list(condition['DISABLED']),
                                        active_ids=list(condition['ACTIVE']), validated_callbacks=7,
                                        active_dispatches={str(truck): 1 for truck in condition['ACTIVE']}),
                       actor_sampling_rng=dict(initial_sha256=str(job['actor_sampling_seed'])),
                       rpc_responses_over_target=0, rpc_receive_to_response_ms={'max': 10.})
    return dict(job, physical=physical, engineering=engineering,
                engineering_acceptance={'passed': True}, status='COMPLETE', normal_completion=True,
                completed_shifts=1, optimizer_updates=0,
                actual_rpc_configuration=dict(condition['OUTAGE_FIELDS'], **{'config.scene': 1}),
                initial_state={'observation_sha256': str(mask)})


def test_all_twenty_jobs_compare_per_slot_to_balanced(tmp_path, monkeypatch):
    suite = _suite(tmp_path)
    ledger = _Ledger(suite['jobs'])
    progress = []
    monkeypatch.setattr(runtime.shared, '_check_startup', lambda _: None)
    monkeypatch.setattr(runtime.shared, '_load_actor', lambda _: (object(), 'actor'))
    monkeypatch.setattr(runtime.shared, '_publish_json', lambda path, report: (path, report))

    def fake_shift(_suite, _startup, ledger, job, config, _actor, _digest):
        mask = job['fleet']['disabled_mask']
        assert config == (None if mask == 448 else {'config.scene': 1})
        receipt = {'path': str(Path(job['python_output']) / 'evaluation_receipt.json')}
        ledger.slots[str(job['run_index'])].update(status='COMPLETE', receipt=receipt)
        return _row(job, mask), receipt

    monkeypatch.setattr(runtime, '_run_shift', fake_shift)
    path, report = runtime.run(suite, ledger, progress=lambda n, job: progress.append((n, job['ordinal'])))
    assert path.name == runtime.REPORT_FILENAME
    assert progress == [(number + 1, number) for number in range(20)]
    assert report['completed_shifts'] == len(report['receipts']) == 20
    assert list(report['summaries']) == list(runtime.SCENARIOS)
    assert report['reference_scenario'] == 'balanced_out'
    assert [slot['pair_ordinal'] for slot in report['slots']] == list(range(5))
    assert report['paired_differences_from_balanced']['heavy_out']['mean_difference']['combined_cost'] == pytest.approx(.312)
    assert report['paired_differences_from_balanced']['heavy_out']['only_balanced_complete'] == 5
    assert report['paired_differences_from_balanced']['small_out']['joint_cost_lower_slots'] == 5

    broken = deepcopy(report['shifts'][:4])
    broken[2]['engineering']['actor_sampling_rng']['initial_sha256'] = 'different'
    with pytest.raises(ValueError, match='Actor RNG start differs'):
        runtime._verify_pair(broken, report['receipts'][:4], suite['jobs'][:4])
