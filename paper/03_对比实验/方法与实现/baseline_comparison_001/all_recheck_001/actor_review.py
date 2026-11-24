"""Replay one sealed s02/u32 nominal shift entirely in process."""
from hashlib import sha256
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

from business_graph_summary_campaign_v1.plan_io import load_reference, reference
from business_graph_summary_v1.admission import frozen_actor
from business_graph_summary_v1.contract import require
from python.constrained_batch import load_dataset
from python.dispatch_bridge import DispatchCallbackBridge
from python.dispatch_env import DispatchEnv
from python.runtime_scenario import PREFIX, fingerprint
from single_critic_task_query_v1.test_scenario_bridge import InProcessDispatch, dispatch_request
from traditional_dispatch_v2.runtime import _collect


RUN_INDEX = 10004025
ARCHIVE = ROOT / 'runs/bg128-continuous-mixed-20261120-s02/evaluation/nominal/new'
RECORDING_FLAGS = {'config.material_event_memory_enabled': 0.,
                   'config.wait_breakdown_enabled': 1., 'config.wait_breakdown_version': 1.}


def request(info, observation, scene_sha, *, terminal=False):
    components = info['reward_components']
    scene = {key: value for key, value in components.items() if key.startswith(PREFIX)}
    require(fingerprint(scene) == scene_sha, 'archived callback differs from the frozen runtime scene')
    values = dict(info, reward_components=dict(components, **RECORDING_FLAGS))
    rpc = dispatch_request(values, terminal=terminal)
    rpc.obs.values[:] = observation
    return rpc


def review():
    episode_dir = ARCHIVE / 'episodes' / f'run_{RUN_INDEX}'
    receipt = json.loads((episode_dir / 'receipt.json').read_bytes())
    phase = load_reference(receipt['phase_config'])
    require(phase == json.loads((ARCHIVE / 'phase_config.json').read_bytes()),
            'historical phase copy differs from its sealed reference')
    job = next(dict(item, policy_kind='bg128') for item in phase['jobs'] if item['run_index'] == RUN_INDEX)
    seed, sampling_seed = job['experiment_seed'], job['actor_sampling_seed']
    require(job['checkpoint_update'] == 32 and job['scene_id'] == 'nominal'
            and seed == receipt['experiment_seed'] and sampling_seed > 0, 'historical job identity differs')
    job['expected_rpc_config'] = dict(job['expected_rpc_config'], **RECORDING_FLAGS)
    scene_sha = job['runtime_scene_sha256']

    rows, episodes, _, _, _ = load_dataset(episode_dir, require_real=True)
    try:
        require(len(episodes) == 1 and len(rows) == receipt['transitions'] == 185,
                'expected the complete 185-action archived shift')
        requests = [request(episodes[0]['start_info'], rows[0]['observation'], scene_sha)]
        requests.extend(request(row['settlement_info'], row['next_observation'], scene_sha,
                                terminal=row['terminated']) for row in rows)
        expected = [row['action'] for row in rows]
    finally:
        rows.close()

    actor, digest = frozen_actor(ARCHIVE / 'actor.pt')
    require(digest == receipt['policy_version'] == episodes[0]['policy_version'],
            'Actor weights differ from those used by the archived shift')
    bridge = DispatchCallbackBridge(expected_episode_metadata=[(RUN_INDEX, seed)])
    env = DispatchEnv(bridge, callback_timeout=3.)
    producer = InProcessDispatch(bridge, requests)
    global_rng = torch.random.get_rng_state()
    producer.start()
    try:
        result = _collect(env, job, actor=actor)
    finally:
        env.close()
        producer.join()
    replies = [response.action for response in producer.responses if response.has_action]
    require(not producer.errors and len(producer.responses) == 186 and not producer.responses[-1].has_action,
            'replay did not complete through the original in-process RPC handler')
    require(replies == expected and torch.equal(global_rng, torch.random.get_rng_state()),
            'historical action sequence or private sampling isolation differs')
    rng = result['engineering']['actor_sampling_rng']
    require(rng['draws'] == 185 and result['terminal_time_seconds'] == 28800
            and len(result['physical']['windows']) == 4, 'replay settlement did not reconcile')
    return dict(status='PASS_OFFLINE_REPLAY', run_index=RUN_INDEX, checkpoint_update=32,
                archived_dataset=reference(episode_dir / 'manifest.json'), archived_phase=receipt['phase_config'],
                actor=reference(ARCHIVE / 'actor.pt'), actor_digest=digest, experiment_seed=seed,
                actor_sampling_seed=sampling_seed, actor_sampling_rng=rng, global_torch_rng_unchanged=True,
                observation_sha256=result['initial_state']['observation_sha256'], runtime_scene_sha256=scene_sha,
                action_sequence_sha256=sha256(bytes(replies)).hexdigest(), matching_actions=len(replies),
                rpc_callbacks=len(producer.responses), terminal_time_seconds=result['terminal_time_seconds'],
                valid_terminal_windows=len(result['physical']['windows']),
                replay_instrumentation_overrides=RECORDING_FLAGS)


if __name__ == '__main__':
    torch.set_num_threads(2)
    print(json.dumps(review(), indent=2, sort_keys=True))
