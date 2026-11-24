"""New-policy synthetic sealed RPC/journal fixtures; never start a simulator."""
import copy
from dataclasses import replace
import hashlib
from types import SimpleNamespace
import numpy as np
import torch
from grade_event_memory_v1.test_admission import _json, _state as _old_state, _window
from grade_event_memory_v1.test_events import synthetic_events
from grade_event_memory_v1.events import SCHEMA
from grade_event_memory_v1.journal import TOKENIZATION, synthetic_journal_bytes
from .targets import REWARD_VERSION, composite_contract
from online_v1_1.synthetic import info
from python.constrained_adapter import RewardCostAdapter
from python.constrained_batch import EpisodeBatchBuffer
from .models import digest
from python.reward_config import configuration_from_components
from python.scenario_contract import from_components
from python.storage_contract import HDF5


def _state(*args, **kwargs):
    value = _old_state(*args, **kwargs)
    value[11:16] = [.8, 1., .26, .28, .27]
    value[1300] = kwargs.get('fifo', 0.) / 500.
    value[1302] = 1. - value[1300]
    value[1303:1305] = 1.
    return value

def prepare_admission(tmp_path, *, old_probability_delta=0., q1=20., fifo1=20.,
                      rpc_windows=(30., 10., 0., 0.), candidate_id='s01_G2'):
    """Create sealed engineering artifacts, never a simulator or learner."""
    from .learner import ACTOR_SCHEMA, default_config, implementation_signature
    from .models import actor_from_config, pack_summary_prefixes
    from .c_features import feature_contract
    from .contract import ablation_contract
    learner_config = default_config(source_kind='synthetic', seed=17,
        candidate_id=candidate_id, execution_profile='synthetic_test_v1')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        actor = actor_from_config(learner_config).eval()
    for parameter in actor.parameters(): parameter.requires_grad_(False)
    policy = digest(actor)
    actor_path = tmp_path / 'actor.pt'
    torch.save(dict(schema=ACTOR_SCHEMA, config=learner_config,
                    actor=actor.state_dict(), actor_digest=policy,
                    implementation=implementation_signature(),
                    graph_contract=ablation_contract(candidate_id)), actor_path)
    episode = "synthetic-material-admission-001"
    recording = "synthetic-material-admission-recording-001"
    states = [
        _state(0),
        _state(20, q=q1, fifo=fifo1, shift=20),
        _state(7200, fifo=10, shift=30),
        _state(28800, terminal=True, shift=sum(rpc_windows)),
    ]
    start = info(0, 0, episode=episode, seed=1)
    cfg = configuration_from_components(start["reward_components"])
    adapter = RewardCostAdapter(cfg)
    adapter.start(start)
    scenario = from_components(start["reward_components"])
    raw_dir = tmp_path / "raw"
    buffer = EpisodeBatchBuffer(raw_dir, expected_episodes=1,
        source_kind="synthetic_unit_test", storage_format=HDF5)
    times = (0., 20., 7200., 28800.)
    windows = [_window(i, q) for i, q in enumerate(rpc_windows)]
    cumulative = 0.
    generator=torch.Generator(device='cpu');generator.manual_seed(29)
    def rng_sha(): return hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(29)
        for index in range(3):
            obs, nxt = states[index:index + 2]
            with torch.no_grad():
                prefix = pack_summary_prefixes(np.stack(states[:-1]), indices=[index],
                    feature_config=actor.feature_config)
                distribution = actor.distribution(torch.from_numpy(obs)[None, :], prefix)
                before=rng_sha()
                action_tensor = torch.multinomial(distribution.probs,1,replacement=True,
                    generator=generator).reshape(1)
                action = int(action_tensor.item())
                old = float(distribution.log_prob(action_tensor).item())
            terminal = index == 2
            delta = 3. if index == 0 else (7. if terminal else 0.)
            cumulative += delta
            newly_closed = windows[:1] if index == 1 else (windows[1:] if terminal else [])
            settlement = info(times[index + 1], index + 1, episode=episode, seed=1,
                windows=newly_closed, delta=delta, cumulative=cumulative,
                elapsed=times[index + 1] - times[index], terminal=terminal,
                previous_action=action, mask=nxt[19:23].astype(int).tolist())
            if terminal:
                total = sum(rpc_windows)
                settlement["reward_components"].update(
                    production=-4 * (1-total/16000), waste=-4., ore_tons_shift=total,
                    waste_tons_shift=0., crusher_received_tons=total, crusher_buffer_tons=0.,
                    mined_ore_tons=total, mined_waste_tons=0., carried_ore_tons=0.,
                    carried_waste_tons=0., load_sites_reserved_tons=0., assigned_not_loaded_tons=0.,
                    shift_grade=.27)
                settlement["transition_reward_raw"] = sum(settlement["reward_components"][k]
                    for k in ("wait", "production", "waste"))
            adapted = adapter.transition(settlement, terminated=terminal, truncated=False)
            buffer.append(dict(**adapted, observation=obs, next_observation=nxt,
                truck_id=0, next_truck_id=settlement["truck_id"], episode_id=episode,
                experiment_seed=1, scene_id="synthetic-scene", scene_group="synthetic-group",
                action=action, mask=obs[19:23].astype(int).tolist(),
                behavior_kind='frozen_actor',critic_only=False,
                actor_sampling_seed=29,actor_sampling_draw_index=index,
                actor_rng_before_sha256=before,actor_rng_after_sha256=rng_sha(),
                old_log_prob=old + (old_probability_delta if index == 0 else 0.),
                policy_version=policy, settlement_info=copy.deepcopy(settlement),
                action_time_seconds=times[index], scenario_fingerprint=scenario.fingerprint))
    buffer.finish_episode(0, start_info=start, initial_observation=states[0],
        job=dict(scene_id="synthetic-scene", scene_group="synthetic-group"),
        scenario_fingerprint=scenario.fingerprint)
    buffer.seal()
    if hasattr(buffer.rows, "close"): buffer.rows.close()
    phase = dict(rpc_configuration_authoritative=True, require_local_alp_audit=False,
        feature_contract=feature_contract(),graph_contract=ablation_contract(candidate_id),
        composite_contract=composite_contract(),training_reward='negative_combined_grade_production_wait_cost_v1',
        candidate_id=candidate_id, execution_profile='synthetic_test_v1',
        recording_id=recording, material_memory=dict(contract=SCHEMA, memory_scope="whole_shift",
            tokenization_version=TOKENIZATION, reward_version=REWARD_VERSION),
        jobs=[dict(run_index=1, experiment_seed=1, actor_sampling_seed=29, actor_seed=17,
            expected_rpc_config=cfg.to_dict())])
    config_path = tmp_path / "phase.json"
    _json(config_path, phase)
    events = tuple(replace(e, request_id=f"{episode}-{e.decision_sequence}")
        if e.kind in ("DECISION", "END") else e for e in synthetic_events())
    header = dict(tokenization_version=TOKENIZATION, source_kind="synthetic",
        recording_id=recording, episode_id=episode,
        config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(), run_index=1, seed=1,
        shift_seconds=28800., window_seconds=7200., target_grade=.27, grade_lower=.26, grade_upper=.28)
    journal_path = tmp_path / "material_events.bin"
    journal_path.write_bytes(synthetic_journal_bytes(header, events))
    return SimpleNamespace(raw=raw_dir, actor=actor_path, config=config_path,
        journal=journal_path, header=header, events=events, output=tmp_path / "admitted.h5",
        actor_object=actor, states=np.stack(states), learner_config=learner_config)

