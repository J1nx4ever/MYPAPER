"""Synthetic RPC -> journal -> admitted HDF5 acceptance, without collection/fit."""

import copy
from dataclasses import replace
import hashlib
import json
import struct
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from online_v1_1.synthetic import info, observation
from python.constrained_adapter import RewardCostAdapter
from python.constrained_batch import EpisodeBatchBuffer
from python.constrained_ppo import Actor, digest
from python.reward_config import configuration_from_components
from python.scenario_contract import from_components
from python.storage_contract import HDF5

from .admission import admit_episode
from .events import SCHEMA, grade_reward
from .journal import TOKENIZATION, synthetic_journal_bytes
from .storage import read_episode_hdf5
from .targets import REWARD_VERSION
from .test_events import synthetic_events


def _json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def _window(index, tons):
    metal = tons * .27
    grade, valid, raw = grade_reward(tons, metal)
    return dict(window_index=index, window_start_seconds=index * 7200.,
        window_end_seconds=(index + 1) * 7200., ore_tons=tons, metal_tons=metal,
        grade=grade, grade_gap=.27-grade, grade_reward_raw=raw,
        grade_reward_weighted=0., violation_upper=False, violation_lower=not valid)


def _state(time, *, terminal=False, q=0., fifo=0., shift=0.):
    value = observation(time, terminal=terminal)
    value[3] = max(0., (min(4, int(time // 7200)) + 1) * 7200. - time) / 7200.
    if terminal: value[3] = 0
    value[4:6] = [q / 5000., q * .27 / 5000.]
    value[6] = q == 0
    value[7:9] = [shift / 20000., shift * .27 / 20000.]
    if fifo:
        value[1391:1394] = [1., fifo / 500., .27]
    return value


def prepare_admission(tmp_path, *, old_probability_delta=0., q1=20., fifo1=20.,
                      rpc_windows=(30., 10., 0., 0.)):
    """Create sealed engineering artifacts, never a simulator or learner."""
    spec = dict(hidden_sizes=[8], activation="tanh", initialization="torch_default")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        actor = Actor(spec).eval()
    for parameter in actor.parameters(): parameter.requires_grad_(False)
    policy = digest(actor)
    actor_path = tmp_path / "actor.pt"
    torch.save(dict(actor=actor.state_dict(), network=spec, actor_digest=policy), actor_path)
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
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(29)
        for index in range(3):
            obs, nxt = states[index:index + 2]
            with torch.no_grad():
                distribution = actor.distribution(torch.from_numpy(obs)[None, :])
                action_tensor = distribution.sample()
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
                old_log_prob=old + (old_probability_delta if index == 0 else 0.),
                policy_version=policy, settlement_info=copy.deepcopy(settlement),
                action_time_seconds=times[index], scenario_fingerprint=scenario.fingerprint))
    buffer.finish_episode(0, start_info=start, initial_observation=states[0],
        job=dict(scene_id="synthetic-scene", scene_group="synthetic-group"),
        scenario_fingerprint=scenario.fingerprint)
    buffer.seal()
    if hasattr(buffer.rows, "close"): buffer.rows.close()
    phase = dict(rpc_configuration_authoritative=True, require_local_alp_audit=False,
        recording_id=recording, material_memory=dict(contract=SCHEMA, memory_scope="whole_shift",
            tokenization_version=TOKENIZATION, reward_version=REWARD_VERSION),
        jobs=[dict(run_index=1, experiment_seed=1, actor_sampling_seed=29,
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
        journal=journal_path, header=header, events=events, output=tmp_path / "admitted.h5")


def _admit(case, **kwargs):
    return admit_episode(case.raw, case.journal, config_path=case.config,
        actor_path=case.actor, output_path=case.output, split="train", **kwargs)


def _no_publication(case):
    assert not case.output.exists()
    assert not case.output.with_name(case.output.name + ".manifest.json").exists()
    assert not case.output.with_name(case.output.name + ".seal.json").exists()


def test_end_to_end_synthetic_rpc_journal_hdf5_without_optimizer(tmp_path, monkeypatch):
    def forbid(*args, **kwargs):
        pytest.fail("data preparation must not construct an optimizer")
    monkeypatch.setattr(torch.optim.Optimizer, "__init__", forbid)
    case = prepare_admission(tmp_path)
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (case.actor, case.config, case.journal, case.raw / "samples.h5")}
    result = _admit(case, allow_synthetic=True)
    assert result["status"] == "PASS_DATA_PREPARATION_ONLY"
    assert result["transitions"] == 3
    assert not result["actor_or_critic_fitted"] and not result["collection_started"]
    assert not result["ppo_authorized"]
    data = read_episode_hdf5(case.output)
    assert data["identity"]["source_kind"] == "synthetic"
    assert data["identity"]["split"] == "synthetic"
    assert data["evaluation"]["ore_tons"] == 40
    assert data["evaluation"]["waste_tons"] == 0
    assert data["evaluation"]["total_wait_seconds"] == 10
    assert data["evaluation"]["production_shortfall"]
    assert data["targets"].mc_returns == pytest.approx((4., 4., 4.))
    assert data["memory"].queries[2].current_cursor == 4  # History survives W0 close.
    with h5py.File(case.output, "r") as handle:
        assert handle["states"].shape == (4, 1519)
        assert "observations" not in handle["transitions"]
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == sha for p, sha in before.items())


def test_synthetic_requires_explicit_opt_in(tmp_path):
    case = prepare_admission(tmp_path)
    with pytest.raises(ValueError, match="explicit opt-in"):
        _admit(case)
    _no_publication(case)


@pytest.mark.parametrize("change,match", [
    ("config_bytes", "configuration byte binding"),
    ("marker_time", "identity or time"),
    ("source_scope", "source scope"),
    ("journal_source_scope", "synthetic data cannot enter formal training"),
    ("window_raw_reward", "reward formula"),
    ("actor_fingerprint", "fingerprint"),
    ("source_payload_bytes", "integrity failure|payload bytes"),
])
def test_changed_evidence_is_rejected_before_publication(tmp_path, change, match):
    case = prepare_admission(tmp_path)
    if change == "config_bytes":
        case.config.write_bytes(case.config.read_bytes() + b" ")
    elif change == "marker_time":
        events = list(case.events)
        events[4] = replace(events[4], time=21.)
        case.journal.write_bytes(synthetic_journal_bytes(case.header, events))
    elif change == "source_scope":
        path = case.raw / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["source_kind"] = "real_anylogic_rpc"
        _json(path, manifest)
    elif change == "journal_source_scope":
        # Adversarially re-seal a synthetic fixture while claiming an actual
        # producer. The fixture writer deliberately cannot create this claim.
        payload = case.journal.read_bytes()
        before, after = b"synthetic", b"actual_anylogic"
        body = payload[:-49].replace(struct.pack(">i", len(before)) + before,
            struct.pack(">i", len(after)) + after, 1)
        footer = payload[-49:-32]
        case.journal.write_bytes(body + footer + hashlib.sha256(body + footer).digest())
    elif change == "window_raw_reward":
        events = list(case.events)
        events[6] = replace(events[6], grade_reward_raw=0.)
        case.journal.write_bytes(synthetic_journal_bytes(case.header, events))
    elif change == "actor_fingerprint":
        artifact = torch.load(case.actor, weights_only=False)
        artifact["actor_digest"] = "0" * 64
        torch.save(artifact, case.actor)
    else:
        with (case.raw / "samples.h5").open("ab") as stream: stream.write(b"tampered")
    with pytest.raises((ValueError, RuntimeError), match=match):
        _admit(case, allow_synthetic=True)
    _no_publication(case)


@pytest.mark.parametrize("options,match", [
    ({"old_probability_delta": -.1}, "old probabilities"),
    ({"q1": 21.}, "window.*Q|window.*tons|window.*state|processed|material.*state"),
    ({"fifo1": 21.}, "FIFO|fifo"),
    ({"rpc_windows": (31., 9., 0., 0.)}, "window tons"),
])
def test_internally_sealed_but_unjoined_rpc_evidence_is_rejected(tmp_path, options, match):
    # These sources pass the inherited EpisodeBatchBuffer seal; only the new
    # Actor/material join can detect their incompatible physical/query evidence.
    case = prepare_admission(tmp_path, **options)
    with pytest.raises((ValueError, RuntimeError), match=match):
        _admit(case, allow_synthetic=True)
    _no_publication(case)
