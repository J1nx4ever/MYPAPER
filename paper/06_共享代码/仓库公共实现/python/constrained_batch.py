"""V1-B: eight complete episodes, immutable inference, storage-only chunks."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.torch_layers import FlattenExtractor

from .constrained_adapter import (ADAPTER_CONTRACT_ID, WEIGHT_CONTRACT_ID,
                                  AdapterConfiguration, RewardCostAdapter, require)
from .masked_policy import DispatchMaskedPolicy
from .observation_contract import OBSERVATION_SCHEMA_ID

BATCH_EPISODES = 8
PREPROCESSING_CONTRACT = "raw-physical-v3-float32-no-running-normalizer-v1"


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def policy_digest(policy):
    digest = hashlib.sha256()
    digest.update(repr(policy).encode())
    digest.update(str(policy.normalize_images).encode())
    for key, value in sorted(policy.state_dict().items()):
        a = value.detach().cpu().contiguous().numpy()
        digest.update(key.encode())
        digest.update(str((a.dtype, a.shape)).encode())
        digest.update(a.tobytes())
    return digest.hexdigest()


class FrozenActor:
    """Owned CPU snapshot of the audited policy; no online transforms accepted.

    Both actor and legacy critic parameters/buffers are copied and frozen.
    No new critic is constructed and the existing value head is never optimized.
    """
    def __init__(self, policy):
        require(type(policy) is DispatchMaskedPolicy, "only audited DispatchMaskedPolicy is supported")
        require(policy.observation_space.shape == (1519,), "V1.1 currently requires raw 1519D policy")
        require(not policy.use_sde, "state-dependent exploration is not supported")
        require(all(type(extractor) is FlattenExtractor for extractor in
                    (policy.features_extractor, policy.pi_features_extractor, policy.vf_features_extractor)),
                "V1.1 requires the audited stateless FlattenExtractor preprocessing")
        self.policy = copy.deepcopy(policy).to("cpu")
        self.policy.set_training_mode(False)
        for parameter in self.policy.parameters():
            parameter.requires_grad_(False)
        self.digest = policy_digest(self.policy)
        self.preprocessing = dict(contract=PREPROCESSING_CONTRACT, dtype="float32",
                                  running_normalization=False, inference_mode="eval/cpu")
        self.source_hashes = {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("masked_policy.py", "observation_schema.json", "observation_contract.py",
                         "constrained_batch.py")}
        identity = dict(parameters=self.digest, preprocessing=self.preprocessing, sources=self.source_hashes)
        self.policy_version = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()

    def assert_frozen(self):
        require(not any(m.training for m in self.policy.modules()), "policy inference mode changed")
        require(not any(p.requires_grad for p in self.policy.parameters()), "policy gradient flag changed")
        require(policy_digest(self.policy) == self.digest, "policy parameters/buffers/preprocessing changed")

    def distribution(self, observation, mask):
        self.assert_frozen()
        obs = np.asarray(observation, dtype=np.float32)
        mask = np.asarray(mask)
        require(obs.shape == (1519,) and np.isfinite(obs).all(), "invalid actor observation")
        require(mask.shape == (4,) and np.isin(mask, (0, 1)).all() and mask.any(), "invalid sampling mask")
        require(np.array_equal(obs[19:23], mask), "sampled mask differs from observation")
        with torch.inference_mode():
            distribution = self.policy.get_distribution(torch.from_numpy(obs.copy()).unsqueeze(0))
        return distribution

    def probabilities(self, observation, mask):
        return self.distribution(observation, mask).distribution.probs[0].cpu().numpy().copy()

    def sample(self, observation, mask):
        with torch.inference_mode():
            distribution = self.distribution(observation, mask)
            action = distribution.get_actions(deterministic=False)
            log_prob = distribution.log_prob(action)
            probs = distribution.distribution.probs[0].cpu().numpy().copy()
        return int(action.item()), float(log_prob.item()), probs.tolist()


class EpisodeBatchBuffer:
    def __init__(self, output_dir, *, chunk_size=128, record_transitions=False):
        require(chunk_size > 0, "invalid storage chunk size")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.chunk_size = chunk_size
        self.record_transitions = record_transitions
        self.rows = []
        self.episodes = []
        self.flushed = 0
        self.chunks = 0
        self.complete = False

    def append(self, row):
        require(not self.complete, "batch is sealed")
        self.rows.append(row)
        if self.record_transitions:
            with (self.output_dir / "transitions.jsonl").open("a") as stream:
                stream.write(json.dumps(self._metadata(row), allow_nan=False) + "\n")
        if len(self.rows) - self.flushed >= self.chunk_size:
            self.flush()

    @staticmethod
    def _metadata(row):
        return {k: v for k, v in row.items() if k not in ("observation", "next_observation")}

    def flush(self):
        rows = self.rows[self.flushed:]
        if not rows:
            return
        # Required training storage, not optional full-observation debug logs.
        np.savez_compressed(self.output_dir / f"samples_{self.chunks:04d}.npz",
                            observations=np.stack([r["observation"] for r in rows]),
                            next_observations=np.stack([r["next_observation"] for r in rows]),
                            metadata_json=np.array([json.dumps(self._metadata(r), allow_nan=False) for r in rows]))
        self.flushed = len(self.rows)
        self.chunks += 1

    def finish_episode(self, start, info):
        rows = self.rows[start:]
        require(rows and rows[-1]["terminated"] and not rows[-1]["truncated"], "incomplete episode")
        require(all(r["episode_id"] == info["episode_id"] for r in rows), "mixed episode rows")
        summary = dict(episode_id=info["episode_id"], experiment_seed=info["experiment_seed"],
                       experiment_run_index=info["experiment_run_index"], start=start, stop=len(self.rows),
                       transitions=len(rows), normal_completion=True, sim_time_seconds=info["sim_time_seconds"],
                       policy_version=rows[0]["policy_version"],
                       totals={k: sum(r[k] for r in rows) for k in ("reward", "grade_cost", "ore_cost", "waste_cost")},
                       terminal_physical=dict(info["reward_components"]))
        summary["business_constraints_met"] = all(summary["totals"][k] == 0 for k in ("grade_cost", "ore_cost", "waste_cost"))
        self.episodes.append(summary)
        with (self.output_dir / "episodes.jsonl").open("a") as stream:
            stream.write(json.dumps(summary, allow_nan=False) + "\n")

    def seal(self):
        require(len(self.episodes) == BATCH_EPISODES, "batch requires exactly eight normal episodes")
        require(sum(e["transitions"] for e in self.episodes) == len(self.rows), "unassigned transition")
        require(len({r["policy_version"] for r in self.rows}) == 1, "mixed policy versions")
        self.flush()
        self.complete = True

    def training_rows(self):
        require(self.complete, "aborted/incomplete batch is not trainable")
        return iter(self.rows)  # no drop_last, padding, truncation, or length normalization

    def episode_returns(self):
        """Undiscounted Monte Carlo targets only; no critic or GAE in Phase A/B."""
        require(self.complete, "returns require a complete valid batch")
        returns = np.zeros((len(self.rows), 4), dtype=np.float64)
        for episode in self.episodes:
            carry = np.zeros(4)
            for i in range(episode["stop"] - 1, episode["start"] - 1, -1):
                carry += [self.rows[i][k] for k in ("reward", "grade_cost", "ore_cost", "waste_cost")]
                returns[i] = carry
        return returns


def collect_batch(env, actor, buffer, expected_metadata, *, source_kind, checkpoint_metadata=None):
    require(len(expected_metadata) == BATCH_EPISODES, "seed manifest must contain exactly eight episodes")
    manifest = dict(status="collecting", source_kind=source_kind, configuration_source="rpc_reward_components",
                    adapter_contract_id=ADAPTER_CONTRACT_ID,
                    weight_contract_id=WEIGHT_CONTRACT_ID, sample_weight=1 / BATCH_EPISODES,
                    objective="(1/8) * sum_episode sum_transition; never divide by episode length",
                    policy_version=actor.policy_version, policy_parameters_sha256=actor.digest,
                    preprocessing=actor.preprocessing, source_hashes=actor.source_hashes,
                    seed_manifest=[dict(run_index=i, experiment_seed=s) for i, s in expected_metadata],
                    optimizer_steps=0, gae_computed=False, observation_schema_id=OBSERVATION_SCHEMA_ID)
    write_json(buffer.output_dir / "manifest.json", manifest)
    actor.policy.save(buffer.output_dir / "frozen_policy.pth")
    adapter = None
    context = {}
    try:
        for index, expected in enumerate(expected_metadata):
            context = dict(expected_run_index=expected[0], expected_seed=expected[1], episode_index=index)
            observation, info = env.reset()
            context.update(episode_id=info.get("episode_id"), actual_seed=info.get("experiment_seed"))
            require((info["experiment_run_index"], info["experiment_seed"]) == tuple(expected), "seed manifest mismatch")
            config = AdapterConfiguration.from_rpc(info["reward_components"])
            if adapter is None:
                if checkpoint_metadata is not None:
                    from .reward_config import configuration_from_components
                    from .scenario_contract import from_components, metadata
                    expected_scenario = metadata(from_components(info["reward_components"]))
                    for item in checkpoint_metadata:
                        require(item["reward_configuration_fingerprint"] ==
                                configuration_from_components(info["reward_components"]).fingerprint,
                                "checkpoint reward configuration differs from RPC")
                        require(all(item.get(k) == v for k, v in expected_scenario.items()),
                                "checkpoint scenario differs from RPC or lacks scenario identity")
                adapter = RewardCostAdapter(config)
                manifest.update(runtime_configuration=dict(config.entries), configuration_fingerprint=config.fingerprint,
                                original_reward_contract_id=info["reward_contract_id"])
                write_json(buffer.output_dir / "manifest.json", manifest)
            adapter.start(info)
            start = len(buffer.rows)
            while True:
                mask = list(info["action_mask"])
                action, old_log_prob, probs = actor.sample(observation, mask)
                context.update(action_request_id=info["request_id"], action=action,
                               decision_sequence=info["decision_sequence"])
                next_obs, raw_reward, terminated, truncated, next_info = env.step(action)
                context["settlement_request_id"] = next_info["request_id"]
                context["settlement"] = next_info
                require(next_info["previous_action"] == action, "settlement action mismatch")
                adapted = adapter.transition(next_info, terminated=terminated, truncated=truncated)
                actor.assert_frozen()
                row = dict(**adapted, observation=np.asarray(observation).copy(),
                           next_observation=np.asarray(next_obs).copy(), action=action, mask=mask,
                           old_log_prob=old_log_prob, action_probabilities=probs, policy_version=actor.policy_version,
                           sample_weight=1 / BATCH_EPISODES, weight_contract_id=WEIGHT_CONTRACT_ID,
                           episode_id=info["episode_id"], experiment_seed=info["experiment_seed"],
                           experiment_run_index=info["experiment_run_index"], action_request_id=info["request_id"],
                           settlement_request_id=next_info["request_id"], decision_sequence=next_info["decision_sequence"],
                           sim_time_seconds=next_info["sim_time_seconds"], sim_time_delta_seconds=next_info["sim_time_delta_seconds"],
                           terminated=bool(terminated), truncated=bool(truncated), episode_start=len(buffer.rows) == start,
                           transition_reward_raw=float(raw_reward), reward_contract_id=next_info["reward_contract_id"],
                           reward_components=dict(next_info["reward_components"]), closed_windows=next_info["closed_windows"])
                buffer.append(row)
                observation, info = next_obs, next_info
                if terminated:
                    buffer.finish_episode(start, info)
                    break
            print(f"Collected {index + 1}/8 episodes; seed={expected[1]}; transitions={len(buffer.rows)-start}", flush=True)
        actor.assert_frozen()
        buffer.seal()
        np.save(buffer.output_dir / "episode_returns.npy", buffer.episode_returns())
        manifest.update(status="complete", completed_episodes=8, transitions=len(buffer.rows),
                        storage_chunks=buffer.chunks, retained_transitions=len(list(buffer.training_rows())),
                        final_policy_parameters_sha256=policy_digest(actor.policy),
                        business_failed_episodes=sum(not e["business_constraints_met"] for e in buffer.episodes))
        write_json(buffer.output_dir / "manifest.json", manifest)
        return buffer
    except BaseException as exc:
        buffer.complete = False
        buffer.flush()
        manifest.update(status="aborted", completed_episodes=len(buffer.episodes), transitions=len(buffer.rows),
                        trainable=False, error_type=type(exc).__name__, error=str(exc), context=context)
        write_json(buffer.output_dir / "manifest.json", manifest)
        write_json(buffer.output_dir / "ABORTED.json", manifest)
        raise


def load_training_batch(output_dir):
    """Read every persisted row in a sealed batch; aborted prefixes are rejected."""
    path = Path(output_dir)
    manifest = json.loads((path / "manifest.json").read_text())
    require(manifest["status"] == "complete" and not (path / "ABORTED.json").exists(),
            "aborted/incomplete batch is not trainable")
    require(manifest["completed_episodes"] == BATCH_EPISODES, "incomplete episode batch")
    rows = []
    for i in range(manifest["storage_chunks"]):
        with np.load(path / f"samples_{i:04d}.npz", allow_pickle=False) as block:
            require(len(block["observations"]) == len(block["next_observations"]) == len(block["metadata_json"]),
                    "sample block length mismatch")
            for obs, next_obs, raw in zip(block["observations"], block["next_observations"], block["metadata_json"]):
                row = json.loads(str(raw))
                require(row["policy_version"] == manifest["policy_version"] and row["sample_weight"] == 1 / BATCH_EPISODES,
                        "policy version/weight mismatch in storage")
                rows.append(dict(row, observation=obs.copy(), next_observation=next_obs.copy()))
    require(len(rows) == manifest["transitions"] == manifest["retained_transitions"], "stored transitions missing")
    episodes = [json.loads(line) for line in (path / "episodes.jsonl").read_text().splitlines()]
    require(len(episodes) == BATCH_EPISODES, "stored episode summaries missing")
    cursor = 0
    for episode in episodes:
        require(episode["start"] == cursor and episode["stop"] > cursor, "episode boundary gap/overlap")
        segment = rows[cursor:episode["stop"]]
        require(all(r["episode_id"] == episode["episode_id"] for r in segment), "mixed episode storage")
        require(segment[-1]["terminated"] and all(not r["truncated"] for r in segment) and
                all(not r["terminated"] for r in segment[:-1]), "invalid terminal storage")
        cursor = episode["stop"]
    require(cursor == len(rows), "unassigned stored transitions")
    return rows, episodes
