"""Phase D: online constrained PPO over real eight-episode batches.

Each round collects eight complete AnyLogic episodes, freezes actor/critics for the
duration of the round, computes the four-channel GAE over exactly those on-policy
transitions, performs one PPO pass, and checkpoints. Nothing from an earlier round
is reused. B3 updates the actor with the multipliers that existed *before* the
batch and then updates those multipliers from the batch's own eight episode costs.

The legacy `python.train` entry and `episode_budget_ppo.py` are untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from .collect_constrained import ConstrainedBridge, load_seeds
from .constrained_adapter import AdapterConfiguration, require
from .constrained_batch import BATCH_EPISODES, EpisodeBatchBuffer, FrozenActor, collect_batch, load_training_batch
from .constrained_batch import write_json
from .constrained_ppo import (CHECKPOINT_FORMAT, COST_NAMES, ConstrainedPPO, ConstrainedPPOConfig,
                              TrainingBatch)
from .dispatch_env import DispatchEnv
from .dispatch_service import create_server
from .episode_runner import ParameterVariationEpisodeRunner
from .masked_policy import DispatchMaskedPolicy

TRAIN_ENTRY_CONTRACT = "dispatch-constrained-online-train-v1"
PILOT_CONFIG_CONTRACT = "dispatch-constrained-pilot-v1"
FROZEN_KEYS = ("mode", "batches", "ppo", "constraints")


def _now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


class RpcBatchCollector:
    """One batch = eight complete episodes served to a real AnyLogic experiment."""

    source_kind = "real_anylogic_rpc"

    def __init__(self, host="127.0.0.1", port=50051, callback_timeout=900.0):
        require(callback_timeout > 0, "callback timeout must be positive")
        self.host, self.port, self.callback_timeout = host, port, callback_timeout

    @contextmanager
    def service(self, bridge):
        server = create_server(bridge, host=self.host, port=self.port)
        server.start()
        print(f"[batch] listening on {self.host}:{self.port}; waiting for exactly "
              f"{BATCH_EPISODES} episodes", flush=True)
        try:
            yield None
        finally:
            server.stop(grace=2).wait()

    def collect(self, actor, expected, output_dir):
        buffer = EpisodeBatchBuffer(output_dir)
        bridge = ConstrainedBridge(output_dir, expected)
        env = DispatchEnv(bridge, ParameterVariationEpisodeRunner(BATCH_EPISODES), self.callback_timeout)
        try:
            with self.service(bridge) as hook:
                collect_batch(env, actor, buffer, expected, source_kind=self.source_kind)
                if hook is not None:
                    hook.result(timeout=self.callback_timeout)
        finally:
            env.close()
        rows, episodes = load_training_batch(output_dir)
        return rows, episodes, bridge


def _batch_identity(rows):
    components = rows[0]["reward_components"]
    config = AdapterConfiguration.from_rpc(components)
    from .scenario_contract import from_components, metadata
    return dict(configuration_fingerprint=config.fingerprint,
                scenario=metadata(from_components(components)))


def batch_metrics(rows, episodes, update, multiplier_update, coefficients_before, coefficients_after):
    """Every metric required for a training round, computed from the stored batch."""
    costs = {name: [float(e["totals"][f"{name}_cost"]) for e in episodes] for name in COST_NAMES}
    joint = [all(cost == 0.0 for cost in (e["totals"]["grade_cost"], e["totals"]["ore_cost"],
                                          e["totals"]["waste_cost"])) for e in episodes]
    physics = [e["terminal_physical"] for e in episodes]
    return dict(
        transitions=len(rows), episodes=len(episodes),
        valid_samples=sum(1 for r in rows if r.get("sample_weight") == 1 / BATCH_EPISODES),
        policy_version=rows[0]["policy_version"],
        update_index=update["update_index"], mode=update["mode"],
        mean_episode_cost={name: float(np.mean(costs[name])) for name in COST_NAMES},
        episode_cost_totals={name: costs[name] for name in COST_NAMES},
        joint_success_rate=float(np.mean(joint)),
        mean_wait_vehicle_seconds=float(np.mean([float(p.get("wait_vehicle_seconds", 0.0)) for p in physics])),
        mean_ore_tons_shift=float(np.mean([float(p.get("ore_tons_shift", 0.0)) for p in physics])),
        mean_waste_tons_shift=float(np.mean([float(p.get("waste_tons_shift", 0.0)) for p in physics])),
        critic_value_loss=update["value_losses"],
        critic_explained_variance=update["explained_variance"],
        approx_kl=update["mean"]["approx_kl"], clip_fraction=update["mean"]["clip_fraction"],
        entropy=update["mean"]["entropy"], actor_grad_norm=update["mean"]["actor_grad_norm"],
        critic_grad_norm=update["mean"]["critic_grad_norm"],
        policy_loss=update["mean"]["policy_loss"], critic_loss=update["mean"]["critic_loss"],
        coefficient_mode=update["mode"],
        coefficients_used=list(coefficients_before), coefficients_after=list(coefficients_after),
        multiplier_update=multiplier_update,
        actor_digest_before=update["actor_digest_before"], actor_digest_after=update["actor_digest_after"],
        critic_digests_after=update["critic_digests_after"],
        critic_changed=update["critics_changed"], actor_changed=update["actor_changed"])


class OnlineTrainer:
    """Sequential 8-episode batches with strict on-policy consumption."""

    def __init__(self, policy, config, output_dir, collector, seed_manifests, resume=None,
                 frozen_config=None):
        self.output_dir = Path(output_dir)
        self.collector = collector
        self.seed_manifests = [Path(p) for p in seed_manifests]
        self.config = config
        self.frozen_config = frozen_config
        self.trainer = ConstrainedPPO(policy, config)
        self.consumed = []
        self.batch_index = 0
        self.expected_fingerprint = None
        self.expected_scenario = None
        if resume is not None:
            self._load_state(Path(resume))

    # ------------------------------------------------------------------ state
    def _load_state(self, path):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        require(payload.get("format") == CHECKPOINT_FORMAT, "unknown checkpoint format")
        state = payload.get("online_trainer")
        require(state is not None, "checkpoint has no online trainer state")
        require(state["output_dir"] == str(self.output_dir),
                f"resume requires the original output directory {state['output_dir']}")
        require(state["config"] == self.config.to_dict(), "resume configuration differs from checkpoint")
        self.trainer = ConstrainedPPO.load_checkpoint(path, self.trainer.policy)
        for item in state["consumed_batches"]:
            batch_dir = Path(item["batch_dir"])
            require(batch_dir.exists(), f"consumed batch directory missing: {batch_dir}")
        self.consumed = list(state["consumed_batches"])
        self.batch_index = int(state["batch_index"])
        self.expected_fingerprint = state.get("expected_configuration_fingerprint")
        self.expected_scenario = state.get("expected_scenario")
        return state

    def _state(self):
        return dict(entry_contract=TRAIN_ENTRY_CONTRACT, output_dir=str(self.output_dir),
                    config=self.config.to_dict(), frozen_config=self.frozen_config,
                    consumed_batches=getattr(self, "consumed", []),
                    batch_index=getattr(self, "batch_index", 0),
                    expected_configuration_fingerprint=getattr(self, "expected_fingerprint", None),
                    expected_scenario=getattr(self, "expected_scenario", None),
                    seeds=[str(p) for p in self.seed_manifests])

    def save(self):
        path = self.output_dir / "checkpoints" / f"step_{getattr(self, 'batch_index', 0):04d}.pt"
        return self.trainer.save_checkpoint(path, extra=dict(online_trainer=self._state()))

    # ------------------------------------------------------------------- loop
    def run(self, batches):
        consumed_dirs = {item["batch_dir"] for item in self.consumed}
        history = []

        for offset in range(batches):
            index = self.batch_index + 1
            manifest = self.seed_manifests[min(offset, len(self.seed_manifests) - 1)]
            expected = load_seeds(manifest)
            batch_dir = self.output_dir / f"batch_{index:04d}"
            require(not batch_dir.exists(), f"batch directory already exists: {batch_dir}")

            # Freeze an independent copy: the trainer's policy keeps the update graph.
            actor = FrozenActor(clone_policy(self.trainer.policy))
            require(actor.digest == self.trainer.actor_digest(), "actor changed before collection")
            critics_before = self.trainer.critic_digests()
            coefficients_before = self.trainer.coefficients.tolist()
            lambdas_before = self.trainer.lambdas.tolist()

            rows, episodes, _bridge = self.collector.collect(actor, expected, batch_dir)

            require(self.trainer.actor_digest() == actor.digest, "actor was updated during collection")
            require(self.trainer.critic_digests() == critics_before, "critics were updated during collection")
            batch = TrainingBatch.from_rows(rows, episodes)
            require(batch.policy_version == actor.policy_version, "stored batch is not on-policy")

            identity = _batch_identity(rows)
            if self.expected_fingerprint is None:
                self.expected_fingerprint = identity["configuration_fingerprint"]
                self.expected_scenario = identity["scenario"]
            require(identity["configuration_fingerprint"] == self.expected_fingerprint,
                    "RPC reward configuration changed between batches")
            require(identity["scenario"] == self.expected_scenario, "RPC scenario changed between batches")
            require(str(batch_dir) not in consumed_dirs,
                    "this batch was already consumed; refusing to reuse it for another update")

            update = self.trainer.update(batch)
            multiplier = (self.trainer.apply_multiplier_update(batch)
                          if self.config.mode == "lagrangian" else None)
            metrics = batch_metrics(rows, episodes, update, multiplier,
                                    coefficients_before, self.trainer.coefficients.tolist())
            metrics.update(batch_index=index, collected_at=_now(),
                           batch_dir=str(batch_dir.relative_to(self.output_dir)),
                           seed_manifest=str(manifest), seeds=batch.provenance["seeds"],
                           configuration_fingerprint=identity["configuration_fingerprint"],
                           scenario=identity["scenario"], lambdas_before=lambdas_before,
                           lambdas_after=self.trainer.lambdas.tolist(),
                           preprocessing=actor.preprocessing, inference_mode="eval/cpu")
            self.consumed.append(dict(batch_index=index, batch_dir=str(batch_dir),
                                      policy_version=batch.policy_version, transitions=batch.size))
            consumed_dirs.add(str(batch_dir))
            self.batch_index = index
            checkpoint = self.save()
            metrics["checkpoint"] = checkpoint["path"]
            with (self.output_dir / "batches.jsonl").open("a") as stream:
                stream.write(json.dumps(metrics, ensure_ascii=False, allow_nan=False) + "\n")
            history.append(metrics)
            print(json.dumps(dict(batch=index, transitions=metrics["transitions"],
                                  policy_version=metrics["policy_version"][:16],
                                  joint_success_rate=metrics["joint_success_rate"],
                                  mean_cost=metrics["mean_episode_cost"],
                                  approx_kl=metrics["approx_kl"],
                                  lambdas=metrics["lambdas_after"]), ensure_ascii=False), flush=True)

        manifest = dict(entry_contract=TRAIN_ENTRY_CONTRACT, output_dir=str(self.output_dir),
                        mode=self.config.mode, config=self.config.to_dict(),
                        frozen_config=self.frozen_config,
                        batches=history, consumed_batches=self.consumed,
                        seed_manifests=[str(p) for p in self.seed_manifests],
                        expected_configuration_fingerprint=self.expected_fingerprint,
                        expected_scenario=self.expected_scenario,
                        final_policy_version=self.trainer.actor_digest(),
                        final_lambdas=self.trainer.lambdas.tolist(),
                        update_index=self.trainer.update_index, finished_at=_now())
        write_json(self.output_dir / "training_manifest.json", manifest)
        return manifest


def load_frozen_config(path):
    """Read the approved Pilot file; it is the single source of truth during a run."""
    path = Path(path)
    payload = json.loads(path.read_text())
    require(payload.get("contract") == PILOT_CONFIG_CONTRACT,
            f"unexpected pilot config contract: {payload.get('contract')!r}")
    require(payload.get("frozen") is True, "pilot config must be marked frozen")
    ppo, constraints = payload["ppo"], payload["constraints"]
    limits = constraints["cost_limits"]
    require(list(limits) == list(constraints["cost_names"]), "cost limit order must match cost names")
    config = ConstrainedPPOConfig(
        mode=payload["mode"], gamma=float(ppo["gamma"]), gae_lambda=float(ppo["gae_lambda"]),
        clip_range=float(ppo["clip_range"]), n_epochs=int(ppo["n_epochs"]),
        minibatch_size=int(ppo["minibatch_size"]),
        actor_learning_rate=float(ppo["actor_learning_rate"]),
        critic_learning_rate=float(ppo["critic_learning_rate"]),
        ent_coef=float(ppo["ent_coef"]), vf_coef=float(ppo["vf_coef"]),
        max_grad_norm=float(ppo["max_grad_norm"]), hidden_sizes=tuple(ppo["hidden_sizes"]),
        seed=int(ppo["seed"]), cost_weights=tuple(float(v) for v in constraints["cost_weights"]),
        cost_limits=(float(limits["grade"]), float(limits["ore"]), float(limits["waste"])),
        lambda_lr=float(constraints["lambda_lr"]), lambda_max=float(constraints["lambda_max"]))
    return payload, config


def clone_policy(policy):
    """Independent copy for sampling: avoids deep-copying tensors left by the update."""
    clone = DispatchMaskedPolicy(
        policy.observation_space, policy.action_space, lambda _: 3e-4,
        net_arch=policy.net_arch, activation_fn=policy.activation_fn, ortho_init=policy.ortho_init,
        use_sde=policy.use_sde, log_std_init=policy.log_std_init,
        share_features_extractor=policy.share_features_extractor,
        features_extractor_class=policy.features_extractor_class,
        features_extractor_kwargs=policy.features_extractor_kwargs)
    clone.load_state_dict(policy.state_dict())
    return clone


def build_policy(net_arch=(128, 128), seed=11):
    import gymnasium as gym
    from .observation_contract import OBSERVATION_SIZE
    observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(OBSERVATION_SIZE,), dtype=np.float32)
    return DispatchMaskedPolicy(observation_space, gym.spaces.Discrete(4), lambda _: 3e-4,
                                net_arch=dict(pi=list(net_arch), vf=list(net_arch)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed-manifest", type=Path, action="append", required=True,
                        help="one manifest per batch; the last one repeats if fewer than --batches")
    parser.add_argument("--config-json", type=Path,
                        help="frozen Pilot configuration; disables all parameter overrides")
    parser.add_argument("--batches", type=int)
    parser.add_argument("--mode", choices=("fixed_weights", "lagrangian"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--callback-timeout", type=float, default=900.0)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--minibatch-size", type=int)
    parser.add_argument("--actor-lr", type=float)
    parser.add_argument("--critic-lr", type=float)
    parser.add_argument("--clip-range", type=float)
    parser.add_argument("--ent-coef", type=float)
    parser.add_argument("--vf-coef", type=float)
    parser.add_argument("--cost-weights")
    parser.add_argument("--cost-limits")
    parser.add_argument("--lambda-lr", type=float)
    parser.add_argument("--lambda-max", type=float)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)

    overrides = {name: getattr(args, name) for name in
                 ("mode", "epochs", "minibatch_size", "actor_lr", "critic_lr", "clip_range", "ent_coef",
                  "vf_coef", "cost_weights", "cost_limits", "lambda_lr", "lambda_max", "seed")
                 if getattr(args, name) is not None}
    frozen = None
    if args.config_json is not None:
        require(not overrides, f"frozen Pilot config forbids overrides: {sorted(overrides)}")
        frozen_payload, config = load_frozen_config(args.config_json)
        frozen = dict(path=str(args.config_json),
                      sha256=hashlib.sha256(Path(args.config_json).read_bytes()).hexdigest(),
                      contract=frozen_payload["contract"], approved=frozen_payload.get("approved"))
        batches = args.batches if args.batches is not None else int(frozen_payload["batches"])
    else:
        config = ConstrainedPPOConfig(
            mode=args.mode or "fixed_weights", n_epochs=args.epochs or 4,
            minibatch_size=args.minibatch_size or 128,
            actor_learning_rate=args.actor_lr or 3e-4, critic_learning_rate=args.critic_lr or 1e-3,
            clip_range=args.clip_range or 0.2, ent_coef=args.ent_coef if args.ent_coef is not None else 0.0,
            vf_coef=args.vf_coef if args.vf_coef is not None else 0.5,
            cost_weights=tuple(float(v) for v in (args.cost_weights or "1,1,1").split(",")),
            cost_limits=tuple(float(v) for v in (args.cost_limits or "0,0,0").split(",")),
            lambda_lr=args.lambda_lr if args.lambda_lr is not None else 0.05,
            lambda_max=args.lambda_max if args.lambda_max is not None else 10.0,
            seed=args.seed or 11)
        batches = args.batches if args.batches is not None else 1

    torch.set_num_threads(1)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    if args.resume is not None:
        require(args.output_dir.exists(), "resume output directory must exist")
    else:
        require(not args.output_dir.exists(), "output directory must not exist; use a new one per run")
        args.output_dir.mkdir(parents=True)

    policy = build_policy(seed=config.seed)
    collector = RpcBatchCollector(args.host, args.port, args.callback_timeout)
    trainer = OnlineTrainer(policy, config, args.output_dir, collector, args.seed_manifest, args.resume,
                            frozen_config=frozen)
    write_json(args.output_dir / "entry.json",
               dict(entry_contract=TRAIN_ENTRY_CONTRACT, started_at=_now(), mode=config.mode,
                    config=config.to_dict(), frozen_config=frozen, batches=batches,
                    seed_manifests=[str(p) for p in args.seed_manifest],
                    resumed_from=str(args.resume) if args.resume else None,
                    source_kind=collector.source_kind))
    manifest = trainer.run(batches)
    print(json.dumps(dict(finished=True, batches=len(manifest["batches"]),
                          final_lambdas=manifest["final_lambdas"]), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
