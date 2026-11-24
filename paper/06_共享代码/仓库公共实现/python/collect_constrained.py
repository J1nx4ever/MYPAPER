"""Independent V1.1 collection entry. Never calls PPO.learn/train/optimizer.step."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import queue
import threading
import time

from google.protobuf.json_format import MessageToDict
import torch

from .constrained_adapter import AdapterConfiguration, SamplingError, require
from .constrained_batch import EpisodeBatchBuffer, FrozenActor, collect_batch, write_json
from .dispatch_bridge import DispatchCallbackBridge, ProtocolError
from .dispatch_env import DispatchEnv
from .dispatch_service import create_server
from .episode_runner import ParameterVariationEpisodeRunner
from .masked_policy import DispatchMaskedPolicy
from .masked_policy import POLICY_CONTRACT_ID
from .observation_contract import OBSERVATION_SCHEMA_ID
from .reward_contract import REWARD_CONTRACT_ID
from .protocol_artifacts import verify_protocol_artifacts
from .scenario_contract import from_components


class ConstrainedBridge(DispatchCallbackBridge):
    """Keep the legacy bridge untouched; fail the entire new batch on RPC errors."""
    def __init__(self, output_dir, expected_metadata):
        super().__init__(runtime_configuration=True, expected_episode_metadata=expected_metadata)
        self.output_dir = Path(output_dir)
        self.adapter_configuration = None
        self.fatal_error = None
        self._failure_lock = threading.Lock()

    def submit_callback(self, request, timeout=None):
        try:
            if self.fatal_error is not None:
                raise self.fatal_error
            return super().submit_callback(request, timeout=timeout)
        except BaseException as exc:
            self.fail(exc, request)
            raise

    def _register(self, request):
        try:
            candidate = AdapterConfiguration.from_rpc(request.reward_components)
            if self.adapter_configuration is not None:
                require(candidate == self.adapter_configuration, "RPC configuration changed within batch")
            # The scenario is declared on the wire, not inferred from local ALP.
            if self.adapter_configuration is None:
                self.expected_scenario = from_components(request.reward_components)
            ticket = super()._register(request)
            self.adapter_configuration = candidate
            return ticket
        except (ValueError, SamplingError) as exc:
            raise ProtocolError(str(exc)) from exc

    def fail(self, exc, request=None):
        with self._failure_lock:
            if self.fatal_error is None:
                self.fatal_error = SamplingError(f"RPC sampling failure: {type(exc).__name__}: {exc}")
            record = dict(error=str(exc), error_type=type(exc).__name__)
            if request is not None:
                record.update(episode_id=request.episode_id, experiment_seed=request.experiment_seed,
                              experiment_run_index=request.experiment_run_index,
                              request=MessageToDict(request, preserving_proto_field_name=True))
            with (self.output_dir / "rpc_errors.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            self.close()

    def next_callback(self, timeout=None):
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if self.fatal_error is not None:
                raise self.fatal_error
            if self._closed:
                raise SamplingError("bridge closed during collection")
            remaining = 0.1 if deadline is None else min(0.1, max(0, deadline - time.monotonic()))
            try:
                ticket = self._incoming.get(timeout=remaining)
                if self.fatal_error is not None:
                    raise self.fatal_error
                return ticket
            except queue.Empty:
                if deadline is not None and time.monotonic() >= deadline:
                    error = TimeoutError("timed out waiting for AnyLogic; batch aborted, seed retained")
                    self.fail(error)
                    raise error


def load_seeds(path):
    data = json.loads(Path(path).read_text())
    require(isinstance(data, list) and len(data) == 8, "manifest must be a list of exactly eight seeds")
    rows = [(row["run_index"], row["experiment_seed"]) for row in data]
    require(all(type(i) is int and type(s) is int and i > 0 and s > 0 for i, s in rows), "invalid seed identity")
    require(len({i for i, _ in rows}) == len({s for _, s in rows}) == 8, "duplicate run index/seed")
    return rows


def checkpoint_contract(path):
    """Accept completed raw F0/raw1519 runs, retaining scenario/config provenance."""
    path = Path(path)
    manifest, success = [json.loads((path.parent / name).read_text())
                         for name in ("run_manifest.json", "SUCCESS.json")]
    for item in (manifest, success):
        require(item.get("reward_contract_id") == REWARD_CONTRACT_ID and
                item.get("policy_contract_id") == POLICY_CONTRACT_ID, "incompatible checkpoint contract")
        require(item.get("agent_observation_mode", "f0") in ("f0", "raw1519") and
                item.get("agent_observation_dimension", 1519) == 1519,
                "checkpoint must use raw1519 preprocessing")
        require("reward_configuration_fingerprint" in item, "checkpoint configuration missing")
    require(manifest.get("observation_schema_id") == OBSERVATION_SCHEMA_ID, "checkpoint schema differs")
    require(success.get("model_sha256") == hashlib.sha256(path.read_bytes()).hexdigest(), "checkpoint hash mismatch")
    return [manifest, success]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--anylogic-seed-manifest", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--callback-timeout", type=float, default=120)
    parser.add_argument("--seed", type=int, default=11, help="Python initialization/action RNG seed")
    parser.add_argument("--checkpoint", type=Path, help="Existing audited raw1519 PPO model.zip")
    parser.add_argument("--record-transitions", action="store_true", help="Opt in to additional per-transition JSONL")
    args = parser.parse_args(argv)
    require(args.callback_timeout > 0, "callback timeout must be positive")
    expected = load_seeds(args.anylogic_seed_manifest)
    verify_protocol_artifacts()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    buffer = EpisodeBatchBuffer(args.output_dir, record_transitions=args.record_transitions)
    # Preserve the exact original manifest bytes, even when handshake fails.
    (args.output_dir / "seed_manifest.json").write_bytes(args.anylogic_seed_manifest.read_bytes())
    write_json(args.output_dir / "entry.json", dict(source_kind="real_anylogic_rpc", python_seed=args.seed,
                                                  checkpoint=str(args.checkpoint) if args.checkpoint else None))
    bridge = ConstrainedBridge(args.output_dir, expected)
    env = DispatchEnv(bridge, ParameterVariationEpisodeRunner(8), args.callback_timeout)
    server = None
    try:
        checkpoint_metadata = None
        if args.checkpoint:
            from .episode_budget_ppo import EpisodeBudgetPPO
            checkpoint_metadata = checkpoint_contract(args.checkpoint)
            model = EpisodeBudgetPPO.load(args.checkpoint, device="cpu")
            policy = model.policy
        else:
            policy = DispatchMaskedPolicy(env.observation_space, env.action_space, lambda _: 0.0,
                                          net_arch=dict(pi=[128, 128], vf=[128, 128]))
        actor = FrozenActor(policy)
        server = create_server(bridge, host=args.host, port=args.port)
        server.start()
        print(f"V1.1 collection listening on {args.host}:{args.port}; exactly 8 episodes; no updates.", flush=True)
        collect_batch(env, actor, buffer, expected, source_kind="real_anylogic_rpc",
                      checkpoint_metadata=checkpoint_metadata)
    except BaseException as exc:
        if not (args.output_dir / "ABORTED.json").exists():
            write_json(args.output_dir / "ABORTED.json", dict(status="aborted", error=str(exc),
                       error_type=type(exc).__name__, seed_manifest=expected, trainable=False))
        raise
    finally:
        env.close()
        if server is not None:
            server.stop(grace=2).wait()  # allow the final no-action ACK to leave the transport


if __name__ == "__main__":
    main()
