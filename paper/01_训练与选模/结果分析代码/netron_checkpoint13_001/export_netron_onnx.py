from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

EXPERIMENT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from business_graph_summary_v1.contract import feature_config_for
from business_graph_summary_v1.graph import GraphBatch, NUM_NODES
from business_graph_summary_v1.models import ACTOR_SPEC, Actor, TYPE_SLICES, digest
from business_graph_summary_v1.graph import build_graph
from single_critic_task_query_v1.c_features import build_features
from single_critic_task_query_v1.features_cache import SummaryFeatureCache
from relation_gat_ego_v1.contract import require


CAMPAIGN = "business-graph-summary-formal-seed1-20261118-001"
CANDIDATE = "s01_BG128"
CHECKPOINT = EXPERIMENT_ROOT / "runs" / CAMPAIGN / "arms" / CANDIDATE / "updates/0013/checkpoint.pt"
SAMPLE_DIR = EXPERIMENT_ROOT / "runs" / CAMPAIGN / "arms" / CANDIDATE / "train_u0014/episodes"
OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_MODEL = OUTPUT_DIR / "BG128_checkpoint13_actor.onnx"
EXPECTED_CHECKPOINT_SHA256 = "a9acc888413ce0861240c04c396a239d9c26e48290eddc9f2c14dd9c97e6afd5"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_states() -> tuple[torch.Tensor, torch.Tensor]:
    candidates = []
    for path in sorted(SAMPLE_DIR.glob("run_*/samples.h5")):
        with h5py.File(path, "r") as stream:
            data = np.asarray(stream["states"][:], dtype=np.float32)
        if data.ndim != 2 or data.shape[1] != 1519:
            continue
        for row in data:
            if row[17] != 0 or not np.isfinite(row).all() or not np.any(row[19:23]):
                continue
            state = torch.from_numpy(row.copy()).reshape(1, -1)
            graph = build_graph(state)
            edge_count = graph.edge_index.shape[1]
            if not candidates or edge_count != candidates[0][0]:
                candidates.append((edge_count, state))
                if len(candidates) == 2:
                    return candidates[0][1], candidates[1][1]
    raise RuntimeError("Could not find two valid observations with different graph sizes.")


class ExportGraphLayer(nn.Module):
    """Same learned layer, expressed with dense masked softmax for ONNX."""

    def __init__(self, layer: nn.Module):
        super().__init__()
        self.score_src = layer.score_src
        self.score_dst = layer.score_dst
        self.score_edge = layer.score_edge
        self.score_a = layer.score_a
        self.value = layer.value
        self.edge_value = layer.edge_value
        self.output = layer.output
        self.norm = layer.norm

    def forward(self, h: torch.Tensor, graph: GraphBatch) -> torch.Tensor:
        batch, _, width = h.shape
        edge_index = graph.edge_index
        source, destination = edge_index
        edge_batch = source // NUM_NODES
        destination_local = destination % NUM_NODES
        flat = h.reshape(-1, width)
        edge_input = torch.cat(
            (
                torch.nn.functional.one_hot(graph.edge_type, 25).to(h.dtype),
                graph.edge_attr,
            ),
            dim=-1,
        )
        score_input = (
            self.score_src(flat)[source]
            + self.score_dst(flat)[destination]
            + self.score_edge(edge_input)
        )
        scores = (
            torch.nn.functional.leaky_relu(score_input, 0.2)
            .reshape(-1, 4, 16)
            .mul(self.score_a)
            .sum(-1)
        )
        messages = torch.nn.functional.leaky_relu(
            self.value(flat)[source] + self.edge_value(edge_input), 0.2
        ).reshape(-1, 4, 16)

        sample_ids = torch.arange(batch, device=h.device)
        node_ids = torch.arange(NUM_NODES, device=h.device)
        group_outputs = []
        group_presence = []
        for group_id in range(4):
            mask = (
                (edge_batch[None, None, :] == sample_ids[:, None, None])
                & (destination_local[None, None, :] == node_ids[None, :, None])
                & (graph.edge_group[None, None, :] == group_id)
            )
            masked_scores = torch.where(
                mask[..., None], scores[None, None, :, :], scores.new_full((), -1e30)
            )
            weights = torch.softmax(masked_scores, dim=2)
            aggregate = (
                weights[..., None] * messages[None, None, :, :, :]
            ).sum(dim=2)
            present = mask.any(dim=2)
            aggregate = aggregate * present[..., None, None]
            group_outputs.append(aggregate.reshape(batch, NUM_NODES, 64))
            group_presence.append(present)

        aggregate = torch.stack(group_outputs, dim=2)
        presence = torch.stack(group_presence, dim=2)
        fused = torch.cat(
            (aggregate.reshape(batch * NUM_NODES, 4 * 64),
             presence.reshape(batch * NUM_NODES, 4).to(h.dtype)),
            dim=-1,
        )
        projected = self.output(fused)
        residual = flat + projected
        physical = self.norm(residual.reshape(batch, NUM_NODES, 64)[:, :18])
        return torch.cat((physical, h[:, 18:19]), dim=1)


class NetronActor(nn.Module):
    """Actor graph with graph construction supplied as explicit tensor inputs."""

    def __init__(self, actor: Actor):
        super().__init__()
        self.actor = actor
        self.graph_layers = nn.ModuleList(
            [ExportGraphLayer(layer) for layer in actor.state_encoder.graph_layers]
        )

    def forward(
        self,
        states: torch.Tensor,
        c128: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        edge_group: torch.Tensor,
        edge_attr: torch.Tensor,
        quantities: torch.Tensor,
        candidate_dump_ids: torch.Tensor,
        ego_ids: torch.Tensor,
    ) -> torch.Tensor:
        graph = GraphBatch(
            edge_index, edge_type, edge_group, edge_attr, quantities,
            candidate_dump_ids, ego_ids,
        )
        tokens = self.actor.state_encoder.token_encoder(states, graph)
        for layer in self.graph_layers:
            tokens = layer(tokens, graph)

        row = torch.arange(states.size(0), device=states.device)
        ego = tokens[row, graph.ego_ids]
        pools = torch.stack(
            [tokens[:, section].mean(1) for section in TYPE_SLICES], dim=1
        )
        quantity_pools = torch.stack(
            [graph.quantities[:, section].sum(1) for section in TYPE_SLICES], dim=1
        )
        raw = torch.cat(
            (pools.flatten(1), ego, tokens[:, 18], quantity_pools.flatten(1)), dim=-1
        )

        action_mask = states[:, 19:23]
        task_summary = self.actor.summary_task_encoder(c128[:, :88])
        action_summary = self.actor.summary_action_encoder(c128[:, 88:].reshape(-1, 4, 10))
        context = self.actor.context_encoder(torch.cat((raw, task_summary), dim=-1))
        ego_per_action = ego[:, None].expand(-1, 4, -1)
        ego_quantities = graph.quantities[row, graph.ego_ids][:, None].expand(-1, 4, -1)
        candidate_dump_tokens = tokens[row[:, None], graph.candidate_dump_ids]
        dump_quantities = graph.quantities[row[:, None], graph.candidate_dump_ids]
        candidate_input = torch.cat(
            (
                ego_per_action,
                tokens[:, 9:13],
                candidate_dump_tokens,
                ego_quantities,
                graph.quantities[:, 9:13],
                dump_quantities,
                action_mask.unsqueeze(-1),
                states[:, 23:27, None],
                action_summary,
            ),
            dim=-1,
        )
        candidates = self.actor.candidate_encoder(candidate_input)
        logits = self.actor.score_candidates(candidates, context)
        return logits.masked_fill(~action_mask.bool(), -torch.inf)


def graph_inputs(states: torch.Tensor) -> tuple[torch.Tensor, ...]:
    graph = build_graph(states)
    return (
        graph.edge_index,
        graph.edge_type,
        graph.edge_group,
        graph.edge_attr,
        graph.quantities,
        graph.candidate_dump_ids,
        graph.ego_ids,
    )


def main() -> None:
    require(CHECKPOINT.is_file(), f"checkpoint missing: {CHECKPOINT}")
    checkpoint_hash = sha256(CHECKPOINT)
    require(checkpoint_hash == EXPECTED_CHECKPOINT_SHA256, "checkpoint SHA256 differs from frozen selection")
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    require(saved.get("schema") == "business-graph-summary-ppo-checkpoint-v1"
            and saved.get("candidate_id") == CANDIDATE and saved.get("update_index") == 13,
            "checkpoint identity is not BG128 update 13")
    config = saved["config"]
    actor = Actor(
        spec=config["network"],
        candidate_id=CANDIDATE,
        feature_config=config["feature_config"],
        pair_seed=config["seed"],
    )
    actor.load_state_dict(saved["actor"], strict=True)
    require(digest(actor) == saved["actor_digest"], "loaded Actor digest differs from checkpoint")
    actor.eval()
    model = NetronActor(actor).eval()

    state_a, state_b = sample_states()
    c_config = feature_config_for(CANDIDATE)
    summary_a = SummaryFeatureCache(state_a.numpy(), feature_config=c_config).prefix()
    c_a = torch.from_numpy(summary_a.features.copy())
    graph_a = graph_inputs(state_a)
    inputs = (state_a, c_a, *graph_a)
    names = (
        "states_1519", "c128",
        "edge_index", "edge_type", "edge_group", "edge_attributes_14",
        "node_quantities_19x21", "candidate_dump_ids", "ego_ids",
    )
    dynamic_axes = {
        "edge_index": {1: "edge_count"},
        "edge_type": {0: "edge_count"},
        "edge_group": {0: "edge_count"},
        "edge_attributes_14": {0: "edge_count"},
    }

    with torch.inference_mode():
        expected = actor(state_a, summary_a)
        actual = model(*inputs)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
        torch.onnx.export(
            model,
            inputs,
            OUTPUT_MODEL,
            input_names=list(names),
            output_names=["masked_action_logits_4"],
            dynamic_axes=dynamic_axes,
            opset_version=18,
            dynamo=False,
            do_constant_folding=True,
        )

    onnx_model = onnx.load(OUTPUT_MODEL)
    onnx.checker.check_model(onnx_model)
    onnx_model.doc_string = (
        "BG128 actor weights from checkpoint update 13. Inputs states_1519 and c128 "
        "plus graph tensors produced by business_graph_summary_v1.build_graph. "
        "Dynamic edge_count is supported. Graph construction/validation and C128 "
        "feature generation are external preprocessing and are not represented here."
    )
    props = {
        "campaign": CAMPAIGN,
        "checkpoint_update": "13",
        "checkpoint_sha256": checkpoint_hash,
        "actor_digest": saved["actor_digest"],
        "input_contract": "states_1519, c128, explicit graph tensors",
        "preprocessing": "business_graph_summary_v1.build_graph + C128 build_features",
        "output": "masked action logits; illegal actions are -inf",
    }
    for key, value in props.items():
        item = onnx_model.metadata_props.add()
        item.key = key
        item.value = value
    onnx.save(onnx_model, OUTPUT_MODEL)

    session = ort.InferenceSession(str(OUTPUT_MODEL), providers=["CPUExecutionProvider"])
    edge_counts = []
    max_abs_error = 0.0
    for state in (state_a, state_b):
        sequence = SummaryFeatureCache(state.numpy(), feature_config=c_config).prefix()
        features = torch.from_numpy(sequence.features.copy())
        graph = graph_inputs(state)
        edge_counts.append(int(graph[0].shape[1]))
        torch_output = actor(state, sequence).detach().numpy()
        feed = {
            "states_1519": state.numpy(),
            "c128": features.numpy(),
            "edge_index": graph[0].numpy(),
            "edge_type": graph[1].numpy(),
            "edge_group": graph[2].numpy(),
            "edge_attributes_14": graph[3].numpy(),
            "node_quantities_19x21": graph[4].numpy(),
            "candidate_dump_ids": graph[5].numpy(),
            "ego_ids": graph[6].numpy(),
        }
        onnx_output = session.run(["masked_action_logits_4"], feed)[0]
        np.testing.assert_allclose(onnx_output, torch_output, rtol=3e-5, atol=3e-5)
        np.testing.assert_array_equal(np.isneginf(onnx_output), np.isneginf(torch_output))
        finite_mask = np.isfinite(torch_output)
        max_abs_error = max(
            max_abs_error,
            float(np.max(np.abs(onnx_output[finite_mask] - torch_output[finite_mask]))),
        )

    manifest = {
        "schema": "netron-onnx-export-v1",
        "status": "PASS",
        "artifact": OUTPUT_MODEL.name,
        "artifact_sha256": sha256(OUTPUT_MODEL),
        "artifact_bytes": OUTPUT_MODEL.stat().st_size,
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": checkpoint_hash,
        "actor_digest": saved["actor_digest"],
        "checkpoint_update": 13,
        "candidate_id": CANDIDATE,
        "opset": 18,
        "batch_size_supported": 1,
        "validation": {
            "loaded_actor_digest_matches_checkpoint": True,
            "pytorch_model_output_matches": True,
            "onnxruntime_cpu_matches_pytorch": True,
            "edge_counts_tested": edge_counts,
            "max_finite_logit_absolute_error": max_abs_error,
            "outputs_checked": "four masked action logits",
        },
        "input_note": (
            "Graph construction from the 1519D observation and C128 feature generation "
            "remain external preprocessing; this ONNX contains the trained token encoder, "
            "two GAT layers, readout, C128 encoders, scorer, and action mask."
        ),
    }
    (OUTPUT_DIR / "export_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
