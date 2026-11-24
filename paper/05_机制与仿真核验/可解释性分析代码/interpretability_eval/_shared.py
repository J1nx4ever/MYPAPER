"""Trajectory I/O and frozen BG128 loading shared by the analysis entry points."""
from contextlib import contextmanager
from functools import lru_cache
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "experiments" / "lagrangian_ppo_v1"
# Project code imports both package names such as business_graph_summary_v1
# and its own "python" namespace. The experiment's namespace has precedence.
for project_path in (ROOT, EXPERIMENT):
    location = str(project_path)
    if location in sys.path:
        sys.path.remove(location)
    sys.path.insert(0, location)

FORMAT = "bg128-interpretability-rollout-v1"
SEEDS = ("s01", "s02", "s03")
POLICIES = ("BG128", "Flat-PPO", "Greedy", "RHO")
GROUPS = ("Transport", "Commitment", "Service", "Material")


def edge_attention_heads(row, layer):
    """Validate the recorded coefficient array against the factual edge order."""
    edges = row.get("edges")
    values = row.get(f"edge_attn_l{layer}")
    if edges is None or values is None:
        return None
    weights = np.asarray(values, dtype=float)
    if weights.ndim == 1:
        weights = weights[:, None]
    if (weights.ndim != 2 or weights.shape[0] != len(edges) or not weights.shape[1]
            or not np.isfinite(weights).all() or ((weights < 0) | (weights > 1.000001)).any()):
        raise ValueError(f"edge_attn_l{layer} must contain a coefficient vector per factual edge")
    return weights


def total_shovel_queue(row):
    """Count queued and Seize-wait trucks once per shovel in dispatch.physical.v3."""
    raw = row.get("raw_obs")
    if raw is not None:
        observation = np.asarray(raw)
        if observation.shape == (1519,):
            slots = observation[1107:1295].reshape(4, 47)[:, 20:38]
            return sum(len(set(waiting[waiting > 0].tolist())) for waiting in slots)
    recorded = row.get("total_shovel_queue")
    if type(recorded) is int and recorded >= 0:
        return recorded
    shovels = row.get("shovels") or []
    queues = [site.get("queue") for site in shovels]
    if queues and all(type(queue) is int and queue >= 0 for queue in queues):
        return sum(queues)
    return None


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path, expected):
    path = Path(path).expanduser().resolve()
    if (not isinstance(expected, str) or len(expected) != 64
            or sha256(path) != expected):
        raise ValueError(f"SHA256 mismatch for {path}")
    return path


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


@contextmanager
def atomic_path(path):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=destination.parent, suffix=destination.suffix,
                            prefix=f".{destination.stem}-", delete=False) as stream:
        staging = Path(stream.name)
    try:
        yield staging
        os.replace(staging, destination)
    finally:
        staging.unlink(missing_ok=True)


def encode(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def write_json(path, value):
    with atomic_path(path) as staging:
        staging.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False,
                                      indent=2) + "\n", encoding="utf-8")


def iter_rows(path):
    path = Path(path)
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        source = pq.ParquetFile(path)
        encoded = (source.schema_arrow.metadata or {}).get(b"schema") == FORMAT.encode()
        for batch in source.iter_batches(batch_size=256):
            for record in batch.to_pylist():
                yield json.loads(record["record_json"]) if encoded else record
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as source:
            if source["schema"].item() != FORMAT:
                raise ValueError(f"Unknown trajectory format in {path}")
            for record in source["record_json"]:
                yield json.loads(record)
    elif path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
    else:
        raise ValueError("Expected .parquet, .npz or .jsonl trajectories")


def read_rows(path):
    return list(iter_rows(path))


def write_rows(path, shifts):
    path = Path(path)
    count = 0
    with atomic_path(path) as staging:
        if path.suffix == ".parquet":
            import pyarrow as pa
            import pyarrow.parquet as pq

            schema = pa.schema([
                ("policy", pa.string()), ("training_seed", pa.string()),
                ("scenario", pa.string()), ("shift_id", pa.string()), ("seed", pa.int64()),
                ("decision_index", pa.int64()), ("sim_time_min", pa.float64()),
                ("record_json", pa.large_string()),
            ], metadata={b"schema": FORMAT.encode()})
            with pq.ParquetWriter(staging, schema, compression="zstd") as writer:
                for rows in shifts:
                    if rows:
                        table = pa.Table.from_pylist(
                            [dict(row, record_json=encode(row)) for row in rows], schema=schema
                        )
                        writer.write_table(table)
                        count += len(rows)
        elif path.suffix == ".npz":
            records = [encode(row) for rows in shifts for row in rows]
            np.savez_compressed(staging, schema=FORMAT, record_json=np.array(records, dtype=str))
            count = len(records)
        else:
            raise ValueError("Output must be .parquet or .npz")
        if not count:
            raise ValueError("No trajectory decisions were collected")
    return count


@lru_cache(maxsize=4)
def load_model(checkpoint_path, checkpoint_sha256):
    """Restore a locally trusted, sealed mixed BG128 learner for read-only inference."""
    path = verify_file(checkpoint_path, checkpoint_sha256)
    import torch

    extension = EXPERIMENT / (
        "change_plans/token_attention_001/business_graph_summary_009/"
        "training_extension_001/train_extension.py"
    )
    spec = importlib.util.spec_from_file_location("_summary_bg128_extension", extension)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with torch.random.fork_rng(devices=[]):
        model = module._extension_learner().from_checkpoint(
            path, expected_sha256=checkpoint_sha256, restore_global_rng=False
        )
    if model.config["candidate_id"] != "s01_BG128":
        raise ValueError(f"Checkpoint does not contain the frozen BG128 architecture: {path}")
    model.actor.eval().requires_grad_(False)
    model.critic.eval().requires_grad_(False)
    return model
