"""Small sealed, typed HDF5 episodes for the independent memory study.

Only N+1 observations are stored.  Histories are stored once, with current/next
exclusive cursors per query; no row contains a copied prefix or GRU hidden state.
The final seal is published last.  Existing files are never overwritten, and an
incomplete publication is retained but rejected rather than treated as a shift.
"""
from __future__ import annotations

from dataclasses import fields
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, get_type_hints

import h5py
import numpy as np

from .events import Event, MemoryTape, Query, Settlement, Token, build_memory
from .targets import REWARD_VERSION, build_grade_targets


DATASET_SCHEMA = "grade-event-memory-episode-hdf5-v1"
OBSERVATION_SCHEMA = "dispatch.physical.v3"
OBSERVATION_DIMENSION = 1519
IDENTITY_FIELDS = (
    "recording_id", "run_index", "environment_seed", "actor_sampling_seed",
    "policy_version", "policy_sha256", "config_sha256", "observation_schema",
    "reward_version", "tokenization_version", "source_kind", "split",
)
Validator = Callable[[dict[str, Any]], None]


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                       separators=(",", ":")) + "\n").encode("utf-8")


def _pairs(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _decode_json(data: str | bytes) -> Any:
    return json.loads(data, object_pairs_hook=_pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_identity(value: Mapping[str, Any]) -> dict:
    if not isinstance(value, Mapping) or not set(IDENTITY_FIELDS) <= set(value):
        raise ValueError("episode identity is missing required explicit fields")
    identity = _decode_json(_json_bytes(dict(value)))
    for key, item in identity.items():
        if not isinstance(key, str) or type(item) not in (str, int, float, bool, type(None)):
            raise ValueError("identity values must be small JSON scalars")
        if type(item) is float and not math.isfinite(item):
            raise ValueError("nonfinite identity value")
    for name in IDENTITY_FIELDS:
        item = identity[name]
        if name in ("run_index", "environment_seed", "actor_sampling_seed"):
            if type(item) is not int or not 0 < item < 2**63:
                raise ValueError(f"{name} must be an explicit positive int64")
        elif not isinstance(item, str) or not item or item != item.strip() or "\x00" in item:
            raise ValueError(f"{name} must be a nonempty identity string")
    for name in ("policy_sha256", "config_sha256"):
        item = identity[name]
        if len(item) != 64 or any(c not in "0123456789abcdef" for c in item) or item == "0" * 64:
            raise ValueError(f"invalid {name}")
    if identity["observation_schema"] != OBSERVATION_SCHEMA:
        raise ValueError("observation schema differs from real dispatch.physical.v3")
    if identity["reward_version"] != REWARD_VERSION:
        raise ValueError("reward version differs from grade-only cost contract")
    allowed_splits = {"synthetic": {"synthetic"}, "fixed_actor": {"train", "validation"},
                      "native_audit": {"audit"}}
    if identity["source_kind"] not in allowed_splits or identity["split"] not in allowed_splits[identity["source_kind"]]:
        raise ValueError("source_kind/split cannot mix native, synthetic and on-policy data")
    return identity


def _valid_evaluation(value: Mapping[str, Any]) -> dict:
    if not isinstance(value, Mapping) or not {"ore_tons", "waste_tons", "total_wait_seconds"} <= set(value):
        raise ValueError("evaluation must retain production and waiting metrics, including failed shifts")
    result = _decode_json(_json_bytes(dict(value)))
    for key, item in result.items():
        if not isinstance(key, str) or type(item) not in (str, int, float, bool, type(None)):
            raise ValueError("evaluation values must be small JSON scalars")
        if type(item) is float and not math.isfinite(item):
            raise ValueError("nonfinite evaluation metric")
    for name in ("ore_tons", "waste_tons", "total_wait_seconds"):
        if type(result[name]) not in (int, float) or not math.isfinite(result[name]) or result[name] < 0:
            raise ValueError(f"invalid nonnegative evaluation metric: {name}")
    result["ore_target_met"] = result["ore_tons"] >= 16000.0
    result["waste_target_met"] = result["waste_tons"] >= 4000.0
    result["production_shortfall"] = not (result["ore_target_met"] and result["waste_target_met"])
    # No implementation of a production assurance layer is implied by an HDF5
    # file or by a particular observed shift meeting both targets.
    result["production_guarantee"] = "pending"
    result["ppo_authorized"] = False
    return result


def _episode(*, identity: Mapping[str, Any], states: Any, actions: Any,
             old_log_probs: Any, memory: MemoryTape, evaluation: Mapping[str, Any],
             validator: Validator | None) -> dict:
    identity = _valid_identity(identity)
    if identity["source_kind"] != "synthetic" and validator is None:
        raise ValueError("nonsynthetic admission requires runtime/config/policy validator")
    if not isinstance(memory, MemoryTape):
        raise ValueError("memory must be a validated MemoryTape")
    rebuilt = build_memory(memory.events)
    if rebuilt != memory:
        raise ValueError("derived token/query/settlement tables disagree with events")
    count = len(memory.queries)
    obs = np.asarray(states)
    if obs.shape != (count + 1, OBSERVATION_DIMENSION) or obs.dtype.kind != "f" or not np.isfinite(obs).all():
        raise ValueError("complete shift requires finite float states[N+1,1519]")
    obs = np.array(obs, dtype=np.float32, copy=True)
    if not np.isfinite(obs).all():
        raise ValueError("states overflow float32")
    act = np.asarray(actions)
    if act.shape != (count,) or act.dtype.kind not in "iu" or (act < 0).any() or (act > 3).any():
        raise ValueError("actions must be N integer load-site actions 0..3")
    act = np.array(act, dtype=np.int8, copy=True)
    logs = np.asarray(old_log_probs)
    if logs.shape != (count,) or logs.dtype.kind != "f" or not np.isfinite(logs).all() or (logs > 1e-7).any():
        raise ValueError("old_log_probs must be N finite, nonpositive sampled-policy log probabilities")
    result = dict(identity=identity, states=obs, actions=act,
                  old_log_probs=np.array(logs, dtype=np.float64, copy=True), memory=memory,
                  targets=build_grade_targets(memory), evaluation=_valid_evaluation(evaluation))
    if validator is not None:
        validator(result)
    return result


def _write_array(group: h5py.Group, name: str, array: Any) -> None:
    value = np.asarray(array)
    kwargs = dict(compression="gzip", compression_opts=4, shuffle=True) if value.size else {}
    if value.dtype.kind == "U":
        group.create_dataset(name, data=value.astype(object), dtype=h5py.string_dtype("utf-8"),
                             compression="gzip" if value.size else None)
    else:
        group.create_dataset(name, data=value, **kwargs)


def _write_table(root: h5py.File, name: str, records: tuple, cls: type) -> None:
    group = root.create_group(name)
    hints = get_type_hints(cls)
    for field in fields(cls):
        values = [getattr(record, field.name) for record in records]
        dtype = {int: np.int64, float: np.float64, bool: np.bool_, str: str}[hints[field.name]]
        _write_array(group, field.name, np.asarray(values, dtype=dtype))


def _read_table(root: h5py.File, name: str, cls: type) -> tuple:
    group = root[name]
    expected = {field.name for field in fields(cls)}
    if not isinstance(group, h5py.Group) or set(group) != expected:
        raise ValueError(f"typed table schema differs: {name}")
    hints = get_type_hints(cls)
    columns, count = {}, None
    for field in fields(cls):
        dataset = group[field.name]
        if not isinstance(dataset, h5py.Dataset) or len(dataset.shape) != 1:
            raise ValueError(f"invalid typed column: {name}/{field.name}")
        if count is None:
            count = len(dataset)
        elif len(dataset) != count:
            raise ValueError(f"column lengths differ: {name}")
        hint = hints[field.name]
        if hint is str:
            string_info = h5py.check_string_dtype(dataset.dtype)
            if string_info is None or string_info.encoding != "utf-8" or string_info.length is not None:
                raise ValueError("string field must use typed HDF5 strings")
            columns[field.name] = dataset.asstr()[...].tolist()
        else:
            expected_dtype = np.dtype({int: np.int64, float: np.float64, bool: np.bool_}[hint])
            if dataset.dtype != expected_dtype:
                raise ValueError(f"column dtype differs: {name}/{field.name}")
            columns[field.name] = dataset[...].tolist()
    return tuple(cls(**{key: column[index] for key, column in columns.items()}) for index in range(count or 0))


def _transitions(episode: dict) -> dict[str, np.ndarray]:
    memory, count = episode["memory"], len(episode["actions"])
    result = dict(state_index=np.arange(count, dtype=np.int64),
                  next_state_index=np.arange(1, count + 1, dtype=np.int64),
                  action=episode["actions"], old_log_prob=episode["old_log_probs"],
                  terminated=np.arange(count) == count - 1,
                  truncated=np.zeros(count, dtype=np.bool_))
    for name in ("current_cursor", "next_cursor", "current_seq", "next_seq", "decision_sequence"):
        result[name] = np.asarray([getattr(query, name) for query in memory.queries], dtype=np.int64)
    return result


def _label_columns(episode: dict) -> dict[str, np.ndarray]:
    targets = episode["targets"]
    return dict(grade_cost=np.asarray(targets.step_costs, dtype=np.float64),
                mc_grade_cost=np.asarray(targets.mc_returns, dtype=np.float64),
                closed_windows_before=np.asarray(targets.closed_windows_before, dtype=np.int64))


def _fsync(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _sidecars(path: Path) -> tuple[Path, Path]:
    return Path(str(path) + ".manifest.json"), Path(str(path) + ".seal.json")


def write_episode_hdf5(path: str | Path, *, identity: Mapping[str, Any], states: Any,
                       actions: Any, old_log_probs: Any, memory: MemoryTape,
                       evaluation: Mapping[str, Any], validator: Validator | None = None) -> dict:
    """Seal a complete episode, publishing each file atomically with no overwrite.

    The seal is the admission point; a crash before it leaves an explicitly
    incomplete episode that read_episode_hdf5 rejects.  No automatic resampling,
    deletion, model creation, fitting, or authorization occurs here.
    """
    episode = _episode(identity=identity, states=states, actions=actions,
                       old_log_probs=old_log_probs, memory=memory,
                       evaluation=evaluation, validator=validator)
    path = Path(path)
    manifest_path, seal_path = _sidecars(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if any(candidate.exists() for candidate in (path, manifest_path, seal_path)):
        raise FileExistsError("episode or admission sidecar already exists; never overwrite/resubmit")
    counts = dict(states=len(episode["states"]), transitions=len(episode["actions"]),
                  events=len(memory.events), tokens=len(memory.tokens), queries=len(memory.queries), settlements=4)
    with tempfile.TemporaryDirectory(prefix=f".{path.name}.staging-", dir=path.parent) as staging:
        stage = Path(staging)
        staged_h5, staged_manifest, staged_seal = stage / "episode.h5", stage / "manifest.json", stage / "seal.json"
        with h5py.File(staged_h5, "x") as target:
            target.attrs["schema"] = DATASET_SCHEMA
            target.attrs["status"] = "complete"
            target.attrs["identity_json"] = _json_bytes(episode["identity"]).decode()
            target.attrs["evaluation_json"] = _json_bytes(episode["evaluation"]).decode()
            target.attrs["terminal_mc_grade_cost"] = 0.0
            _write_array(target, "states", episode["states"])
            transitions = target.create_group("transitions")
            for name, array in _transitions(episode).items():
                _write_array(transitions, name, array)
            for name, records, cls in (("material_events", memory.events, Event),
                                        ("material_tokens", memory.tokens, Token),
                                        ("queries", memory.queries, Query),
                                        ("grade_settlements", memory.settlements, Settlement)):
                _write_table(target, name, records, cls)
            labels = target.create_group("labels")
            for name, array in _label_columns(episode).items():
                _write_array(labels, name, array)
            target.flush()
        _fsync(staged_h5)
        digest = file_sha256(staged_h5)
        manifest = dict(schema=DATASET_SCHEMA, status="complete", identity=episode["identity"],
                        counts=counts, hdf5_sha256=digest,
                        created_utc=datetime.now(timezone.utc).isoformat(),
                        labels="grade-only MC; no waiting or production in training targets",
                        production_guarantee="pending", ppo_authorized=False)
        manifest_bytes = _json_bytes(manifest)
        staged_manifest.write_bytes(manifest_bytes)
        _fsync(staged_manifest)
        seal = dict(schema=DATASET_SCHEMA, status="complete", hdf5_sha256=digest,
                    manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest())
        staged_seal.write_bytes(_json_bytes(seal))
        _fsync(staged_seal)
        for source, destination in ((staged_h5, path), (staged_manifest, manifest_path), (staged_seal, seal_path)):
            os.link(source, destination)  # atomic CREATE_NEW semantics, unlike replace/rename
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    return manifest


def _compare_columns(group: h5py.Group, expected: Mapping[str, np.ndarray]) -> None:
    if set(group) != set(expected):
        raise ValueError("stored derived columns differ from schema")
    for name, value in expected.items():
        dataset = group[name]
        if not isinstance(dataset, h5py.Dataset) or dataset.dtype != value.dtype or not np.array_equal(dataset[...], value):
            raise ValueError(f"stored derived column disagrees: {name}")


def read_episode_hdf5(path: str | Path, *, expected_identity: Mapping[str, Any] | None = None,
                      validator: Validator | None = None) -> dict:
    """Verify external file/manifest seals, rebuild causal tables, and admit data."""
    path = Path(path)
    manifest_path, seal_path = _sidecars(path)
    if not all(candidate.is_file() for candidate in (path, manifest_path, seal_path)):
        raise ValueError("incomplete episode: HDF5, manifest and final complete seal required")
    manifest_bytes = manifest_path.read_bytes()
    manifest, seal = _decode_json(manifest_bytes), _decode_json(seal_path.read_bytes())
    if manifest.get("schema") != DATASET_SCHEMA or seal.get("schema") != DATASET_SCHEMA or manifest.get("status") != "complete" or seal.get("status") != "complete":
        raise ValueError("invalid or incomplete episode seal")
    if hashlib.sha256(manifest_bytes).hexdigest() != seal.get("manifest_sha256"):
        raise ValueError("manifest SHA256 differs from seal")
    digest = file_sha256(path)
    if digest != manifest.get("hdf5_sha256") or digest != seal.get("hdf5_sha256"):
        raise ValueError("HDF5 SHA256 differs from complete manifest/seal")
    with h5py.File(path, "r") as source:
        if source.attrs.get("schema") != DATASET_SCHEMA or source.attrs.get("status") != "complete":
            raise ValueError("HDF5 internal schema/status differs")
        if set(source) != {"states", "transitions", "material_events", "material_tokens", "queries", "grade_settlements", "labels"}:
            raise ValueError("HDF5 root schema differs")
        identity = _decode_json(source.attrs["identity_json"])
        if identity != manifest.get("identity"):
            raise ValueError("HDF5 identity differs from manifest")
        if expected_identity is not None and any(identity.get(key) != value for key, value in expected_identity.items()):
            raise ValueError("episode identity differs from requested config/policy/recording/seeds")
        events = _read_table(source, "material_events", Event)
        memory = build_memory(events)
        if _read_table(source, "material_tokens", Token) != memory.tokens or _read_table(source, "queries", Query) != memory.queries or _read_table(source, "grade_settlements", Settlement) != memory.settlements:
            raise ValueError("stored material memory tables disagree with canonical events")
        if source["states"].dtype != np.dtype("float32"):
            raise ValueError("states must use compact float32")
        episode = _episode(identity=identity, states=source["states"][...],
                           actions=source["transitions/action"][...],
                           old_log_probs=source["transitions/old_log_prob"][...], memory=memory,
                           evaluation=_decode_json(source.attrs["evaluation_json"]), validator=validator)
        _compare_columns(source["transitions"], _transitions(episode))
        _compare_columns(source["labels"], _label_columns(episode))
        if source.attrs.get("terminal_mc_grade_cost") != 0.0:
            raise ValueError("terminal MC cost must be exactly zero")
    counts = dict(states=len(episode["states"]), transitions=len(episode["actions"]),
                  events=len(memory.events), tokens=len(memory.tokens), queries=len(memory.queries), settlements=4)
    if manifest.get("counts") != counts or manifest.get("production_guarantee") != "pending" or manifest.get("ppo_authorized") is not False:
        raise ValueError("manifest counts/gates differ from validated episode")
    episode["manifest"] = manifest
    return episode


def validate_episode_splits(episodes: list[dict]) -> None:
    """Reject row-level/random split substitutes and seed leakage between arms."""
    recordings: set[tuple] = set()
    runs: set[int] = set()
    seed_splits: dict[tuple[str, int], str] = {}
    cohort = None
    for episode in episodes:
        identity = _valid_identity(episode["identity"])
        recording = (identity["recording_id"],identity["run_index"],identity["environment_seed"])
        if recording in recordings or identity["run_index"] in runs:
            raise ValueError("episode recording appears more than once; split only whole episodes")
        recordings.add(recording)
        runs.add(identity["run_index"])
        binding = tuple(identity[k] for k in ('policy_version','policy_sha256','source_kind',
            'observation_schema','reward_version','tokenization_version'))
        if cohort is not None and binding != cohort:
            raise ValueError("fixed Actor cohort mixes policy/source/schema contracts")
        cohort = binding
        for name in ("environment_seed", "actor_sampling_seed"):
            key = (name, identity[name])
            previous = seed_splits.setdefault(key, identity["split"])
            if previous != identity["split"]:
                raise ValueError(f"{name} leaks across complete-episode splits")
