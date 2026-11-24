"""Read sealed historical shifts and revalidate their offline admission."""

from collections.abc import Iterator
from contextlib import closing, nullcontext
from functools import lru_cache
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import sqlite3
import sys
from unittest.mock import patch


EXPERIMENT = Path(__file__).resolve().parents[2] / "experiments/lagrangian_ppo_v1"
if str(EXPERIMENT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT))

from business_graph_summary_campaign_v1 import collection
from business_graph_summary_campaign_v1.budget import digest
from business_graph_summary_campaign_v1.plan_io import load_phase, load_reference, reference
from python.constrained_batch import load_dataset
from python.reward_config import configuration_from_components


_BUSINESS_PLAN = EXPERIMENT / "change_plans/token_attention_001/business_graph_summary_009"
_MIXED_PLAN = _BUSINESS_PLAN / "generalization_001/grade_assignment_generalization_001/continuous_training_001"
_RUNS = (
    ("BG128", "s01", "bg128-continuous-mixed-20261119-001"),
    ("BG128", "s02", "bg128-continuous-mixed-20261120-s02"),
    ("BG128", "s03", "bg128-continuous-mixed-20261120-s03"),
    ("Flat-PPO", "s01", "bg128-ablation-20261121-flat-s01"),
)
_SCENARIOS = {"nominal": "Nominal", "grade_swap": "Swapped", "new_numeric": "Shifted"}


def _check(condition, message):
    if not condition:
        raise ValueError(message)


@lru_cache(maxsize=None)
def _load_module(name, path):
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _historical_slot(ledger, run_index, job, receipt, dataset_ref):
    status, payload, checksum = ledger.execute(
        "SELECT status, payload, checksum FROM slots WHERE run = ?", (run_index,)
    ).fetchone() or (None, "null", None)
    record = json.loads(payload)
    _check(status == "COMPLETE" and record is not None and record["status"] == status
           and checksum == digest(record) and record["identity_sha256"] == digest(job),
           f"Invalid completed ledger slot {run_index}")
    completion = dict(dataset_ref=dataset_ref, receipt=receipt)
    _check(record["completion_sha256"] == digest(completion)
           and all(record[key] == value for key, value in completion.items()),
           f"Material receipt differs from completed ledger slot {run_index}")


def historical_sources(scenarios: set[str] | None = None,
                       limit_per_group: int | None = None) -> Iterator[dict]:
    """Yield registered shifts, grouped by policy, training seed and scenario.

    A limit selects the same initial pair ordinals across archive groups.
    """
    if limit_per_group is not None and limit_per_group < 1:
        raise ValueError("limit_per_group must be positive")
    requested = None if scenarios is None else {scenario.casefold() for scenario in scenarios}
    for policy, training_seed, campaign in _RUNS:
        run_root = EXPERIMENT / "runs" / campaign
        completion = json.loads((run_root / "suite_complete.json").read_bytes())
        checkpoint = completion["new_checkpoint"]
        suite = completion.get("suite") or completion["campaign"]
        _check(completion["status"] == "COMPLETE" and completion["training_updates"] == 32
               and completion["evaluation_optimizer_updates"] == 0
               and reference(checkpoint["path"]) == checkpoint,
               f"Invalid frozen u32 completion for {campaign}")
        suite_config = load_reference(suite)
        _check(suite_config["campaign_id"] == campaign, f"Frozen suite differs for {campaign}")
        phases = {ref["path"]: ref for ref in suite_config["phase_configs"]}
        with closing(sqlite3.connect((run_root / "budget.sqlite").as_uri() + "?mode=ro", uri=True)) as ledger:
            for phase in completion["phase_results"]:
                if phase["checkpoint"] != checkpoint:
                    continue
                report_ref = phase["report"]
                scene_id = Path(report_ref["path"]).parents[1].name
                scenario = _SCENARIOS.get(scene_id, scene_id)
                if requested is not None and requested.isdisjoint((scenario.casefold(), scene_id.casefold())):
                    continue
                report = load_reference(report_ref)
                config_ref = report["phase_config"]
                _check(report["checkpoint_provenance"] == checkpoint and report["scenario_id"] == scene_id
                       and report["policy_arm"] == "new" and report["evaluation_optimizer_updates"] == 0
                       and phases.get(config_ref["path"]) == config_ref
                       and reference(config_ref["path"]) == config_ref,
                       f"Invalid frozen evaluation report for {campaign}/{scene_id}")
                config = load_phase(config_ref["path"])
                jobs = sorted(config["jobs"], key=lambda job: job["pair_ordinal"])
                sources = {Path(ref["path"]).name: ref for ref in report["source_refs"]}
                identities = {record["identity"]["run_index"]: record["identity"] for record in report["episodes"]}
                _check(len(jobs) == len(sources) == len(identities) == report["n_shifts"],
                       f"Incomplete evaluation inventory for {campaign}/{scene_id}")
                for job in jobs[:limit_per_group]:
                    run_index = job["run_index"]
                    receipt_path = (Path(job["python_output"]) / "episodes"
                                    / f"run_{run_index}" / "material_receipt.json")
                    raw_receipt = receipt_path.read_bytes()
                    receipt = json.loads(raw_receipt)
                    dataset_ref = sources[f"run_{run_index}.h5.manifest.json"]
                    _check(receipt["identity"] == identities[run_index]
                           and receipt["phase_config"] == config_ref
                           and receipt["references"]["actor"] == report["actor_artifact"]
                           and receipt["references"]["episode_manifest"] == dataset_ref,
                           f"Material receipt differs from evaluation report for run {run_index}")
                    _historical_slot(ledger, run_index, job, receipt, dataset_ref)
                    yield dict(
                        policy=policy, training_seed=training_seed, scenario=scenario, scene_id=scene_id,
                        pair_ordinal=job["pair_ordinal"], run_index=run_index,
                        swap_time_min=0.0 if scene_id == "grade_swap" else None,
                        receipt_path=str(receipt_path), receipt_sha256=sha256(raw_receipt).hexdigest(),
                        checkpoint_path=checkpoint["path"], checkpoint_sha256=checkpoint["sha256"],
                        suite_path=suite["path"], suite_sha256=suite["sha256"],
                    )


def load_source(source: dict) -> tuple[dict, dict]:
    """Return admitted episode data and the actual configuration read from RPC."""
    import torch

    receipt = load_reference(dict(path=source["receipt_path"], sha256=source["receipt_sha256"]))
    refs = receipt["references"]
    _check(reference(refs["config"]["path"]) == refs["config"], "Material receipt configuration changed")
    phase = load_phase(refs["config"]["path"])
    job, = (job for job in phase["jobs"] if job["run_index"] == source["run_index"])
    _check(job["scene_id"] == source["scene_id"] and job["pair_ordinal"] == source["pair_ordinal"],
           "Historical shift locator differs from its frozen configuration")
    runtime = _load_module("interpretability_mixed_runtime", _MIXED_PLAN / "mixed_runtime.py")
    storage = _load_module("interpretability_bg128_storage", _MIXED_PLAN / "recovery_002/storage_scope.py")
    suite = load_reference(dict(path=source["suite_path"], sha256=source["suite_sha256"]))
    actor_scope = nullcontext()
    if source["policy"] == "Flat-PPO":
        ablation = _load_module("interpretability_ablation_runtime",
                                _BUSINESS_PLAN / "structural_ablation_001/ablation_runtime.py")
        _check(suite["ablation_variant"] == "flat_mlp", "Expected frozen Flat-PPO architecture")
        _, load_actor = ablation._learner(runtime, suite)
        actor_scope = patch.object(collection, "frozen_actor", load_actor)
    else:
        _check(source["policy"] == "BG128", "Unknown historical policy")
    with runtime._exact_verification([job]), storage.bg128_storage(), actor_scope:
        threads = torch.get_num_threads()
        try:
            # Exact replay uses the same CPU reduction order as collection.
            torch.set_num_threads(suite["cpu_threads"])
            episode = collection.load_admitted_episode(receipt)
        finally:
            torch.set_num_threads(threads)
    _check(episode["identity"] == receipt["identity"], "Admitted episode identity changed")

    reader, summaries, _, _, _ = load_dataset(Path(refs["raw_manifest"]["path"]).parent, require_real=True)
    try:
        summary, = summaries
        actual = configuration_from_components(summary["start_info"]["reward_components"], terminal=False).to_dict()
        _check(all(actual.get(key) == value for key, value in job["expected_rpc_config"].items()),
               "Raw RPC configuration differs from admitted job")
        return episode, actual
    finally:
        reader.close()
