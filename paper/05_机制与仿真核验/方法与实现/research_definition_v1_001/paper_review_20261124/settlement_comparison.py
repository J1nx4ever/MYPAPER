"""Compare UNLOAD and PROCESS grade on the same frozen Full u32 trajectories.

Run with the project Python. Standard output is the reproducible JSON report.
"""
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


PROJECT = next(path for path in Path(__file__).resolve().parents if path.name == "lagrangian_ppo_v1")
CAMPAIGNS = {
    "s01": "bg128-continuous-mixed-20261119-001",
    "s02": "bg128-continuous-mixed-20261120-s02",
    "s03": "bg128-continuous-mixed-20261120-s03",
}
SCENES = ("nominal", "grade_swap", "held_order_1", "held_order_2", "new_numeric", "joint_shift", "waste_swap")


def verified_bytes(path, expected):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"File digest mismatch: {path}")
    return data


def compliant(grade):
    return grade is not None and bool(.26 - 1e-12 <= grade <= .28 + 1e-12)


def window_totals(events):
    kinds, windows = events["kind"][:], events["window"][:]
    tons, metal = events["tons"][:], events["metal"][:]
    result = {}
    for kind in (b"UNLOAD", b"PROCESS"):
        selected = kinds == kind
        physical_windows = windows[selected]
        if not np.all((0 <= physical_windows) & (physical_windows < 4)):
            raise ValueError("Material event outside the four physical windows")
        qty = np.bincount(physical_windows, weights=tons[selected], minlength=4)
        mass = np.bincount(physical_windows, weights=metal[selected], minlength=4)
        result[kind] = [(q, m, m / q if q > 0 else None) for q, m in zip(qty, mass, strict=True)]
    return result


def read_episodes(report_path, expected_digest, seen):
    report = json.loads(verified_bytes(report_path, expected_digest))
    episodes = report["episodes"]
    if len(episodes) != 100 or report["evaluation_optimizer_updates"] != 0:
        raise ValueError(f"Unexpected evaluation inventory: {report_path}")
    results = []
    for item, source in zip(episodes, report["source_refs"], strict=True):
        run = item["identity"]["run_index"]
        material = report_path.parent / "material" / f"run_{run}.h5"
        manifest_path = material.with_suffix(".h5.manifest.json")
        if Path(source["path"]).name != manifest_path.name or run in seen:
            raise ValueError(f"Episode inventory mismatch for run {run}")
        seen.add(run)
        manifest = json.loads(verified_bytes(manifest_path, source["sha256"]))
        if manifest["identity"] != item["identity"]:
            raise ValueError(f"Episode identity mismatch for run {run}")
        verified_bytes(material, manifest["hdf5_sha256"])
        with h5py.File(material, "r") as handle:
            totals = window_totals(handle["material_events"])
        observed = item["physical"]["windows"]
        if [row["window"] for row in observed] != list(range(4)):
            raise ValueError(f"Unexpected windows for run {run}")
        shift = []
        for unloading, processing, row in zip(totals[b"UNLOAD"], totals[b"PROCESS"], observed, strict=True):
            q, m, grade = processing
            errors = [abs(q - row["tons"]), abs(m - row["metal_tons"])]
            if (grade is None) != row["empty"] or (grade is not None and abs(grade - row["grade"]) > 1e-11):
                raise ValueError(f"Processed grade disagrees with the archived result for run {run}")
            if max(errors) > 1e-5 or compliant(grade) != row["compliant"]:
                raise ValueError(f"Processed quantity, metal or outcome differs for run {run}")
            shift.append((unloading[2], grade, *errors))
        results.append(shift)
    return results


def summarize(shifts):
    windows = [window for shift in shifts for window in shift]
    deltas = [100 * abs(u - p) for u, p, _, _ in windows if u is not None and p is not None]
    return {
        "shifts": len(shifts), "windows": len(windows), "both_grades_defined": len(deltas),
        "unload_compliant_only_windows": sum(compliant(u) and not compliant(p) for u, p, _, _ in windows),
        "process_compliant_only_windows": sum(compliant(p) and not compliant(u) for u, p, _, _ in windows),
        "all_four_compliant_unload_shifts": sum(all(compliant(w[0]) for w in shift) for shift in shifts),
        "all_four_compliant_process_shifts": sum(all(compliant(w[1]) for w in shift) for shift in shifts),
        "mean_absolute_grade_gap_pp": float(np.mean(deltas)),
        "p95_absolute_grade_gap_pp": float(np.quantile(deltas, .95)),
        "max_process_ore_reconstruction_error_t": max(w[2] for w in windows),
        "max_process_metal_reconstruction_error_t": max(w[3] for w in windows),
    }


def main():
    by_scene = {name: [] for name in SCENES}
    by_seed = {}
    seen = set()
    for seed, campaign in CAMPAIGNS.items():
        root = PROJECT / "runs" / campaign
        receipt = json.loads((root / "suite_complete.json").read_bytes())
        if receipt["status"] != "COMPLETE" or receipt["training_updates"] != 32:
            raise ValueError(f"Incomplete campaign: {campaign}")
        refs = {Path(phase["report"]["path"]).parts[-3:]: phase["report"]["sha256"]
                for phase in receipt["phase_results"] if phase["checkpoint"] == receipt["new_checkpoint"]}
        seed_shifts = []
        for scene in SCENES:
            path = root / "evaluation" / scene / "new" / "evaluation_complete.json"
            shifts = read_episodes(path, refs[path.parts[-3:]], seen)
            by_scene[scene].extend(shifts)
            seed_shifts.extend(shifts)
        by_seed[seed] = summarize(seed_shifts)
    report = {
        "campaigns": CAMPAIGNS,
        "definition": "Actual ore entering crusher FIFO (UNLOAD) versus ore crushed (PROCESS), each event assigned to its recorded 2 h window. Grade = metal tons / ore tons. Empty fails; compliant is 26% to 28% inclusive with epsilon 1e-12. Differences are absolute percentage points. Both calculations use the same trajectory. Four-window rates ignore production targets.",
        "all": summarize([shift for shifts in by_scene.values() for shift in shifts]),
        "by_training_seed": by_seed,
        "by_scenario": {name: summarize(shifts) for name, shifts in by_scene.items()},
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
