"""Compare C128 and raw observations on future physical-state prediction."""
from __future__ import annotations

import argparse
from collections import defaultdict
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.linalg import solve

from _shared import iter_rows, total_shovel_queue, write_json

TRAINING_SEEDS = ("s01", "s02", "s03")
TARGETS = (
    ("total_shovel_queue_15min", "total_shovel_queue", 15),
    ("grade_margin_30min", "grade_margin_pp", 30),
)
REPRESENTATIONS = (("c128", "summary_emb", 128), ("raw", "raw_obs", 1519))
BATCH_ROWS = 4096
ROW_FIELDS = ("training_seed", "shift_id", "seed", "scenario", "sim_time_min",
              "decision_index", "grade_margin_pp")


def number(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def stage_features(input_path, directory):
    """Keep small row metadata in memory and put dense vectors on local scratch."""
    paths = {name: directory / f"{name}.f32" for name, _, _ in REPRESENTATIONS}
    rows = []
    with paths["c128"].open("wb") as c128, paths["raw"].open("wb") as raw:
        streams = {"c128": c128, "raw": raw}
        for row in iter_rows(input_path):
            if row["policy"] != "BG128" or row.get("summary_emb") is None:
                continue
            for name, column, width in REPRESENTATIONS:
                vector = np.asarray(row[column], dtype=np.float32)
                if vector.shape != (width,) or not np.isfinite(vector).all():
                    raise ValueError(f"{column} requires finite vectors of length {width}")
                streams[name].write(vector.tobytes())
            metadata = {field: row.get(field) for field in ROW_FIELDS}
            metadata["total_shovel_queue"] = total_shovel_queue(row)
            rows.append(metadata)
    present = {row["training_seed"] for row in rows}
    if present != set(TRAINING_SEEDS):
        raise ValueError(f"expected BG128 inputs for s01–s03, received {sorted(map(str, present))}")
    matrices = {name: np.memmap(paths[name], dtype=np.float32, mode="r", shape=(len(rows), width))
                for name, _, width in REPRESENTATIONS}
    return rows, matrices


def group_rollouts(rows):
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        key = (row["training_seed"], row["shift_id"])
        groups[key].append(index)
    for indices in groups.values():
        indices.sort(key=lambda index: (float(rows[index]["sim_time_min"]),
                                        int(rows[index]["decision_index"])))
        shift = [rows[index] for index in indices]
        identities = {(row["scenario"], row["seed"]) for row in shift}
        if len(identities) != 1:
            raise ValueError(f"shift {shift[0]['shift_id']} has inconsistent scenario or seed")
        decisions = [int(row["decision_index"]) for row in shift]
        if len(set(decisions)) != len(decisions):
            raise ValueError(f"shift {shift[0]['shift_id']} has duplicate decision indices")
        if any(number(row["sim_time_min"]) is None for row in shift):
            raise ValueError(f"shift {shift[0]['shift_id']} has invalid simulation time")
    return groups


def future_labels(rows, groups, field, horizon, maximum_lag):
    result = np.full(len(rows), np.nan, dtype=np.float64)
    for indices in groups.values():
        times = np.array([float(rows[index]["sim_time_min"]) for index in indices])
        for position, index in enumerate(indices):
            future_time = times[position] + horizon
            if future_time > times[-1]:
                continue
            observed = np.searchsorted(times, future_time, side="right") - 1
            if observed > position and future_time - times[observed] <= maximum_lag:
                value = number(rows[indices[observed]].get(field))
                if value is not None:
                    result[index] = value
    return result


def choose_test_seeds(rows, random_seed):
    env_seeds = sorted({int(row["seed"]) for row in rows})
    if len(env_seeds) < 2:
        raise ValueError("ridge probing needs at least two distinct environment seeds")
    rng = np.random.default_rng(random_seed)
    count = min(len(env_seeds) - 1, max(1, round(0.2 * len(env_seeds))))
    return sorted(int(seed) for seed in rng.choice(env_seeds, size=count, replace=False))


def standardize(train, other):
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale == 0.0] = 1.0
    return (train - mean) / scale, None if other is None else (other - mean) / scale


def batches(indices):
    for i in range(0, len(indices), BATCH_ROWS):
        yield indices[i:i + BATCH_ROWS]


def fit_ridge(matrix, expected, train, test):
    """Fit exact Ridge alpha=1 with train-only scaling and a free intercept."""
    width = matrix.shape[1]
    mean = np.zeros(width)
    for ids in batches(train):
        mean += matrix[ids].sum(axis=0, dtype=np.float64)
    mean /= len(train)
    center = expected[train].mean()
    gram = np.zeros((width, width))
    right = np.zeros(width)
    for ids in batches(train):
        block = matrix[ids].astype(np.float64) - mean
        gram += block.T @ block
        right += block.T @ (expected[ids] - center)

    scale = np.sqrt(gram.diagonal() / len(train))
    scale[scale == 0.0] = 1.0
    gram /= scale[None, :]
    gram /= scale[:, None]
    gram.flat[::width + 1] += 1.0
    coefficients = solve(gram, right / scale, assume_a="pos", overwrite_a=True) / scale
    return np.concatenate([(matrix[ids].astype(np.float64) - mean) @ coefficients + center
                           for ids in batches(test)])


def score(expected, predicted):
    squared_error = np.square(expected - predicted)
    centered_expected = expected - expected.mean()
    total_variation = np.square(centered_expected).sum()
    r2 = None if total_variation == 0.0 else float(1 - squared_error.sum() / total_variation)
    pearson_r = None
    if expected.size >= 2 and np.ptp(expected) > 0.0 and np.ptp(predicted) > 0.0:
        centered_predicted = predicted - predicted.mean()
        expected_norm = np.linalg.norm(centered_expected)
        predicted_norm = np.linalg.norm(centered_predicted)
        if expected_norm > 0.0 and predicted_norm > 0.0:
            correlation = (centered_expected / expected_norm) @ (centered_predicted / predicted_norm)
            pearson_r = float(np.clip(correlation, -1.0, 1.0))
    return {"r2": r2, "rmse": float(np.sqrt(squared_error.mean())), "pearson_r": pearson_r}


def across_seeds(values):
    values = [value for value in values if value is not None]
    if not values:
        return None, None
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else None


def probe_metrics(rows, groups, matrices, test_seeds, maximum_lag):
    held_out = np.isin([int(row["seed"]) for row in rows], test_seeds)
    model_seeds = np.asarray([row["training_seed"] for row in rows])
    results = []
    for target, field, horizon in TARGETS:
        target_values = future_labels(rows, groups, field, horizon, maximum_lag)
        for representation, matrix in matrices.items():
            per_seed = {}
            for seed in TRAINING_SEEDS:
                valid = (model_seeds == seed) & np.isfinite(target_values)
                train, test = valid & ~held_out, valid & held_out
                if train.sum() < 2 or test.sum() < 2:
                    raise ValueError(
                        f"{seed} / {target} needs at least two valid rows in each partition "
                        f"(got {train.sum()} train and {test.sum()} test). Check observed target "
                        "coverage, simulation times and environment-seed split."
                    )
                predicted = fit_ridge(matrix, target_values, np.flatnonzero(train), np.flatnonzero(test))
                metrics = score(target_values[test], predicted)
                baseline_rmse = float(np.sqrt(np.mean(np.square(
                    target_values[test] - target_values[train].mean()
                ))))
                reduction = 100 * (1 - metrics["rmse"] / baseline_rmse) if baseline_rmse > 0 else None
                per_seed[seed] = {
                    **metrics,
                    "training_mean_baseline_rmse": baseline_rmse,
                    "rmse_reduction_percent": reduction,
                    "train_rows": int(train.sum()),
                    "test_rows": int(test.sum()),
                    "train_shifts": len({rows[i]["shift_id"] for i in np.flatnonzero(train)}),
                    "test_shifts": len({rows[i]["shift_id"] for i in np.flatnonzero(test)}),
                }
            r2_mean, r2_std = across_seeds(row["r2"] for row in per_seed.values())
            rmse_mean, rmse_std = across_seeds(row["rmse"] for row in per_seed.values())
            correlations = [row["pearson_r"] for row in per_seed.values() if row["pearson_r"] is not None]
            pearson_r_mean, pearson_r_std = across_seeds(correlations)
            reduction_mean, reduction_std = across_seeds(
                row["rmse_reduction_percent"] for row in per_seed.values()
            )
            results.append({
                "target": target, "representation": representation, "per_seed": per_seed,
                "r2_mean": r2_mean, "r2_std": r2_std,
                "rmse_mean": rmse_mean, "rmse_std": rmse_std,
                "pearson_r_mean": pearson_r_mean, "pearson_r_std": pearson_r_std,
                "pearson_r_n_seeds": len(correlations),
                "rmse_reduction_mean": reduction_mean, "rmse_reduction_std": reduction_std,
            })
    return results


def compute_projection(rows, matrices, count, random_seed, *, skip):
    if skip:
        return {"status": "skipped", "rows": []}
    try:
        from umap import UMAP
    except ImportError as exc:
        raise ValueError(
            "UMAP needs umap-learn in anylogic-env. Use --skip-umap for ridge metrics only."
        ) from exc

    eligible = [
        index for index, row in enumerate(rows)
        if number(row.get("grade_margin_pp")) is not None
        and number(row.get("total_shovel_queue")) is not None
    ]
    if len(eligible) <= 30:
        raise ValueError("UMAP with n_neighbors=30 needs at least 31 rows with both physical labels")
    rng = np.random.default_rng(random_seed)
    indices = np.sort(rng.choice(eligible, size=min(count, len(eligible)), replace=False))
    projections = {
        name: UMAP(
            n_neighbors=30, min_dist=0.1, n_components=2, random_state=random_seed, n_jobs=1,
        ).fit_transform(standardize(matrix[indices], None)[0])
        for name, matrix in matrices.items()
    }
    return {
        "status": "computed", "requested_rows": count, "eligible_rows": len(eligible),
        "n_neighbors": 30, "min_dist": 0.1,
        "rows": [
            {
                "training_seed": rows[index]["training_seed"], "shift_id": rows[index]["shift_id"],
                "decision_index": rows[index]["decision_index"],
                "grade_margin_pp": number(rows[index].get("grade_margin_pp")),
                "total_shovel_queue": rows[index]["total_shovel_queue"],
                **{name: points[position].tolist() for name, points in projections.items()},
            }
            for position, index in enumerate(indices)
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Collected rollout Parquet or NPZ")
    parser.add_argument("--output", type=Path, required=True, help="Destination linear_probing.json")
    parser.add_argument("--random-seed", type=int, default=128, help="Split and UMAP seed")
    parser.add_argument("--projection-count", type=int, default=5000)
    parser.add_argument("--maximum-target-lag-min", type=float, default=5.0)
    parser.add_argument("--skip-umap", action="store_true", help="Calculate ridge metrics only")
    args = parser.parse_args(argv)
    if args.maximum_target_lag_min < 0 or not math.isfinite(args.maximum_target_lag_min):
        parser.error("--maximum-target-lag-min must be finite and nonnegative")
    if not args.skip_umap and args.projection_count <= 30:
        parser.error("--projection-count must be at least 31")

    try:
        with TemporaryDirectory(prefix="bg128-linear-probing-") as directory:
            rows, matrices = stage_features(args.input, Path(directory))
            groups = group_rollouts(rows)
            test_seeds = choose_test_seeds(rows, args.random_seed)
            metrics = probe_metrics(rows, groups, matrices, test_seeds, args.maximum_target_lag_min)
            projection = compute_projection(rows, matrices, args.projection_count,
                                            args.random_seed, skip=args.skip_umap)
        write_json(args.output, {
            "schema": "bg128-linear-probing-v3",
            "representation": {"c128": "certified 128-dimensional engineering summary",
                               "raw": "1519-dimensional raw model observation"},
            "alpha": 1.0, "random_seed": args.random_seed,
            "holdout": "approximately 20% of environment seeds, kept together across all shifts "
                       "and training seeds",
            "test_environment_seeds": test_seeds,
            "ridge_preprocessing": "feature centering and scaling fitted only on training rows",
            "projection_preprocessing": "the same randomly sampled rows for both representations, "
                                        "scaled within each representation",
            "target_alignment": "most recent recorded observation at or before the target time "
                                "in the same shift, within maximum lag and observed shift coverage",
            "maximum_target_lag_min": args.maximum_target_lag_min,
            "target_grade_field": "grade_margin_pp",
            "target_queue_field": "total_shovel_queue",
            "queue_definition": "sum across the four shovels of distinct trucks in explicit queues and seize-wait lists",
            "aggregate": "unweighted training-seed mean and sample standard deviation",
            "pearson_definition": "correlation between held-out targets and Ridge predictions within each training seed, "
                                  "with null for constant inputs and seed summaries over defined correlations",
            "rmse_reduction_definition": "100 times (1 minus model RMSE divided by baseline RMSE) on "
                                         "the held-out rows. The constant baseline is the mean of "
                                         "the corresponding training targets. Zero baseline RMSE "
                                         "leaves the reduction undefined.",
            "metrics": metrics, "projection": projection,
        })
    except (KeyError, ValueError, OSError) as exc:
        parser.exit(1, f"{parser.prog}: {exc}\n")


if __name__ == "__main__":
    main()
