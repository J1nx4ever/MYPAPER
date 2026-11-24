"""Read-only snapshot timing on sealed nominal data, with a separate report."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import h5py
import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT))

from business_graph_summary_v1.admission import frozen_actor
from traditional_dispatch_v1.decision import POLICIES, choose
from traditional_dispatch_v1.runtime import _actor_choice


HISTORY = PROJECT / "runs/bg128-continuous-mixed-20261119-001/evaluation/nominal/new"
RUNS = (10001125, 10001126, 10001127)
POLICY_KEYS = (*POLICIES, "bg128")


def inputs():
    for run_index in RUNS:
        path = HISTORY / f"material/run_{run_index}.h5"
        with h5py.File(path, "r") as source:
            states = source["states"][:-1]
        window = np.rint(states[:, 2] * 4).astype(int)
        masks = states[:, 19:23].sum(axis=1)
        start = int(np.flatnonzero(masks >= 2)[0])
        yield run_index, start, states[start]
        for current in range(4):
            eligible = np.flatnonzero((window == current) & (masks >= 2))
            index = int(eligible[len(eligible) // 2])
            yield run_index, index, states[index]


def _observations():
    torch.set_num_threads(2)
    actor, _ = frozen_actor(HISTORY / "actor.pt")
    generator = torch.Generator(device="cpu").manual_seed(430125)
    rows = []
    for run_index, index, obs in inputs():
        outcomes = {}
        mask = obs[19:23]
        for policy in POLICY_KEYS:
            started = perf_counter()
            if policy == "bg128":
                choice = _actor_choice(actor, generator, obs)
                elapsed_ms = (perf_counter() - started) * 1000
                result = dict(action=choice, kind="actor", timed_out=False)
            else:
                choice = choose(policy, obs, mask)
                elapsed_ms = (perf_counter() - started) * 1000
                result = vars(choice).copy()
            if not mask[result["action"]]:
                raise RuntimeError(f"illegal replay action for {policy}, run {run_index}, frame {index}")
            outcomes[policy] = dict(result, elapsed_ms=elapsed_ms)
        fifo, rolling = outcomes["fifo_greedy"], outcomes["rolling"]
        if fifo["kind"] == rolling["kind"] == "planned":
            if fifo["selected_cost"] != rolling["initial_cost"]:
                raise RuntimeError("rolling and FIFO root comparison disagrees on the same snapshot")
            if rolling["selected_cost"] > rolling["initial_cost"]:
                raise RuntimeError("rolling returned a worse candidate than its FIFO incumbent")
        rows.append(dict(run_index=run_index, frame_index=index, window=int(round(obs[2] * 4)),
                         observation_sha256=hashlib.sha256(obs.tobytes()).hexdigest(), policies=outcomes))
    return rows


def report():
    rows = _observations()
    summary = {}
    for policy in POLICY_KEYS:
        results = [row["policies"][policy] for row in rows]
        latencies = [row["elapsed_ms"] for row in results]
        summary[policy] = dict(
            decisions=dict(Counter(row["kind"] for row in results)), max_ms=max(latencies),
            p50_ms=float(np.percentile(latencies, 50)), p95_ms=float(np.percentile(latencies, 95)),
            p99_ms=float(np.percentile(latencies, 99)), timeouts=sum(row["timed_out"] for row in results),
            fallback_reasons=dict(Counter(row["fallback_reason"] for row in results if row.get("fallback_reason"))),
            requests_with_multistep=sum(bool(row.get("multi_step_plans")) for row in results),
            scored_depths=dict(Counter(row["max_scored_depth"] for row in results if "max_scored_depth" in row)),
        )
    ready = (
        all(row["max_ms"] <= 1000 for row in summary.values())
        and all(summary[policy]["decisions"].get("planned", 0) == len(rows) for policy in POLICIES)
        and summary["rolling"]["requests_with_multistep"] > 0
    )
    return dict(scope="sealed_nominal_snapshot_replay_no_anylogic", wall_clock_budget_ms=800,
                response_target_ms=1000, snapshot_count=len(rows), snapshot_runs=list(RUNS),
                ready=ready, summary=summary, snapshots=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = report()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps(dict(ready=result["ready"], snapshot_count=result["snapshot_count"],
                          summary=result["summary"]), indent=2, allow_nan=False))
    return 0 if result["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
