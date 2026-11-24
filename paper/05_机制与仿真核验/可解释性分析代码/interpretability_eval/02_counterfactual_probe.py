"""Replay three observation-level interventions with the frozen BG128 models."""
from __future__ import annotations

import argparse
import hashlib
from heapq import heappop, heappush
from itertools import chain
import math
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t
import torch

from _shared import iter_rows, load_model, total_shovel_queue, write_json

TRAINING_SEEDS = ("s01", "s02", "s03")
STRATA = ("normal", "grade_critical", "congestion")
GRID = {
    "A": np.linspace(-0.25, 0.35, 25),
    "B": np.arange(6, dtype=np.float64),
    "C": np.arange(120, 14, -5, dtype=np.float64),
}
METRICS = {
    "A": ("high_grade_probability", "critic_value"),
    "B": ("site_probability",),
    "C": ("waste_probability",),
}

# Wire coordinates in dispatch.physical.v3, as consumed by the frozen C128.
SHIFT_ELAPSED, SHIFT_REMAINING, WINDOW_INDEX, WINDOW_REMAINING = range(4)
SHIFT_CLOCK_COLUMNS = (SHIFT_ELAPSED, SHIFT_REMAINING, WINDOW_REMAINING)
WINDOW_TONS, WINDOW_METAL = 4, 5
SHIFT_ORE, WASTE_DELIVERED, WASTE_SHORTFALL = 7, 9, 10
ORE_TARGET, WASTE_TARGET, GRADE_LOWER, TERMINAL = 11, 12, 13, 17
MASK = slice(19, 23)
ACTION_SIDECAR = slice(19, 27)
TRUCKS = slice(27, 1107)
LOADS = slice(1107, 1295)
ROADS = slice(1439, 1519)
LOAD_QUEUE_AND_SEIZE = slice(20, 38)
INCOMING_COUNT = 17
SHIFT_MINUTES, WINDOW_MINUTES = 480, 120
ORE_SCALE, WASTE_SCALE, RESERVE_SCALE, TRUCK_SCALE = 20000, 4000, 30000, 193
LOAD_TARGET = slice(26, 30)
ACTIVE_ROAD = slice(45, 61)
ROUTE_ORDER = slice(63, 79)
EGOVEC, TO_LOAD, CAPACITY, LOADED, ORE, RESERVED = 0, 6, 16, 17, 18, 20


def stratum(row):
    grade = row.get("grade_margin_pp")
    queue = total_shovel_queue(row)
    if grade is None or queue is None:
        return None
    if float(grade) <= 0.05:
        return "grade_critical"
    if queue >= 2:
        return "congestion"
    if float(grade) > 0.15:
        return "normal"
    return None


def route_to_load(roads, current_road, target):
    """Continue along the active road before choosing the shortest open tail."""
    from business_graph_summary_v1.graph import PHYSICAL_NODE_TO_TOKEN, ROAD_ENDPOINTS

    origin = ROAD_ENDPOINTS[current_road][1]
    terminal = PHYSICAL_NODE_TO_TOKEN.index(9 + target)
    frontier = [(0., (), origin)]
    visited = set()
    while frontier:
        distance, tail, node = heappop(frontier)
        if node == terminal:
            return (current_road, *tail)
        if node in visited:
            continue
        visited.add(node)
        for road, (start, end) in enumerate(ROAD_ENDPOINTS):
            length, _, closed = roads[road, 2:5]
            if road != current_road and start == node and end not in visited and not closed:
                heappush(frontier, (distance + float(length), (*tail, road), end))
    return None


def move_reservation(state, target, move):
    truck_id, source, route = move
    trucks = state[TRUCKS].reshape(9, 120)
    loads = state[LOADS].reshape(4, 47)
    roads = state[ROADS].reshape(16, 5)
    truck = trucks[truck_id]
    quantity = float(truck[RESERVED]) * TRUCK_SCALE / RESERVE_SCALE
    truck[LOAD_TARGET] = 0
    truck[LOAD_TARGET.start + target] = 1
    truck[ROUTE_ORDER] = 0
    for rank, road in enumerate(route, start=1):
        truck[ROUTE_ORDER.start + road] = rank / 16
    truck[62] = float(truck[61]) + sum(float(roads[road, 2]) for road in route[1:])
    for site, direction in ((source, -1), (target, 1)):
        reserve = float(loads[site, 6]) + direction * quantity
        incoming = round(float(loads[site, INCOMING_COUNT]) * 9) + direction
        if reserve < -1e-7 or reserve > float(loads[site, 5]) + 1e-7 or not 0 <= incoming <= 8:
            return False
        loads[site, 6] = max(0., reserve)
        loads[site, 7] = max(0., float(loads[site, 5]) - float(loads[site, 6]))
        loads[site, INCOMING_COUNT] = incoming / 9
        # The source and target are legal before the move. Holding the mask and
        # candidate tonnage requires enough unreserved material for the ego.
        capacity = float(trucks[int(trucks[:, EGOVEC].argmax()), CAPACITY])
        after = min(float(loads[site, 7]) * RESERVE_SCALE / TRUCK_SCALE, capacity)
        if after <= 0 or not math.isclose(after, float(state[23 + site]), abs_tol=8e-6):
            return False
    return True


def reservation_sweep(raw, possible_targets, legal):
    """Find a physical, nested 0–5 sequence using only genuine en-route empty trucks."""
    trucks = raw[TRUCKS].reshape(9, 120)
    loads = raw[LOADS].reshape(4, 47)
    roads = raw[ROADS].reshape(16, 5)
    owners = set(np.rint(loads[:, 20:47].ravel() * 9).astype(int) - 1)
    reserved = (trucks[:, EGOVEC] == 0) & (trucks[:, LOADED] == 0) & (trucks[:, RESERVED] > 0)
    enroute = reserved & (trucks[:, TO_LOAD] == 1) & (trucks[:, ACTIVE_ROAD].sum(axis=1) == 1)
    if enroute.sum() < int(GRID["B"][-1]):
        return None, ()
    for target in possible_targets:
        if (round(float(loads[target, INCOMING_COUNT]) * 9)
                or np.any(reserved & (trucks[:, LOAD_TARGET.start + target] == 1))):
            continue
        changed = raw.copy()
        moves = []
        for truck_id in np.flatnonzero(enroute & (trucks[:, ORE] == loads[target, 1])):
            truck = trucks[truck_id]
            sources = np.flatnonzero(truck[LOAD_TARGET])
            if truck_id in owners or len(sources) != 1 or int(sources[0]) not in legal:
                continue
            source = int(sources[0])
            route = route_to_load(roads, int(truck[ACTIVE_ROAD].argmax()), target)
            if route is None:
                continue
            trial = changed.copy()
            move = int(truck_id), source, route
            if move_reservation(trial, target, move):
                changed = trial
                moves.append(move)
            if len(moves) == int(GRID["B"][-1]):
                return target, tuple(moves)
    return None, ()


def candidate(row, *, probe_only=None):
    if row["dispatch_kind"] != "LOAD" or row.get("raw_obs") is None:
        return None
    label = stratum(row)
    if label is None:
        return None
    raw = np.asarray(row["raw_obs"], dtype=np.float32)
    if raw.shape != (1519,) or not np.isfinite(raw).all() or raw[TERMINAL]:
        raise ValueError(f"invalid decision input at {row['shift_id']} / {row['decision_index']}")
    telemetry_mask = np.asarray(row["action_mask"], dtype=np.float32)
    if not np.array_equal(raw[MASK], telemetry_mask):
        raise ValueError(f"action mask mismatch at {row['shift_id']} / {row['decision_index']}")

    sites = {}
    for site in row["shovels"]:
        action = int(site["action"])
        if not 0 <= action < 4 or action in sites:
            raise ValueError(f"invalid load action map at {row['shift_id']} / {row['decision_index']}")
        sites[action] = site
    legal = set(np.flatnonzero(telemetry_mask).tolist())
    if not legal <= sites.keys():
        raise ValueError(f"missing legal load site at {row['shift_id']} / {row['decision_index']}")
    high = [action for action in sorted(legal) if sites[action]["is_high_grade"]]
    waste = [action for action in sorted(legal) if sites[action]["is_waste"]]
    loads = raw[LOADS].reshape(4, 47)
    empty_queue = [action for action in sorted(legal)
                   if sites[action].get("queue") == 0
                   and not loads[action, LOAD_QUEUE_AND_SEIZE].any()]
    probes = []
    if probe_only is None and high and raw[WINDOW_TONS] > 0:
        probes.append("A")
    # Rotate simultaneous sites without consulting the executed action.
    if empty_queue and probe_only in (None, "B"):
        rotation = int(row["decision_index"]) % len(empty_queue)
        action_b, moves = reservation_sweep(raw, empty_queue[rotation:] + empty_queue[:rotation], legal)
    else:
        action_b, moves = None, ()
    if action_b is not None:
        probes.append("B")
    if (probe_only in (None, "C") and waste and raw[SHIFT_ORE] >= raw[ORE_TARGET] > 0
            and 0 <= raw[WASTE_DELIVERED] < raw[WASTE_TARGET] and raw[WASTE_SHORTFALL] > 0
            and raw[WINDOW_INDEX] == 0.75
            and math.isclose(raw[SHIFT_ELAPSED] + raw[SHIFT_REMAINING], 1., abs_tol=1e-6)
            and math.isclose(float(raw[WINDOW_REMAINING]), float(raw[SHIFT_REMAINING]) * 4,
                             abs_tol=1e-6)):
        probes.append("C")
    if not probes:
        return None
    fields = ("training_seed", "shift_id", "seed", "scenario", "sim_time_min",
              "decision_index", "grade_margin_pp")
    return {"row": {key: row[key] for key in fields}, "raw": raw, "stratum": label,
            "high": high, "waste": waste, "action_b": action_b, "moves": moves, "probes": probes}


def select_snapshots(rows, count, random_seed, *, b_rows=(), c_rows=()):
    rng = np.random.default_rng(random_seed)
    eligible_counts = {probe: dict.fromkeys(STRATA, 0) for probe in GRID}
    reservoirs = {probe: {label: [] for label in STRATA} for probe in GRID}
    refs = {}
    seen = {"B": {}, "C": {}}
    inputs = chain(((row, None) for row in rows), ((row, "B") for row in b_rows),
                   ((row, "C") for row in c_rows))
    for row, probe_only in inputs:
        if row["policy"] != "BG128":
            continue
        seed = row["training_seed"]
        ref = Path(row["checkpoint_path"]), row["checkpoint_sha256"]
        if refs.setdefault(seed, ref) != ref:
            raise ValueError(f"{seed} rows refer to multiple BG128 checkpoints")
        snapshot = candidate(row, probe_only=probe_only)
        if snapshot is None:
            continue
        label = snapshot["stratum"]
        for probe in snapshot["probes"]:
            if probe != "A":
                identity = (seed, row["shift_id"], row["decision_index"])
                digest = hashlib.sha256(snapshot["raw"].tobytes()).hexdigest()
                if identity in seen[probe]:
                    if seen[probe][identity] != digest:
                        raise ValueError(f"Conflicting Intervention {probe} snapshot {identity}")
                    continue
                seen[probe][identity] = digest
            eligible_counts[probe][label] += 1
            reservoir = reservoirs[probe][label]
            if len(reservoir) < count:
                reservoir.append(snapshot)
            else:
                replacement = rng.integers(eligible_counts[probe][label])
                if replacement < count:
                    reservoir[replacement] = snapshot
    if set(refs) != set(TRAINING_SEEDS) or len({ref[1] for ref in refs.values()}) != 3:
        raise ValueError("M3 requires distinct frozen BG128 checkpoints for s01, s02, and s03")
    selected = {}
    for probe, strata in reservoirs.items():
        chosen = []
        for samples in strata.values():
            rng.shuffle(samples)
        available = min(count, sum(len(samples) for samples in strata.values()))
        while len(chosen) < available:
            for samples in strata.values():
                if samples and len(chosen) < available:
                    chosen.append(samples.pop())
        if chosen:
            selected[probe] = chosen
    if not selected:
        raise ValueError("No BG128 observations satisfy the three intervention definitions")
    return selected, refs, eligible_counts


def verify_commitment_sweep(states, snapshot):
    from business_graph_summary_v1.graph import NUM_NODES, RELATION_NAMES, build_graph

    graph = build_graph(np.ascontiguousarray(states))
    source, dest = graph.edge_index.numpy()
    relation = graph.edge_type.numpy()
    forward_id = RELATION_NAMES.index('reserved_load')
    reverse_id = RELATION_NAMES.index('reserved_load_reverse')
    target = snapshot["action_b"] + 9
    selected = [move[0] for move in snapshot["moves"]]
    verified = []
    for count in range(len(states)):
        load = count * NUM_NODES + target
        incoming = sorted((source[(dest == load) & (relation == forward_id)] % NUM_NODES).tolist())
        outgoing = sorted((dest[(source == load) & (relation == reverse_id)] % NUM_NODES).tolist())
        expected = sorted(selected[:count])
        serialized = round(float(states[count, LOADS.start + snapshot["action_b"] * 47
                                         + INCOMING_COUNT]) * 9)
        if incoming != expected or outgoing != expected or serialized != count:
            raise RuntimeError(f"Intervention B graph differs from the requested {count} commitments")
        verified.append({"incoming_count": count, "reserved_load_from_trucks": incoming,
                         "reserved_load_reverse_to_trucks": outgoing})
    return verified


def shift_clocks(remaining_minutes):
    remaining = np.asarray(remaining_minutes, dtype=np.float64)
    return np.stack(((SHIFT_MINUTES - remaining) / SHIFT_MINUTES,
                     remaining / SHIFT_MINUTES, remaining / WINDOW_MINUTES), axis=-1).astype(np.float32)


def interventions(snapshots, probe):
    x_values = GRID[probe]
    original = np.stack([snapshot["raw"] for snapshot in snapshots])
    modified = np.repeat(original[:, None, :], len(x_values), axis=1)
    if probe == "A":
        grade = original[:, GRADE_LOWER, None].astype(np.float64) + x_values[None, :] / 100.0
        if (grade < 0.0).any() or (grade > 1.0).any():
            raise ValueError("grade-margin sweep would leave the calibrated grade domain")
        modified[:, :, WINDOW_METAL] = original[:, WINDOW_TONS, None] * grade
    elif probe == "B":
        for i, snapshot in enumerate(snapshots):
            for point, move in enumerate(snapshot["moves"], start=1):
                modified[i, point] = modified[i, point - 1]
                if not move_reservation(modified[i, point], snapshot["action_b"], move):
                    raise RuntimeError("Intervention B replay could not reproduce an admissible move")
            snapshot["verified_commitments"] = verify_commitment_sweep(modified[i], snapshot)
    else:
        # Physical V3 uses denominators 28,800 s and 7,200 s respectively.
        # Keep the raw clocks consistent for the C128 branch as well.
        modified[:, :, SHIFT_CLOCK_COLUMNS] = shift_clocks(x_values)
    if not np.array_equal(modified[:, :, ACTION_SIDECAR],
                          np.repeat(original[:, None, ACTION_SIDECAR], len(x_values), axis=1)):
        raise RuntimeError("counterfactual changed a frozen action mask or dispatch quantity")
    return modified


def shift_node_logits(model, states, prefix, clocks):
    """Intervene on the Global node features entering the frozen token projection."""
    from business_graph_summary_v1.graph import NODE_NAMES

    encoder = model.actor.state_encoder
    token_encoder = encoder.token_encoder
    node = NODE_NAMES.index("Global")
    trace = {"layers": set()}

    def inject(_module, args):
        (features,) = args
        if features.shape != (len(clocks), 19) or features.dtype != clocks.dtype:
            raise RuntimeError("Unexpected Global node feature shape or dtype")
        trace["raw_matches"] = int((features[:, SHIFT_CLOCK_COLUMNS] == clocks).all(dim=1).sum())
        shifted = features.clone()
        shifted[:, SHIFT_CLOCK_COLUMNS] = clocks
        return (shifted,)

    def confirm_projection(_module, args, _output):
        if not torch.equal(args[0][:, SHIFT_CLOCK_COLUMNS], clocks):
            raise RuntimeError("Frozen Global projection did not receive the injected clocks")
        trace["projection_received"] = True

    def capture_token(_module, _args, output):
        trace["token"] = output

    def confirm_layer(module, args):
        tokens, graph = args
        if "token" not in trace or not torch.equal(tokens[:, node], trace["token"]):
            raise RuntimeError("Projected Global token did not reach the frozen R-GAT layer")
        if (graph.edge_index.remainder(len(NODE_NAMES)) == node).any():
            raise RuntimeError("Unexpected physical edge on the frozen Global bypass")
        trace["layers"].add(id(module))

    handles = [token_encoder.global_final.register_forward_pre_hook(inject),
               token_encoder.global_final.register_forward_hook(confirm_projection),
               token_encoder.global_ln.register_forward_hook(capture_token)]
    handles.extend(layer.register_forward_pre_hook(confirm_layer) for layer in encoder.graph_layers)
    try:
        logits = model.actor(states, prefix)
        if not trace.get("projection_received") or trace["layers"] != {id(layer) for layer in encoder.graph_layers}:
            raise RuntimeError("Global shift-node verification did not observe every graph layer")
        return logits, trace["token"].numpy(), trace["raw_matches"]
    finally:
        for handle in handles:
            handle.remove()


def infer(model, modified, batch_size, *, probe):
    # BG128 uses certified current rows only, with no recurrent history.
    from business_graph_summary_v1.features_cache import pack_summary_prefixes

    n_snapshots, n_points, width = modified.shape
    decision_rows = modified.reshape(-1, width)
    probs = []
    values = []
    grid_clocks = shift_clocks(GRID["C"]) if probe == "C" else None
    intervention_clocks = np.tile(grid_clocks, (n_snapshots, 1)) if grid_clocks is not None else None
    global_tokens, raw_matches = [], 0
    with torch.inference_mode():
        for i in range(0, len(decision_rows), batch_size):
            rows = decision_rows[i:i + batch_size]
            prefix = pack_summary_prefixes(rows, feature_config=model.actor.feature_config)
            states = torch.from_numpy(rows)
            if intervention_clocks is None:
                logits = model.actor(states, prefix)
            else:
                clocks = torch.from_numpy(intervention_clocks[i:i + len(rows)])
                logits, token, matches = shift_node_logits(model, states, prefix, clocks)
                global_tokens.append(token)
                raw_matches += matches
            batch_probs = torch.softmax(logits, dim=-1).numpy()
            if np.any(batch_probs[~rows[:, MASK].astype(bool)] != 0.0):
                raise RuntimeError("frozen Actor assigned probability to an illegal action")
            probs.append(batch_probs)
            if probe == "A":
                values.append(model.critic(states, prefix).numpy())
    probs = np.concatenate(probs).reshape(n_snapshots, n_points, 4)
    values = np.concatenate(values).reshape(n_snapshots, n_points) if probe == "A" else None
    verified = None
    if global_tokens:
        tokens = np.concatenate(global_tokens).reshape(n_snapshots, n_points, -1)
        distance = np.linalg.norm(tokens[:, -1] - tokens[:, 0], axis=-1)
        verified = {
            "verified_rows": len(decision_rows),
            "projection_rows_already_matching_from_raw": raw_matches,
            "snapshots_with_changed_endpoint_token": int(np.count_nonzero(distance)),
            "mean_endpoint_token_l2": float(distance.mean()),
            "min_endpoint_token_l2": float(distance.min()),
            "max_endpoint_token_l2": float(distance.max()),
        }
    return probs, values, verified


def responses(model, modified, selected, batch_size):
    result = {}
    shift_verification = None
    for probe, states in modified.items():
        snapshots = selected[probe]
        probs, values, verified = infer(model, states, batch_size, probe=probe)
        if probe == "A":
            result[probe] = {
                "high_grade_probability": np.stack([
                    probs[i][:, snap["high"]].sum(axis=-1) for i, snap in enumerate(snapshots)
                ]),
                "critic_value": values,
            }
        elif probe == "B":
            result[probe] = {
                "site_probability": np.stack([
                    probs[i, :, snap["action_b"]] for i, snap in enumerate(snapshots)
                ]),
            }
        else:
            result[probe] = {
                "waste_probability": np.stack([
                    probs[i][:, snap["waste"]].sum(axis=-1) for i, snap in enumerate(snapshots)
                ]),
            }
            shift_verification = verified
    return result, shift_verification


def aggregate(scores, probes):
    result = []
    coefficient = float(student_t.ppf(0.975, len(TRAINING_SEEDS) - 1))
    for probe in probes:
        for metric in METRICS[probe]:
            units = np.stack([scores[seed][probe][metric] for seed in TRAINING_SEEDS])
            seed_means = units.mean(axis=1, dtype=np.float64)
            for i, point in enumerate(GRID[probe]):
                seed_values = seed_means[:, i]
                mean = float(seed_values.mean())
                std = float(seed_values.std(ddof=1))
                half_width = coefficient * std / math.sqrt(len(TRAINING_SEEDS))
                result.append({
                    "probe": probe, "metric": metric, "x": float(point),
                    "mean": mean, "std": std, "n_seeds": len(TRAINING_SEEDS),
                    "n_snapshots": int(units.shape[1]),
                    "snapshot_std": float(units[:, :, i].mean(axis=0).std(ddof=1))
                    if units.shape[1] > 1 else None,
                    "ci95_low": mean - half_width, "ci95_high": mean + half_width,
                    "per_seed": dict(zip(TRAINING_SEEDS, seed_values.tolist())),
                })
    return result


def snapshot_manifest(probe, snapshots):
    result = []
    for item in snapshots:
        row, raw = item["row"], item["raw"]
        record = {
            **row, "probe": probe, "stratum": item["stratum"],
            "observation_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
            "fixed_window_grade_margin_pp": 100. * (
                float(raw[WINDOW_METAL]) / float(raw[WINDOW_TONS]) - float(raw[GRADE_LOWER])
            ) if raw[WINDOW_TONS] > 0 else None,
            "action_mask": raw[MASK].astype(int).tolist(),
            "high_grade_actions": item["high"], "waste_actions": item["waste"],
            "probe_b_action": item["action_b"],
        }
        if probe == "B":
            record["rerouted_trucks"] = [
                {"truck_id": truck, "original_load_action": load, "new_remaining_route_roads": list(route)}
                for truck, load, route in item["moves"]
            ]
            record["graph_verification"] = item["verified_commitments"]
        elif probe == "C":
            record.update(observed_ore_t=float(raw[SHIFT_ORE]) * ORE_SCALE,
                          ore_target_t=float(raw[ORE_TARGET]) * ORE_SCALE,
                          observed_waste_t=float(raw[WASTE_DELIVERED]) * WASTE_SCALE,
                          waste_target_t=float(raw[WASTE_TARGET]) * WASTE_SCALE,
                          observed_remaining_min=float(raw[SHIFT_REMAINING]) * SHIFT_MINUTES)
        result.append(record)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Collected rollout Parquet or NPZ")
    parser.add_argument("--output", type=Path, required=True, help="Destination counterfactual.json")
    parser.add_argument("--b-input", type=Path, help="Optional extra authenticated observations for B only")
    parser.add_argument("--c-input", type=Path, help="Optional extra authenticated observations for C only")
    parser.add_argument("--snapshot-count", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)
    if args.snapshot_count < 3:
        parser.error("--snapshot-count must be at least 3")
    if not 1 <= args.batch_size <= 4096:
        parser.error("--batch-size must be between 1 and 4096")
    try:
        selected, checkpoints, eligible_counts = select_snapshots(
            iter_rows(args.input), args.snapshot_count, args.random_seed,
            b_rows=iter_rows(args.b_input) if args.b_input else (),
            c_rows=iter_rows(args.c_input) if args.c_input else (),
        )
        modified = {probe: interventions(snapshots, probe) for probe, snapshots in selected.items()}
        scores = {}
        shift_verification = {}
        torch.set_num_threads(1)
        for seed in TRAINING_SEEDS:
            model = load_model(*checkpoints[seed])
            model.actor.eval()
            model.critic.eval()
            scores[seed], verified = responses(model, modified, selected, args.batch_size)
            if verified is not None:
                shift_verification[seed] = verified
        coverage = {}
        for probe, eligible in eligible_counts.items():
            collected = dict.fromkeys(STRATA, 0)
            for snapshot in selected.get(probe, []):
                collected[snapshot["stratum"]] += 1
            total = sum(collected.values())
            missing = [label for label in STRATA if collected[label] == 0]
            status = ("unavailable" if total == 0 else
                      "partial" if total < args.snapshot_count or missing else "computed")
            coverage[probe] = {"status": status, "eligible": eligible, "selected": collected,
                               "selected_shifts": len({snap["row"]["shift_id"]
                                                       for snap in selected.get(probe, [])}),
                               "missing_strata": missing}
        write_json(args.output, {
            "schema": "bg128-counterfactual-v3", "requested_snapshots_per_probe": args.snapshot_count,
            "random_seed": args.random_seed,
            "coverage": coverage,
            "checkpoints": {seed: {"path": str(path), "sha256": digest}
                            for seed, (path, digest) in checkpoints.items()},
            "target_units": {"A": "grade percentage points", "B": "incoming truck count",
                             "C": "shift minutes remaining"},
            "shift_node_verification": {
                "node": "Global",
                "feature_projection": "actor.state_encoder.token_encoder.global_final",
                "clock_columns": list(SHIFT_CLOCK_COLUMNS),
                "input_remaining_minutes": GRID["C"].tolist(),
                "normalized_shift_remaining": shift_clocks(GRID["C"])[:, 1].tolist(),
                "global_token_reaches_both_graph_layers_unchanged": bool(shift_verification),
                "has_physical_message_edges": False if shift_verification else None,
                "seeds": shift_verification,
            },
            "method": {
                "scope": "synthetic observations with frozen masks and dispatch quantities",
                "sampling": "separate admissible pools per probe, uniform decision reservoirs "
                            "within exclusive regimes, and round-robin regime allocation up to "
                            "the requested count. Each probe's sample is shared across frozen models.",
                "A": "hold current fixed-window ore mass; set its metal mass from the swept "
                     "grade margin. Rolling telemetry is used for sample selection.",
                "B": "start from a legal, empty and uncommitted shovel; redirect five existing "
                     "same-material and not-yet-served inbound empty trucks in a nested sequence. "
                     "Preserve each active segment, recompute its open route tail and move station "
                     "reservation mass and count. Verify the frozen builder gives exactly 0–5 "
                     "reserved_load edges and matching reverses; keep legal actions and tons fixed.",
                "C": "require the recorded final-window state already meets its configured ore target "
                     "and has positive waste shortfall with a legal waste action. Preserve observed "
                     "production and vary the compatible clocks using a 480-minute shift and "
                     "120-minute window denominators. Inject these normalized features directly "
                     "at the frozen Global token projection and verify that the result reaches "
                     "both R-GAT layers unchanged. The isolated Global token is read out by the "
                     "Actor context; the same clock values feed C128.",
                "critic": "scalar joint cost value",
                "aggregate": "sample means within each model, then equal-weight seed mean, "
                             "sample standard deviation and two-sided t interval across models",
            },
            "snapshots": [item for probe, snapshots in selected.items()
                          for item in snapshot_manifest(probe, snapshots)],
            "curves": aggregate(scores, selected),
        })
    except (KeyError, ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"{parser.prog}: {exc}\n")


if __name__ == "__main__":
    main()
