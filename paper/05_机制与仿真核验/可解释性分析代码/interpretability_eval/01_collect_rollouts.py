"""Export read-only historical rollouts and capture frozen BG128 inference diagnostics."""
import argparse
from itertools import islice
import math
from pathlib import Path

import numpy as np

from _shared import (GROUPS, POLICIES, SEEDS, iter_rows, load_model, read_json,
                     total_shovel_queue, verify_file, write_rows)


def diagnostics(model, observations):
    """Instrument one inference call. The input rows and legal actions stay factual."""
    import torch

    from business_graph_summary_v1.graph import NODE_NAMES
    from business_graph_summary_v1.features_cache import pack_summary_prefixes

    states = torch.from_numpy(np.ascontiguousarray(observations, dtype=np.float32))
    captured = {}

    def save_actor(_module, _inputs, result):
        captured["detail"] = result[1]

    with torch.inference_mode():
        prefixes = pack_summary_prefixes(states, feature_config=model.actor.feature_config)
        hook = model.actor.register_forward_hook(save_actor)
        try:
            logits, _ = model.actor(states, prefixes, diagnostic=True)
            values = model.critic(states, prefixes)
        finally:
            hook.remove()
        detail = captured.pop("detail")
        probabilities = logits.softmax(dim=1).tolist()
        graph = detail["graph"]
        last_layer = detail["layer2"]
        output_weight = model.actor.state_encoder.graph_layers[1].output.weight
        group_messages = last_layer["group_messages"]
        group_presence = last_layer["group_masks"]
        by_group = []
        for group in range(len(GROUPS)):
            w = output_weight[:, group * 64:(group + 1) * 64]
            message = group_messages[:, :, group] @ w.T
            presence = group_presence[:, :, group, None] * output_weight[:, 256 + group]
            by_group.append((message + presence).norm(dim=-1))
        projected_norms = torch.stack(by_group, dim=-1).tolist()
        edge_weights = {}
        for layer in (1, 2):
            edge_weights[layer] = detail[f"layer{layer}"]["weights"].tolist()

    tensor_ego = graph.ego_ids.tolist()
    group_edge_records = [[] for _ in range(len(states))]
    group_layer1 = [[] for _ in range(len(states))]
    group_layer2 = [[] for _ in range(len(states))]
    from business_graph_summary_v1.graph import RELATION_NAMES
    for index, ((src, dst), relation, group) in enumerate(zip(
            graph.edge_index.T.tolist(), graph.edge_type.tolist(), graph.edge_group.tolist())):
        sample, src = divmod(src, len(NODE_NAMES))
        group_edge_records[sample].append(dict(
            src=src, dst=dst % len(NODE_NAMES), group=GROUPS[group], relation=RELATION_NAMES[relation]
        ))
        group_layer1[sample].append(edge_weights[1][index])
        group_layer2[sample].append(edge_weights[2][index])
    result = []
    for index, (summary, value) in enumerate(zip(detail["current_summary"].tolist(), values.tolist())):
        norms = {str(node): dict(zip(GROUPS, group_values))
                 for node, group_values in enumerate(projected_norms[index])}
        result.append(dict(summary_emb=summary, critic_value=value,
                           action_probs=probabilities[index],
                           edge_attn_l1=group_layer1[index], edge_attn_l2=group_layer2[index],
                           edges=group_edge_records[index], ego_node=tensor_ego[index],
                           edge_group_contribution_l2=norms[str(tensor_ego[index])],
                           node_group_contribution_l2=norms))
    return result


def distance_matrix(state):
    roads = state[1439:1519].reshape(16, 5)
    distance = np.full((9, 9), np.inf)
    np.fill_diagonal(distance, 0)
    for start, end, length, _, closed in roads:
        if not closed:
            i, j = round(float(start) * 9) - 1, round(float(end) * 9) - 1
            distance[i, j] = min(distance[i, j], float(length) * 10)
    for through in range(9):
        distance = np.minimum(distance, distance[:, through, None] + distance[None, through, :])
    return distance


def physical_fields(state, time_s, amounts, grade_lower):
    trucks = state[27:1107].reshape(9, 120)
    loads = state[1107:1295].reshape(4, 47)
    dump = state[1295:1343]
    ego_id = int(np.flatnonzero(trucks[:, 0])[0])
    ego = trucks[ego_id]
    ego_nodes = np.flatnonzero(ego[32:41])
    distances = distance_matrix(state)
    q_roll, m_roll = amounts
    if not math.isclose(float(state[13]), grade_lower, abs_tol=1e-7, rel_tol=0):
        raise ValueError("Observation grade lower bound differs from validated RPC configuration")
    grade_min = grade_lower * 100
    roll_grade = 100 * m_roll / q_roll if q_roll > 1e-8 else None
    inbound = (trucks[:, 17] == 1) & (trucks[:, 21] == 1) & (trucks[:, 30] == 1)
    inbound_tons = trucks[inbound, 19].astype(float) * 193
    inbound_metal = float(inbound_tons @ trucks[inbound, 22].astype(float))
    combined_tons = q_roll + float(inbound_tons.sum())
    expected = 100 * (m_roll + inbound_metal) / combined_tons if combined_tons > 1e-8 else None
    grade_high = max((float(load[2]) for load in loads if load[1]), default=None)
    shovels = []
    for action, load in enumerate(loads):
        node = action + 5
        shovels.append(dict(
            action=action, site_id=f"Load{action}", node_id=action + 9,
            queue=int(np.count_nonzero(load[20:29])),
            commitment=int(round(float(load[17]) * 9)),
            grade_pp=float(load[2]) * 100 if load[1] else None,
            distance_km=finite_distance(distances[ego_nodes[0], node]) if len(ego_nodes) else None,
            haul_distance_km=finite_distance(distances[node, 0 if load[1] else 1]),
            is_high_grade=bool(load[1] and float(load[2]) == grade_high),
            is_waste=not bool(load[1]),
        ))
    return dict(
        sim_time_min=time_s / 60, dispatch_kind="LOAD", truck_id=ego_id,
        truck_type=("Cat777", "Cat785", "Cat789")[int(ego[1:4].argmax())],
        truck_capacity_t=float(ego[16]) * 193, cargo_t=float(ego[19]) * 193,
        grade_roll_pp=roll_grade, grade_min_pp=grade_min,
        grade_margin_pp=roll_grade - grade_min if roll_grade is not None else None,
        grade_expected_pp=expected, grade_window_pp=float(state[5]) / float(state[4]) * 100
        if state[4] > 0 else None,
        total_shovel_queue=total_shovel_queue({"raw_obs": state}),
        crusher_queue=int(np.count_nonzero(dump[12:21])),
        crusher_capacity_wait=int(np.count_nonzero(dump[30:39])),
        remaining_min=float(state[1]) * 480, ore_t=float(state[7]) * 20000,
        waste_t=float(state[9]) * 4000, shovels=shovels,
        action_mask=state[19:23].astype(bool).tolist(), raw_obs=state.tolist()
    )


def finite_distance(value):
    return float(value) if math.isfinite(value) else None


def rolling_amounts(process, time_s):
    """Mass in the preceding two hours from causally visible PROCESS intervals."""
    lo = max(0, time_s - 7200)
    keep = [event for event in process if event.interval_end > lo]
    fractions = [max(0, min(time_s, event.interval_end) - max(lo, event.interval_start))
                 / (event.interval_end - event.interval_start) for event in keep]
    return keep, (math.fsum(event.tons * fraction for event, fraction in zip(keep, fractions)),
                  math.fsum(event.metal * fraction for event, fraction in zip(keep, fractions)))


def annotate_grade_margin_plus20(rows, memory):
    """Reconstruct the exact t+20 physical window from the admitted PROCESS journal."""
    process = [event for event in memory.events if event.kind == "PROCESS"]
    end_s = max(event.time for event in memory.events)
    for row in rows:
        target_s = (row["sim_time_min"] + 20) * 60
        margin = None
        if target_s <= end_s + 1e-7:
            _, (tons, metal) = rolling_amounts(process, target_s)
            if tons > 1e-8:
                margin = 100 * metal / tons - row["grade_min_pp"]
        row["grade_margin_plus20_pp"] = margin
        row["grade_margin_plus20_method"] = "process_interval_overlap"


def collect_shift(source, episode, config, rules, batch_size):
    states, actions = episode["states"], episode["actions"]
    identity = episode["identity"]
    queries = episode["memory"].queries
    events = iter(episode["memory"].events)
    checkpoint_path = source["checkpoint_path"]
    checkpoint_sha = source["checkpoint_sha256"]
    rows = []
    process = []
    last_dispatch = {}

    def consume(event):
        if event.kind == "PROCESS":
            process.append(event)
        if event.kind == "UNLOAD":
            previous = last_dispatch.pop(event.truck_id, None)
            if previous is not None and previous["action"] == event.source_id:
                previous["ore_unloads"] = [dict(
                    truck_id=event.truck_id, dispatch_decision_index=previous["decision_index"],
                    sim_time_min=event.time / 60, grade_pp=event.grade * 100, tons=event.tons
                )]

    for index, (query, state, action) in enumerate(zip(queries, states, actions)):
        for event in events:
            if event.seq == query.current_seq:
                if event.kind != "DECISION" or event.request_id != query.request_id:
                    raise ValueError("Canonical decision event does not match the admitted query")
                break
            consume(event)
        else:
            raise ValueError("Required canonical decision event is missing")
        process, amounts = rolling_amounts(process, event.time)
        row = physical_fields(state, event.time, amounts, config["config.grade_lower"])
        action = int(action)
        if not row["action_mask"][action]:
            raise ValueError(f"Observed action outside the Action Mask at {query.request_id}")
        row.update(policy=source["policy"], training_seed=source["training_seed"],
                   scenario=source["scenario"], seed=identity["environment_seed"],
                   shift_id=f'{identity["recording_id"]}:{identity["run_index"]}',
                   decision_index=query.decision_sequence, action=action,
                   truck_payload_t=float(state[23 + action]) * 193,
                   run_index=identity["run_index"], recording_id=identity["recording_id"],
                   observation_schema=identity["observation_schema"],
                   checkpoint_path=checkpoint_path, checkpoint_sha256=checkpoint_sha,
                   actor_sampling_seed=identity["actor_sampling_seed"],
                   config_sha256=identity["config_sha256"],
                   source_policy_sha256=identity["policy_sha256"],
                   source_receipt_sha256=source["receipt_sha256"],
                   pair_id=source.get("pair_id"),
                   swap_time_min=source.get("swap_time_min"))
        if rules and source["policy"] == "BG128":
            from traditional_dispatch_v2.decision import choose

            row["baseline_actions"] = {}
            row["baseline_diagnostics"] = {}
            for label, name in (("Greedy", "unload_greedy"), ("RHO", "rolling")):
                choice = choose(name, state, row["action_mask"])
                row["baseline_actions"][label] = choice.action
                row["baseline_diagnostics"][label] = dict(
                    kind=choice.kind, timed_out=getattr(choice, "timed_out", None),
                    fallback_reason=getattr(choice, "fallback_reason", None)
                )
        rows.append(row)
        last_dispatch[row["truck_id"]] = row
    for event in events:
        consume(event)
    annotate_grade_margin_plus20(rows, episode["memory"])

    if source["policy"] == "BG128":
        model = load_model(checkpoint_path, checkpoint_sha)
        if model.policy_version != identity["policy_version"]:
            raise ValueError("Frozen checkpoint Actor differs from the admitted trajectory Actor")
        for start in range(0, len(rows), batch_size):
            stop = min(start + batch_size, len(rows))
            batch = diagnostics(model, states[start:stop])
            for row, details, old_log in zip(rows[start:stop], batch,
                                              episode["old_log_probs"][start:stop]):
                sampled_probability = details["action_probs"][row["action"]]
                if sampled_probability <= 0 or not math.isclose(
                        math.log(sampled_probability), float(old_log), rel_tol=0, abs_tol=3e-5):
                    raise ValueError(f"Replayed Actor differs from stored log probability for {row['shift_id']}")
                row.update(details)
    return rows


def import_rows(path):
    """Accept pre-exported explicit telemetry from a separately controlled run."""
    configs = {}
    shifts = {}
    for row in iter_rows(path):
        required = {"shift_id", "policy", "training_seed", "seed", "scenario",
                    "sim_time_min", "decision_index", "action", "action_mask", "config_path", "config_sha256",
                    "observation_schema", "raw_obs", "recording_id", "run_index"}
        if not required.issubset(row):
            raise ValueError(f"Telemetry missing fields {sorted(required - row.keys())}")
        if row["policy"] not in POLICIES or row["training_seed"] not in SEEDS:
            raise ValueError("Unknown policy or reference training seed in telemetry")
        mask, action = row["action_mask"], row["action"]
        raw = np.asarray(row["raw_obs"], dtype=np.float32)
        from relation_gat_ego_v1.adapter import validate_states

        if row["observation_schema"] != "dispatch.physical.v3" or raw.shape != (1519,):
            raise ValueError("Telemetry must retain the exact dispatch.physical.v3 observation")
        validate_states(raw[None, :])
        if (type(action) is not int or not 0 <= action < 4 or len(mask) != 4 or not mask[action]
                or not all(isinstance(flag, bool) for flag in mask)
                or not np.array_equal(raw[19:23], mask)):
            raise ValueError("Telemetry action differs from the original Action Mask")
        cfg_ref = (str(path.parent / row["config_path"]), row["config_sha256"])
        if cfg_ref not in configs:
            configs[cfg_ref] = verify_file(*cfg_ref)
        row["config_path"] = str(configs[cfg_ref])
        key = (row["recording_id"], row["run_index"])
        binding = tuple(row[name] for name in ("policy", "training_seed", "shift_id", "seed",
                                               "scenario", "config_sha256"))
        index, now = row["decision_index"], row["sim_time_min"]
        if type(index) is not int or index < 0 or not isinstance(now, (int, float)) or not math.isfinite(now):
            raise ValueError("Invalid telemetry decision index or simulation time")
        prior = shifts.get(key)
        if prior is not None and (binding != prior[0] or index != prior[1] + 1 or now < prior[2]):
            raise ValueError(f"Telemetry changes shift identity or skips/reorders decisions for {key}")
        shifts[key] = (binding, index, now)
        if row["policy"] == "BG128" and row.get("checkpoint_path"):
            checkpoint = path.parent / row["checkpoint_path"]
            model = load_model(str(checkpoint.resolve()), row["checkpoint_sha256"])
            row.update(diagnostics(model, raw[None, :])[0], checkpoint_path=str(checkpoint.resolve()))
        yield row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    input_options = parser.add_mutually_exclusive_group(required=True)
    input_options.add_argument("--historical", action="store_true", help="Discover completed frozen test episodes")
    input_options.add_argument("--manifest", type=Path, help="JSON list of pinned material receipt sources")
    input_options.add_argument("--telemetry", type=Path, help="Import external JSONL or Parquet decision records")
    parser.add_argument("--scenario", action="append", help="With --historical, select scenario(s)")
    parser.add_argument("--limit-per-group", type=int, help="With --historical, limit shifts per policy/seed/scenario")
    parser.add_argument("--rule-actions", action="store_true", help="Replay Greedy and RHO on the same BG128 states")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", required=True, type=Path, help=".parquet (streamed) or .npz")
    args = parser.parse_args()
    if args.batch_size <= 0 or (args.limit_per_group is not None and args.limit_per_group <= 0):
        parser.error("batch size and limit must be positive")
    if not args.historical and (args.scenario or args.limit_per_group is not None):
        parser.error("--scenario and --limit-per-group are historical filters")

    def shifts():
        if args.telemetry:
            rows = iter(import_rows(args.telemetry))
            while batch := list(islice(rows, 512)):
                yield batch
            return
        import torch

        from _sources import historical_sources, load_source
        torch.set_num_threads(1)
        sources = (historical_sources(set(args.scenario) if args.scenario else None, args.limit_per_group)
                   if args.historical else read_json(args.manifest))
        seen = set()
        for item in sources:
            source = dict(item)
            episode, config = load_source(source)
            identity = episode["identity"]
            key = (identity["recording_id"], identity["run_index"])
            if key in seen:
                raise ValueError(f"Duplicate physical shift {key}")
            seen.add(key)
            yield collect_shift(source, episode, config, args.rule_actions, args.batch_size)

    count = write_rows(args.output, shifts())
    print(f"{count} decisions → {args.output}")


if __name__ == "__main__":
    main()
