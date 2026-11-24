"""Render available M1–M6 results as vector PDF and 300 dpi PNG."""
import argparse
from pathlib import Path

import numpy as np

from _shared import GROUPS, atomic_path, read_json, write_json

COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9")
VEHICLES = ("Cat777", "Cat785", "Cat789")
POLICY_COLORS = dict(zip(("BG128", "Flat-PPO", "Greedy", "RHO"), COLORS))


def empty(axis, label):
    axis.text(.5, .5, label, ha="center", va="center", transform=axis.transAxes,
              wrap=True, fontsize=9, color="#555555")
    axis.set_xticks([])
    axis.set_yticks([])


def legend(axis, **kwargs):
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(handles, labels, **kwargs)


def ribbon(axis, rows, x_key, label, color, *, stairs=False, confidence=False):
    points = sorted((point for point in rows if point.get("mean") is not None
                     and point.get(x_key) is not None), key=lambda point: point[x_key])
    if not points:
        return False
    x = np.array([point[x_key] for point in points])
    y = np.array([point["mean"] for point in points])
    if confidence:
        low, high = (np.array([point.get(key) if point.get(key) is not None else np.nan
                               for point in points]) for key in ("ci95_low", "ci95_high"))
    else:
        sd = np.array([np.nan if point.get("std") is None else point["std"] for point in points])
        low, high = y - sd, y + sd
    style = {"drawstyle": "steps-post"} if stairs else {}
    seed_counts = sorted({point["n_seeds"] for point in points if point.get("n_seeds") is not None})
    if seed_counts:
        count = str(seed_counts[0]) if len(seed_counts) == 1 else f"{seed_counts[0]}–{seed_counts[-1]}"
        label = f"{label} ({count} seed{'s' if count != '1' else ''})"
    axis.plot(x, y, color=color, linewidth=1.5, label=label, **style)
    axis.fill_between(x, low, high, color=color, alpha=.17, step="post" if stairs else None)
    return True


def grouped_bars(axis, rows, category, series, categories, series_values, major,
                 minor="std", *, use_ci=False):
    width = .78 / len(series_values)
    drawn = False
    for j, name in enumerate(series_values):
        records = {row[category]: row for row in rows if row.get(series) == name}
        x, y, lower, upper = [], [], [], []
        for i, value in enumerate(categories):
            row = records.get(value, {})
            if row.get(major) is None:
                continue
            center = row[major]
            x.append(i + width * (j - (len(series_values) - 1) / 2))
            y.append(center)
            sd = row.get(minor)
            lo = row.get("ci95_low") if use_ci else (center - sd if sd is not None else None)
            hi = row.get("ci95_high") if use_ci else (center + sd if sd is not None else None)
            lower.append(max(0., center - lo) if lo is not None else np.nan)
            upper.append(max(0., hi - center) if hi is not None else np.nan)
        if y:
            axis.bar(x, y, width=width * .94, color=COLORS[j], label=str(name))
            known = np.isfinite(lower) & np.isfinite(upper)
            if known.any():
                axis.errorbar(np.asarray(x)[known], np.asarray(y)[known],
                              yerr=np.asarray([lower, upper])[:, known], fmt="none",
                              color="#222222", capsize=2, linewidth=.8)
            drawn = True
    axis.set_xticks(range(len(categories)), categories)
    axis.set_xlim(-.55, len(categories) - .45)
    return drawn


def fig1(plt, payload):
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.2, 3.6))
    m1 = payload.get("m1", {})
    states = ("Normal", "Grade-Critical", "Congestion")
    counts = m1.get("regime_decisions", {})
    labels = [f"{state}\n{counts[state]:,} decisions" if state in counts else state for state in states]
    attention = m1.get("attention", [])
    if grouped_bars(left, attention, "state", "group", states, GROUPS, "pooled_edge_mean", minor=None):
        left.set_xticks(range(len(states)), labels)
        left.set_ylabel("Mean attention per edge")
        handles, groups = left.get_legend_handles_labels()
        layers = {row["group"]: row["layer"] for row in attention}
        left.legend(handles, [f"{group} · L{layers[group]}" for group in groups],
                    loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2,
                    frameon=False, borderaxespad=0)
    else:
        empty(left, "No measured relation-edge coefficients")
    seen = [ribbon(right, [p for p in m1.get("swapped", []) if p.get("policy") == policy],
                   "step", policy, POLICY_COLORS[policy], stairs=True)
            for policy in ("BG128", "Flat-PPO")]
    if any(seen):
        right.set(xlabel="Decision after grade swap", ylabel="Cumulative high-grade dispatch share", ylim=(0, 1.04))
        legend(right, frameon=False)
    else:
        empty(right, "No observed grade-swap comparison")
    return fig


def projection(axis, rows, representation, color_key, fig, title, unit):
    points = [r for r in rows if r.get(representation) is not None and r.get(color_key) is not None]
    axis.set_title(title)
    if not points:
        empty(axis, "UMAP projection unavailable")
        return
    xy = np.asarray([r[representation] for r in points])
    values = np.asarray([r[color_key] for r in points])
    if np.ptp(values) == 0:
        axis.scatter(xy[:, 0], xy[:, 1], color="#6F7980", s=5, alpha=.68, rasterized=True)
        axis.text(.97, .97, f"{unit} = {values[0]:g} for all observations",
                  ha="right", va="top", transform=axis.transAxes, color="#555555")
    else:
        dots = axis.scatter(xy[:, 0], xy[:, 1], c=values, s=5, alpha=.68,
                            cmap="cividis", rasterized=True)
        colorbar = fig.colorbar(dots, ax=axis, label=unit, pad=.02)
        if color_key == "total_shovel_queue":
            colorbar.set_ticks(np.unique(values))
    axis.set(xlabel="UMAP 1", ylabel="UMAP 2")


def fig2(plt, data):
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.7))
    rows = data.get("projection", {}).get("rows", [])
    projection(axes[0, 0], rows, "c128", "grade_margin_pp", fig, "C128", "Grade margin (pp)")
    projection(axes[0, 1], rows, "c128", "total_shovel_queue", fig, "C128", "Total shovel queue")
    projection(axes[1, 0], rows, "raw", "grade_margin_pp", fig, "Raw observation", "Grade margin (pp)")
    panel = axes[1, 1]
    targets = ("total_shovel_queue_15min", "grade_margin_30min")
    if grouped_bars(panel, data.get("metrics", []), "target", "representation",
                    targets, ("c128", "raw"), "rmse_reduction_mean", "rmse_reduction_std"):
        panel.set_xticks([0, 1], ["Shovel queue +15 min", "Grade +30 min"])
        for index, target in enumerate(targets):
            scores = [row for row in data["metrics"] if row["target"] == target]
            if scores and all(row["rmse_reduction_mean"] is None for row in scores):
                panel.text(index, .12, "RMSE reduction unavailable", ha="center",
                           transform=panel.get_xaxis_transform(), color="#555555")
        panel.axhline(0, color="#777777", linewidth=.75)
        panel.set_ylabel("RMSE reduction vs training-mean baseline (%)\nMean ± SD")
        handles, labels = panel.get_legend_handles_labels()
        panel.legend(handles, ["C128" if name == "c128" else "Raw observation" for name in labels],
                     frameon=False, loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2)
    else:
        empty(panel, "No shift-held-out Ridge metrics")
    return fig


def fig3(plt, data):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.15))
    curves = data.get("curves", [])
    specs = (("A", "high_grade_probability", "Fixed-window grade margin (pp)", "High-grade choice probability"),
             ("B", "site_probability", "Inbound commitment edges", "Selected shovel probability"),
             ("C", "waste_probability", "Remaining shift time (min)", "Waste choice probability"))
    for axis, (probe, metric, xlabel, ylabel) in zip(axes, specs):
        points = [row for row in curves if row.get("probe") == probe and row.get("metric") == metric]
        snapshots = points[0].get("n_snapshots") if points else None
        name = "Intervention C · Global node" if probe == "C" else f"Intervention {probe}"
        axis.set_title(name + (f"\n{snapshots} snapshots" if snapshots else ""))
        if not ribbon(axis, points, "x", ylabel, COLORS[0], confidence=True):
            empty(axis, "No admissible snapshots")
            continue
        axis.set(xlabel=xlabel, ylabel=ylabel, ylim=(0, 1.03))
        if probe == "A":
            axis.axvline(0, color="#888888", linestyle="--", linewidth=.8)
            critic = axis.twinx()
            if ribbon(critic, [r for r in curves if r.get("probe") == "A"
                               and r.get("metric") == "critic_value"], "x", "Critic joint cost",
                      COLORS[3], confidence=True):
                critic.set_ylabel("Critic joint cost", color=COLORS[3])
                critic.spines["right"].set_visible(True)
        if probe == "B":
            high = max(point.get("ci95_high") or point["mean"] for point in points)
            axis.set_ylim(0, min(1.03, high + .04))
        if probe == "C":
            low = min(point.get("ci95_low") if point.get("ci95_low") is not None else point["mean"]
                      for point in points)
            high = max(point.get("ci95_high") if point.get("ci95_high") is not None else point["mean"]
                       for point in points)
            padding = max(.016, (high - low) * .07)
            axis.set_ylim(max(0, low - padding), min(1.03, high + padding))
            axis.set_xticks([15, 30, 60, 90, 120])
            axis.invert_xaxis()
    fig.suptitle("Frozen-model response (95% confidence bands)", fontsize=10)
    return fig


def violins(axis, data, key, label):
    any_data = False
    for i, vehicle in enumerate(VEHICLES, 1):
        vals = np.array([r[key] for r in data if r.get("truck_type") == vehicle and r.get(key) is not None])
        if len(vals) >= 2 and np.ptp(vals) > 0:
            density = axis.violinplot(vals, positions=[i], widths=.72, showmedians=True,
                                      showextrema=False)
            for body in density["bodies"]:
                body.set_facecolor(COLORS[i - 1])
                body.set_alpha(.55)
            density["cmedians"].set_color("#222222")
        elif len(vals):
            axis.scatter(i, vals[0], color=COLORS[i - 1])
        any_data |= bool(len(vals))
    if any_data:
        axis.set_xticks(range(1, 4), VEHICLES)
        axis.set_ylabel(label)
    else:
        empty(axis, "No measured dispatch data")


def holm_label(tests, metric, state="All"):
    test = next((row for row in tests if row.get("metric") == metric
                 and row.get("state") == state), None)
    adjusted = test.get("p_holm") if test else None
    return f"Holm p = {adjusted:.3g}" if adjusted is not None else "Holm p unavailable"


def fig4(plt, data):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5))
    m4 = data.get("m4", {})
    tests = m4.get("tests", [])
    states = ("Grade-Critical", "Grade-Ample")
    if grouped_bars(axes[0], m4.get("assignment", []), "state", "truck_type",
                    states, VEHICLES, "mean"):
        axes[0].set_ylabel("High-grade dispatch share")
        axes[0].set_xticks(range(len(states)),
                           [f"{state}\n{holm_label(tests, 'high_grade_share', state)}"
                            for state in states])
        legend(axes[0], frameon=False, ncol=3, loc="lower left",
               bbox_to_anchor=(0, 1.01), borderaxespad=0)
    else:
        empty(axes[0], "No eligible vehicle allocation data")
    samples = m4.get("shift_level", [])
    for axis, key, label in zip(axes[1:], ("distance_km", "grade_contribution_t"),
                                ("Shift mean loaded route distance (km)", "Shift mean grade contribution (t metal)")):
        violins(axis, [row for row in samples if row.get("metric") == key], "value", label)
        axis.set_title(holm_label(tests, key))
    fig.suptitle("Shift-level Kruskal–Wallis tests with Holm correction", fontsize=10)
    return fig


def plot_grade_change(axis, m5, baselines):
    change = m5.get("grade_change_20min", [])
    if grouped_bars(axis, change, "baseline", "agreement", baselines,
                    ("agree", "divergent"), "mean"):
        axis.axhline(0, color="#777777", linewidth=.8)
        axis.set_ylabel("20 min grade-margin change (pp)")
        legend(axis, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2)
    else:
        empty(axis, "No 20 min grade follow-up after compared decisions")


def same_state_divergence(plt, m5):
    fig, (rates, grades) = plt.subplots(1, 2, figsize=(8.9, 3.55))
    baselines = ("Greedy", "RHO")
    counts = m5.get("counts", {})
    shown = [row for row in m5.get("rates", []) if row["baseline"] in baselines]
    for i, baseline in enumerate(baselines):
        row = next((row for row in shown if row["baseline"] == baseline), None)
        if row is None:
            continue
        std = row.get("std")
        rates.bar(i, row["mean"], width=.52, color=POLICY_COLORS[baseline],
                  yerr=std, capsize=3, error_kw={"linewidth": .8})
        sample = counts.get(baseline, {}).get("comparable")
        if sample:
            rates.text(i, .96, f"{sample:,} choices", ha="center",
                       transform=rates.get_xaxis_transform(), fontsize=9)
    if shown:
        rates.set(xticks=range(len(baselines)), xticklabels=baselines, ylim=(0, 1),
                  ylabel="Share of actions differing from BG128")
    else:
        empty(rates, "No same-state rule decisions")
    plot_grade_change(grades, m5, baselines)
    fig.suptitle("Same-state action comparison and observed BG128 follow-up", fontsize=10)
    return fig


def fig5(plt, data):
    m5 = data.get("m5", {})
    rows = m5.get("trajectory", [])
    if not rows and m5.get("rates"):
        return same_state_divergence(plt, m5)
    fig, (paired, grade_change) = plt.subplots(1, 2, figsize=(8.9, 3.4))
    plotted = False
    for baseline, policy in (("Greedy", "BG128"), ("Greedy", "Greedy"),
                             ("RHO", "BG128"), ("RHO", "RHO")):
        series = [r for r in rows if r.get("metric") == "grade_margin_pp"
                  and r.get("baseline") == baseline and r.get("policy") == policy]
        color = POLICY_COLORS[policy] if policy != "BG128" or baseline == "Greedy" else COLORS[5]
        label = f"BG128 vs {baseline}" if policy == "BG128" else policy
        plotted |= ribbon(paired, series, "offset_min", label, color)
    if plotted:
        paired.axvline(0, color="#666666", linestyle="--", linewidth=.8)
        paired.axhline(0, color="#999999", linewidth=.75)
        paired.set(xlabel="Minutes from BG128 divergence", ylabel="Rolling grade margin (pp)")
        legend(paired, frameon=False, ncol=2, loc="lower left",
               bbox_to_anchor=(0, 1.01), borderaxespad=0)
    else:
        empty(paired, "No verified paired divergence window")
    plot_grade_change(grade_change, m5, ("Greedy", "RHO"))
    return fig


def decision_path_panel(axis, case):
    candidates = ("near", "far")
    rows = case.get("attention", [])
    if not grouped_bars(axis, rows, "group", "candidate", GROUPS, candidates, "mean"):
        empty(axis, "No eligible attention paths")
        return
    values = {(row["group"], row["candidate"]): row["mean"] for row in rows
              if row.get("mean") is not None}
    axis.set_xticks(range(len(GROUPS)), GROUPS, rotation=24, ha="right")
    axis.set_ylim(0, max(.05, max(values.values()) * 1.18))
    axis.set_ylabel("Mean attention path score")
    axis.set_title(f'Attention to Truck{case["physical"]["truck_id"]}', pad=27)
    handles, labels = axis.get_legend_handles_labels()
    axis.legend(handles, [case[name]["site_id"] for name in labels], loc="lower left",
                bbox_to_anchor=(0, 1.01), ncol=2, frameon=False, borderaxespad=0)
    if len(values) < len(GROUPS) * len(candidates):
        for j, group in enumerate(GROUPS):
            missing = [i for i, name in enumerate(candidates) if (group, name) not in values]
            positions = [j] if len(missing) == 2 else [j + .39 * (i - .5) for i in missing]
            for x in positions:
                axis.text(x, .018, "—", ha="center", va="bottom",
                          transform=axis.get_xaxis_transform(), color="#52636F")
        axis.set_xlabel("— No eligible path", color="#455560")

    material = [values.get(("Material", name)) for name in candidates]
    if all(value is not None and value > 0 for value in material):
        zoom = axis.inset_axes([.55, .40, .43, .42], facecolor="white")
        scaled = [value * 1000 for value in material]
        for index, value in enumerate(scaled):
            zoom.bar(index, value, color=COLORS[index], width=.62)
            zoom.text(index, value, f"{value:.3f}", ha="center", va="bottom")
        zoom.set(ylim=(0, max(scaled) * 1.25), xticks=[], yticks=[])
        zoom.spines["left"].set_visible(False)
        relays = {path["relay_node"] for row in rows if row["group"] == "Material"
                  for path in row.get("paths", [])}
        title = f"Material via Dump{next(iter(relays)) - 13}" if relays in ({13}, {14}) else "Material"
        zoom.set_title(f"{title}\nScore ×10³", fontsize=9, pad=2)


def fig6(plt, data):
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.9), width_ratios=(1, 1.38, 1, 1.1))
    case = data.get("case")
    if not case:
        for axis, label in zip(axes, ("Mine state", "R-GAT groups", "Actor choice", "Observed follow-up")):
            axis.set_title(label)
            empty(axis, "No matching dispatch case")
        return fig
    criteria = data.get("criteria", {})
    if criteria.get("far_queue") != criteria.get("strict_far_queue"):
        fig.suptitle(f'Alternate observed case · farther shovel queue '
                     f'= {criteria["far_queue"]}', fontsize=10)
    topology, attention, choices, followup = axes
    topology.axis("off")
    topology.text(.13, .87, "Truck", ha="center", transform=topology.transAxes)
    for label, y in (("near", .55), ("far", .18)):
        site = case[label]
        d, grade = site.get("distance_km"), site.get("grade_pp")
        text = f'{site["site_id"]}\n{d:.2f} km · {grade:.2f}%\nQueue {site["queue"]} · Inbound {site["commitment"]}'
        topology.annotate("", xy=(.47, y), xytext=(.13, .79), xycoords="axes fraction",
                          arrowprops={"arrowstyle": "->", "color": COLORS[0]})
        topology.text(.53, y, text, ha="left", va="center", transform=topology.transAxes)
    decision_path_panel(attention, case)
    actions = [r for r in case.get("actions", []) if r.get("probability") is not None]
    if actions:
        positions = np.arange(len(actions))
        choices.bar(positions, [row["probability"] for row in actions], color=COLORS[0], label="BG128")
        greedy = [i for i, row in enumerate(actions) if row.get("greedy")]
        if greedy:
            choices.scatter(greedy, [1.03] * len(greedy), marker="v", color=COLORS[1], label="Greedy choice")
        choices.set(xticks=positions, xticklabels=[row.get("site_id") or str(row["action"]) for row in actions],
                    ylim=(0, 1.12), ylabel="Masked probability")
        legend(choices, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.01),
               borderaxespad=0)
    else:
        empty(choices, "No action distribution")
    series = [row for row in case.get("followup", []) if row.get("grade_roll_pp") is not None]
    if series:
        followup.plot([r["offset_min"] for r in series], [r["grade_roll_pp"] for r in series], color=COLORS[0])
        for index, arrival in enumerate(case.get("arrivals", [])):
            followup.axvline(arrival["offset_min"], color=COLORS[2], linestyle=":", linewidth=.8,
                             label="Linked ore unload" if index == 0 else None)
        cutoff = case.get("physical", {}).get("grade_min_pp")
        if cutoff is not None:
            followup.axhline(cutoff, color="#888888", linestyle="--", linewidth=.8)
        followup.set(xlabel="Minutes after dispatch", ylabel="Observed rolling grade (%)",
                     xlim=(0, case.get("followup_horizon_min", 25)))
        legend(followup, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.01),
               borderaxespad=0)
    else:
        empty(followup, "No observed follow-up grade")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", type=Path, help="M1/M4/M5 stats.json")
    parser.add_argument("--probes", type=Path, help="M3 counterfactual.json")
    parser.add_argument("--linear", type=Path, help="M2 linear_probing.json")
    parser.add_argument("--case", type=Path, help="M6 case.json")
    parser.add_argument("--figures", type=int, choices=range(1, 7), nargs="+",
                        help="Export only the selected figure numbers")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if not any((args.stats, args.probes, args.linear, args.case)):
        parser.error("supply at least one analysis result")

    def checked(path, schema):
        if path is None:
            return None
        payload = read_json(path)
        if payload.get("schema") != schema:
            parser.error(f"{path} requires {schema}; regenerate it using the corresponding analysis script")
        return payload

    stats = checked(args.stats, "bg128-interpretability-stats-v3")
    probes = checked(args.probes, "bg128-counterfactual-v3")
    linear = checked(args.linear, "bg128-linear-probing-v3")
    case = checked(args.case, "bg128-interpretability-case-v3")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        parser.exit(1, f"Plotting requires matplotlib in the analysis environment: {error}\n")
    matplotlib.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 9, "axes.ymargin": .05, "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelsize": 9, "axes.titlesize": 10, "legend.fontsize": 9, "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures = ((1, fig1, stats), (2, fig2, linear), (3, fig3, probes),
               (4, fig4, stats), (5, fig5, stats), (6, fig6, case))
    available = {number for number, _, payload in figures if payload is not None}
    selected = set(args.figures) if args.figures else available
    if selected - available:
        parser.error(f"missing analysis input for figures {sorted(selected - available)}")
    for number, draw, payload in figures:
        if number not in selected:
            continue
        fig = draw(plt, payload)
        fig.tight_layout(pad=1.1)
        for extension in ("pdf", "png"):
            path = args.output_dir / f"Fig{number}.{extension}"
            with atomic_path(path) as staging:
                fig.savefig(staging, dpi=300, bbox_inches="tight")
        plt.close(fig)
    outputs = {}
    for number, _, _ in figures:
        files = [f"Fig{number}.{suffix}" for suffix in ("pdf", "png")]
        if all((args.output_dir / name).is_file() for name in files):
            outputs[f"Fig{number}"] = files
    write_json(args.output_dir / "figures.json", outputs)
    print(f"{len(selected)} figure(s) → {args.output_dir}")


if __name__ == "__main__":
    main()
