"""Build Figure 4 from one verified joint-shift trajectory with anylogic-env."""
from __future__ import annotations

from collections import Counter, defaultdict
import ctypes as C
from ctypes.util import find_library
from datetime import datetime, timezone
from hashlib import sha256
from html import escape
import json
import math
from pathlib import Path
import sqlite3
import sys

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = next(path for path in HERE.parents if (path / "grade_event_memory_v1").is_dir())
sys.path.insert(0, str(PROJECT))
from grade_event_memory_v1.events import build_memory
from grade_event_memory_v1.journal import read_journal
from python.constrained_batch import load_dataset


RUN = PROJECT / "runs/bg128-continuous-mixed-20261119-001"
PHASE = RUN / "evaluation/joint_shift/new"
STEM = "figure4_fifo_timeline"
BIN_SECONDS = 600
WINDOW_SECONDS = 7200
SHIFT_SECONDS = 28800
GOAL = .27
BOUNDS = (.26, .28)
COLORS = {"H": "#b75320", "M": "#3d6fa4", "L": "#16847c", "W": "#858b96"}
INK = "#172b3a"
FAINT = "#506675"
GRID = "#dce5eb"
W, H = 1240, 1002


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_bytes())


def ref(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256(path.read_bytes()).hexdigest()}


def selection():
    report_path = PHASE / "evaluation_complete.json"
    report = read(report_path)
    require(report["scenario_id"] == "joint_shift" and report["policy_arm"] == "new"
            and report["training_state_unchanged"] is True,
            "expected the complete fixed mixed-u32 joint-shift report")
    require(len(report["per_episode"]) == 100, "expected all 100 sealed shifts")
    names = ("ore_tons", "waste_tons", "total_wait_seconds", "four_window_grade_rmse")
    pool = []
    with sqlite3.connect((RUN / "budget.sqlite").as_uri() + "?mode=ro", uri=True) as db:
        for row in report["per_episode"]:
            run = row["identity"]["run_index"]
            status, = db.execute("SELECT status FROM slots WHERE run=?", (run,)).fetchone()
            require(status == "COMPLETE" and row["identity"]["policy_version"] == report["fingerprints"]["actor"],
                    "evaluation report contains an incomplete or different-Actor shift")
            physical = row["physical"]
            if physical["complete_task"]:
                require(len(physical["windows"]) == 4 and all(w["grade"] is not None for w in physical["windows"]),
                        "complete task lacks four actual grades")
                rmse = math.sqrt(math.fsum((w["grade"] - GOAL) ** 2 for w in physical["windows"]) / 4)
                pool.append((row, [physical[name] for name in names[:3]] + [rmse]))
    samples = np.array([features for _, features in pool], dtype=float)
    medians = np.median(samples, axis=0)
    q1, q3 = np.quantile(samples, [.25, .75], axis=0)
    iqr = q3 - q1
    normalized = (samples[:, iqr > 0] - medians[iqr > 0]) / iqr[iqr > 0]
    distances = np.linalg.norm(normalized, axis=1)
    require(len(pool) > 0 and np.isfinite(distances).all(), "invalid representative cohort")
    index = min(range(len(pool)), key=lambda i: (distances[i], pool[i][0]["identity"]["run_index"]))
    row, features = pool[index]
    return row, {
        "universe": "All 100 sealed fixed mixed-u32 joint_shift validation shifts in the bound report.",
        "eligible": len(pool),
        "total_completed": 100,
        "frozen_report": ref(report_path),
        "checkpoint": report["checkpoint_provenance"],
        "frozen_report_mtime_utc": datetime.fromtimestamp(report_path.stat().st_mtime, timezone.utc).isoformat(),
        "criterion": "Minimum Euclidean distance from the four conditional cohort medians after IQR scaling. "
                     "Exclude a dimension with zero IQR. Break ties by run_index.",
        "medians": dict(zip(names, medians.tolist())),
        "iqr": dict(zip(names, iqr.tolist())),
        "selected_features": dict(zip(names, features)),
        "distance": float(distances[index]),
    }


def load_trajectory(row):
    ident = row["identity"]
    run = ident["run_index"]
    directory = PHASE / "episodes" / f"run_{run}"
    receipt_path = directory / "material_receipt.json"
    receipt = read(receipt_path)
    require(receipt["normal_completion"] is True and receipt["identity"] == ident,
            "material receipt does not match the sealed report row")
    refs = receipt["references"]
    require(all(ref(item["path"]) == item for item in refs.values()), "selected provenance changed")
    manifest = read(refs["raw_manifest"]["path"])
    paths = {name: ref(directory / name) for name in manifest["files_sha256"]}
    require(all(item["sha256"] == manifest["files_sha256"][name] for name, item in paths.items()),
            "selected raw RPC payload changed")
    cfg = read(refs["config"]["path"])
    job, = [job for job in cfg["jobs"] if job["run_index"] == run]
    require(job["experiment_seed"] == ident["environment_seed"]
            and job["actor_sampling_seed"] == ident["actor_sampling_seed"],
            "selected seed differs from the frozen job")
    params = job["expected_rpc_config"]
    means = [params[f"config.runtime_scene.load.{site}.grade"] for site in range(4)]
    ore = [site for site in range(4) if params[f"config.runtime_scene.load.{site}.isOre"] == 1]
    roles = {site: role for site, role in zip(sorted(ore, key=means.__getitem__), "LMH", strict=True)}
    waste, = set(range(4)) - set(ore)
    roles[waste] = "W"
    journal = read_journal(refs["journal"]["path"])
    require(journal.header["run_index"] == run and journal.header["episode_id"] == ident["episode_id"]
            and journal.logical_sha256 == ident["journal_logical_sha256"], "journal identity differs")
    memory = build_memory(journal.events)
    reader, episodes, _, _, _ = load_dataset(directory, require_real=True)
    try:
        rows = list(reader)
    finally:
        reader.close()
    episode, = episodes
    infos = [episode["start_info"]] + [sample["settlement_info"] for sample in rows[:-1]]
    decisions = [
        {"decision": i, "request_id": info["request_id"], "seconds": info["sim_time_seconds"],
         "truck": sample["truck_id"], "site": sample["action"], "role": roles[sample["action"]]}
        for i, (sample, info) in enumerate(zip(rows, infos, strict=True))
    ]
    with h5py.File(refs["episode"]["path"], "r") as hf:
        require(json.loads(hf.attrs["identity_json"]) == ident and
                np.array_equal(hf["transitions/action"][...], [d["site"] for d in decisions]),
                "sealed derived actions differ from raw RPC")
    require(len(decisions) == row["transitions"] == len(memory.queries),
            "decision inventory differs")
    return roles, means, decisions, journal, memory, {"receipt": ref(receipt_path), **refs, **paths}


def measure(roles, means, decisions, journal, memory, row):
    n = SHIFT_SECONDS // BIN_SECONDS
    dispatch_bins = np.zeros((n, 4), dtype=int)
    unload_bins, process_bins = np.zeros((n, 4)), np.zeros((n, 4))
    windows = [defaultdict(lambda: dict(tons=0., metal_tons=0.)) for _ in range(4)]
    inbound = [defaultdict(lambda: dict(tons=0., metal_tons=0.)) for _ in range(4)]
    last_dispatch = {}
    batches = {}
    used_decisions = set()
    for event in journal.events:
        if event.kind == "DECISION":
            d = decisions[event.decision_sequence]
            require(event.request_id == d["request_id"] and event.time == d["seconds"],
                    "raw dispatch does not match its material marker")
            dispatch_bins[int(event.time // BIN_SECONDS), d["site"]] += 1
            last_dispatch[d["truck"]] = d
        elif event.kind == "UNLOAD":
            d = last_dispatch[event.truck_id]
            require(d["site"] == event.source_id and roles[d["site"]] != "W"
                    and d["seconds"] <= event.time and d["decision"] not in used_decisions,
                    "received batch does not bind to one preceding ore dispatch")
            used_decisions.add(d["decision"])
            unload_bins[int(event.time // BIN_SECONDS), event.source_id] += event.tons
            entry = inbound[int(event.time // WINDOW_SECONDS)][event.source_id]
            entry["tons"] += event.tons
            entry["metal_tons"] += event.metal
            batches[event.batch_id] = {
                "batch_id": event.batch_id, "role": d["role"], "site": d["site"], "truck": d["truck"],
                "dispatch_decision": d["decision"], "dispatch_seconds": d["seconds"], "unload_seconds": event.time,
                "unloaded_tons": event.tons, "unloaded_metal_tons": event.metal, "actual_grade": event.grade,
                "processing": [],
            }
        elif event.kind == "PROCESS":
            start, end = event.interval_start, event.interval_end
            for bin_id in range(int(start // BIN_SECONDS), min(int(end // BIN_SECONDS) + 1, n)):
                overlap = max(0., min(end, (bin_id + 1) * BIN_SECONDS) - max(start, bin_id * BIN_SECONDS))
                process_bins[bin_id, event.source_id] += event.tons * overlap / (end - start)
            entry = windows[event.window][event.source_id]
            entry["tons"] += event.tons
            entry["metal_tons"] += event.metal
            batches[event.batch_id]["processing"].append(
                {"window": event.window + 1, "physical_start_seconds": start, "physical_end_seconds": end,
                 "visible_seconds": event.time, "tons": event.tons, "metal_tons": event.metal})
    window_records = []
    for i, (win, settle, physical) in enumerate(zip(windows, memory.settlements,
                                                     row["physical"]["windows"], strict=True)):
        require(math.isclose(math.fsum(v["tons"] for v in win.values()), settle.tons, abs_tol=1e-7)
                and math.isclose(math.fsum(v["metal_tons"] for v in win.values()), settle.metal, abs_tol=1e-7)
                and math.isclose(physical["grade"], settle.grade, abs_tol=1e-12)
                and BOUNDS[0] <= settle.grade <= BOUNDS[1], "actual window grade or mass mismatch")
        for aggregate in (win, inbound[i]):
            for record in aggregate.values():
                record["signed_metal_surplus_tons"] = record["metal_tons"] - GOAL * record["tons"]
        arrivals = inbound[i]
        window_records.append(
            {"window": i + 1, "interval_seconds": [i * WINDOW_SECONDS, (i + 1) * WINDOW_SECONDS],
             "settlement_visible_seconds": settle.time,
             "actual_processed_tons": settle.tons, "actual_processed_metal_tons": settle.metal,
             "actual_grade": settle.grade, "grade_deviation_percentage_points": 100 * (settle.grade - GOAL),
             "processed_by_site": dict(win), "ore_unloaded_by_site": dict(arrivals),
             "ore_unloaded_weighted_grade": math.fsum(x["metal_tons"] for x in arrivals.values()) /
                                             math.fsum(x["tons"] for x in arrivals.values())})
    require(math.isclose(float(process_bins.sum()), math.fsum(s.tons for s in memory.settlements), abs_tol=1e-7),
            "physical PROCESS interval binning lost mass")
    begin_delays = []
    arrival_delays = []
    crossers = []
    for item in batches.values():
        processing = item["processing"]
        arrival_delays.append(item["unload_seconds"] - item["dispatch_seconds"])
        if processing:
            first_start = processing[0]["physical_start_seconds"]
            delay = first_start - item["unload_seconds"]
            require(delay >= -1e-7, "negative FIFO delay")
            begin_delays.append(max(0., delay))
            item["fifo_wait_seconds"] = delay
            per_window = defaultdict(float)
            for segment in processing:
                per_window[segment["window"]] += segment["tons"]
            if len(per_window) > 1:
                item["processed_tons_by_window"] = dict(per_window)
                item["signed_metal_surplus_tons_by_window"] = {
                    window: tons * (item["actual_grade"] - GOAL) for window, tons in per_window.items()
                }
                crossers.append(item)
    crossers.sort(key=lambda item: min(item["processed_tons_by_window"]))
    require(bool(crossers), "representative shift contains no boundary-crossing batch")
    boundary = min(crossers[0]["processed_tons_by_window"]) * WINDOW_SECONDS
    preceding = boundary - 1800
    recent = Counter(d["role"] for d in decisions if preceding <= d["seconds"] < boundary)
    return {
        "sites": {site: {"role": roles[site], "mean_grade": means[site]} for site in range(4)},
        "dispatches": decisions,
        "first_10_minute_dispatches": [d for d in decisions if d["seconds"] < BIN_SECONDS],
        "bins": {"seconds": BIN_SECONDS, "dispatch_count_by_site": dispatch_bins.tolist(),
                 "ore_unloaded_tons_by_site": unload_bins.tolist(),
                 "physically_processed_tons_by_site": process_bins.tolist()},
        "ore_unload_batches": [{key: value for key, value in item.items() if key != "processing"}
                               for item in batches.values()],
        "pre_boundary_dispatches": {"interval_seconds": [preceding, boundary],
                                    "interval_convention": "left closed, right open",
                                    "count_by_role": {role: recent[role] for role in "HMLW"}},
        "windows": window_records, "boundary_crossing_batches": crossers,
        "ore_unloaded_batches": len(batches), "processed_batches": len(begin_delays),
        "unloaded_minus_processed_tons_at_shift_end": float(unload_bins.sum() - process_bins.sum()),
        "dispatch_to_unload_seconds_10th_50th_90th": np.quantile(arrival_delays, [.1, .5, .9]).tolist(),
        "unload_to_first_process_seconds_10th_50th_90th": np.quantile(begin_delays, [.1, .5, .9]).tolist(),
        "raw_process_calls": journal.raw_process_calls, "canonical_events": len(journal.events),
        "time_semantics": {
            "dispatch": "Original RPC time and selected action; counts include all four sites.",
            "unload": "Physical arrival and deposit at the crusher FIFO; ore only.",
            "process": "Physical first-in/first-out intervals, not recorder visibility timestamps. "
                       "Mass allocated to 10-minute bins by interval overlap at constant rate within each event.",
            "signed_metal_surplus": "Each contribution is metal_tons minus 0.27 times ore tons.",
            "scope": "One descriptive complete shift. Lags combine observed travel, loading, queues and FIFO "
                     "as labeled. No counterfactual effect or intent is inferred from the trajectory."
        },
    }


def svg_for(meta):
    sample = meta["data"]
    fig = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">']

    def rect(x, y, width, height, fill):
        if height > 0:
            fig.append(f'<rect x="{x:.3f}" y="{y:.3f}" width="{width:.3f}" height="{height:.3f}" fill="{fill}"/>')

    def line(x1, y1, x2, y2, color=GRID, width=1, dash=""):
        pattern = f' stroke-dasharray="{dash}"' if dash else ""
        fig.append(f'<line x1="{x1:.3f}" y1="{y1:.3f}" x2="{x2:.3f}" y2="{y2:.3f}" stroke="{color}" stroke-width="{width}"{pattern}/>')

    def circle(x, y, radius, fill):
        fig.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="{radius:.3f}" fill="{fill}" fill-opacity=".84"/>')

    def text(x, y, value, size=17, *, color=INK, weight=400, anchor="start"):
        fig.append(f'<text x="{x:.3f}" y="{y:.3f}" text-anchor="{anchor}" fill="{color}" '
                   f'font-family="Noto Sans CJK SC, sans-serif" font-size="{size}" font-weight="{weight}">{escape(value)}</text>')

    left, right = 116, W - 62
    space = right - left
    xhour = lambda hour: left + hour * space / 8
    times = np.arange(0, 8.01, 1)
    grades = [100 * w["actual_grade"] for w in sample["windows"]]
    by_site = sample["sites"]
    stack_order = [next(site for site, record in by_site.items() if record["role"] == role) for role in "LMHW"]
    focus = sample["boundary_crossing_batches"][0]
    rect(0, 0, W, H, "#ffffff")
    text(left, 54, "图 4　派车、入仓与 FIFO 破碎的时间错位", 32, weight=700)
    text(left, 91, f"混合 BG128/u32 · 联合变化独评 · 代表班 {meta['identity']['run_index']}", 19, color=FAINT)
    for i, role in enumerate("HMLW"):
        site = stack_order["LMHW".index(role)]
        origin = left + 258 * i
        label = {"H": "高矿", "M": "中矿", "L": "低矿", "W": "废石"}[role]
        entry = by_site[site]
        grade = f"  {entry['mean_grade'] * 100:.1f}%" if role != "W" else ""
        rect(origin, 113, 16, 14, COLORS[role])
        text(origin + 26, 127, f"{label}  Load{site}{grade}", 16)

    def panel(header, data, top, bottom, step, maximum, order):
        text(left, top - 12, header, 19, weight=700)
        yscale = lambda value: bottom - value * (bottom - top) / maximum
        for tick in np.arange(0, maximum + 1, step):
            level = yscale(tick)
            line(left, level, right, level)
            text(left - 13, level + 5, str(int(tick)), 14, color=FAINT, anchor="end")
        for i, values in enumerate(data):
            floor = 0
            for site in order:
                value = values[site]
                rect(left + i * space / len(data) + 0.9, yscale(floor + value),
                     space / len(data) - 1.8, yscale(floor) - yscale(floor + value),
                     COLORS[by_site[site]["role"]])
                floor += value
        for hour in times[2::2][:-1]:
            line(xhour(hour), top, xhour(hour), bottom, "#627887", 1.2, "5 4")

    bins = sample["bins"]
    dispatch_array = np.asarray(bins["dispatch_count_by_site"])
    process_array = np.asarray(bins["physically_processed_tons_by_site"])
    max_dispatch = 4 * math.ceil(max(dispatch_array.sum(1)) / 4)
    max_mass = 100 * math.ceil(max(process_array.sum(1)) / 100)
    panel("A　派车次数（每 10 分钟）", dispatch_array, 172, 279, 4, max_dispatch, stack_order)

    text(left, 310, "B　入仓批次的实测品位（按实际卸载时刻）", 19, weight=700)
    text(right, 310, "点面积表示入仓吨位", 15, color=FAINT, anchor="end")
    btop, bbottom = 322, 429
    batch_grade_y = lambda grade: bbottom - (grade - 13) / 27 * (bbottom - btop)
    for grade in (15, 20, 25, 30, 35, 40):
        y = batch_grade_y(grade)
        line(left, y, right, y)
        text(left - 13, y + 5, f"{grade}%", 14, color=FAINT, anchor="end")
    for hour in (2, 4, 6):
        line(xhour(hour), btop, xhour(hour), bbottom, "#627887", 1.2, "5 4")
    for batch in sample["ore_unload_batches"]:
        circle(xhour(batch["unload_seconds"] / 3600), batch_grade_y(100 * batch["actual_grade"]),
               3.5 * math.sqrt(batch["unloaded_tons"] / 193), COLORS[batch["role"]])

    panel("C　FIFO 实际破碎（吨／10 分钟）", process_array, 472, 579, 100, max_mass, stack_order[:3])
    lag = sample["unload_to_first_process_seconds_10th_50th_90th"][1] / 60
    text(right, 460, f"入仓至首破碎中位等待 {lag:.2f} 分钟", 15, color=FAINT, anchor="end")

    text(left, 614, "D　每个两小时窗口的实际破碎品位", 19, weight=700)
    rect(right - 160, 601, 16, 13, "#e7f4ed")
    text(right, 614, "合规带 26%–28%", 15, color=FAINT, anchor="end")
    top, bottom = 626, 723
    yp = lambda grade: bottom - (grade - 25.6) / 2.8 * (bottom - top)
    rect(left, yp(28), space, yp(26) - yp(28), "#e7f4ed")
    for tick in (26, 27, 28):
        level = yp(tick)
        line(left, level, right, level, "#a3c7b2" if tick == 27 else "#c9ddd5", 1.1, "5 4" if tick == 27 else "")
        text(left - 13, level + 5, f"{tick}%", 14, color=FAINT, anchor="end")
    for i, grade in enumerate(grades):
        x0, x1 = xhour(i * 2), xhour((i + 1) * 2)
        line(x0, yp(grade), x1, yp(grade), "#173c56", 3)
        if i:
            line(x0, yp(grades[i-1]), x0, yp(grade), "#173c56", 1.4)
        text((x0 + x1) / 2, yp(grade) - 10, f"{grade:.2f}%", 17, weight=700, anchor="middle")
    for hour in times:
        line(xhour(hour), bottom, xhour(hour), bottom + 5, "#8296a4")
        text(xhour(hour), bottom + 23, f"{hour:g} 小时", 14, color=FAINT, anchor="middle")
        if hour in (2, 4, 6):
            line(xhour(hour), top, xhour(hour), bottom, "#627887", 1, "5 4")

    panel_y = 766
    rect(left, panel_y, space, 190, "#f4f7f9")
    line(left, panel_y, right, panel_y, "#cad9e2")
    text(left + 24, panel_y + 33,
         f"E　跨 2 小时边界的实际批次　{focus['batch_id']} · Truck{focus['truck']} · "
         f"Load{focus['site']} · 品位 {100 * focus['actual_grade']:.2f}%", 19, weight=700)
    axis_start, axis_stop = left + 58, right - 43
    start = focus["dispatch_seconds"]
    middle = focus["unload_seconds"]
    fifo_start = focus["processing"][0]["physical_start_seconds"]
    end = focus["processing"][-1]["physical_end_seconds"]
    x = lambda seconds: axis_start + (seconds - start) / (end - start) * (axis_stop - axis_start)
    y = panel_y + 85
    line(x(start), y, x(middle), y, "#748896", 3)
    line(x(middle), y, x(fifo_start), y, "#748896", 3, "5 4")
    for interval in focus["processing"]:
        line(x(interval["physical_start_seconds"]), y, x(interval["physical_end_seconds"]), y,
             COLORS[focus["role"]], 8)
    boundary = min(focus["processed_tons_by_window"]) * WINDOW_SECONDS
    line(x(boundary), y - 26, x(boundary), y + 30, "#314b5a", 1.2, "4 4")
    for moment, caption in [(start, "派车"), (middle, "入仓"), (fifo_start, "开始破碎")]:
        line(x(moment), y - 7, x(moment), y + 7, INK, 1.4)
        text(x(moment), y - 12, caption, 15, anchor="middle")
    text(x(start), y + 24, f"{start / 60:.1f} 分", 14, color=FAINT, anchor="middle")
    text(x(middle), y + 24, f"{middle / 60:.1f} 分", 14, color=FAINT, anchor="middle")
    text(x(fifo_start) - 4, y + 24, f"{fifo_start / 60:.1f} 分", 14, color=FAINT, anchor="middle")
    text(x(boundary) + 8, y - 11, f"{boundary / 60:.0f} 分", 14, color=FAINT, anchor="start")
    tonnes = list(focus["processed_tons_by_window"].values())
    text(left + 24, panel_y + 147,
         f"该批矿石：第一窗破碎 {tonnes[0]:.1f} 吨，第二窗破碎 {tonnes[1]:.1f} 吨。"
         f"入仓后等待 {focus['fifo_wait_seconds'] / 60:.2f} 分钟才开始破碎。", 16)
    prior = sample["pre_boundary_dispatches"]
    count = prior["count_by_role"]
    start_min, stop_min = [seconds // 60 for seconds in prior["interval_seconds"]]
    text(left + 24, panel_y + 172,
         f"边界前 {start_min}–{stop_min} 分钟派车　高 {count['H']} · 中 {count['M']} · 低 {count['L']} · "
         f"废 {count['W']}　高∶低 = {count['H']}∶{count['L']}", 16, color=FAINT)
    text(left, 981,
         "派车含矿废四站，入仓与破碎仅为矿石。破碎按物理区间分箱，窗口品位按实际金属吨／矿石吨计算。", 15, color=FAINT)
    fig.append("</svg>")
    return "".join(fig).encode("utf-8")


def export(svg, directory):
    """Render the same vector source to a double-resolution PNG and a vector PDF."""
    cairo = C.CDLL(find_library("cairo"))
    rsvg = C.CDLL(find_library("rsvg-2"))
    gobject = C.CDLL(find_library("gobject-2.0"))
    void, integer, double = C.c_void_p, C.c_int, C.c_double

    class Viewport(C.Structure):
        _fields_ = [("x", double), ("y", double), ("width", double), ("height", double)]

    def method(lib, name, args, result=None):
        function = getattr(lib, name)
        function.argtypes, function.restype = args, result
        return function

    open_svg = method(rsvg, "rsvg_handle_new_from_data", [C.c_char_p, C.c_size_t, C.POINTER(void)], void)
    render = method(rsvg, "rsvg_handle_render_document", [void, void, C.POINTER(Viewport), C.POINTER(void)], integer)
    unref = method(gobject, "g_object_unref", [void])
    image = method(cairo, "cairo_image_surface_create", [integer, integer, integer], void)
    pdf = method(cairo, "cairo_pdf_surface_create", [C.c_char_p, double, double], void)
    context = method(cairo, "cairo_create", [void], void)
    write_png = method(cairo, "cairo_surface_write_to_png", [void, C.c_char_p], integer)
    surface_status = method(cairo, "cairo_surface_status", [void], integer)
    destroy = method(cairo, "cairo_destroy", [void])
    finish = method(cairo, "cairo_surface_finish", [void])
    destroy_surface = method(cairo, "cairo_surface_destroy", [void])
    show_page = method(cairo, "cairo_show_page", [void])
    err = void()
    handle = open_svg(svg, len(svg), C.byref(err))
    require(handle and not err, "could not parse chart SVG")
    try:
        for extension, scale in (("png", 2), ("pdf", .72)):
            path = str(directory / f"{STEM}.{extension}").encode()
            viewport = Viewport(0, 0, W * scale, H * scale)
            surface = (image(0, int(viewport.width), int(viewport.height)) if extension == "png"
                       else pdf(path, viewport.width, viewport.height))
            ctx = context(surface)
            try:
                require(render(handle, ctx, C.byref(viewport), C.byref(err)) and not err,
                        f"could not render chart {extension}")
                if extension == "png":
                    require(write_png(surface, path) == 0, "PNG write failed")
                else:
                    show_page(ctx)
                finish(surface)
                require(surface_status(surface) == 0, f"{extension} output surface failed")
            finally:
                destroy(ctx)
                destroy_surface(surface)
    finally:
        unref(handle)


def main():
    row, selected = selection()
    roles, means, decisions, journal, memory, sources = load_trajectory(row)
    stats = measure(roles, means, decisions, journal, memory, row)
    result = {"schema": "bg128-figure4-fifo-timeline-v1", "identity": row["identity"],
              "selection": selected, "source_references": sources, "data": stats}
    export(svg_for(result), HERE)
    destination = HERE / f"{STEM}.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(f"Rendered run {row['identity']['run_index']} in {HERE}")


if __name__ == "__main__":
    main()
