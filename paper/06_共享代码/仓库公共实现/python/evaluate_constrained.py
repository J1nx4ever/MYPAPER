"""Phase E: fixed-policy holdout evaluation of a constrained-PPO checkpoint.

Inference only. No optimizer is constructed, no parameter is updated, no reward,
cost, multiplier or observation feature is changed. Two execution modes are
evaluated separately and never mixed: `random` (masked categorical sampling) and
`argmax` (masked deterministic action).

Identity is verified before any episode is scored: checkpoint integrity digest,
actor digest, 1519D raw observation contract, action mask carried in
`observation[19:23]`, and the RPC reward configuration/scenario recorded when the
checkpoint was written.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from .collect_constrained import ConstrainedBridge
from .constrained_adapter import AdapterConfiguration, RewardCostAdapter, require
from .constrained_batch import FrozenActor, write_json
from .constrained_ppo import (CHECKPOINT_FORMAT, COST_NAMES, ConstrainedPPOConfig,
                              _digest_payload)
from .dispatch_bridge import ProtocolError
from .dispatch_env import DispatchEnv
from .dispatch_service import create_server
from .episode_runner import ParameterVariationEpisodeRunner
from .masked_policy import DispatchMaskedPolicy, POLICY_CONTRACT_ID
from .observation_contract import (ACTION_MASK_SLICE, OBSERVATION_SCHEMA_ID,
                                   OBSERVATION_SIZE, validate_observation)
from .scenario_contract import from_components, metadata

EVAL_CONTRACT = "dispatch-constrained-holdout-eval-v1"
FIFO_BASE, FIFO_STRIDE, FIFO_SLOTS, FIFO_TONS_SCALE = 1391, 3, 16, 500.0
MODES = ("random", "argmax")


# ----------------------------------------------------------------- identity
def load_actor(checkpoint_path):
    """Rebuild the Actor from a training checkpoint and prove it matches."""
    path = Path(checkpoint_path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    require(payload.get("format") == CHECKPOINT_FORMAT, "unknown checkpoint format")
    expected = payload.pop("integrity_sha256", None)
    require(expected is not None and expected == _digest_payload(payload),
            "checkpoint integrity check failed")
    config_payload = payload["config"]
    config = ConstrainedPPOConfig(**{k: tuple(v) if isinstance(v, list) and k in
                                     ("cost_weights", "cost_limits", "hidden_sizes") else v
                                     for k, v in config_payload.items()
                                     if k in ConstrainedPPOConfig.__dataclass_fields__})
    import gymnasium as gym
    observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(OBSERVATION_SIZE,),
                                       dtype=np.float32)
    policy = DispatchMaskedPolicy(observation_space, gym.spaces.Discrete(4), lambda _: 3e-4,
                                  net_arch=dict(pi=list(config.hidden_sizes), vf=list(config.hidden_sizes)))
    policy.load_state_dict(payload["policy_state"])
    policy.set_training_mode(False)
    actor = FrozenActor(policy)
    actor.assert_frozen()
    require(actor.digest == payload["actor_digest"], "restored actor digest differs from checkpoint")
    require(actor.policy_version and len(actor.policy_version) == 64, "invalid policy version")
    online = payload.get("online_trainer") or {}
    identity = dict(
        eval_contract=EVAL_CONTRACT, checkpoint=str(path),
        checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        checkpoint_integrity_sha256=expected,
        actor_digest=actor.digest, policy_version=actor.policy_version,
        policy_contract_id=POLICY_CONTRACT_ID, observation_schema_id=OBSERVATION_SCHEMA_ID,
        observation_size=int(policy.observation_space.shape[0]),
        preprocessing=actor.preprocessing,
        training_mode=any(module.training for module in actor.policy.modules()),
        # Training-side update counter; the evaluation itself performs none.
        checkpoint_update_index=int(payload.get("update_index", 0)),
        lambdas=list(payload.get("lambdas", [])), config=config.to_dict(),
        training_output_dir=online.get("output_dir"),
        expected_configuration_fingerprint=online.get("expected_configuration_fingerprint"),
        expected_scenario=online.get("expected_scenario"),
        consumed_batches=[item.get("batch_index") for item in online.get("consumed_batches", [])])
    require(not identity["training_mode"], "policy must be in eval mode")
    require(identity["observation_size"] == OBSERVATION_SIZE, "checkpoint is not a 1519D policy")
    require(identity["expected_configuration_fingerprint"] is not None,
            "checkpoint carries no RPC configuration identity")
    return actor, identity


def load_holdout_seeds(path):
    payload = json.loads(Path(path).read_text())
    require(isinstance(payload, list) and payload, "holdout manifest must be a non-empty list")
    rows = [(int(row["run_index"]), int(row["experiment_seed"])) for row in payload]
    require(all(i > 0 and s > 0 for i, s in rows), "invalid holdout seed identity")
    require(len({i for i, _ in rows}) == len(rows) and len({s for _, s in rows}) == len(rows),
            "duplicate run index or seed in holdout manifest")
    return rows


def assert_disjoint_from_training(holdout, training_manifests):
    trained = set()
    for path in training_manifests:
        for _, seed in load_holdout_seeds(path):
            trained.add(seed)
    overlap = sorted(set(seed for _, seed in holdout) & trained)
    require(not overlap, f"holdout seeds were used for training: {overlap}")
    return dict(training_seeds=sorted(trained), holdout_seeds=[s for _, s in holdout],
                disjoint=True)


# ------------------------------------------------------------------ policy
class EvaluationActor:
    """Wraps FrozenActor with an explicit execution mode."""

    def __init__(self, actor, mode):
        require(mode in MODES, f"unknown evaluation mode {mode!r}")
        self.actor, self.mode = actor, mode

    def act(self, observation, mask):
        distribution = self.actor.distribution(observation, mask)
        deterministic = self.mode == "argmax"
        action = distribution.get_actions(deterministic=deterministic)
        probs = distribution.distribution.probs[0].cpu().numpy().copy()
        return int(action.item()), probs

    def assert_frozen(self):
        self.actor.assert_frozen()


# --------------------------------------------------------------- recording
def fifo_remaining(observation):
    obs = np.asarray(observation)
    total = 0.0
    for slot in range(FIFO_SLOTS):
        base = FIFO_BASE + slot * FIFO_STRIDE
        if obs[base] > 0.5:
            total += float(obs[base + 1]) * FIFO_TONS_SCALE
    return total


def classify_window(window, lower, upper, eps):
    ore, grade = float(window["ore_tons"]), float(window["grade"])
    if ore <= 0.0:
        return dict(failure_type="empty", direction="empty", below_margin=0.0,
                    above_margin=0.0, violated=True)
    below = max(0.0, (lower - eps) - grade)
    above = max(0.0, grade - (upper + eps))
    direction = "below" if below > 0 else ("above" if above > 0 else "within")
    return dict(failure_type=("below_lower" if below > 0 else
                              "above_upper" if above > 0 else "none"),
                direction=direction, below_margin=below, above_margin=above,
                violated=bool(below > 0 or above > 0))


def collect_shift(env, evaluation_actor, expected):
    """One complete episode. Derives the config from the RPC payload and proves it matches."""
    observation, info = env.reset()
    config = AdapterConfiguration.from_rpc(info["reward_components"])
    scenario = metadata(from_components(info["reward_components"]))
    if expected["configuration_fingerprint"] is None:
        expected["configuration_fingerprint"] = config.fingerprint
        expected["scenario"] = scenario
    require(config.fingerprint == expected["configuration_fingerprint"],
            "RPC reward configuration differs from the training checkpoint")
    require(scenario == expected["scenario"], "RPC scenario differs from the training checkpoint")
    require(info["reward_contract_id"] == expected["reward_contract_id"],
            "RPC reward contract differs from the training checkpoint")
    adapter = RewardCostAdapter(config)
    adapter.start(info)
    identity = (info["episode_id"], info["experiment_run_index"], info["experiment_seed"])
    actions, windows, transitions, wait = [], [], 0, 0.0
    terminal_info, grade_cost_total, ore_cost, waste_cost = None, 0.0, 0.0, 0.0
    while True:
        mask = list(info["action_mask"])
        action, probs = evaluation_actor.act(observation, mask)
        require(mask[action] == 1, "selected a masked-out action")
        evaluation_actor.assert_frozen()
        next_observation, _raw_reward, terminated, truncated, next_info = env.step(action)
        adapted = adapter.transition(next_info, terminated=terminated, truncated=truncated)
        actions.append(action)
        transitions += 1
        grade_cost_total += adapted["grade_cost"]
        wait = float(next_info["reward_components"]["wait_vehicle_seconds"])
        for window in next_info["closed_windows"]:
            verdict = classify_window(window, config.get("grade_lower"), config.get("grade_upper"),
                                      config.get("grade_epsilon"))
            require(bool(window["violation_upper"]) == (verdict["direction"] == "above"),
                    "ALP upper-violation flag disagrees with RPC bounds")
            require(bool(window["violation_lower"]) == (verdict["direction"] in ("below", "empty")),
                    "ALP lower-violation flag disagrees with RPC bounds")
            windows.append(dict(window, **verdict))
        if terminated or truncated:
            require(terminated and not truncated, "evaluation requires a normal terminal")
            ore_cost, waste_cost = adapted["ore_cost"], adapted["waste_cost"]
            terminal_info = next_info
            terminal_observation = next_observation
            break
        observation, info = next_observation, next_info
    require(len(windows) == 4, "a scored shift requires exactly four windows")
    terminal_observation = np.asarray(terminal_observation, dtype=np.float32)
    validate_observation(OBSERVATION_SCHEMA_ID, terminal_observation, terminal=True)
    components = terminal_info["reward_components"]
    counts = [actions.count(i) for i in range(4)]
    return dict(
        episode_id=identity[0], experiment_run_index=identity[1], experiment_seed=identity[2],
        transitions=transitions, actions=actions, action_counts=counts,
        windows=windows, grade_cost=float(grade_cost_total), ore_cost=float(ore_cost),
        waste_cost=float(waste_cost),
        joint_success=bool(grade_cost_total == 0.0 and ore_cost == 0.0 and waste_cost == 0.0),
        crushed_ore_tons=float(components["ore_tons_shift"]),
        waste_delivered_tons=float(components["waste_tons_shift"]),
        wait_vehicle_seconds=wait,
        fifo_remaining_tons=fifo_remaining(terminal_observation),
        crusher_buffer_tons=float(components.get("crusher_buffer_tons", float("nan"))),
        window_violations=sum(1 for w in windows if w["violated"]),
        terminal_sim_time=float(terminal_info["sim_time_seconds"]),
        configuration_fingerprint=config.fingerprint, scenario=scenario,
        reward_contract_id=info["reward_contract_id"],
        observation_schema_id=info["observation_schema_id"])


# ------------------------------------------------------------------ outputs
SHIFT_COLUMNS = ["mode", "optimizer_steps", "policy_version", "run_index", "experiment_seed",
                 "episode_id", "transitions", "grade_cost", "ore_cost", "waste_cost",
                 "joint_success", "window_violations", "crushed_ore_tons", "waste_delivered_tons",
                 "wait_vehicle_seconds", "fifo_remaining_tons", "crusher_buffer_tons",
                 "action_0", "action_1", "action_2", "action_3", "terminal_sim_time"]
WINDOW_COLUMNS = ["mode", "run_index", "experiment_seed", "window_index", "window_start_seconds",
                  "window_end_seconds", "ore_tons", "metal_tons", "grade", "violated",
                  "failure_type", "direction", "below_margin", "above_margin",
                  "alp_violation_lower", "alp_violation_upper"]


def write_csv(path, columns, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize(mode, shifts, identity):
    windows = [w for shift in shifts for w in shift["windows"]]
    waits = [s["wait_vehicle_seconds"] for s in shifts]
    per_window = []
    for index in range(4):
        subset = [w for w in windows if int(w["window_index"]) == index]
        per_window.append(dict(
            window_index=index, n=len(subset),
            violation_rate=float(np.mean([w["violated"] for w in subset])),
            empty_rate=float(np.mean([w["failure_type"] == "empty" for w in subset])),
            below_rate=float(np.mean([w["failure_type"] == "below_lower" for w in subset])),
            above_rate=float(np.mean([w["failure_type"] == "above_upper" for w in subset])),
            mean_grade=float(np.mean([w["grade"] for w in subset])),
            mean_ore_tons=float(np.mean([w["ore_tons"] for w in subset])),
            max_below_margin=float(np.max([w["below_margin"] for w in subset])),
            max_above_margin=float(np.max([w["above_margin"] for w in subset]))))
    failure_types = {}
    for w in windows:
        failure_types[w["failure_type"]] = failure_types.get(w["failure_type"], 0) + 1
    action_counts = [int(sum(s["action_counts"][i] for s in shifts)) for i in range(4)]
    total_actions = sum(action_counts)
    sorted_waits = sorted(waits)
    return dict(
        eval_contract=EVAL_CONTRACT, mode=mode, optimizer_steps=0,
        policy_version=identity["policy_version"], actor_digest=identity["actor_digest"],
        checkpoint=identity["checkpoint"], checkpoint_sha256=identity["checkpoint_sha256"],
        shifts=len(shifts), transitions=int(sum(s["transitions"] for s in shifts)),
        seeds=[s["experiment_seed"] for s in shifts],
        mean_grade_cost=float(np.mean([s["grade_cost"] for s in shifts])),
        grade_failed_shifts=int(sum(s["grade_cost"] > 0 for s in shifts)),
        mean_ore_cost=float(np.mean([s["ore_cost"] for s in shifts])),
        ore_failed_shifts=int(sum(s["ore_cost"] > 0 for s in shifts)),
        mean_waste_cost=float(np.mean([s["waste_cost"] for s in shifts])),
        waste_failed_shifts=int(sum(s["waste_cost"] > 0 for s in shifts)),
        joint_success_rate=float(np.mean([s["joint_success"] for s in shifts])),
        four_window_violation_rate=float(np.mean([w["violated"] for w in windows])),
        window_failure_types=failure_types,
        per_window=per_window,
        wait=dict(mean=float(np.mean(waits)), median=float(statistics.median(waits)),
                  stdev=float(np.std(waits, ddof=1)) if len(waits) > 1 else 0.0,
                  min=float(min(waits)), max=float(max(waits)),
                  p25=float(np.percentile(sorted_waits, 25)),
                  p75=float(np.percentile(sorted_waits, 75)),
                  total=float(sum(waits))),
        crushed_ore=dict(mean=float(np.mean([s["crushed_ore_tons"] for s in shifts])),
                         min=float(min(s["crushed_ore_tons"] for s in shifts)),
                         max=float(max(s["crushed_ore_tons"] for s in shifts)),
                         target=16000.0),
        waste=dict(mean=float(np.mean([s["waste_delivered_tons"] for s in shifts])),
                   min=float(min(s["waste_delivered_tons"] for s in shifts)),
                   max=float(max(s["waste_delivered_tons"] for s in shifts)),
                   band=[3999.999, 4193.001]),
        fifo_remaining=dict(mean=float(np.mean([s["fifo_remaining_tons"] for s in shifts])),
                            min=float(min(s["fifo_remaining_tons"] for s in shifts)),
                            max=float(max(s["fifo_remaining_tons"] for s in shifts))),
        action_counts=dict(total=total_actions,
                           counts={str(i): action_counts[i] for i in range(4)},
                           fractions={str(i): (action_counts[i] / total_actions if total_actions else 0.0)
                                      for i in range(4)}),
        identity=identity)


def baseline_comparison(mode_summary, baseline_path, seed_summary_path):
    """Compare against the frozen Policy4 baseline, declaring pairing honestly."""
    if not baseline_path.exists():
        return dict(available=False, reason=f"baseline missing: {baseline_path}")
    rows = list(csv.DictReader(baseline_path.open()))
    seeds = set(mode_summary["seeds"])
    overlap = [r for r in rows if int(r["experiment_seed"]) in seeds]
    baseline_manifest = json.loads((baseline_path.parent / "manifest.json").read_text())
    evidence = baseline_manifest.get("input_files_sha256", {})
    alp_pin = None
    for key, value in evidence.items():
        if key.endswith("DRL-PPO.alp"):
            alp_pin = value
    current_alp = hashlib.sha256((Path(__file__).resolve().parents[1] / "anylogic/DRL-PPO.alp")
                                 .read_bytes()).hexdigest()
    comparison = dict(
        available=True, baseline=str(baseline_path),
        baseline_status=baseline_manifest.get("status"),
        baseline_shifts_total=len(rows),
        baseline_seeds_requested=sorted(seeds),
        baseline_seeds_found=sorted(int(r["experiment_seed"]) for r in overlap),
        seed_paired=len(overlap) == len(seeds),
        baseline_alp_sha256=alp_pin, current_alp_sha256=current_alp,
        alp_build_identical=(alp_pin == current_alp),
        baseline_valid_column_all_false=all(r["valid"] == "False" for r in overlap))
    if not overlap:
        comparison["verdict"] = "non-paired: no shared seeds; no advantage may be claimed"
        return comparison
    def numbers(column):
        return [float(r[column]) for r in overlap if r.get(column) not in (None, "")]
    comparison["baseline"] = dict(
        joint_met_shifts=sum(1 for r in overlap if r["observed_jointMet"] == "True"),
        window_violations_total=sum(int(float(r["windowViolations"] or 0)) for r in overlap),
        crushed_ore_mean=float(np.mean(numbers("crushedOre"))),
        waste_delivered_mean=float(np.mean(numbers("wasteDelivered"))),
        wait_mean=float(np.mean(numbers("waitVehicleSeconds"))),
        window_satisfaction_mean=float(np.mean(numbers("observed_window_satisfaction"))))
    comparison["ours"] = dict(
        joint_met_shifts=int(round(mode_summary["joint_success_rate"] * mode_summary["shifts"])),
        window_violations_total=int(sum(mode_summary["window_failure_types"].values())
                                    - mode_summary["window_failure_types"].get("none", 0)),
        crushed_ore_mean=mode_summary["crushed_ore"]["mean"],
        waste_delivered_mean=mode_summary["waste"]["mean"],
        wait_mean=mode_summary["wait"]["mean"],
        window_satisfaction_mean=1.0 - mode_summary["four_window_violation_rate"])
    if not comparison["alp_build_identical"]:
        comparison["verdict"] = ("seed-paired only; the baseline ran on a different ALP build, "
                                 "so differences may come from the model rather than the policy. "
                                 "No advantage is claimed.")
    else:
        comparison["verdict"] = ("seed-paired on the same ALP build; still a single 8-shift sample, "
                                 "descriptive only")
    comparison["baseline_valid_evidence_gap"] = comparison.pop("baseline_valid_column_all_false")
    return comparison


def report(mode_summary, comparison, disjoint, baseline_notes):
    lines = [f"# B3 固定策略 Holdout 评估（{mode_summary['mode']} 模式）", "",
             f"- 评估契约：`{EVAL_CONTRACT}`；执行模式：**{mode_summary['mode']}**（两种模式不得混合统计）",
             f"- Checkpoint：`{mode_summary['checkpoint']}`",
             f"- Checkpoint SHA256：`{mode_summary['checkpoint_sha256']}`",
             f"- Actor 摘要 / 策略版本：`{mode_summary['actor_digest']}` / `{mode_summary['policy_version']}`",
             f"- optimizer_steps：**0**（推理入口不构造优化器）",
             f"- Holdout episodes：{mode_summary['shifts']}，transitions {mode_summary['transitions']}",
             f"- Seeds：{mode_summary['seeds']}",
             f"- 与训练 seed 不相交：{disjoint['disjoint']}（训练集 {disjoint['training_seeds']}）", "",
             "## 汇总", "",
             "| 指标 | 值 |", "|---|---|",
             f"| 四窗口违规率 | {mode_summary['four_window_violation_rate']:.4f} |",
             f"| 班次 Grade Cost 均值（失败班数） | {mode_summary['mean_grade_cost']:.4f}（{mode_summary['grade_failed_shifts']}/{mode_summary['shifts']}） |",
             f"| 班次 Ore Cost 均值（失败班数） | {mode_summary['mean_ore_cost']:.4f}（{mode_summary['ore_failed_shifts']}/{mode_summary['shifts']}） |",
             f"| 班次 Waste Cost 均值（失败班数） | {mode_summary['mean_waste_cost']:.4f}（{mode_summary['waste_failed_shifts']}/{mode_summary['shifts']}） |",
             f"| 联合成功率 | {mode_summary['joint_success_rate']:.4f} |",
             f"| 累计车辆等待 均值/中位/P25/P75/标准差 | {mode_summary['wait']['mean']:.1f} / {mode_summary['wait']['median']:.1f} / {mode_summary['wait']['p25']:.1f} / {mode_summary['wait']['p75']:.1f} / {mode_summary['wait']['stdev']:.1f} |",
             f"| 累计车辆等待 最小/最大 | {mode_summary['wait']['min']:.1f} / {mode_summary['wait']['max']:.1f} |",
             f"| 破碎矿石 均值/最小/最大（目标 16000） | {mode_summary['crushed_ore']['mean']:.1f} / {mode_summary['crushed_ore']['min']:.1f} / {mode_summary['crushed_ore']['max']:.1f} |",
             f"| 废石交付 均值/最小/最大（区间 3999.999–4193.001） | {mode_summary['waste']['mean']:.1f} / {mode_summary['waste']['min']:.1f} / {mode_summary['waste']['max']:.1f} |",
             f"| FIFO 剩余 均值/最小/最大 | {mode_summary['fifo_remaining']['mean']:.1f} / {mode_summary['fifo_remaining']['min']:.1f} / {mode_summary['fifo_remaining']['max']:.1f} |", "",
             "## 四窗口违规率与失败类型", "",
             "| 窗口 | 违规率 | 空窗口 | 低于下限 | 高于上限 | 平均品位 | 平均吨位 | 最大低于幅度 | 最大高于幅度 |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in mode_summary["per_window"]:
        lines.append(f"| {row['window_index']} | {row['violation_rate']:.3f} | {row['empty_rate']:.3f} | "
                     f"{row['below_rate']:.3f} | {row['above_rate']:.3f} | {row['mean_grade']:.5f} | "
                     f"{row['mean_ore_tons']:.1f} | {row['max_below_margin']:.6f} | {row['max_above_margin']:.6f} |")
    lines += ["", f"失败类型计数（全部窗口）：{mode_summary['window_failure_types']}", "",
              "## 动作选择次数", "",
              "| 动作（装载点） | 次数 | 占比 |", "|---:|---:|---:|"]
    for index in range(4):
        lines.append(f"| {index} | {mode_summary['action_counts']['counts'][str(index)]} | "
                     f"{mode_summary['action_counts']['fractions'][str(index)]:.4f} |")
    lines += ["", "## 与 Policy4 Baseline 对照", ""]
    if not comparison.get("available"):
        lines.append(f"Baseline 不可用：{comparison.get('reason')}")
    else:
        paired = "配对（同 seed 集）" if comparison["seed_paired"] else "**非配对**"
        lines += [f"- 配对状态：{paired}",
                  f"- Baseline：`{comparison['baseline']}`（status `{comparison['baseline_status']}`，共 {comparison['baseline_shifts_total']} 班）",
                  f"- 共享 seed：{comparison['baseline_seeds_found']}",
                  f"- ALP 构建：baseline `{str(comparison['baseline_alp_sha256'])[:16]}…` vs 当前 `{comparison['current_alp_sha256'][:16]}…` → 相同={comparison['alp_build_identical']}",
                  f"- Baseline 证据缺口：Policy4 的 `valid` 列全为 False（缺独立 RPC 等待 delta 与控制台双源证据），因此其业务指标只能算观察值",
                  "", f"**结论口径：{comparison['verdict']}**", ""]
        if comparison.get("baseline"):
            lines += ["| 指标 | 本策略 | Policy4 Baseline |", "|---|---:|---:|",
                      f"| 联合达标班数 | {comparison['ours']['joint_met_shifts']}/{mode_summary['shifts']} | {comparison['baseline']['joint_met_shifts']}/{len(comparison['baseline_seeds_found'])} |",
                      f"| 窗口违规总数 | {comparison['ours']['window_violations_total']} | {comparison['baseline']['window_violations_total']} |",
                      f"| 窗口满足率 | {comparison['ours']['window_satisfaction_mean']:.4f} | {comparison['baseline']['window_satisfaction_mean']:.4f} |",
                      f"| 破碎矿石均值 (t) | {comparison['ours']['crushed_ore_mean']:.1f} | {comparison['baseline']['crushed_ore_mean']:.1f} |",
                      f"| 废石交付均值 (t) | {comparison['ours']['waste_delivered_mean']:.1f} | {comparison['baseline']['waste_delivered_mean']:.1f} |",
                      f"| 平均等待 (vehicle·s) | {comparison['ours']['wait_mean']:.1f} | {comparison['baseline']['wait_mean']:.1f} |", ""]
    lines += ["## 身份与不变量", "",
              "| 项 | 值 |", "|---|---|",
              f"| 观测 schema / 维度 | `{mode_summary['identity']['observation_schema_id']}` / {mode_summary['identity']['observation_size']} |",
              f"| Policy 契约 | `{mode_summary['identity']['policy_contract_id']}` |",
              f"| 预处理 | `{mode_summary['identity']['preprocessing']['contract']}`（{mode_summary['identity']['preprocessing']['inference_mode']}） |",
              f"| 训练配置指纹 | `{mode_summary['identity']['expected_configuration_fingerprint']}` |",
              f"| 训练 scenario | `{json.dumps(mode_summary['identity']['expected_scenario'], ensure_ascii=False)}` |",
              f"| 训练输出目录 | `{mode_summary['identity']['training_output_dir']}` |",
              f"| 乘子（未改动） | {mode_summary['identity']['lambdas']} |",
              f"| 已消费批次（未再消费） | {mode_summary['identity']['consumed_batches']} |", ""]
    if baseline_notes:
        lines += ["## 说明", ""] + [f"- {note}" for note in baseline_notes] + [""]
    return "\n".join(lines)


# --------------------------------------------------------------------- main
def run_evaluation(args, client=None):
    actor, identity = load_actor(args.checkpoint)
    holdout = load_holdout_seeds(args.seed_manifest)
    disjoint = assert_disjoint_from_training(holdout, args.training_manifest)
    evaluation_actor = EvaluationActor(actor, args.mode)
    torch.manual_seed(args.sampler_seed)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "entry.json", dict(
        eval_contract=EVAL_CONTRACT, mode=args.mode, sampler_seed=args.sampler_seed,
        checkpoint=str(args.checkpoint), seed_manifest=str(args.seed_manifest),
        seeds=[{"run_index": i, "experiment_seed": s} for i, s in holdout],
        training_manifests=[str(p) for p in args.training_manifest],
        baseline_shifts=str(args.baseline_shifts),
        started_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), source_kind=(
            "synthetic_protobuf_client" if client is not None else "real_anylogic_rpc")))

    bridge = ConstrainedBridge(output, holdout)
    env = DispatchEnv(bridge, ParameterVariationEpisodeRunner(len(holdout)),
                      callback_timeout=args.callback_timeout)
    server = create_server(bridge, host=args.host, port=args.port)
    server.start()
    print(f"[eval:{args.mode}] listening on {args.host}:{args.port}; "
          f"{len(holdout)} holdout shifts", flush=True)

    client_thread = None
    if client is not None:
        import threading
        client_thread = threading.Thread(target=client, args=(bridge,), daemon=True)
        client_thread.start()

    expected = dict(configuration_fingerprint=identity["expected_configuration_fingerprint"],
                    scenario=identity["expected_scenario"],
                    reward_contract_id=None)
    # The checkpoint records a reward contract id in its batch metadata; recover it from
    # the first callback and require the frozen V1.0 contract.
    from .reward_contract import REWARD_CONTRACT_ID
    expected["reward_contract_id"] = REWARD_CONTRACT_ID

    actor_digest_before = actor.digest
    shifts = []
    try:
        for _ in range(len(holdout)):
            shifts.append(collect_shift(env, evaluation_actor, expected))
            print(f"[eval:{args.mode}] shift {len(shifts)}/{len(holdout)} "
                  f"seed={shifts[-1]['experiment_seed']} "
                  f"grade={shifts[-1]['grade_cost']:g} ore={shifts[-1]['ore_cost']:g} "
                  f"waste={shifts[-1]['waste_cost']:g}", flush=True)
    finally:
        if client_thread is not None:
            client_thread.join(timeout=60)
        env.close()
        server.stop(grace=2).wait()

    evaluation_actor.assert_frozen()
    require(actor.digest == actor_digest_before, "policy parameters changed during evaluation")

    summary = summarize(args.mode, shifts, identity)
    summary.update(disjoint=disjoint, sampler_seed=args.sampler_seed,
                   configuration_fingerprint=expected["configuration_fingerprint"],
                   scenario=expected["scenario"], reward_contract_id=expected["reward_contract_id"],
                   actor_digest_unchanged=(actor.digest == actor_digest_before),
                   optimizer_steps=0, policy_parameters_updated=False)

    shift_rows = [dict(mode=args.mode, optimizer_steps=0, policy_version=identity["policy_version"],
                       run_index=shift["experiment_run_index"],
                       experiment_seed=shift["experiment_seed"], episode_id=shift["episode_id"],
                       transitions=shift["transitions"], grade_cost=shift["grade_cost"],
                       ore_cost=shift["ore_cost"], waste_cost=shift["waste_cost"],
                       joint_success=shift["joint_success"],
                       window_violations=shift["window_violations"],
                       crushed_ore_tons=shift["crushed_ore_tons"],
                       waste_delivered_tons=shift["waste_delivered_tons"],
                       wait_vehicle_seconds=shift["wait_vehicle_seconds"],
                       fifo_remaining_tons=shift["fifo_remaining_tons"],
                       crusher_buffer_tons=shift["crusher_buffer_tons"],
                       action_0=shift["action_counts"][0], action_1=shift["action_counts"][1],
                       action_2=shift["action_counts"][2], action_3=shift["action_counts"][3],
                       terminal_sim_time=shift["terminal_sim_time"])
                   for shift in shifts]
    window_rows = [dict(mode=args.mode, run_index=shift["experiment_run_index"],
                        experiment_seed=shift["experiment_seed"],
                        window_index=window["window_index"],
                        window_start_seconds=window["window_start_seconds"],
                        window_end_seconds=window["window_end_seconds"],
                        ore_tons=window["ore_tons"], metal_tons=window["metal_tons"],
                        grade=window["grade"], violated=window["violated"],
                        failure_type=window["failure_type"], direction=window["direction"],
                        below_margin=window["below_margin"], above_margin=window["above_margin"],
                        alp_violation_lower=window["violation_lower"],
                        alp_violation_upper=window["violation_upper"])
                    for shift in shifts for window in shift["windows"]]

    write_csv(output / "shifts.csv", SHIFT_COLUMNS, shift_rows)
    write_csv(output / "windows.csv", WINDOW_COLUMNS, window_rows)
    comparison = baseline_comparison(summary, Path(args.baseline_shifts),
                                    Path(args.baseline_shifts).parent / "seed_summary.csv")
    notes = [
        "两种执行模式（random / argmax）必须分别运行、分别出表，不做合并统计。",
        "random 模式的动作来自掩码类别分布抽样，sampler_seed 已记录且仅在评估期使用。",
        "argmax 模式在掩码后 logits 上取确定性动作。",
        "评估入口不构造优化器，optimizer_steps=0，Actor 参数在整轮评估前后摘要一致。",
        "Policy4 baseline 的 valid 列全为 False 是证据缺口（缺独立 RPC 等待 delta 与控制台双源），"
        "不代表业务失败；其 observed_* 列只是观察值。",
    ]
    summary["baseline_comparison"] = comparison
    write_json(output / "summary.json", summary)
    (output / "report.md").write_text(report(summary, comparison, disjoint, notes) + "\n")
    print(json.dumps(dict(mode=args.mode, shifts=summary["shifts"],
                          joint_success_rate=summary["joint_success_rate"],
                          four_window_violation_rate=summary["four_window_violation_rate"],
                          wait_mean=summary["wait"]["mean"],
                          optimizer_steps=0,
                          output=str(output)), ensure_ascii=False, indent=2))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed-manifest", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--baseline-shifts", type=Path,
                        default=Path("runs/policy4-statistics-20261026-f17d11f/shifts.csv"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--callback-timeout", type=float, default=900.0)
    parser.add_argument("--sampler-seed", type=int, default=20261026)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    run_evaluation(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
