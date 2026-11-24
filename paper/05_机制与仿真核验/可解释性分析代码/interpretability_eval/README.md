# BG128 机制与可解释性分析

使用冻结的 BG128 `s01`、`s02`、`s03` 的 u32 checkpoint 重放决策，测量图关系权重、C128 的预测能力、输入敏感性、车型分工和策略分歧。脚本默认离线运行。历史入口从归档读取实测决策、实际 RPC 配置和物料事件，并复核班次登记、数据回执、checkpoint 字节 SHA256、观测 schema 和 Action Mask。

## 文件与结果

| 脚本 | 分析 | 输出 |
| --- | --- | --- |
| [01_collect_rollouts.py](01_collect_rollouts.py) | 历史轨迹准入、冻结推理和可选同状态 Greedy / RHO 复算 | `rollouts.parquet` 或 `rollouts.npz` |
| [02_counterfactual_probe.py](02_counterfactual_probe.py) | M3 三项单变量输入敏感性，复用原始 Action Mask | `counterfactual.json` |
| [03_linear_probing.py](03_linear_probing.py) | M2 UMAP 与按完整班次隔离的 Ridge 线性探针 | `linear_probing.json` |
| [04_stats_and_divergence.py](04_stats_and_divergence.py) | M1 工况注意力和品位互换、M4 车型分工、M5 分歧与配对后效 | `stats.json` |
| [05_case_retrieval.py](05_case_retrieval.py) | M6 典型分歧案例、动作概率、在途态势和后续品位 | `case.json` |
| [06_plot_publication_figs.py](06_plot_publication_figs.py) | Fig. 1–6，Okabe–Ito 色板 | 矢量 PDF、300 dpi PNG、`figures.json` |

## 图表预览

本版图表位于 `figure11221215/`。Fig. 1、2、4、5 使用 16 个 Nominal 和 Swapped 历史班次的 2,940 条决策，覆盖 BG128 的三个训练种子及 Flat-PPO 的一个训练种子。Fig. 3 的可行干预快照及 Fig. 6 的实测案例另见绘图记录。

### Fig. 1 边组注意力与品位互换

![边组注意力与品位互换](figure11221215/Fig1.png)

拥堵定义为电铲总排队数 ≥ 2，共 30 条决策。左图按工况对真实有向边计算四头平均后的单边均值。Transport、Commitment 取第二层，Service、Material 取第一层。三种工况的四个边组均有观测。[矢量 PDF](figure11221215/Fig1.pdf)

### Fig. 2 C128 投影与未来状态线性预测

![C128 投影与未来状态线性预测](figure11221215/Fig2.png)

UMAP 使用 2,093 条成对观测。右下用训练集目标均值作为基线，比较两个 Ridge 探针在留出集的相对 RMSE 降幅。C128 与原始观测的排队降幅为 −77.99% 和 −112.92%，品位降幅为 −2.41% 和 −18.64%。C128 在两个任务的误差更小，两种表示的误差都高于均值基线。[矢量 PDF](figure11221215/Fig2.pdf)

### Fig. 3 三项输入干预

![三项输入干预](figure11221215/Fig3.png)

B 在 50 个实测快照上反事实重指派在途卡车。目标电铲收到的 Commitment 边数由 0 增至 5 时，选择概率由 34.95% 降至 2.29%。C 将 120 至 15 分钟对应的归一化时钟显式注入冻结模型的 `Global` 节点，验证更新后的 token 到达两层 R-GAT。100 个快照上的剥离选择概率端点为 28.26% 和 28.97%。[矢量 PDF](figure11221215/Fig3.pdf)

### Fig. 4 车型分工与载重贡献

![车型分工与载重贡献](figure11221215/Fig4.png)

每种车型在每个班次形成一个统计样本，用 Kruskal–Wallis 检验并对四项检验作 Holm 校正。左图品位临界和品位充裕组的校正后 $p$ 分别为 0.425 和 0.962，中图运距为 0.962。[矢量 PDF](figure11221215/Fig4.pdf)

### Fig. 5 同状态策略分歧

![同状态策略分歧](figure11221215/Fig5.png)

右图用物料事件重算同一 BG128 班次内的 $\Delta g(t+20\,\mathrm{min})-\Delta g(t)$。与 Greedy 一致、分歧组分别为 +0.094、+0.134 个百分点，与 RHO 一致、分歧组分别为 −0.330、+0.237 个百分点。两组均来自观察到的 BG128 轨迹。[矢量 PDF](figure11221215/Fig5.pdf)

### Fig. 6 单步派车案例

![单步派车案例](figure11221215/Fig6.png)

Swapped 班次 10001327 中，Greedy 选择 Load0，BG128 实际选择 Load2，约 22.98 分钟后卸矿。图中按实际边计算两跳分数。经 Dump0 到 Truck1 的 Material 分数，Load0 为 0.003155，Load2 为 0.003398，高 7.69%。该差异来自单个决策。[矢量 PDF](figure11221215/Fig6.pdf)

可用保存的数据重绘本次更新的 Fig. 1、2、3、6。[输入与验证记录](../../experiments/lagrangian_ppo_v1/reports/interpretability_eval/figure11221215/final_20261122/README.md)

## 运行方式

从仓库根目录运行以下命令。分析产物写入实验项目的 `reports/`，图表写入本目录的 `figure11221215/`。默认环境已有轨迹读取和 Ridge 所需依赖，UMAP 和绘图分别需要 `umap-learn` 和 `matplotlib`。可以把这两项安装到单独的临时目录，以保持 Conda 环境中用于历史回放的版本。

```bash
PY=/home/jinx/miniconda3/envs/anylogic-env/bin/python
EVAL=summary/interpretability_eval
OUT=experiments/lagrangian_ppo_v1/reports/interpretability_eval

"$PY" -m pip install --target /tmp/faith-interpretability-deps 'umap-learn>=0.5,<0.6' 'matplotlib>=3.8,<3.11'

"$PY" -B "$EVAL/01_collect_rollouts.py" --historical --rule-actions --output "$OUT/rollouts.parquet"
"$PY" -B "$EVAL/02_counterfactual_probe.py" "$OUT/rollouts.parquet" --output "$OUT/counterfactual.json"
PYTHONPATH=/tmp/faith-interpretability-deps "$PY" -B "$EVAL/03_linear_probing.py" \
    "$OUT/rollouts.parquet" --output "$OUT/linear_probing.json"
"$PY" -B "$EVAL/04_stats_and_divergence.py" "$OUT/rollouts.parquet" --output "$OUT/stats.json"
"$PY" -B "$EVAL/05_case_retrieval.py" "$OUT/rollouts.parquet" --output "$OUT/case.json"
PYTHONPATH=/tmp/faith-interpretability-deps "$PY" -B "$EVAL/06_plot_publication_figs.py" \
    --stats "$OUT/stats.json" --probes "$OUT/counterfactual.json" \
    --linear "$OUT/linear_probing.json" --case "$OUT/case.json" --output-dir "$EVAL/figure11221215"
```

初次核验可在 01 加 `--scenario Nominal --scenario Swapped --limit-per-group 2`，用独立输出目录。样本不足时，02 可降低 `--snapshot-count`，03 可降低 `--projection-count` 或用 `--skip-umap` 只算 Ridge。完整历史入口包含七个工况、四个策略与训练种子组合、每组每工况各 100 班，逐班准入和 `--rule-actions` 的规则复算会增加耗时。未启用规则复算时，同状态分歧和案例需要输入已经记录的规则动作。

## 数据约定

- 归档场景 `nominal`、`grade_swap`、`new_numeric` 分别映射到 `Nominal`、`Swapped`、`Shifted`。历史 Swapped 在开班前完成品位互换，因此 Fig. 1 右图从开班后第一个决策计数。Flat-PPO 对应历史 Flat-MLP + C128，只有 `s01`。
- Parquet 按班次写入，常用检索列单独存储，完整记录在 `record_json`。NPZ 使用压缩字符串数组，读取时关闭 pickle。核心字段包括班次、环境 seed、训练 seed、场景、仿真分钟、决策序号、1519 维原始观测、Action Mask、执行动作、候选电铲属性及冻结模型诊断。
- 01 的 `--manifest` 接受 `_sources.historical_sources()` 导出的来源清单。每个条目保留预先固定的回执和 checkpoint 路径及 SHA256、`policy`、`training_seed`、`scenario`、`scene_id`、`pair_ordinal`、`run_index` 和必要的 suite 路径及 SHA256。
- `grade_roll_pp` 从决策可见的 PROCESS 区间计算最近 120 分钟破碎矿石的金属量除以矿量。窗口边缘按 PROCESS 区间重叠时长折算。窗口没有矿量时留空。品位数值用百分数，品位裕度用百分点，距离用 km，时间用 min，吨位用 t。预期品位是滚动窗口内已破碎矿石与当前可见的在途已装矿车辆的质量加权混合值。
- `total_shovel_queue` 将每台电铲的普通队列与 Seize 资源等待按车号去重后求和。`grade_margin_plus20_pp` 在同一历史班次内通过封存 PROCESS 区间与 $t+20$ 分钟滚动窗口的重叠量重算。
- 01 的 `--telemetry` 可导入另行记录的 JSONL 或 Parquet。每行提供 `policy`、`training_seed`、`scenario`、`seed`、`shift_id`、`run_index`、`recording_id`、`decision_index`、`sim_time_min`、`observation_schema`、`raw_obs`、`action_mask`、`action`、`config_path` 和 `config_sha256`。相对路径以输入文件所在目录为起点。BG128 行提供 `checkpoint_path` 和 `checkpoint_sha256` 时会重放网络诊断。建议保留 `shovels`、`grade_min_pp`、`grade_roll_pp`、`grade_margin_pp`、`grade_expected_pp`、`crusher_queue` 等物理字段，以支持全部模块。
- 新的实物配对轨迹应在同一 `pair_id` 下同时保留场景、环境 seed 和策略来源。Greedy / RHO 的同状态选择写在 BG128 行的 `baseline_actions` 字典中，独立物理班的 Greedy / RHO 决策写成独立策略轨迹。线上采集需要按项目约定另行冻结并登记每个班次身份。

## 指标与解释

- **M1** 每条观测边先对四个头取均值，再按关系组和工况汇总。冻结模型对每个目标节点、关系组和头分别归一化。
- **M2** `summary_emb` 是冻结模型从当前观测构造的 C128 向量。UMAP 默认使用 `n_neighbors=30`、`min_dist=0.1`。Ridge 使用 `alpha=1.0`，按完整环境 seed 组切分，输出电铲总排队和滚动品位裕度的 Pearson $r$、RMSE 及相对均值基线的 RMSE 降幅。
- **M3** 三个训练种子的模型共用各项的合格快照，默认每项最多 100 个。A 扰动固定窗品位。B 在构建器中同步改写在途卡车的 Commitment 边及相关字段，并保留 Action Mask。C 保留实测产量，将归一化时钟直接写入 `Global` 投影输入。JSON 记录快照覆盖与节点注入的验证结果。
- **M4** 用可调度载重和选中电铲的品位估计金属贡献。距离为开放路网上从电铲到卸点的最短距离。每个车型的检验输入是各班次内的均值。
- **M5** 用同状态规则动作划分一致与分歧，再比较实际 BG128 轨迹的 20 分钟滚动品位裕度变化。要评估规则动作在同一状态下的后果，需要真实配对或状态分支轨迹。
- **M6** 检查候选节点流向当前卡车的实际有向边和两跳路径。两跳分数取两层同头系数的乘积，再平均各头。JSON 同时记录 Actor 概率、Critic 状态成本和后续观测。候选动作的即时回报及后继价值需要分支观测 `action_outcomes`。
