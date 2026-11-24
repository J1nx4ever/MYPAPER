# Fig. 1–6 数据与验证

本页保留首次重绘的输入和验证记录。[Fig. 1、2、3、6 最终重绘记录](final_20261122/README.md)

[重绘版图表预览](../../../../../summary/interpretability_eval/README.md)

主样本沿用 16 个 Nominal 和 Swapped 历史班次的 2,940 条决策，含 2,220 条 BG128 和 720 条 Flat-PPO。Fig. 3 的干预使用冻结的三个 BG128 checkpoint，Fig. 6 沿用经归档认证的 Swapped 班次 10001327 第 70 次决策。

| 图 | 数据范围与结果 |
| --- | --- |
| Fig. 1 | 电铲总排队数包括普通队列和 Seize 资源等待，每台电铲内按车号去重。共 30 条决策达到拥堵阈值。落入统计工况的 2,048 条不同决策均没有当前卡车直连合法候选电铲的事实边。 |
| Fig. 2 | 按电铲总排队数着色并预测 +15 分钟的电铲总排队。两种输入在三个训练种子的样本上均得到有限的队列 $R^2$，平均为 C128 −2.448、原始观测 −3.578，平均 RMSE 分别为 0.728、0.840 辆。 |
| Fig. 3 | A、C 各用 100 个快照。B 用 25 个班次的 50 个快照，逐点验证目标电铲恰有 0 至 5 条前向 `reserved_load` 边及匹配的反向边。改写途中的路线和预约字段，保留原动作掩码及其可派吨位。C 从矿石已达标、剥离不足的第四窗口实测状态中抽样，按班长 480 分钟和窗口 120 分钟的归一化口径扫描剩余时间，保持原产量不变。 |
| Fig. 4 | 以班次内车型均值作为 Kruskal–Wallis 检验输入，对四项检验作 Holm 校正。品位临界和品位充裕的高品位派车率校正后 $p$ 为 0.425 和 0.962，运距为 0.962，预计金属贡献为 6.94×10⁻⁷。品位临界组按 Cat777、Cat785、Cat789 计有 9、10、9 个班次级观测，其他三项各为 12、12、12。 |
| Fig. 5 | 封存 PROCESS 区间给出精确 $t+20$ 分钟的同班滚动窗口。与 Greedy 一致、分歧组的平均裕度变化为 +0.094、+0.134 个百分点，样本数 597、1,426。与 RHO 一致、分歧组为 −0.330、+0.237 个百分点，样本数 48、124。两组都执行了 BG128，现有轨迹支持描述性比较。 |
| Fig. 6 | 第 2 子图仅使用 Truck1→Load0 和 Truck1→Load2 的第二层实际有向边。八个组别均标为 `not_in_graph`，原图 Material 边只连接 Load 与 Dump。不能比较指定节点之间的 Material 系数。 |

绘图输入为 [统计数据](stats.json)、[线性探针](linear_probing.json)、[干预数据](counterfactual.json) 和 [案例数据](case_for_figure.json)。补充快照来自 [B 班次清单](b_supplement_sources.json) 和 [C 班次清单](c_supplement_sources.json)。

B 的 50 个合格样本覆盖正常 47 条、品位临界 3 条，目标电铲的平均选择概率随 Commitment 边数 0 至 5 从 34.95% 降至 2.29%。C 的剥离选择概率在 120 分钟和 15 分钟时分别为 28.26% 和 28.97%，区间内为 27.19% 至 29.17%。

六份 PDF 均已渲染核对，六份 PNG 均为 300 dpi。受保护的 `python/episode_budget_ppo.py` 前后 SHA256 均为 `8e4cfa31c622f5e33756e304eea3f7526547db72ae93f7d9f00808845d61ef96`。

从仓库根目录使用已安装的绘图依赖复现。

```bash
EVAL=summary/interpretability_eval
INPUT=experiments/lagrangian_ppo_v1/reports/interpretability_eval/figure11221215
PYTHONPATH=/tmp/faith-interpretability-deps /home/jinx/miniconda3/envs/anylogic-env/bin/python -B \
    "$EVAL/06_plot_publication_figs.py" --stats "$INPUT/stats.json" \
    --probes "$INPUT/counterfactual.json" --linear "$INPUT/linear_probing.json" \
    --case "$INPUT/case_for_figure.json" --output-dir "$EVAL/figure11221215"
```
