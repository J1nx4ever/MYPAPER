# BG128 图表的输入与验证

这是旧版图表输入归档。重绘版的脚本读取更新后的统计定义和数据格式。[重绘版输入与图表](../figure11221215/README.md)

[六张图与说明](../../../../../summary/interpretability_eval/README.md)

Fig. 1–5 的轨迹输入来自 16 个 Nominal 和 Swapped 历史班次。12 个 BG128 班次提供 2,220 条决策，4 个 Flat-PPO 班次提供 720 条决策。Fig. 6 使用独立的历史 Swapped 班次 10001327。

- [stats.json](stats.json) 提供 Fig. 1、4、5 的绘图数据。Fig. 5 的 Greedy 与 BG128 在 2,220 条有效同状态决策中分歧 1,560 次，RHO 在 191 条有效决策中分歧 137 次。图上柱高先按班次取均值，再按训练种子等权平均。RHO 从每个 BG128 班次的 16 个等距决策序号区间中各取中间一条。一次超过规则计划时限的结果剔除。此图使用同状态复算，归档没有用于事件对齐的 Greedy / RHO 决策级后效轨迹。[规则复算覆盖](rule_replay_coverage.json)
- [linear_probing.json](linear_probing.json) 提供 Fig. 2 的 2,093 条成对 UMAP 输入观测及 Ridge 指标。测试队列标签恒定时，测试队列的 $R^2$ 留空。
- [counterfactual.json](counterfactual.json) 提供 Fig. 3 的三项干预数据。每项使用 100 个快照和三份冻结 checkpoint，记录跨训练种子的 95% 置信区间。
- [case_for_figure.json](case_for_figure.json) 提供 Fig. 6 的实测时序和冻结模型诊断。相对于原筛选条件，该 Swapped 案例的远端电铲队列为 0。关联卸矿发生在派车后约 22.98 分钟，图中品位曲线为同班观测。

从仓库根目录使用已安装的绘图依赖重绘这批图。

```bash
EVAL=summary/interpretability_eval
INPUT=experiments/lagrangian_ppo_v1/reports/interpretability_eval/figures_20261122
PYTHONPATH=/tmp/faith-interpretability-deps /home/jinx/miniconda3/envs/anylogic-env/bin/python -B \
    "$EVAL/06_plot_publication_figs.py" --stats "$INPUT/stats.json" \
    --probes "$INPUT/counterfactual.json" --linear "$INPUT/linear_probing.json" \
    --case "$INPUT/case_for_figure.json" --output-dir "$EVAL/figures"
```
