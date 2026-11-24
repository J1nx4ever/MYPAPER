# BG128 四张图的最终重绘记录

[Fig. 1–6 预览](../../../../../../summary/interpretability_eval/README.md)

本次重绘 Fig. 1、2、3、6，使用与前版相同的历史观测、冻结模型和案例。

## Fig. 1 单边平均注意力

在实际有向边上先平均四个头，再按工况和边组对全部实测边取均值。Transport、Commitment 取第二层，Service、Material 取第一层。正常、品位临界和拥堵工况分别有 1,858、190 和 30 条决策，工况允许重叠，12 个工况与边组组合均有边。Material 在三组中的均值均为 0.750。冻结注意力按目标节点、关系组和头分别归一化，因此组间均值受各归一化组的入边数量影响。[统计数据](stats.json)

## Fig. 2 独立测试集误差

保留原 Ridge 拟合、留出划分和 UMAP。右下显示 $100(1-\mathrm{RMSE}_{\mathrm{model}}/\mathrm{RMSE}_{\mathrm{baseline}})$，基线固定预测训练集目标的均值。每个种子在同一测试样本上计算两个 RMSE，图中显示三个种子的平均值与样本标准差。较大的百分比表示误差较低，负值表示该模型的误差高于均值基线。

| 目标 | C128 的 RMSE 降幅 | 原始观测的 RMSE 降幅 | Pearson $r$（C128 / 原始观测） |
| --- | ---: | ---: | ---: |
| 电铲总排队 +15 min | −77.99% | −112.92% | 0.0814 / −0.0008 |
| 滚动品位裕度 +30 min | −2.41% | −18.64% | 0.3805 / 0.4530 |

C128 在两个目标上的 RMSE 都低于 raw。两种表示的 RMSE 都高于均值基线，品位目标的 Pearson 相关性以 raw 较高。[线性探针数据](linear_probing.json)

## Fig. 3 全局时钟注入

C 在冻结 Actor 的 `TokenEncoder.global_final` 投影输入显式注入环境定义的班次已用、班次剩余和窗口剩余特征。班次剩余时间从 120 min 降至 15 min，归一化值为 0.25 至 0.03125。三个种子各验证 2,200 行，确认每行更新后的 `Global` token 都传入两层 R-GAT。该节点没有物理消息边，模型将它直接读入 Actor 上下文。此前的 raw 时钟已经到达这一投影，显式注入后原曲线保留，端点为 28.26% 和 28.97%。[干预与注入验证](counterfactual.json)

## Fig. 6 两跳路径分数

案例为历史 Swapped 班次 10001327 第 70 次决策。冻结图中没有 Load0 或 Load2 指向 Truck1 的直接边，因此经电铲再直达 Truck1 的路径也不存在。图中比较实际存在的转发路径，对每条路径先计算同一头的第一、二层系数乘积，再平均四个头。

| 边组 | Load0 | Load2 | 实际路径 |
| --- | ---: | ---: | --- |
| Material | 0.003155 | 0.003398 | Load→Dump0→Truck1 |
| Commitment | 缺失 | 0.231061 | Load2→Truck2→Truck1 |
| Transport | 缺失 | 缺失 | — |
| Service | 缺失 | 缺失 | — |

同一案例中 Load2 的 Material 路径分数比 Load0 高 7.69%。系数乘积描述注意力沿路径的权重。模型还含跨头线性投影、残差与归一化，单案例的路径分数不能确定对决策概率的因果效应或统计显著性。[案例路径与逐头系数](case_for_figure.json)

四份 PDF 已逐页渲染核对，四份 PNG 均通过 300 dpi 检查。Fig. 4 和 Fig. 5 的现有文件均通过原字节 SHA256 复核。受保护的 `python/episode_budget_ppo.py` 前后 SHA256 均为 `8e4cfa31c622f5e33756e304eea3f7526547db72ae93f7d9f00808845d61ef96`。

## 重现导出

```bash
EVAL=summary/interpretability_eval
INPUT=experiments/lagrangian_ppo_v1/reports/interpretability_eval/figure11221215/final_20261122
PYTHONPATH=/tmp/faith-interpretability-deps /home/jinx/miniconda3/envs/anylogic-env/bin/python -B \
    "$EVAL/06_plot_publication_figs.py" --stats "$INPUT/stats.json" \
    --probes "$INPUT/counterfactual.json" --linear "$INPUT/linear_probing.json" \
    --case "$INPUT/case_for_figure.json" --figures 1 2 3 6 --output-dir "$EVAL/figure11221215"
```
