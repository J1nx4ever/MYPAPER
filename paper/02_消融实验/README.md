# 消融实验

五组结构对照按 s01 训练日程比较。传播深度先核对 s01–s03，再追加 s04、s05，现有五对训练日程。重复出现在两项比较中的 s01 模型沿用相同的物理评测班。

- [Full、Flat、无 C128、定向屏蔽与一层模型的封存结果](封存摘要/gat_weak_ablation_001/final_ablation_comparison_20261123.json)
- [首批一层与两层三个种子的分项统计](封存摘要/gat_weak_ablation_001/seed_replication_001/paired_evaluation_summary.json)
- [Flat 与无 C128 的方法定义](方法与实现/structural_ablation_001/README.md)
- [方向屏蔽与层数消融的方法定义](方法与实现/gat_weak_ablation_001/README.md)

## 五训练日程扩展

两层 Full 与一层 GAT 加逐节点 MLP 的五留出工况达标为 2,464/2,500 与 2,418/2,500。按训练日程配对，一层减两层的差为 −1.84 个百分点，95% 区间 [−5.66, +1.98]。五个日程的推断使用五对重复，固定权重下的评测精度另按 100 个共享槽重采样。新增四个模型的训练、逐工况指标和汇总见 M21–M26。

- [五对训练日程的结果与区间](封存摘要/gat_weak_ablation_001/seed_extension_001/statistics_001/report_zh.md)
- [五个 Full 的标称百槽配对](封存摘要/gat_weak_ablation_001/seed_extension_001/statistics_001/full_nominal_wilcoxon.md)
- [新训练与固定评测的代码和协议](方法与实现/gat_weak_ablation_001/seed_extension_001/README.md)
