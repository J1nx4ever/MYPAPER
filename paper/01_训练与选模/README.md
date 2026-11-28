# 训练与选模

本目录收录早期 BG128 的训练、学习率配对、检查点独评与延训方法。用于论文正式对比的混合 Full s02/u32，来自三个 u32 模型在历史标称开发槽上的选模。混合训练和相应泛化材料集中在第四类。

- [正式训练方案](方法与实现/formal_training_001/README.md)
- [u13 固定评测方法](方法与实现/checkpoint_evaluation_001/README.md)
- [延训与 u32 评测方法](方法与实现/training_extension_001/README.md)
- [混合 Full 三种子训练与评测](../04_泛化与鲁棒性/封存摘要/generalization_001/mixed_seed_replication_001/three_seed_analysis.md)
- [标称比较的冻结选模记录](../03_对比实验/封存摘要/baseline_comparison_001/paired_100_001/checkpoint_selection.json)

## 11 月 25–28 日新增训练

- [七个活动的逐批训练指标](封存摘要/paper_review_20261128/training_progress.json)覆盖四个新增传播深度模型、历史 Full s02 和两版整班 REINFORCE。每个活动 32 批，每批 32 班。第 k 批采样使用更新前的权重 u(k−1)。每行附原始摘要路径与 SHA256。
- [五日程传播深度对照](../02_消融实验/README.md)沿用 s01–s03，并以全新运行完成 s04/s05 的 Full 和一层模型。
- [步数对齐的整班回报训练](../03_对比实验/方法与实现/reinforce_comparison_001/ppo_steps_matched_001/README.md)逐批匹配 Full s02 的 Actor Adam 步数，共 23,612 步。
- [B 图热启动与选模](../04_泛化与鲁棒性/封存摘要/variable_topology_training_001/frozen_policy_trial_001/b_finetune_001/finetune_report.md)收录另外 128 班训练、4 次更新、独立验证和 B50 复测。
