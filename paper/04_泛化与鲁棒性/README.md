# 泛化与鲁棒性

矿源配置泛化、固定六车零样本、车队出勤与班中退场在不同活动中评测。阅读结果时以证据总表中的模型身份、实际工况和物理班分母为准。含准备包的历史目录同时保存了预设工况和手工接入说明。

- [混合 Full 三种子与矿源工况](封存摘要/generalization_001/mixed_seed_replication_001/three_seed_analysis.md)
- [固定六车零样本评测方案](方法与实现/fleet_outage_003/README.md)
- [固定六车零样本逐班原报告](../../experiments/lagrangian_ppo_v1/reports/token_attention_001/business_graph_summary_009/fleet_outage_003/fleet_outage_evaluation.json)
- [出勤与退场活动方法](方法与实现/fleet_outage_resilience_001/README.md)
- [适应性训练分析](封存摘要/fleet_outage_resilience_001/bg128-fleet-resilience-20261124-repair01/training_analysis.md)
- [训练后验证与测试分层摘要](封存摘要/fleet_outage_resilience_001/bg128-fleet-resilience-20261124-repair01/evaluation.json)

## 跨布局迁移与装载点停工

- T03–T04 使用冻结动态出勤 u9。A 树图九车和 B 环图七车的独立评测分别为 50/50、22/50 班达标，两组矿废吨位均全数达标。[A/B 各 50 班的结果](封存摘要/variable_topology_training_001/frozen_policy_trial_001/paired_50_001/summary.md)
- T05 从动态 u9 在 B 图微调。入选 u2 在相同历史 B50 种子上完成 47/50，25 个槽从失败变为成功，0 个槽反向变化。[训练、验证选模和同槽复测](封存摘要/variable_topology_training_001/frozen_policy_trial_001/b_finetune_001/finetune_report.md)
- T06–T07 使用冻结 Full s02/u32。先核对同种子停工前的派车与物料轨迹，再让三种策略在 30 个新槽各跑 0、30、60 分钟停工，共 270 班。[单槽配对](封存摘要/variable_topology_training_001/frozen_policy_trial_001/a_full_u32_high_grade_outage_001/same_seed_control_001/summary.md) · [停工强度配对](封存摘要/variable_topology_training_001/frozen_policy_trial_001/a_full_u32_high_grade_outage_001/paired_severity_001/summary.md)
- A04 对应另行登记的车辆退场现场演示，回执状态 UNKNOWN，完整班数为 0。车队正式独立测试的分母仍为 36 个四方配对槽。
