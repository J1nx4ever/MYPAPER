# 对比实验

标称正式评测的八种策略共用 100 个环境槽。综合报告合并初始五组 500 班、FIFO 修订 100 班和两种强基线 200 班。十槽运筹方法比较与一层模型十槽配对使用独立评测面板。

- [正式八组逐班原报告](../../experiments/lagrangian_ppo_v1/reports/token_attention_001/business_graph_summary_009/baseline_comparison_001/or_paper_comparison_001/paired100_strong_001/strong100_evaluation.json)
- [原五组逐班原报告](../../experiments/lagrangian_ppo_v1/reports/token_attention_001/business_graph_summary_009/baseline_comparison_001/paired_v2_100_001/paired_comparison.json)
- [两种 MILP 的十槽原报告](../../experiments/lagrangian_ppo_v1/reports/token_attention_001/business_graph_summary_009/baseline_comparison_001/or_paper_comparison_001/paired10_001/milp_paired10_evaluation.json)
- [一层模型与品位修订规则的十槽原报告](../../experiments/lagrangian_ppo_v1/reports/token_attention_001/business_graph_summary_009/baseline_comparison_001/dispatch_flow_matching_001/grade_control_001/gat1_s01_paired_001/paired_evaluation.json)
- [同主机离线决策耗时摘要](封存摘要/baseline_comparison_001/or_paper_comparison_001/paired100_strong_001/same_host_compute_probe.json)
- [传统策略方法与实现](方法与实现/baseline_comparison_001)

## 同一 BG128 表示下的训练方法对比

B19–B23 记录初版整班 REINFORCE、优化预算核查、逐批步数对齐和七工况固定权重结果。正式比较沿用历史 Full s02/u32 的 700 个班作为 PPO 对照。步数对齐的整班回报变体有 42 个原完整班和 658 个接续完整班。PPO 与该变体的联合达标为 693/700 与 139/700，四窗均合规为 694/700 与 144/700。

- [步数对齐版学习器副本](算法实现/business_graph_matched_reinforce_v1/learner.py) · [初版学习器副本](算法实现/business_graph_reinforce_v1/learner.py)
- [训练封存](封存摘要/reinforce_comparison_001/ppo_steps_matched_001/training_complete.json) · [配对独评最终结果](封存摘要/reinforce_comparison_001/ppo_steps_matched_001/paired_inference_001/recovery_001/paired_summary.json)
- [旧版训练与独评清理清单](封存摘要/reinforce_comparison_001/ppo_steps_matched_001/legacy_cleanup.json)
- [A 图停工下的三策略配对结果](../04_泛化与鲁棒性/封存摘要/variable_topology_training_001/frozen_policy_trial_001/a_full_u32_high_grade_outage_001/paired_severity_001/summary.md)
