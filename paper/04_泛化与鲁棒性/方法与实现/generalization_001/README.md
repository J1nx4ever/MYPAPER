# BG128 u32 单训练种子工况泛化

原批次在品位互换首班 `9991101` 的轨迹验收处停止。该班离线复核后记为已完成，四窗品位超限和废石欠产纳入统计。保留原 101 个完成班次，使用[续跑启动卡](recovery_001/manual_anylogic.md)从 `9991102` 继续原计划剩余 499 班。

冻结原 BG128 `3e-4/u32`。该检查点在已完成的 100 班同槽配对独评中，完整任务 100/100，品位 RMSE 0.00250，平均等待 13,364.40 车辆秒。它与 u24 的完整任务数相同，品位 RMSE 和等待更低。选择证据位于[延训配对独评](../../../../reports/token_attention_001/business_graph_summary_009/training_extension_001/paired_evaluation.md)。

六组使用同一套 100 个新评估槽。每槽环境和 Actor 采样种子在六组逐一配对。物理班次各自使用新 run、recording、配置指纹和事件目录，合计 600 班固定推理，训练和优化器更新为零。

| 组别 | 单项扰动 | 班次 |
| --- | --- | ---: |
| 标称 | 矿源均值 38% / 30% / 15%，半区间 ±0.5 个百分点，破碎站入口路段限速 30 km/h | 100 |
| 品位互换 | 交换 Load0 和 Load2 的均值，变为 15% / 30% / 38% | 100 |
| 入口限速 18 | `intersection1 → Crusher` 的限速 18 km/h | 100 |
| 入口限速 15 | `intersection1 → Crusher` 的限速 15 km/h | 100 |
| 均值变化 | Load0、Load1、Load2 的均值为 36% / 29% / 17% | 100 |
| 波动增大 | 三个矿源的半区间均为 ±1.0 个百分点，沿用标称均值和 500 t 块大小 | 100 |

主结果为各组相对标称组的完整任务率及逐班改善、退化数。补充各窗合规、首窗实际处理吨位与金属量、品位 RMSE、矿废达产、等待、联合成本及 Critic MSE/EV。报告同时对照各组真实 RPC 工况、入仓矿源品位与破碎站入口的载矿车决策时刻车速，以核验扰动。所有正常结束的欠产和品位违规均纳入统计。

运行操作见 [AnyLogic 一键启动卡](manual_anylogic.md)。[600 班身份清单](prepared_001/identities.csv) · [六个工况冻结配置和输出位置](prepared_001/launch_identity.json)。

品位换位结果见[状态、身份和四类关系审查](../../../../reports/token_attention_001/business_graph_summary_009/generalization_001/grade_swap_state_relation_audit.md)。后续改动与验收集中在[摘要重构与泛化方案](grade_assignment_generalization_001/README.md)。

固定混合训练 u32 的标称工况速度扰动独评见[内生拥堵压力测试](speed_congestion_001/README.md)。两档标准差各配对评估 20 班。
