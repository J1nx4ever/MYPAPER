# BG128 u32 泛化批次续跑启动卡

从 `9991102` 续跑原批次剩余的 **499 班**。其中品位互换 **99 班**，之后四组各 **100 班**。原标称组的 100 班和品位互换首班 `9991101` 均已完成。首班离线接纳、账本复核和 499 班的 Java 映射预检均已通过。 [首班恢复记录](../../../../../reports/token_attention_001/business_graph_summary_009/generalization_001/recovery_001/admission/reconciliation.json) · [离线预检](../../../../../reports/token_attention_001/business_graph_summary_009/generalization_001/recovery_001/preparation_validation.json)

恢复采用原 `3e-4/u32`、原工况和原双种子清单。品位互换首班的准入按上线时相同的逐观测 Actor 推理形状重放，核对已存采样对数概率、动作和随机数状态，再绑定原物料与 RPC 证据。后续班次也使用该准入口径。

## 手工启动

先关闭已停机的旧 `RL_Training` 实例。在已恢复四个装载点的泛化模型中，执行下面两处修改。

| AnyLogic 位置 | 修改前 | 修改后 | 验证 |
| --- | --- | --- | --- |
| Additional Class → `Baseline1CampaignPlan` → 全部 Java 源码 | 原 600 班 helper | 全文替换为[续跑 helper](prepared_001/Baseline1CampaignPlan.java) | `jobCount()=499`，`resumeOffset()=101`，`frozenPlanJobCount()=600`。AnyLogic 索引 0 对应原 master ordinal 101。 |
| `RL_Training → Properties → Number of runs` | `600` | **499** | 首班为 `9991102`，末班为 `9991600`。保持串行，关闭并行评估和重复。 |

Freeform 继续用 `Baseline1CampaignPlan.*((int) index)`，由 helper 在内部计算偏移。确认 `deterministicCheck=false`、`Main → On startup` 使用现有 `awaitConfigSha256(...)`，`Main → init1` 有站号 0、1、2、3 四站，限速注入仍在 `addBidirectionalPath`。

**Save → Build → Run 一次。** Java 会按下表首班身份启动恢复控制器。它读取独立的恢复控制目录，原 `STOP`、原 claim、事件和启动证据请一并保留。

## 首个恢复班次身份

| 字段 | 值 |
| --- | --- |
| Campaign | `bg128-u32-generalization-20261119-001` |
| 工况 | 品位互换 `grade_swap` |
| AnyLogic 索引 / 原 ordinal / 组内槽 | `0 / 101 / 1`（从零计数） |
| `run_index` | `9991102` |
| 环境种子 / Actor 采样种子 | `250558918536851 / 232706493228058` |
| `recording_id` | `bg128-u32-generalization-20261119-001-s01_BG128_eval_grade_swap` |
| 首班配置字节 SHA256 | `027196ad7b80bec2a0645d9852f7f6413786df895cc23ebee00457d01ea7bd0c` |
| u32 检查点字节 SHA256 | `5d42df7d8e33b5f11424094268a4da894a36454e46d86889fc0218ecce6b2914` |
| 新恢复 Java helper 字节 SHA256 | `fe60d338528a146294b8df6bbac1c8c6cd49a82e1ac130ec1aaa0ec329d2a714` |
| 恢复描述字节 SHA256 | `0cbdda2370867ccc0a3d2e01103cb98b9b57a6df8c6dc1870d37bb03ee1072fd` |
| Python 首组输出目录 | `/home/jinx/Faith/experiments/lagrangian_ppo_v1/runs/bg128-u32-generalization-20261119-001/evaluation/grade_swap` |
| AnyLogic 首组物料目录 | `/home/jinx/Faith/experiments/lagrangian_ppo_v1/runs/bg128-u32-generalization-20261119-001_materials/s01_BG128_eval_grade_swap` |
| 新恢复控制目录 | `/home/jinx/Faith/experiments/lagrangian_ppo_v1/runs/bg128-u32-generalization-20261119-001/recovery_001/control` |

[冻结恢复身份](prepared_001/launch_identity.json) · [499 班完整 run/seed 清单](prepared_001/remaining_identities.csv)

完成时账本应累计 **600 班 COMPLETE**、六组各 100 个配对班次、训练及梯度更新为零。续跑日志为 `runs/bg128-u32-generalization-20261119-001/recovery_001/launch/controller.log`。原批次账本与各工况输出目录继续累计进度。
