# s01 GAT1 u32 与品位控制十班配对独评

活动 `bg128-gat1-s01-u32-grade-paired10-20261123-001` 使用冻结的 s01 **1-Round GAT + Node-MLP u32** Actor。新物理 run 为 `10036531–10036540`，依序对应已完成的品位控制 run `10036521–10036530`。每槽复用对应班的环境 seed，并核对 RPC 场景、完整业务配置、首帧哈希和首车掩码。新班使用独立的 recording_id、配置字节 SHA256、Python 输出目录和 AnyLogic 事件目录。Actor 只在 Python 使用本槽的独立采样 seed，整个活动为冻结策略独评。 [逐班身份清单](prepared_001/run_seed.csv) · [启动卡](prepared_001/launch_identity.json) · [品位控制十班设置](../manual_anylogic.md)

## AnyLogic 手工设置

使用上次品位控制十班的标准九车工作副本。弱 GAT 复现与本次独评共用 **127.0.0.1:50051**，两项活动按顺序运行。启动入口会验证独占端口租约。 [本次 Java helper](prepared_001/Baseline1CampaignPlan.java)

| 位置 | 品位控制十班设置 | s01 GAT1 配对设置及验证 |
| --- | --- | --- |
| Additional Class → Baseline1CampaignPlan → 类正文 | `CAMPAIGN_ID=bg128-dispatch-flow-grade-control10-20261123-001`，`JOB_COUNT=10` | 替换为本次 helper 类正文，保留编辑器生成的包声明。确认 `CAMPAIGN_ID=bg128-gat1-s01-u32-grade-paired10-20261123-001` 和 `JOB_COUNT=10`。helper 将新班身份、SHA 和启动入口一并绑定 |
| RL_Training → Properties → Number of runs | 10 | 使用 **10**，Freeform、串行、关闭 Replications，Stop at 28,800 秒。一次运行覆盖十个已冻结槽 |
| RL_Training → Properties → Parameters → Freeform | 品位控制身份由旧 helper 提供 | 核对 `experimentRunIndex=Baseline1CampaignPlan.runIndex((int) index)`、`experimentSeed=Baseline1CampaignPlan.environmentSeed((int) index)`、`boundaryRecordingId=Baseline1CampaignPlan.recordingId((int) index)`、`boundaryEventRoot=Baseline1CampaignPlan.eventRoot((int) index)`。首个新 run 为 `10036531`，对应环境 seed 以逐班清单为准 |
| 同上 → 派车、工况与记录 | Python RPC 派车，标称九车，低频记录 | 核对 `currentDispatchPolicy=Baseline1CampaignPlan.dispatchPolicy((int) index)` 返回 3，`usePhysicalObservation=true`，`numTrucks=9`。配置 `enableBoundaryEventRecording`、`enableMaterialEventMemory`、`offlineNativeCollection` 分别调用 helper 的 `eventRecordingEnabled`、`materialEnabled`、`offlineEnabled`，三项均返回 false |
| Main → On startup | 第一段 `boundaryAcceptanceConfigSha256 = Baseline1CampaignPlan.awaitConfigSha256(...)` | 核对该调用仍在初始化首段。Main 使用新 helper 拉起 Python，逐班等待 ready 并核对配置字节 SHA256 |

在实验根目录执行预检。返回 `PREFLIGHT_PASS_NO_LISTENER` 且 50051 空闲后，在 AnyLogic **Save → Build → Run RL_Training** 一次。Python 日志出现 `JAVA_START_VERIFIED` 表示身份握手通过。Python 在发出首个动作前核对已封存配对槽的实际环境条件。

```bash
cd /home/jinx/Faith/experiments/lagrangian_ppo_v1
paired_plan_dir=change_plans/token_attention_001/business_graph_summary_009/baseline_comparison_001/dispatch_flow_matching_001/grade_control_001/gat1_s01_paired_001
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B "$paired_plan_dir/paired_launch.py" --check
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B "$paired_plan_dir/paired_launch.py" --status
```

## 配对验收

状态和日志位于 `runs/bg128-gat1-s01-u32-grade-paired10-20261123-001/` 下的 `run_status.json` 和 `launch/controller.log`。每班收到 END 并完成结算后发布 `evaluation_receipt.json`。如果状态为 `STOP`，按状态和日志核对已领取的 run、逐班回执和审计身份。

十班完成后，`suite_complete.json` 给出最终配对报告及 SHA256。报告逐槽列出完整达标、四窗品位差、矿废达产、品位代价、生产代价、等待及联合代价。联合代价差按 **GAT1 减品位控制** 计算，负差表示 GAT1 代价较低。相同环境 seed 与初始状态构成配对，后续仿真轨迹随两组动作变化。配对报告以十个环境槽为重采样单位。
