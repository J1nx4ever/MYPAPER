# DISPATCH 路径流量匹配五班测试

活动 `bg128-dispatch-flow-matching-smoke5-20261123-002` 在标准九车工况串行运行五班。当前车装点由 Python 经既有 RPC 返回。AnyLogic 执行物理装运和四窗 FIFO 结算。五个 run 的新环境 seed、recording_id、配置字节 SHA256、Python 输出目录和 AnyLogic 事件目录都在同一行。 `prepared_001` 为第一阶段可行状态映射修复前的未启动审计包。 [逐班运行清单](prepared_002/run_seed.csv) · [完整启动卡](prepared_002/launch_identity.json)

## AnyLogic 手工接入

在标准九车模型的工作副本中按下表设置。Python 控制器由 Main 的既有启动握手自动拉起。 [本次 Java 类正文](prepared_002/Baseline1CampaignPlan.java)

| 位置 | 修改前或既有表达式 | 本次设置、影响及验证 |
| --- | --- | --- |
| Additional Class → Baseline1CampaignPlan → 类正文 | 上一次活动的类正文及控制目录 | 整段替换为本次 Java 类，保留编辑器生成的包声明。核对 `CAMPAIGN_ID` 末尾为 `smoke5-20261123-002`，`JOB_COUNT=5`。该类按顺序提供五个新物理身份，先核验配置字节 SHA256 再允许 Main 启动 |
| RL_Training → Properties → Number of runs | 工作副本已有班数 | 设为 **5**。Freeform 模式，串行，关闭 Replications，Stop at 28,800 秒。运行一次 RL_Training 会按顺序完成五个独立班 |
| RL_Training → Properties → Parameters → Freeform | run、seed、recording、事件路径由 helper 返回 | 核对 `experimentRunIndex=Baseline1CampaignPlan.runIndex((int) index)`、`experimentSeed=Baseline1CampaignPlan.environmentSeed((int) index)`、`boundaryRecordingId=Baseline1CampaignPlan.recordingId((int) index)`、`boundaryEventRoot=Baseline1CampaignPlan.eventRoot((int) index)`。首班 run 为 `10036511` |
| 同上 → 派车与记录 | Python RPC 动作和低频记录 | `currentDispatchPolicy=Baseline1CampaignPlan.dispatchPolicy((int) index)`，返回 3。核对 `usePhysicalObservation=true`、`numTrucks=9`，以及 `enableBoundaryEventRecording`、`enableMaterialEventMemory` 和 `offlineNativeCollection` 分别调用 helper 的 `eventRecordingEnabled`、`materialEnabled` 和 `offlineEnabled`，三项均返回 false |
| Main → On startup | `boundaryAcceptanceConfigSha256 = Baseline1CampaignPlan.awaitConfigSha256(...)` | 核对该握手仍是初始化首段。Python RPC 使用 `127.0.0.1:50051`，终止班次以 AnyLogic END 为准 |

预检返回 `PREFLIGHT_PASS_NO_LISTENER` 后，在 AnyLogic **Save → Build → Run RL_Training** 一次。启动后查看状态和回执。若出现 `STOP`，先核对已登记的物理身份和失败原因，使用独立新身份处理确需补跑的班。

```bash
cd /home/jinx/Faith/experiments/lagrangian_ppo_v1
flow_plan_dir=change_plans/token_attention_001/business_graph_summary_009/baseline_comparison_001/dispatch_flow_matching_001
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B "$flow_plan_dir/flow_matching_launch.py" --check
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B "$flow_plan_dir/flow_matching_launch.py" --status
```

## 物理验收

活动目录包含逐班 `evaluation_receipt.json`、控制器日志、预算账本和最终 `suite_complete.json`。班末物理结果核对四窗破碎矿吨、金属吨、矿废产量及等待。工程验收核对路线匹配、RPC 回显、流量计划与响应延时。完整任务达标率是该五班的试运行统计。 [算法设计](README.md)
