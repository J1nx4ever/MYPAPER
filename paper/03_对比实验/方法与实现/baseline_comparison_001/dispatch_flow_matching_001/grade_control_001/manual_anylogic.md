# 路径流量匹配品位修订版十班测试

活动 `bg128-dispatch-flow-grade-control10-20261123-001` 使用标准九车工况和品位修订策略，串行评估十个完整班。run 为 `10036521–10036530`。逐班清单包含独立录制 ID、配置字节 SHA256、环境种子、Python 输出目录及 AnyLogic 事件目录。策略为确定性派车，配置中的 Actor seed 以 0 表示。 [逐班身份和目录](prepared_001/run_seed.csv) · [完整启动卡](prepared_001/launch_identity.json)

## AnyLogic 手工接入

使用上次五班测试的标准九车模型副本。活动与占用本机 `127.0.0.1:50051` 的 GAT 复现共用 RPC 通道，按顺序运行。启动入口会获取端口锁，只有获得独占锁并通过身份核对后才注册班次。 [十班 Java helper](prepared_001/Baseline1CampaignPlan.java) · [上次五班活动配置](../manual_anylogic.md)

| 位置 | 上次五班 | 十班设置与核验 |
| --- | --- | --- |
| Additional Class → Baseline1CampaignPlan → 类正文 | `CAMPAIGN_ID=bg128-dispatch-flow-matching-smoke5-20261123-002`，`JOB_COUNT=5` | 将类正文替换为链接中的新 helper，保留编辑器生成的包声明。确认 `CAMPAIGN_ID` 末尾为 `grade-control10-20261123-001`，`JOB_COUNT=10`。helper 校验新活动表和每班配置字节 |
| RL_Training → Properties → Number of runs | 5 | 设为 **10**。使用 Freeform、串行、关闭 Replications，仿真 Stop at 28,800 秒。一次运行对应十个独立班 |
| RL_Training → Properties → Parameters → Freeform | 由五班 helper 提供物理身份 | 核对 `experimentRunIndex=Baseline1CampaignPlan.runIndex((int) index)`、`experimentSeed=Baseline1CampaignPlan.environmentSeed((int) index)`、`boundaryRecordingId=Baseline1CampaignPlan.recordingId((int) index)`、`boundaryEventRoot=Baseline1CampaignPlan.eventRoot((int) index)`。本次首班应为 `10036521` |
| 同上 → 派车与记录 | Python RPC 派车与低频记录 | 核对 `currentDispatchPolicy=Baseline1CampaignPlan.dispatchPolicy((int) index)` 返回 3，`usePhysicalObservation=true`，`numTrucks=9`。确认 `enableBoundaryEventRecording`、`enableMaterialEventMemory` 和 `offlineNativeCollection` 分别调用 helper 的 `eventRecordingEnabled`、`materialEnabled`、`offlineEnabled`，三项均返回 false |
| Main → On startup | `boundaryAcceptanceConfigSha256 = Baseline1CampaignPlan.awaitConfigSha256(...)` | 核对该握手是初始化首段。Main 会拉起本次控制器，校验配置 SHA256，并等待 Python 发布逐班 ready |

预检返回 `PREFLIGHT_PASS_NO_LISTENER` 且端口空闲后，在 AnyLogic **Save → Build → Run RL_Training** 一次。Python 日志中的 `JAVA_START_VERIFIED` 表示该班身份握手已通过。每班只有在收到 END 并完成结算后才发布 `evaluation_receipt.json`。若出现 `STOP`，查看状态及日志，保留已有批次和逐班审计记录。

```bash
cd /home/jinx/Faith/experiments/lagrangian_ppo_v1
grade_plan_dir=change_plans/token_attention_001/business_graph_summary_009/baseline_comparison_001/dispatch_flow_matching_001/grade_control_001
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B "$grade_plan_dir/grade_launch.py" --check
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B "$grade_plan_dir/grade_launch.py" --status
```

## 验收

状态文件位于 `runs/bg128-dispatch-flow-grade-control10-20261123-001/run_status.json`。同目录内的 `launch/controller.log` 记录控制器进度。十班通过身份及 END 回执核对后，`suite_complete.json` 提供最终报告及 SHA256。

物理验收使用 RPC 品位窗上下限与班末矿废结算。报告包含完整达标率、按窗超限方向、矿废达产、运输等待、策略规划耗时以及品位复核改派原因。匹配规划时限为 800 ms，RPC 响应工程目标为 1,000 ms。与上次五班的比较为独立种子样本描述。 [品位修订设计](README.md)
