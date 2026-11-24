# 两种在线 MILP 各十班闭环测试

活动 `bg128-or-milp-paired10-20261124-001` 使用标准九车工况，按十个新环境槽串行运行 20 个完整班。每槽先滚动整数配额，再车次级 FIFO，两班共用该槽环境 seed。两种策略均为确定性派车。各物理班有独立的 run、recording_id、配置字节 SHA256、Python 输出目录与 AnyLogic 事件目录。 [逐班身份清单](prepared_001/run_seed.csv) · [启动卡](prepared_001/launch_identity.json)

## AnyLogic 工作副本接入

沿用上轮标准九车工作副本，在 `127.0.0.1:50051` 串行运行。表中的原值以两班试跑包为参照。若仍用十槽 GAT1 配对包，原 helper 与运行数分别是该包的类正文和 10。

| 位置 | 原值 | 本轮值、作用与核验 |
| --- | --- | --- |
| Additional Class → Baseline1CampaignPlan → 类正文 | 两班试跑 helper，`JOB_COUNT=2` | 替换为[本轮 helper](prepared_001/Baseline1CampaignPlan.java)类正文，保留 AnyLogic 管理的包声明。核对活动名和 `JOB_COUNT=20`。helper 按序绑定每班策略、身份及配置 SHA256。 |
| RL_Training → Properties → Number of runs | 2 | 设为 **20**，保持 Freeform、串行、关闭 Replications，`Stop at=28800` 秒。 |
| RL_Training → Properties → Parameters → Freeform → 运行身份 | 由旧 helper 供值 | 核对 `experimentRunIndex=Baseline1CampaignPlan.runIndex((int) index)`、`experimentSeed=Baseline1CampaignPlan.environmentSeed((int) index)`、`boundaryRecordingId=Baseline1CampaignPlan.recordingId((int) index)` 和 `boundaryEventRoot=Baseline1CampaignPlan.eventRoot((int) index)`。首班 run 以逐班清单为准。 |
| 同上 → 派车、观测与记录 | Python RPC，标准九车 | 核对 `currentDispatchPolicy=Baseline1CampaignPlan.dispatchPolicy((int) index)` 返回 3，`usePhysicalObservation=true`，`numTrucks=9`。`enableBoundaryEventRecording`、`enableMaterialEventMemory`、`offlineNativeCollection` 对应 helper 的 `eventRecordingEnabled`、`materialEnabled`、`offlineEnabled`，均返回 false。 |
| Main → On startup → 第一段 | `boundaryAcceptanceConfigSha256 = Baseline1CampaignPlan.awaitConfigSha256(...)` | 核对该调用仍在首段，参数依次为 `experimentRunIndex, experimentSeed, boundaryRecordingId, boundaryEventRoot, boundaryInstrumentationRevision`。首班在这里启动控制器，各班在运行前获取对应配置 SHA256。 |

```bash
cd /home/jinx/Faith/experiments/lagrangian_ppo_v1
milp_paired10=change_plans/token_attention_001/business_graph_summary_009/baseline_comparison_001/or_paper_comparison_001/paired10_001
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B "$milp_paired10/paired10_launch.py" --check
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B "$milp_paired10/paired10_launch.py" --status
```

预检返回 `PREFLIGHT_PASS_NO_LISTENER` 后，在 AnyLogic 执行 **Save → Build → Run RL_Training** 一次。Java 启动参数核验后登记本轮 20 班，`JAVA_START_VERIFIED` 表示对应班身份握手通过。每槽第二班发出首个动作前核对第一班的环境 seed、首帧和实际 RPC 配置。

## 结果与状态

`runs/bg128-or-milp-paired10-20261124-001/` 下的 `run_status.json` 和 `launch/controller.log` 显示进度。各班收到真实 END 并核对动作回显后发布 `shifts/run_*/evaluation_receipt.json`。20 班正常收齐时，`suite_complete.json` 引用封存报告，列出完整达标、四窗品位、矿废产量、三类等待、联合成本、延迟、MILP 采用和回退。配对差按车次级 FIFO 减滚动配额计算。若出现 `STOP`，保留已领取身份和现有回执，按日志定位问题。

后续预算与 run/seed 禁重从 [paired10_campaign.budget_snapshot](paired10_campaign.py) 读取，合并原两班试跑账本及以前活动。准备本包时按已冻结的试跑身份检查冲突。
