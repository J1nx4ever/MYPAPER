# 两种在线 MILP 的标称闭环验证

两个策略使用同一个新标准九车环境槽，分别运行一个 8 小时班次。滚动配额为 `10036541`，车次级 FIFO 为 `10036542`。不同 run 各有独立 recording_id、配置字节 SHA256、Python 输出目录与 AnyLogic 事件目录。两组均为确定性派车。 [逐班清单](prepared_smoke_001/run_seed.csv) · [启动卡与配置引用](prepared_smoke_001/launch_identity.json)

## AnyLogic 工作副本手工接入

沿用上轮标准九车工作副本及原有 Main 启动握手。两组在本机 `127.0.0.1:50051` 串行运行。下表“修改前”以十槽 GAT1 配对工作副本为参照，若当前已切换其他活动，则以现有 helper 和运行数为准。

| 位置 | 修改前或现有值 | 本轮设置与验证 |
| --- | --- | --- |
| Additional Class → Baseline1CampaignPlan → 类正文 | GAT1 十槽配对 helper，`JOB_COUNT=10` | 用[本轮 helper](prepared_smoke_001/Baseline1CampaignPlan.java)替换类正文，保留 AnyLogic 管理的包声明。核对 `CAMPAIGN_ID=bg128-or-milp-smoke-20261124-001`、`JOB_COUNT=2`。新 helper 绑定独立清单和控制器 SHA256，分别选择 Python 策略。 |
| RL_Training → Properties → Number of runs | 10 | 改为 **2**，保持 Freeform、串行、关闭 Replications。核对 `Stop at = 28800` 秒。一次点击 Run 完成配对的两个班。 |
| RL_Training → Properties → Parameters → Freeform → 运行身份 | 使用 `Baseline1CampaignPlan` | 核对 `experimentRunIndex=Baseline1CampaignPlan.runIndex((int) index)`、`experimentSeed=Baseline1CampaignPlan.environmentSeed((int) index)`、`boundaryRecordingId=Baseline1CampaignPlan.recordingId((int) index)`、`boundaryEventRoot=Baseline1CampaignPlan.eventRoot((int) index)`。首班 run 应为 `10036541`。 |
| 同上 → 派车、观测与记录 | Python RPC，标准九车 | 核对 `currentDispatchPolicy=Baseline1CampaignPlan.dispatchPolicy((int) index)` 返回 3，`usePhysicalObservation=true`，`numTrucks=9`。三项 `enableBoundaryEventRecording`、`enableMaterialEventMemory`、`offlineNativeCollection` 分别采用 `eventRecordingEnabled`、`materialEnabled`、`offlineEnabled`，均返回 false。 |
| Main → On startup → 首段配置握手 | `boundaryAcceptanceConfigSha256 = Baseline1CampaignPlan.awaitConfigSha256(...)` | 核对握手仍在首段。首班将启动本轮控制器，逐班等待对应 ready 并校验配置字节 SHA256。 |

握手调用的参数顺序为 `experimentRunIndex, experimentSeed, boundaryRecordingId, boundaryEventRoot, boundaryInstrumentationRevision`。工作副本里的标准工况参数和场景绑定继续由既有回调传入，并由本次预存配置逐字段核对。

```bash
cd /home/jinx/Faith/experiments/lagrangian_ppo_v1
milp_smoke_plan=change_plans/token_attention_001/business_graph_summary_009/baseline_comparison_001/or_paper_comparison_001
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B "$milp_smoke_plan/milp_smoke_launch.py" --check
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B "$milp_smoke_plan/milp_smoke_launch.py" --status
```

预检返回 `PREFLIGHT_PASS_NO_LISTENER` 后，在 AnyLogic **Save → Build → Run RL_Training** 一次。预检只核对清单、历史预算及策略依赖。启动器获得端口锁并验证首班身份后才登记两班。控制器日志中的 `JAVA_START_VERIFIED` 表示握手通过。

## 首轮验收

状态和日志位于 `runs/bg128-or-milp-smoke-20261124-001/` 下的 `run_status.json`、`launch/controller.log`。各 run 收到真实 END 且动作回显计数对齐后，才发布 `shifts/run_*/evaluation_receipt.json`。封存报告对账两班首帧与真实 RPC，并分别列出四窗品位、破碎矿量、卸达废石、三类等待、联合成本、求解统计、回退和端到端响应。工程门槛包括完整业务结算、至少实际采用一次可行 MILP 解、动作回显和单请求 1000 ms 目标。车次级另需观察至少一次实际采用联合车次方案。业务达标作为独立结果，并结合工程门槛判断是否扩班。

若状态为 `STOP`，先核对问题、已领取 run 与存在的回执，保留现有审计证据。
