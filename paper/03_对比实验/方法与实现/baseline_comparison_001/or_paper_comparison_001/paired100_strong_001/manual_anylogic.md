# 两组强基线接入原 100 个标称槽

活动 `bg128-or-strong-paired100-20261124-001` 补跑品位控制版 DISPATCH 与滚动配额 MILP＋Policy4。每个历史槽增加两班，先 DISPATCH 后滚动配额。两班都复用该槽的原环境 seed，分别使用独立物理 run、recording_id、配置字节 SHA256、Python 输出和 AnyLogic 事件目录。原 500 班 V2 和同槽 100 班 V3 的已封存回执进入配对表。新运行数共 200，历史评估保持原身份。 [逐班清单](prepared_001/run_seed.csv) · [启动卡](prepared_001/launch_identity.json)

## ALP 入口

实验目录的原件 `baseline/DRL-PPO.alp` 使用 4,000 吨废石停派线及旧版 RPC。既有百班的业务废石门槛也是 4,000 吨，停派线是 6,000 吨。停派线决定可执行动作集合。以原件为起点时，先 Save As 工作副本，按[原件兼容清单](alp_audit.md)补齐废石动作规则、实际 RPC 工况和等待统计。只有同槽实际 RPC 和首帧核对通过后才执行策略。

项目内已经接入该合同的工作模型为 `anylogic/3d/DRL-PPO-baseline1/DRL-PPO-baseline1.alp`。如果本次所指的 baseline 是该实验模型，可在它的工作副本按下表接入。若当前使用减车工作副本，需将已有 `fleetDisabledTruckMask` 的 Freeform 值设为 `0`，保证标准九车全部运行。

| AnyLogic 位置 | 之前的设置 | 本次设置与目的 |
|---|---|---|
| Additional Class → Baseline1CampaignPlan → 类正文 | 旧活动的 helper | 粘贴[本轮 Java helper](prepared_001/Baseline1CampaignPlan.java)类正文，保留编辑器管理的 package。确认 `CAMPAIGN_ID=bg128-or-strong-paired100-20261124-001` 和 `JOB_COUNT=200`。 |
| RL_Training → Properties → Number of runs | 配对十槽副本为 20，旧原件为 55 | 设为 **200**，使用 Freeform、串行、关闭 Replications，Stop at 28,800 秒。 |
| RL_Training → Properties → Parameters → Freeform → 身份 | 配对副本调用旧 helper，旧原件使用递增 run 和 seed | 将 `experimentRunIndex`、`experimentSeed`、`boundaryRecordingId`、`boundaryEventRoot` 分别绑定 `Baseline1CampaignPlan.runIndex((int) index)`、`environmentSeed((int) index)`、`recordingId((int) index)`、`eventRoot((int) index)`。四个方法均从同一新 helper 调用。 |
| 同上 → 派车与工况 | 标称配对副本使用 Python RPC | 核对 `currentDispatchPolicy=Baseline1CampaignPlan.dispatchPolicy((int) index)` 返回 3，`freeSpeedFactorStd=Baseline1CampaignPlan.freeSpeedFactorStd((int) index)` 返回 0.05，`numTrucks=9`，`usePhysicalObservation=true`，`enableOnnxDispatch=true`，RPC 为 `127.0.0.1:50051`。 |
| 同上 → 记录与诊断 | 兼容副本默认关闭高频记录 | 通过 helper 的 `eventRecordingEnabled`、`materialEnabled`、`offlineEnabled` 将 `enableBoundaryEventRecording`、`enableMaterialEventMemory`、`offlineNativeCollection` 设为 false。核对 `obsDryRun=false`、`diagEvalEnabled=false`、`bcCollectEnabled=false`、`deterministicCheck=false`。 |
| Main → On startup → 配置握手 | 兼容副本有握手，旧原件需新增 | 核对首段调用 `boundaryAcceptanceConfigSha256 = Baseline1CampaignPlan.awaitConfigSha256(experimentRunIndex, experimentSeed, boundaryRecordingId, boundaryEventRoot, boundaryInstrumentationRevision)`。该调用确认本班配置字节 SHA256，再由通信接口提供实际工况。 |

```bash
cd /home/jinx/Faith/experiments/lagrangian_ppo_v1
strong100_plan=change_plans/token_attention_001/business_graph_summary_009/baseline_comparison_001/or_paper_comparison_001/paired100_strong_001
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B "$strong100_plan/strong100_launch.py" --check
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B "$strong100_plan/strong100_launch.py" --status
```

`--check` 核对 Python、冻结配置和历史回执，`PREFLIGHT_PASS_NO_LISTENER` 表示启动包通过核验。完成兼容设置后，在 AnyLogic **Save → Build → Run RL_Training** 一次。首个 run 为 `10036701`。每一班在首个动作前按该槽核对旧 Full 的种子、首帧、车辆、掩码和实际 RPC。若出现 `STOP`，以 `run_status.json`、控制器日志和已领取身份为准。

## 业务与算力结果

- 全班 END 使用一个业务口径。四个非空 FIFO 品位窗均需在 26%–28%，本班破碎矿石至少 16,000 吨、卸达废石至少 4,000 吨。主效应为旧 Full `s02/u32` 与两个新基线分别逐槽比较的联合成本差。报告也包含旧 V2、V3 及其与新方法的配对结果。旧环境槽已用于既有评估，新比较属于历史槽扩展。
- 新运行逐班汇总端到端 RPC 响应、策略墙钟耗时、策略进程 CPU 秒、每次决策 CPU 毫秒、求解接受和回退。策略 CPU 和墙钟包括观测处理、模型构建、求解、匹配和复核，动作回显后的台账更新时间单列。历史回执使用当时记录的策略/RPC 墙钟统计，CPU 时间标为缺失。
- 同机计算对照复用三班的 24 个封存观测，分别计量 Full 推理、DISPATCH 冷重算、滚动配额冷重算与纯 Policy4 跟踪。配额冷重算在独立帧上使用诊断初始化，实际整班通过回执统计真实重算频率。该探针的时间范围是策略调用，物理仿真与 RPC 延迟来自实跑回执。 [同机算力报告](../../../../../../reports/token_attention_001/business_graph_summary_009/baseline_comparison_001/or_paper_comparison_001/paired100_strong_001/same_host_compute_probe.json)

运行期查看 `runs/bg128-or-strong-paired100-20261124-001/run_status.json` 与同目录 `launch/controller.log`。全部 200 份新回执封存后，`suite_complete.json` 引用 `strong100_evaluation.json`。后续预算与 run/seed 禁重使用 [strong100_campaign.budget_snapshot](strong100_campaign.py)，准备包按 0 班登记，实际启动登记 200 班。
