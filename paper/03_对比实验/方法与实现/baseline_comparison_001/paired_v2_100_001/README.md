# BG128 与四种传统策略的标准工况配对评估

活动 ID 为 `bg128-baseline-v2-paired100-20261122-001`。使用 100 个新标称槽，每槽依次运行卸载归窗贪心、FIFO 贪心、多步滚动、DISPATCH-inspired 和冻结 BG128 `s02/u32`，合计 500 个完整 8 小时班。多步滚动沿用 V2 的 3 步搜索深度、20 个扩展节点和 800 ms 安全时限。DISPATCH-inspired 沿用 900 s 的规划更新阈值和最多 3 辆车的实时匹配。 [V2 五组复验](../all_recheck_001/README.md)

同槽五组共用环境种子、起始状态与实际 RPC 工况。每个槽使用独立的新环境种子，BG128 每班另用自己的 Actor 采样种子。旧四组启动包和两轮单槽联调保留历史记录。正式统计使用本轮 100 个槽。 [逐班 recording_id、配置字节 SHA256、种子和输出目录](prepared_001/run_seed.csv) · [启动清单](prepared_001/launch_identity.json)

## AnyLogic 手工启动

| 位置 | 当前复验副本 | 正式实验设置与验收 |
| --- | --- | --- |
| **Additional Class → Baseline1CampaignPlan → 类正文** | `bg128-baseline-all-recheck-20261122-001`，`JOB_COUNT=5` | 粘贴本轮冻结类正文，保留 AnyLogic 生成的 package。确认 `CAMPAIGN_ID="bg128-baseline-v2-paired100-20261122-001"` 和 `JOB_COUNT=500`。 [本轮 Java 类正文](prepared_001/Baseline1CampaignPlan.java) |
| **RL_Training → Properties → Number of runs** | `5` | 改为 `500`。标称槽内按五组顺序串行，使用 Freeform，关闭 Replications，每班结束时间 28,800 s。 |
| **RL_Training → Properties → Parameters → Freeform** | 来自 `Baseline1CampaignPlan` 的策略、种子、工况、身份和记录设置 | 核对 `currentDispatchPolicy=Baseline1CampaignPlan.dispatchPolicy((int) index)` 返回 `3`，`usePhysicalObservation=true`。`experimentRunIndex`、`experimentSeed`、`boundaryRecordingId`、`boundaryEventRoot` 分别调用新 helper 的 `runIndex`、`environmentSeed`、`recordingId`、`eventRoot`。`freeSpeedFactorStd` 由 helper 返回 `0.05`。 |
| **Main → On startup** | `Baseline1CampaignPlan.awaitConfigSha256(...)` | 继续以逐班配置 SHA256 握手。确认 `enableOnnxDispatch=true`，RPC 使用 `127.0.0.1:50051`。 [字段位置](../dispatch_inspired_001/README.md#anylogic-操作) |

启动前在下方命令中使用 `--check`。看到 `PREFLIGHT_PASS_NO_LISTENER` 后，在 50051 空闲时 **Save → Build → Run RL_Training** 一次。首班身份握手会启动 Python。每个班结束后更新进度。用 `--status` 查询已完成的班次数和配对槽数。 [预检记录](../../../../../reports/token_attention_001/business_graph_summary_009/baseline_comparison_001/paired_v2_100_001/preflight_review.json)

```bash
cd /home/jinx/Faith/experiments/lagrangian_ppo_v1
plan=change_plans/token_attention_001/business_graph_summary_009/baseline_comparison_001/paired_v2_100_001
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B "$plan/launch.py" --check
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B "$plan/launch.py" --status
```

## 统计与验收

主效应是逐槽 `BG128 − 滚动优化` 的联合成本差的均值。按 100 个配对槽重采样 10,000 次，计算 95% bootstrap 区间。完整任务报告配对差、两个不一致方向的计数和精确 McNemar 检验。其他比较为描述性。各组同时报告矿废达产、四窗品位、等待、合法回退及响应时限。真实 END 的品位失败、欠产和合法回退计入结果。

运行完成以 `suite_complete.json`、500 个班次回执和 100 个完整配对槽为准。报告保留配置、终末物理量、结算分项、汇总工程计数与逐班回执引用。逐请求事件、全量观测、逐 transition 和 rollout 快照关闭。若遇 `STOP`，依据实际回执和账本对账已经领取的身份。

```bash
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B "$plan/verify_results.py" \
  runs/bg128-baseline-v2-paired100-20261122-001
```
