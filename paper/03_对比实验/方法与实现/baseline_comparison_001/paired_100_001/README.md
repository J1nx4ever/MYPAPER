# BG128 与三种传统策略的 100 槽配对独评

> 此处保存旧版 400 班冻结方案。基线逻辑已复审并修复，后续启动请按[修复版五组单槽验证](../audit_002/README.md)执行。

2026-11-22（Asia/Shanghai）。按用户确认，标称工况每策略 100 班，共 400 个完整物理班。保留四班联调作为先验工程证据。新活动的配置、控制目录及运行身份见[启动清单](prepared_001/launch_identity.json)。

## 检查点选择

三种混合训练初始化共同具备固定 `u32` 独评。从这三个已独评检查点中，以历史标称 100 班的平均冻结整班联合成本最小为主选择规则，完整任务、两类达产、品位和等待单独列出。此前的三个种子共享同一组历史独评槽，本轮使用新种子确认与传统策略的差异。

| 历史固定策略 | 标称联合成本 | 标称完整任务 | 标称等待（车辆秒/班） | 五共同留出联合成本 | 五共同留出完整任务 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 混合 s01/u32 | 0.363799 | 97/100 | 17,349 | 0.330527 | 498/500 |
| **混合 s02/u32** | **0.271360** | **99/100** | **15,182** | **0.281778** | 496/500 |
| 混合 s03/u32 | 0.336237 | 98/100 | 20,967 | 0.376994 | 488/500 |

s02 标称历史矿废各 100/100 达产，标称四窗品位 RMSE 为 0.2880 个百分点。本轮固定 **s02/u32**、原私有随机采样与零优化器更新。三次初始化的较早更新提供训练曲线，冻结比较的共同检查点是 u32。 [三种子原始分析](../../../../../reports/token_attention_001/business_graph_summary_009/generalization_001/mixed_seed_replication_001/three_seed_analysis.md) · [选模核验结果](../../../../../reports/token_attention_001/business_graph_summary_009/baseline_comparison_001/paired_100_001/checkpoint_selection.json)

## 正式独评

每个标称槽使用一个独立的新环境 seed，同槽按卸载贪心、FIFO 贪心、多步滚动、BG128 各完成一班。每班各有独立 run、recording、配置字节 SHA256、Python 输出及 AnyLogic 事件目录。每个 BG128 班消费自己的新私有 Actor 采样 seed，三个确定性策略由当步快照和冻结参数确定。正式评估前冻结四种方法与 `s02/u32`。滚动采用四班联调的同一名义模型、深度 3、800 ms 搜索预算及回退规则。 [方法规格](../methods.md)

主效应为同槽 **BG128 减多步滚动的平均联合成本**。按环境种子槽计算差值并做 10,000 次配对 bootstrap，报告 95% 区间。完整任务同时报告两种不一致方向及精确 McNemar 检验。次级对照为 FIFO 减卸载、滚动减 FIFO。各臂补报每窗品位与合规、空窗、矿废吨位及达产、全班等待、各等待阶段、搜索深度、时限和回退。正常结束的品位违规、欠产与合法回退均保留。完整报告要求 100 个四臂配对槽，单层场景与固定策略是本轮推断范围。

## AnyLogic 手工启动

| 精确位置 | 改前 | 改后与验收 |
| --- | --- | --- |
| `Additional Class → Baseline1CampaignPlan → 类正文` | 仓库副本是已完成的四班联调 helper，`jobCount()=4` | 替换成 [本轮 Java helper](prepared_001/Baseline1CampaignPlan.java) 全文，沿用现有 Java package。新 helper 的 `jobCount()=400`，index 0–3 为第一配对槽。 |
| `RL_Training → Properties → Number of runs` | 核对界面当前值，仓库副本为 `3448` | 明确填 `400`。继续 Freeform、串行、关闭 Replications，班末 28,800 s。 |

继续使用 `currentDispatchPolicy=3`、`usePhysicalObservation=true`、`Main.enableOnnxDispatch=true` 和 50051。保留 `Main → On startup → Baseline1CampaignPlan.awaitConfigSha256(...)`，既有身份及工况表达式继续从 helper 取值。标称 `freeSpeedFactorStd=0.05`。边界事件、紧凑物料、逐请求、全量观测和 rollout 的详细记录关闭。通过实际 RPC 验收配置、schema、动作与四窗材料及等待。字段位置可参照[四班联调的 Freeform 和启动位置](../single_shift_anylogic.md#构建前核对现有设置)。

在项目目录先运行预检。待 50051 的已有活动正常结束，在 AnyLogic **Save → Build → Run** 一次，由 helper 启动 Python。以 `run_status.json` 的已完成物理班和配对槽作为进度依据，完整封存以 `suite_complete.json` 与配对报告为准。

```bash
cd /home/jinx/Faith/experiments/lagrangian_ppo_v1
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B change_plans/token_attention_001/business_graph_summary_009/baseline_comparison_001/paired_100_001/launch.py --check
```

运行后可把 `--check` 换成 `--status` 查看进度。出现 STOP 时保留账本、回执及当前场景。已领取身份的班次依据真实证据核对，补位另立新身份。按四班联调的实际墙钟简单外推约 4.6 小时，其中滚动优化占主要计算时间。该估计来自一个标称槽。 [四班封存结果](../../../../../runs/bg128-baseline-smoke-20261122-001/suite_complete.json)
