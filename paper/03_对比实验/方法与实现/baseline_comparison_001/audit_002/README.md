# BG128 对比基线复审与修复版闭环验证

修复版五种策略已在同一个新标称槽各完整运行一个 8 小时班，五组均通过工程验收。真实业务指标沿用四窗破碎品位、矿废达产、车辆等待和整班联合成本。此前冻结的四组 400 班启动包使用旧版本，后续正式对比采用独立冻结的修复版。 [封存的五班报告](../../../../../reports/token_attention_001/business_graph_summary_009/baseline_comparison_001/audit_002/single_shift_validation.json) · [原方法规格](../methods.md) · [旧正式方案](../paired_100_001/README.md)

同槽首帧与实际 RPC 配置一致。930 次派车中，Python 回退和超过 1 s 的响应均为 0。五组均达到矿石和废石吨位目标，四窗合规数按卸载、FIFO、滚动、DISPATCH-inspired、BG128 顺序为 2、3、3、1、4。该槽用于验证同条件下的实际闭环，正式效应以独立环境槽的配对差值估计。

## 发现及修复

| 策略与位置 | 问题与修复 |
| --- | --- |
| 卸载归窗、FIFO 归窗、滚动及 DISPATCH 的名义装载时间 | 观测的载量与铲斗吨位分别经 float32 编码。同一物理倍数解码后略超整斗边界时，旧计算可能多算一斗。现按编码精度吸收边界舍入，同时保留超过该精度的真实加斗。 |
| 卸载归窗与 FIFO 归窗 | 核对了两组对同一候选应用相同名义轨迹、相同矿废和等待评分。卸载组仅改变品位记账时刻。跨窗测试将一整批卸载吨记到卸完窗口，将实际可破碎部分分别计入两个窗口。 |
| 逐事件多步滚动 | 旧版 800 ms 截断会将机器快慢产生的不同部分搜索结果用于派车。修复版先对所有合法首动作完成 FIFO 比较，再按固定顺序评估 20 个后续节点，深度最多 3。深层优先覆盖各首动作中表现最好的前缀，第三动作依首动作轮换候选起点。达到固定节点数采用当前最优完整轨迹；触及墙钟安全线则回到完整的一步 FIFO 解。 |
| DISPATCH 上层 | 停放或无法释放的车辆曾占用可派车时，班末也可能使用新矿尚未到仓时的破碎能力。修复版按可见释放时刻计算车时，并扣除因果可用时间段内既有料的处理量。分析算例中最后一窗剩 900 s，新矿最早在 345.9 s 后可破碎，预测由 625.00 t 改为 384.79 t 的时间上界。 |
| DISPATCH 下层 | 旧版用铲端服务开始时刻比较派车流率的需求时刻，空驶和排队已经单独计价。修复版用派车时刻比较，并以 LP 的车型释放时点重新合成站点分段流率。虚拟车按余料分配部分载荷，三车不可行时仍比较可行的两车匹配。 |
| 冻结 BG128 和五组共用接入 | 保留历史标称选出的混合 s02/u32 及私有随机推理。在实际桥接器上，DECISION 和 END 均校验上一动作回显，DISPATCH 只在回显通过后增加已派吨数。五组核对相同 seed、首帧摘要和实际 RPC 配置。 |

[共用名义预测](../../../../../traditional_dispatch_v2/nominal.py) · [三种经典策略](../../../../../traditional_dispatch_v2/decision.py) · [流量规划](../../../../../traditional_dispatch_v2/flow_plan.py) · [实时匹配](../../../../../traditional_dispatch_v2/dispatch_policy.py) · [共用评估入口](../../../../../traditional_dispatch_v2/runtime.py)

DISPATCH 上层的新增流仍用窗口内连续流量近似。未来车次之间的仓容和 FIFO 次序由 AnyLogic 兑现。名义轨迹对后续服务、启动间隔和路况采用可见参数预测。共同环境 seed 配对约束相同的启动条件，策略改变事件顺序后，车辆获得的随机服务或扰动可以随轨迹分叉。正式报告按配对槽统计。

旧滚动验证班的 178 个请求中有 131 个碰到墙钟上限。修复后，对 3 个已封存标称班取 51 个历史快照，四种传统策略各返回 51 个合法动作，无回退或墙钟中断。滚动有 48 个快照达到深度 3，37 个快照的名义目标低于同帧一步 FIFO，测得 P95 为 656 ms。历史快照来自其他策略的已观察轨迹。闭环比较仍需重跑各策略实际选择产生的轨迹。 [旧滚动计数](../../../../../reports/token_attention_001/business_graph_summary_009/baseline_comparison_001/single_shift_validation.json) · [修复版离线回归](../../../../../reports/token_attention_001/business_graph_summary_009/baseline_comparison_001/audit_002/offline_replay.json)

## 本次 AnyLogic 手工设置记录

使用实际运行副本 `DRL-PPO-baseline1.alp`。冻结清单提供每班独立 recording、配置字节 SHA256、Python 输出目录、AnyLogic 事件目录和 run/seed。五班共用新环境 seed `32396102817785`。仅 BG128 消费私有 Actor seed `64097867756419`。

| Java index | 策略 | run_index |
| --- | --- | --- |
| 0 | 一步卸载归窗 | `10023001` |
| 1 | 一步 FIFO 归窗 | `10023002` |
| 2 | 逐事件多步滚动 | `10023003` |
| 3 | DISPATCH-inspired | `10023004` |
| 4 | 冻结 BG128 s02/u32 | `10023005` |

[逐班身份、配置 SHA256 与目录](prepared_001/launch_identity.json) · [run/seed 清单](prepared_001/run_seed.csv)

1. 在 **Additional Class → Baseline1CampaignPlan → 类正文**，将当前活动的类正文更换为下方冻结文件，保留 AnyLogic 生成的 package。核对 `CAMPAIGN_ID="bg128-baseline-audit-20261122-002"`、`JOB_COUNT=5`。这是新活动的工况和 RPC 启动入口。 [复制冻结 Java 类正文](prepared_001/Baseline1CampaignPlan.java)
2. 在 **RL_Training → Properties → Number of runs**，若沿用刚验收的 DISPATCH 副本，将原来的 `1` 改为 **`5`**。必须为 Freeform、串行、关闭 Replications，结束时间 `28800 s`。策略槽的 Freeform 表达式为 `Baseline1CampaignPlan.dispatchPolicy((int) index)`，返回 `3`。同处 `usePhysicalObservation=true`，run、seed、recording、event root 和记录开关继续由新 helper 对应方法提供。已有 `Main → On startup` 保留 `Baseline1CampaignPlan.awaitConfigSha256(...)`，gRPC 仍由 `enableOnnxDispatch=true`、`127.0.0.1:50051` 启用。字段的精确位置见上轮说明。 [Freeform 字段及启动握手](../dispatch_inspired_001/README.md#anylogic-操作)
3. 在终端执行只读预检。看到 `PREFLIGHT_PASS_NO_LISTENER` 后，等其他活动释放 `50051`，在 AnyLogic **Save → Build → Run RL_Training** 一次。类按索引串行启动五班。状态检查可把命令末尾的 `--check` 替换为 `--status`。

```bash
cd /home/jinx/Faith/experiments/lagrangian_ppo_v1
plan=change_plans/token_attention_001/business_graph_summary_009/baseline_comparison_001/audit_002
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B "$plan/launch.py" --check
```

验证记录保留在本活动的 `reports/` 下，物理运行根为 `runs/bg128-baseline-audit-20261122-002/`。核对五班完成、真实工况相同、合法动作和终止计数一致、每班四窗及吨和等待守恒。检查滚动的完整首步基准、后续扩展和安全线计数，以及 DISPATCH 的 LP 和实时匹配见证。联合成本及工程计数只按班汇总，逐请求事件、全量观测和 rollout 快照默认关闭。 [历史快照回归](../../../../../reports/token_attention_001/business_graph_summary_009/baseline_comparison_001/audit_002/offline_replay.json)
