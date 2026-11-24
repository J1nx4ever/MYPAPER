# 五组基线独立复核与同槽复验

本次复验沿用已封存 V2 的卸载贪心、FIFO 贪心、多步滚动、DISPATCH-inspired 和混合 s02/u32 Actor。每组运行一班，共用一个新标称环境种子。指标包括按实际 FIFO 结算的四窗品位、矿废吨位、等待及联合成本。两种匹配时钟的归因属于单独的对照问题。

本轮五班已完成并通过工程验收。五组均达产，FIFO 贪心和 BG128 同时满足四窗品位要求。DISPATCH-inspired 的前三窗合规，末窗品位为 28.8818%。 [班末报告](../../../../../reports/token_attention_001/business_graph_summary_009/baseline_comparison_001/all_recheck_001/single_shift_validation.json) · [复核记录](../../../../../reports/token_attention_001/business_graph_summary_009/baseline_comparison_001/all_recheck_001/result_review.json)

## 核对方法

| 检查对象 | 复核方式 |
| --- | --- |
| 两种贪心和滚动 | 运行既有回归测试。在 300 棵具有两种选择的三层决策树中，独立计算共同尾部以及全部 2,400 个叶子的成本，再核对三个策略的 900 次动作和成本。覆盖卸载与 FIFO 采用不同品位时钟、完整搜索与受预算约束的实现。 |
| DISPATCH-inspired | 在三个已封存标称班抽取的 141 帧上，独立计算站点需求时刻、空驶成本及铲位排队，再对 7,965 个候选分配逐一重算损失。Python 返回动作达到同一帧最大可行匹配规模下的最低分，分数核对容差为 1e-8。同时核对每站分段流量积分和上层车型分配吨数。 |
| BG128 | 从原始封存结果复算三训练种子的选模排序，核对 s02/u32 权重与签名。用真实 Actor 和原班采样种子重放一个封存标称班，185 次派车与归档动作逐一相同，四个结算窗和 END 均通过。 |
| 五组共同结算 | 使用下方独立核验器复核五班的活动、配置、Java START、ledger、receipt 及源码字节。按实际破碎矿吨和金属吨重算品位，按四窗软成本、矿废欠产与等待重算联合成本，并检查请求数和站点动作计数。上轮全部五班通过独立重算。 |

本次复查验证当前代码实现其冻结评分公式。DISPATCH 上层以流量近似未来仓容和跨窗 FIFO，下层按派车流量、空驶及等铲损失进行匹配。模型可能选择出真实窗口品位不合格的动作。业务是否达标以本次真实 END 回执为准。

四种传统策略还在标称工况及品位交换工况各取八帧，逐一重复计算。64 个策略与状态组合均给出相同的合法动作，Python 回退与超时计数均为零。滚动策略在其中 12 帧改善了单步方案的预测成本，其余四帧持平。独立评分核验复用了生产端的可行候选生成与载荷预留函数，已预留吨数、部分载荷和禁派条件另由回归算例覆盖。 [独立复算脚本](algorithm_review.py)

上一轮真实五班合计 930 次派车，Python 回退和超过 1 s 的响应均为零。独立核验器还逐项拒绝了六份在内存中构造的错误回执，包括错误金属吨、错误联合成本、错报矿石达产、缺失多车匹配证据、Actor 少一次采样和错误工程验收结论。 [审查记录](../../../../../reports/token_attention_001/business_graph_summary_009/baseline_comparison_001/all_recheck_001/prelaunch_review.json)

Actor 全班回放使用历史观测、原采样种子和实际模型权重。为接入当前验证器，仅在内存中的 RPC 副本适配三个记录开关。全局 Torch 随机状态在回放前后逐字节一致。 [回放脚本](actor_review.py) · [实际回放记录](../../../../../reports/token_attention_001/business_graph_summary_009/baseline_comparison_001/all_recheck_001/actor_replay.json)

## 手工启动

1. 打开实际运行的模型副本。在 **Additional Class → Baseline1CampaignPlan → 类正文** 将上一轮 `audit_002` 的类正文替换为本轮冻结文件，保留编辑器的 package。核对活动 ID 和主清单路径均属于本轮。 [本轮 Java 类正文](prepared_001/Baseline1CampaignPlan.java)
2. 在 **RL_Training → Properties → Number of runs** 核对为 **5**。沿用串行、Freeform、关闭 Replications、28800 s、策略槽 3、物理观测、既有启动握手和低频记录配置。 [参数逐项说明](../audit_002/README.md)
3. 核对每班的 recording_id、run_index、种子、配置实际字节 SHA256、Python 和 AnyLogic 事件目录，然后 **Save → Build → Run RL_Training** 一次。 [逐班身份和输出路径](prepared_001/run_seed.csv)

班末使用项目 Conda 环境执行 `verify_results.py <本轮 runs 目录>`。输出列出五组实际派车数、合规窗数、联合成本和工程验收状态。 [独立核验器](verify_results.py)
