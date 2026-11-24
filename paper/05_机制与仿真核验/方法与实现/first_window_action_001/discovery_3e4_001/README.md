# 3e-4/u13 首窗失败发现批次

100/100 班已完成。发现 4 个首窗失败，2 个偏低、2 个偏高，均有锁定前的名义候选。矿废分别 100/100 达产，完整任务 96/100。 [结果与候选复核](../../../../../reports/token_attention_001/business_graph_summary_009/first_window_action_001/discovery_3e4_001/README.md)

按用户授权冻结 100 班新独立种子。固定原始 BG128/u13 和原随机采样，保留原 GAT、C128、Reward 与 PPO 数值配置。班次采集及准入复用已完成的 BG128 严格 RPC 和紧凑物料日志路径。原 100 班独评、2 个成对动作反事实及各自结果继续作为独立证据。

启动步骤、冻结 SHA、run/seed 清单和路径见 [AnyLogic 启动卡](manual_anylogic.md)。研究终点见 [冻结协议](protocol.md)，以真实收集的全部 100 班报告首窗失败范围以及原任务指标。

本批独立账本 `runs/bg128-u13-discovery100-20261119-001/independent_discovery_ledger.json` 由控制器按登记状态推进。后续预算与身份检查入口为 [discovery_budget.py](discovery_budget.py) 的 `budget_snapshot`，合并四班反事实及其所有历史前缀。正式登记上限 21799，执行状态以账本、Java 启动声明和封存班数据为准。
