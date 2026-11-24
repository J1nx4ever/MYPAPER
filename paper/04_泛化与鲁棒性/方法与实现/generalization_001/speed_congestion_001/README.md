# 标称工况的内生拥堵压力测试

用同一个混合训练 BG128/u32 检查点，在标称矿源与道路工况下固定推理。自由速度因子的截断前标准差设为 **0.15、0.25**，各做 **20 个物理班**。各档按槽复用环境和 Actor 采样种子。速度档位改变运输时间和随机数消耗顺序，因此同槽只保证初始种子一致。

| 固定项 | 设置 |
| --- | --- |
| 矿源 | Load0、Load1、Load2 均值 38%、30%、15%，半区间 ±0.5 个百分点，Load3 为废石 |
| 车辆和设施 | 9 辆车，沿用标称道路限速、装卸资源、FIFO 缓冲仓和破碎机 |
| 任务 | 8 小时，4 个 2 小时窗。矿石 16,000 t、废石 4,000 t，品位 26%–28% |
| 速度模型 | 自由速度因子截断在 [0.85, 1.10]，内生跟驰和制动开启 |
| 推理 | 混合训练 u32，stochastic，独立的逐班 Actor RNG，优化器更新为零 |

本次活动的档位和班数在 [campaign_plan.json](campaign_plan.json) 声明。准备程序展开成有字节 SHA256 的工况配置、master、速度表和 Java `Baseline1CampaignPlan`。冻结后用独立 run、recording 和种子启动。旧 0.05 组来自相同策略和标称工况，作为历史参考。正式配对差值为 0.25 档减 0.15 档。

主要指标是每班装卸资源累计等待。报告同时给出矿废吨位、四窗品位、任务达标、卸点容量排队、决策时刻近距跟驰代理和可观察速度离散程度。等待统计覆盖装卸资源及破碎仓容量排队。道路指标来自决策采样，不能直接换算成全班跟车或制动车秒。各档假如实测差距很小，需要结合截断后理论标准差约 0.06876、0.07095 解释。

按 [AnyLogic 一键启动说明](manual_anylogic.md) 接入。启动前核对[两档 recording、配置 SHA256 和目录](prepared_001/launch_identity.json)及[40 班 run 与双种子清单](prepared_001/identities.csv)。正式报告位于 `reports/token_attention_001/business_graph_summary_009/generalization_001/speed_congestion_001/`。

当前历史合并登记量为 25,636 班。AnyLogic Run 后会追加 40 个新物理槽，登记量为 25,676。预算与身份禁重通过 [speed_budget.py](speed_budget.py) 的 `budget_snapshot` 读取，包括连续混合活动的替补和历史 UNKNOWN。
