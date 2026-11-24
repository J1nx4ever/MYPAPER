# BG128 最优训练观测 Checkpoint 的100班独立推理

用户于2026-11-18明确要求：“训练结束，选择最好的checkpoint推理100轮次”。该请求授权本轮100个新班次，不增加训练更新。

选用 `updates/0013/checkpoint.pt`。完整训练为512班、16次已提交更新；按同一冻结联合成本，`train_u0014` 的32班均值最低，为0.3223485481，采样策略对应更新13。第14次更新的训练summary成本来自更新前策略，不能归给更新14。最终更新16没有后续实际采样，因此不按其summary声称最优。

选择依据和逐批指标见[冻结审查](../../../../reports/token_attention_001/business_graph_summary_009/checkpoint_evaluation_001/selection_review.md)。这是训练期间观测最优，32班且不同seed带来选择偏差；新100班不再用于重新挑选模型。

- 模型：固定BG128 Actor及单Critic，保留两层GAT、四组关系及原C128。
- 动作：沿用训练策略的随机采样，使用每班独立私有RNG；不改为argmax。
- 范围：100个validation班次、400个窗口，零梯度、零优化器更新。
- 成本：保持008冻结的品位deadband .004、品位scale .006、每窗cap2、空窗2、矿/废短缺8/8、等待0.2×累计等待/28800。矿石16000、废石4000为达产阈值，6000为废石停派阈值。
- 评价：四窗口品位RMSE、四窗口联合合规率、矿石达产率、废石达产率、完整任务完成率、累计等待、Critic MC MSE和EV。保留失败班次；空窗计不合规并单列。
- 记录：关闭transition JSONL、RPC JSONL和rollout快照；沿用既有封存typed HDF5与必要物料journal，以验证采样身份、因果摘要、MC目标及八项指标。

使用[手工清单](manual_anylogic.md)替换helper并设置100次运行。Java将在一次AnyLogic Run中启动唯一Python控制器。

完成状态：100班、18625个决策、400个窗口，零训练/优化更新；原控制器正常完成全部回读与评价，无离线finalizer或重跑。结果见[运行结果](../../../../runs/business-graph-summary-best-checkpoint-eval100-20261118-001/reports/README.md)、[完成独立审计](../../../../reports/token_attention_001/business_graph_summary_009/checkpoint_evaluation_001/completion_independent_audit.md)与[训练后综合分析](../../../../reports/token_attention_001/business_graph_summary_009/trained_analysis_001/README.md)。下述准备/启动身份保留为原始方案记录。

预算以旧独评账本为前缀，使用 `runs/independent-evaluation-budget-v1.bg128.sqlite` 兼容扩展。原独评JSON固定64班且被后续账本按SHA引用，不能原位追加100班而保持旧链可读；两个原JSON及其他账本保留。完整合并入口为 `business_graph_summary_evaluation_v1.budget.budget_snapshot`，总容量20983→21083。注册在本次Run触发时发生，不复用旧512班身份。

准备目录为 [prepared_001](prepared_001/)，含冻结配置、用户100班授权记录、Java helper、master、[完整身份CSV](prepared_001/identities.csv)与[启动身份](prepared_001/launch_identity.json)。运行数据和结果将位于本轮独立 `runs/business-graph-summary-best-checkpoint-eval100-20261118-001/`。
