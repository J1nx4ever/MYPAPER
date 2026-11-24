# BG128 单种子训练预算只读审查

2026-11-18，Asia/Shanghai。未登记预算、未启动 RPC、AnyLogic 或训练。

完整合并视图使用 `single_critic_task_query_v1.budget.budget_snapshot()`，包含固定 Checkpoint 独评、随机/确定性配对独评、后续训练及 representation-ablation SQLite 账本；`grade_sampling_ab_v1` 的视图仅是它的历史子集。

当前 active 合并预留 **20,471 班**：COMPLETE **20,461**、UNKNOWN **1**、PLANNED **9**、CLAIMED **0**，共 22 个 campaign。H2 历史另存完成 795、UNKNOWN 1，活动未来预留为 0；旧 UNKNOWN 和 PLANNED 不得当作新实验额度。

完成的 008 每组为 512 班训练（16×32）及最终固定 u16 的 100 班独评。本次用户明确要求一个随机种子训练，建议沿用 **512 班训练**，合并上限为 **20,983**。本次暂不自动附加 100 班独评；如后续独立授权这 100 班，上限才为 **21,083**。

候选新身份 `business-graph-summary-formal-seed1-20261118-001`，训练 run_index **9900001..9900512**，初始化 seed **75597143230906**。可选独评范围 **9900513..9900612** 也未占用。包括初始化、512 个环境和 512 个采样 seed 的 1025 个训练 seed 均无历史或跨类型冲突；进一步检查可选独评后的 1225 个 seed 也无冲突。禁重核对覆盖当前任务及历史 H2/retired 共 31,767 条身份行。此身份仅审查建议，未冻结配置字节、未登记或占用。

预算政策 README 原文是“后续实验按具体方案和班数单独取得执行授权。”当前训练执行意图明确，但用户本条未直接给出班数。512 是沿用 008 的推荐默认；本审查不能自行批准预算，也不能把独评默认为本轮训练授权。Root 已确定本轮具体范围为 512 班训练、0 独评、0 独立冒烟，并将此范围与用户明确的单种子训练授权绑定；审批记录保留用户原话和范围解释，不将 512 冒称为用户原话。

可复用 `budget_snapshot`、`preceding_reference`、`check_collisions` 和 `_history_rows` 进行只读预检。旧 `BudgetLedger` 绑定 008 的 gate/reader，不能伪装 008 合同承载 BG009；可借用其经测试的状态机，接入独立 BG009 合同和读取器。完整快照引用、SHA256、候选 seed 推导及旧 008 campaign 状态见 [budget_review.json](budget_review.json)。

新 BG009 独立账本模块已经实现。以实际冻结 512-job suite 做临时 `test_only=True` 注册、32 班合成记账和 commit 完整性验证，**29 项通过**；全部生产账本 SHA256 前后不变，新 BG009 生产账本仍不存在，临时数据库已删除。无模型 Forward、优化器、RPC 或 AnyLogic。详见 [budget_integrity.json](budget_integrity_001/budget_integrity.json)。
