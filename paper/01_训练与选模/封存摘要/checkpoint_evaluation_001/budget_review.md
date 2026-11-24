# BG128 最佳 Checkpoint 100 班独评预算审查

用户明确要求“训练结束，选择最好的checkpoint推理100轮次”，已授权新增100班固定策略推理；无需重复请求额度授权。此次不训练、不执行PPO更新。

完整合并视图为 **20983班**：COMPLETE 20973、UNKNOWN 1、PLANNED 9；包括旧独评JSON、采样对比JSON、后续训练账本以及BG128已完成512班。原UNKNOWN及旧预留均保留。新增100班登记后上限为 **21083班**，H2未来预留为0。

候选身份：`business-graph-summary-best-checkpoint-eval100-20261118-001`，`run_index=9950001..9950100`，独立环境和动作采样各100个新seed，已核对全部活动、H2历史及已归档恢复身份，无重复或交叉类型碰撞。固定父checkpoint的初始化seed 75597143230906 是模型身份，按只读checkpoint允许复用；它不用于新环境或动作抽样。完整待冻结身份见 `proposed_identity_review.csv`，root最终方案应另行核对字节和输出路径。

原 `runs/independent-evaluation-budget-v1.json` 的实现写死64班/u8且其字节SHA被后续sampling、u8、physical、008及BG009账本绑定，原位追加会使整个历史只读验证失败。根据root明确选择，使用该独评账本的SQLite扩展 `runs/independent-evaluation-budget-v1.bg128.sqlite`，保留两个旧JSON原件。统一入口为 `business_graph_summary_evaluation_v1.budget.budget_snapshot()`，它先读取完整BG009历史，再合并扩展；不得后续只读旧JSON而漏算本次100班。

`budget.py` 仅在显式构造BudgetLedger时登记，read-only snapshot不建文件、不建锁。严格100班、零训练、固定checkpoint；UNKNOWN/CLAIMED禁止自动重采，PPO commit永拒绝。临时SQLite验证38项通过，覆盖身份禁重、跨类型seed、父Actor身份复用、幂等登记、第二次扩展精确计数、未授权不落盘、单次claim/completion、UNKNOWN保留、checksum破坏等。旧396 source pins与受保护PPO SHA仍一致。

本审查未登记生产额度、未启动RPC或AnyLogic。实际登记和运行需由root完成最终冻结与预检；不需要用户再次批准100班。
