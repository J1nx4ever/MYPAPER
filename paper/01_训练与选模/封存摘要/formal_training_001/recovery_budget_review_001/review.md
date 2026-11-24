# BG128 首班故障恢复预算只读审查

2026-11-18，Asia/Shanghai。本审查没有生产写入、预算登记、RPC、AnyLogic 或训练。

首班 run9900001 的原始采样已经正常结束：184 transitions、4 windows、normal_completion=true。生产 BG 账本 cap512、UNKNOWN1、PLANNED511、commits0。UNKNOWN 的原因是新 BG admission 引用了旧 TQ storage 候选合同；不是仿真尚未完成。原始 normal receipt 只能证明物理班消耗，不能代替尚未完成的 BG 强校验和派生数据验收。

当前 BudgetLedger 没有 reconciliation 接口；complete 只支持 CLAIMED→COMPLETE，resume 拒绝 UNKNOWN/CLAIMED，commit 和原只读 _state 都要求批内 SQL row 为 COMPLETE。原 controller 还拒绝保留的 STOP。因此不能直接调用 complete、改 status、删 STOP、重采首班或登记另一套512班作为“恢复”。

若严格保留原 UNKNOWN 状态，建议隔离恢复 sidecar：

1. 用独立 BG storage writer+strong reader 完整验收现存 raw、journal、原 Actor、原 phase config，向新的恢复证据目录写派生 HDF5；原 pinned 源码和原数据保持不变。校验实际动作/RNG、旧 log probability、schema/RPC、因果物料、等待和冻结成本。
2. 新 reconciliation 证据永久绑定原 UNKNOWN payload 前像和 checksum、原 suite/approval、原 receipt/raw/Actor/config/journal，以及新派生数据、seal、manifest 和验收报告。原 UNKNOWN 和 STOP 不改。
3. 恢复 BudgetAdapter 的有效视图仅把已验收的 run9900001 视作训练可用 COMPLETE；该 run 永远禁止重新 claim。其余仅原 run9900002..9900512，共511个 PLANNED slots 按原身份继续，不加预算。
4. 原账本可以继续记录这511班的 claim/complete；**全部恢复 commit/checkpoint 必须写 sidecar**。仅覆盖 job/resume 不够：如果把包含首班 UNKNOWN 的 u1 写进原 commits，会破坏原 _state 的一致性规则。
5. sidecar 对16次连续更新逐次绑定恰好32个有效 COMPLETE、尚未提交的 canonical jobs、唯一 batch_id、数据 SHA 和 checkpoint SHA。恢复 controller 使用独立控制和启动目录，以明确恢复 manifest 认可旧 STOP；不删除旧 STOP、Java once claim、旧 master/config 或旧审批。
6. 报告双重口径：原账本 UNKNOWN1 保留为故障历史；恢复有效视图可包含验收完成1及后续完成511。物理新班数仅511，总授权仍512，合并预留保持20983。

如果“不能清 UNKNOWN”允许经完整证据验收后把 SQL 当前状态显式迁移为 COMPLETE，同时永久保存 UNKNOWN 前像、原 error、不可变审计事件与 STOP，则可设计独立离线事务工具，原状态机校验也允许额外字段。但现有 API 不支持该操作；它与严格保留 UNKNOWN 当前状态的约束不同，本审查没有选择或执行此方案。

所需恢复测试、精确 source/数据 SHA 与首班账本前像见 [review.json](review.json)。
