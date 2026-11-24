# 首班 READY 前停止与恢复验证

原控制器在查询第 1 次 PPO 更新的提交记录时抛出 `ValueError: committed update absent`。首次更新进入训练前尚无提交记录，`mixed_runtime._train` 将该状态定义为 `None`。原账本继承的严格查询把它当作异常，导致监听和 READY 均未启动。[原控制器日志](../../../../../../runs/bg128-continuous-mixed-20261119-001/launch/controller.log)

| 阶段 | 验证结果 |
| --- | --- |
| 原停止位置 | 2,424 个 `PLANNED`、0 个提交、0 个物理班，没有 READY、Java claim、样本目录或 AnyLogic 物料目录。初始化所得 u0 已写入和封存。 |
| 恢复查询 | 对原 SQLite 只读验证，u1 和 u32 的未提交查询返回 `None`，越界查询拒绝。存在的提交沿用严格的账本与校验。原数据库字节 SHA256 为 `c0cda253e209ce3a542773c2a0f827a07c754768e2cf9b4aa46467380849549d`，原账本和 STOP 保留。 |
| Java 恢复入口 | 编译并比对全部 2,424 个冻结工况和身份。隔离进程测试验证首班参数、新控制目录、错误 ordinal 拒绝，以及旧 STOP 可留存。 |
| 首批采集接口 | 在临时目录走原 32 班的顺序、READY 发布、Java claim 核验、完成回执和材料角色字段检查，32/32 通过。 |
| 优化与衔接 | 用单独的 32 班离线测试数据执行真实 4-epoch 优化流程，写出靶值、保存并精确恢复 u1、提交临时账本。再次读取 u1 使用已提交检查点，随后 u2 首班可正常认领。 |

恢复沿用原 run、种子、配置及 u0，不增加物理班次或新预算。原 PPO 核心 SHA256 为 `8e4cfa31c622f5e33756e304eea3f7526547db72ae93f7d9f00808845d61ef96`，核验前后一致。上述恢复测试在临时目录完成，实际 AnyLogic 恢复尚未启动。

[恢复接入清单](../../../../../../change_plans/token_attention_001/business_graph_summary_009/generalization_001/grade_assignment_generalization_001/continuous_training_001/recovery_001/manual_anylogic.md)
