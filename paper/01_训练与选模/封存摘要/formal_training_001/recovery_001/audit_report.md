# BG128 正式训练首班存储失败与恢复审计

2026-11-18，Asia/Shanghai。恢复准备完成时的状态：首班已完成并接纳，余下511班尚未启动，PPO更新为0。

## 原因与影响

首班 `9900001` 的 AnyLogic 仿真正常结束；Python将原始采样转为训练HDF5时，沿用了 `single_critic_task_query_v1.storage`。该存储器按旧TQ候选合同校验 `s01_BG128`，抛出 `ValueError('unknown task-query candidate')`。错误发生在派生数据写入之前，没有发布不完整的训练HDF5，也未执行PPO更新。

原首次接入测试遗漏了实际BG身份的 canonical write → seal → strong read 闭环，这是本次修复覆盖的边界。网络前向、业务边和联合成本定义没有因此改变。

## 修复

新增独立 `business_graph_summary_storage_v1`，仅切换BG身份、图合同/hash与schema；保留原canonical存储、封存和强读规则。新增 `business_graph_summary_recovery_v1` 继续原512班计划，调用新的writer和strong reader，使用独立恢复控制目录。

原396个源码pin、原准备包、BG128网络、C128、四组关系、两层GAT、成本函数与训练设置保持不变。受保护 `python/episode_budget_ppo.py` 的SHA256仍为 `8e4cfa31c622f5e33756e304eea3f7526547db72ae93f7d9f00808845d61ef96`。没有直接编辑ALP。

## 首班证据

184条transition、185个1519D观测、四窗口、物料journal、RPC配置、Actor逐步采样与log probability均完成离线复核。派生HDF5已封存并由新strong reader及正式collection接口再次读取。联合成本和MC目标核验通过；随机初始化Critic给出有限预测，不将其解释为已学习语义或性能提升。

原首班策略digest为 `08f301996e7c944b566e398497c68d3b1cd9731ff7f35b97d4872a7067a91908`，Actor文件SHA256为 `c4aeef10c24533427234ca360e497840bb30810bcd880a9ee2d2cf896ba64653`。恢复使用同一初始化策略，两套Adam状态均为空。首批按原顺序组成已完成run9900001加其后31班，32班全部通过同策略核对后才执行首次更新。

## 预算与历史保留

依据完整原始证据，仅将本次首班UNKNOWN审计迁移为COMPLETE。迁移前的整数据库、UNKNOWN原文、checksum及error被单独只读封存；原STOP、首班once claim、启动日志和配置保留。当前canonical行保留原error并附 `reconciled_from`、审计和前像引用。没有清除其他历史UNKNOWN。

原512班上限和511个PLANNED身份逐字节不变，commit为0；恢复不登记新预算、不增加班次。独立合并视图容量仍为20983：COMPLETE20462、UNKNOWN1、PLANNED520，包含其他历史实验。该统计是恢复前的审查快照，后续正常采集会改变活动账本。

## 验证

- 首班完整离线证据：15项通过，零优化、零RPC。
- 新存储与恢复runtime专项：12项通过，含真实184行读写、错合同、错hash及封存篡改拒绝。
- 账本迁移工具：临时数据库27项通过，含整事务rollback、重复迁移拒绝与其他511行不变。
- 生产迁移独立复核：27项通过；原BudgetLedger可resume。仅在临时副本核验原首批32班commit接口，生产没有模拟班次或commit。
- 恢复包完整门禁：13项通过；原396个源码pin和PPO核心校验通过。
- 实际恢复helper已Java编译，全部511行逐项映射原master的后511行；index0→9900002，index510→9900512；已完成首班和越界索引在启动桥前拒绝。
- 根代理另行运行正式 `runtime preflight --recovery`，结果 `RECOVERY_PREFLIGHT_PASS_NO_LISTENER`。

## 用户恢复步骤与交付

见[手工清单](../../../../../change_plans/token_attention_001/business_graph_summary_009/formal_training_001/recovery_001/manual_anylogic.md)：替换Additional Class → Baseline1CampaignPlan全文，将RL_Training → Number of runs从512改为511，Freeform不加偏移，Save → Build → Run一次。Python由Java启动。

恢复描述SHA256：`4783b9f195ac01f64b30da1e617759a6b00fa532a634a87630492ab1e864efa6`。helper SHA256：`2d86c0ef9332d44b58f735a2014e57057a4bb80e448b1da657d801985831e0eb`。

此报告不声明恢复采样已经开始、512班训练已完成或独立评价已完成。后续实际状态以恢复controller日志、活动账本和训练commit为准。
