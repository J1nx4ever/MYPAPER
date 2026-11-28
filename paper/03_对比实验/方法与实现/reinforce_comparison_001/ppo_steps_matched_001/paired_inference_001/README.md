# 步数对齐版 REINFORCE S02/u32 固定策略评测

本轮启动 42 班后，原窗口在第 43 班切换了模型。后续见[接续评测](recovery_001/README.md)。

训练于北京时间 2026 年 11 月 28 日 07:31 封存了 1,024 个完整班、32 批和 23,612 个 Actor 梯度步。本次推理从封存的最终 Actor 开始，保留合法动作掩码并按策略概率采样，更新步数为零。[训练封存报告](../../../../../../runs/bg128-reinforce-ppo-steps-20261128-s02/suite_complete.json)

正式评测使用七个工况，每组 100 班，依次为标称、高低矿换位、两个留出角色排列、新品位数值、联合变化、废石换位。每班匹配历史 Full S02 PPO 固定 u32 的工况、环境和 Actor 采样种子。700 个物理 run 使用本活动自己的身份。四个品位窗口、矿废达产、联合成本和车队累计等待沿用训练合同。报告汇总全部 700 对物理结果。[历史 Full S02 评测封存](../../../../../../runs/bg128-continuous-mixed-20261120-s02/suite_complete.json)

本轮使用独立 AnyLogic 评测模型的 `RL_Training` 实验作为仿真入口。启动记录核对首班的 READY、Java 领取、START RPC 和完整物料回执。[七阶段配置和启动身份](prepared_001/launch_identity.json) · [700 班双种子](prepared_001/identities.csv) · [AnyLogic 模型和接入清单](manual_anylogic.md)
