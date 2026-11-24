# BG128 从 u16 续训到 u32

从完成 512 班训练的原始 `3e-4/u16` 恢复 Actor、Critic、两个 Adam 优化器、训练计数和随机数状态。续训使用原有的 BG128、C128、训练 Reward、PPO 数值设置和每次更新 32 班的训练批量。u17–u32 合计追加 512 班，每次完整提交从前一更新的封存 checkpoint 接续。

u16、u24、u32 各做 100 班固定策略随机推理。三组逐槽沿用原 BG128/u13 独评的环境和 Actor 采样种子，对应全新的 run、recording 和文件目录。训练样本使用新种子。训练与评估串行，分别入账，合计新增 812 班。

主要比较是 u32 相对 u16 的同槽差异，u24 用来查看变化是在前 8 次还是后 8 次更新出现的。优先看四窗合规且矿废都达产的班次比例，另汇总各窗口、品位 RMSE、矿废达产、累计等待、联合成本和 Critic MC MSE、EV。完成后报告每个 checkpoint 的全部 100 班和逐槽结果。

[AnyLogic 启动卡](manual_anylogic.md) · [812 班 run/seed 清单](prepared_001/identities.csv) · [阶段身份及配置 SHA](prepared_001/launch_identity.json)
