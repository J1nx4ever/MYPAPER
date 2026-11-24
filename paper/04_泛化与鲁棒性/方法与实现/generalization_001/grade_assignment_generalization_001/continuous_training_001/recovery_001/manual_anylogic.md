# BG128 连续混合训练首班启动恢复

本页记录第一次恢复。首班 `10000001` 后续已完成，当前续跑使用[第二次恢复清单](../recovery_002/manual_anylogic.md)。

原启动在 Python 查询 u1 的提交记录时停止。此时 u1 尚未开始，提交表应为空，旧查询把空表当成错误。2,424 个班仍为 `PLANNED`，没有 READY、Java claim、物料事件或训练样本。恢复继续使用已封存的 u0、原账本及原首班，新增登记为 0。

## AnyLogic 接入

先结束原先已经报 STOP 的实验窗口。在当前保存的 [DRL-PPO-baseline1.alp](../../../../../../../anylogic/3d/DRL-PPO-baseline1/DRL-PPO-baseline1.alp) 中，用[恢复 helper](prepared_001/Baseline1CampaignPlan.java)全文替换 **Additional Class → Baseline1CampaignPlan**。原 helper 连接原启动路径，恢复 helper 使用同一工况表，把启动和 READY/STOP 握手切到独立恢复目录。其余代码与原 helper 一致。

核对 `RL_Training` 为 **2,424 班**、串行、关闭重复、从 index 0 开始。四个站点的 `Main.init1`、Main 启动握手及 `enableMaterialAudit=true` 沿用已保存的混合设置。保存和构建后，在 `RL_Training` 点 Run 一次。

首班 run_index 是 `10000001`，环境种子 `251214294863295`，Actor 动作采样种子 `88752031272144`，recording_id 为 `bg128-continuous-mixed-20261119-001-s01_BG128_train_u0001`。首阶段配置字节 SHA256 为 `5161204b2b3b2f7adc2b6f1218d417b23f0f3fa44dc6d86b6888ed95704ced94`。[恢复启动身份及目录](prepared_001/launch_identity.json) · [逐班 run 与种子清单](../prepared_001/identities.csv)

## 预检和查看状态

```bash
cd /home/jinx/Faith/experiments/lagrangian_ppo_v1
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B change_plans/token_attention_001/business_graph_summary_009/generalization_001/grade_assignment_generalization_001/continuous_training_001/recovery_001/recovery_bridge.py --check
```

该命令检查原账本、原停止记录、保留的 u0 和恢复工件。启动后用同一入口的 `--status` 查看进度。恢复控制日志写到原 Python 输出根目录下的 `recovery_001/launch/controller.log`，恢复状态写到 `recovery_001/run_status.json`。训练样本和 AnyLogic 物料日志仍落在原冻结目录。正常的首班应出现 `LISTENING`、`JAVA_START_VERIFIED run_index=10000001`，班末出现 `[MATERIAL_AUDIT_PASS]`，32 班准入后进度应为 `UPDATE_COMMITTED`、`update_index=1`。

原 `control/STOP.properties`、原 `run_status.json`、首次启动日志均为第一次故障的封存证据。恢复使用自己的 STOP 和状态文件。若恢复窗口也报 STOP，依据恢复状态与账本判断运行是否已被认领，再确定下一步。
