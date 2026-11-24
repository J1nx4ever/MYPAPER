# BG128 第一个完整班后的续跑

本页记录第二次恢复。前两班后出现跨班配置锁错误，当前继续使用[逐班 RPC 恢复清单](../recovery_003/manual_anylogic.md)。

`recovery_001` 在 `10000001` 实际结束后写出 STOP。Python 保存训练数据时使用了旧 Task Query 的候选契约，把 `s01_BG128` 拒为未知候选。原 RPC 与 Java 物料日志通过全量重放，184 次动作、四窗和班末均有效。本次已用 BG128 存储重新准入，原账本为 **1 COMPLETE、2,423 PLANNED、0 次训练提交**。已结束的班仍参加 u1 的 32 班批次。

## AnyLogic 只改两个位置

1. 在 [DRL-PPO-baseline1.alp](../../../../../../../anylogic/3d/DRL-PPO-baseline1/DRL-PPO-baseline1.alp) 中，打开 **Additional Class → `Baseline1CampaignPlan`**。将旧 2,424 班启动 helper 或 `recovery_001` helper 全文替换为[第二次恢复的 helper](prepared_001/Baseline1CampaignPlan.java)。该类的 `runIndex(0)` 指向 **10000002**，重用原主表和工况表，控制文件写入独立的 `recovery_002/control`。
2. 在 **`RL_Training` → Properties → Number of runs** 将 **2424 改为 2423**，保持串行、关闭重复、索引从 0 开始。保留 Freeform 中的 `Baseline1CampaignPlan.*((int) index)`，以及保存过的 `Main.init1` 四站物料参数和物料审计。

关闭先前的报错实验，**Save → Build → Run** 一次。需要核对保存生效时，查看 `Baseline1CampaignPlan` 中的 `CONTROL_DIRECTORY` 以 `/recovery_002/control` 结尾，并核对实验显示 2,423 班。两份旧 STOP 与原启动日志继续保存。

[首个剩余班身份、配置字节 SHA256 及输出路径](prepared_001/launch_identity.json) · [原全部 run 和双种子](../prepared_001/identities.csv)

## 检查运行状态

```bash
cd /home/jinx/Faith/experiments/lagrangian_ppo_v1
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B change_plans/token_attention_001/business_graph_summary_009/generalization_001/grade_assignment_generalization_001/continuous_training_001/recovery_002/recovery_bridge.py --check
```

`--check` 只读核对已完成的首班数据和未用身份。启动后用 `--status` 查看恢复进度。新日志位于 `runs/bg128-continuous-mixed-20261119-001/recovery_002/launch/controller.log`，新 STOP 会显示其完整文件路径和原因。首条新 Java 凭据应为 `JAVA_START_VERIFIED run_index=10000002`。再完成 31 班后，由已恢复的首班和新准入数据一起提交 u1。

[首班接纳报告](../../../../../../../reports/token_attention_001/business_graph_summary_009/generalization_001/continuous_training_001/recovery_002/admission/reconciliation.json)
