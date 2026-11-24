# BG128 四训练种子：单 ALP 一键串行

在同一个 [原 ALP](../../../../../../anylogic/3d/DRL-PPO-baseline1/DRL-PPO-baseline1.alp#L12) 内，**一次 Run 自动串行 s02 → s03 → s04 → s05**。每个种子 1,724 班（1,024 训练 + 700 固定 u32 随机评测），总计 **6,896 班**。本次只生成和核验配置，未直接编辑 ALP、未启动或登记预算。速度沿用首轮：`freeSpeedFactorStd=0.05`、截断 `0.85–1.10`、内生跟驰与局部动态速度开启。

## 核对一键套件和运行身份

只粘贴 [组合 Java helper](prepared_oneclick_003/Baseline1CampaignPlan.java#L1)，**不要粘贴子目录里的单种子 helper**。[总启动清单](prepared_oneclick_003/launch_identity.json#L1) 绑定组合 helper 的 SHA256 及以下四套身份：

| AnyLogic index | 种子 / 首班 run | 首班身份及全部阶段 SHA256 / 目录 | 完整 run／双种子 |
| --- | --- | --- | --- |
| 0–1723 | s02 / 10003001 | [启动清单](prepared_oneclick_003/s02/launch_identity.json) | [逐班表](prepared_oneclick_003/s02/identities.csv) |
| 1724–3447 | s03 / 10005001 | [启动清单](prepared_oneclick_003/s03/launch_identity.json) | [逐班表](prepared_oneclick_003/s03/identities.csv) |
| 3448–5171 | s04 / 10007001 | [启动清单](prepared_oneclick_003/s04/launch_identity.json) | [逐班表](prepared_oneclick_003/s04/identities.csv) |
| 5172–6895 | s05 / 10009001 | [启动清单](prepared_oneclick_003/s05/launch_identity.json) | [逐班表](prepared_oneclick_003/s05/identities.csv) |

保留原模型、Java 包 `fusion` 和单一端口 `127.0.0.1:50051`。不得与使用该端口的速度压力测试等活动同时运行。四个种子的配置、输出、物料目录、READY/STOP/claim、账本与 checkpoint 仍分开。旧双路和旧手动串行套件仅留作审计。

## 只改两处

| AnyLogic 位置 | 修改前 | 修改后、影响与验证 |
| --- | --- | --- |
| Additional Class → `Baseline1CampaignPlan` | 速度压力或单种子 helper | 全文替换为组合 helper。构建后 `jobCount()` 为 6896，`runIndex(0)` 为 10003001，`runIndex(1724)` 为 10005001。 |
| RL_Training → Properties → Number of runs | 原件为 **40**；如手动接过串行版则为 **1724** | 改为 **6896**。保持串行，关闭仿真实验内部并行和重复。 |

保留 Main → Variables → `dispatchInferencePort` 的初始值 `50051` 和 `dispatchInferenceHost` 的 `"127.0.0.1"`，不改 `speedControlPort`。保留 `RL_Training → Properties → Parameters → Freeform → freeSpeedFactorStd` 当前表达式 `Baseline1CampaignPlan.freeSpeedFactorStd((int) index)`。串行版 helper 每班返回 **0.05**。四站初始化、启动握手、物料审计和物理步长继续沿用；若误用压力测试值，Python 会在 START 拒绝。

## 只读预检与运行

从实验项目目录执行：

```bash
cd /home/jinx/Faith/experiments/lagrangian_ppo_v1
PLAN=change_plans/token_attention_001/business_graph_summary_009/generalization_001/grade_assignment_generalization_001/mixed_seed_replication_001
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B "$PLAN/replica_run.py" --suite "$PLAN/prepared_oneclick_003/s02/suite.json" --check
```

整个实验只点一次 **Save → Build → Run**，不手动启动 Python。首班检查 s02 的 `launch/controller.log` 已监听 50051 并显示 `JAVA_START_VERIFIED run_index=10003001`。下一种子的首班在前一轮 1724 班结算、u32 独评封存且端口锁释放前等待；出现 STOP 或交接超时不启动后续种子。将 `--check` 换成 `--status` 并更换 suite 中的种子名可查各轮。最后以 s05 的状态为 `COMPLETE` 为准。保留高频诊断关闭，不因显示等待而再次点击 Run。

## 校验提速范围

每班首次完整验证 Actor、精确概率和私有采样 RNG；第二次准入读回与训练入口验证相同字节、Actor、种子、来源和数据完整性后，复用这一证明。不降级实际 RPC/schema、品位与物料、完整班次或 checkpoint 校验。不改变 32 次 PPO 更新、每次训练样本、优化器配置及独评种子。每个 phase 的 `validation_summary.json` 保留回放复用计数。

[离线校验基准](../../../../../../reports/token_attention_001/business_graph_summary_009/generalization_001/mixed_seed_replication_001/oneclick_validation_benchmark.json#L1)：三个实际历史班次的同一校验链平均 11.62 → 7.91 秒；这不是全程训练加速比。
