# 全班 REINFORCE 步数对齐实验的 AnyLogic 接入

本活动的独立模型为 `BG128-REINFORCE-S02-MATCHED`。Python 控制器给该副本提供每班冻结的配置与策略权重。[可启动的 AnyLogic 模型副本](../../../../../anylogic/3d/BG128-REINFORCE-S02-MATCHED/BG128-REINFORCE-S02-MATCHED.alp) · [模型与资源核验](../../../../../reports/token_attention_001/business_graph_summary_009/reinforce_comparison_001/ppo_steps_matched_001/model_copy_s02_matched.json)

需要手动重建时，在 AnyLogic 打开原 S02 九车训练模型并另存为 `BG128-REINFORCE-S02-MATCHED`。将源模型相邻的资源复制到新模型相同的相对位置，再按下表设置。[源模型](../../../../../anylogic/3d/BG128-REINFORCE-S02/BG128-REINFORCE-S02.alp)

| AnyLogic 位置 | 当前 S02 训练副本 | 本活动设置和验证 |
| --- | --- | --- |
| 项目树 → Additional classes → `Baseline1CampaignPlan` | 原整班 REINFORCE 的逐班路径、种子和配置 | 将类正文替换为本活动 Java 类，使准备信号、采样身份和事件记录指向新活动。Java 包为 `fusion`。Build 后核对 `runIndex(0)=10094001`、`runIndex(1023)=10095024`。[Java 类](prepared_001/Baseline1CampaignPlan.java) · [启动清单](prepared_001/launch_identity.json) |
| 项目树 → `RL_Training` → Number of runs | 1,024 | 设为 1,024，启用 Use Freeform Parameters，关闭 Allow parallel evaluations 和 Use replications。覆盖零基索引 0–1023。 |
| Main → Variables → `dispatchInferenceHost`、`dispatchInferencePort` | `127.0.0.1`、`50051` | 核对和启动清单相符。Main 启动回调通过 Java 的 `awaitConfigSha256` 核验 READY 配置并领取身份后发送 START RPC。 |

Python 入口位于项目目录。离线预检命令如下。

```bash
cd /home/jinx/Faith/experiments/lagrangian_ppo_v1
PLAN=change_plans/token_attention_001/business_graph_summary_009/reinforce_comparison_001/ppo_steps_matched_001
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B "$PLAN/run.py" verify
```

正式启动时，在该目录执行以下命令登记本活动并启动控制器，随后在独立副本里运行一次 `RL_Training`。共享 RPC 端口可通过正在运行的控制器状态核对。

```bash
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B "$PLAN/run.py" run --authorize-start
```

检查首班的 Java 领取凭证、Python 物料回执、环境和 Actor 采样种子及配置 SHA256。第一批收集齐 32 班后，训练回执应记录 4 轮和 732 次 Actor Adam。最终回执应封存 1,024 班和 23,612 个 Actor 步。
