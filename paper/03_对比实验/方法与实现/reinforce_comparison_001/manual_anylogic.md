# Full S02 整班 REINFORCE 手工接入

已准备好 [S02 独立模型副本](../../../../anylogic/3d/BG128-REINFORCE-S02/BG128-REINFORCE-S02.alp)，可直接打开其中的 `RL_Training`。副本的模型名称为 `BG128-REINFORCE-S02`，保留 Java 包 `fusion`。副本已应用下表两项设置，并附带全部 25 个模型资源。[源模型、副本及资源摘要](../../../../reports/token_attention_001/business_graph_summary_009/reinforce_comparison_001/model_copy_s02.json)

在标准九车 [DRL-PPO-baseline1.alp](../../../../anylogic/3d/DRL-PPO-baseline1/DRL-PPO-baseline1.alp) 的运行副本中操作。该版本已把四装点矿废角色、品位和半幅、路段限速、物料事件目录及每班配置握手接到 `Baseline1CampaignPlan`。本次在界面中改两处。

| 界面位置 | 当前工程中的内容 | 修改与作用 | 验证 |
| --- | --- | --- | --- |
| Additional Class → `Baseline1CampaignPlan` → 类源码 | 原 Full/GAT1 恢复活动的分组 helper，`resumeOffset()` 为 3254 | 将整个类正文替换为[本次 Java 类](prepared_001/Baseline1CampaignPlan.java)，保留编辑器管理的包声明。这会将 Freeform 的 run、种子、工况和目录指向本次 1,024 行清单。 | Build 后 `jobCount()` 应为 1024，`runIndex(0)` 为 10090001，`runIndex(1023)` 为 10091024。 |
| `RL_Training` → Properties → Number of runs | 3642 | 改为 `1024`，覆盖第 0–1023 个索引。Use Freeform Parameters 开启，Allow parallel evaluations 关闭，串行运行。 | 第 31、32 个零基索引跨入下一批，recording 和 phase 配置摘要随批次切换。 |

核对 `Main → Variables → dispatchInferenceHost / dispatchInferencePort` 为 `127.0.0.1 / 50051`。Main 的 On startup 调用 `Baseline1CampaignPlan.awaitConfigSha256(...)`。派车策略、随机速度标准差和物料开关由 Freeform 的 `dispatchPolicy`、`freeSpeedFactorStd`、`materialEnabled` 提供，值依次为 `3`、`0.05`、`true`。每班运行到 28,800 秒。握手检查配置字节，Python 用实际 START RPC 校验当前矿源角色、品位、速度和观测合同。

### 离线预检

```bash
cd /home/jinx/Faith/experiments/lagrangian_ppo_v1
PLAN=change_plans/token_attention_001/business_graph_summary_009/reinforce_comparison_001
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B "$PLAN/run.py" verify
```

在[启动清单](prepared_001/launch_identity.json)核对本活动、各阶段配置 SHA256、Python 输出与 AnyLogic 物料目录。[逐班清单](prepared_001/identities.csv)按运行顺序列出身份。训练种子 S02 对应 Full 的归档初始化，候选名称仍使用其已有的 `s01_BG128`。

### 实际启动时

1. 保存并 Build 上述副本。确认上一次使用 `50051` 的仿真已经结束。
2. 在上面的项目目录执行下面的命令并保留该终端。这个命令会在复核最新合并账本后，**正式登记 1,024 班并启动 Python 控制器**。

   ```bash
   /home/jinx/miniconda3/envs/anylogic-env/bin/python -B "$PLAN/run.py" run --authorize-start
   ```

3. 在 AnyLogic 点击一次 **Run RL_Training**。Java 会等控制器写好本班 READY，再核对配置 SHA256、领取身份并发送 RPC。该 Java 类使用手动启动的 Python 控制器。启动后查看 `runs/bg128-reinforce-full-shift-20261127-s02/training_progress.json`。
4. 结束时核对 `suite_complete.json` 的 `training_shifts=1024`、`training_updates=32` 及 `final_actor`。每个 `arms/s01_BG128/updates/NNNN/training_summary.json` 含班均回报、基线、实际决策数和 Adam 步数。若出现 STOP，保留现有回执与领取记录，以出错 run 和 phase 排查，防止重复消耗班次。
