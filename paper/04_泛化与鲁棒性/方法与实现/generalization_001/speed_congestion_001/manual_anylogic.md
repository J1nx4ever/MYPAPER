# BG128/u32 速度扰动独评的 AnyLogic 接入

活动标识为 `bg128-speed-congestion-20261120-001`。固定混合训练 u32，使用 40 个独立物理班。前 20 班的 `freeSpeedFactorStd` 为 0.15，后 20 班为 0.25。每班实际运行身份以冻结清单为准。 [首班身份与配置 SHA256](prepared_001/launch_identity.json) · [40 班 run、环境种子和 Actor 采样种子](prepared_001/identities.csv)

## 手工修改位置

打开项目中的 [DRL-PPO-baseline1.alp](../../../../../anylogic/3d/DRL-PPO-baseline1/DRL-PPO-baseline1.alp)。结束已完成的上一轮实验窗口后，在新的 `RL_Training` 运行前完成以下三项修改。

| 位置 | 修改前 | 修改后 |
| --- | --- | --- |
| Additional Class → `Baseline1CampaignPlan` → 类正文 | 混合品位活动的 helper | 全文替换为本活动的 [Baseline1CampaignPlan.java](prepared_001/Baseline1CampaignPlan.java)。保留编辑器已有的 package 设置。 |
| `RL_Training` → Properties → Parameters → Freeform → `freeSpeedFactorStd` | 空白，使用 Main 默认 0.05 | 填 `Baseline1CampaignPlan.freeSpeedFactorStd((int) index)` |
| `RL_Training` → Properties → Number of runs | `2422` | `40` |

`RL_Training` 继续使用 Freeform、单线程串行、关闭 Replications，终止时间为 28,800 秒。现有 `runIndex`、`environmentSeed`、`recordingId`、`eventRoot` 和 `materialRoot` 的 helper 表达式接入本活动同名接口。保留当前 `Main` 启动握手、四站矿料初始化及路线限速函数。 [参数位置](../../../../../anylogic/3d/DRL-PPO-baseline1/DRL-PPO-baseline1.alp#L25554) · [速度标准差所在参数行](../../../../../anylogic/3d/DRL-PPO-baseline1/DRL-PPO-baseline1.alp#L25621)

默认的 `freeSpeedFactorMin=0.85`、`freeSpeedFactorMax=1.10`、`useLocalDynamicSpeed=true`、`useEndogenousCongestion=true`、`deterministicCheck=false` 已符合本活动，Freeform 中可沿用现有默认。若曾手工覆盖这些参数，请恢复以上值。Python 在每次实际 RPC 核对 `config.runtime_scene.env` 中的标准差、上下限和启用状态，以及冻结的完整工况。 [运行工况字段](../../../../../python/runtime_scenario.py#L17) · [实际 RPC 校验](../../../../../python/collect_offline.py#L143)

Java helper 用 `index` 对齐 master 的物理身份，读取经过 SHA256 校验的 [速度表](prepared_001/speed_scenarios.csv) 和 [矿料与道路表](prepared_001/scenarios.csv)。索引 0–19 和 20–39 分别返回 0.15 和 0.25。Freeform 读取发生在仿真初始化前，空载与重载运输腿的原抽样函数继续使用当前班参数。保留原 `Main` 启动握手即可启动 Python 控制器，端口仍为 50051。 [速度抽样](../../../../../anylogic/3d/DRL-PPO-baseline1/DRL-PPO-baseline1.alp#L7798) · [空载运输腿](../../../../../anylogic/3d/DRL-PPO-baseline1/DRL-PPO-baseline1.alp#L16065) · [重载运输腿](../../../../../anylogic/3d/DRL-PPO-baseline1/DRL-PPO-baseline1.alp#L16187)

## 离线核对与实际启动

从项目目录运行以下只读检查。确认新清单与界面配置一致后，执行 **Save → Build → Run**，运行 `RL_Training` 一次。

```bash
cd /home/jinx/Faith/experiments/lagrangian_ppo_v1
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B change_plans/token_attention_001/business_graph_summary_009/generalization_001/speed_congestion_001/speed_campaign.py --check
```

运行期间用同一入口的 `--status` 查看账本状态。第一班的 `JAVA_START_VERIFIED` 与 `launch_identity.json` 的首个 run 应一致。切至第 21 班时，按清单核对实际 START 已读取 0.25。以封存 RPC 和实际 40 班完成记录为验收依据。

## 扰动和拥堵的判读

`freeSpeedFactorStd` 是截断前正态分布的标准差。原有 0.85–1.10 截断和路段限速继续生效，实际速度的离散程度由采样、限速和跟驰共同决定。两档评估的合规、吞吐和等待差值描述整个工况的结果。若核验跟驰强度，按运输腿查看实际旅行时间和前车最小车距，按在途车秒比较 `FOLLOW` 和 `BRAKE` 状态占比。现有短时运动 RPC 字段和决策点队列只能描述采样到的状态。每班汇总或低频运输腿摘要已经能支持这项区分。 [跟驰与制动分支](../../../../../anylogic/3d/DRL-PPO-baseline1/DRL-PPO-baseline1.alp#L7381) · [空载及重载旅行摘要](../../../../../anylogic/3d/DRL-PPO-baseline1/DRL-PPO-baseline1.alp#L16493) · [决策点队列口径](../../../../../single_critic_task_query_v1/evaluation.py#L88)
