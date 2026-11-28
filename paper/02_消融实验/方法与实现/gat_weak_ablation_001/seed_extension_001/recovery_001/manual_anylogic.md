# 从 GAT1 s04 的固定 u32 评测继续

恢复列表从原 GAT1 s04 的第 1530 个索引（零起算）开始。首班为 `joint_shift` 的配对槽 6，环境 seed 为 `238429010339408`，Actor 采样 seed 为 `162247748257471`，recording 为 `bg128-gat-depth-20261125-gat1-s04-s01_BG128_eval_joint_shift_new`，RPC 配置字节 SHA256 为 `34a5e7657d02ebe1c44cc42d280dbb699c194134c1c51e4009066cf19e875c59`。对应 Python 输出在 `runs/bg128-gat-depth-20261125-gat1-s04/evaluation/joint_shift/new/`，物料事件在 `runs/bg128-gat-depth-20261125-gat1-s04_materials/s01_BG128_eval_joint_shift_new/`。[原逐班身份表](../prepared_001/gat1-s04/identities.csv)

| 恢复索引 | 模型与训练种子 | 原物理 run | 剩余班次 |
| --- | --- | --- | ---: |
| 0–193 | GAT1 s04 固定 u32 | 10049255–10049448 | 194 班评测 |
| 194–1917 | Full s05 | 10049449–10051172 | 1024 班训练、700 班评测 |
| 1918–3641 | GAT1 s05 | 10051173–10052896 | 1024 班训练、700 班评测 |

两组 s05 在各自首班运行原冻结启动器。逐班训练及七工况评测身份、配置、Python 输出与事件目录见 [Full s05](../prepared_001/full-s05/identities.csv) 和 [GAT1 s05](../prepared_001/gat1-s05/identities.csv)。

## AnyLogic 正式启动入口

本机独立 Java 宿主通过了引擎工厂预检，但在创建物理模型时返回 `Invalid start location for AnyLogic Runtime`。本轮使用 AnyLogic 桌面程序及其官方 `-r` 入口启动。

在 AnyLogic 中从项目的 `anylogic/3d/DRL-PPO-baseline1/DRL-PPO-baseline1.alp` 另存独立运行副本。仅在副本的模型树中修改以下两处。

| 位置 | 原设置 | 恢复设置 | 作用与验证 |
| --- | --- | --- | --- |
| 九车模型 → Additional classes → `Baseline1CampaignPlan` → Java 源码 | [原四组合并类](../prepared_001/Baseline1CampaignPlan.java) | 用[恢复合并类](prepared_001/Baseline1CampaignPlan.java)全文替换 | `runIndex(0)` 应为 10049255，`runIndex(194)` 为 10049449，`runIndex(1918)` 为 10051173，`runIndex(3641)` 为 10052896。 |
| `RL_Training` → 参数变化 → 运行次数 | 6896 | 3642 | 每个恢复索引运行一次。保留原单线程执行、无重复与 Freeform 参数表达式。 |

保存副本，按 F7 构建。在副本中运行 `RL_Training` 一次，也可使用以下命令启动同一实验。

```bash
/home/jinx/apps/AnyLoigc/anylogic/anylogic -r "/独立运行副本的绝对路径.alp" RL_Training
```

首班 Main startup 会用原 run、环境 seed 和 recording 回调恢复桥。检查 `control/java_claims/run_10049255.once.claim`、恢复控制器日志和该班的 `COMPLETE` 账本回执。最后核对三个活动的 `suite_complete.json` 及固定 u32 七工况每组 100 班报告。此次两组 s05 应分别有 32 个训练更新，三组固定评测均应有 700 个配对槽及零优化器更新。
