# AnyLogic 启动卡

已通过离线预检。正式物理班次由 AnyLogic 的现有 `RL_Training` 实验启动。一次运行依次执行 u16 独评、u17–u24 训练、u24 独评、u25–u32 训练和 u32 独评。

| 精确位置 | 修改前 | 修改后 | 理由和验证 |
| --- | --- | --- | --- |
| Additional Class → `Baseline1CampaignPlan` → Java 源码 | 最近一次完成的 100 班发现批次 helper | 全文替换为 [本次冻结的 Java helper](prepared_001/Baseline1CampaignPlan.java) | 载入新 master、启动 Python 控制器并按阶段绑定配置 SHA。Build 后 `jobCount()` 和 `frozenPlanJobCount()` 均为 812，`resumeOffset()` 为 0。 |
| `RL_Training → Properties → Number of runs` | `100` | `812` | 串行执行 512 班续训及 300 班固定推理。最后一个 run 为 `9990812`。 |
| `RL_Training → Parameters → Freeform` | 现有 `Baseline1CampaignPlan.*((int) index)` | 沿用现有表达式 | 参数按冻结 master 读取。 |
| `Main → On startup` | 现有 `Baseline1CampaignPlan.awaitConfigSha256(...)` 及 RPC、物料初始化 | 沿用现有调用 | 首班自动启动控制器，根据 ready 配置确立身份，然后创建物料事件记录。 |
| `RL_Training` 运行设置 | 串行，并行评估和重复执行均关闭 | 沿用现有设置 | 每班按顺序与 Python 端完成一次握手。 |

Save → Build → **Run 一次**。Python 随首次 Main 启动，不需要另开训练命令。进入 u24 和 u32 固定推理时模型自动切换，每个阶段使用对应封存 checkpoint。若出现 STOP，保留现场和本次 claim。

## 启动身份

| 项目 | 值 |
| --- | --- |
| Campaign | `bg128-u16-to-u32-20261119-001` |
| 首班 recording_id | `bg128-u16-to-u32-20261119-001-s01_BG128_eval_u0016` |
| 首班配置字节 SHA256 | `ebf19402f50dbb40d1c83dc34c5141c4b32ef31664f5a0af9fde6e93bccb0197` |
| Suite 字节 SHA256 | `c52e840dfdd2387e0056eb0073925a8061aab82417691bf5c6b2d946563b4aa5` |
| u16 原始 checkpoint 字节 SHA256 | `f0bf60a3091f85eb4f163806805f495ff70d0f7aa325906203c82c92ecc30e43` |
| 首班 run_index | `9990001` |
| 首班环境 seed / Actor 采样 seed | `93625302859898 / 196394957391198` |
| 本次源 Actor 初始化 seed | `75597143230906` |
| Python 总输出 | `/home/jinx/Faith/experiments/lagrangian_ppo_v1/runs/bg128-u16-to-u32-20261119-001` |
| 首阶段 Python 输出 | `/home/jinx/Faith/experiments/lagrangian_ppo_v1/runs/bg128-u16-to-u32-20261119-001/evaluation/eval_u0016` |
| AnyLogic 物料事件父目录 | `/home/jinx/Faith/experiments/lagrangian_ppo_v1/runs/bg128-u16-to-u32-20261119-001_materials` |
| 首阶段 AnyLogic 物料事件目录 | `/home/jinx/Faith/experiments/lagrangian_ppo_v1/runs/bg128-u16-to-u32-20261119-001_materials/s01_BG128_eval_u0016` |

| 顺序 | 新 run_index | 班次 |
| --- | --- | ---: |
| u16 固定推理 | `9990001–9990100` | 100 |
| u17–u24 续训 | `9990101–9990356` | 256 |
| u24 固定推理 | `9990357–9990456` | 100 |
| u25–u32 续训 | `9990457–9990712` | 256 |
| u32 固定推理 | `9990713–9990812` | 100 |

[全部 812 班 run/seed/配置 SHA 清单](prepared_001/identities.csv) · [全部 19 阶段 recording、配置字节 SHA 和输出路径](prepared_001/launch_identity.json) · [离线预检记录](../../../../reports/token_attention_001/business_graph_summary_009/training_extension_001/preparation_validation.json)

运行进度见 Python 总输出的 `run_status.json`、`training_progress.json` 和各独评阶段的 `run_status.json`。完成判据为 812 班 COMPLETE、16 次追加更新全部提交、三个独评阶段各 100 班零优化，输出逐班配对报告。
