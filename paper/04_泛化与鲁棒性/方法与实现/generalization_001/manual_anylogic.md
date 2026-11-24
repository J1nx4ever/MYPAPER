# BG128 u32 工况泛化一键启动卡

原 600 班入口已经使用。要从 `9991102` 接着跑，使用[499 班续跑启动卡](recovery_001/manual_anylogic.md)。

使用刚完成延训的 AnyLogic 模型，在它的泛化专用副本上操作。六个工况由同一个 `RL_Training` 批次串行执行，Python 自动加载冻结的 `3e-4/u32`。

## 手工修改

| AnyLogic 位置 | 修改前 | 修改后与验证 |
| --- | --- | --- |
| Additional Class → `Baseline1CampaignPlan` → 全部 Java 源码 | u16–u32 的 helper | 全文替换为[本次的冻结 helper](prepared_001/Baseline1CampaignPlan.java)。它读取本次 master 和工况表。首班为 `9991001`，末班为 `9991600`，`jobCount()` 为 600。 |
| Main → Functions → `init1` → 开头三个矿石 `LoadSite` 初始化块 | [修改前片段](Main.init1.LoadSites.before.java)中的 `s0`、`s1`、`s2`，其品位实参为 `0.38`、`0.30`、`0.15` | 将这三个块替换为[修改后片段](Main.init1.LoadSites.after.java)，粘贴范围到 `s2.initLoadSite(...);` 结束。各矿石站在生成品位块前采用工况参数。确认 `s2` 与 `for(LoadSite s: loadSites)` 之间保有完整的 `s3` 废石站块。若已漏掉，在此处插入[恢复片段](Main.init1.WasteSite.restore.java)。验证 `init1` 中有四个 `add_loadSites()` 调用及站号 0、1、2、3。 |
| Main → Functions → `addBidirectionalPath` → 两个 `new PathSegment(...)` | [修改前片段](Main.addBidirectionalPath.before.java)给两个构造器末参都传 `speedLimit` | 用[修改后片段](Main.addBidirectionalPath.after.java)替换，分别按各方向的起终点读取限速；此次只覆盖 `intersection1 → Crusher`。 |
| `RL_Training → Properties → Number of runs` | `812` | 设为 **600**。保持串行，关闭并行评估与重复。 |
| `RL_Training → Parameters → Freeform`、`Main → On startup` | 现有 helper 表达式与 `awaitConfigSha256(...)` | 沿用已有表达式，并核对实验 Freeform 的 `deterministicCheck=false`。首次 Main 启动由 Java 触发 Python，按冻结运行身份完成握手。 |

完成后 **Save → Build → Run 一次**。首次启动会登记本批 600 班并自动开启 Python 评估。每个工况用同一个冻结 Actor 的随机采样模式。若出现 STOP，保留当前启动记录和班次 claim，供核对失败点。

## 冻结运行身份

| 项目 | 值 |
| --- | --- |
| 首班 recording_id | `bg128-u32-generalization-20261119-001-s01_BG128_eval_nominal` |
| 首班配置字节 SHA256 | `fe76a53e60de6ea783d8afe906d29d7406d82a6b7f6022259e52bf215f9dd6c1` |
| Suite 字节 SHA256 | `2430b87f21477b0de3102acfaf3083559badd0c779e4cdedbb032fc51fd052cc` |
| Checkpoint 字节 SHA256 | `5d42df7d8e33b5f11424094268a4da894a36454e46d86889fc0218ecce6b2914` |
| 首班 run_index | `9991001` |
| 首班环境 seed / Actor 采样 seed | `138685446164644` / `257951068098007` |
| 源 Actor 初始化 seed | `75597143230906` |
| Python 输出总目录 | `/home/jinx/Faith/experiments/lagrangian_ppo_v1/runs/bg128-u32-generalization-20261119-001` |
| 首阶段 Python 输出 | `/home/jinx/Faith/experiments/lagrangian_ppo_v1/runs/bg128-u32-generalization-20261119-001/evaluation/nominal` |
| AnyLogic 物料事件总目录 | `/home/jinx/Faith/experiments/lagrangian_ppo_v1/runs/bg128-u32-generalization-20261119-001_materials` |
| 首阶段 AnyLogic 物料事件目录 | `/home/jinx/Faith/experiments/lagrangian_ppo_v1/runs/bg128-u32-generalization-20261119-001_materials/s01_BG128_eval_nominal` |

| 工况 | run_index | 配置字节 SHA256 |
| --- | --- | --- |
| 标称 | `9991001–9991100` | `fe76a53e60de6ea783d8afe906d29d7406d82a6b7f6022259e52bf215f9dd6c1` |
| 品位互换 | `9991101–9991200` | `027196ad7b80bec2a0645d9852f7f6413786df895cc23ebee00457d01ea7bd0c` |
| 入口限速 18 km/h | `9991201–9991300` | `9547e879d20c2a915fa53b24a637e3676b19bd167bcb407c614b120f9cd025aa` |
| 入口限速 15 km/h | `9991301–9991400` | `1150c22b5ef00c4bfb358b8ba8fef52437895b737036bcb6bd658a36bc6e7de6` |
| 均值变化 | `9991401–9991500` | `f90025ef477f092ab5c8f51b12caf359794847d93b4d4fae8cd5087d3527260b` |
| 波动增大 | `9991501–9991600` | `3233344fe751f07b58d1212ab0524bf8d4be01fafdc3993558b072cd91bdafc3` |

[600 班完整 run、环境与 Actor 采样 seed 清单](prepared_001/identities.csv) · [每组 recording、配置及目录](prepared_001/launch_identity.json) · [Java 外部工况参数表](prepared_001/scenarios.csv) · [离线预检](../../../../reports/token_attention_001/business_graph_summary_009/generalization_001/preparation_validation.json)

实际运行查看 Python 输出根目录下的 `run_status.json`，当前工况的 `evaluation/<工况>/run_status.json` 和 `launch/controller.log`。完成应为六组各 100 个 COMPLETE 班次、零训练和零梯度更新。每组用实际 RPC 对照冻结工况，最终按同槽双种子给出相对标称组的配对报告。
