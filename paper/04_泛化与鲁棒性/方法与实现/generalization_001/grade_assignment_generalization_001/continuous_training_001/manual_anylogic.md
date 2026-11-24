# 连续混合品位训练手工接入清单

在当前 BG128 泛化专用模型副本上操作。已核对的本地参考是 [DRL-PPO-baseline1.alp](../../../../../../anylogic/3d/DRL-PPO-baseline1/DRL-PPO-baseline1.alp)。启动前关闭旧活动，核对本活动的[运行身份](prepared_001/launch_identity.json)和[完整 run／双种子清单](prepared_001/identities.csv)。

| AnyLogic 位置 | 修改前 | 修改后、理由与验证 |
|---|---|---|
| Additional Class → `Baseline1CampaignPlan` → 全部 Java 源码 | 上一轮泛化或续跑 helper | 全文替换为[本活动冻结 helper](prepared_001/Baseline1CampaignPlan.java)。它绑定本活动场景表、master 和 Python 入口。构建后 `jobCount()` 为 2,424，首班 `runIndex(0)` 为 10000001，末班为 10002424。 |
| Main → Functions → `init1` → 四个 `LoadSite` 初始化块 | [旧四块](Main.init1.LoadSites.before.java)，s0–s2 是固定矿站，s3 是固定废石站 | 用[新四块](Main.init1.LoadSites.after.java)完整替换，从 `LoadSite s0 = add_loadSites();` 到 `s3.initLoadSite(...);`。保留上方 `dynamic = true;` 和下方 `for (LoadSite s : loadSites)` 循环。四站在 `initLoadSite` 生成品位块前按 `experimentRunIndex` 设置矿废类型、均值和半区间。验证仍有四个 `add_loadSites()`、固定站号 0–3 和对应的物理 `loadNode`。 |
| `RL_Training → Properties → Number of runs` | 参考副本为 499 | 改为 **2424**，保持串行、关闭试验重复。验证实验 Freeform 中 dispatcher、run index、环境种子、recording ID 和 material root 仍由 `Baseline1CampaignPlan` 按 `(int) index` 提供。 |
| `RL_Training → Parameters → Freeform` 和 `Main → On startup` | 原模式 3 的 helper 映射和 `awaitConfigSha256(...)` | 继续沿用现有表达式。核对 `deterministicCheck=false`，原启动握手使用本 helper 的 ready。当前 `Main → Functions → addBidirectionalPath` 已调用 `roadSpeedLimit(...)`，新 helper 提供同名接口并按冻结的标称限速核验。 |

**Save → Build → Run 一次。** 首班由 Java 以冻结 run、环境种子和 recording ID 启动 Python。后续训练和独评依次执行。RPC 会逐班验证 1519D schema、四站矿废类型和品位、实际吨位守恒、废石停派及物料来源。图构建与 C128 从该班实际观测确定矿石、废石和兼容卸点。矿废误配会在更新准入前报错。

首次启动前给本模型的关键身份以[启动身份文件](prepared_001/launch_identity.json)为准，包含首班 recording ID、配置字节 SHA256、Python 输出、AnyLogic 事件目录、run_index 和双种子。阶段级配置和逐班身份由同一冻结套件生成。

保存后的本地副本已核对为上表修改后状态，`enableMaterialAudit` 已开启。实跑时核对 `[MATERIAL_AUDIT_PASS]`、新废石站实际交付，以及 Load3 担任矿石站班次的入仓和加工记录。[计算审计和实跑验收依据](../../../../../../reports/token_attention_001/business_graph_summary_009/generalization_001/continuous_training_001/anylogic_calculation_audit.md)
