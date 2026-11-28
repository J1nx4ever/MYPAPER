# S02/u32 七工况推理的 AnyLogic 副本

运行[独评模型](../../../../../anylogic/3d/BG128-REINFORCE-S02-U32-EVAL/BG128-REINFORCE-S02-U32-EVAL.alp)的 `RL_Training`。这是现有参数变化实验的名称。推理控制器向它提供封存的 S02 REINFORCE u32 策略，在七个工况下各执行 100 个完整班。AnyLogic 每班等候 Python 的 READY，核对该班配置 SHA256，发送 START RPC 并记录物料流水。

| AnyLogic 界面位置 | S02 训练模型 | 独立评测副本 | 理由与验证 |
| --- | --- | --- | --- |
| 项目树 → 模型 → Name | `BG128-REINFORCE-S02` | `BG128-REINFORCE-S02-U32-EVAL` | 在单独目录生成评测工程，Java 包仍为 `fusion`。 |
| 项目树 → Additional classes → `Baseline1CampaignPlan` → Java 源码 | 训练活动的 1,024 行清单与物料目录 | 粘贴[推理活动 Java 类](prepared_001/Baseline1CampaignPlan.java)全文 | Freeform 使用当班的 run、环境种子、recording、物料目录、四个装载点和道路速度。Build 后验证 `runIndex(0)` 为 10092001、`runIndex(699)` 为 10092700。 |
| 项目树 → `RL_Training`（参数变化）→ Number of runs | 1024 | 700 | 每个工况恰好 100 班。Use Freeform Parameters 勾选，Allow parallel evaluations 与 Use replications 关闭。 |

副本随机械模型附带 23 个 JAR、一张 PNG 与卡车 DAE。启动前核对[各阶段配置与目录](prepared_001/launch_identity.json)和[逐班双种子](prepared_001/identities.csv)。启动后用首班 Java 领取凭证及完整物料回执核对双向通信。[副本与资源核验证据](../../../../../reports/token_attention_001/business_graph_summary_009/reinforce_comparison_001/paired_inference_001/model_copy_s02_u32_eval.json)
