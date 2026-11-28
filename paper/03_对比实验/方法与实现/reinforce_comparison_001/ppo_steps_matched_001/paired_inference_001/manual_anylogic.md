# S02 步数对齐模型的独立评测副本

评测启动使用 `BG128-REINFORCE-S02-MATCHED-EVAL`。评测的七个工况各执行 100 班，共 700 班。训练模型是副本的来源。[评测模型](../../../../../../anylogic/3d/BG128-REINFORCE-S02-MATCHED-EVAL/BG128-REINFORCE-S02-MATCHED-EVAL.alp) · [模型与资源核验](../../../../../../reports/token_attention_001/business_graph_summary_009/reinforce_comparison_001/ppo_steps_matched_001/paired_inference_001/model_copy_s02_matched_eval.json)

需要手动重建时，从 AnyLogic 打开已完成的 S02 步数对齐训练模型，另存为上述评测模型，并把相邻资源文件按原相对路径复制到新目录。然后检查以下位置。[训练模型](../../../../../../anylogic/3d/BG128-REINFORCE-S02-MATCHED/BG128-REINFORCE-S02-MATCHED.alp)

| AnyLogic 位置 | 训练模型的值 | 评测副本的值和验证 |
| --- | --- | --- |
| 项目树 → 顶层模型名称 | `BG128-REINFORCE-S02-MATCHED` | `BG128-REINFORCE-S02-MATCHED-EVAL`。窗口及模型路径对应新副本。 |
| 项目树 → Additional classes → `Baseline1CampaignPlan` → Java 正文 | S02 训练 helper，run `10094001–10095024` | 换成本评测冻结的 Java 正文，使 READY、逐班种子和物料目录使用评测身份。Java 包为 `fusion`。首末索引 `0` 和 `699` 分别绑定 run `10096001` 和 `10096700`。[评测 Java](prepared_001/Baseline1CampaignPlan.java) |
| 项目树 → `RL_Training` → Number of runs | `1024` | `700`。Use Freeform Parameters 开启，Allow parallel evaluations 关闭，Use replications 关闭。每个物理 run 恰好有一个索引。 |
| Main → Variables → `dispatchInferenceHost`、`dispatchInferencePort` | `127.0.0.1`、`50051` | 与评测控制器一致。启动回调从匹配的 READY 校验配置 SHA256，并写出 Java 独占 claim。 |

保存并按 F7 构建后，官方命令行入口可以执行这个副本的 `Run RL_Training`。训练最终 Actor 已封存、旧控制器已退出、评测控制器已发布首班 READY 后，自动交接执行一次该命令。

```bash
/home/jinx/apps/AnyLoigc/anylogic/anylogic -r \
  /home/jinx/Faith/experiments/lagrangian_ppo_v1/anylogic/3d/BG128-REINFORCE-S02-MATCHED-EVAL/BG128-REINFORCE-S02-MATCHED-EVAL.alp RL_Training
```

首班的 Java claim、RPC 物料回执和评测账本是接管验证依据。由同一进程完成运行交接，避免再次触发同一评测实验。[启动身份与配置](prepared_001/launch_identity.json) · [逐班 run 和双种子](prepared_001/identities.csv)
