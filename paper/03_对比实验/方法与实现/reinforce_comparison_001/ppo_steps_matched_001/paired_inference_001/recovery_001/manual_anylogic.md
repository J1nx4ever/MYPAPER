# S02 固定策略评测的 AnyLogic 接续

现有评测在第 43 班开始后停止推进。新活动用冻结的 S02/u32 Actor 补齐剩余 658 个配对槽，合并前 42 个完整班。西采场模型和原评测模型各保留在原目录，新活动使用独立评测副本和独立 AnyLogic 工作区。

## 副本中的修改

复制整个 `anylogic/3d/BG128-REINFORCE-S02-MATCHED-EVAL` 目录，在副本中保留原来的相对资源结构。将副本目录、`.alp` 文件及模型显示名设为 `BG128-REINFORCE-S02-MATCHED-EVAL-RECOVERY`，Java package 沿用 `fusion`。[接续模型](../../../../../../../anylogic/3d/BG128-REINFORCE-S02-MATCHED-EVAL-RECOVERY/BG128-REINFORCE-S02-MATCHED-EVAL-RECOVERY.alp)

| AnyLogic 界面位置 | 接续前 | 接续后 | 作用与验证 |
| --- | --- | --- | --- |
| 模型树 → Additional classes → `Baseline1CampaignPlan` | 原始评测 Java 启动计划 | 将类正文替换为 [接续 Java 类](prepared_001/Baseline1CampaignPlan.java) | Freeform 参数按接续顺序读取独立身份。先离线核验首末身份、各阶段和 658 行。 |
| 模型树 → `RL_Training` → Properties → Number of runs | 700 | 658 | 从接续计划的零号索引开始。Use Freeform Parameters 开启，Allow parallel evaluations 和 Use replications 关闭。 |
| `Main` → Variables → `dispatchInferenceHost` / `dispatchInferencePort` | `127.0.0.1` / `50051` | `127.0.0.1` / `50051` | 用该地址连接接续 Python 控制器。启动回调校验 READY 的配置字节，随后发送本班的 START RPC。 |

## 启动与验收

核对[接续启动清单](prepared_001/launch_identity.json)中的配置 SHA256、run、recording、双种子及 Python 和物料事件路径。确认旧评测控制器停止、端口已释放，并运行接续方案的 Python 只读预检。

打开独立评测副本，Build 后只运行一次 `RL_Training`。先完成西采场编辑器中的未保存工作，再从已激活的 AnyLogic 实例加载接续副本。若已通过 AnyLogic 激活向导获得可用的独立工作区，可以用以下命令避免复用西采场实例。

```bash
LANG=zh_CN.UTF-8 LC_ALL=zh_CN.UTF-8 /home/jinx/apps/AnyLoigc/anylogic/anylogic \
  -data "$RECOVERY_WORKSPACE" -nl zh_CN -r "$RECOVERY_ALP" RL_Training
```

空白工作区已实测取得独立锁，使用有效的中文 locale 会进入 AnyLogic 激活向导。现有桌面许可状态随现有工作区保留，空白探针未完成激活。[隔离启动检查](../../../../../../../reports/token_attention_001/business_graph_summary_009/reinforce_comparison_001/ppo_steps_matched_001/paired_inference_001/recovery_001/isolated_workspace_probe.json)

以评测所用工作区日志的 Instance location、接续控制目录的首班唯一 Java 领取和完整物料回执核对运行。推理活动的优化器步数为零。
