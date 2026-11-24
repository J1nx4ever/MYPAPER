# AnyLogic 手工接入 · GAT1 s02 和 s03

同一个标准九车、支持 1,519D 物理观测和物料记忆的 [DRL-PPO-baseline1.alp](../../../../../anylogic/3d/DRL-PPO-baseline1/DRL-PPO-baseline1.alp) 运行副本，点一次 Run 依次完成 `gat1-s02` 和 `gat1-s03`。每组 1,024 班训练、700 班固定评测，共 3,448 班。使用 `127.0.0.1:50051`，在同端口其他活动完成后启动。

## 修改位置、原因和核对

| 精确位置 | 修改前 | 本轮设置与验证 |
| --- | --- | --- |
| Additional Class → `Baseline1CampaignPlan` → 类正文 | 流量匹配五班 helper，`jobCount() == 5` | 用[本轮合并 Java 类正文](prepared_001/Baseline1CampaignPlan.java)替换，保留编辑器的 package。新类逐班发放新身份、工况和配置绑定。核对 `jobCount() == 3448`、`runIndex(0) == 10038001`、`runIndex(1724) == 10040001`。 |
| `RL_Training` → Properties → Number of runs | 流量匹配副本设为 `5` | 改为 `3448`，Freeform、串行、关闭 Replications、结束时间为 `28800 s`。 |
| `RL_Training` → Properties → Parameters → Freeform → `enableMaterialEventMemory` | 流量匹配 helper 返回 `false` | 使用 `Baseline1CampaignPlan.materialEnabled((int) index)`。本轮返回 `true`，用于固定四窗结算和 FIFO 准入。实际 START 必须收到物料记忆开启。 |

合并 helper 字节 SHA256 为 `a5e0d7f6b50cd08c51660b2dd8565cae1eace0810093de445f2d823893467171`。

| 界面位置 | 需保留的设置或表达式 |
| --- | --- |
| `RL_Training` → Properties → Parameters → Freeform | `numTrucks=9`，`usePhysicalObservation=true`，`currentDispatchPolicy=Baseline1CampaignPlan.dispatchPolicy((int) index)`（返回 3）。 |
| 同上 → `experimentRunIndex`、`experimentSeed` | `Baseline1CampaignPlan.runIndex((int) index)`、`Baseline1CampaignPlan.environmentSeed((int) index)`。 |
| 同上 → `boundaryRecordingId`、`boundaryEventRoot`、`materialEventRoot` | `Baseline1CampaignPlan.recordingId((int) index)`、`Baseline1CampaignPlan.eventRoot((int) index)`、`Baseline1CampaignPlan.materialRoot((int) index)`。 |
| 同上 → `enableBoundaryEventRecording`、`offlineNativeCollection`、`freeSpeedFactorStd` | 分别用 helper 的 `eventRecordingEnabled((int) index)`、`offlineEnabled((int) index)`、`freeSpeedFactorStd((int) index)`，对应 `false`、`false`、`0.05`。 |
| `Main` → Variables | `dispatchInferenceHost="127.0.0.1"`，`dispatchInferencePort=50051`，`enableOnnxDispatch=true`。 |
| `Main` → On startup | 首段用 `boundaryAcceptanceConfigSha256 = Baseline1CampaignPlan.awaitConfigSha256(...)` 将实际配置 SHA 绑定到首次 RPC。 |

## 封存运行身份

| 组别 | AnyLogic index | run_index | 首班环境 seed / Actor 采样 seed | 清单 |
| --- | ---: | ---: | --- | --- |
| `gat1-s02` | 0–1723 | 10038001–10039724 | `12999697798769` / `233569925063516` | [阶段身份与配置](prepared_001/gat1-s02/launch_identity.json) · [逐班 run 与 seed](prepared_001/gat1-s02/identities.csv) |
| `gat1-s03` | 1724–3447 | 10040001–10041724 | `35262946706242` / `273736645455026` | [阶段身份与配置](prepared_001/gat1-s03/launch_identity.json) · [逐班 run 与 seed](prepared_001/gat1-s03/identities.csv) |

| 组别 | 首班 recording_id | 首班配置字节 SHA256 |
| --- | --- | --- |
| `gat1-s02` | `bg128-gat1-replication-20261123-s02-s01_BG128_train_u0001` | `119f94d00efbec295edea3b8327a44ebc109074797a63ab939e15e5928fb6f93` |
| `gat1-s03` | `bg128-gat1-replication-20261123-s03-s01_BG128_train_u0001` | `6fd7400fc44b13c376adee74c8c646e6db79e4691bf7b116848cd4dba43bba86` |

Python 输出根分别为项目的 `runs/bg128-gat1-replication-20261123-s02/` 和 `runs/bg128-gat1-replication-20261123-s03/`。AnyLogic 物料事件根分别为 `runs/bg128-gat1-replication-20261123-s02_materials/` 和 `runs/bg128-gat1-replication-20261123-s03_materials/`。所有阶段的完整目录和配置 SHA 见[合并启动清单](prepared_001/launch_identity.json)及其子清单。

## 启动与验收

```bash
cd /home/jinx/Faith/experiments/lagrangian_ppo_v1
PLAN=change_plans/token_attention_001/business_graph_summary_009/gat_weak_ablation_001/seed_replication_001
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B "$PLAN/replication_run.py" --check
```

看到 `OFFLINE_CHECKED` 后，在 AnyLogic **Save → Build → Run RL_Training**。首班由 Java 拉起 Python，`s02` 第一班应出现 `JAVA_START_VERIFIED run_index=10038001`。该组完整封存并释放端口后，`s03` 第一班应出现 `JAVA_START_VERIFIED run_index=10040001`。日志位于两组 Python 输出根下的 `launch/controller.log`。

可用同入口加 `--suite "$PLAN/prepared_001/gat1-s02/suite.json" --status` 查询进度。第二组用 `gat1-s03`。两组完成后，`reports/token_attention_001/business_graph_summary_009/gat_weak_ablation_001/seed_replication_001/paired_evaluation_summary.json` 汇总三种子的配对指标。若出现 STOP，保留回执和账本供审查。
