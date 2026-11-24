# AnyLogic 手工接入 · BG128 两组 s01 弱消融

一次 Run 串行执行第二层 `Dump → Load` 定向屏蔽，再执行一层 GAT 加逐节点 MLP。每组 1,724 班，两组合计 3,448 班。使用支持物料记忆和 1,519D 观测的 [DRL-PPO-baseline1.alp](../../../../anylogic/3d/DRL-PPO-baseline1/DRL-PPO-baseline1.alp) 运行副本。接入前让已有活动结束并释放 `127.0.0.1:50051`。

## 修改位置和验收值

| AnyLogic 精确位置 | 修改前 | 本批修改与验证 |
| --- | --- | --- |
| 项目树 → Additional Class → `Baseline1CampaignPlan` → 类正文 | 最近五组基线复验的 helper，`jobCount() == 5` | 替换为[本批合并 Java helper](prepared_001/Baseline1CampaignPlan.java)，保留 AnyLogic 生成的 package。它负责新活动的工况、身份及 Python 启动。核对 `jobCount() == 3448`、`runIndex(0) == 10027001`、`runIndex(1724) == 10029001`。 |
| `RL_Training` → Properties → Number of runs | 沿用最近的基线复验副本时为 `5` | 改为 `3448`。使用 Freeform、串行、关闭 Replications，班末时间 `28800 s`。 |
| `RL_Training` → Properties → Parameters → Freeform → `enableMaterialEventMemory` | 最近五组基线 helper 的 `materialEnabled((int) index)` 返回 `false`，部分副本也可能写死为 `false` | 使用 `Baseline1CampaignPlan.materialEnabled((int) index)`，本批每班返回 `true`。这是 FIFO 结算和物料准入所需的物理记录。RPC START 会检查物料记忆为 1。 |

合并 helper 的文件 SHA256 为 `5a5c3a3ede88c6b5d75e030f46a479e21f45ee842c49177d603ec77142f6083d`。本批只粘贴合并文件。若运行副本已有 Freeform 覆盖，按下表恢复。

| 位置 | 字段 | 本批表达式或值 |
| --- | --- | --- |
| `RL_Training` → Properties → Parameters → Freeform | `currentDispatchPolicy`、`usePhysicalObservation` | `Baseline1CampaignPlan.dispatchPolicy((int) index)` 返回 `3`，`true` |
| 同上 | `experimentRunIndex`、`experimentSeed` | `Baseline1CampaignPlan.runIndex((int) index)`、`Baseline1CampaignPlan.environmentSeed((int) index)` |
| 同上 | `boundaryRecordingId`、`boundaryEventRoot`、`materialEventRoot` | `Baseline1CampaignPlan.recordingId((int) index)`、`Baseline1CampaignPlan.eventRoot((int) index)`、`Baseline1CampaignPlan.materialRoot((int) index)` |
| 同上 | `enableBoundaryEventRecording`、`offlineNativeCollection` | `Baseline1CampaignPlan.eventRecordingEnabled((int) index)`、`Baseline1CampaignPlan.offlineEnabled((int) index)`，均返回 `false` |
| 同上 | `freeSpeedFactorStd` | `Baseline1CampaignPlan.freeSpeedFactorStd((int) index)` 返回 `0.05` |
| `Main` → Variables | `dispatchInferenceHost`、`dispatchInferencePort`、`enableOnnxDispatch` | `"127.0.0.1"`、`50051`、`true`，通过已有 gRPC 派车路径服务 |
| `Main` → On startup | `boundaryAcceptanceConfigSha256` | 保留首段通过 `Baseline1CampaignPlan.awaitConfigSha256(...)` 绑定配置的握手。 |

## 两组独立运行身份

| 组别 | AnyLogic index | run_index | 首班环境 seed / Actor 采样 seed | 完整清单 |
| --- | ---: | ---: | --- | --- |
| `nodump-s01` | 0–1723 | 10027001–10028724 | `184753289302782` / `204257354822860` | [阶段身份与配置](prepared_001/nodump-s01/launch_identity.json) · [逐班 run/seed](prepared_001/nodump-s01/identities.csv) |
| `gat1-s01` | 1724–3447 | 10029001–10030724 | `29622541569038` / `230600439570088` | [阶段身份与配置](prepared_001/gat1-s01/launch_identity.json) · [逐班 run/seed](prepared_001/gat1-s01/identities.csv) |

| 组别 | 首班 recording_id | 首班配置字节 SHA256 |
| --- | --- | --- |
| `nodump-s01` | `bg128-gat-weak-20261122-nodump-s01-s01_BG128_train_u0001` | `1271324a8edaa993f40a331c548afce7404e7b4be69110fd77ce855e3496dfda` |
| `gat1-s01` | `bg128-gat-weak-20261122-gat1-s01-s01_BG128_train_u0001` | `37c4071542a969758d728507c420640b2a13fba3c1b910688f354056d4cf86d8` |

Python 输出根位于项目 `runs/bg128-gat-weak-20261122-nodump-s01/` 和 `runs/bg128-gat-weak-20261122-gat1-s01/`。AnyLogic 物料事件根分别为 `runs/bg128-gat-weak-20261122-nodump-s01_materials/` 和 `runs/bg128-gat-weak-20261122-gat1-s01_materials/`。逐阶段配置和目录列在[合并启动清单](prepared_001/launch_identity.json)及其两个子清单。

## 启动与检查

```bash
cd /home/jinx/Faith/experiments/lagrangian_ppo_v1
PLAN=change_plans/token_attention_001/business_graph_summary_009/gat_weak_ablation_001
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B "$PLAN/weak_run.py" --check
```

看到 `OFFLINE_CHECKED` 后，在 AnyLogic 执行 **Save → Build → Run RL_Training**。Python 从 AnyLogic 首班发起。第一组日志应出现 `JAVA_START_VERIFIED run_index=10027001`；第一组封存且释放端口后，第 1,725 班应出现 `JAVA_START_VERIFIED run_index=10029001`。各组日志位于各自 Python 输出根下的 `launch/controller.log`。要查询进度，运行 `weak_run.py --suite "$PLAN/prepared_001/nodump-s01/suite.json" --status`，查第二组时把组名改为 `gat1-s01`。

两组均完成后，`reports/token_attention_001/business_graph_summary_009/gat_weak_ablation_001/` 将包含 `paired_evaluation_summary.json` 和 `fixed_grade_response_replay.json`。若需要单独复核固定状态报告，可执行 `weak_run.py --grade-replay`。出现 STOP 时保留回执和账本，已领取物理身份按历史规则审查。
