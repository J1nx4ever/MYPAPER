# AnyLogic 手工接入 · Full/GAT1 s04、s05

以标准九车的 [DRL-PPO-baseline1.alp](../../../../../anylogic/3d/DRL-PPO-baseline1/DRL-PPO-baseline1.alp) 保存一个运行副本。在该副本的 `RL_Training` 中点一次 Run，按 `full-s04 → gat1-s04 → full-s05 → gat1-s05` 完成四个独立训练与固定 `u32` 的推理。每组 1,024 班训练、700 班评测，共 **6,896 班**。Python RPC 使用 `127.0.0.1:50051`，开始前确认其他使用该端口的实验已结束。

## 修改项及核对

| AnyLogic 位置 | 副本当前设置 | 本轮设置、作用和验证 |
| --- | --- | --- |
| Additional Class → `Baseline1CampaignPlan` → 类正文 | 既有单组或对照活动的 helper | 用本轮[组合 Java 类正文](prepared_001/Baseline1CampaignPlan.java)替换并保留编辑器管理的 package。它负责四组顺序、每班的身份与工况下发，以及 Python 启停。核对 `jobCount() == 6896` 和下表四个起始索引。 |
| `RL_Training` → Properties → Number of runs | 标准模型当前保存 200 班 | 设为 **6896**。Use Freeform Parameters 开启，Allow parallel evaluations 关闭，Replications 关闭，Stop at 设为 `28800` 秒。每个索引对应一班。 |
| `RL_Training` → Properties → Parameters → Freeform → 运行身份 | 由原 helper 提供 | `experimentRunIndex`、`experimentSeed`、`boundaryRecordingId`、`boundaryEventRoot`、`materialEventRoot` 分别调用新 helper 的 `runIndex((int) index)`、`environmentSeed((int) index)`、`recordingId((int) index)`、`eventRoot((int) index)`、`materialRoot((int) index)`。按[启动清单](prepared_001/launch_identity.json)核对。 |
| 同上 → 派车、工况和观测 | 标准九车物理观测 | `numTrucks=9`，`usePhysicalObservation=true`，`currentDispatchPolicy=Baseline1CampaignPlan.dispatchPolicy((int) index)` 返回 `3`，`freeSpeedFactorStd=Baseline1CampaignPlan.freeSpeedFactorStd((int) index)` 返回 `0.05`。保留速度截断 `0.85–1.10`、内生跟驰及动态速度。 |
| 同上 → 记录与诊断 | 基线比较可将物料记忆关闭 | `enableMaterialEventMemory=Baseline1CampaignPlan.materialEnabled((int) index)`，本实验为 `true`。`enableBoundaryEventRecording`、`offlineNativeCollection` 分别使用 helper 的 `eventRecordingEnabled`、`offlineEnabled`，本实验均为 `false`。核对 `obsDryRun`、`diagEvalEnabled`、`bcCollectEnabled`、`deterministicCheck` 均为 `false`。物料结算凭证供四窗审核，逐请求和全量观测等高频诊断保持关闭。 |
| `Main` → Variables → 派车连接 | 标准 Python RPC | 核对 `dispatchInferenceHost="127.0.0.1"`、`dispatchInferencePort=50051`、`enableOnnxDispatch=true`。 |
| `Main` → Agent actions → On startup → 首段 | 标准模型已安装配置握手 | 核对 `boundaryAcceptanceConfigSha256 = Baseline1CampaignPlan.awaitConfigSha256(experimentRunIndex, experimentSeed, boundaryRecordingId, boundaryEventRoot, boundaryInstrumentationRevision)`。它绑定本班配置字节 SHA256，实际 RPC 工况随后由 Python 核验。 |
| `Main` → Functions → `init1`（装载点初始化）及道路限速构造 | 原模型从 helper 读取工况 | 核对四个装载点继续按 `experimentRunIndex` 调用 `loadIsOre`、`loadGrade`、`loadGradeHalfRange`，道路继续调用 `roadSpeedLimit`。训练和独评都通过同一映射读取实际工况。 |

如已经打开减车或故障副本，优先回到上述九车模型。标准九车要求停运掩码为 `0`、无定时故障，且矿石目标 `16000` 吨、废石目标 `4000` 吨、废石停派线 `6000` 吨。减车副本的 `Simulation` 和 `RL_Training` 可能仍有对旧 helper 停运或目标接口的引用，应以标准九车模型作为本轮入口。

## 运行身份

| 组别 | AnyLogic index | 新物理 run_index | 完整身份与配置 |
| --- | ---: | ---: | --- |
| `full-s04` | 0–1723 | 10046001–10047724 | [启动清单](prepared_001/full-s04/launch_identity.json) · [逐班 run 与双种子](prepared_001/full-s04/identities.csv) |
| `gat1-s04` | 1724–3447 | 10047725–10049448 | [启动清单](prepared_001/gat1-s04/launch_identity.json) · [逐班 run 与双种子](prepared_001/gat1-s04/identities.csv) |
| `full-s05` | 3448–5171 | 10049449–10051172 | [启动清单](prepared_001/full-s05/launch_identity.json) · [逐班 run 与双种子](prepared_001/full-s05/identities.csv) |
| `gat1-s05` | 5172–6895 | 10051173–10052896 | [启动清单](prepared_001/gat1-s05/launch_identity.json) · [逐班 run 与双种子](prepared_001/gat1-s05/identities.csv) |

| 组别 | 首班 recording_id | 首班配置字节 SHA256 | 首班环境种子 / Actor 采样种子 |
| --- | --- | --- | --- |
| `full-s04` | `bg128-gat-depth-20261125-full-s04-s01_BG128_train_u0001` | `286db5f8e381ae3dec67f8d833ef6c6f30dbf97c3535fa62cbb4034ee9b59051` | `242214385619528` / `180066249680641` |
| `gat1-s04` | `bg128-gat-depth-20261125-gat1-s04-s01_BG128_train_u0001` | `7563d7b81d8cc476700f5920d6e1e0bb4f4c448ef134ca5a5fc192c8e7bf7907` | `221279399269009` / `132376109434437` |
| `full-s05` | `bg128-gat-depth-20261125-full-s05-s01_BG128_train_u0001` | `9bf9c018764a46a62a5aaac8657cf22a4593ae68e71d133a0aa9affad82a8e73` | `136019205914594` / `195291969824924` |
| `gat1-s05` | `bg128-gat-depth-20261125-gat1-s05-s01_BG128_train_u0001` | `1b50ea505298e557b1451b259114f890028c5c7f41e5a0144e12a5317eb20ae0` | `273002174717929` / `172289262760590` |

每组子清单给出首班与全部阶段的 `recording_id`、配置字节 SHA256、Python 输出目录、AnyLogic 事件目录。逐班表给出 `run_index` 和环境种子、Actor 采样种子。组别为上表的 `<组别>`，Python 输出根为项目的 `runs/bg128-gat-depth-20261125-<组别>/`，物料事件根为 `runs/bg128-gat-depth-20261125-<组别>_materials/`。复制 Java 前核对[组合启动清单](prepared_001/launch_identity.json)里的 helper SHA256。粘贴组合 helper 后，核对 `runIndex(0)`、`runIndex(1724)`、`runIndex(3448)` 和 `runIndex(5172)` 都与上表首班相符。

合并 Java 类的 SHA256 为 `c57a2bc6cb7c55cfed89bcd72dcab370c6449384a6df97b1b6203eda34300573`。四组首班的 `recording_id` 格式为 `bg128-gat-depth-20261125-<组别>-s01_BG128_train_u0001`，对应种子和首阶段配置摘要如下。

| 组别 | 首班环境 seed | 首班 Actor 采样 seed | 首阶段配置字节 SHA256 |
| --- | ---: | ---: | --- |
| `full-s04` | `242214385619528` | `180066249680641` | `286db5f8e381ae3dec67f8d833ef6c6f30dbf97c3535fa62cbb4034ee9b59051` |
| `gat1-s04` | `221279399269009` | `132376109434437` | `7563d7b81d8cc476700f5920d6e1e0bb4f4c448ef134ca5a5fc192c8e7bf7907` |
| `full-s05` | `136019205914594` | `195291969824924` | `9bf9c018764a46a62a5aaac8657cf22a4593ae68e71d133a0aa9affad82a8e73` |
| `gat1-s05` | `273002174717929` | `172289262760590` | `1b50ea505298e557b1451b259114f890028c5c7f41e5a0144e12a5317eb20ae0` |

## 预检、启动和验收

```bash
cd /home/jinx/Faith/experiments/lagrangian_ppo_v1
PLAN=change_plans/token_attention_001/business_graph_summary_009/gat_weak_ablation_001/seed_extension_001
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B "$PLAN/extension_run.py" --check
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B "$PLAN/extension_run.py" \
  --suite "$PLAN/prepared_001/full-s04/suite.json" --status
```

`--check` 返回 `OFFLINE_CHECKED` 后，在九车副本执行 **Save → Build → Run RL_Training** 一次。由 Java 启动 Python。查询其余三组时更换 `--suite` 对应目录，启动前应显示 `PREPARED_OFFLINE_ONLY`。每组首班在其 `launch/controller.log` 可见 `JAVA_START_VERIFIED`，run 依次为 `10046001`、`10047725`、`10049449`、`10051173`。下一组在前一组全部封存、释放端口后接续。若遇 STOP 或交接失败，保留控制器日志、已领取身份和回执，依据 `--status` 查明停在哪组。

完成时核对四个 `suite_complete.json` 均为 `COMPLETE`，各记录 `training_shifts=1024`、`training_updates=32`、`evaluation_shifts=700` 和 `evaluation_optimizer_updates=0`。推理阶段的品位越界或矿废欠产属于结果数据。配对报告检查每个新 `u32` 仍使用原七工况的同一套 700 槽。五个训练日程的最终对比写到 `reports/token_attention_001/business_graph_summary_009/gat_weak_ablation_001/seed_extension_001/paired_evaluation_summary.json`。
