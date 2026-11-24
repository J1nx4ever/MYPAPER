# 原始 baseline ALP 的百班配对接入核对

原始依据为实验目录下的 `baseline/DRL-PPO.alp`，SHA256 为 `4406a8094ea796d6c10231aafbf8354dd2ad9bc2474e85efa9c8dca869271330`。已通过配对闭环的代码参考为 `anylogic/3d/DRL-PPO-baseline1/DRL-PPO-baseline1.alp`，审阅时 SHA256 为 `79dc4f8591c0e1a4d5e2d1ca02aa90bd442e1e334528fb799682b6e775330c04`。两文件的 Main、Truck、LoadSite 和 DumpSite 的 Presentation XML 字节一致，Main 的 `init_network`、`obsInit` 和 `buildPhysicalObservation` 函数体一致。以下通过工作副本迁移运行合同，保留原件的路网和工艺布局。

## 原件与既有百班合同的差异

| 项目 | 原件 | 既有 100 槽 START / END 合同 |
|---|---|---|
| 启动与身份 | 55 个 Freeform 班，run 为 `(long) index + 1L`，seed 为 `202610280401L + (long) index` | 本批 helper 一次认领一个新物理班，校验配置字节 SHA256，run、recording 和两个输出目录逐班独立 |
| 奖励与废石掩码 | `dispatch-reward-v1.0`。已承诺废石达 4,000 t 即停派，最多 4,193 t | `dispatch-production-wait-shortfall-v2`，`rewardWaitKappa=1.0/25000.0`。业务目标 4,000 t，已承诺 6,000 t 停派，最多 6,193 t |
| RPC 校验 | 1519 维观测存在。schema 字节仅在旧可选诊断分支输出，缺少 `config.runtime_scene.*` | 每次请求核验观测字段顺序和实际工况配置，START 核对历史槽的环境 seed、首帧、车号和合法动作 |
| 等待与记录 | 原件只有累计车秒，`RL_Training` 的 `diagEvalEnabled=true`，Main 的 `obsDryRun=true` | END 提供装载、卸载资源、仓容准入三段等待与对账残差。高频记录、诊断和干跑关闭 |
| 资源 | 原件目录缺少 ALP 引用的同目录 JAR、图片及相对 `3d/mine_truck.dae` | 在带全相对依赖的实验工作目录中 Build，确保 Java 使用项目冻结的 RPC 库 |

废石停派差异直接改变合法动作集合。在原件上只粘贴启动 helper 会在 START 缺少运行工况和 schema 契约字段。复制历史配置值到 Python 无法修正实际掩码或提供真实分段等待。

## 工作副本手工迁移

1. 在 AnyLogic 打开 `baseline/DRL-PPO.alp`，Save As 为新名称。工作副本可放在 `anylogic/3d/DRL-PPO-baseline1/`，该目录的相对 JAR 和呈现资源已核对；也可放在已配置相同依赖的独立工作目录。以下“代码参考”指本页首段的文件。按同名对象从代码参考读取 Java 正文和属性，在工作副本逐项粘贴。原件、既有 100 槽实验用工作模型及其回执均保留。
2. 在 **Additional Class** 新建 `Baseline1CampaignPlan`，粘贴[本轮 Java helper](prepared_001/Baseline1CampaignPlan.java)，保留 AnyLogic 生成的 package。按代码参考新建 `WaitBreakdownLedger`、`AcceptanceEventRecorder`、`MaterialEventRecorder` 三个类，分别使用同名类正文。后两个类供兼容函数编译，百班运行时对应记录开关为 false。现有 `RLClient` 的代码一致。
3. 在 **Main → Variables** 修改 `rewardContractId` 初始值为 `"dispatch-production-wait-shortfall-v2"`，将 `obsDryRun` 设为 false。新增普通变量 `rewardWaitKappa`（double，`1.0/25000.0`）、`waitBreakdownLedger`（WaitBreakdownLedger，null）、`experimentVerboseTrace`（boolean，false）、`boundaryEventRecorder`（AcceptanceEventRecorder，null）、`materialEventRecorder`（MaterialEventRecorder，null）、`offlineDecisionObservation`（Observation，null）。按代码参考的同名元素补齐 Main 参数。

   | 参数 | 类型与初值 |
   |---|---|
   | `wasteDispatchStopTonsShift` | double，`6000.0` |
   | `boundaryRecordingId`、`boundaryEventRoot`、`boundaryAcceptanceConfigSha256`、`materialEventRoot`、`offlineBehaviorSha256` | String，空字符串即可，运行身份由本轮 Freeform 配置绑定 |
   | `boundaryInstrumentationRevision` | String，`"boundary-events-v1-native-rpc-runtime-scene-v1"` |
   | `enableBoundaryEventRecording`、`enableMaterialEventMemory`、`offlineNativeCollection`、`acceptanceLongLoadEnabled` | boolean，false |
   | `acceptanceBoundaryMode`、`acceptanceLoadNotBeforeSec` | int，`0`。double，`14500.0` |

4. 在 **Main → On startup / On destroy** 用代码参考的相应正文替换原内容。启动代码首条语句为 `boundaryAcceptanceConfigSha256 = Baseline1CampaignPlan.awaitConfigSha256(experimentRunIndex, experimentSeed, boundaryRecordingId, boundaryEventRoot, boundaryInstrumentationRevision)`。该调用等待控制器并发布独占 Java claim。启动正文另外初始化等待台账，在 `init1()` 后注册车辆，保持原有随机种子初始化和车辆激活顺序。
5. 在 **Main → Functions** 将现有的 `getWasteTaskUpperTons`、`buildRewardConfigComponents`、`selectNextLoadSiteAI`、`selectNextLoadSite`、`selectNextLoadSitePlan`、`finishEpisodeAndNotify`、`settleCurrent2hWindow`、`printMaterialAuditSummary`、`init1`、`addBidirectionalPath` 的正文替换为代码参考同名正文。新增函数如下，并复制同名正文。

   | 新函数 | 返回类型 | 参数 |
   |---|---|---|
   | `checkedWaitIncrement` | double | `double cumulativeWait` |
   | `calculateProductionWaitReward` | double | `double deltaWait` |
   | `exchangeDispatch` | int | `Truck t, Observation obs, int proposedAction` |
   | `nativeDecisionMetadata` | `java.util.Map<String, Double>` | `int proposedAction` |
   | `buildRuntimeScenarioComponents` | `java.util.Map<String, Double>` | 无 |
   | `buildWaitBreakdownTerminalComponents` | `java.util.Map<String, Double>` | 无 |

6. 在 **LoadSite → Functions** 更新 `getDispatchableTons`、`reserveLoad`、`loadServiceMinutes`，新增 `double loadServiceBoundaryMinutes(Truck t)`。在流程块属性用代码参考的同名字段覆盖 `enterLoad.On enter`、`seize.On enter`、`delay.On enter`、`exitLoad.On exit`，并更新 `delay.Delay time`。其中 `getDispatchableTons` 按 `main.wasteDispatchStopTonsShift` 检查已承诺废石，等待阶段由真实块回调记账。
7. 在 **DumpSite → Functions** 更新 `advanceCrusherTo`、`completeUnloading`、`receiveCrusherBatch`，新增 `double dumpServiceBoundaryMinutes(Truck t)`。在流程块属性覆盖 `enterDump.On enter`、`waitCrusherCapacity.On enter`、`delayDump.On enter`、`exitDump.On exit` 和 `delayDump.Delay time`。保留原 FIFO 处理顺序和队首整车准入，分别统计卸载位排队与仓容等待。正常班由代码参考按普通服务时长计算。
8. 在 **Truck → Variables** 新增两个 boolean `acceptanceBoundaryLoadUsed`、`acceptanceBoundaryDumpUsed`，默认 false。复制代码参考的 `initTruck`、`onLoadingFinished`、`onDumpingFinished` 正文与 `event` 动作。代码参考用 `main.experimentVerboseTrace` 包裹其余逐车 `traceln`，如工作副本还有未受开关控制的相同语句，按代码参考对应状态或转移的动作补上该开关。做算力对比时以 Python 策略时间及 RPC 响应时间为口径，并保留实际遥测。

## RL_Training 与首次验证

- **Properties → Number of runs** 从 55 改为 200，保留 Freeform、串行、无 Replications，Stop at 28,800 s。
- **Properties → Parameters → Freeform** 将 `experimentRunIndex`、`experimentSeed`、`boundaryRecordingId`、`boundaryEventRoot` 设为 `Baseline1CampaignPlan.runIndex((int) index)`、`environmentSeed((int) index)`、`recordingId((int) index)`、`eventRoot((int) index)`。后三个缩写均通过 `Baseline1CampaignPlan` 调用。`boundaryAcceptanceConfigSha256` 可保留空字符串，实际值由 Main 启动握手回填。
- `currentDispatchPolicy=Baseline1CampaignPlan.dispatchPolicy((int) index)`、`freeSpeedFactorStd=Baseline1CampaignPlan.freeSpeedFactorStd((int) index)`，标称返回 3 和 0.05。启用 `usePhysicalObservation=true`、`enableOnnxDispatch=true`，`numTrucks=9`，RPC `127.0.0.1:50051`。
- Freeform 中将旧 `diagEvalEnabled=true` 改为 false，删除旧 BC 输出路径表达式并保持 `bcCollectEnabled=false`。三个记录参数 `enableBoundaryEventRecording`、`enableMaterialEventMemory`、`offlineNativeCollection` 分别调用 helper 的 `eventRecordingEnabled`、`materialEnabled`、`offlineEnabled`，均应返回 false。核对 `enableP0SnapshotAudit=false`、`acceptanceBoundaryMode=0`、`acceptanceLongLoadEnabled=false`、`deterministicCheck=false`。
- 先 Build，再按本活动启动说明预检及启动。Python 在第一班发出动作前确认 RPC 字段和实际标准工况与历史同槽一致。验收重点为废石停止派车 6,000 t、承诺上限 6,193 t、四窗 7,200 s、总班长 28,800 s、3/2 个卸位、500 t 破碎缓冲、2,500 t/h 破碎率，以及真实首帧 SHA 和蒙版。END 要求真实的三段等待且与总车秒对账。接入状态以实际 RPC 与账本判定。
