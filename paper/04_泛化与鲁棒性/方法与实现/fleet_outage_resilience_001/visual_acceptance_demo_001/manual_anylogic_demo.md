# AnyLogic 单班退场演示

本包用冻结的动态分支 `u9` 演示一班。工况沿用历史独立测试 `10043852` 的开班名单、物料布局和停派时刻，使用本次封存的独立环境种子与 Actor 采样种子。目标是从现场动画和班末回执核对退场流程，实际退场时刻和产量以演示本班为准。[本班身份与封存配置](prepared_001/run_seed.csv)

## 一、另存演示模型

1. 在 AnyLogic 打开已经接入 `FleetResilienceState` 的 `DRL-PPO-baseline2.alp`。用 **File → Save As…** 在独立目录另存为 `DRL-PPO-baseline2-visual-demo.alp`。检查副本的 `3d/`、`image2.png`、数据库和 JAR 依赖都能在模型资源中解析。如果要从文件管理器复制，应在源模型关闭后复制整个模型目录，使相对路径保持有效。[AnyLogic 另存为](https://www.anylogic.help/anylogic/ui/saving-models.html) · [相对资源路径](https://www.anylogic.help/anylogic/ui/relative-filepaths.html)
2. 在**演示副本**的项目树进入 **Java classes → Baseline1CampaignPlan**，用本目录 [单班版 Java 类](prepared_001/Baseline1CampaignPlan.java)替换现有类声明，保留 IDE 管理的 `fusion` 包设置。确认新主表只有一行，读取索引为 `0`，活动为 `bg128-fleet-outage-visual-demo-20261124-001`。旧的 `Main → On startup` 首次握手应保留，启动桥会从它自动启动。
3. 演示入口是副本的 **Simulation**。它和正式批量实验共用 Main 的引擎及派车程序，使用此处接入的一班身份。

## 二、配置 Simulation

选择项目树中的 **Simulation → Properties → Parameters**，逐行填写下表右列的 Java 表达式，同名参数的默认复选框应取消。手册的原值取自已保存的 `baseline2`。

| 参数 | 默认或原历史值 | 演示设置 |
| --- | --- | --- |
| `currentDispatchPolicy` | `4` | `Baseline1CampaignPlan.dispatchPolicy(0)` |
| `experimentRunIndex` | 默认 `1` | `Baseline1CampaignPlan.runIndex(0)` |
| `experimentSeed` | `202610280422L` | `Baseline1CampaignPlan.environmentSeed(0)` |
| `freeSpeedFactorStd` | 默认 `0.05` | `Baseline1CampaignPlan.freeSpeedFactorStd(0)` |
| `boundaryEventRoot` | 历史边界试验的输出目录 | `Baseline1CampaignPlan.eventRoot(0)` |
| `boundaryRecordingId` | `"boundary-supplement-002-C2"` | `Baseline1CampaignPlan.recordingId(0)` |
| `boundaryInstrumentationRevision` | `"boundary-events-v1-outer-cutoff"` | `"boundary-events-v1-native-rpc-runtime-scene-v1"` |
| `boundaryAcceptanceConfigSha256` | 历史配置 SHA | `"0000000000000000000000000000000000000000000000000000000000000000"` |
| `acceptanceLongLoadEnabled` | `true` | `false` |
| `enableBoundaryEventRecording` | 默认 `false` | `Baseline1CampaignPlan.eventRecordingEnabled(0)` |
| `enableMaterialEventMemory` | 默认 `false` | `Baseline1CampaignPlan.materialEnabled(0)` |
| `materialEventRoot` | 默认空字符串 | `Baseline1CampaignPlan.materialRoot(0)` |
| `offlineNativeCollection` | 默认 `false` | `Baseline1CampaignPlan.offlineEnabled(0)` |
| `offlineBehaviorSha256` | 默认空字符串 | `Baseline1CampaignPlan.offlineSha256(0)` |
| `wasteDispatchStopTonsShift` | Main 默认值 | `Baseline1CampaignPlan.wasteStop(0)` |
| `fleetDisabledTruckMask` | Main 默认值 | `Baseline1CampaignPlan.fleetDisabledTruckMask(0)` |
| `fleetFaultTruckId` | Main 默认值 | `Baseline1CampaignPlan.faultTruckId(0)` |
| `fleetFaultTriggerSeconds` | 默认 `-1.0` | `Baseline1CampaignPlan.faultTriggerSeconds(0)` |
| `fleetOreTarget` | Main 默认值 | `Baseline1CampaignPlan.oreTarget(0)` |
| `fleetWasteTarget` | Main 默认值 | `Baseline1CampaignPlan.wasteTarget(0)` |

`boundaryAcceptanceConfigSha256` 的全零值只用于创建 Main，启动握手会校验并换成单班包的真实配置哈希。`deterministicCheck` 保持 `false`，`usePhysicalObservation` 保持 `true`。`acceptanceBoundaryMode` 保持默认 `0`，`bcCollectEnabled`、`diagEvalEnabled`、`enableP0SnapshotAudit` 保持关闭，`enableWasteTaskLimit`、`useLocalDynamicSpeed` 保持开启。Main 的 `dispatchInferencePort` 保持 `50051`。关闭历史长装载开关，使场景采用正常装载及卸载事件。

**Simulation → Model time** 设从 0 秒启动，结束时刻设为 **28,801 秒**。Main 的 `rlShiftSeconds` 保持 **28,800 秒**，已有 `endSimulationEvent` 在班末结算、发送终端 RPC 并调用 `getEngine().finish()`。外层停止时间比结算时刻晚 1 秒，使班末 RPC 有机会先完成。`Simultaneous events` 保持 LIFO，`Load top-level agent from snapshot` 保持关闭。为在启动前看到控制栏，可在 Simulation 的 General 取消 **Skip experiment screen and run the model**，并在 Window 开启 **Enable developer panel** 和 **Show developer panel on start**。[AnyLogic Simulation 属性](https://www.anylogic.help/anylogic/experiments/simulation-experiment.html)

## 三、在现场图上标明退场状态

1. 在演示副本的 **Main → Advanced Java → Additional class code**，把 [状态显示方法](demo_display.java.txt)追加到已有代码末尾。它读取现场已有的双掩码、车辆工序和载荷，显示三个时刻及两个车队人数。
2. 在 **Main** 的图面从 **Presentation** 面板拖入一个 **Text**，命名 `fleetDemoStatusText`，放在矿区图上方或留白处。将 **Text** 切到动态表达式，填 `fleetDemoStatus()`。建议在 2D 界面显示为三行，让时间和载荷都可读。
3. 在 **Truck → Presentation → level → text**，把现有动态内容 `this.truckId` 改为下式，6 号车会在原位带着状态标识移动，其他车辆仍显示车号。该 Text 当前为 2D，使用现有矿区图即可看车。需要 3D 时，在 **Main → Presentation → 3D** 放入一个 **3D Window**，并将该 Truck Text 的 **Advanced → Show in** 改为 **In 2D and 3D**。演示可同时保留 2D 的状态总览和 3D 的车辆视图。[AnyLogic Text](https://www.anylogic.help/anylogic/presentation/text.html) · [AnyLogic 3D Window](https://www.anylogic.help/anylogic/3d/view3d.html)

```java
main == null ? Integer.toString(truckId) : main.fleetDemoTruckLabel(this)
```

## 四、手动验收一次

1. 保存副本，执行 **Build**。确认无属性表达式错误后启动 **Simulation**。当前关卡只验收一班，整个过程用同一个窗口的 Pause 和 Run 继续。不要在同一演示身份上使用 Stop 后重开。
2. 若要快速推进，在开发面板展开 **Run until**，以模型秒填 `18780`（开班后 05:13:00），在故障前约 48 秒暂停。故障计划在 `18827.877717961626` 秒（05:13:47.878）。降速后继续，观察屏幕实际状态。可以在第一次状态变化后手动暂停，拍下当前工序、载荷和车队人数，再用 Run 从同一状态继续。[AnyLogic 指定时刻运行](https://anylogic.help/anylogic/running/enhanced-execution-control.html) · [AnyLogic 播放控制](https://anylogic.help/anylogic/running/control-panel.html)
3. 核对下面三个逻辑阶段。同一仿真秒内已有卸载先完成记账，停派后才处理新的合法派车。故障时车若仍有已承诺任务，就先完成当前任务；如果当时已经满足退场安全条件，停派和退出可发生在同一仿真时刻。

| 阶段 | 现场应看到 |
| --- | --- |
| 故障前 | `正常接单`，可派／在网为 `9 / 9` |
| 停派后仍在网 | `停派，仍在网`，可派／在网为 `8 / 9`，原任务可继续行驶、排队和卸载 |
| 安全退出后 | `已退场`，可派／在网为 `8 / 8`，载荷和装卸资源已释放，标签随车出现在停车区 |

版内 `fleetTryExit` 确认没有当前路段、载荷、预约和工序占用后，直接把静止车辆放入停车区。停车区瞬间出现目标车辆符合现有动画逻辑。其余 8 辆继续工作，不会另派目标车去停车区。

4. 看完后可加速至自然班末，等待终端回执写盘。确认记录的 `run_index`、配置 SHA、实际停派／退出时刻以及矿石和金属的物料守恒结果。班末回执中的四窗口、矿废实际产量和退场后首窗按本次实测显示。逐次 RPC 只保存目标车辆每个状态的**首个有效观测**，其秒数可能晚于状态栏显示的真实事件。具体可用字段和路径见 [封存身份](prepared_001/run_seed.csv)。

## 五、运行身份与状态查询

- Run 为 `10045001`，recording ID 为 `bg128-fleet-outage-visual-demo-20261124-001-10045001`。
- 环境 seed 为 `173391452439347`，Actor 采样 seed 为 `88655618899752`。
- [配置字节 SHA256 对应的文件](prepared_001/configs/run_10045001.json)为 `49e7d5745772fb844211f58218c9ef57db38944b71af93f8a0f8ee82ad77b48c`。
- Python 输出在项目的 `runs/bg128-fleet-outage-visual-demo-20261124-001/shifts/run_10045001/`。其中 `receipt.json` 由班末实际 RPC 写入。AnyLogic 事件路径为同一活动下的 `anylogic_events/run_10045001/`，详细事件采集关闭。

在本目录可只读查询运行状态，`--check` 用于启动前校验。

```bash
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B demo_launch.py --suite prepared_001/suite.json --status
```

单班演示使用完整冻结策略，不执行 PPO 更新。准备时合并账本为 43,046 班，用户在 AnyLogic 实际启动后加 1 班。[准备验证记录](../../../../../reports/token_attention_001/business_graph_summary_009/fleet_outage_resilience_001/visual_acceptance_demo_001/preflight.json)
