# baseline2 手工接线清单

源模型为 `/home/jinx/Models/DRL-PPO-baseline2/DRL-PPO-baseline2.alp`，核对的 SHA256 为 `eb8c46b7f7410daeb9997bf18ac0aa07e53ca5e5a652e0d79e15ceb425baae26`，模型时间单位为秒。下文的行号指这个版本的源 XML，操作位置使用 IDE 的对象名。本次交付的是手工改动和离线验证，实际闭环要用按本活动生成的身份。

## 1　绑定一次实验的冻结参数

1. 项目树 → Java classes → `Baseline1CampaignPlan`，用本活动准备目录中生成的同名 Java 类替换原类的内容。保留 IDE 的包 `fusion`，类文件只保留一个 package 声明。旧类仅接受 10 班、停运掩码 292；新类读取对应活动的主表和场景表并核对文件 SHA。项目树中新建同包 Java 类 `FleetResilienceState`，将 [FleetResilienceState.java.txt](FleetResilienceState.java.txt) 的完整类声明贴在包声明之后，替换 IDE 生成的空类声明。
2. Main → Parameters 添加下表的四个参数。基线的三个产量字段都是普通 variable（1338、1353、1726 行）；由 Startup 在校验前赋值。`fleetDisabledTruckMask` 与 `wasteDispatchStopTonsShift` 已是 parameter。
3. 实验 `RL_Training` → Parameters → Freeform 填入下表；`index` 是从 0 开始的物理班行序。把原 Number of runs `10` 改为新准备包 `suite.json` 的 jobs 数量（与生成类的 `jobCount()` 相同），保持并行评估关闭。原有实验 run、seed、RPC port、recording、eventRoot 和 Python 握手绑定沿用生成类的同名方法。

| Main 参数 | 默认值 | RL_Training Freeform 值 |
|---|---:|---|
| 新 `int fleetFaultTruckId` | `-1` | `Baseline1CampaignPlan.faultTruckId((int) index)` |
| 新 `double fleetFaultTriggerSeconds` | `-1.0` | `Baseline1CampaignPlan.faultTriggerSeconds((int) index)` |
| 新 `double fleetOreTarget` | `16000.0` | `Baseline1CampaignPlan.oreTarget((int) index)` |
| 新 `double fleetWasteTarget` | `4000.0` | `Baseline1CampaignPlan.wasteTarget((int) index)` |
| 已有 `int fleetDisabledTruckMask` | 0 | `Baseline1CampaignPlan.fleetDisabledTruckMask((int) index)` |
| 已有 `double wasteDispatchStopTonsShift` | `6000.0` | `Baseline1CampaignPlan.wasteStop((int) index)` |
| 已有 `boolean enableMaterialEventMemory` | false | `Baseline1CampaignPlan.materialEnabled((int) index)`，本活动为 false |

基线的 `RL_Training` 只在 Number of runs 处实际限制为 10。原 Replications 的 Maximum=10 由关闭的 Use replication 控制；实验的 0–28800 秒指单班仿真时间。主表模板还有限定行数的 Java `JOB_COUNT`，准备程序用当次行数替换。启动时从第 0 行开始，Python 启动器等首班 ready 最多 840 墙钟秒，Java 对该启动器最多等待 15 分钟，随后每个物理 run 独立等待其 ready 最多 60 分钟，RPC 单次调用的 `dispatchInferenceTimeoutMs` 为 300000 毫秒。这些时间限制均对应一次握手或请求。

保留 Policy 3、物理观测、RPC、实际窗结算与废石停派开关。训练与测试默认关闭逐请求观测、物料逐事件文件、逐 transition 和 rollout 快照。原 `Main.init1` 的四组 `loadIsOre/loadGrade/loadGradeHalfRange(experimentRunIndex, site)` 读取主表关联的场景，物理道路与装点 ID 保持稳定，车号对应车型 `0–2 / 3–5 / 6–8` 的额定载重分别为 91.7、147、193 吨。

## 2　初始化出勤与仿真定时事件

在 Main → Advanced Java → Additional class code 追加 [Main.additional.java.txt](Main.additional.java.txt)。在 Truck 的同一属性追加 [Truck.additional.java.txt](Truck.additional.java.txt)。后者是本版模型 15768 行原 `NeedDispatchToLoadSite` 完整 Entry action 的方法形式，保留原统计、抽样和派单提交点。

**Main → Events** 新建 `evFleetFault`，Trigger type 为 Timeout，Mode 为 User control，单位秒。Action 填：

```java
fleetSync();
```

**Main → Agent actions → On startup** 在原 `if (!Double.isFinite(targetOreTonsShift)` 之前插入以下内容。每次 Main 实例只在这里创建一次状态，班中不重算吨位目标。

```java
targetOreTonsShift = fleetOreTarget;
targetOreTons2h = fleetOreTarget / 4.0;
targetWasteTonsShift = fleetWasteTarget;
fleetResilience = new FleetResilienceState(fleetDisabledTruckMask, fleetFaultTruckId, fleetFaultTriggerSeconds);
```

本函数中紧接原 `init1();` 添加：

```java
DumpSite fleetCrusher = getDumpSiteById(0);
if (!fleetCrusher.crusherFifo.isEmpty() || fleetCrusher.crusherStoredTons != 0.0
    || fleetCrusher.receivedTons != 0.0 || fleetCrusher.receivedMetalTons != 0.0
    || crusherOreTons != 0.0 || crusherMetalTons != 0.0 || time(SECOND) != 0.0)
    throw new IllegalStateException("Fleet shift must start with an empty physical ledger");
if (fleetFaultTruckId >= 0) evFleetFault.restart(fleetFaultTriggerSeconds, SECOND);
```

原 Startup 最后（215 行）只对 `!isFleetTruckDisabled(t.truckId)` 的车送 `ACTIVATE`，保留此循环。无故障班参数为 `(-1,-1)`，故障班安排一次真实仿真时钟回调。按班初实际车辆容量之和除以 1295.1，`oreTarget / wasteTarget / wasteStop` 分别等于 `16000 / 4000 / 6000` 乘此系数。核对 `config.target_ore_tons_2h` 为班矿目标的四分之一。三型车开班都有在役车，一车容差沿用原 `getWasteTaskUpperTons()` 的 193 吨。

**Main → Functions → init1** 用以下代码替换限制 `numTrucks != 9 || (fleetDisabledTruckMask != 0 && fleetDisabledTruckMask != 292)` 的整个 if（3929 行）。后面的 `add_trucks`、停车位、初始前车查找和激活流程按原模型运行。

```java
if (numTrucks != 9 || fleetResilience == null)
    throw new IllegalStateException("Initialize the nine-slot fleet before creating physical trucks");
```

**Main → Functions → isFleetTruckDisabled(int truckId)** 原末行直接读取开班掩码（10766 行）。替换整个函数体为：

```java
return !fleetResilience.contains(fleetResilience.inSystemMask, truckId);
```

这使原 `Truck.appendDispatchObservation` 在实际退场后写 phase.11。IdleAtParking 中的未出勤槽持有停车坐标和车型，Python 用实时物理掩码消除其节点、边和池化贡献。无需在 1519 维原观测向量中增列。基线 `buildPhysicalObservation` 的字段顺序与 `buildRewardConfigComponents` 的八个 schema 指纹字必须通过 RPC 核验。

## 3　挂接派单、道路和卸载

**Truck → Statechart → NeedDispatchToLoadSite → Entry action** 用下式替换从 `initialDispatchDone = true;` 至 `this.deliver("TOLOADSITE", this);` 的原完整动作。原动作已在步骤 2 的 `fleetDispatchNow()` 中。

```java
main.fleetDispatch(this);
```

**Main → Functions → selectNextLoadSite(Truck t)** 原函数首行 `if (isFleetTruckDisabled(t.truckId)) {` 改为：

```java
if (!fleetResilience.contains(fleetResilience.dispatchableMask, t.truckId)) {
```

保留原异常和后续 `obsCheckDecision`。处于 (0,1) 的车继续按原状态机行驶、加载、排队、占用资源、卸载；其后续派单请求由 `fleetDispatch` 在进入 RPC 前拦截。

**Main → Events → evDynamicSpeed → Action** 原函数 3732 行包含三次对全车更新。替换完整 Action：

```java
if (episodeClosed || time(SECOND) >= rlShiftSeconds) return;
if (fleetResilience.triggerSeconds >= 0 && fleetResilience.exitSeconds < 0) fleetSync();
if (useLocalDynamicSpeed) {
    for (Truck t : trucks) if (!isFleetTruckDisabled(t.truckId)) t.updateBasicSpeedState();
    updateFrontRearVehicles();
    updateNextSegmentFrontVehicles();
    for (Truck t : trucks) if (!isFleetTruckDisabled(t.truckId)) applyDynamicSpeed(t, useEndogenousCongestion);
    for (Truck t : trucks) if (!isFleetTruckDisabled(t.truckId)) t.obsRecordMotion();
}
for (Truck t : trucks) {
    if (fleetResilience.contains(fleetResilience.dispatchableMask, t.truckId) && !t.initialDispatchDone)
        t.onChange();
}
```

在 `evUpdateSpeedObservation` 与 `evApplySpeedActions` 两个 Event 的 Action 中，分别在第一个 `for (Truck t : trucks) {` 紧后加入 `if (isFleetTruckDisabled(t.truckId)) continue;`，保留这些事件现有的触发配置。基线实际用 `moveTo`、`currentSegment` 与 `updateFrontRearVehicles` 建立车道占用。到达卸点时 `CheckDestination1` 已清空路段。基线没有单独的卸后空返停车路程；下一趟的空驶已经占有装载预约。只有到站停稳、预约和载荷清空、队列和服务块释放时 `fleetTryExit` 才移至 `parking`，并清理前车引用、速度、路线及历史。若今后加入空返，须在最终停稳状态使用同一安全出口，提供对应的状态名与回调。

**DumpSite → delayDump → Delay time** 把 21073 行的 `dumpServiceBoundaryMinutes((Truck)agent)`（分钟）替换为以下表达式，并把该属性的单位切为 **seconds**。登记使用同一次采样值；同时到期的多台卸载服务分别由车号跟踪。

```java
main.fleetResilience.startDump(((Truck)agent).truckId, time(SECOND),
    60.0 * dumpServiceBoundaryMinutes((Truck)agent))
```

**DumpSite → exitDump → On exit** 原 `t.onDumpingFinished(this);`、`t.send("restart", t);` 之间插入两行，得到：

```java
t.onDumpingFinished(this);
main.fleetResilience.finishDump(t.truckId, time(SECOND));
main.fleetSync();
t.send("restart", t);
```

保留该动作开头的班末截止检查、`Truck t = (Truck)agent;` 和 `totalServed++;`。原卸载函数负责入仓或废石交付、清载和状态；`restart` 负责进入“收集数据并重置”，计入最后完成循环。原 `Truck.transition7` 收到 restart 后清 `dumpFinishedFlag` 并进入“收集数据并重置”，在那里登记本车周期、装点周期与 `main.completedTrips++`。本补丁把最后一趟的 `targetLoadSiteId` 保留至此；车辆已停派时下一次进入 NeedDispatch 将直接返回。如果卸完恰好等于 28800 秒，exitDump 顶部的班末判断先于 book/restart，这一趟既不入仓也不增加完成行程；此前已卸完的行程按原回调记账。

`RL_Training` 的同时事件选择规则为 LIFO。三个入口共同执行业务顺序。派单撞上到期卸载时先放入短期 FIFO；同一模型时刻最后一次 `finishDump` 后才剥夺目标车资格，再放行其他合法派单。只有派单撞上故障而没有到期卸载时，`fleetDispatch` 自行先提交故障。三者都读未经四舍五入的模型秒，两个不同模型时刻即使显示为同一整数秒也各自处理。

## 4　物料账本、滚动窗及传输

**DumpSite → Functions → advanceCrusherTo(double nowSec)** 在原 `boundaryAuditCursorSec = boundaryAuditEndSec;`（20148 行）之前插入：

```java
main.fleetResilience.processed(boundaryAuditCursorSec, boundaryAuditEndSec, tons, batch[1]);
```

此处每段是单批次恒品位、恒速的实际处理区间；按与 `[exit,exit+7200)` 的时间交集计量，跨边界时只计重叠部分。矿石到仓时间、派车时间与墙钟均不参与切窗。班末不足两小时的窗可输出部分吨位供诊断。正式分母只计实际退场且 `exit <= 21600` 的班，完整窗无矿计不合规。原四个固定窗按各自的原始质量加权统计继续结算。

**Main → Functions → buildRewardConfigComponents** 原 `config.num_trucks` 后有一整段 `if (fleetDisabledTruckMask != 0) {...}`，输出旧版静态信息（9206–9216 行）。替换这一整个 if：

```java
fleetResilience.components(c);
fleetAuditWasteCommitment();
if (time(SECOND) >= rlShiftSeconds) {
    DumpSite d = getDumpSiteById(0);
    fleetResilience.terminal(c, d.receivedTons, d.receivedMetalTons,
        crusherOreTons, crusherMetalTons, d.crusherFifo, cpMassBalanceEpsilon);
}
```

原固定字段 `config.num_trucks=9` 继续用于物理 schema。新增开班字段 `config.fleet.outage_version=2`、`disabled_mask`、`active_count` 在各决策和 END 均固定。实时字段 `fleet.dispatchable_mask`、`fleet.in_system_mask` 供策略；`fleet.trigger_seconds` 与 `fleet.exit_seconds` 为实际发生时刻，发生前为 -1。故障车 ID 和预定时刻仅供仿真及 Python 计划核对。

保留 `Main.getWasteTaskCommittedTons` 原 `Σ(废石点.producedTons + reservedTons)`。上段在每次 RPC 对账时与“已交付废石 + 所有在网实载废石 + 尚未装载的预约”独立比较，退场途中的废石仍占停派线。终端原函数 `finishEpisodeAndNotify` 先调用 `getDumpSiteById(0).advanceCrusherTo(rlShiftSeconds)`。END 从卸点累计量、全班累计破碎量和实际 FIFO 批次分别取值，以 `cpMassBalanceEpsilon`（本版默认 1e-5 吨）分别核对矿石和金属守恒；预约和在途吨位属于入仓前的物料。

## 5　验收位置

| 工况 | 在何处核查 |
|---|---|
| 9 车无故障 | START、每次派单、END 的两个 mask 均为 511；实际触发、退场均为 -1。RPC 检查 1519 字段指纹。迁移验收独立比较 4 个 Actor 概率与 Critic。 |
| 6–9 车开班 | 核对计划车型、容量比、四装点角色，初始停运车 phase.11，路段、装卸队列与资源均无此车；原位置不会生成派单。 |
| 装载预约、载矿或载废期间故障 | 触发时只有 dispatchable 的目标位消失。到卸载前仍能看到车辆、预约/实载以及排队。最后一次卸载产生一笔交付，随后退出并计入最后一次完成行程；目标车此后无新决策。 |
| 相同时刻 | 单独走“同车卸载、触发、派单”和“另一辆车的请求先进入、还有两次卸载待完成”两种顺序，最终先卸载、后撤资格、再产生合法派单。初始主体、动作屏蔽、最终卸载、前后车及所有队列引用由回调与接收端交叉核验。 |
| 班末和窗 | `exit=21600` 纳入，略晚于 21600 排除，未退场或无故障排除；完整但零吨的窗保留分母。END 六个矿石/金属计量字段做两条相互独立的等式校验。 |
| 班末仍在途 | 有触发且 exit=-1，预约/载荷随在网车辆截止，按实际入仓结算。保留未完成退场计数，故障后完整窗不入分母。 |

既有 1519 个字段名称以 UTF-8 按顺序每行一个，包含末尾换行，SHA256 为 `1e347d28170348e43b87229cd98198e3d29084eebeabcdd6a19281551b84405f`。本清单的 Java 回调与两个 Additional class code 已使用此模型 IDE 生成的类型声明、本机 AnyLogic 8.9 的 SDK 和过程库做过 javac 类型检查。手工接线后仍需在 IDE 生成并编译完整模型，实际班次另行验证。

在本目录可复查原模型中的全部挂点、相同时间事件测试及原行程完成转移，脚本在临时副本内编译，默认将结果输出到控制台：

```bash
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B verify_anylogic_offline.py \
  --alp /home/jinx/Models/DRL-PPO-baseline2/DRL-PPO-baseline2.alp \
  --ide-project /home/jinx/.AnyLogicUniversity/Workspace8.8/DRL-PPO-baseline2
```

仅运行独立 Java 账本用例可使用：

```bash
fleet_qa_dir=$(mktemp -d)
cp FleetResilienceState.java.txt "$fleet_qa_dir/FleetResilienceState.java"
cp FleetResilienceChecks.java.txt "$fleet_qa_dir/FleetResilienceChecks.java"
javac -encoding UTF-8 "$fleet_qa_dir/FleetResilienceState.java" "$fleet_qa_dir/FleetResilienceChecks.java"
java -cp "$fleet_qa_dir" FleetResilienceChecks
```

一次性 Event 与 `restart(timeout, units)` 对应 [AnyLogic Event 文档](https://anylogic.help/anylogic/statecharts/events.html)。
