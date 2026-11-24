# AnyLogic 六车运行手工清单

本次使用九车训练的冻结 s02/u32 Actor。模型照常创建 `Truck` 0–8。编号 `2、5、8` 各占一种车型，整班停放。其余编号 `0、1、3、4、6、7` 参与运输。班长为 28,800 秒，沿用四个品位窗及原矿、废目标。当前验证包有五班，每班用独立环境种子和 Actor 私有采样种子。

在独立的模型副本中手工操作，等正在使用 RPC 端口的活动结束后再启动。对照项目中的模型以 **Main**、**Truck** 和 **RL_Training** 下的对象名定位。当前工作副本可能包含另一个活动的 Java helper。 [模型基准](../../../../anylogic/3d/DRL-PPO-baseline1/DRL-PPO-baseline1.alp) · [逐段 Java 片段](anylogic_fleet_outage.java.txt)

## 设置活动参数

1. **Main → Parameters** 新建参数 `fleetDisabledTruckMask`，类型 `int`，默认值 `0`。第 `i` 位为 `1` 表示 `Truck.truckId == i` 停运。`292 = (1 << 2) | (1 << 5) | (1 << 8)`。
2. **Main → Functions** 新建返回 `boolean` 的 `isFleetTruckDisabled`，参数 `int truckId`。将片段 1 粘到函数体。编号范围、允许的掩码和车数在创建车辆时校验。
3. **Additional Classes → Baseline1CampaignPlan** 替换为本活动准备包中的类正文，保留编辑器现有的 `package`。**RL_Training → Properties → Parameters → Freeform** 中，将 `fleetDisabledTruckMask` 填为 `Baseline1CampaignPlan.fleetDisabledTruckMask((int) index)`，由该行冻结清单提供掩码。保持 `numTrucks = 9`。现模型的 `Main.obsInit` 要求九个 Truck 对象，Python 也校验九个车辆槽、类型和吨位。 [本次 Java helper](prepared_001/Baseline1CampaignPlan.java)

## 停放车辆和阻止派车

4. **Main → Functions → init1** 定位 `Truck population` 下的 `for(int i = 0; i < numTrucks; i++)`。只把这个车辆循环替换为片段 2。旧循环把每辆车放在破碎机或排土场发车位置，随后按同节点前车设置初次发车的 30 米间隔。新循环仍按原编号和次序调用九次 `initTruck(i, cap)`，保留类型、容量、启动随机数调用和六辆活跃车的原发车位置。停运车使用 Main 已有的 `parking` 点作视觉停放位置，并将路网节点名称置为空字符串。活跃车的初始前车链跳过停运车。`parking` 不属于 `obsNodeNames` 或十六段活动调度路网。`obsNodeIndex("")` 返回 `-1`，原 `oneHot` 写出九个零。停运车保留真实的停放坐标。
5. **Main → Agent actions → On startup** 定位函数尾段给 Truck 广播 `ACTIVATE` 的循环，以片段 3 的循环替换。Truck 默认进入 `IdleAtParking`，离开该状态的 `transition` 接收 `ACTIVATE`，因此停运车整班继续停放。**Main → Functions → selectNextLoadSite** 在原首行 `obsCheckDecision(t);` 之前加片段 4 的保护。如果停运车误入派车路径，直接中断这一班。
6. **Main → Events → evDynamicSpeed → Action** 修改尾部遍历 Truck 调用 `t.onChange()` 的循环，使用片段 5。既有 `updateFrontRearVehicles`、`updateNextSegmentFrontVehicles` 和 `applyDynamicSpeed` 已按 `currentSegment == null` 跳过停放车，所以这三处沿用原实现。停运车既不占道路，也不进入装卸队列。`getCumulativeWaitVehicleSeconds` 统计装卸资源等待，不把全班停放计入等待成本。界面若显示 `Truck` 的 idle 时间，该诊断可包含停运车。

## 上报停运配置与正确的车辆状态

7. **Truck → Functions → appendDispatchObservation** 找到从 `int phase=11;` 到 `if(decision) phase=0;` 的两行，用片段 6 替换，原来的 `else if(!initialDispatchDone)` 及以下分支接在其后。停放车仍保持 `initialDispatchDone=false`，通过显式 `phase.11=1` 区别于 `phase.1` 的初始待派车。车辆 ID、类型、容量、槽顺序、字段名及维度仍走原写入逻辑。这样观测不会把停运车交给规划模型作为可释放车辆，也不会把停车点纳入活动路网图。
8. **Main → Functions → buildRewardConfigComponents** 在 `c.put("config.num_trucks", (double) trucks.size());` 后加片段 7。START、DECISION 和 END 都走现有 RPC 配置构造。六车配置信息为 `config.fleet.outage_version=1`、`config.fleet.disabled_mask=292` 和 `config.fleet.active_count=6`。`config.num_trucks=9` 和 `config.runtime_scene.*` 沿用原契约。新增字段在 `config.fleet.*` 命名空间，冻结工况通过显式 RPC 配置检查区分。全车默认掩码 `0` 时沿用旧九车 RPC 字段。

## 启动前和运行验收

- 核对 **RL_Training → Properties** 的 Freeform、串行、关闭 Replications、五班、28,800 秒、`currentDispatchPolicy=3`、物理观测和低频记录。逐班核对新 `recording_id`、run、两个种子、配置字节 SHA256、Python 输出和 AnyLogic 事件路径。保持已有 `Main.On startup` 的 ready/claim 握手。然后 **Save → Build → Run RL_Training**。 [冻结身份表](prepared_001/run_seed.csv)
- 首次验证在 START 和 END 检查观测长度 1519、原物理 schema、九车辆槽和三个 `config.fleet.*` 字段。停运 2、5、8 始终 `phase.11=1`，无 ego、前车、载荷、预留、节点、道路和运动。活跃车的前车不能引用停运车。每次实际派车请求均来自 `0、1、3、4、6、7`。Python 班级签收将保留调用和停运见证；发现异常时保留 STOP、账本和已完成回执。
- 班末核对四窗按顺序封闭、真实矿废吨位和等待守恒、临界班末 RPC 完成，PPO 更新为零。当前五班用于验证九车策略在六车资源约束下能否正常运行及其物理表现。
