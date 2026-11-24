# AnyLogic 四种六车停运工况手工清单

使用九车训练的冻结 BG128 s02/u32 Actor，四种配置各五班，共 20 班。每班保留九个 Truck 对象和 `dispatch.physical.v3` 的 1519 个输入值，三个指定车辆整班停放。六辆活跃车保留它们在原九车班中的发车位置。班长 28,800 秒，沿用四个品位窗及原矿、废产量目标。

| 停运配置 | 掩码 | 停运 ID | 排土场 `dumpNode` 发车 | 破碎机 `Crusher` 发车 |
| --- | ---: | --- | --- | --- |
| 重型全停 | 448 | 6、7、8 | 0、1、2 | 3、4、5 |
| 每种车型停一辆 | 292 | 2、5、8 | 0、1 | 3、4、6、7 |
| 轻型全停 | 7 | 0、1、2 | 无 | 3、4、5、6、7、8 |
| 中型全停 | 56 | 3、4、5 | 0、1、2 | 6、7、8 |

轻型全停时，六车都从 `Crusher` 发车。这个工况同时改变了活跃车的车型构成和发车点分布，反映从原九车布置直接抽走指定车型的效果。保留各车原始容量和启动初始化次序。 [模型基准](../../../../anylogic/3d/DRL-PPO-baseline1/DRL-PPO-baseline1.alp)

## 从已设置 `fleet_outage_001` 的副本升级

1. **Main → Functions → init1** 找到原停运参数检查 `if (numTrucks != 9 || (fleetDisabledTruckMask != 0 && fleetDisabledTruckMask != 292))`。用新版片段 2 开头的 `if` 块替换，允许 `0、448、292、7、56`。保留随后九车创建循环及原停运逻辑。 [Java 片段](anylogic_fleet_outage.java.txt)
2. **Additional Classes → Baseline1CampaignPlan** 替换为本轮已冻结的类正文，保留 package。**RL_Training → Properties → Parameters → Freeform** 检查 `fleetDisabledTruckMask` 仍为 `Baseline1CampaignPlan.fleetDisabledTruckMask((int) index)`，本轮 helper 按冻结清单返回四种掩码。 [本轮 Java helper](prepared_001/Baseline1CampaignPlan.java)
3. **RL_Training → Properties → Number of runs** 改为 **20**。其余按下方运行验收核对。

## 从九车原始副本设置

1. **Main → Parameters** 新建 `int fleetDisabledTruckMask`，默认值 `0`。第 `i` 位表示停运 `Truck.truckId == i`。**Main → Functions** 新建 `isFleetTruckDisabled`，返回 `boolean`，参数 `int truckId`，粘贴片段 1 到函数体。
2. **Main → Functions → init1** 定位 `Truck population` 下的 `for(int i = 0; i < numTrucks; i++)`。用片段 2 替换这个循环并加入参数检查。停运车放到现有 `parking` 点，路网节点设为空字符串。`obsNodeIndex("")` 返回 `-1`，原 `oneHot` 写出九个零。停车点坐标仍可见。活跃车从原位置发车，初始前车链跳过停运车辆。
3. **Main → Agent actions → On startup** 用片段 3 替换末尾 `ACTIVATE` 循环，保留启动 ready/claim 握手。`Truck` 从 `IdleAtParking` 出发，原状态转换需接收 `ACTIVATE`。**Main → Functions → selectNextLoadSite** 在 `obsCheckDecision(t);` 前插入片段 4，遇停运车辆请求派车立即终止该班。
4. **Main → Events → evDynamicSpeed → Action** 用片段 5 替换 `t.onChange()` 所在的尾部循环。既有前后车识别与速度控制已跳过 `currentSegment == null` 的车辆。停放车不进入装卸队列，资源等待统计不计入停放时间。
5. **Truck → Functions → appendDispatchObservation** 用片段 6 替换 `int phase=11;` 和 `if(decision) phase=0;` 两行，后续 `else if(!initialDispatchDone)` 等分支接在其后。停运车写 `phase.11=1`，保留真实的 `initialDispatchDone=false`。
6. **Main → Functions → buildRewardConfigComponents** 在 `c.put("config.num_trucks", (double) trucks.size());` 后加片段 7。非零掩码的 START、DECISION 和 END 均上报 `config.fleet.outage_version=1`、`config.fleet.disabled_mask` 等于当前工况、`config.fleet.active_count=6`。`config.num_trucks=9`。默认掩码 `0` 仍激活九车并保持旧九车 RPC 配置。
7. 按上节第 2、3 项更换本轮 helper、填 Freeform 表达式、将 Number of runs 设为 **20**。 [逐段 Java 片段](anylogic_fleet_outage.java.txt)

## 初始发车与验收

- 各发车点内，最小活跃 ID 没有前车，后车只指向同一发车点中最近的较小活跃 ID。重型全停的链头为排土场 `0` 和破碎机 `3`。轻型全停的唯一链头为破碎机 `3`，链为 `3 → 4 → 5 → 6 → 7 → 8`。中型全停时破碎机的链头变为 `6`。原有 `Truck → statechart → transition8` 仍等待活跃前车完成初派并拉开 30 米。各链严格按 ID 递增，且从无需等待前车的活跃车开始。 [发车转换位置](../../../../anylogic/3d/DRL-PPO-baseline1/DRL-PPO-baseline1.alp#L16737)
- 在独立模型副本确认 Freeform、串行、关闭 Replications、20 班、28,800 秒、`numTrucks=9`、`currentDispatchPolicy=3`、物理观测和低频记录。等现有 RPC 活动结束后，核对各班新 `recording_id`、run、双种子、配置字节 SHA256、Python 输出和 AnyLogic 事件路径。然后 **Save → Build → Run RL_Training**。 [冻结身份表](prepared_001/run_seed.csv)
- 核对每班的 `config.fleet.disabled_mask` 与清单一致，活跃车计数为 6。停运 ID 只处于 `phase.11`，无 ego、前车、道路、运动、载荷或预留。活跃车的前车不得引用停运 ID。班末核对 28,800 秒 END、四窗结算及吨位与等待守恒，确认零优化器更新。出现 STOP 时保留账本及已完成签收。
