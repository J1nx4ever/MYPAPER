# A/B/C 原始观测干预合同

2026-11-18，Asia/Shanghai。只针对已保存的 `float32[1519]` 观测构造离线外生分支。固定模型初始化 seed 为 `2026111801`，本脚本不实例化模型、不训练、不启动 RPC/AnyLogic、不使用新班次预算。没有修改 `.alp`、正式实现包或历史记录。

实现入口：[build_scenarios.py](build_scenarios.py)。输出为 [scenarios.json](../../../../reports/token_attention_001/business_graph_summary_009/physical_intervention_001/scenarios.json)、[逐字段变化](../../../../reports/token_attention_001/business_graph_summary_009/physical_intervention_001/raw_changes.csv) 和 [物理约束检查](../../../../reports/token_attention_001/business_graph_summary_009/physical_intervention_001/physical_checks.json)。

## 1. 共同基点与接口

选择已认证历史快照 **T1**：run `9810998`、state_index `62`、Ego `Truck0`，时间 `9717.867211564166 s`。T1 比 T0 更适合独立干预：Truck1 正在装载且品位完全可推知；Truck8 尚在前往装载点的共同道路前缀上，可改未来路线尾而不移动车辆。

`scenarios.json` 顶层 `base.raw` 为历史 T1，`scenarios.A/B/C.raw` 为三个独立分支。每个 raw 长度1519、序列化后按 float32 重新读入；`raw_sha256` 对连续 float32 字节计算。`intervention` 保存主体车辆、old_load/new_load、吨位/品位、外生设定和边界说明。A/B/C 都相对基点，不能依次叠加。

原始 HDF 字节 SHA 与历史 selection 中记录一致，raw 与 HDF 原行逐元素一致。代码读取当前本地 Java 定义作为语义依据并记录 SHA；这不证明历史快照来自该 ALP 的某一内部事件时序，也不代替运行时配置合同。

## 2. A：一辆当前正在装载的车完成已知物料

主体为 **Truck1、Load0、91.7 t**。T1 已提供 `loadingGradeInferred=1`、`loadingGradeStd=0`、`loadingGradeMean=0.37571972608566284`，因此载荷品位使用当前可观测值，未访问 T1 之后的快照或隐藏随机块。当前装载起点约2525.3 t、终点2617 t，全部在第5号500 t品位块；Load0 的 `lastKnownBlock/lastKnownGrade` 已与之相同。

外生设定是在同一时刻令当前已分配装载提前完成。总车数仍为9。为了保留所有已观测运动历史，取 `onLoadingFinished` 之后、下一状态机行驶状态进入之前的内部边界：

|字段|改动|依据|
|---|---|---|
|Truck1 phase|4→11|`isLoaded=true` 后 `loading=false`；旧 executionPhase 尚为 LOADING，观测默认分支为11|
|Truck1 loaded/payload/reserved|loaded=1，payload=91.7 t，reserved=0|`onLoadingFinished` 的完成规则|
|Truck1 gradeKnown/payloadGrade|1 / 当前已知品位|不使用未来品位|
|Truck1 loadingGrade、loadAge/loadMean/loadMaterialStart|归零|这些字段受 `loading` 门控|
|Truck1 loadTarget/dumpTarget|清空 / Dump0|装载预留完成；已载矿石的卸载目标为0|
|Load0 remaining/reserved/produced|−91.7 / −91.7 / +91.7 t|`completeLoading` 物料守恒|
|Load0 service/shovelBusy/loadingCount|移除Truck1，1→0，1→0|结束当前唯一装载服务并释放资源|

`available=remaining-reserved` 在此代数上不变，保留原 float32 序列化值，避免重新相减引入无业务含义的1ULP变化。`allocationCursor` 已在进入装载服务时推进，完成时不再推进。原先已知块记录保持；当前 FIFO、已破碎吨位/金属、Dump0 bufferReserved、车辆位置、路段、路线和运动/自由速度历史均不变。**Crusher bufferReserved 只在开始卸载时增加，不能在装载完成时增加。**

该分支新增的是一辆“已载且目标Dump0”的车辆，它仍在装载点，尚未开始道路行驶。禁止把它描述成在某路段上增加了一辆车。固定图中 `inbound_loaded_*` 的现有定义包括所有有卸载目标的已载车辆，并不要求车辆正在道路上。

**边界限制：** 这是同步事件内部暂态的合成观测。原生产 RPC 不保证在这里产生决策快照；本次没有证明该完整状态能被 AnyLogic 直接输出。若要求车辆进入 `MoveToDump`，Java 会调用 `obsResetMotion`，需要另行定义当前腿历史重置，不能一面声称精确复现该状态转换，一面保留旧腿运动缓存。

## 3. B：尚未开始服务的预留任务转移

主体为 **Truck8，Load0→Load1，193 t**。它处于空载 `TO_LOAD`，当前道路7为 `Node2→Node3`，不在任何Load的queue、seize或service列表。两个装载点都是矿石。

|字段|改动|同步含义|
|---|---|---|
|Truck8 loadTarget|Load0→Load1|预留任务身份变化|
|Truck8 剩余路线|[7,8]→[7,12,13]|保留当前道路7；未来尾路由转向Load1对应Node6|
|Truck8 routeRemaining|当前路段剩余距离＋道路12和13长度|不改变当前车辆位置、segmentRemaining和laneRank|
|Load0 reserved/incoming/available|−193 t / −1车 / +193 t|取消旧尚未服务预留|
|Load1 reserved/incoming/available|+193 t / +1车 / −193 t|建立新尚未服务预留|

新路线有序、连续、无重复路段且终点对应新Load。当前道路未改变，因此 road occupancy 与其他车辆laneRank无需改动。站点都没有Truck8的服务/排队成员，不得凭空添加或删除队列。两站remaining、produced、allocationCursor、knownGrade、FIFO和已发生历史不变。

float32量化核查：Load0 的available净增约193.001032 t、reserved净减192.999998 t，二者差0.001034 t；Load1 的available净减192.999244 t、reserved净增192.999998 t，二者差0.000754 t，均在本合同0.006 t容差内。物理设定仍是193 t的预留转移，这些差异来自已有/重新序列化的 `/30000` float32，而非新增物料。

这里严格区分**预留**与**物料分配游标**：`reserveLoad` 只增加 reservedTons/incomingTrucks；在 `LoadSite.delay.onEnter` 中调用 `takeMaterialGrade` 才推进 cursor。不得把“cursor = produced + 所有reserved”作为守恒式。本快照应为“cursor = produced + 已进入装载服务的已分配吨位”。

外生原子改派在观测层保留了物料与路线约束；当前模拟器没有把运行中改派暴露为PPO动作，本次不是执行其在线派车API，也不证明服务事件日程已被模拟器重新安排。

## 4. C：当前破碎处理能力降容

只修改 raw index `1304`：`dump.0.crusherRate` 从1变0.5，即 **2500→1250 t/h**。Java 参数为double，原初始化/推进检查要求正值，1250满足该限制。外生变化发生在当前时刻，因此历史累计吨位/金属、FIFO、buffer容量/库存/预留/可用量、排队和所有车辆数据保持。

当前运行配置仍是2500 t/h。本分支是离线参数变化，不是已部署环境配置，也不能被标成原历史真实数据。后续完整前向应重新计算C128中的处理容量/可用容量/窗口需求等；不得手工修改C128或篡改原定义中的名义ETA。

## 5. 共同验收与前向要求

原始 schema 校验之外，独立检查：九车与Ego、固定时间/累计全局历史、所有运动和初始派车历史、原FIFO、地理位置、道路配置；每Load的库存守恒、预留与车辆合计、服务成员与计数、已分配游标、在途数量；每Truck的载荷/品位/目标门控、剩余路线次序/连续性/终点/距离；Dump的buffer守恒、仅卸载服务车辆占用bufferReserved；总矿石完成装载=已破碎+FIFO库存+现有已载吨位。float32归一化值允许明确的舍入容差，未clamp非法输入。

重新计算动作mask/tons，三个分支均未改变Ego的可派吨数或合法动作。A仅Load0服务和载荷目标事实变化；B仅移转指定预留事实及道路尾编码；C不改变边结构，但站点内容和C128会变。

后续消费方必须由各分支 raw 完整重建 **graph、quantity、C128及其原始字节证书、H0/H1/H2、Actor候选表示与logits、Critic值**，不得沿用基点中间缓存。当前固定随机初始化下的输出差异仅验证数值传播路径，不能解释为“已经学会正确业务”。

这些检查覆盖可观察字段约束，不覆盖不可观察的状态机事件队列、完整隐藏物料序列、资源库内部成员与随机时钟，所以不是AnyLogic可达性证明。

## 6. Java/schema依据

- [观测schema](../../../../python/observation_schema.json)：当前字段、尺度和门控。
- [Truck.onLoadingFinished](../../../../anylogic/3d/DRL-PPO-baseline1/DRL-PPO-baseline1.alp:17846)：完成装载、清空预留并设置isLoaded。
- [Truck.appendDispatchObservation](../../../../anylogic/3d/DRL-PPO-baseline1/DRL-PPO-baseline1.alp:18042)：phase11、loaded/target/route和历史字段。
- [LoadSite.reserveLoad](../../../../anylogic/3d/DRL-PPO-baseline1/DRL-PPO-baseline1.alp:23027)、[completeLoading](../../../../anylogic/3d/DRL-PPO-baseline1/DRL-PPO-baseline1.alp:23101)、[takeMaterialGrade](../../../../anylogic/3d/DRL-PPO-baseline1/DRL-PPO-baseline1.alp:23137)：预留、完成、分配游标的不同更新时点。
- [装载服务onEnter](../../../../anylogic/3d/DRL-PPO-baseline1/DRL-PPO-baseline1.alp:24185)：进入服务才消耗物料分配游标。
- [Dump.delayDump.onEnter](../../../../anylogic/3d/DRL-PPO-baseline1/DRL-PPO-baseline1.alp:21025)：开始卸载才预留buffer。
- [crusher参数检查](../../../../anylogic/3d/DRL-PPO-baseline1/DRL-PPO-baseline1.alp:19807)、[rate观测](../../../../anylogic/3d/DRL-PPO-baseline1/DRL-PPO-baseline1.alp:20288)：正值参数与/2500尺度。
