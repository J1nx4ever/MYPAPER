# AnyLogic 现场案例手工调整清单

在单独的 `baseline2` 副本中接线。下面的修改前内容来自既有运行接线文档和源快照。修改后的现场参数从[现场数据对照](README.md#第一版可用的现场依据)和同班实绩填入。准备以实际 RPC 核实模型和配置，首班身份随正式准备包另行冻结。本清单用于现阶段确定挂点和验收内容。

## 固定去向版

| AnyLogic 位置 | 修改前 | 修改后、理由和验证 |
|---|---|---|
| **Java classes → `Baseline1CampaignPlan`**，**Simulation 或 RL_Training → Parameters** | 历史 helper 读取既有活动表和物理身份。 | 新活动用同类生成 helper 读取按班冻结的实际场景。按所用实验选择 `0` 或 `(int) index` 绑定新配置、环境 seed、recording、目录和 run ID。保留 Main 启动握手。验证 helper 读取的工况与首个实际 RPC 一致。 [演示参数布局](../fleet_outage_resilience_001/visual_acceptance_demo_001/manual_anylogic_demo.md#L13) |
| **Main → Functions → `init1` → s0–s3 `initLoadSite(...)`** | 四个固定编号，品位和矿废角色通过 `loadIsOre/loadGrade/loadGradeHalfRange(experimentRunIndex, site)` 读工况。设备参数仍沿用模型设定。 | 四槽填同一班的铲位、当班物料角色、工作面名义品位和波动。铲装时长有实测时，在 `initLoadSite` 的装载周期、斗容、物料密度或服务能力对应字段接入实测参数。核对第一帧角色及名义品位、实际装载吨数和服务时长，站 ID 与边端点不变。 [四站接线参考](../generalization_001/grade_assignment_generalization_001/station_role_control_001/README.md#anylogic-手工接入位置) |
| **Main → Functions → `init1` → `Truck population`** | `k=i/3`，`cap` 为 `91.7/147/193` 吨，再调用 `t.initTruck(i, cap)` 和 `t.truckType=k`。 | 兼容版仅加载实际候选吨量不超过 schema 上限的已核对车组，按真实车号、车型和容量填固定槽。班初在役名单落实为 `fleetDisabledTruckMask`。导出实际 `truck.*.capacity_t`，核对首帧候选吨量、实际载荷、出勤人数和停运槽。若车组含 220/236 吨满载，先完成下方载重兼容性改造。 [车辆初始化参考](../fleet_outage_001/anylogic_fleet_outage.java.txt#L28) |
| **Main → Functions → `init_network`、`addBidirectionalPath`**，项目树中的 **Main 道路几何与相应节点** | 原 16 边，双向对象由两个 `new PathSegment(...)` 建立，构造器末参为按方向读取的限速。 | 保持边两端及排序。用可对应的现场路段几何驱动物理行驶长度，限速按方向传入构造器。先核对路线逐段求和、实际行驶时间和 RPC `road.*.length_scaled_f32`。`length_scaled_f32 × 10000` 的长短须与车辆实际行驶一致。只有整趟运距时先校准路线级距离和时长，逐段值保留来源或估计标记。 [双向构造器](../generalization_001/Main.addBidirectionalPath.after.java) · [运行场景快照](../../../offline_data_readiness_001/runtime_scenario_v1/source_snapshot.json) |
| **Main → Functions → `init1` → d0/d1 `initDumpSite(...)`**，**DumpSite → Variables → `crusherRateTph`、`crusherBufferCapacityTons`**，**DumpSite → Functions → `initDumpSite`、`dumpServiceBoundaryMinutes`** | d0 为 `Crusher`、d1 为 `Dump`，标称破碎能力 2,500 t/h、缓冲 500 t。运行服务时间走当前卸载函数和资源。 | 确认 d0 是当前案例计量的矿石破碎站，d1 是直排点。按现场班期设置服务槽、实际卸载服务、活跃处理能力和可用缓冲，核对 RPC 参数、实际资源并发、排队和整车能否入仓。保留卸载到达与破碎兑现的两个时钟。 [场景导出中的卸点字段](../../../offline_data_readiness_001/runtime_scenario_v1/buildRuntimeScenarioComponents.after.java#L83) |
| **Main → Parameters → `fleetOreTarget`、`fleetWasteTarget`、`wasteDispatchStopTonsShift`**，**Main → Agent actions → On startup** | 历史演示按班写入 16,000 吨矿、4,000 吨废，`targetOreTons2h` 按四分之一设置。标称废石停派线为 6,000 吨。 | 用已确认的同作业范围、同计量环节的班目标和停派规则覆盖。现场期间二小时吨位若有实际配额，先决定是否需要不同窗目标。达产线、停派线和最后一车允许量分别核对。品位目标若变更，贯通 Java `targetGrade/gradeLowerBound/gradeUpperBound`、RPC 合同、基线规划约束、评测与代价定义。旧 `configuration_from_components` 固定标称吨位，场景活动另用逐班参数准入。 [目标和启动位置](../fleet_outage_resilience_001/manual_anylogic.md#L35) · [当前 RPC 准入](../../../../python/reward_config.py#L65) · [逐班目标范例](../fleet_outage_resilience_001/resilience_rpc.py#L208) |
| **Main → Agent actions → On startup → 紧接 `init1();`** | 退场分支检查时钟为零、期初料仓和矿石／金属计数为空。 | 第一轮取能够证明期初为空的实际开班，或明确只评估能够独立计量的段。若要复现连续生产班，另做开班 FIFO、外部来料和在途状态的初始化，按账源构建队列与资源状态，保存实际初值。用交接班前后吨量和金属量检验状态。 [空仓开班检查](../fleet_outage_resilience_001/manual_anylogic.md#L35) |

现场班长、窗口制度若与 8 小时、四个 2 小时窗一致，可采用现行时钟。其余工况先明确统计窗口和交接班边界。需要变更决策窗数量或长度时，配置、时钟、窗结算与评测合同一起升级。C128 的名义速度、车型卸载均值及若干尺度是冻结假设。重新定标前对照实测循环时间与模型名义 ETA，并记录影响。 [C128 冻结配置与 ETA](../../../../grade_commitment_ppo_v1/features.py#L142)

## 三维路网批量导入

首版在 `baseline2` 副本中，使用同一份现场中心线生成 AnyLogic 的 `PointNode`／`Path` 和调度用的有向 `PathSegment` 数据。约定导入器完成校验后按稳定 ID 更新对象。样本文件格式及 AnyLogic 版本确定后，再写对应的解析器和可粘贴接线代码。

### 输入与导入方式

- 输入需含**路段编号、起终点节点编号、按行驶顺序排列的 XYZ 中心线点、通行方向**，以及装载点、粗破和直排点对应的路网节点。记录数据日期、平面坐标系、XY／Z 单位和高程基准。道路限速、空重车通行限制、宽度和共用车道信息有则保留。
- 有 `SHP PolyLineZ`、`GeoPackage LineStringZ` 或三维 CAD 中心线时，提取上述字段。标准中间表可用 `nodes.csv`（`node_id,x_m,y_m,z_m,role`）、`edges.csv`（`edge_id,from_id,to_id,direction` 等）与 `vertices.csv`（`edge_id,seq,x_m,y_m,z_m`）。只有地形网格或点云时，需另取中心线与节点。OBJ、GLB 等可通过 `Presentation → 3D Object` 加载用作地形展示。 [AnyLogic 三维对象导入](https://www.anylogic.help/anylogic/3d/import-3d.html)
- **编辑器中一键生成。** AnyLogic 8.9.7+ 的 Professional／University Researcher 版可在副本中启动 **Start graphical editor API connector**，由设计期 Python 脚本批量更新 `Main` 所在层的节点、路径折点和端点连接，再保存并构建。路径编辑器的点表使用相对起点的 XYZ。优先原位更新节点对象，保留名称和已有设备引用。若新建替换对象，重绑路径端点及设备位置。 [设计期 API 和版本要求](https://www.anylogic.help/advanced/code/dt-api.html) · [Path 三维点表](https://www.anylogic.help/markup/path.html)
- **随启动选文件。** 如需按案例换路网，可在模型初始化阶段用 Java 读取同一份中间表，将中心线逐段建立为带 XYZ 的 `MarkupSegmentLine`，连接节点，组成 `Network` 后初始化 `Level`。当前模型需先将固定节点引用接到显式节点表。导入、初始化和校验全部成功后才创建或激活在路网中运行的车辆。 [AnyLogic 按代码创建网络](https://anylogic.help/markup/create-network-by-code.html)
- `GIS Map → Shapefiles → Convert Shapefile to Space Markup → Roads` 可在测试副本批量生成 AnyLogic Road Traffic 路网。官方示例的转换结果默认双向且需要另设限速。本项目使用 `Path` 和自有路段逻辑，若采用 GIS 转换，还需验证高程、实际方向、连通性及自有路段映射。 [AnyLogic Shapefile 转路网](https://anylogic.help/markup/converting-roads.html)

### 现有模型接线位置

| 位置 | 现有内容 | 导入后与验收 |
|---|---|---|
| **Main → `PointNode`、`Path` 所在层**，**Main → Agent actions → On startup → `init1()` 前** | 装卸点、停车点及移动网络来自编辑器固定对象。 | 保留兼容模式的既有节点名，在编辑器生成或替换折线几何。运行时方案在 `init1()` 前创建和绑定节点。用至少两个已知位置和高程核对坐标、方向和缩放。经纬度先转到适用的米制投影，再换算为模型局部坐标。不同高程的投影交叉口按现场节点关系连通。 [启动顺序](../../../../manual_alp/01_Main_Main_StartupCode.after.txt#L61) |
| **Main → Functions → `initRoadPointNodeMap`、`init_network`、`init_routeNodeMap`** | 八组双向连接写入 `segmentMap`，OD 路线按固定节点序列查路。 | 使用同一批路段 ID 和实测方向构建路段及 OD 路线。进入 `Truck → MoveSegment / MoveSegment1` 的目标节点与实际几何段一致。若存在平行道路或多条可替代路径，需为路段使用独立 ID 并显式选择几何路径。用空车去装点、重车去粗破及直排分别核对逐段经过顺序。 [旧路网与寻路源码快照](../../../offline_data_readiness_001/runtime_scenario_v1/source_snapshot.json) · [重车移动动作](../../../../manual_alp/13_Truck_MoveSegment1_EntryAction.after.txt#L4) |
| **Main → Functions → `getNodesPosition`、`initDynamicSpeedModel`、`obsInit`** | 距离函数以 XY 端点请求网络路径，速度和观测按有向索引初始化。 | 从导入折线核算每段三维弧长和有向坡度。检查同一长度口径实际用于车辆运动、剩余距离和 RPC。若行驶时间需体现坡度，按空／重车和方向从实绩标定速度或时长。核对 `config.runtime_scene.road.*.length_scaled_f32`、限速、首车沿坡运动及实际耗时。 [运行字段导出](../../../offline_data_readiness_001/runtime_scenario_v1/buildRuntimeScenarioComponents.after.java#L97) |

兼容模式保留当前 BG128 的九个道路节点及 16 条有向边端点和顺序，只将道路中无分流、无独立排队作用的折点作为路径顶点。新分流、单行连接、关键共享路段或装卸点无法对应时，建立新的路网与观测合同，并重新验证策略。用先前的现场轨迹对照通过距离和时间，验收后将路网配置及其指纹纳入拟开展活动的 RPC 预期。 [BG128 边端点约束](../../../../business_graph_summary_v1/graph.py#L150)

## 双去向和大载重的扩展挂点

| 精确位置 | 修改前 | 双去向场景中的修改与验证 |
|---|---|---|
| **Truck → Statechart → `NeedDispatchToDumpSite` → Entry action**，**Truck → Functions → `onLoadingFinished`** | 装完后按 `payloadIsOre` 遍历 `dumpSites`，匹配 `acceptsOre/acceptsWaste`，无策略 RPC。 | 在载荷确认后、计算新行驶路线之前，取得同一载荷的合法卸点。唯一合法卸点按现场规则提交。双合法时调用独立的卸点选择规则或新策略，提交实际 `targetDumpSiteId`、目的节点和在途承诺。保留原装载结束记账。验证重复选择、无合法卸点、两边都合法、路被封和班末同时事件。 [现有动作挂点](../../constrained_wait_006/network_architecture_v1_001/engineering_003/alp_audit.md#L29) |
| **Main → Functions → `selectNextLoadSiteAI`、`exchangeDispatch`**，**`proto-src/src/main/proto/dispatch.proto`** | `DECISION` 的单个整数 action 被解释为装点 0–3。 | 空车请求继续返回装点。装后卸点请求以版本化决策阶段、卸点合法掩码和已知的载荷信息进入独立协议或端点，回执携带阶段及最终去向。到达已关闭卸点前的处置使用明确业务规则。验证阶段错配、非法去向和重试时重复预约均能被识别。 [现有请求与回执](../../../../proto-src/src/main/proto/dispatch.proto#L73) |
| **DumpSite → Functions → `completeUnloading`、`receiveCrusherBatch`、`advanceCrusherTo`**，**Main → Functions → `auditMaterialBalances`、`getWasteTaskCommittedTons`** | 矿石只允许进入 d0，废石才可交付 d1。既有守恒按这两种固定物料流编写。 | 按现场材料类别和已提交的实际去向分别确认合法交付，保留物料身份和含铁量。入矿石破碎的车载只入仓一次。直排按类别进入直排账，粉碎后的岩石若由胶带转运，还需明确其中的转运环节与最终交付时钟。核对装载、在途、预约、卸达、库存和破碎产量，逐类对吨量与金属量。 [卸载分支的历史源摘录](../../../../reports/boundary_alp_review_002/synthetic_current_alp/extracted_java/DumpSite.java#L150) |
| **Main → Functions → `collectPhysicalObservation`、`buildRewardConfigComponents`**，**Python → `observation_schema.json`、图编码、C128、准入和评价** | `actionTons[0:4]` 按 `/193` 归一化并限制于 `[0,1]`，图和 C128 用 `isOre` 预测单个卸点。 | 大载重场景建立版本化吨量合同，核对 220/236 吨候选值、Java/Python 反算、真实载荷与队列入仓条件，以及最大末车吨量。合法双去向场景同时编码已知的可达卸点和实际目的地，逐候选核对 ETA、FIFO 兑现和模型输出。新版检查点按其观测与决策合同独立验收。 [动作吨量合同](../../../../python/observation_schema.json#L234) · [观测域实现](../../../../python/observation_contract.py#L29) · [候选卸点与在途校验](../../../../business_graph_summary_v1/graph.py#L154) · [C128 候选](../../../../grade_commitment_ppo_v1/features.py#L361) |

原模型的行驶、跟车、装卸状态机、统计与破碎 FIFO 可作为骨架。现场如果同时有生产粗破、岩石短倒破碎和直排，三处需共享资源时要作为独立卸点建模。只有两条互斥的预定方案时，可分别做情景，并说明各自的运输和结算范围。新版策略控制卸点时，已有检查点用作可比合同下的固定去向参照或初始化来源。

## 首班验收顺序

1. 离线确认作业单元、吨位可行性、品位目标、实际去向和决策时间。按新活动生成独立 `recording_id`、配置字节 SHA256、Python 输出目录、AnyLogic 事件目录和完整 run／环境 seed／Actor 采样 seed 清单，并用当前累加预算入口检查身份冲突。准备与实际登记按项目现行规则执行。
2. 在独立副本 **Save → Build**。首个实际 START、派单和 END 核对 RPC 字段与冻结配置、schema、动作/阶段、实际路由、循环时间和吨量。失败身份与回执保留。逐请求追踪、逐 transition 和快照默认关闭。只在定位某个出错环节时使用最少的专项记录。
3. 自然班末核对矿石和金属守恒，四窗输出按处理时钟重算。固定去向场景检查每种物料到达正确终点。双去向场景至少覆盖唯一合法、两者合法、路线失效，以及已有预约或在途物料的班末结算。正式比较再用相同的外生输入进行多策略配对。
