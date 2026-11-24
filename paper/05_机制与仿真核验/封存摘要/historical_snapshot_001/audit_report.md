# BG128 历史32班与相邻决策快照离线构图审计

2026-11-18，Asia/Shanghai。结论：**32班、5841个有效决策全部成功构图；选中的同班相邻T0/T1均通过19节点、16条道路、原始C128及数量独立核验。当前008成本和MC与冻结目标逐元素相同。**

本次仅读取历史数据、构图、计算确定性特征、绘图和重建已有成本。模型实例化、模型Forward、参数更新、RPC、AnyLogic启动、新班次、预算登记均为0。图是BG009正式输入结构的重构，历史动作来自旧TQ128策略，不是BG128训练结果或注意力图。

## 1. 数据、Schema与相邻性

历史批次为`s01_TQ128/train_u0016`，共32班、5873行N+1原始状态、5841条transition/有效决策。每班一个真实终态，构图覆盖全部非终态决策。原始1519D float32观测、typed transition时间/奖励、window ledger、sealed物料事件及冻结训练目标齐全，没有阻止正式构图的缺失字段。

Schema实际位置为`change_plans/token_attention_001/business_graph_summary_009/schema.json`；包内没有第二份独立schema。实际复用`business_graph_summary_v1.graph.build_graph`和原C128 `build_features`，未复制重定义正式构图规则。地图严格核对16条有向道路端点顺序，Load地理节点5/6/7/8、Dump0/1、中间节点2/3/4。

选中源：`runs/single-critic-task-query-formal-seed1-20261117-002/arms/s01_TQ128/train_u0016/episodes/run_9810998/samples.h5`。

|身份|T0|T1|
|---|---|---|
|run_index|9810998|9810998|
|原始state索引 / decision sequence|61 / 61|62 / 62|
|精确RPC动作时刻（秒）|9603.707937186524|9717.867211564166|
|float32观测解码时间（秒）|9603.708171844482|9717.867279052734|
|决策车辆|Truck8|Truck0|
|该时刻历史选择|Load0|Load2|

间隔114.159274377642秒，源H5行严格相邻，属于同一个episode、同一个recording和环境seed `220829429904417`。精确时刻取RPC float64元数据，输入仍保留原float32字节，未用元数据修补输入。快照筛选是事后展示选择，不是随机样本性能统计。

T0与T1分别由各自原始观测独立构图；T1未补充T0。PNG的变化颜色和下方事后对照属于展示元数据，不是事实图、数量或C128的额外输入。

## 2. 两张快照与变化图

三张图使用相同的19节点固定示意布局，位置不代表真实坐标。道路箭头和业务消息方向可见；正/逆消息对在PNG可合为双箭头，CSV始终保留每条有向记录和多重性。颜色区分四组关系；快照图外沿与差异图线色区分新增、消失、属性变化。候选仅画在独立读取卡片中，不加入事实图。

### T0

![T0业务图](snapshot_T0.png)

### T1

![T1业务图](snapshot_T1.png)

### 关系变化

![相邻关系变化](snapshot_diff.png)

两图各88条有向记录：16道路＋72其他关系。Schema词表为25种有向类型（其中1种road、24种其他类型），本对出现19种；32班出现21种。未出现的`admission, admission_reverse, queue, queue_reverse`反映相应原始列表为空，不是缺字段，不补造关系。

按关系类型、端点、有序槽位及实例序号对齐，多重图差异为：55不变、28消失、28新增、5属性变化；属性变化按原float32字段严格比较。一个实际业务状态变化可能同时改变目标、定位、服务及正/逆边，不能把边数当独立事件数。

## 3. 本对快照的业务变化

|车辆|T0事实|T1事实|
|---|---|---|
|Truck3|Load0正在装载，预留147t，载荷品位未观测|已载147t矿石，品位约0.383127，目标Dump0|
|Truck5|Load2正在装载，预留147t，载荷品位未观测|已载147t矿石，品位约0.145767，目标Dump0|
|Truck7|Load1正在装载，预留193t，载荷品位未观测|已载193t矿石，品位约0.296477，目标Dump0|
|Truck0|Dump0卸载服务中，仍载91.7t|已卸空，成为当前Ego；没有新派发事实|
|Truck8|当前Ego，无loadTarget|历史动作派Load0，形成193t实际预留，开始空车运输|

Truck3/5/7合计487t由待装承诺转为已载事实，Truck0卸出91.7t，因此Dump0载货目标吨位净增约395.3t，车辆数3→5。`inbound_loaded`包含到站但仍载货车辆，不仅指正在道路上行驶者。装载后的历史Load来源不在T1的loadTarget内；可用两时刻对照解释变化，但没有回填T1事实边。

|节点|T0待装预留：车数/吨位|T1待装预留：车数/吨位|T0载货目标：车数/吨位|T1载货目标：车数/吨位|
|---|---:|---:|---:|---:|
|Load0|2 / 238.7t|2 / 284.7t|0 / 0.0t|0 / 0.0t|
|Load1|1 / 193.0t|0 / 0.0t|0 / 0.0t|0 / 0.0t|
|Load2|1 / 147.0t|0 / 0.0t|0 / 0.0t|0 / 0.0t|
|Load3|1 / 147.0t|1 / 147.0t|0 / 0.0t|0 / 0.0t|
|Dump0|0 / 0.0t|0 / 0.0t|3 / 376.4t|5 / 771.7t|
|Dump1|0 / 0.0t|0 / 0.0t|0 / 0.0t|0 / 0.0t|

服务列表变化按原观测保留：Load0服务Truck3→Truck1，Load1/2服务清空，Load3仍服务Truck4；Dump0服务Truck0→空，容量等待仍为Truck6、Truck2。`truck_predecessor`只表示首次派车前驱，不是当前运动前车。

## 4. 两层传播路径

对四个候选Load与两个Dump，逐条列出直接入边和精确两条有向边的walk，保留关系类型、方向、全部14边属性、原始实体与变化状态。包含回返walk并标记`return_to_source`，没有把回返当作新增事实。见[two_hop_paths.csv](two_hop_paths.csv)。

|快照|目标|第一层直接入边记录|精确两边walk|其中Truck→Dump→Load|
|---|---|---:|---:|---:|
|T0|Load0|7|42|3|
|T0|Load1|5|32|3|
|T0|Load2|5|32|3|
|T0|Load3|5|23|0|
|T0|Dump0|14|60|0|
|T0|Dump1|2|11|0|
|T1|Load0|7|44|5|
|T1|Load1|3|24|5|
|T1|Load2|3|24|5|
|T1|Load3|5|24|0|
|T1|Dump0|14|59|0|
|T1|Dump1|2|12|0|

核心路径变化：T0为`Truck0/2/6 → Dump0 → Load0/1/2`；T1为`Truck2/3/5/6/7 → Dump0 → Load0/1/2`。每个矿石Load由3条变成5条此类通道，新增Truck3/5/7并移除已卸空的Truck0。关系分别是`loaded_dump`和`compatible_material_reverse`。废石Load3只兼容Dump1，不伪造Dump0→Load3物料边。

这说明已载车辆及对应卸点条件**在拓扑上可经两层进入矿石候选Load**。最终Actor直接读取Load/Dump，不要求再传回Ego。此处没有运行随机网络，也没有计算注意力权重；拓扑通道存在不证明模型已经学会利用它。原初始编码本身还含列表/路径/数量信息，因此两层仅约束GNN新增组合范围。道路边是信息关系，关闭状态并不自动删除边，两跳不等价于物理通行路线。

缺失的历史载荷Load来源保持未知，未生成`historical_load_origin`事实；完整缺失清单见[path_summary.json](path_summary.json)。Ego的四个候选边是按设计不存在的读取索引，不是待补齐的派车事实。

## 5. 当前Reward/成本对照与可见时间

实际RPC reward与当前008训练成本定义不同，分别保留。原RPC reward为`wait+production+waste`，本批wait系数`4e-5`、终态矿/废缺口权重4/4，grade相关RPC项为0；窗口品位/旧grade_reward独立记录在window ledger与物料事件中。

当前008成本严格复用原函数：四窗`min(max(abs(G-.27)-.004,0)/.006,2)`、空窗2；真实班末矿石16000/废石4000吨缺口权重8/8；每区间车队等待`.2*delta_wait/28800`。6000吨是废石停派阈值。没有把RPC reward直接取负替代训练成本。

|此刻已收到上一动作结果|T0|T1|
|---|---:|---:|
|对应transition索引|60|61|
|RPC reward|-0.013284320152097|-0.009132741950211|
|同区间008成本|0.002306305581961|0.001585545477467|
|累计已结算品位成本|0.452694192911265|0.452694192911265|
|已结算窗口（0起）|[0]|[0]|
|尚未结算窗口（0起）|[1, 2, 3]|[1, 2, 3]|

**T0动作Truck8→Load0，到T1收到结果**：车队等待增加228.318548755285车辆秒；原RPC reward为-0.009132741950211；当前008成本为0.001585545477467，全部等待，品位和生产成本均0。T0时未知该区间结果，T1时才收到。T1选择的后续结果另标future audit-only。

该区间物料事件为PROCESS79.277273873t、Truck0 UNLOAD91.7t，没有WINDOW_CLOSE。487t装完、91.7t卸入缓冲与79.277t实际破碎分别是不同业务量，不能相互替代。当前Reward可以与阶段变化按时间区间对照，但没有对“装完487t”即时给出独立品位/达产收益；品位要等窗口结算，产量缺口等班末结算。车队等待也不能仅凭两张快照因果分摊给某辆车或某条边。

32班5841条成本和MC与冻结`updates/0016/training_targets.h5`逐元素完全相同，最大误差0；RPC分项加和、等待守恒、raw/material状态最大误差0。见[reward_audit.csv](reward_audit.csv)、[reward_details.json](reward_details.json)、[reward_findings.md](reward_findings.md)。

选中整班后验成本2.631603848978＝品位2.452694192911＋生产0.000000000000＋等待0.178909656066；这些全班和未来结算结果仅用于audit，不进入T0/T1输入。未结算窗口的当时最终成本留空，没有用0冒充已结算结果。

## 6. 数量、候选、输入与无法恢复的信息

21维数量逐项由原字段独立重建，32班最大绝对误差为4.3556236e-05（float32缩放/还原误差），无重复承诺计数。目标、服务、排队可保留多重语义，但吨位只按符合条件的Truck目标计一次。service实例与unique数分开；本批未发现同一有向类型/端点/属性的重复实例，不据此删除Schema允许的多重性。

候选吨位减半的干预不改变事实边、边属性、数量、候选Dump绑定或Ego索引。C128由各快照独立计算且保留原字段顺序；Actor使用88＋4×10，Critic使用完整128，事实输入可只读共享，神经网络参数仍按正式结构独立，但本次没有实例化它们。

`node_features.csv`输出可审计的原始节点字段/引用及Junction确定性描述，**不输出随机神经64D嵌入**。原始道路和动作sidecar仍完整保存在`snapshots.json`及对应边/候选信息中。`count_features.csv`为2×19×21＝798行，包含raw、scaled和独立重建值；`c128_diff.csv`保留全部128维。

无法从各自历史快照直接恢复：已载Truck已清空的历史装载来源；尚未实际观测的载荷品位；两决策间未记录的精确装载完成时刻；未来实际ETA；未结算窗口的最终品位/成本及班末产量；单车/单边对聚合Reward的因果贡献。完整历史中的后续事件可用于审计，但没有补进任何历史输入。

## 7. 验收结论

|报告问题|结论|
|---|---|
|两个快照是否成功构建？|是，各自原始观测独立构图，源行61→62严格相邻。|
|19节点、16道路、25关系词表符合Schema？|是；每图88条记录，本对出现19种类型，不能要求每图都出现所有25种。|
|是否重复计数？|独立21维oracle通过；没有将目标/服务/候选重复加入承诺吨位。|
|是否候选污染事实图？|否；四候选仅直接读取，候选吨位干预不改变事实图。|
|是否缺失补猜？|否；历史Load来源/未知品位保留未知，未从另一时刻回填。|
|哪些变化可两层到Load？|新已载Truck3/5/7通过Dump0可达Load0/1/2；Truck0卸空相应通道消失。|
|Reward能否对应承诺变化？|可按区间核对；本区间只有等待成本，没有窗口结算，不构成单车/单边信用分配。|
|哪些信息无法恢复？|上述历史来源、精确装载时刻、未来实际结果及因果分项。|

独立复核见[independent_review.md](independent_review.md)。所有历史输入、正式源码和受保护PPO SHA前后相同。核心SHA：`8e4cfa31c622f5e33756e304eea3f7526547db72ae93f7d9f00808845d61ef96`。本次全部输出索引及哈希见[manifest.json](manifest.json)。

## 8. 文件索引

|交付|内容|
|---|---|
|[snapshot_T0.png](snapshot_T0.png)、[snapshot_T1.png](snapshot_T1.png)|同布局19节点快照|
|[snapshot_diff.png](snapshot_diff.png)|新增/消失/属性变化对照|
|[graph_edges_T0.csv](graph_edges_T0.csv)、[graph_edges_T1.csv](graph_edges_T1.csv)|完整有向多重边与14属性|
|[two_hop_paths.csv](two_hop_paths.csv)|直接入边及全部两边walk，459行|
|[node_features.csv](node_features.csv)|原始物理节点字段与确定性描述，2886行|
|[count_features.csv](count_features.csv)|21维raw/scaled/oracle，798行|
|[c128_diff.csv](c128_diff.csv)|完整128维变化|
|[reward_audit.csv](reward_audit.csv)|实际RPC、当前成本、已知/未结算/后验分离|
|[manifest.json](manifest.json)|身份、范围、源与交付哈希|

源脚本及复现步骤见[实现目录说明](../../../../change_plans/token_attention_001/business_graph_summary_009/historical_snapshot_001/README.md)。
