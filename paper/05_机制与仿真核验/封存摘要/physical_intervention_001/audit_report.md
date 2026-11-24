# BG128 业务关系原始观测干预：A/B/C

2026-11-18，Asia/Shanghai。**三个独立原始观测分支均已完成：raw→事实图→21维数量→Token/H0/H1/H2，以及重新认证的C128→Actor候选表示→四个Logits。最终读出接口已经接通，本轮直接使用正式实现。独立426项核验全部通过。**

模型仍为固定随机初始化 `pair_seed=2026111801`，Actor/Critic参数独立，各场景共用同一角色参数。没有加载训练权重、优化参数、启动RPC或AnyLogic，没有新班次和预算登记。所有输出仅用于数值敏感性诊断，不表示已学到正确业务语义，也不验证动作偏好、价值预测或调度收益。

## 共同起点与物理约束边界

基点为此前认证的历史 **T1**：run_index `9810998`、state_index `62`、仿真时间 `9717.867211564166秒`、Ego **Truck0**。A/B/C都从这同一份原始float32[1519]独立复制修改，不叠加，不引入未来观测。原始基点与HDF5逐元素一致。

本轮落实了可观察字段的自洽：物料守恒、预留/已载互斥、列表/服务计数、路线方向/长度/终点、动作mask/吨位、FIFO与buffer恒等式等。**尚未证明这些分支对应完整AnyLogic内部状态和事件队列的可达决策。** A是源码兼容的事件内部边界，B是当前系统未暴露为PPO动作的外生改派，C是原运行配置之外的离线参数阶跃。未将它们冒充历史实况或已执行的仿真事件。

|分支|原始干预与联动|改动字段|图记录数|C128变化项|
|---|---|---:|---:|---:|
|BASE|历史T1|0|88|0|
|A|Truck1在Load0完成当前91.7吨装载，已载、目标Dump0；同步库存、预留、完成量及服务资源|20|86|23|
|B|Truck8的193吨待装预留从Load0转至Load1；同步预留、incoming、available和未行驶路线尾|12|88|10|
|C|Dump0当前处理速率2500→1250吨/小时|1|88|8|

所有分支仍为9辆车、19个节点、16条有向道路，动作mask均为四个合法动作，Ego候选吨位均约91.7吨。25种关系的Schema不变，不宣称样本中出现了全部类型。原始输入、逐字段修改及检查见 [scenarios.json](scenarios.json)、[raw_changes.csv](raw_changes.csv)、[physical_checks.json](physical_checks.json)。精确字段位置、前后内容、Java依据和舍入容差见 [修改合同](../../../../change_plans/token_attention_001/business_graph_summary_009/physical_intervention_001/observation_intervention_contract.md)。

## A：增加一辆已载且目标Dump0的车辆

固定9车Schema下，“增加一辆已载车辆”通过现有Truck1完成装载实现，未新增第10辆车。其BASE已给出当前装载物料 `loadingGradeInferred=1`、std=0、mean=`0.37571972608566284`；载荷品位直接用此当前可知值，没有借用后续真实品位。

Truck1清空装载预留，载荷变为91.7吨、gradeKnown=1、dumpTarget=0；Load0的remaining/reserved各减少91.7吨，produced增加91.7吨，移除当前唯一服务成员并释放资源。可用吨位在代数上不变，保留原始float32字节，避免1ULP伪变化。allocationCursor已在进入服务时推进，本次不再推进。Dump0的buffer预留仅在开始卸载时增加，本次保持。

为保留车辆位置和已有运动历史，快照边界取 `onLoadingFinished` 之后、下一行驶状态进入之前；此时按源码门控phase由4落到11（OTHER）。Truck1仍在Load0，**不能称其已经在道路上行驶**。这个同步事件内部暂态未证明能被生产RPC实际采到。

事实图移除 Truck1↔Load0 的reserved_load和service两对边，增加Truck1↔Dump0的loaded_dump一对边：新增2条、移除4条。Dump0已载目标由5辆、约771.7吨变为6辆、约863.4吨。原服务、预留、已载三个口径分别统计，不相加为重复物料。

`Truck1 H0→Dump0 H1→Load0/1/2 H2` 仍具备完整通道；实际前向中Dump0和对应Load均响应。与此同时，Dump0的明确数量已先进入H0、Load0自身状态及服务关系也改变，所以完整响应不能独占归因于这一条两跳路径。Load1/2在H1就有小幅变化，正体现这种多入口。

## B：改变一辆待装车辆的loadTarget

Truck8为空载TO_LOAD、当前在道路7（Node2→Node3），尚未进入任何Load的queue/seize/service列表。Load0和Load1均为矿石。将193吨预留原子移转，同时把有序剩余路线 `[7,8]` 改为 `[7,12,13]`，保留当前道路和段内位置，只改未来尾段及剩余总路程。

Load0的reserved/incoming减少193吨/1车，Load1增加对应数量，available反向更新；remaining、produced、allocationCursor、既往已知品位和FIFO保持。新路线方向连续、无重复段且抵达Load1。仅改变loadTarget而保留旧路由和站点账本是不合格干预，本轮未采用该做法。

图中Truck8↔Load0的reserved_load变为Truck8↔Load1，共移除2条、新增2条，其他边不变。总预留车数/吨位守恒，旧、新Load的承诺表示及候选表示均改变。Load1的commitment组由缺失变为存在，与其较大H1/H2响应同时发生；本轮未将组存在性、属性和状态变化分别量化归因。

## C：改变Dump处理状态

仅将原始索引1304的 `dump.0.crusherRate` 从1改为0.5，对应当前处理能力2500→1250吨/小时。采用严格正值，不用0伪造源码不允许的停机参数。

同一时刻的库存、FIFO、buffer、预留、服务和历史累计处理量均不变；图的88条边及其属性、21维数量也全部不变。站点内容从raw重新编码，Dump0的H0因此变化，关联Load在H1/H2响应。**无拓扑变化不等于无图传播响应。**

重新计算的C128改变8项：`crusher.rate`、当前capacity/free_capacity/required_grade_lower/required_grade_upper/required_interval_intersects_ore_support、下一窗口capacity/free_capacity。当前名义需求区间与矿石支持区间的相交标志由1变0；这是冻结C128中的当前状态估计，并非后续真实窗口结果。四个action10D块完全不变；原定义的候选名义ETA不读crusherRate，未手工改写ETA。

之前单独接口probe使用T0，C128变化为7项；**主实验固定T1，结果为8项**，不混用两个基准。完整128维数值与差异见 [c128_features.csv](c128_features.csv)。

## H0/H1/H2与候选表示比较

所有变化均为干预相对BASE的L2距离，单位是隐藏向量坐标，不是吨位或业务贡献。全层比较见 [编码器响应图](encoder_response.png)，该图每个面板独立缩放；完整19节点各64维见 [node_representations.csv](node_representations.csv) 与 [analysis_tensors.npz](analysis_tensors.npz)。

Actor的逐层Load响应：

|分支|Load|ΔH0|ΔH1|ΔH2|
|---|---|---:|---:|---:|
|A|Load0|0.186673|1.303348|1.621764|
|A|Load1|0|0.013834|0.020643|
|A|Load2|0|0.013838|0.020596|
|A|Load3|0|0|0.000252|
|B|Load0|0.093637|0.094417|0.117684|
|B|Load1|0.095576|1.092841|1.432969|
|B|Load2|0|0|0.000145|
|B|Load3|0|0|0.000270|
|C|Load0|0|0.003717|0.004286|
|C|Load1|0|0.003916|0.004919|
|C|Load2|0|0.004054|0.005009|
|C|Load3|0|0|0.000291|

Critic使用自己的独立编码器，也已保存H0/H1/H2；H2结果如下，H0/H1详见同一响应图和 [逐层变化CSV](representation_changes.csv)。

|分支|Load0 ΔH2|Load1 ΔH2|Load2 ΔH2|Load3 ΔH2|
|---|---:|---:|---:|---:|
|A|1.564186|0.022250|0.022286|0.000268|
|B|0.118230|1.686560|0.000146|0.000262|
|C|0.003504|0.003954|0.004118|0.000213|

Actor最终候选64D表示在每个动作上直接读取Ego/候选Load/兼容Dump、对应数量、mask/tons以及候选C128编码。公共context另融合类型池化、Ego、Global、类型数量和C128公共部分。两者通过乘性交互评分，接口已实际运行。

|分支|Load0候选 ΔL2|Load1候选 ΔL2|Load2候选 ΔL2|Load3候选 ΔL2|
|---|---:|---:|---:|---:|
|A|0.122441|0.026813|0.026622|0.015526|
|B|0.019805|0.172833|0.003909|0.000106|
|C|0.003421|0.003476|0.003468|0.000235|

[候选与Logit响应图](actor_readout_response.png)、[候选各输入块变化](candidate_changes.csv)、[候选64D明细](actor_candidates.csv) 均已交付。块范数仅为输入敏感性，不可相加成贡献比例。最终Critic标量也保存用于接口核验，随机初始化数值不作为估值精度结果。

## 四个动作Logits

本例全部动作合法，所以unmasked与masked logits相同：

|场景|Load0|Load1|Load2|Load3|
|---|---:|---:|---:|---:|
|BASE|−0.037963|−0.034621|−0.036097|−0.035499|
|A|−0.037100|−0.031155|−0.032604|−0.031613|
|B|−0.038645|−0.037270|−0.036203|−0.035305|
|C|−0.035866|−0.032407|−0.033870|−0.033615|

完整Mask、候选吨位、概率和差分见 [actor_logits.csv](actor_logits.csv)。[任意两动作分数差](logit_pairwise_differences.csv) 以及图中减去四动作平均变化后的Logit，用于排除Softmax不敏感的公共平移。**有响应只能证明接口与信息通道工作；正负方向不能说明网络会正确回避拥堵或合理平衡品位/等待。**

完整原始干预同时改变图编码、明确数量、直接Dump读出与C128等通道。因此不能要求Load3的总Logit响应为零，也不能把总评分变化全部归于两跳GAT。

## 为什么C也会影响废石Load3？

C中Load0..3的H0全部不变；但已载Truck2/3/5/6/7在TokenEncoder中读取Dump目标引用，连同Dump0构成发生变化的H0来源。实际图提供以下到Load3的两跳通道：

```text
Truck3 → Junction3 → Load3
     truck_segment_to   road

Truck3 → Truck4 → Load3
     truck_predecessor_reverse   truck_at_node / reserved_load / service_reverse

Truck5 → Truck4 → Load3
     truck_predecessor           truck_at_node / reserved_load / service_reverse
```

每种角色共7条关系路径，详见 [路径CSV](cross_material_paths_C.csv) 与 [内容源/路径JSON](C_Load3_content_paths.json)。`truck_predecessor` 是**首次派车前驱**，不是当前道路上的运动前车。共享道路与历史派车关系能够传播其他物料相关内容；这不代表矿石承诺被计入废石，也不证明这种耦合有利。路径重叠且经过非线性，不能用各路径范数相加分摊总响应。

这也解释了为何本轮与此前仅替换Dump0 H1的隔离实验不同：完整raw重建会先改变多个H0来源；本轮没有任何隐藏张量替换。后续若研究关系选择性，首次派车前驱和目标引用重复传播应纳入训练后关系消融，本轮保持正式架构不变。

## 完整性与交付

[独立核验报告](independent_review.md) 的426项检查包括：HDF基点、20/12/1字段白名单、独立21D数量oracle、物料/路线/动作sidecar、raw重构图、H0/H1/H2、重算C128/新证书、候选输入与64D、显式交互评分、四logits/概率/Critic值；普通与diagnostic前向逐元素一致，参数和buffer摘要前后一致。旧raw证书用于新输入时会被拒绝。

原始数据、此前构图及编码器审计保持原样，正式模型和受保护PPO核心未修改。核心SHA仍为 `8e4cfa31c622f5e33756e304eea3f7526547db72ae93f7d9f00808845d61ef96`。本轮未变更Reward、未计算新实际结算结果，也未用合成分支拟合Actor或Critic。

[manifest.json](manifest.json) 封存脚本、原始输入、当前本地语义依据、正式实现和全部证据签名。[复现入口](../../../../change_plans/token_attention_001/business_graph_summary_009/physical_intervention_001/README.md) 仅离线构建/Forward/检查/绘图，不启动训练或仿真。
