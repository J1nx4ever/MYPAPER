# BG128 相邻历史快照：编码器数值传播审计

2026-11-18，Asia/Shanghai。**两张原始快照已送入正式 BG128 图编码器，Actor/Critic 的 H0、H1、H2 与逐边消息均已保存。Truck→Dump→Load 的两层数值通道通过核验。**

本次使用固定随机初始化 `pair_seed=2026111801`，未加载训练权重。同一角色的 T0/T1 使用完全相同参数，Actor/Critic 参数独立。结果只说明当前实现的数值响应与传播路径，不能解释为学到了正确业务语义、业务重要性或更优策略。仅执行图编码器和内部张量干预；未执行最终动作评分、价值头或 C128 神经融合，无训练、RPC、AnyLogic、新班次和预算登记。

## 输入与逐层表示

复用 [历史快照审计](../historical_snapshot_001/audit_report.md) 的同班相邻有效决策，未重新挑选样本，未用 T1 补充 T0。历史采样策略是 TQ128，不是 BG128。

|项目|T0|T1|
|---|---:|---:|
|run_index|9810998|9810998|
|原始 state_index|61|62|
|仿真时间（秒）|9603.707937|9717.867212|
|决策 Ego|Truck8|Truck0|
|节点 / 有向边记录|19 / 88|19 / 88|
|道路边 / 其他关系记录|16 / 72|16 / 72|

两时刻相距 114.159274 秒；同时存在车辆状态、关系、数量、时间年龄及 Ego 变化，不能视为单因素物理实验。19 个节点顺序固定，每层为 `[19,64]`。H0 是正式 TokenEncoder 输出；H1/H2 分别是两层同步图传播结果。Global 在两层中保持旁路。原始 C128 在后续 Actor/Critic 读出支路融合，不参与这里的 H0/H1/H2。

- [Actor：T0/T1 × H0/H1/H2 热图](actor_H_layers.png)
- [Critic：T0/T1 × H0/H1/H2 热图](critic_H_layers.png)
- [全部 12 个矩阵](representations.npz)，键为 `{actor,critic}_{T0,T1}_H{0,1,2}`。
- [节点64维明细](node_representations.csv) 与 [变化范数/RMS/余弦](representation_changes.csv)。

每张热图的六个面板共用对称色标。通道编号没有预设业务含义；Actor 与 Critic 的独立坐标轴不能逐通道对齐解释。LayerNorm 后节点向量范数接近8，不能凭范数大小判断信息量。

## 第一层：哪些 Truck 的消息进入 Dump？

这里按 `source→destination` 表示信息方向，列的是直接消息来源，不等于可逆恢复这些车辆全部状态。

|快照|进入 Dump0 的 Truck|关系明细|实际 loaded_dump|
|---|---|---|---|
|T0|0、2、6、8|Truck0：truck_at_node、loaded_dump、service_reverse；Truck2/6：truck_at_node、loaded_dump、capacity_wait_reverse；Truck8：truck_at_node|Truck0/2/6，共3辆|
|T1|0、2、3、5、6、7|Truck0：truck_at_node；Truck2/6：truck_at_node、loaded_dump、capacity_wait_reverse；Truck3/5/7：loaded_dump|Truck2/3/5/6/7，共5辆|

两时刻均为10条直接 Truck→Dump0 关系记录，来源车辆从4辆变为6辆。**4→6 不是已载承诺车辆数，后者是3→5。** T0 Truck8、T1 Truck0 的这里的关系仅为交通定位。Dump1 两时刻均无直接 Truck 入边，但有其他实体消息，不是孤立节点。

Truck3/5/7 从装载转为已载并面向 Dump0，新的 loaded_dump 边将其载荷吨位、已知品位和状态编码送入 Dump0；Truck0 卸载完成，原 loaded_dump/service 关系消失。Dump0 的已载目标数量从3辆、376.4吨变为5辆、771.7吨；这些物理数量见历史数量审计，不能从神经消息范数中读取。

逐边信息保存了4个 head 的 alpha、value/message、加权消息、14维边属性和64维投影。见 [重点消息](message_evidence.csv)、[全部704条逐层边记录](edge_messages.csv)、[608条关系组记录](group_messages.csv)、[完整消息张量](message_tensors.npz)。本样本未出现全部25种关系类型；类型合同保持正式 Schema，不将“Schema定义25种”写成“每张图实际出现25种”。

## 第二层：哪些 Dump 的消息进入候选 Load？

第二层从 H1 同步计算 H2，因此以下链路成立：

```text
Truck 的 H0 → Dump0 的 H1 → Load0 / Load1 / Load2 的 H2
                        compatible_material_reverse
Dump1 的 H1 → Load3 的 H2
```

兼容边表示物料兼容和下游信息读取，不表示反向运输。每个 Load 的 material 组只有这一条入边，所以4个 head 的 alpha 均严格为1；同一时刻同一编码器中，Dump0 发给 Load0/1/2 的 message 和投影向量逐元素相同。**这些边上的相同权重不是网络学会了候选区分。** 候选差异仍来自自身表示、其他关系组、边属性的差异及后续读出。

[两层消息流图](two_layer_message_flow.png) 同时展示 Actor/Critic、T0/T1。线旁数字是同一来源车辆多条边的64D投影先相加再取L2范数；Dump→Load 则为对应单边投影范数。它们位于残差和 LayerNorm 之前，省略其他入边、presence项和残差，不是总输出的贡献比例。

另外做了隔离的内部路径检验：固定 G1、其余节点 H1 和目标 Load H1，只把 Dump0 H1 换成 T0；以及先在 G1 中替换 Truck3/5/7 的 H0，再只把所得 Dump0 H1 送入第二层，阻断经其他节点的并行变化。

|内部张量替换|角色|Load0 响应L2|Load1|Load2|Load3|
|---|---|---:|---:|---:|---:|
|Dump0 H1(T0) 放入 T1|Actor|0.050077|0.053515|0.053441|0|
|Dump0 H1(T0) 放入 T1|Critic|0.057976|0.060040|0.060160|0|
|Truck3/5/7 H0 经 Dump0 H1 的隔离通道|Actor|0.004884|0.005278|0.005263|0|
|Truck3/5/7 H0 经 Dump0 H1 的隔离通道|Critic|0.004520|0.004779|0.004788|0|

这证明目标数值路径存在，并且这个隔离通道不会直接进入 Load3。它不是删除边实验，也不是物理自洽的改派实验：G1及新 loaded_dump 边始终固定，替换的是隐藏张量。见 [路径干预CSV](path_interventions.csv)、[定义与结果JSON](path_interventions.json) 和 [向量NPZ](path_interventions.npz)。

## T0→T1 哪些候选表示变化最大？

使用同角色同参数的 `||H(T1)-H(T0)||_2`；以 **H2** 比较，Actor/Critic 的排序均为 **Load2 > Load1 > Load0 > Load3**。不同层的排序不必相同，Critic 的 H0/H1 均为 Load1 > Load2。

|候选|Actor H0|Actor H1|Actor H2|Critic H0|Critic H1|Critic H2|
|---|---:|---:|---:|---:|---:|---:|
|Load0|0.103591|0.142141|0.184468|0.101591|0.147165|0.171858|
|Load1|0.205432|1.600872|2.285449|0.333033|1.601965|2.307866|
|Load2|0.244450|1.631172|2.302036|0.313090|1.589217|2.316160|
|Load3|0.002072|0.019337|0.022931|0.003921|0.017551|0.027640|

见 [变化与内部路径分解图](load_changes_and_attribution.png)。该排序只针对这对快照和固定初始化，未做多种子稳定性检验，不是任务重要性排序。

## 变化来自自身还是邻居？

先在编码器内部区分三个因素：目标 Load 的整行 H0、其余18个节点的 H0、G（拓扑、关系类型/组、组存在性和边属性）。每个候选完整计算8种张量组合，采用三因素 Shapley 向量分解分摊交互。混合时21D数量对初始编码的影响留在 H0 中，不随 G 重新编码，避免把同一通道重复归入两个因素。

下表为各向量分量沿真实 ΔH2 的**带符号投影份额**：`dot(phi, ΔH2) / ||ΔH2||²`。不是向量范数占比，可以为负；本例每行和约100%。这是内部计算分解，不是原始物理状态的因果贡献。

|角色 / Load|自身 H0|其余节点 H0|关系图/边属性|
|---|---:|---:|---:|
|Actor Load0|23.3%|20.0%|56.7%|
|Actor Load1|1.8%|0.7%|97.5%|
|Actor Load2|2.6%|0.2%|97.2%|
|Actor Load3|1.8%|90.4%|7.8%|
|Critic Load0|30.2%|−4.7%|74.5%|
|Critic Load1|1.0%|0.9%|98.1%|
|Critic Load2|1.9%|1.0%|97.1%|
|Critic Load3|约0.0%|94.0%|6.0%|

向量、八格组合和闭合误差见 [hybrid_attribution.csv](hybrid_attribution.csv) 与 [attribution_vectors.npz](attribution_vectors.npz)。**Load1/2 的 G 分量大，不能直接解释为模型已理解装载完成。** 它们在 T0 的 commitment、service 两组各有1条入边，T1 两组均为空；组消息和 presence 投影同步归零，分组结构本身会带来显著表示变化。Load0 的对应组仍存在，但成员、数量和属性发生变化。本次未进一步把 G 因素拆成拓扑、属性和 presence 的独立因果比例。

“自身 H0”也不等于“自身原始物理属性”。正式 TokenEncoder 在 GAT 之前已经编码站点车辆列表、服务/FIFO引用、地理在场车辆、路径内容和确定性数量：

- Load1/2 的服务列表分别由 Truck7/5 变为空，预留吨位193/147吨变为0，自身local20、数量和关系都变化。
- Load0 的服务成员 Truck3→Truck1，预留吨位238.7→284.7吨。
- **Load3 的原始47维和21维数量全部不变，但引用的 Truck4 的 loadAge/运动历史age增长，经服务引用和地理编码使 H0 改变。** 因而没有新增/消失事实边，也可能收到变化的邻居内容。

为细查 H0，单独冻结 local20基础投影、业务引用、地理和数量四条内部支路。四条同时冻结能精确恢复目标 Load 的 H0(T0)；但在其余节点H0(T1)、G1上继续传播，H2仍不恢复T0。例如 Actor Load2 的剩余距离为2.250775，实际两快照距离为2.302036。单支路敏感性不能相加为份额。完整定义见 [分支审计](raw_sensitivity_notes.md)、[120条敏感性记录](raw_sensitivity.csv)、[原始字段差异](raw_field_changes.csv)、[数量差异](raw_quantity_changes.csv)。

## 核验、边界与可复现性

[独立复核](independent_review.md) 的146项检查全部通过：保存的H0/H1/H2与正式 `scene()` 前向逐元素一致；逐边投影加总和presence项恢复LayerNorm前结果；独立重算三因素分解闭合；Global旁路；参数/buffer前后摘要一致。逐边投影重建最大误差 `3.58e-7`，分解/端点最大误差 `5.96e-8`。

310项历史来源引用校验通过，两行快照与原HDF5逐元素一致。受保护PPO核心SHA仍为 `8e4cfa31c622f5e33756e304eea3f7526547db72ae93f7d9f00808845d61ef96`。模型完整参数/buffer摘要：

```text
Actor  fc33cda5d679a6277b2774b0d49c55c87201d2782b113741e7fc9b8dceff4694
Critic 5be16494983376b3c4c1ca0df6f466c37e38bf2e7a3aa2816f65a0019c104f0b
```

结论限于：事实图能驱动正式编码器；关注的两层路径有非零数值响应；候选表示变化可以在内部编码与关系通道之间分解。尚未验证训练后是否利用这些通道、是否正确区分业务、能否改善Actor动作或Critic误差。本次也不能推出摘要可删除、奖励有效或应立即改变聚合结构。

Reward不进入本次图编码输入，未重新拟合或更改目标。原始RPC奖励、008联合成本/MC及窗口结算边界继续引用 [历史Reward审计](../historical_snapshot_001/reward_audit.csv)；隐藏向量变化不转换成奖励改善。

复现入口与执行顺序见 [本轮方案](../../../../change_plans/token_attention_001/business_graph_summary_009/encoder_snapshot_001/README.md)。[summary.json](summary.json) 记录数值与模型身份，[manifest.json](manifest.json) 固定本轮产物、脚本、原始快照和正式实现签名。此前构图审计保持原样，另行封存本轮有Forward、无优化的证据。
