# BG009 图与 C128 融合：离线工程验收

2026-11-18，Asia/Shanghai。实现与详细字段见 [方案](../../../change_plans/token_attention_001/business_graph_summary_009/README.md) 和 [schema](../../../change_plans/token_attention_001/business_graph_summary_009/schema.json)。最终证据为 [engineering_003](engineering_003/validation.json)；001/002保留为实施过程证据，最终源文件签名以003为准。

新增 [历史32班相邻决策快照审计](historical_snapshot_001/audit_report.md)：5841个有效决策全部构图通过，两张同班相邻图及差异图、全部边/路径/数量/C128/Reward CSV已交付。该次严格无模型实例化/Forward/参数更新/新班次，当前008成本及MC与旧冻结目标逐元素一致。

随后新增 [相邻快照编码器数值传播审计](encoder_snapshot_001/audit_report.md)：固定随机初始化下，Actor/Critic的12个H0/H1/H2矩阵、逐边投影与两层路径干预均已保存，独立146项核验通过。该轮仅无梯度Forward和内部张量干预，无优化或新班次，不代表训练结果。

## 实现

- 19×64D节点，16条有向道路属性边，25种有向关系、4个关系组。
- 原始目标/服务列表构造事实业务边；候选仅用于直接读出，不增加实际承诺。物料与观测目标不一致时拒绝。
- 按接收节点与关系组分别softmax；道路属性同时进入score/value；组间拼接，缺失组消息为零。
- 21维明确数量，计数和吨位不经attention加权平均；进入节点投影及独立读出通道。已载车辆残留reservation不重复计入待装，当前有segment的车辆不重复计为节点已到达。
- Actor直接读取Ego/候选Load/兼容Dump，C128按88＋4×10拆分，采用共享候选评分与显式候选×上下文交互。Critic独立图编码及完整C128融合，单标量输出。无最终Query。
- 保留同一成本、MC、冻结old-V GAE、全批联合优势标准化及PPO默认配置。旧TQ checkpoint/配置不可静默迁入，新Actor及模型封印绑定新身份。

## 验证证据

最终 [pytest记录](engineering_003/pytest.txt) 包含45项通过：16项图构造语义检查、11项模型机制检查、18项学习器/封印/独评检查。测试包含两个完整synthetic班次组成的批次更新、精确检查点恢复及继续训练；这不消耗AnyLogic物理班次。

[真实状态forward](engineering_003/real_forward.json)读取旧TQ128最终100班独评，每班等距8个决策，共800状态。新模型以固定种子随机初始化，只做只读前向；所有有限值、硬mask、19节点、数量通道、batch隔离、组内归一、缺失组归零、无最终Query及参数不变检查通过。该结果只证明历史输入兼容和数值通道成立，不是新策略的独评成绩，也不证明已经学会业务。

800状态中实际出现道路、定位、前车、预留装载、已载卸载、seize、capacityWait、service及物料关系；queue/admission恰未出现，其正确性由针对性的synthetic单测覆盖，不宣称已在真实样本中覆盖所有关系。

|参数|旧TQ128|BG009|
|---|---:|---:|
|Actor|190321|260481|
|Critic|186033|185297|
|实体Token|32|19|

节点减少并不等于容量匹配。历史方案与本版同时存在图、融合、评分等多项差异，性能归因需后续同环境、同成本的匹配实验。

## 完整性与范围

[validation.json](engineering_003/validation.json)确认受保护PPO核心SHA仍为`8e4cfa31c622f5e33756e304eea3f7526547db72ae93f7d9f00808845d61ef96`；旧正式运行359项源文件pin全部未变。100份历史H5逐份校验前后SHA一致。完整新架构、学习器配置、依赖源签名见 [architecture.json](engineering_003/architecture.json)。

真实仿真新增班次0、历史真实数据优化更新0。没有修改ALP，没有准备或启动新AnyLogic控制器。当前交付是模型、学习器、封印/恢复、独评和一键离线验收；真实RPC端到端冒烟需后续单独准备009身份与启动包。
