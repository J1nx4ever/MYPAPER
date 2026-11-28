# 按班次切换训练路网

目标是让每次 `RL_Training` 迭代用该班配置创建独立的原生路网、装卸站和车队。路网在这一班内保持固定，按新的 `Main` 初始化下一班。每份配置明确节点 ID、米制坐标、物理道路以及正反向限速、车辆槽位和车型载重、出勤与起点、装点矿废角色和品位，以及矿石破碎站和废石卸点的位置。

## 本次实现

| 场景 | 节点 | 有向路段 | 配置在役车辆 | 装点和卸点 |
| --- | ---: | ---: | ---: | --- |
| `tree_A` | 7 | 12 | 9 | 4 装点，2 卸点 |
| `loop_B` | 8 | 16 | 7 | 4 装点，2 卸点 |

两图从原演示的分支树和环路扩展到训练的业务槽位。B 增加连接关系，A 有四个补零道路槽。路网节点仍使用 9 个槽，空路口排除在图池化之外，动作仍是选四个装点。矿石去破碎，废石去废石卸点。两图使用站点的固定业务 ID 绑定目标，坐标和路段 ID 可按班变化。[场景定义](layouts.json) · [场景编译器](compile_layouts.py) · [原生路网辅助类](TrainingTopology.java)

演示版的 `Topology_Scene_Preflight` 使用与 `RL_Training` 同类的 Parameter Variation / Freeform `index`，两次启动分别从 [tree_A.properties](compiled/tree_A.properties) 和 [loop_B.properties](compiled/loop_B.properties) 读配置和校验 SHA256。它使用演示顶层 `TopologyVisualMain`，构建原生 Level，校验装卸站点位置和角色，将配置中在役的探针车辆放到对应起点。实际生产 Truck 状态机、装卸队列、RPC 与梯度更新还未接入这个入口。查看控制台的 `[TOPOLOGY_PREFLIGHT]` 行可辨认两个场景、实建路数与在役数。生产 `Main` 和旧 `RL_Training` 在原有模型中继续使用各自的历史配置。

Python 原型保留原版 BG128 业务关系和 C128 特征，路段边由每条观测实际携带的端点构建，补零边及未用路口会被排除。同一个批次中的 A/B 具有独立邻接。使用冻结的 `full-s04/u32` Actor 和 Critic，旧地图的动作概率和价值在单图标称回放中逐位相同，两个新图也能在一次合成输入批次内输出概率和价值。该策略验证用九车的合成观测。B 的 7 辆车出勤在原生启动预检中单独验证，7 车训练仍需接入现有出勤掩码。当前采用的图适配器用于原型验证，接入训练前应直接生成新图的业务和路网边，并冻结独立的模型与数据合同。[图适配器](topology_policy.py) · [离线核验](offline_probe.py)

## 生产接线顺序

当前四单元恢复训练计划的嵌入源码与项目内的 `DRL-PPO-baseline1.alp` 一致。演示副本中的旧 `RL_Training` 则仍绑定历史单班退场的 helper。新拓扑训练应准备全新 CampaignPlan、RPC 配置和身份，再新增对应实验。可沿用 Freeform、配置字节哈希、单线程和每次新建顶层 Agent 的流程。[当前训练入口](/home/jinx/Faith/experiments/lagrangian_ppo_v1/anylogic/3d/DRL-PPO-baseline1/DRL-PPO-baseline1.alp:25569)

| 位置 | 现有实现 | 变拓扑训练接法和验收 |
| --- | --- | --- |
| `RL_Training → Freeform` 与 `Baseline1CampaignPlan` | `index` 读取固定班次行、种子、品位及道路扰动 | 每行新增 `layout_id`、场景路径、配置 SHA 和场景 SHA。新运行身份入新清单。开班先核准配置字节，再校验场景，构建成功后再激活车辆 |
| `Main → On startup / init1` | 通过固定画布节点和 Attractor 放置装点、卸点、车辆 | 在 `init1` 前构建 `TrainingTopology`。用业务 ID 绑定四铲、矿石和废石两个卸点及九槽车辆。启用 6–9 辆时沿用车队的 `dispatchable`、`in_system` 两个掩码。补齐停车区、接驳道、泊位和首车顺序，新图中核对每车的实际位置以及进入原生路网的 Level |
| `Main → initRoadPointNodeMap / init_network / init_routeNodeMap`，`PathSegment`，`Truck → prepareRoute / calculateDisOnSeg` | 固定站点键、固定 OD 和道路列表，旧实现有对全网最短路径的几何查询 | 路由、车速、跟车和分段占用使用同一个实际路段注册表。每段长度直接读取其原生 `Path`，每个 OD 用场景路段生成，空重车限速和 ETA 按同一口径。首版只接两端原生最短路唯一的路段，逐段核对被选择的边，处理闭路时重算后再发车 |
| `Main → obsInit / 观测 road 行 / buildRuntimeScenarioComponents`，Truck 路段编码 | 16 个有向路段写死，编号和回传字段分别维护 | 把 16 槽改为构建后的有向路段表及其零填充。位置、当前路段、路线顺序及 START 场景快照都使用同一编号。给这份语义单独的协议版本。Python 在 START 校验**实际**节点和道路、坐标、车点绑定及场景摘录，后续每帧校验身份和有效动作 |
| `DispatchEnv / collector / BG128 policy` | 只承认冻结生产地图，存 1519D 和选四动作 | 新版允许同一批次混合两图，每个决策和 PPO 回放从该样本的道路行重建路网并检查掩码。沿用 C128 的材料与 FIFO 语义。适配层接入活跃路口和车辆池化，核对 old log-prob 回放，PPO 损失和约束估计沿原逻辑 |

四装两卸固定时可以做真正的拓扑变化，包括路口和道路数变化。若要超过 9 路网节点或 16 有向路段，或要改变装卸点数量，单独升级观测、图批处理及候选索引。让策略主动选卸点还要增加决策阶段和动作合同。

## 开通训练前的验收

1. 同一名义基准配置核对原版与新版的有效动作、候选概率、价值和首趟轨迹。使用新拓扑 checkpoint 命名和 optimizer，标明迁移的权重，旧恢复包保持独立。
2. 用生产车辆先在两种地图各过一条完整的装卸车次，逐事件核对路段、站点坐标、区间用时、资源占用以及装载承诺与卸载兑现。至少有一条 OD 因换边出现不同路线，验证新图的物理移动和 RPC 实发边一致。
3. 自然完整班验收矿石与金属守恒、四窗口、废石截止以及无效动作拒绝。随后验证一次小批次闭环、PPO 前后旧 log-prob 对齐、可断点恢复。训练与测试按布局组分开，不能只按种子分开。

本次仅做启动和离线策略预检，未登记或启动训练班次。正式准备包需冻结各班的独立 recording ID、配置字节 SHA256、Python 与 AnyLogic 目录、run_index 和双种子，再按登记流程启动。

[同目录 ALP 新副本](/home/jinx/Models/DRL-PPO-baseline-topologydemo/DRL-PPO-baseline-topologydemo-training-topology.alp) · [Java 启动预检](/home/jinx/Faith/experiments/lagrangian_ppo_v1/reports/token_attention_001/business_graph_summary_009/variable_topology_training_001/native_preflight.txt) · [Python 混合批次结果](/home/jinx/Faith/experiments/lagrangian_ppo_v1/reports/token_attention_001/business_graph_summary_009/variable_topology_training_001/offline_probe.json)

后续的两张图生产车单趟已通过。验证使用实际行驶、装卸资源和 FIFO 守恒，策略 RPC 尚待接入。[单趟验收与收据](physical_trip_001/README.md)

A 图 9 车和 B 图 7 车均通过联合运力初筛及原生完整班回放。证书适用于冻结的确定性物理配置。变拓扑训练实验与 RPC 尚待接入。 [可行性门](feasibility_001/README.md) · [完整班结果](feasibility_001/full_shift_001/README.md)
