# 合成路网与装卸点换位验证

每个 8 小时班开始前选择一套人工构造的场景。一个场景包含装载点、破碎站、废石卸点和路口的坐标，以及道路的两端与限速。首轮用平面道路，固定 9 辆车、4 个装载点、1 个矿石破碎站和 1 个废石卸点。车辆初始停靠位置跟随对应卸载区，物料类型、品位配置、装卸服务规则和班次矿废目标沿用同一实验配置。

## 离线验证

Python 探针使用冻结动态 `u9`、正式 Actor 和摘要构造器。三个场景共用一份合成派单状态，只替换道路端点与长度。

| 场景 | 合成变更 | Python 结果 |
| --- | --- | --- |
| A | 参考布局 | 0 号装载点候选任务预计 1,029 s，动作概率 15.56% |
| B | 移动 0 号装载点、破碎站和废石卸点，保持道路连通关系 | 候选任务预计 1,524 s，动作概率 15.72% |
| C | 保留 B 的坐标，将 0 号装载点从路口 2 改接到路口 3 | 固定道路端点校验拒绝输入 |

从工作区根目录运行下列命令可重新生成结果到标准输出。证据保存在 [layout_probe.json](../../../../reports/token_attention_001/business_graph_summary_009/synthetic_layout_probe_001/layout_probe.json)。

```bash
/home/jinx/miniconda3/envs/anylogic-env/bin/python experiments/lagrangian_ppo_v1/change_plans/token_attention_001/business_graph_summary_009/synthetic_layout_probe_001/layout_probe.py
```

AnyLogic SDK 探针使用独立的合成坐标。装卸点换位后，最短路径长度从 80 变为 282.84 个模型长度单位。保持新坐标并改由路口中转后，路径变为 400。每次都检查了 Agent 绑定坐标、原生道路长度和路径查询结果。两项探针分别验证物理路网接口与策略输入，尚未通过 RPC 联动。动作概率变化只说明策略能读到布局引起的运距变化，完整班次的产量和品位仍需闭环测量。[Java 源码](SyntheticSiteNetworkProbe.java) · [SDK 输出及复现命令](../../../../reports/token_attention_001/business_graph_summary_009/synthetic_layout_probe_001/anylogic_network_probe.txt) · [AnyLogic 路网构建文档](https://anylogic.help/markup/create-network-by-code.html)

## baseline2 的手工接线位置

| 位置 | 现有实现 | 同拓扑换位需要的改动 |
| --- | --- | --- |
| `Main → On startup` 和 CampaignPlan | 启动时直接使用模型自带的地理对象 | 由逐班配置读取 `layout_id`，首次派单前创建、初始化并绑定该班网络，校验坐标、连通性及单位 |
| `Main → Functions → initRoadPointNodeMap` | 固定名称映射到画布上预置的 `PointNode` | 由场景构建过程按相同名称绑定新的节点。把这一步移到 `init1` 之前 |
| `Main → Functions → init1` | `loadNode`、`dumpNode` 和车辆初始位置绑定预置对象 | `add_loadSites()`、`add_dumpSites()`、`add_trucks()` 继续创建原有 Agent，通过新节点表绑定装卸点。车位随对应卸载区生成，并核对车位与车辆逻辑起点 |
| `Main → Functions → init_network`、`getNodesPosition`、`DistanceRecorder` 和 `Truck → Functions → calculateDisOnSeg` | 读取固定道路对象与默认 `network` | 道路长度、车辆寻路和发给 Python 的道路观测都取自该班实际使用的网络。保留 16 个有向路段的序号与端点顺序 |
| `Main → Functions → init_routeNodeMap` | 按固定站点序列写死路线 | A/B 的站点编号、路口编号与路径序列保持一致。扩展到 C 时再改为按场景图生成路径；同步修改 BG128 的道路校验和所有依赖固定端点的特征计算 |

验收以两张合成布局 A/B 的首次派单 RPC、道路长度、实际行驶路径、目标站队列、矿废卸载和矿石及金属守恒为依据。B 的车辆应走到移动后的站点，并在正确的装卸队列记账。随后如需验证换连线，再跑 C，确认从新路口驶入且策略收到对应端点。完整未见布局泛化需要把整张布局留作独立测试场景。每个班内保持布局固定。

本轮仅进行了离线验证，未修改 ALP 或启动任何实验班次。
