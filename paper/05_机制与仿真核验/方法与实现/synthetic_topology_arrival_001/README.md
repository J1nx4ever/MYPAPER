# 合成路网拓扑与到达坐标验证

两张地图使用同一份 [campaign.json](campaign.json)。A 有 4 个节点、3 条双向道路，是以中心路口为分叉点的树。B 有 6 个节点、6 条双向道路，西、东、南路口与装载点形成一条环，西路口和南路口之间是备选道路。每张地图都从起点到装载点，再从装载点到卸载点。

[拓扑与路径对比图 PNG](../../../../reports/token_attention_001/business_graph_summary_009/synthetic_topology_arrival_001/topology_comparison.png) · [SVG](../../../../reports/token_attention_001/business_graph_summary_009/synthetic_topology_arrival_001/topology_comparison.svg)

| 地图 | 装载点坐标 | 卸载点坐标 | 装载段 / 卸载段长度 | 记录的目标坐标误差 |
| --- | --- | --- | --- | --- |
| A | (80, 60) | (170, 0) | 140 / 150 | 两点均为 0 |
| B | (100, 105) | (245, 25) | 180 / 225 | 两点均为 0 |

坐标和距离使用模型空间单位。Java 从 AnyLogic 8.9 原生路网查询两段路径，然后按每秒 8 个模型单位和 0.25 秒的间隔向 SDK 取样，在终点验收两维欧氏误差 `<= 1e-6`。Python 用事先指定的途经节点复算每个位置，独立核对 352 条记录。验证同时检查了节点数、路段数、路由来源、途经位置和端点坐标，能发现直线穿越、错走环路或到错节点。[坐标验收记录](../../../../reports/token_attention_001/business_graph_summary_009/synthetic_topology_arrival_001/coordinate_checks.json)

对校验器做了两项反例检查。在记录副本中改动地图 B 的一条途中记录，以及将卸点到达记录的坐标改成距离目标 0.1 个模型单位，两项均被拒绝。

两张地图分别初始化一个独立 SDK 网络。独立运行时拒绝推进 AnyLogic 原生仿真时钟，本轮的时间列来自设定速度下的路径偏移量，图中的轨迹来自 AnyLogic 原生路径采样。未加载或修改任何 ALP，未启动生产班次，未调用 BG128。若在 AnyLogic Simulation 验收车辆运动，使用同一份地图配置在独立演示模型的 `Main → On startup` 初始化网络。在 `Truck → On arrival to target location` 读取 `getX()` 和 `getY()`，按同一阈值判断到达，再发下一段 `moveTo`。用 `Main → Event` 每 0.25 秒采样途中坐标，并复用本方案的途经节点判定。[AnyLogic 路网移动文档](https://anylogic.help/anylogic/agentbased/movement-network.html)

运行身份、配置字节 SHA256、事件目录和 Python 输出目录见 [launch_identity.json](launch_identity.json)。从工作区以 `anylogic-env` 的 Python 运行 `run_validation.py`，使用身份清单指定的全新输出目录。已封存的 CSV 可用 `plot_topologies.py --events <anylogic_events目录> --output <图件目录>` 重绘。
