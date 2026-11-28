# baseline2 的两张地图 Simulation 动画验收

在 `baseline2` 的演示副本里新增两个 Simulation。两者共用一个连续空间顶层 Agent，班次参数决定用 A 图还是 B 图。两个实验都使用 AnyLogic 原生 `moveTo(PointNode)` 行驶，以车辆自己的 `getX()`、`getY()` 和到达回调作为验收依据。先用探针车验证地图、路线和可视化。

生产 `Main` 在启动时等待 CampaignPlan 配置，并初始化固定路网、车辆与 RPC 合同。A、B 的 4/6 个路网节点和 1 个装点、1 个卸点不能直接接入 BG128 的固定道路端点检查。本验收将顶层 Agent 指向 `TopologyVisualMain`。后续若把动态地图用于生产派单，需要用布局驱动生产 `Main` 的站点、路径、分段占用和 RPC 图。可变节点数还要修改观测 schema 和策略的固定图输入校验。[生产 Main 启动](/home/jinx/Models/DRL-PPO-baseline2/DRL-PPO-baseline2.alp:116) · [固定路网校验](/home/jinx/Faith/experiments/lagrangian_ppo_v1/business_graph_summary_v1/graph.py:154)

## 在 AnyLogic 8.9 中手工接线

| 位置 | 修改前 | 在演示副本中添加 | 作用 |
| --- | --- | --- | --- |
| 文件层级 | `DRL-PPO-baseline2.alp` | 在 AnyLogic 中用 **File → Save As** 将模型另存为同目录的 `DRL-PPO-baseline2.topology-demo.alp` | 同目录保留已有相对资源路径 |
| 项目 → New → Java Class | 无辅助类 | 新建 `TopologyVisualDemo`，Java 编辑器中粘贴 [完整辅助类](TopologyVisualDemo.java)，包名应与模型的 `fusion` 一致 | 读配置、建原生路网、绘制地图、记录实际坐标 |
| 项目 → New → Agent Type | 已有生产 `Main` | 新建 `TopologyVisualMain`，在 Properties → Space and network 选择 **Continuous**，空间宽、高可取 720、480 | 给验收路网和车提供独立的连续空间 |
| 项目 → New → Agent Type | 已有依赖生产状态机的 `Truck` | 新建空的 `TopologyProbeTruck`。拖到 `TopologyVisualMain`，选择 **Population of agents**，名称 `probeTrucks`，设置 **Initially empty** | 由 AnyLogic 创建原生测试车辆，画面上的移动标记由辅助类跟踪 |
| `TopologyVisualMain` 图中两个 Parameter | 无 | 添加 String `configPath`、String `configSha256`，默认表达式均为 `""` | 由两个 Simulation 覆盖各自配置文件及字节 SHA256 |
| 项目 → New → Experiment | 现有 `Simulation` 指向生产 Main | 新建两个 Simulation，命名 `Simulation_Topology_A`、`Simulation_Topology_B`，顶层 Agent 都选 `TopologyVisualMain` | 两次各自初始化独立的原生网络 |

AnyLogic 在属性里支持选择每个 Simulation 的顶层 Agent。程序构建的网络在启动时初始化，8.9 代码用 `getPresentationShape().add(level)` 把网络加到画面中。[AnyLogic Simulation 属性](https://anylogic.help/anylogic/experiments/simulation-experiment.html) · [按代码创建路网](https://anylogic.help/markup/create-network-by-code.html)

### 复制这四处代码

**`TopologyVisualMain → Advanced Java → Additional class code`**

```java
TopologyVisualDemo topologyDemo;
```

**`TopologyVisualMain → Agent actions → On startup`**

```java
topologyDemo = new TopologyVisualDemo(this, configPath, configSha256);
topologyDemo.start(add_probeTrucks(), time(SECOND));
```

**`TopologyProbeTruck → Agent actions → On arrival to target location`**

```java
TopologyVisualMain root = (TopologyVisualMain) getOwner();
root.topologyDemo.onArrival(this, time(SECOND));
```

**`TopologyVisualMain → Agent actions → On destroy`**

```java
if (topologyDemo != null) topologyDemo.close(time(SECOND));
```

### 加坐标采样事件

在 `TopologyVisualMain` 上添加 **Event**，命名 `topologyTick`。Trigger type 选 **Timeout**，Mode 选 **Cyclic**。First occurrence time 和 Recurrence time 都填 `0.25`，时间单位选 **seconds**。Action 填以下代码。画面每次采样更新车辆位置，正式到达只通过车辆的原生回调登记。[AnyLogic 的路网移动](https://anylogic.help/anylogic/agentbased/movement-network.html) · [到达动作](https://anylogic.help/anylogic/agentbased/on-arrival.html)

```java
if (topologyDemo != null) topologyDemo.tick(time(SECOND));
```

## 两次运行

1. 在两个 Simulation 的 **Parameters** 中分别粘贴 [可直接复制的配置参数](experiment_parameters.txt)。**Model time** 从 `0` 开始，`80 s` 停止。动画用 **Real time**，倍率 `1×`。视图中可看到从 `(30, 30)` 到约 `(700, 425)` 的路网和状态提示。
2. 按 **F7** 编译，先右键 `Simulation_Topology_A` → **Run**。启动后应看到 4 个节点、3 条道路。探针车经中心路口，先到装点、停约 1 秒，再经同一路口去卸点。待底部出现 `PASS` 后截图。
3. 关闭 A 的运行窗口并右键 `Simulation_Topology_B` → **Run**。启动后应看到 6 个节点、6 条道路。车从装点驶入东路口，再经过南路口到卸点，西—南斜线保留为备选。待 `PASS` 后截图。

| Simulation | 装载点坐标 | 卸载点坐标 | 预计到达时刻，含装点 1 秒停留 |
| --- | --- | --- | --- |
| A | `(80, 60)` | `(170, 0)` | `17.500 s`、`37.250 s` |
| B | `(100, 105)` | `(245, 25)` | `22.500 s`、`51.625 s` |

坐标误差阈值为 `1e-6` 模型单位。可在 `PASS` 画面暂停或等 80 秒停止，再截取全图。截图保存到 [本轮 reports 目录](/home/jinx/Faith/experiments/lagrangian_ppo_v1/reports/token_attention_001/business_graph_summary_009/synthetic_topology_simulation_001)，建议命名为 `A_simulation_pass.png`、`B_simulation_pass.png`。首次填写实验参数前可查阅 [独立 run、seed、recording、配置字节 SHA256 和输出目录](launch_identity.json)。每个 recording 一次运行，`events.csv` 采用独占创建方式保留历史。

两个窗口运行后，用以下命令核对实际轨迹、到达坐标、耗时、两段路线和对应身份。输出同时写入各自的 Python 输出目录和 reports。需要先看单图时，可加 `--layout A` 或 `--layout B`。

```bash
/home/jinx/miniconda3/envs/anylogic-env/bin/python experiments/lagrangian_ppo_v1/change_plans/token_attention_001/business_graph_summary_009/synthetic_topology_simulation_001/verify_receipts.py
```

已完成 8.9 SDK 构建与启动预检。两个窗口的实际动画、到达回调和收据仍待在演示副本中运行。[启动预检记录](/home/jinx/Faith/experiments/lagrangian_ppo_v1/reports/token_attention_001/business_graph_summary_009/synthetic_topology_simulation_001/preflight.txt)
