# A/B 原生整班可行性见证

在 topologydemo 的单趟验收模型基础上生成了一个独立副本。实验 `Topology_Full_Shift_A` 和 `Topology_Full_Shift_B` 各运行一班，按自身冻结配置在开班初始化路网、站点和车辆。两图使用同一固定派车规则。模型复用原生车辆状态机、设备资源队列、装卸账本和破碎 FIFO。

- A 图使用 7 个节点、12 条有向路段和 9 辆在役车。
- B 图使用 8 个节点、16 条有向路段和 7 辆在役车。
- 两班检验固定限速、确定性装卸与品位、无故障及关闭动态跟驰的物理配置。车速遵循每条路段限速，空车上限 6.5 m/s，重车上限 5.0 m/s。固定派车规则计入空程、在途及排队的装铲任务和目标品位。
- 原生仿真在出发和到达时检查路网对象绑定、节点坐标、行驶距离及运动时间。班末记录按方向计数，逐车轨迹和观测快照无需开启。
- 通过条件为 8 小时自然结束，实破矿石至少 16,000 t、实卸废石 4,000–6,193 t、四个 2 小时窗各自有实际产量且品位 26%–28%，并通过矿石、金属、FIFO 和派车重放校验。设备能力、实际路网和开班车队必须与冻结场景相符。

## 原生验收结果

| 场景 | 实破矿石 t | 实卸废石 t | 第 1–4 窗品位 | 原生到达次数 | 准入 |
| --- | ---: | ---: | --- | ---: | --- |
| A · 9 车 | 19,402.3 | 4,473.8 | 26.91%、26.89%、26.90%、26.95% | 696，覆盖 12/12 个方向 | ADMITTED |
| B · 7 车 | 17,894.7 | 4,446.5 | 26.31%、27.26%、26.82%、26.63% | 919，覆盖 14/16 个方向 | ADMITTED |

B 图开班时校验了 16 条有向路段的实际几何和限速，本次派车未使用 `intersection1 ↔ intersection2`。两班都自然结束并通过矿石、金属、FIFO、等待积分和派车重放。固定规则分别实际派车 177 次、181 次。证书的适用范围是这两组冻结的路网、车队、装卸能力和确定性运行配置。

[封存结果](/home/jinx/Faith/experiments/lagrangian_ppo_v1/reports/token_attention_001/business_graph_summary_009/variable_topology_training_001/feasibility_001/full_shift_001/witness_collection.json) · [证据索引](/home/jinx/Faith/experiments/lagrangian_ppo_v1/reports/token_attention_001/business_graph_summary_009/variable_topology_training_001/feasibility_001/full_shift_001/witness_index.json)

## AnyLogic 操作与手工接线定位

如需查看验收配置，打开 topologydemo 目录的 `DRL-PPO-baseline-topologydemo-full-shift-witness.alp`。在项目树查看分别负责 A、B 的 `Topology_Full_Shift_A` 与 `Topology_Full_Shift_B`。两份已完成身份供复核。Freeform 参数已经写入，两者分别给 Main 的 `topologyWitnessCase` 传入 1 和 2。原生班末事件在 28,800 s 结转，28,801 s 是异常情况的停止上限。配置与路径已预设，重复运行会被既有事件目录拒绝。各班身份和路径见 [启动身份](launch_identity.json)。

| 精确位置 | 原单趟版 | 整班副本及验证 |
|---|---|---|
| `Main → Parameters → topologyTripCase`，`Main → On startup / On destroy`，`Main → destinationToInt / evUpdateSpeedObservation` | 加载单趟开班配置，跟踪一辆车和途中采样 | 改用 `topologyWitnessCase` 及 `TopologyWitness`。冻结输入与实建场景匹配之后才激活在役车，关闭逐秒采样 |
| `Truck → Advanced Java → fleetDispatchNow`，`Truck → transition18 / transition24` | 单车只派一次，已有路段出发与到达检查 | 多车同规则派车，成功预留后记动作，复用两原生移动到达回调检查全部车辆 |
| `Truck → transition15` 与 `收集数据并重置 → Entry action` | 检查分支只跑第一趟并提前完成 | 条件恢复为 `true`，去掉单趟结束调用，进入下一轮原生派车 |
| `Main → finishEpisodeAndNotify` | 原班末逻辑在统计完成后进入常规 RPC 分支 | 见证分支在已有四窗和物料审计后封存并结束，其他实验仍走原路径 |
| `Java Classes → AcceptanceEventRecorder → finishWitness` | 原通用收据 | 见证分支增加零回退、零无效动作及各方向原生到达次数。复用已有装卸、FIFO、等待与品位记录器 |

副本从历史 ALP 通过 [生成器](prepare_witness.py) 生成。实现集中在 [原生验收](TopologyWitness.java) 和 [固定派车](TopologyFeasibilityDispatch.java)。同一地图使用其他出勤、随机品位、动态拥堵或故障配置时，按新场景验收。

## 复核

用 `anylogic-env` 调用 `collect_witness.py` 查看状态。正常终止后加 `--collect` 登记已跑班，生成 `witness_index.json` 和 `witness_collection.json`。可加 `--case A` 或 `--case B` 单独复核。实际班已停止且 manifest 未完成时，以 `--collect --case A --finalize-incomplete`（或 B）记 UNKNOWN。正常自然终止而业务目标未达时，物理班记为 COMPLETE，场景为 NOT_ADMITTED。

上一级 `admit_layouts.py --witness-index` 可独立复核同一证据。准备身份本身不占训练预算。两班均使用固定派车，不消费 Actor 采样 seed。
