# A/B 生产车单趟验收

在同目录的 [单趟验收 ALP 副本](/home/jinx/Models/DRL-PPO-baseline-topologydemo/DRL-PPO-baseline-topologydemo-physical-trip.alp) 中运行 `Topology_Physical_Trip_Check`。实验每次启动全新的生产 `Main`，依次读两份冻结配置，并调用已验证的 `TrainingTopology` 构建路网。实际运输使用生产 `Truck` 状态机和原有 LoadSite、DumpSite 的队列、资源释放、料源记账与粗破 FIFO。实验固定选择同一辆车和同一装点，单独核验物理链路。实验未接入 Python RPC。

| 场景 | 运输车与装点 | 期望空车路线（重车反向） | 运行编号 |
| --- | --- | --- | --- |
| `tree_A` | 车 3，装点 1，147 t 矿 | `0 → 3 → 6` | 10054021 |
| `loop_B` | 车 3，装点 1，147 t 矿 | `0 → 2 → 5 → 3 → 6` | 10054022 |

车队开班保留 9/7 出勤和 9 个物理槽位。单趟只给车 3 发送激活，其他车保持停放。车 3 逐段调用 AnyLogic 的原生 `moveTo`，空／重车以 6.5／5.0 m/s 为上限并服从各有向道路限速。周期事件只采样坐标，每段保留一个段内点。入站由原生到达回调和生产状态机推进。B 途中通过节点 5 时不会请求装载。卸完后先走原有释放与车次统计，再检查吨量、金属和 FIFO 并结束。在役车队拥堵、班中退场和 8 小时产量留待后续完整班。

## 接线和验证

| 位置 | 修改前 | 验收分支 |
| --- | --- | --- |
| `Main → Parameters → topologyTripCase` 与 `On startup` | 以旧训练计划握手开班 | 参数默认 0。新实验 Freeform 将两次迭代映射到 1、2。在旧握手前绑定固定验收配置并初始化真实代理 |
| `Main → evUpdateSpeedObservation` 与 `Functions → destinationToInt` | 旧图采样和固定节点引用 | 仅单趟分支取活动段内点和活动图节点编号 |
| `Truck → Advanced Java → fleetDispatchNow`、`NeedDispatchToDumpSite` | 生产策略与固定站点引用 | 单趟首次派单固定到装点 1。两端的目标节点按当前场景绑定，正常矿石去粗破 |
| `Truck → MoveSegment / MoveSegment1` 和 `transition18 / transition24` | 原生逐段移动与状态机到达 | 发车核对原生路径，到达转移的动作核对实际目标坐标，然后进入原有状态 |
| `Truck → 收集数据并重置`、`transition15` | 计入车次后再派 | 构成一趟后验证释放、矿石和金属守恒，并停止。单趟分支禁止再次派单 |

**运行前**查看 [冻结的身份清单](launch_identity.json)，其中逐次记录 recording ID、配置和场景的 SHA256、run、环境 seed、AnyLogic 事件目录及 Python 输出目录。两个用例都写 `events.jsonl`，采用独占创建。停表上限 2400 模型秒，超时得到 `INCOMPLETE`。正式生产班预算独立，单趟不记作完整班。

打开副本并构建后，运行新实验 `Topology_Physical_Trip_Check`。期望控制台出现两条 `[TOPOLOGY_TRIP_PASS]`，两次分别校验 **矿石 147 t、金属 41.895 t**。Python 复核实际路线、三点坐标证据、行驶阶段、车速与最终物料：

```bash
/home/jinx/miniconda3/envs/anylogic-env/bin/python experiments/lagrangian_ppo_v1/change_plans/token_attention_001/business_graph_summary_009/variable_topology_training_001/physical_trip_001/verify_trip_receipts.py
```

2026-11-27 由 AnyLogic 原生实验运行，两图均通过实际轨迹及物料复核。A 完成于 855.866 s，B 完成于 1,555.392 s，分别通过 4 段和 8 段坐标校验。每次只完成一趟。粗破在收尾时有 147 t 矿石和 41.895 t 金属量，均在 FIFO 中，符合入仓守恒。[A 复核](/home/jinx/Faith/experiments/lagrangian_ppo_v1/reports/token_attention_001/business_graph_summary_009/variable_topology_training_001/physical_trip_001/tree_A_trip_check.json) · [B 复核](/home/jinx/Faith/experiments/lagrangian_ppo_v1/reports/token_attention_001/business_graph_summary_009/variable_topology_training_001/physical_trip_001/loop_B_trip_check.json) · [启动验证](/home/jinx/Faith/experiments/lagrangian_ppo_v1/reports/token_attention_001/business_graph_summary_009/variable_topology_training_001/physical_trip_001/preflight.txt)

本轮两个 recording 已有原生事件记录。若重跑，准备新的身份与输出目录。
