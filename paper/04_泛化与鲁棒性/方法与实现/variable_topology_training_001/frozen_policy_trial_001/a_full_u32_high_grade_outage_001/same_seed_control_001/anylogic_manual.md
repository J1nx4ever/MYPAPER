# A 图同种子无停工对照的 AnyLogic 核对清单

运行对象是独立文件 `DRL-PPO-baseline-topologydemo-a-full-u32-no-outage-control.alp`。其源文件为已完成 10056051 班的 `DRL-PPO-baseline-topologydemo-a-full-u32-high-grade-outage.alp`。可在 AnyLogic 里打开两个模型，按下表复核。逐行的 Java 替换可见 [副本生成脚本](build_control_model.py)。

| 在副本中的位置 | 停工班设置 | 无停工设置与用途 |
|---|---|---|
| Model → Experiments → Parameter Variation | 名称 `Topology_A_Full_HighGradeOutage`，参数 `Main.topologyWitnessCase = 4001 + index` | 名称 `Topology_A_Full_NoOutageControl`，参数改为 `4002 + index`。设 1 次迭代，关闭并行，采用同一 8 小时班长。 |
| Main → Events | 两个 user-controlled 秒制超时事件 `evHighGradeStop` 和 `evHighGradeResume` 的 Action 调用 `topologyWitness.syncOutage()` | 在副本中移除这两个事件。原有 `evFleetFault` 等业务事件由正常 A 图设置控制。 |
| Java Classes → TopologyWitness → open / 构造函数 | 槽号 4001 映射至停工目录 `configs/O_001.json`，`readOutage()` 读时间和站点 | 只接受 4002，读取本目录的 `configs/C_001.json`，核对 `topology_a_full_no_outage_control`、运行身份、配置字节 SHA、冻结 Full s02/u32 的 SHA，并检查 `config.outage` 明确为 JSON null。 |
| TopologyWitness → start / actualScene | 对两个事件按计划调用 `restart(..., SECOND)`，记录实际 `load_site_outage` 对象 | 移除两个 `restart` 调用，`verifyNormalLoader()` 检查实际铲位。原生场景将 `exogenous_config.load_site_outage` 记为 false。 |
| TopologyWitness → dispatch / finish | 派车前调用 `syncOutage()`，班末封存停复工快照 | 在每次策略请求前及班末执行只读检查。实际 0 号矿点必须位于 5 号节点、中心品位 32%，物理可用且电铲和资源池容量均为 1。若正在装车，检查该车的原生 Delay 正处于运行状态。记录整班检查次数和第 14,400–16,200 秒的检查次数。 |
| AcceptanceEventRecorder → bindControl / 构造函数 | 以 `outage_evidence_sha256` 绑定停机快照；记录器使用停工版标识 | 班末写入 `control_site_id`、`control_available_decisions`、`control_available_window_decisions` 三个核对量。记录器使用 `topology-no-outage-control-v1`，仍执行相同的 FIFO、金属和累计等待记账。 |

**启动前检查。** 使用 [本班独立清单](run_index_seed_manifest.csv) 核对环境种子与停工班一致，并核对本班独立 recording、run、配置 SHA 与输出目录。确认 A 图九车、四座铲位、矿废目标和 u32 的 checkpoint 与已完成停工班一致。

**运行后检查。** 在 `events_manifest.json` 中核对 `control_site_id=0`、`control_available_decisions=native_policy_decisions`，再把 `control_available_window_decisions` 与推理记录在 14,400–16,200 秒内的有效派车数逐一核对。对照班和停工班首次干预前的物理事件、动作及观测按实测记录比对。业务结果采用四个破碎品位窗、全班矿废吨位及累计等待时间。
