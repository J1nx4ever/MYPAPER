# A 图三策略与三档停工的 AnyLogic 配置

将已封存的 `DRL-PPO-baseline-topologydemo-a-full-u32-high-grade-outage.alp` 复制为独立模型 `DRL-PPO-baseline-topologydemo-a-paired-severity.alp`，与原项目共用 3D 资源和 Java 依赖。源文件字节 SHA256 为 `9022e92f066b4e9709e1636a177e10b5efd26dff02d22d56b41962af7ed65280`，30 个配对槽生成的模型为 `f40360615fb24d4083460682091e0afda503ef9baa0d966c6e3439cacf7b1e86`。[逐处的原文与替换代码](build_paired_model.py)

| AnyLogic 对象与位置 | 原值 | 本轮值与核验 |
|---|---|---|
| Model → Name | `DRL-PPO-baseline-topologydemo-a-full-u32-high-grade-outage` | 改为 `DRL-PPO-baseline-topologydemo-a-paired-severity`。确认项目树展开的是副本。 |
| Experiments → Parameter Variation | `Topology_A_Full_HighGradeOutage`，1 次，`Main.topologyWitnessCase = 4001 + index` | 改为 `Topology_A_HighGradePairedSeverity`，270 次，`5001 + index`，串行运行。固定 30 个独立种子，每个种子内依次为 Full、流量匹配与 Flat-PPO，每种策略再按 0、30 和 60 分钟排序。 |
| Java Classes → TopologyWitness → open、构造函数 | `O_001` 与一个固定 Full 检查点 | 按 case code 读取本目录的 `configs/P_001.json` 至 `P_270.json`。逐班校验配置字节 SHA、槽号、策略、停工时长；每次 RPC 必须回传配置与策略 SHA。 |
| Java Classes → TopologyWitness → start、actualScene、syncOutage、dispatch、finish | 两个事件固定在停工班排程，始终发布停工证据 | 正常班不开启计时器，初始实建工况 `load_site_outage=false`。每次派车和班末查验 0 号电铲处于可用状态、容量为 1、装载队列中未发生暂停。30 与 60 分钟工况分别于 14,400–16,200 秒、14,400–18,000 秒触发原生停复工，实建工况记录相应时间。 |
| Main → Events → evHighGradeStop、evHighGradeResume | user-controlled 的秒制事件，触发 `topologyWitness.syncOutage()` | 继续使用相同的事件与派车前同步。已承诺的车辆保留运量、位置和任务。停工时正在作业的卡车逐台调用 `delay.suspend(truck)`，复工前检查原车仍暂停、剩余时间一致，复工时 `delay.resume(truck)`。 |
| Java Classes → AcceptanceEventRecorder → bindSeverity、bindAvailability、bindOutage | 班末封存一个停工证据文件 | 每班绑定 `native_strategy`、`native_policy_id`、`native_policy_update`、`native_duration_minutes` 与策略实际返回的 SHA。正常班另记录全班及 4:00–4:30、4:00–5:00 的高品位电铲可用决策数。停工班封存 `outage_evidence.json` 的 SHA256。 |

**查看停工与恢复。** 0 号装载点为 `ShovelOreHigh1`，中心品位 32%。停工开始前后与恢复前后各有一份快照，可比对已预约吨位、库存、在途和排队车号、铲下车号与剩余装载秒数。通过原生 `WAIT_ENTER`、`WAIT_EXIT`、`LOAD_COMPLETE` 和 `ACTION_APPLIED` 重放停工区间，核对该点的新派车、开装和完装事件。正常班在相应时段的实际装载量可直接衡量同槽停工强度。

**区分等待与作业中断。** 原生排队等待从车辆进入装卸队列起计，开装或开卸时截止。发生在装载中的暂停时间属于作业中断；需要量化时可由触发时正处于装载的车数乘实际停工时长得到车辆·秒。

**编译核验。** 270 次串行实验、两个 Java 类及原生 Main 与 AnyLogic 8.9.0 的真实库在 Java 17 通过编译。额外用独立的组件替身针对本模型实际 `syncOutage()` 代码验证空闲与忙碌、重复触发、剩余装载 15 秒、0/30/60 分钟六种组合。真实车辆的暂停、排队、恢复和吨位守恒以每个物理班的原生封存证据为准。
