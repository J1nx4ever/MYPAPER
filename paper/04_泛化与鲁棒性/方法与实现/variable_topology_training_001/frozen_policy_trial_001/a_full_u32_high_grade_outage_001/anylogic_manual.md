# 标准 A 图高品位点停工的 AnyLogic 手工核对

实验仅运行一个 8 小时班，标准 A 图九车出勤，冻结 Full BG128 的 s02/u32。现场装载点 0 为 ShovelOreHigh1，路网节点 5，坐标 (800, 600) m。开班时读取四个装载点的实际中心品位及位置，核对该点为唯一最高品位矿点，中心 32%，料块波动范围 31%–33%。

从已封存的 `DRL-PPO-baseline-topologydemo-b-finetune.alp` 制作新副本 `DRL-PPO-baseline-topologydemo-a-full-u32-high-grade-outage.alp`。旧模型字节 SHA256 为 `60bedf6c07e6f75058597cfa928faf31e289a47740394d34d62a3657c610211c`，本轮新模型为 `9022e92f066b4e9709e1636a177e10b5efd26dff02d22d56b41962af7ed65280`。手工对照按下表定位对象，生成器与 Java 方法文件给出每处替换的完整代码。 [模型生成器](build_outage_model.py) · [装载中断与恢复方法](outage_methods.java.inc)

| AnyLogic 项目树位置 | 原值或代码 | 本轮值或代码及影响 |
|---|---|---|
| Experiments → Parameter Variation | `Topology_B_Test50`，50 次，`Main.topologyWitnessCase = 3001 + index`，另有四个训练及一个验证入口 | 新建独立实验 `Topology_A_Full_HighGradeOutage`，1 次，`4001 + index`，串行、单次重复，仿真终点沿用 28,801 s。28,800 s 由模型结班。 |
| Main → Events | `evFleetFault` 为已有秒制、user-controlled 超时事件 | 以此为模板增加 `evHighGradeStop` 和 `evHighGradeResume`，Action 均为 `if (topologyWitness != null) topologyWitness.syncOutage();`。 |
| Java Classes → TopologyWitness → open、构造函数、start | 从原训练、验证、测试槽读取 B 图与检查点 activation | 新案例映射到 `configs/O_001.json`，检查字节 SHA、A 图布局 `tree_A`、角色、Full s02/u32 的固定 SHA 和完整开班场景。由 `outage.start_seconds` 与 `end_seconds` 启动两个事件。本轮采用第 14,400 至 16,200 秒。|
| Java Classes → TopologyWitness → dispatch | 校验车辆后直接调用 `dispatcher.choose` | 在 Actor 推理前调用 `syncOutage()`。同刻派单也能读取对应可用状态和动作掩码。 |
| Java Classes → TopologyWitness → syncOutage | 正常开采期间，LoadSite.available 为 true，shovelCapacity 和 shovelResource.capacity 为 1 | 关闭时置 available=false、shovelCapacity=0，逐辆 `delay.suspend(truck)`，再设置 `shovelResource.set_capacity(0)`。恢复时重设两个容量，继续同一辆车剩余的延迟。LoadSite 自身的 getAvailableTons、getDispatchableTons 和 appendDispatchObservation 将变化传给既有策略字段。 |
| Java Classes → TopologyWitness → actualScene、finish；AcceptanceEventRecorder → outageCursor、bindOutage | 记录开班设施、矿量、动作、FIFO、原生检查点等 | 开班证据加入实际装载点和定时配置。停工前后及恢复前后各抓取一份快照。在班末将四份快照写入 `anylogic_events/outage_evidence.json`，并将 SHA 写入原生 `events_manifest.json`。 |
 
**核对物理效果。** 停工开始后，在途车辆继续到场，持有的预约吨位及排队次序随原生流程保存。已开工车辆的剩余作业秒数冻结。恢复时先恢复原车的计时，然后开放装载资源，避免重新抽样装载时长或同一批矿石品位。原生事件序号可以区分同一时刻先后发生的操作。逐项比对暂停和恢复快照的产矿量、金属量、剩余量、预约吨位及采掘游标，同时核对停工区间无新派单、装载开始及完成。若触发时铲位空闲，快照中的服务车清单为空。

**核对运行。** 使用已封包的 O_001 案例，在已有控制器启动该案例时观察这个实验。班末检查 action 与原生记账一致、0 次备用或非法派单、8 小时物料及金属守恒、四个 2 小时品位窗的结果。已完成的运行通过回执核对。各班配置 SHA、run/seed、recording 与两个输出目录由本活动清单给出。
