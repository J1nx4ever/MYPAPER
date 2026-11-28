# 动态出勤 u9 的 A/B 冻结策略试跑

树形 A 图出勤 9 辆车，环形 B 图出勤 7 辆车，班长均为 8 小时。按动态出勤训练的口径，两组各自按开班车队额定载重锁定矿废目标。原试跑把九车目标也传给了七车 B，请使用独立的 B 运力缩放实验核对该组结果。模型使用动态出勤训练第 9 次更新封存的 Actor，权重 SHA256 为 `b13e86feb74656def9077e412dbf4443857fba777ecec5e8ac758fd79a191f36`。每次在物理可派装载点中取概率最高的动作，车位和铲位的排列种子为 0。 [B 组开班目标与核验](capacity_scaled_001/README.md)

## AnyLogic 中的改动与验证

A 图九车与 B 图七车各 50 个随机工况班的验证结果见 [100 班统计报告](/home/jinx/Faith/experiments/lagrangian_ppo_v1/reports/token_attention_001/business_graph_summary_009/variable_topology_training_001/frozen_policy_trial_001/paired_50_001/summary.md)。两组按开班运力分别锁定吨位目标。 [实验设计与全部运行身份](paired_50_001/README.md)

打开 [试跑模型](/home/jinx/Models/DRL-PPO-baseline-topologydemo/DRL-PPO-baseline-topologydemo-frozen-policy-trial.alp)。项目树中的 `Topology_Frozen_Policy_A` 和 `Topology_Frozen_Policy_B` 各绑定一份独立的开班配置。两份已使用的身份供历史复核。

| 位置 | 完整班固定规则版 | 冻结策略试跑版及验证 |
|---|---|---|
| `Main → obsInit` 和 `collectPhysicalObservation` | 旧观测默认读取固定的 16 条有向路段 | 从本班实建道路读取正、反向路段，按编译顺序写入 16 槽，不足补零。开班校验端点，推理时逐帧校验端点、路长及限速 |
| `TopologyWitness → start / instantiateFleet / dispatch` | 调用固定派车规则 | 初始化物理观测，向本地 Actor 发送当时的 1519 维观测及车队掩码。B 图停运两槽采用训练时的停车状态表示 |
| `TopologyPolicyDispatch → choose / committed` | 本地估算运量和品位后选铲 | 校验 Actor 回应中的运行配置、权重、派车序号，再由原生状态机验证资格并预留载荷。原生收据记录实际派车及车辆 ID |
| `AcceptanceEventRecorder → finishPolicyTrial` | 结束后封存物料与路段记录 | 附加实际 Actor 派车次数和 checkpoint SHA。复核器逐项比对模型响应、实际应用动作、车号和时间 |

[ALP 生成器](prepare_trial_alp.py) · [Java 派车接口](TopologyPolicyDispatch.java) · [冻结模型及图适配器](frozen_topology_policy.py)

## 配置与复核

项目使用 `anylogic-env`。准备器 `prepare_trial.py --model ALP` 核对已通过的见证、动态 u9 权重、推理源码及运行身份后生成配置。 [两班启动身份](launch_identity.json)

`trial_service.py` 在本机回环接口接受两个实验发出的派车请求，使用 CPU 执行冻结策略。原生模拟记录四个窗口与 FIFO 物料事件，Python 仅记录用于对账的派车序号、观测摘要、掩码、所选动作、车号与时间。班末运行 `collect_trial.py --collect` 独立重放并核对决策和物料账。物理完成状态、推理一致性和业务是否达标分别记录，未达标的自然完整班同样计入结果。
