# BG128 九槽车队出勤与班中退场训练

研究问题是在相同开班计划、单车故障和评价口径下，加入班中退场训练能否比仅训练班初出勤变化更好地守住品位并减少欠产。基线为 `/home/jinx/Models/DRL-PPO-baseline2/DRL-PPO-baseline2.alp`，权重来自完整混合 BG128 的 `s02/u32`。实验使用 6–9 辆出勤车、4 个物理装载点、8 小时班长。

## 本轮 992 班

| 阶段 | 批次或配对槽 | 班数 | run_index |
|---|---|---:|---|
| 共同热身 | 4 次更新，每次 32 班，每个初始车数各 8 班 | 128 | 10043001–10043128 |
| 静态训练 | 热身后 9 次更新，每次 32 班 | 288 | 10043129–10043416 |
| 动态训练 | 同一热身模型后 9 次更新，每次 16 个健康班、16 个故障班 | 288 | 10043417–10043704 |
| 验证 | 36 个独立槽，每槽静态、动态策略各跑有故障和无故障两班 | 144 | 10043705–10043848 |
| 测试 | 另取 36 个独立槽，沿用四方配对 | 144 | 10043849–10043992 |

活动名为 `bg128-fleet-resilience-20261124-001`。每班的 recording、配置**文件字节** SHA256、环境与动作采样 seed、Python 输出路径、AnyLogic 事件路径已列在[启动清单](prepared_001/run_seed.csv)。AnyLogic 读取[主表](prepared_001/master_plan.csv)和[装点与道路场景表](prepared_001/scenario_table.csv)。[封存文件](prepared_001/suite.json)及其[字节 SHA256](prepared_001/suite.sha256)绑定所有班次。准备包不占用物理班次。实际启动时由 [resilience_budget.py](resilience_budget.py) 的 `budget_snapshot` 合并历史预算并核验 run 与 seed。

## 在 baseline2 中操作

原活动在 32 班后遇到首次 PPO 概率回放错误。本轮补 32 班的**实际启动入口**使用[修复版操作清单与 CampaignPlan](replay_repair_001/README.md)。

针对已手工保存的 baseline2，见[保存版本复核与修正](saved_alp_audit_001/README.md)，其中补充了六车班单车容差的精确上报方式。

按[逐对象手工清单](manual_anylogic.md)设置 Main、Truck、DumpSite 和 RL_Training，再在 IDE 中 Build。使用本轮生成的 [Baseline1CampaignPlan.java](prepared_001/Baseline1CampaignPlan.java)。`RL_Training` 设置为 **992 次串行运行**，从索引 0 开始。Java 在 Main 启动时校验主表、场景表、Python 桥和本班配置 SHA，并按序领取物理 run。启动前可在本目录只读核对：

```bash
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B launch.py --suite prepared_001/suite.json --check
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B verify_package.py --suite prepared_001/suite.json
```

运行从 IDE 的 `RL_Training` 启动。训练与通信以实时 RPC 配置、schema 指纹及班次身份为准。默认关闭逐请求、逐 transition、全量观测、物料事件文件和 rollout 快照。每班回执保存四窗、开班目标、实际退场时刻、终端物料对账和训练完整性计数。项目 `runs/bg128-fleet-resilience-20261124-001/` 下的 `shifts/run_<run>/` 是 Python 输出，`anylogic_events/run_<run>/` 是每班预分配的事件路径，`models/<arm>/uXXXX/` 保存模型与更新摘要。`launch.py --suite prepared_001/suite.json --status` 可查状态。若发生中断，依据物理领取、回执和已封存更新边界核对；尚在内存的半个训练批不能安全重放。

首班为 `10043001`，recording 为 `bg128-fleet-resilience-20261124-001-10043001`，环境 seed 为 `235067115305789`，采样 seed 为 `159572978294450`，配置字节 SHA256 为 `26b2c716a0a6b77c3910e7c0875b2555cb23b1ff59f276e35ec993de6e5231ab`。封包时已有 42,022 班登记，若启动前其他任务没有新增登记，启动后合计 43,014 班。[首班配置](prepared_001/configs/run_10043001.json)

## 如何训练

- **热启动。** Actor、Critic 原输入投影按原尺寸执行，将两个状态位相对满编正常值中心化，用零初始化的无偏置投影加到车辆和全局表示。沿用原特征归一化和物理字段顺序。共同热身建立新优化器。分支从热身 u4 复制权重，两边都重建相同设置的优化器，分支编号从 u0 到 u9。迁移验证直接回放旧权重与新 u0，检查每个合法动作、动作分布和价值。[迁移代码](fleet_policy.py) · [实测数据](../../../../reports/token_attention_001/business_graph_summary_009/fleet_outage_resilience_001/u0_equivalence.json)
- **初始出勤与布局。** 每种车型先选两辆，其余 0–3 个位置按车型无放回抽取，车型内抽实际车号。四个装载点轮换高、中、低品位矿和废石角色，逐班生成品位与波动范围。物理道路、装点 ID 和约定的 RPC 观测坐标一致。[工况生成](schedule.py)
- **索引置换。** 每班由开班名单与 Actor 采样 seed 派生独立的排列流，打乱开班车辆和四装点在图中的位置，班内固定。同步变换道路与预约边、请求车、候选装点和掩码，动作回传前还原物理装点号。父模型显式编码车号、站点及地理引用，C128 也保留业务顺序。因此验收覆盖图张量顺序等变，物理 ID 的影响仍存在。[置换下的迁移复核](../../../../reports/token_attention_001/business_graph_summary_009/fleet_outage_resilience_001/u0_permuted_equivalence.json)
- **两组对照。** 两分支每个对应训练班的初始出勤、装点角色与品位参数相同。训练的物理随机流与 Actor 采样流逐班独立。动态组每批的 16 个故障班中，7、8、9 辆开班及三种退出车型各按 5/5/6 轮换，四个废石点各四班。每个九更新周期内，36 种三因素组合各出现四次。触发时刻在开班后 5400–19800 秒连续抽取，目标车从指定在役车型中选定。计划中的故障车号和未来时刻只进入仿真与核验。
- **吨位与成本。** 九车总额定载重 1295.1 吨。本班出勤载重比例为 α，班初锁定矿石目标 16000α、废石目标 4000α、废石停派线 6000α 吨。停派线上界沿用单大车的 193 吨超量容差。总矿量按实破、废石按实卸计算。班末欠产成本取两种相对缺口的正部，各乘 8 再相加；达标线处连续。品位按原四个两小时窗结算，等待按原单 Critic 的成本口径。训练保存各分项成本、Actor 与 Critic 梯度及分层样本数。本轮父模型采用固定成本权重，无动态欠产乘子。[训练实现](fleet_policy.py)
- **评估。** 各分支固定取末次 u9。在每个验证或测试槽内，四次物理运行有独立 run、recording 和配置，同槽共享环境及 Actor 采样 seed；两个数据集之间隔离。两个集合各覆盖 36 种车数、退场车型和废石点组合。验证阶段采用预先固定的零百分点容差，先要求动态组在健康和故障两类四窗全合规率、及双方均有有效退场窗的配对合规率上均不下降，再要求动态组故障班的平均矿废**相对缺口之和**严格更小，否则选静态组。此规则用样本点估计选模；独立测试同时报告两组及各自有效样本数。[配对分析](reports.py)

## 物理边界与评价口径

| 环节 | 规则 |
|---|---|
| 开班至触发 | 在役车状态为 (1,1)。同一模型时刻先完成所有到期卸载记账，再取消目标车资格，随后处理合法派单。 |
| 触发至实际退出 | 状态为 (0,1)。该车不能开始下一趟。当前预约、道路占用、排队、实载及品位保留，载废车辆持续计入废石承诺量。 |
| 实际退出 | 满足停稳、路段及服务资源释放、载荷和预约清空后才进入 (0,0)。清理前后车引用，屏蔽该车的图消息和池化。故障后仍在途到班末的车保留当时物理状态。 |
| 退场后首个完整窗 | 由真实破碎 FIFO 恒品位处理区间与 `[t_exit, t_exit+7200)` 的交集累加吨位和金属量。仅实际退出且 `0 ≤ t_exit ≤ 21600` 纳入。完整零产窗计失败，晚退、未退及健康班分别计数。26% 与 28% 均算合规。 |
| 入仓账本 | 开班 FIFO 为空。END 时以卸点累计入仓、累计已破碎、逐批实盘的 FIFO 残余作三方比较，分别检查矿石和金属两条恒等式。绝对容差取 RPC 的质量守恒阈值，最多 1e-5 吨。 |
| 报告 | 各组给出四窗全合规率、滚动窗合规数/有效数、空窗数、联合达标、实际产量、相对和绝对缺口、等待和同槽故障产量损失。每组的有效分母及两组共有的配对分母分别保留。 |

[实时 RPC 接收与按班结算](resilience_rpc.py) · [训练编排](resilience_runtime.py) · [终端验收](physical.py) · [离线日程与边界检查](verify_design.py)

离线验收涵盖 992 行 Java 参数、实际历史决策的热启动和置换等价、RPC 回放、模拟撤车、合成训练更新以及矿石和金属对账。界面手工接线后的物理效果由本轮运行实测。[验收汇总](../../../../reports/token_attention_001/business_graph_summary_009/fleet_outage_resilience_001/package_acceptance.json) · [baseline2 Java 编译及边界用例](../../../../reports/fleet_outage_resilience_001/anylogic_offline.json)
