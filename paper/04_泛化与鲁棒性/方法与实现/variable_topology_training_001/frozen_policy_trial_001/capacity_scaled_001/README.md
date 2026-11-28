# B 图按开班运力缩放目标

环形 B 的车队为 3 辆 91.7 t、2 辆 147 t、2 辆 193 t 矿卡。7 辆出勤车的额定载重合计 955.1 t，9 个槽位的满编参考为 1,295.1 t。与冻结动态 u9 的训练设置一致，开班时用二者的比值缩放 16,000 t 矿石目标、4,000 t 废石目标和 6,000 t 废石停派线。超量容差为一辆大车的 193 t。 [训练目标生成方法](../../../fleet_outage_resilience_001/schedule.py)

| 班内约束 | B 图七车 |
| --- | ---: |
| 矿石班目标 | 11,799.55 t |
| 两小时矿石目标 | 2,949.89 t |
| 废石班目标 | 2,949.89 t |
| 废石停派线（实卸加已承诺） | 4,424.83 t |
| 废石允许上限 | 4,617.83 t |
| 四个两小时品位窗 | 每窗 26%–28% |

### 实际结果

10054046 班在 28,800 秒正常结束。冻结动态 u9 完成 155 次派车，逐次核对了策略实际收到的目标、可选装点及实际下发动作。矿石实破 15,161.85 t，完成目标的 128.50%。废石实卸 4,490.80 t，超过停派线 65.97 t，在单车允许上限以内。四窗品位分别为 27.44%、28.33%、27.17%、27.56%。第二窗超过 28% 上限 0.33 个百分点，本班共 3/4 窗合规。 [原生班末清单](/home/jinx/Faith/experiments/lagrangian_ppo_v1/runs/site-layout-frozen-policy-scaled-20261127-001-B/anylogic_events/events_manifest.json) · [逐班复核](/home/jinx/Faith/experiments/lagrangian_ppo_v1/reports/token_attention_001/business_graph_summary_009/variable_topology_training_001/frozen_policy_trial_001/capacity_scaled_001/trial_evaluation_B.json)

### AnyLogic 手工核对

打开缩放版模型，项目树选中 `Topology_Frozen_Policy_B_Scaled`，参数 `topologyWitnessCase = 2`，单次运行。 [缩放版 ALP](/home/jinx/Models/DRL-PPO-baseline-topologydemo/DRL-PPO-baseline-topologydemo-frozen-policy-scaled.alp)

| 位置 | 原实现 | 缩放版 |
| --- | --- | --- |
| `TopologyWitness → DIRECTORY` | 读取通信修复版 `repair_001/configs` | 读取本目录 `configs/B.json` 和 SHA256。准备器按开班车辆名册算出完整精度目标 |
| `TopologyWitness → start()` | 从旧任务读取三个班级吨位参数，两小时矿石目标使用原默认值 | 将三个班级参数读为 double，并设置 `main.targetOreTons2h = main.targetOreTonsShift / 4.0;`。实例化车队后、激活车辆前调用 `verifyOpeningTargets(task)` 核对实际运力、四个吨位和两小时目标 |
| `TopologyWitness → actualScene()` | `Math.round(...)` 对吨位取整 | 直接写入吨位 double，包括 `ore_target_tons_2h`，开班回执与 JSON 逐项核对 |
| Python 推理服务 → `ScaledTrial.decide()` | 校验车号、路网及动作掩码 | 另核对 Actor 每次实际收到的矿废目标，以及由累计实卸量计算的剩余废石缺口 |

生成器列出了逐项替换的 Java 代码。运行身份 10054046 独立于原 B 班，采用相同路网、装备、无故障确定性工况和冻结权重。班后可从本目录执行 `/home/jinx/miniconda3/envs/anylogic-env/bin/python -B collect_scaled.py --case B` 只读重放校验。 [ALP 生成器](build_model.py) · [开班目标计算](opening_targets.py) · [新班身份](launch_identity.json)
