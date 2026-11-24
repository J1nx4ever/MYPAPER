# DISPATCH-inspired FIFO 工业基线 · 标准工况 100 个配对班次

当前使用 **prepared_002**。活动 ID 为 `bg128-dispatch-fifo-v3-paired100-20261122-002`，物理 run 为 `10036001–10036100`。100 个班全部运行同一版 Python 策略。AnyLogic 执行运输、装卸、破碎及班末结算。每班的 recording、环境 seed、配置字节 SHA256、对照 run 和两个输出目录都可在同一行查到。 [逐班身份清单](prepared_002/run_seed.csv) · [启动卡](prepared_002/launch_identity.json)

工业 DISPATCH 公布了面向装运的实时优化和破碎站、堆场混配功能。这里实现与本矿山仿真条件一致的公开思路。决策使用现有 1519 维观测和四个装载动作。班次仍按实际 AnyLogic 四个两小时 FIFO 窗、16,000 吨矿石、4,000 吨废石及车辆等待结算。 [Komatsu DISPATCH 功能说明](https://www.komatsu.com/en-us/technology/smart-mining/loading-and-haulage/dispatch)

## 决策

1. 上层 LP 在班内滚动求装点和车型的目标吨流。用可见既有 FIFO 积压随派车时刻变化的函数计算归窗系数。求解器最终发布的非负解须复验硬约束。窗口聚合的 LP 吨流用于指导派车排序。
2. 下层用全车吨量和装点的尚未兑现吨流比较候选，显式检查部分载荷、余量容差和事件同刻顺序。先完成所有合法首动作的共同尾部预测。按工业评分给每个根动作提名至多三个多车组合，轮流评估各根，最多扩展 12 个候选。只有车辆 ID、合法装点和载荷吻合的组合才能进入最终排序。规划安全限为 800 ms，扩展触及安全限时使用已完成全根比较的动作。
3. 每个候选从同一份已承诺状态开始，重放占卸载位、仓容队首、整车预约、卸完入队和 FIFO 破碎。以实际窗边界切分矿吨和金属吨。已有载荷用可见实际品位，未装载批次用可见名义品位。其后的派车使用相同的确定性续派规则。
4. 若存在名义预测中余下各窗皆达到 26%–28%、班末矿废皆达产的候选，在这些候选中按工业吨流匹配损失排序。若均无法同时满足，选原冻结品位、欠产和等待联合成本最低的候选。最终只执行当前车动作，下一次真实请求重新评估。

候选的第 w 窗品位为 g_w = M_w / Q_w，其中 Q_w 是破碎矿吨，M_w 是破碎金属吨。同步考虑仓内矿量 + 已预约吨位 + 当前拟卸载吨位 ≤ 500 吨与破碎速率上限 2,500 吨／小时。以 Q_w > 0 且 0.26 ≤ g_w ≤ 0.28 判定窗口合规。联合成本 J 沿用已有标准工况定义。

```text
J = Σw grade_cost(w)
    + 8 · max(0, 1 − P_ore / 16000)
    + 8 · max(0, 1 − P_waste / 4000)
    + 0.2 · W_vehicle / 28800

grade_cost(w) = min(2, max(0, |g_w − 0.27| − 0.004) / 0.006)
grade_cost(empty window) = 2
```

派车筛选只评估当前及未来可影响的品位窗，并以班末总产量判断达产。到达时间、未装载矿料品位和后续派车采用可见信息下的名义预测，真实品位达标由 AnyLogic 班末结算。

对新分配的车辆 i，用 q_i 表示车载量，v_i 表示装点计划尚能覆盖的吨位，t_i 为派车时刻，τ_i 为首个待满足吨位的需求时刻，f_i 为该时刻的目标吨率。A_i、S_i 分别为到铲和铲服务开始时刻。e_i 为该路线的空驶秒，e_min 为车辆最短空驶秒，T_i 为名义运输循环秒。工业损失在相同长度的派车前缀上求和。

```text
L_i = (q_i − v_i)
    + min(v_i, f_i · |t_i − τ_i|)
    + (q_i / T_i) · (S_i − A_i + e_i − e_min)
```

## 100 班同种子复验

原标准工况五组正式实验结束后，取原 DISPATCH 的 100 个环境种子，按配对槽原顺序逐一复跑修正版。每班新 `recording_id`、`run_index`、配置字节 SHA256、Python 输出和 AnyLogic 事件目录。修正版不消费 Actor 采样种子。旧 DISPATCH 回执用于环境、首帧校验和配对统计。正式运行前核对旧 500 班完整封存、旧 DISPATCH 身份与班末回执。 [逐班冻结身份](prepared_002/run_seed.csv)

对比使用 V3 减 V2 的逐槽联合成本差、配对 bootstrap 95% 区间、完整任务配对差和两种方向的完成差异。报告四窗品位、矿废产量和三个等待分项。验收对比同时保留超时、回退及名义分支统计。配对槽共用起始环境种子和实测首帧，后续事件随策略演化。此次活动验证同一批环境下的算法修订效果。

前 60 槽旧结果参与问题诊断，本试验属于同种子配对复验。主指标为每班 V3 减 V2 的联合成本差，负值对应 V3 成本较低。完整任务要求四窗全部达标且矿废均达产。按环境槽整体重采样 10,000 次，随机种子为 90221，完整任务同时给出精确双侧 McNemar 检验。与同槽冻结 BG128 的差异作为描述性结果。所有已完成班次都纳入统计，分别报告业务表现和工程验收。

旧准备版会把提名的多车组合计作已匹配，即使组合全部被拒绝或比较中途超时。最终版分别记录提名、完成重放、进入最终比较和中选。完整多车匹配须至少一条可执行的多车候选实际进入比较。超时恢复完整首动作方案时，单列已经做过的试算；模型异常进入工程验收。历史 `prepared_001` 的清单和源码快照留存供审计。 [策略入口](fifo_policy.py) · [LP](v3_flow.py) · [匹配损失](v3_matching.py) · [RPC 回执](v3_runtime.py)

## AnyLogic 启动

| 位置 | 原 500 班副本 | 单独复验设置 |
| --- | --- | --- |
| **Additional Class → Baseline1CampaignPlan** | 旧五组配对 helper | 在标准九车模型的独立副本粘贴复验 helper，保留 AnyLogic 生成的 package。核对 `CAMPAIGN_ID` 末尾为 `-002`、`JOB_COUNT=100`。新类让每个 run 绑定对应的种子和配置。 [本次 Java 类正文](prepared_002/Baseline1CampaignPlan.java) |
| **RL_Training → Properties → Number of runs** | 500 | 设置为 **100**。保持 Freeform、串行、关闭 Replications、每班 28,800 秒。 |
| **RL_Training → Properties → Parameters → Freeform** | `experimentRunIndex`、`experimentSeed`、`boundaryRecordingId`、输出路径及场景参数由 helper 提供 | 继续引用替换后的 helper。核对 `currentDispatchPolicy=3`、`usePhysicalObservation=true`、`numTrucks=9`、标称矿源和路网。 |
| **Main → On startup** | ready/Java START 握手 | 沿用按配置字节 SHA256 的启动握手及 `127.0.0.1:50051`。 |

活动根目录为 `runs/bg128-dispatch-fifo-v3-paired100-20261122-002`。Python 每班输出为 `shifts/run_<run_index>/evaluation_receipt.json`，AnyLogic 事件目录配置为 `anylogic_events/run_<run_index>`。逐请求、全量观测和物料事件采集关闭。正式启动包使用本项目绝对路径和固定 Conda 解释器，适用于当前工作站。

预检显示 `PREPARED_AWAITING_V2_COMPLETE` 时，等原运行封存。显示 `PREFLIGHT_PASS_NO_LISTENER` 且端口 50051 空闲后，**Save → Build → Run RL_Training** 一次。

```bash
cd /home/jinx/Faith/experiments/lagrangian_ppo_v1
dispatch_plan_dir=change_plans/token_attention_001/business_graph_summary_009/baseline_comparison_001/dispatch_fifo_v3_001
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B "$dispatch_plan_dir/launch.py" --check
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B "$dispatch_plan_dir/launch.py" --status
```

## 可复现代码和班末验收

在上述实验目录执行以下命令。第一条从三份带 SHA256 封存的 HDF5 独立读取 24 个决策帧，核对动作、车辆、名义可行性和成本，同时重编 Java 并校验 100 行；可用 `--output 新文件.json` 保存本机报告。第二条在正式活动完成后独立复算 100 个配对槽、四窗结算、联合成本、配对统计和工程验收。

```bash
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B "$dispatch_plan_dir/release_checks/reproduce.py"
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B "$dispatch_plan_dir/release_checks/verify_results.py"
```

共同首帧取自标准九车工况。名义模型选择车辆 `(3,0,1)` 的装点前缀 `(Load2,Load0,Load2)`，联合成本约 0.2492；本次真实请求仅向车 3 返回 Load2。车间放行间距、行驶和未装料品位在名义模型中作了简化。重放时限为 800 ms，机器负载可能触发超时恢复。实际达标班数、工程通过数和成本差取自班末 END。Java 原生回退标志的覆盖数按回执另报，标志缺失时保留“未上报”。 [离线复现程序](release_checks/reproduce.py) · [物理终验程序](release_checks/verify_results.py)

本机验收通过 80 项测试，24 帧共 22 种不同观测，全部完成所有合法首动作的比较。零超时，冷启动规划耗时 P95 为 545.0 ms，最大 556.3 ms。复现报告记录本次解释器、包版本、JDK 和实际导入的项目源码 SHA256。 [最终版离线验收报告](../../../../../reports/token_attention_001/business_graph_summary_009/baseline_comparison_001/dispatch_fifo_v3_002/release_reproduction.json)

工作站上的正式物理复验依赖已有项目、原 500 班封存、历史预算及标准模型。测试包内含 100 份配置、Java helper、策略、RPC、配对统计及终验代码，`PACKAGE.json` 列出文件 SHA256。单独复现历史决策可使用便携包，在解压目录使用配有包内所列依赖的 Python 执行 `python -I -B replay.py`。 [100 班正式测试包](../../../../../reports/token_attention_001/business_graph_summary_009/baseline_comparison_001/dispatch_fifo_v3_002/dispatch_fifo_paired100_002.zip) · [可独立解压的算法复现包](../../../../../reports/token_attention_001/business_graph_summary_009/baseline_comparison_001/dispatch_fifo_v3_002/dispatch_fifo_replay.zip)
