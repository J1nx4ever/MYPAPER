# BG128 标称单班验证 · DISPATCH-inspired

标称 8 小时班 `10022001` 已结束并通过工程验收。Python 在既有 BG128 决策槽运行流量规划和车辆匹配。AnyLogic 执行物理运输、装卸、FIFO 破碎和四个品位窗结算。 [冻结启动身份](prepared_001/launch_identity.json) · [实际单班验收](../../../../../reports/token_attention_001/business_graph_summary_009/baseline_comparison_001/dispatch_inspired_001/single_shift_validation.json)

## 策略怎样决策

公开 DISPATCH 架构依据路网出行时间计算路线，以 LP 生成目标流量，再用车辆可用时间和装载点需求匹配派车。论文可使用 **DISPATCH-inspired flow planning and truck assignment**，并明确下面的窗口品位约束、15 分钟重规划和三车匹配均为本研究对 BG128 的实现选择。 [Modular Mining 专利公开说明，第 12 页](https://patentimages.storage.googleapis.com/60/fe/7c/92aa3718a55c42/US20090096637A1.pdf#page=12) · [车辆匹配，第 13 页](https://patentimages.storage.googleapis.com/60/fe/7c/92aa3718a55c42/US20090096637A1.pdf#page=13)

1. **名义路线。** 读取同一份 1519D 物理快照、实时合法掩码和现有路网时间表。按空车到装载点、重车到对应卸点、装卸服务估计名义循环。预测台账先推演已经承诺的任务和当前 FIFO 存量。 [现有名义模型](../../../../../traditional_dispatch_v1/nominal.py) · [策略实现](dispatch_policy.py)
2. **流量规划。** 在各剩余品位窗 \(p\) 上求车型 \(k\) 至装载点 \(j\) 的派出吨数 \(x_{kjp}\ge 0\)。装载点库存、车时、装卸能力、破碎能力及 6000 t 废石承诺停派约束限制可排吨数。预计破碎矿吨和金属吨为 \(Q_w=Q_w^0+\sum_{kjp}a_{w,kjp}x_{kjp}\) 和 \(M_w=M_w^0+\sum_{kjp}g_j a_{w,kjp}x_{kjp}\)，其中系数 \(a\) 用均匀派出、名义卸载和可见仓余估计各窗兑现比例。约束 \(M_w\ge0.266Q_w-s_w^-\)、\(M_w\le0.274Q_w+s_w^+\)，并为每个未结束窗口设一个最小车型载量的软供料目标。目标最小化矿石 16,000 t 和废石 4,000 t 的归一化欠产、品位松弛、供料缺口与占用车时的加权和。每隔 900 仿真秒到达可规划请求时重算，窗口、路网或资源可用状态、废石停派状态变化也触发。窗口最终成绩由真实 FIFO 破碎结算。 [流量规划与目标权重](flow_plan.py)
3. **实时匹配。** 将计划按站汇总为速率 \(r_j(t)\)。令 \(A_j\) 为本轮计划以来获 RPC 确认的派出吨数，站点需求时刻为 \(d_j=\inf\{t:\int_{t_0}^t r_j(u)\,du>A_j\}\)。对当前车以及最多两辆预计即将可调度的车辆，枚举合法车铲匹配，用铲位可用时间、现有排队和已发运到达预测服务时刻。匹配分数由等候、相对最短空驶路线的额外行驶时间、以及偏离目标流量需求的吨位损失组成。只向 AnyLogic 返回当前车的装载点。其他车辆仍在各自真实请求时重新决策。单一合法点直出。遇计算截止取已完成的合法匹配，预测无可用解时按短 ETA 回退。单次规划预算 800 ms，RPC 响应验收目标 1,000 ms。 [匹配与已确认派出量](dispatch_policy.py)

上层 LP 的目标为

\[
\min\quad
8\frac{u_{\mathrm{ore}}}{16000}
+8\frac{u_{\mathrm{waste}}}{4000}
+\sum_w\left(\frac{s_w^-+s_w^+}{0.006\,Q_w^{\mathrm{ref}}}
+2\frac{b_w}{q_{\min}}\right)
+0.02\frac{\sum_{kjp}\tau_{kj}x_{kjp}/q_k}{9\times28800}.
\]

\(u\) 是班末欠产，\(b_w\) 是当窗到最小车型载量 \(q_{\min}\) 的供料缺口，\(Q_w^{\mathrm{ref}}\) 按该窗的已有破碎量及可用破碎能力归一化，\(\tau_{kj}\) 是名义循环秒数。下层对每个候选匹配累计各车的吨位损失。记 \(a_i\) 为装载点到达时刻，\(f_i\) 为匹配后的铲位可用时刻，\(s_i=\max(a_i,f_i)\) 为服务开始，\(e_i-e_i^{\min}\) 为相对最短合法空驶的额外时间。每辆车的分数为

\[
\frac{q_i}{\tau_i}\bigl[s_i-a_i+e_i-e_i^{\min}\bigr]
+\min\left(q_i,\,r_j(d_j)\,\left|s_i-\max(f_i,d_j)\right|\right).
\]

当站点本轮的目标吨量已用尽，第二项取 \(q_i\)。匹配按总分择优，同分按站点索引序列确定。 [LP 系数与资源余量](flow_plan.py) · [匹配评分和同分处理](dispatch_policy.py)

## 冻结身份

下列目录均相对 `/home/jinx/Faith/experiments/lagrangian_ppo_v1`。Java helper 按清单自动填入，无需在 AnyLogic 逐项输入。

| 字段 | 本班值 |
| --- | --- |
| recording_id | `bg128-dispatch-inspired-20261122-001` |
| Java index / run_index | `0` / `10022001` |
| 环境种子 | `29555357148808` |
| Actor 采样 | 确定性策略，保留字段 `0`，运行时不消费 |
| 配置文件字节 SHA256 | `268c6fbf6359c96dd0f93ea62752b68af86381dd72c4c8f0585677833b518c69` |
| Python 输出根 / 单班产物 | `runs/bg128-dispatch-inspired-20261122-001/` / `shift/` |
| AnyLogic 事件根 | `runs/bg128-dispatch-inspired-20261122-001/anylogic_events/` |

[启动身份及绝对路径](prepared_001/launch_identity.json) · [配置字节](prepared_001/phase_config.json) · [run/seed 清单](prepared_001/master_plan.csv)

## AnyLogic 操作

打开实际运行副本 `DRL-PPO-baseline1.alp`。本次需要替换一个 Java 类正文并设置一次运行。旧单班四组、正式 400 班和本活动各用自己的 helper。若另一活动还占用 `127.0.0.1:50051`，在它正常结束后再启动本班。 [模型位置](../../../../../anylogic/3d/DRL-PPO-baseline1/DRL-PPO-baseline1.alp)

| 界面位置 | 修改前 | 修改后及验证 |
| --- | --- | --- |
| `Additional Class → Baseline1CampaignPlan → 类正文` | 当前运行副本的活动类。仓库快照中的旧类包含上轮单班验证的活动标识。 | 粘贴下方冻结 Java 文件全文。文件从注释和 `public final class` 开始，保留 AnyLogic 生成的 package。核对 `CAMPAIGN_ID="bg128-dispatch-inspired-20261122-001"`、`JOB_COUNT=1`。类负责工况、身份和 Python 启动握手。 |
| `RL_Training → Properties → Number of runs` | 当前运行副本的值。仓库快照为 `3448`，已完成四组单班的运行副本通常设为 `4`。 | 设为 **`1`**，只运行 `index=0`。保持 Freeform、串行、关闭 Replications，班末 `28800 s`。 |

[复制此 Java 类正文](prepared_001/Baseline1CampaignPlan.java) · [仓库中的实验设置](../../../../../anylogic/3d/DRL-PPO-baseline1/DRL-PPO-baseline1.alp#L25553)

核对以下字段。任何已有覆盖按给出的表达式恢复。表内 helper 表达式均以 `Baseline1CampaignPlan.` 开头。

| 位置 | 字段 | 表达式或值 |
| --- | --- | --- |
| `RL_Training → Properties → Parameters → Freeform` | `currentDispatchPolicy` / `usePhysicalObservation` | `dispatchPolicy((int) index)`，返回 `3` / `true` |
| 同上 | `experimentRunIndex` / `experimentSeed` | `runIndex((int) index)` / `environmentSeed((int) index)` |
| 同上 | `boundaryRecordingId` / `boundaryEventRoot` / `materialEventRoot` | `recordingId((int) index)` / `eventRoot((int) index)` / `materialRoot((int) index)` |
| 同上 | `enableBoundaryEventRecording` / `enableMaterialEventMemory` / `offlineNativeCollection` | `eventRecordingEnabled((int) index)` / `materialEnabled((int) index)` / `offlineEnabled((int) index)`，三项均返回 `false` |
| 同上 | `freeSpeedFactorStd` | `freeSpeedFactorStd((int) index)`，返回 `0.05` |
| `Main → Variables` | `dispatchInferenceHost` / `dispatchInferencePort` / `enableOnnxDispatch` | `"127.0.0.1"` / `50051` / `true`，启用已有 gRPC 派车路径 |
| `Main → On startup` | 配置 SHA 绑定 | 第一段保持已有 `boundaryAcceptanceConfigSha256 = Baseline1CampaignPlan.awaitConfigSha256(...)` |

[Freeform 策略与种子](../../../../../anylogic/3d/DRL-PPO-baseline1/DRL-PPO-baseline1.alp#L25636) · [事件记录配置](../../../../../anylogic/3d/DRL-PPO-baseline1/DRL-PPO-baseline1.alp#L25700) · [Main 启动握手](../../../../../anylogic/3d/DRL-PPO-baseline1/DRL-PPO-baseline1.alp#L27)

在终端执行只读预检，看到 `PREFLIGHT_PASS_NO_LISTENER` 后，在 AnyLogic **Save → Build → Run `RL_Training`** 一次。Java 首班自动拉起 Python，然后等待 ready 并绑定配置字节。逐请求事件和全量快照记录保持关闭。

```bash
cd /home/jinx/Faith/experiments/lagrangian_ppo_v1
plan=change_plans/token_attention_001/business_graph_summary_009/baseline_comparison_001/dispatch_inspired_001
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B "$plan/launch.py" --check
```

运行后可用同一条命令把 `--check` 换成 `--status`。查看 `runs/bg128-dispatch-inspired-20261122-001/launch/controller.log`，期望出现 `JAVA_START_VERIFIED` 和 `COMPLETE run_index=10022001 policy=dispatch_inspired`。看到 `STOP` 时保留目录、日志与账本，由新身份处理确需重跑的班次。

## 单班通过条件

- 真实 START 核对 run、环境 seed、配置字节、标称工况指纹和 1519D schema。真实动作均在掩码内，RPC 下一请求回显上个动作，再计入派出吨数。
- 正常完成 28,800 s、收到 END 回执，四个品位窗各结算一次。窗口和全班矿吨、金属、废石与等待按现行账本核对，业务违规和欠产照常记录。
- 单班报告的 `engineering_acceptance.passed=true` 要求上述闭环完成，至少一次上层计划成功、至少一次实时多车匹配，且所有 RPC 响应满足 1 s 目标。报告汇总响应时延、回退率、各装载点动作数量和实际联合成本。

运行根的 `suite_complete.json` 指向已封存报告。557 个历史快照验证了动作合法性，其中 190 个在多车与单车模式间给出不同的当前动作。已完成的实际单班用于核验策略调用和物理结算。纳入正式表时，可按原正式实验的 100 个环境槽扩展这一基线，每个新物理班绑定新 run、recording 和配置，并为跨活动复用环境种子明确登记槽级授权。 [实际结算和时延](../../../../../reports/token_attention_001/business_graph_summary_009/baseline_comparison_001/dispatch_inspired_001/single_shift_validation.json) · [离线证据](../../../../../reports/token_attention_001/business_graph_summary_009/baseline_comparison_001/dispatch_inspired_001/offline_preflight.json) · [四组正式方案](../paired_100_001/README.md)
