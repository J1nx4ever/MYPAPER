# 冻结 BG128 s02/u32 的六车零样本泛化操作

活动为 `bg128-full-s02-u32-fleet6-zero-shot10-20261124-001`，物理 run 为 `10042001–10042010`。沿用此前六车验证的均衡停运组，开班即停运 **2、5、8**，在役车为 **0、1、3、4、6、7**。三种车型各保留两辆。`numTrucks=9` 表示保留 9 个 Truck 槽，实际在役 6 辆。策略使用九车训练的完整 BG128 冻结 `s02/u32`，十班均为标准矿源和道路的 8 小时评估，矿废产量目标、四窗及品位条件沿用标准工况。 [十班身份清单](prepared_001/run_seed.csv) · [启动卡](prepared_001/launch_identity.json)

## 副本接入

以下修改在用户指定的 [DRL-PPO-baseline2.alp](/home/jinx/Models/DRL-PPO-baseline2/DRL-PPO-baseline2.alp) 中手工完成。已核对其保存版本当前绑定 20 班在线 MILP 对照，尚无停运掩码、停运函数与 RPC 停运字段。首次接入请按下表更新 helper，并补齐下一节的七段停运逻辑。

| 对象与属性 | 原值或现有写法 | 本次设置、作用与核对 |
| --- | --- | --- |
| **Additional Class → Baseline1CampaignPlan → 类正文** | 副本保存值为 `bg128-or-milp-paired10-20261124-001`、`JOB_COUNT=20` | 粘贴[本次 helper](prepared_001/Baseline1CampaignPlan.java)类正文，保留 AnyLogic 管理的包声明。核对新活动名、`JOB_COUNT=10` 和首班 run `10042001`。每班取独立的身份与配置 SHA256。 |
| **RL_Training → Properties → Number of runs** | 副本保存值为 `20` | 设为 **10**。核对 Freeform、串行、关闭 Replications、`Stop at=28800` 秒。 |
| **Main → Parameters → fleetDisabledTruckMask** 和 **RL_Training → Parameters → Freeform** | 标准模型尚无该参数。已接入的六车副本可能绑定旧 helper | 有参数时直接复用，缺失时新建 `int`、初始值 `0`。本次实验的 Freeform 表达式填 `Baseline1CampaignPlan.fleetDisabledTruckMask((int) index)`，班内值为 `292`。 |
| **RL_Training → Parameters → Freeform → 运行身份** | 由旧 helper 供值 | 确认 `experimentRunIndex=Baseline1CampaignPlan.runIndex((int) index)`、`experimentSeed=Baseline1CampaignPlan.environmentSeed((int) index)`、`boundaryRecordingId=Baseline1CampaignPlan.recordingId((int) index)`、`boundaryEventRoot=Baseline1CampaignPlan.eventRoot((int) index)`。与逐班清单核对。 |
| **RL_Training → Parameters → Freeform → 派车、工况和观测** | 标准 Python RPC 与九槽物理观测 | `currentDispatchPolicy=Baseline1CampaignPlan.dispatchPolicy((int) index)` 返回 3，`usePhysicalObservation=true`，`numTrucks=9`，`freeSpeedFactorStd=Baseline1CampaignPlan.freeSpeedFactorStd((int) index)` 返回 0.05。 |
| **RL_Training → Parameters → Freeform → 记录** | 由旧 helper 提供低频设置 | `enableBoundaryEventRecording`、`enableMaterialEventMemory`、`offlineNativeCollection` 分别调用新 helper 的 `eventRecordingEnabled`、`materialEnabled`、`offlineEnabled`，均返回 false。事件目录用于运行身份绑定，详细事件录制关闭时该目录可为空。 |
| **Main → Agent actions → On startup → 首段** | 已调用 `awaitConfigSha256` | 核对首段仍赋值 `boundaryAcceptanceConfigSha256 = Baseline1CampaignPlan.awaitConfigSha256(...)`，依次传入 `experimentRunIndex, experimentSeed, boundaryRecordingId, boundaryEventRoot, boundaryInstrumentationRevision`。首班通过该调用唤起 Python。 |

## 六车副本的停运逻辑核对与补齐

下列片段适用于本次固定掩码 `292=(1<<2)|(1<<5)|(1<<8)`。已安装等价逻辑的副本逐项核对即可。缺少逻辑时按对象和原行定位，粘贴对应片段。 [七段 Java 片段](../fleet_outage_001/anylogic_fleet_outage.java.txt)

| 位置 | 原写法或定位标记 | 本次改动及影响 |
| --- | --- | --- |
| **Main → Functions → isFleetTruckDisabled** | 标准模型尚无该函数 | 新建函数，返回 `boolean`，入参 `int truckId`，函数体用片段 1。它统一解释停运掩码并检查 ID。 |
| **Main → Functions → init1** | `Truck population` 下的 `for(int i = 0; i < numTrucks; i++)` | 只替换车辆创建循环为片段 2。九辆车保持同一 ID、车型、载重和初始化顺序。停运车位于原有 `parking`，`currentNode=""`、`currentSegment=null`、速度为零。在役车沿用原发车点，初始前车链跳过停运车。 |
| **Main → Agent actions → On startup → ACTIVATE 循环** | 对所有 Truck 调用 `t.deliver("ACTIVATE", t)` | 用片段 3，只激活在役车辆。保留前面的配置握手。 |
| **Main → Functions → selectNextLoadSite** | 首行 `obsCheckDecision(t);` | 在其前插入片段 4。停运车若触发派车，立即暴露配置错误。 |
| **Main → Events → evDynamicSpeed → Action** | 尾部遍历未完成初派的车辆并调用 `t.onChange()` | 只替换最后一个循环为片段 5，停运车不参与初派。中间的速度和跟驰更新沿用当前模型。 |
| **Truck → Functions → appendDispatchObservation** | `int phase=11;` 紧跟 `if(decision) phase=0;` | 用片段 6 替换这两行，其后的 `else if(!initialDispatchDone) phase=1;` 接在片段之后。停运车写 `phase.11` 且永远不能成为 ego。1519 维观测保留原布局。 |
| **Main → Functions → buildRewardConfigComponents** | `c.put("config.num_trucks", (double) trucks.size());` | 紧随该行插入片段 7。非零停运掩码时，START、DECISION、END 上报 `config.fleet.outage_version=1`、`config.fleet.disabled_mask=292`、`config.fleet.active_count=6`，同时 `config.num_trucks=9`。 |

若副本已经装有旧四种停运方案的通用逻辑，而且接受掩码 292，则可继续使用现有逻辑。两种接入方式都检查停运车不占道路、料斗或装卸队列，在役车的初始前车引用不指向停运车。

## 启动及验收

```bash
cd /home/jinx/Faith/experiments/lagrangian_ppo_v1
plan=change_plans/token_attention_001/business_graph_summary_009/fleet_outage_003
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B "$plan/launch.py" --check
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B "$plan/launch.py" --status
```

预检通过且共享 RPC 端口 `127.0.0.1:50051` 可用时，在 AnyLogic 六车副本执行 **Save → Build → Run RL_Training** 一次。Python 根据每班实际 RPC 验证场景、停运字段、观测和合法动作，实时状态见 `runs/bg128-full-s02-u32-fleet6-zero-shot10-20261124-001/run_status.json`。

验收时，停运三辆车的派车数为 0，动作回显只来自六个在役 ID。十班均需收到正常的 28,800 秒 END，核对四窗结算、矿废产量、等待和联合成本，并写出十份班次签收与 `suite_complete.json`。品位越界或欠产是待报告的业务结果。若出现 `STOP`，按状态和日志核对已领取身份及回执。

班次封存后执行 `/home/jinx/miniconda3/envs/anylogic-env/bin/python -B "$plan/verify_results.py"`，独立核对十份回执、四窗物理结算和实际在役车派车账。 [终验脚本](verify_results.py)
