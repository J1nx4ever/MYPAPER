# AnyLogic 单班闭环验证方案

2026-11-21 定义，2026-11-22 准备四班验证（Asia/Shanghai）。[方法规格](methods.md)定义首版控制、目标和预测时限。[四班启动身份](prepared_001/launch_identity.json)逐班给出 recording_id、配置字节 SHA256、run/seed、Python 输出及 AnyLogic 事件目录。

## 0. 策略在哪里运行

四组决策都由 **Python** 执行。卸载贪心、FIFO 贪心和逐事件滚动共用 Python 名义预测器，冻结 BG128 在同一入口推理。Python 控制器在每班开始前按冻结清单中的 `run_index` 绑定策略，每次回传当前车的一个合法装载站索引 \(0,\ldots,3\)。身份和种子用于控制器校验，传统策略的决策输入是当步快照和冻结参数。

**AnyLogic** 使用 `currentDispatchPolicy=3` 进入现有 `Main.selectNextLoadSiteAI → exchangeDispatch`。首个 `EPISODE_START` 同时请求首个动作，后续按派车事件发 `DECISION`，班末 `EPISODE_END` 只收终止回执。AnyLogic 按返回站点执行真实运输、装卸、排队、FIFO 破碎和窗口结算。Java 的 `Baseline1CampaignPlan` 负责活动参数、身份和启动握手。这个分工把策略差异限制在同一个 RPC 决策槽。 [派车方法分支](../../../../anylogic/3d/DRL-PPO-baseline1/DRL-PPO-baseline1.alp#L4214) · [请求与动作回执](../../../../anylogic/3d/DRL-PPO-baseline1/DRL-PPO-baseline1.alp#L10288)

## 1. 验证范围

使用同一个标称场景槽，每条策略各完整运行一次 8 小时班，共 **4 个物理班**。沿用冻结混合 u32 标称独评的**物理工况字段**，独立生成一个新环境 seed。本活动允许四组按该槽共用这个初始 seed。每个物理班有独立的 run_index、recording_id、配置字节 SHA256、Python 输出和 AnyLogic 事件根。固定 BG128 的 Actor 另用一个新的私有采样 seed。确定性策略的决策由当步观测和冻结参数给定。启动格式如需 sampling seed 字段，标注为不消费。动作分叉后，真实随机值的分配和物理轨迹可分叉。 [标称工况参考配置](../../../../runs/bg128-continuous-mixed-20261119-001/evaluation/nominal/new/phase_config.json)

| Java index | Python 策略 | 本班必须留下的行为证据 |
| --- | --- | --- |
| 0 | 一步卸载归窗贪心 | 可比较请求中确实按 \(U\) 评分，汇总合法候选、完整候选比较和回退计数。 |
| 1 | 一步 FIFO 贪心 | 确实按 \(F\) 评分，汇总相同的班级计数及被预测跨窗的候选计数。 |
| 2 | 逐事件多步滚动 | 至少一个多合法动作且可预测的请求完成后续动作扩展。汇总单合法快路、搜索节点和完整搜索次数、初始解与最终名义成本差、预算截断及回退。 |
| 3 | 冻结 BG128 | 校验固定检查点 SHA256、私有采样 seed、推理次数和优化器更新数 0。 |

固定模型为混合训练 `s01_BG128/u32`，检查点 SHA256 为 `b8b81c0821535cdfdff3c9352c11d69bc91d14fbe97694ab5f45c08fb97c3c9a`。受保护文件 `python/episode_budget_ppo.py` 的设计阶段 SHA256 为 `8e4cfa31c622f5e33756e304eea3f7526547db72ae93f7d9f00808845d61ef96`，实现前、准备和验收时再次校验。Python 使用 `/home/jinx/miniconda3/envs/anylogic-env/bin/python` 或 `conda run -n anylogic-env`。

单槽检验策略接口、预测流程和评价账本。实际品位违规、空窗或欠产照常入账。策略效果与排名由后续多槽独立配对实验估计。

## 2. 启动前离线验收

1. **跨窗算例。** 设 \(t=7140\) s、2500 t/h、FIFO 前方 60 t、品位 0.24，当前已破碎 1000 t、金属 258 t，候选 100 t、品位 0.38，在 7180 s 卸完。到窗末最多破碎 \(125/3\) t，全部来自已有 FIFO，候选本窗破碎贡献为零。卸载代理给当前窗计 \(Q=1160,\ M=310.4\)，品位约 26.759%。FIFO 给 \(Q=1041.6667,\ M=268\)，品位约 25.728%。同一名义轨迹的产量和等待在两种评分中一致。
2. **物理边界。** 检验 \(B+R+q>C\) 时延后卸载、单批跨窗的吨和金属守恒、6000 t 废石承诺停派、班末截断、一个合法动作及零质量有效 FIFO 槽。单合法动作走确定性快路。构造计划不可用、强制短时限、候选同分用例，确认回退仍由原掩码约束。
3. **相同快照上的受控比较。** 固定同一帧，确认两个一步组的候选名义轨迹、班末矿废量和等待相同。完整候选可评价时核对滚动的初始解与一步 FIFO 一致，返回的已确认名义目标值不高于初始解。替换测试身份与后续日志引用，传统策略得分和动作应一致。
4. **同机计时。** 用开发快照覆盖早、中、末班和多合法候选状态，检查四组响应时延、一步候选比较完成率、三步搜索覆盖率与回退。800 ms 计算预算和 1 秒响应目标须在此步验证。用强制慢分支确认截止前可以返回已准备的合法动作。
5. **冻结启动包。** 从[合并预算入口](smoke_budget.py)核验完整历史及 run/seed 禁重。冻结清单声明本活动同槽配对，配置固定 K=3、800 ms 规划预算、1000 ms 响应目标和回退规则。清单通过只读校验后再进入界面操作。

## 3. AnyLogic 手工位置及改动

模型用现有 [DRL-PPO-baseline1.alp](../../../../anylogic/3d/DRL-PPO-baseline1/DRL-PPO-baseline1.alp)。以下用仓库快照定位，实际启动依据当前运行副本界面及 RPC 回执。

| 精确位置 | 当前内容 | 本活动手工设置 | 理由和验证 |
| --- | --- | --- | --- |
| `Additional Class → Baseline1CampaignPlan → 类正文` | 仓库快照为混合复现 s02–s05 helper，运行副本可能为结构消融 helper | 替换为本活动冻结的 [Baseline1CampaignPlan.java](prepared_001/Baseline1CampaignPlan.java) 全文，保留编辑器已有的 Java package。 | `jobCount()=4`，索引只接受 0–3，分别绑定上表的物理身份和各自配置。核对首条 Java 启动回执及对应的 Python ready。 |
| `RL_Training → Properties → Number of runs` | 当前界面原值，仓库快照为 6896 | `4` | 同一场景四个独立班。核对最后一个 index 为 3。 |

### 构建前核对现有设置

`RL_Training → Properties → Parameters → Freeform` 的已有表达式按下表核对。只有实际副本的表达式曾被覆盖时，才按表恢复。表中 helper 调用的完整形式为 `Baseline1CampaignPlan.<方法>((int) index)`。

| Freeform 字段 | 表达式或值 | 新 helper 实际返回 |
| --- | --- | --- |
| `currentDispatchPolicy` | `dispatchPolicy((int) index)` | 四班均为 `3` |
| `usePhysicalObservation` | `true` | 字面量，使用 1519D 物理快照 |
| `experimentRunIndex`、`experimentSeed` | `runIndex((int) index)`、`environmentSeed((int) index)` | 四班独立 run，按槽共用一个新环境 seed |
| `boundaryRecordingId`、`boundaryEventRoot`、`materialEventRoot` | `recordingId((int) index)`、`eventRoot((int) index)`、`materialRoot((int) index)` | 当班清单中的标识与路径 |
| `enableBoundaryEventRecording`、`enableMaterialEventMemory`、`offlineNativeCollection` | `eventRecordingEnabled((int) index)`、`materialEnabled((int) index)`、`offlineEnabled((int) index)` | 均为 `false` |
| `freeSpeedFactorStd` | `freeSpeedFactorStd((int) index)` | 四班均为标称 `0.05` |

继续使用 Freeform、串行、关闭 Replications 和 28,800 秒班长。核对 `Main → Variables` 中 `dispatchInferenceHost="127.0.0.1"`、`dispatchInferencePort=50051`、`enableOnnxDispatch=true`。`enableOnnxDispatch` 是现有 gRPC 派车路径的启用开关。保持常规场景 `deterministicCheck=false`、`acceptanceLongLoadEnabled=false`、`acceptanceBoundaryMode=0`，逐请求诊断开关保持关闭。800 ms 规划预算和 1 秒响应目标由 Python 计时与回退实现。AnyLogic 的 `dispatchInferenceTimeoutMs` 沿用现有通信 deadline。

`Main → On startup` 沿用 `Baseline1CampaignPlan.awaitConfigSha256(...)`。该调用校验实际 run、seed、recording 和事件根，并读取 Python ready 所绑定的配置字节 SHA256。界面现有 `boundaryAcceptanceConfigSha256` 初值继续由这个握手覆盖。 [实验及班数](../../../../anylogic/3d/DRL-PPO-baseline1/DRL-PPO-baseline1.alp#L25554) · [策略/run/seed](../../../../anylogic/3d/DRL-PPO-baseline1/DRL-PPO-baseline1.alp#L25636) · [记录开关和根目录](../../../../anylogic/3d/DRL-PPO-baseline1/DRL-PPO-baseline1.alp#L25700) · [Main 启动握手](../../../../anylogic/3d/DRL-PPO-baseline1/DRL-PPO-baseline1.alp#L27)

### 一键启动与交接顺序

在项目目录运行以下只读预检。看到 `PREFLIGHT_PASS_NO_LISTENER` 后，待占用 50051 的既有活动正常结束，在 AnyLogic **Save → Build → Run** 一次。

```bash
cd /home/jinx/Faith/experiments/lagrangian_ppo_v1
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B change_plans/token_attention_001/business_graph_summary_009/baseline_comparison_001/launch.py --check
```

新 helper 在首班拉起 Python 控制器。Python 为当班创建独立 gRPC 监听器，正常监听后发布 ready。Java 校验清单和实际配置字节，写启动回执，Python 在首发动作前核对回执。每班正常 END 确认并排空 RPC 后，控制器释放当班监听器，再用下一班身份启动新的监听器并发布 ready。现有桥接器按监听实例检查环境 seed 重复，这样四班可以使用活动内授权的同槽 seed。 [桥接器防重](../../../../python/dispatch_bridge.py#L295) · [分班监听先例](../first_window_action_001/runtime.py#L206)

运行时在 `runs/bg128-baseline-smoke-20261122-001/launch/controller.log` 查看逐班 `JAVA_START_VERIFIED` 和 `COMPLETE` 回执。四班封存后，运行根目录生成 `suite_complete.json`，验证结果写入报告目录的 `single_shift_validation.json`。出现 STOP 时保留回执和账本。

准备 Python 侧比对配置时，只从历史标称文件继承物理工况字段。新配置明确记录 `config.material_event_memory_enabled=0.0`，并校验现行 `config.material_event_contract_version=1.0`、`config.wait_breakdown_enabled=1.0`、`config.wait_breakdown_version=1.0`。逐 RPC 核对这些字段和场景指纹。四组区分由活动清单的 `run_index` 映射完成，Java 的策略槽始终为 3。对于仅本活动允许的环境 seed 配对，优先沿用首窗实验的局部清单实现，保留通用生成器与桥接器的禁重判定。 [RPC 配置字段](../../../../anylogic/3d/DRL-PPO-baseline1/DRL-PPO-baseline1.alp#L9325) · [同槽清单先例](../first_window_action_001/prepare.py#L74)

## 4. 闭环验收与输出

| 层级 | 验收条件 |
| --- | --- |
| START | 从真实 RPC 核对 run/seed、配置、工况指纹、1519D schema 及字段顺序。从活动清单、ready 和 Java 启动回执绑定 recording、事件根、配置字节 SHA256 及当班冻结策略。每班保留第一请求快照的摘要与合法掩码用于同槽初始化核验。 |
| DECISION | 请求序号连续、时钟不回退、上一动作回显、当前返回动作在掩码内。传统方法只读取白名单快照和冻结参数。同一请求的重试沿用已有票据和动作。 |
| 策略计数 | 各策略的派车类别计数之和等于已处理请求数。传统组分类为合法单选、模型选择、Python 端简化回退，BG128 计成功推理动作。滚动记录至少一个有效的多步扩展，核对同帧名义目标相对初始解不升。预算截断另计。 |
| END | 在 28800 s 正常结束并收到终止回执，窗口 0–3 各结算一次。按现行容差核对矿石入仓、破碎、仓余和车载量，四窗矿吨/金属吨与整班实际一致，废石及等待增量守恒。 |
| 低频证据 | 每班只持久化身份、实际配置、班级指标、四窗 \(Q/M\)、各站动作计数和工程汇总。内存在线统计端到端时延 P50/P95/P99、策略计算时延及回退率。保留异常现场。 |

评估器复用实际 RPC 的 `DispatchEnv`、`DispatchCallbackBridge` 和 `RewardCostAdapter`，并按方法规格重建整班联合成本。实现可参照轻量验收循环。现有固定 Actor 收集器绑定 Actor 准入、紧凑物料事件和逐步数据，低频单班验证需独立评估入口。业务结果包括四窗 \(Q_w,M_w,M_w/Q_w\) 与合规、矿废吨位、累积等待、完整任务和联合成本。通信、schema、终止或守恒异常保留日志，已开始但未封存的物理班按账本规则处理，替代运行使用独立身份。

验证报告存放在所属 `reports/` 目录，实际运行数据留 `runs/`，方案与生成的手工清单留当前 `change_plans/` 目录。四个单班及离线用例通过后，再另行冻结多槽正式对比。 [轻量循环参考](../../../../python/acceptance_service.py#L49) · [终止和守恒验收](../../../../python/constrained_adapter.py#L47) · [原固定 Actor 收集器](../../../../business_graph_summary_campaign_v1/collection.py#L296)
