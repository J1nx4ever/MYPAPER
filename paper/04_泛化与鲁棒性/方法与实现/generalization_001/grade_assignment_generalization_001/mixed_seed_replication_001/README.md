# BG128 混合工况训练：补齐四个随机种子

状态：**单 ALP 一键串行套件已冻结，未登记，未运行。** 本轮检验原 BG128 混合训练的跨训练种子稳定性。模型、奖励、PPO、车队、物料规则和 RPC 校验口径不变。自由速度因子截断前标准差 **0.05**，截断仍为 **0.85–1.10**。见[一键串行手工接入](manual_anylogic.md#L1)与[一键配置核验](../../../../../../reports/token_attention_001/business_graph_summary_009/generalization_001/mixed_seed_replication_001/oneclick_configuration_verification.json#L1)。

## 班次预算

| 范围 | 训练 | 新模型固定随机推理 | 新增物理班 |
| --- | ---: | ---: | ---: |
| 每个新种子 | 1,024（32 班/次 × 32 次更新） | 700（7 工况 × 100 班） | 1,724 |
| 四个新种子 | 4,096 | 2,800 | **6,896** |

首轮的 1,400 班独评由混合 u32 的 700 班与固定旧标称 u32 的 700 班组成。旧标称 u32 检查点与逐班结果已封存；本轮用同一批旧结果做配对对照，不重跑它。每个混合模型的评测样本量不变。若要求每轮也物理重跑旧策略，则另加 2,800 班，不属于本方案。

新登记使用 [replica_budget.py](replica_budget.py) 的 `budget_snapshot`：保留首轮 2,424 班 COMPLETE、1 班历史 UNKNOWN，并纳入速度压力活动实际注册的槽位。生成时已登记 25,636 班，H2 活动预留为 0。四轮登记后合并上限为 **32,532** 班；若另启动 40 班速度压力活动，则合并上限为 **32,572** 班。

## 四个独立的训练复现

从各自 u0 重新初始化 Actor、Critic、优化器和训练随机状态；不续接 s01，不做早停或事后挑选检查点。逐班连续抽样沿用[首轮混合工况合同](../continuous_training_001/README.md#训练与独评)：同一品位及波动范围、同一留出规则，每批每个物理站各承担 8 班废石。仅更换初始化、训练场景抽样、训练环境和训练 Actor 采样的随机流。

| 种子标记 | 独立活动 ID | 初始化 seed | 训练 run_index | 评测 run_index |
| --- | --- | ---: | --- | --- |
| s02 | `bg128-continuous-mixed-20261120-s02` | 96988463657515 | 10003001–10004024 | 10004025–10004724 |
| s03 | `bg128-continuous-mixed-20261120-s03` | 160321093819405 | 10005001–10006024 | 10006025–10006724 |
| s04 | `bg128-continuous-mixed-20261120-s04` | 218752489478849 | 10007001–10008024 | 10008025–10008724 |
| s05 | `bg128-continuous-mixed-20261120-s05` | 105592708357017 | 10009001–10010024 | 10010025–10010724 |

种子生成沿用首轮的 `seed(namespace, kind, ordinal)`。初始化用新活动 ID、`initialization`、`1`；训练场景按新活动 ID、`configuration`、更新序号 1–32 生成；环境和 Actor 采样分别用 `<活动 ID>-train` 下的 `environment`、`sampling` 和训练槽序号 0–1023。登记前用最新合并账本复核所有新 run、recording、双种子及 Java 引擎与地质 RNG 状态。历史 UNKNOWN 仍参与禁重。

[预排只读核验](../../../../../../reports/token_attention_001/business_graph_summary_009/generalization_001/mixed_seed_replication_001/seed_scope_audit.json#L1)已核对 6,896 个 run、8,196 个新初始化/训练 seed 与 20,480 个 Java RNG 状态；启动包生成后还需验证配置和实际运行身份。

## 评测配对合同

- 使用首轮原顺序：`nominal`、`grade_swap`、`held_order_1`、`held_order_2`、`new_numeric`、`joint_shift`、`waste_swap`。每个工况的槽 0–99 按 `(scenario_id, pair_ordinal)` 从[首轮封存身份清单](../continuous_training_001/prepared_001/identities.csv#L1)的 `new` 臂逐槽复制环境 seed、Actor 采样 seed 和工况配置；用 `runtime_scene_sha256` 及冻结 RPC 配置验证。首轮跨工况共用 100 对双种子，本轮保留这一结构。
- 起点锚定[首轮完成清单](../../../../../../runs/bg128-continuous-mixed-20261119-001/suite_complete.json#L1)（字节 SHA256 `e32326901f5c070c5bd64160c8978ea0b491947da7eb159f380c8ddef945bc9d`），核对其中的配置、检查点和评测报告引用。仅允许上述配对槽复用两种评测 seed；训练不复用。新物理班的 `run_index`、阶段 `recording_id`、Python 输出和 AnyLogic 事件目录必须独立。
- 每个种子只评冻结 u32，沿用随机推理，评测期优化器更新为 0。旧标称策略使用封存的逐班结果，仅作为共同基准。报告按 s01–s05 分别列出完整任务率、四窗合规、矿废达产、等待和成本，再给五个训练种子的均值与离散度。标称、历史故障复验和五个共同留出工况分别汇报；不把同槽重复评测当成独立样本，不据此估计旧标称训练的种子波动。

## 启动

按[手工接入清单](manual_anylogic.md#L1)使用现有 ALP、Java 包 `fusion` 和端口 `50051`。只粘贴一次 `prepared_oneclick_003/` 根目录的组合 helper，设为 6,896 班并点 Run 一次。顺序为 **s02 → s03 → s04 → s05**；前一轮完整封存并释放端口后自动接续，遇 STOP 不跳过。旧套件 `prepared_001/` 和 `prepared_serial_002/` 仅保留作审计。

每班仍完全校验 RPC/schema、物料、完整结算及 Actor 行为。相同 Actor、原始数据、动作、旧概率和采样 RNG 证据的回放在同一 phase 中只做一次；后续复核文件和输入字节，不重复跑同一神经网络。训练 minibatch、优化器步数及 checkpoint 精确恢复校验不变，高频诊断关闭。
