# B 图 PPO 微调与同种子复测

使用 `loop_B` 的七车班次，从动态出勤 u9 热启动。本活动内将原权重记为 u0。每批 32 班采样派单后做一次 PPO 更新，共四批。u0 至 u4 在同一组 16 个验证种子上分别运行。以验证结果选出的权重重跑原 B50 的 50 个环境种子。全部 258 班各有独立 run、recording 和配置字节 SHA256。[逐班身份、种子与输出路径](run_index_seed_manifest.csv) · [启动身份索引](launch_identity.json)

## 一次启动与进度查询

本轮已经启动。在本目录执行以下命令可查看各阶段的回执数、已完成的更新及入选权重。把 `--status` 改为 `--follow` 可持续查看至最终报告生成，按 Ctrl+C 可随时退出观察。

```sh
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B run_remaining.py --status
```

本活动完成后，再次运行查询会直接给出报告路径。同一目录的 `finetune_curves.svg` 是完整矢量曲线。[运行控制器](run_remaining.py)

对已经准备好全新身份、尚未开班的活动，在其方案目录执行一次下列命令即可。控制器依次启动 AnyLogic、完成四次更新、验证选模、B50 复测与最终审计。每个原生实验启动前，控制器会锁定活动，并记录该实验涉及的 run、recording 和配置 SHA。

```sh
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B run_remaining.py
```

重复执行时，已有控制器或本轮旧版启动记录会转为显示进度。由此入口启动的活动若在阶段间中断，再次执行会核对已有回执与 checkpoint，从下一个阶段衔接。若某组已领取 run 但回执尚不全，查看活动 `logs` 中的策略服务及 AnyLogic 日志核对原进程。后续独立重跑使用新的活动身份和配置清单。

## 在 AnyLogic 中复核实验入口

冻结模型文件名为 `DRL-PPO-baseline-topologydemo-b-finetune.alp`。以下步骤适用于手工重建副本或对照现有文件。保留原统计模型作为对照。训练中的文件由 `model_path` 和字节 SHA256 共同绑定，手工另存产生的副本供离线核对。[首班冻结配置](configs/T_001.json) · [逐项变更代码](build_model.py)

1. 在项目树中找到 `DRL-PPO-baseline-topologydemo-frozen-policy-stats50`，另存为上述新模型名。在副本的 **Experiments** 中以 `Topology_Frozen_Stats_B50` 为模板复制六个 Parameter Variation 实验，名称和标题设为下表中的实验名。副本中由这六个入口取代原来的 `Topology_Frozen_Stats_A50` 和 `Topology_Frozen_Stats_B50`。
2. 逐个进入 **Properties → Parameters**，启用 freeform，设置表中班次数和 `Main.topologyWitnessCase` 的表达式。原 B50 是 50 班、`201 + index`，这里的 `index` 从 0 开始。四批训练各有独立入口，以便在下一批开班前封存本批更新后的权重。

| 实验名 | Number of runs | `topologyWitnessCase` | 配置范围 | 实际使用权重 |
|---|---:|---|---|---|
| `Topology_B_Train1` | 32 | `1001 + index` | T_001–032 | u0 |
| `Topology_B_Train2` | 32 | `1033 + index` | T_033–064 | u1 |
| `Topology_B_Train3` | 32 | `1065 + index` | T_065–096 | u2 |
| `Topology_B_Train4` | 32 | `1097 + index` | T_097–128 | u3 |
| `Topology_B_Validate80` | 80 | `2001 + index` | V_001–080 | 每 16 班依次使用 u0–u4 |
| `Topology_B_Test50` | 50 | `3001 + index` | E_001–050 | 已选权重 |

六个实验均关闭并行评估与重复试验，仿真起止时间是 0 和 28,801 s。原生班末事件在 28,800 s 结算、写入回执并结束。保留同时刻事件顺序 LIFO。实验模板的 `fixedSeed=1` 用于初始化，开班后 `TopologyWitness.start()` 用各自配置中的 `environment_seed` 设置引擎随机数生成器。

## 在项目树核对两个 Java 类

| 位置 | 原统计版 → 微调版 | 核验方法与用途 |
|---|---|---|
| **Java Classes → TopologyWitness** 的 `DIRECTORY`、`open(Main,int)` | 配置目录从 `paired_50_001/configs` 指向本目录的 `configs`。两位槽号解析改为 `number / 1000` 分阶段、`number % 1000` 取槽号。根据阶段读取 `T_`、`V_` 或 `E_` 配置 | 对照实验表首末编号。检查 `experiment`、`layout_id=loop_B`、`phase`、`case`、`policy_update` 与 `native_case_code`，防止错跑阶段或地图 |
| 同类的构造函数与 `start()` | 活动用途由 `topology_frozen_policy_trial` 改为 `topology_b_finetune`。原来从静态配置读取固定权重 SHA，改为读取该班 `activation_path` 中的 `checkpoint_sha256` 和 `policy_update`，传给 `TopologyPolicyDispatch` | 开班校验 activation 中的 run、配置 SHA、更新编号。测试配置的 `policy_update=-1` 表示预留选模槽，activation 中填写已选的实际编号。派单回复的权重 SHA 必须匹配 |
| 同类的 `writeTrainingTerminal(double)`、`finish(double)` | 原来由 `request(cumulativeWait,true)` 直接进入班末封存。现在训练阶段在两者之间调用 `main.buildPhysicalObservation(null)`，原子写入该班 `python/terminal_observation.json`，再调用 `bindPolicyUpdate` | 写入时核对 `dispatch.physical.v3`、1519 个有限值、终止位为 1、四个动作位全为 0。文件带末刻时间、车队掩码、矿废量、金属量、等待量和权重身份，供最后一步 PPO 回放使用 |
| **Java Classes → AcceptanceEventRecorder** 的构造函数、`bindPolicyUpdate(int,String)` | 原有 `topology-stats50-v1` 的压缩 FIFO 记录条件增加 `topology-b-finetune-v1`。新增绑定函数把 `native_policy_update` 写进 manifest，训练班同时写入 `terminal_observation_sha256` | 班末逐班核对权重编号。训练班额外验证末态文件 SHA 与 manifest 相符，待策略和物料审计全部通过后才计入更新 |

完整 Java 方法可直接在封存副本的同名对象中对照。修改限于上述两类、六个实验入口和模型名。[封存模型](../../../../../../reports/token_attention_001/business_graph_summary_009/variable_topology_training_001/frozen_policy_trial_001/b_finetune_001/DRL-PPO-baseline-topologydemo-b-finetune.alp)

## 场景、随机性与选模口径

- 物理五项设置为 `deterministicCheck=false`、`dynamic=true`、`useLocalDynamicSpeed=false`、`useEndogenousCongestion=false`、`enableWasteTaskLimit=true`。九槽中 0、1、2、3、4、6、7 出勤，两个掩码均为 223，班中故障关闭。班目标按开班运力缩放，矿石约 11,799.55 t、废石约 2,949.89 t，废石停派线约 4,424.83 t。计算使用配置中的全精度数值。四个两小时品位窗使用 26%–28%。
- 训练按各班独立 Actor 种子采样，同一种子确定该班节点顺序。Actor、Critic 学习率分别是 `1e-4`、`3e-4`。验证及 B50 复测采用贪心派单，节点顺序种子为 0。验证依次按联合达标班数、合规窗口总数、品位累计偏离幅度、较早更新选模。B50 仅与其已封存的历史槽逐项配对。[协议与种子生成](campaign.py)
- 用 AnyLogic 的 **Build** 检查类与实验入口。复核冻结副本时对照新 ALP SHA256 `60bedf6c07e6f75058597cfa928faf31e289a47740394d34d62a3657c610211c`、父 ALP SHA256 `e3bc2214ba64493d27ad2a0ea5346570ecc1c5fa4df2b2d884528e66465ceccd`。若需在项目环境离线复建同字节副本，用 `/home/jinx/miniconda3/envs/anylogic-env/bin/python build_model.py <原统计版.alp> <新临时路径.alp>`。此命令只生成模型文件。
- 运行回执和事件位于逐班身份表中的 `python_output_directory` 和 `event_directory`。完整验收覆盖 128 个训练班、80 个验证班和 50 个复测班，逐班检查派单、原生结束、实际配置、矿石及金属守恒，并核对最终 50 对环境种子。[审计与结果汇总](results.py)
