# 两种拓扑下冻结 u9 的各 50 班验证

使用同一冻结动态出勤 u9。每个班为 8 小时，每次在合法装载点中取 Actor 分数最高的动作。A 图使用树形路网和九车满编。B 图使用环形路网和七车出勤。A 和 B 的各 50 个环境种子独立，每班使用唯一的物理 run、recording 和配置。前期确定性单班作为装卸设施、路网、车队和目标的参照，统计分母从本活动的 100 个新班取得。

| 目标（t） | A，9 车 | B，7 车 |
| --- | ---: | ---: |
| 矿石实破下限 | 16,000 | 11,799.55 |
| 废石实卸下限 | 4,000 | 2,949.89 |
| 废石停派线 | 6,000 | 4,424.83 |
| 废石允许上限 | 6,193 | 4,617.83 |

启动时核对完整精度的目标与实建车队。两组均采用原生随机料块品位和装卸服务耗时，沿用各自装载点的位置、品位中心与范围。路段按该拓扑的限速和空重车速度行驶。班内故障关闭。每班核对四个两小时窗的破碎矿石加权品位是否均处于 26%–28%，并逐次核对模型动作和实际派车。

整班成功要求矿石、废石和四窗品位同时合规。每组报告成功班数及 Wilson 95% 置信区间，并分别报告品位合规和吨位达标。均值和样本标准差按完成且核验通过的班次计算。两组车数、目标和拓扑同时变化，组间差异描述整体工况。

## 100 班实测结果

A 组 50/50 班联合达标，B 组 22/50 班联合达标。A、B 各自均有 50/50 班同时达到矿废吨位要求。B 组第 1 至第 4 个窗口超过 28% 的班数依次为 2、3、13、19，其中部分班次在多个窗口超过上限。两组全部窗口有实际破碎产出，最低品位均在 26% 下限以上。共核对 16,912 次实际派车。 [统计汇总](/home/jinx/Faith/experiments/lagrangian_ppo_v1/reports/token_attention_001/business_graph_summary_009/variable_topology_training_001/frozen_policy_trial_001/paired_50_001/summary.md) · [逐班核验](/home/jinx/Faith/experiments/lagrangian_ppo_v1/reports/token_attention_001/business_graph_summary_009/variable_topology_training_001/frozen_policy_trial_001/paired_50_001/cases.json)

## AnyLogic 中的复核位置

| 位置 | 原单班 | 本次实现与核对 |
| --- | --- | --- |
| 实验属性 → Parameters / Number of runs | `Topology_Frozen_Policy_B_Scaled`，固定参数 2，运行一班 | `Topology_Frozen_Stats_A50` 与 `Topology_Frozen_Stats_B50` 分别串行运行 50 次，参数自 101 与 201 起随迭代增加 |
| Java Classes → `TopologyWitness.open()` | 读取单一 `B.json` | 按实验与迭代读取 `configs/A_001.json` 至 `B_050.json` 及各自的 SHA256 |
| Java Classes → `TopologyWitness.start()` | `main.deterministicCheck = true` | 按班种子初始化，再设置 `main.deterministicCheck = false` 并核对开班运力目标。检查实际建立的各装载点料块品位 |
| Python → `ValidationTrial.decide()` | 检查车队、目标和动作掩码 | 另核对 Actor 输入中的随机工况标志。推理输出保留实际决策所需的摘要和动作 |

生成器提供取数、参数绑定和原生记录的具体代码替换。运行完成后可用 `/home/jinx/miniconda3/envs/anylogic-env/bin/python -B collect_stats.py` 只读重放。 [ALP 生成器](build_model.py) · [全部运行身份及配置 SHA](run_index_seed_manifest.csv) · [逐班核验与统计](collect_stats.py)
