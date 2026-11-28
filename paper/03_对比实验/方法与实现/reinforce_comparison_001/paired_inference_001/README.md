# Full S02 与整班 REINFORCE 的固定 u32 配对推理

此活动已停在 130 个完整独评班。原账本保留 700 个已登记槽，旧推理样本和结果由新训练启动前的清理记录登记。

两组都使用混合训练 S02 第 32 次更新后的策略。已有 Full PPO 的七份独评报告共覆盖 700 个完整班。本次只启动 REINFORCE 一侧，用同一组工况参数、环境种子和 Actor 采样种子逐槽配对；每个新班使用独立物理 run。两种策略按各自的合法动作概率随机采样。推理期间固定 Actor 参数，优化器更新数为零。

| 工况 | 配对槽数 |
| --- | ---: |
| 标称工况 `nominal` | 100 |
| 高低矿品位换位 `grade_swap` | 100 |
| 留出角色排列 `held_order_1` | 100 |
| 留出角色排列 `held_order_2` | 100 |
| 新品位数值 `new_numeric` | 100 |
| 联合变化 `joint_shift` | 100 |
| 废石位置变化 `waste_swap` | 100 |

活动为 `bg128-reinforce-s02-u32-paired700-20261127-001`。[启动身份](prepared_001/launch_identity.json)与[逐班清单](prepared_001/identities.csv)给出配置、run、双种子和输出目录。[训练封存报告](../../../../../runs/bg128-reinforce-full-shift-20261127-s02/suite_complete.json)标识本次实际使用的 Actor。[历史 PPO 封存报告](../../../../../runs/bg128-continuous-mixed-20261120-s02/suite_complete.json)关联原 700 班的评测证据。

每班保留四个品位窗口、产量和车队累计等待。所有班次封存后，控制器生成逐槽结果 `paired_shifts.csv` 和按工况及全体汇总的 `paired_summary.json`。差值按 REINFORCE 减 PPO 计算，完整达标数同时列出仅其中一种策略达标的配对班数。[配对统计程序](paired_metrics.py)
