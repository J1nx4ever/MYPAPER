# BG128 s02 u32 六车零样本评估

在用户指定的 `/home/jinx/Models/DRL-PPO-baseline2` 副本，评估从九车训练冻结的完整 BG128 s02/u32。沿用标称矿源和道路，2、5、8 号车从班初整班停运，其他 6 辆在役。四个装点、1519 维观测、九车 ID 槽、矿废达产目标、四窗品位和等待记账沿用正式标准工况。10 班均使用独立环境和 Actor 采样种子。

活动 `bg128-full-s02-u32-fleet6-zero-shot10-20261124-001`，物理 run `10042001–10042010`。Python 每班输出为 `runs/bg128-full-s02-u32-fleet6-zero-shot10-20261124-001/shifts/run_<run_index>`，事件根为该活动目录下的 `anylogic_events`。真实结算在 AnyLogic 结束后由 Python 验收并封存，优化器更新为零。 [手工改动与启动命令](manual_anylogic.md) · [每班配置 SHA256 与独立身份](prepared_001/run_seed.csv)

保存的副本起初绑定 20 班 MILP 实验，六车停运钩子按手工清单补全。新 helper 的首班启动握手会调用 `launch.py`，该入口可独立预检和读取状态。班末再次用 `verify_results.py` 核对签收、动作计数和物理指标。 [Java helper](prepared_001/Baseline1CampaignPlan.java) · [启动脚本](launch.py) · [终验脚本](verify_results.py)

## 理论运力筛查

取十班封存 RPC 的在役车、路段、装卸资源和品位，在 480 分钟内先求解整班吨流松弛。模型保留车辆从上一卸点驶向下一装点的流守恒，允许分数趟，按整班计算品位，省略排队和逐事件时间。标称车速及平均装卸时间下，这个模型在完成 4,000 吨废石时的最大破碎矿量为 16,687.74 吨。自由速度统一乘 0.95 时降到 15,964.19 吨。两个值的适用范围都是这个静态模型。 [吨流脚本](theoretical_capacity.py) · [求解结果](../../../../reports/token_attention_001/business_graph_summary_009/fleet_outage_003/theoretical_capacity.json)

离散事件回放使用六辆车逐车固定的装点序列，逐次检查合法动作、铲装、卸点服务和 500 吨破碎仓的库存加预约容量。固定自由速度系数为 1.08、装卸时间系数为 0.92 时，得到 16,235.22 吨已破碎矿、4,126.50 吨已卸废和四窗品位 27.247%、26.966%、27.175%、27.002%。回放用九车首帧投影出六车初态，采用各矿源名义品位与固定车速，省略加减速、30 米首发间距和跟驰及路口损失。它给出简化情景的可行解。原十班 AnyLogic 实测仍是冻结策略的联合任务压力测试。 [回放脚本](six_truck_fifo_witness.py) · [回放结果](../../../../reports/token_attention_001/business_graph_summary_009/fleet_outage_003/six_truck_fifo_witness.json)

用项目 Conda 环境从任意目录运行这两个脚本可重现 JSON。两者均为离线计算，直接读取已封存的输入与回执。

命令 `campaign.py --prepare` 用于首次冻结本活动。已经冻结的身份以 `prepared_001` 为准。实际登记通过 `campaign.py::budget_snapshot` 合并最新 20 班 MILP、GAT 复现和全部历史预算。活动启动时才追加 10 班。
