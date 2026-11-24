# 首窗单动作反事实

4 班已完成。两条原动作班均复现历史整班，替代动作将首窗品位分别从 25.8217% 提高到 26.5915%、从 25.9118% 提高到 27.3377%。147 吨和 193 吨替代矿均在首窗全部处理，两条替代班四窗合规且矿废达产。 [结果](../../../../reports/token_attention_001/business_graph_summary_009/first_window_action_001/experiment_result.json) · [完成验收](../../../../reports/token_attention_001/business_graph_summary_009/first_window_action_001/completion_validation.json)

后续复核已明确原结果中的 `policy_suggestion` 是原 Actor 的随机样本，两处最高概率动作均为 Load0。复核中也重建了四分支的物料时间链和 Q/M，筛查了六条来源于既有 1e-4/u13 的首窗失败。新增解释字段集中在补充证据中，原封存结果继续保留。 [概率、物料和扩展筛查复核](../../../../reports/token_attention_001/business_graph_summary_009/first_window_action_001/followup_001/README.md)

P2 已按用户授权冻结 [3e-4/u13 的 100 班新种子发现批次](discovery_3e4_001/README.md)。后续 run/seed 与预算检查入口更新为 [discovery_budget.py](discovery_3e4_001/discovery_budget.py) 的 `budget_snapshot`。

本实验在两条原 3e-4/u13 失败轨迹中各检验一个早期动作。实验固定 BG128、原 Reward、原 PPO 数值配置及已有独立评价的 u13。每例先复现原动作的完整班次，再从相同班初种子和动作前缀运行替代班。替代班只覆盖预先选定的一次派车，后续由原 u13 根据新的实际状态随机采样。

|历史失败 / 同种子成功|决策序号、时间、车辆|原动作 → 替代动作|候选吨位|首次观察到锁定|
|---|---|---|---:|---:|
|9950012 / 9970012|38，4876.78 s，Truck3|Load2 → Load0|147 t|6842.64 s|
|9950029 / 9970029|26，3387.78 s，Truck8|Load1 → Load0|193 t|6934.20 s|

筛选逐一查看证书前的合法高品位矿替代。以当时状态构造名义运输、装卸与 FIFO 投影，筛出原动作预计低于 0.26、替代区间落在 [0.262, 0.278] 且候选矿料预计在窗内处理至少 30 吨的决策。每例取最晚合格决策，同一决策选最接近 0.27 的投影。投影冻结了部分负荷，名义到达受真实排队和道路反馈影响，实跑以卸载和 PROCESS 事件验收。两处原 Actor 都给 Load0 最高概率，分别为 63.27% 和 59.75%。[候选与已实现对照](../../../../reports/token_attention_001/business_graph_summary_009/first_window_action_001/candidate_selection.json)

原动作班逐决策核对 1519D 字节、时间、车辆、Mask、物理回报、累计等待、私有 Actor 抽样及随机状态，并检查终止状态和完整物料事件。通过后才开放对应的替代班。替代班验证同一前缀及实时 Mask，发生覆盖时消耗一次原策略抽样，并分别记录建议和执行动作。终止后再核对物料事件前缀、FIFO、四窗 Q/M 与 RPC 结算。该记录带独立诊断标记，训练批次入口会拒收。

原策略与替代分支从相同环境 seed 及采样 seed 出发。矿源品位块按环境 seed 和矿源生成，动作会改变块何时被开采。装卸服务会使用引擎随机流，分支后事件顺序可改变不同车辆分配到的随机值，因此结论限定在同种子和该仿真实现下。历史低学习率成功对照在决策 1 和 10 已分别与失败轨迹分叉，用于解释实际处理组成差异。

首要结果为替代动作自身截至 7200 s 的卸载和破碎吨位、首窗各矿源 Q/M、首窗品位及是否合格。同时报告其余三个窗口、矿废达产、累计等待和原成本。替代动作修复首窗即建立该早期状态存在可行路径的实例证据。仅换一次后仍失败时，结论限于这次替换。证书所约束的是原轨迹后来进入的物理状态。若分支前证据或原样复现不一致，以验收失败处理。

历史启动材料包括 [启动卡](manual_anylogic.md)、[完整身份](prepared_002/launch_identity.json) 和 [run/seed 清单](prepared_002/run_seed.csv)。完整预算从 21695 班起，4 个新槽完成后为 21699。后续合并入口为 [merged_budget.py](followup_001/merged_budget.py) 的 `budget_snapshot`，它读取已移至本轮运行目录的关闭账本及全部既有账本。 [路径与字节 SHA 迁移记录](../../../../reports/token_attention_001/business_graph_summary_009/first_window_action_001/followup_001/ledger_relocation.json)

原 RPC 桥在每个监听器内禁止复用 seed。最终包为每个班次依次开启只接受该 run/seed 的监听器，在上一班完成验证并关闭后才开放下一个 ready。初版准备记录独立保留。 [退役原因](prepared_001/RETIRED.md)

最终包通过 Java 编译及四班身份映射、4 次进程内 RPC 身份检查、437 次封存动作核对和 4 次前缀不匹配拒绝测试。两条历史物料账本均通过对账，PPO 核心字节 SHA 保持一致。实跑通过原身份恢复入口完成，合并账本为 21699 班。 [离线验收证据](../../../../reports/token_attention_001/business_graph_summary_009/first_window_action_001/preparation_validation_002.json) · [恢复记录](recovery_001/README.md)
