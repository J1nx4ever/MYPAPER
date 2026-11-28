# 机制与仿真核验

- [卸入与实际破碎口径的同轨迹对照](封存摘要/paper_review_20261124/settlement_comparison.json)
- [首窗单动作闭环反事实方法](方法与实现/first_window_action_001/README.md)
- [首窗物料分解](封存摘要/first_window_action_001/followup_001/README.md)
- [冻结模型的图、摘要和策略诊断](可解释性分析代码/interpretability_eval/README.md)
- [已有可解释性图表原位置](../../summary/interpretability_eval/figure11221215)

## 跨窗兑现、输入干预与现场资料

- A01–A02 在五种子 2,500 个留出班中复算矿石的实际破碎窗口，平均每班有 764.28 t 矿石跨窗加工。300 个同槽内窗边界中有 16 对符合临界批次匹配条件。[跨窗回放](封存摘要/delayed_realization_001/material_replay.json) · [近截止批次配对](封存摘要/delayed_realization_001/matched_batches.json)
- A03 固定其余观测，交换 FIFO 前两批次的次序。133 个状态的显式分窗投影改变，五个 Actor 在每个状态均改变动作概率，另 13 个投影不变的对照状态无变化。[方法与分析代码](方法与实现/delayed_realization_001/README.md)
- T01–T02 记录路网、单趟和整班物理可行性验收。T08 的西采场演示按现场资料构建，有两份完整的装运卸守恒验收。[西采场资料与假设](方法与实现/site_data_integration_001/west_pit_demo_001/README.md)
