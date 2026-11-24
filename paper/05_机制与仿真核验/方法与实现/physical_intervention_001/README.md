# BG128 业务关系原始观测干预

从同一历史T1原始观测独立构造A/B/C，完整重算事实图、数量、Token/H0/H1/H2、C128证书、Actor候选表示和四Logits。正式Actor最终接口已接通，直接运行现有读出，没有修改架构。结果见 [审计报告](../../../../reports/token_attention_001/business_graph_summary_009/physical_intervention_001/audit_report.md)。

|场景|外生干预|约束|
|---|---|---|
|A|Truck1完成Load0当前91.7吨已知品位装载，成为已载、目标Dump0|同步物料/预留/服务；仍9车；采用源码兼容的完成后、下一行驶状态前内部边界|
|B|Truck8将193吨未开始服务的预留从Load0转至Load1|同步账本与未来路线尾；当前路段/位置/历史保持|
|C|Dump0处理速率2500→1250吨/小时|瞬时正值降容，当前库存/FIFO/累计历史保持|

精确字段位置、修改前后、源码依据及容差见 [原始修改合同](observation_intervention_contract.md)。共同基点run9810998、state_index62、Ego Truck0、时间9717.867211564166秒。三个分支不依次叠加，不使用未来字段。

本轮只证明可观察字段约束与数值通道，未运行AnyLogic，不证明完整隐藏事件/资源状态的可达决策。A是内部暂态，B是未暴露为当前策略动作的外生改派，C是离线参数变化。模型固定随机初始化seed2026111801，无训练、优化器、RPC、新班次或预算登记；不把分数变化解释成学会业务。

从仓库根目录依次执行，Python固定为Conda `anylogic-env`：

```bash
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B experiments/lagrangian_ppo_v1/change_plans/token_attention_001/business_graph_summary_009/physical_intervention_001/build_scenarios.py
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B experiments/lagrangian_ppo_v1/change_plans/token_attention_001/business_graph_summary_009/physical_intervention_001/analyze_interventions.py
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B experiments/lagrangian_ppo_v1/change_plans/token_attention_001/business_graph_summary_009/physical_intervention_001/review_interventions.py
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B experiments/lagrangian_ppo_v1/change_plans/token_attention_001/business_graph_summary_009/physical_intervention_001/render_interventions.py
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B experiments/lagrangian_ppo_v1/change_plans/token_attention_001/business_graph_summary_009/physical_intervention_001/finalize_interventions.py
```

脚本使用固定输入/输出，复现会覆盖本轮同名文件，保留封存证据时先复制本轮目录。独立检查不调用构造函数，以另写的字段白名单和数量oracle验证输出；不运行原工程合成训练测试。

[接口与C128审查](readout_contract_review.md) 和 `readout_contract_probe.py` 是独立接口预检，使用此前T0；主A/B/C统一用T1。主C改变8个C128特征，T0预检为7个，不能混用。主脚本显式使用Actor的enabled=True特征配置，每个raw生成新证书；禁止复用原基点prefix或手工编辑C128。

本轮不增加隐藏张量替换。因H0已含引用/数量，最终评分还有直接Ego/Load/Dump、池化与C128支路，总Logit变化不是某条两跳路径的独占因果贡献。完整图、数量、逐层/候选/评分数据与图件均在报告目录，旧审计不修改。
