# BG128 相邻快照编码器追踪

本轮在已完成的 [历史构图审计](../historical_snapshot_001/README.md) 之后，复用原始相邻观测，运行正式 `business_graph_summary_v1` 图编码器。用户允许使用随机初始化检查数值通道，明确不将向量变化解释为已学习正确业务语义。

结果见 [审计报告](../../../../reports/token_attention_001/business_graph_summary_009/encoder_snapshot_001/audit_report.md)：Actor/Critic各两时刻的H0/H1/H2、逐边消息、两层路径干预、三因素向量分解和原始编码支路敏感性。参数固定 `pair_seed=2026111801`，无训练权重、优化器更新、最终策略/价值头或C128融合Forward，无RPC、AnyLogic、新班次、预算登记或ALP变更。

输入为历史 `run_9810998` 的原始 state_index61/62。两时刻各自独立编码；Actor/Critic参数独立、角色内部两时刻相同。载荷/服务等关系严格复用正式图构建器；候选动作只作为读出索引，不生成事实承诺。

脚本均显式优先导入本实验目录，使用Conda `anylogic-env`。从仓库根目录依次执行：

```bash
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B experiments/lagrangian_ppo_v1/change_plans/token_attention_001/business_graph_summary_009/encoder_snapshot_001/trace_encoder.py
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B experiments/lagrangian_ppo_v1/change_plans/token_attention_001/business_graph_summary_009/encoder_snapshot_001/raw_sensitivity.py
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B experiments/lagrangian_ppo_v1/change_plans/token_attention_001/business_graph_summary_009/encoder_snapshot_001/review_trace.py
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B experiments/lagrangian_ppo_v1/change_plans/token_attention_001/business_graph_summary_009/encoder_snapshot_001/render_trace.py
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B experiments/lagrangian_ppo_v1/change_plans/token_attention_001/business_graph_summary_009/encoder_snapshot_001/finalize_trace.py
```

前3个脚本进行无梯度Forward及内部张量替换；第4个仅读取保存张量绘图；第5个只校验完整性并生成本轮manifest。脚本固定输入与输出目录，复现会覆盖本轮同名产物；保留封存证据时先复制本轮目录，不运行历史构图脚本或原工程验收脚本。内部张量混合不要求是物理可达的新状态，不能当作业务改派因果效应。

H0已含原始字段、站点车辆引用、地理、路径和21D数量；“自身H0”不是纯自身物理属性。逐边投影只在LayerNorm之前可加，范数不可相加；三因素使用沿真实变化的带符号向量投影份额。C128在图编码器之后融合，本轮不评价摘要与图读出的最终效果。
