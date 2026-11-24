# BG128 训练后分析入口

本目录仅存放离线分析脚本，不启动AnyLogic/RPC、训练或新增预算。

- `analyze_evaluation.py`：读取已完成的100班固定checkpoint13独评及历史TQ128的100班独评，生成相同指标的描述性对比与完整班次非配对区间，不重新选择checkpoint。
- `analyze_training.py`：核对16批相同cost/MC/GAE及五个冻结前策略的直接梯度加和，零优化更新。输入只含原训练数据，不把100班独评反用于训练。
- 首窗口物料流水诊断脚本与来源见报告中的机器证据；不使用缺失的窗口边界Truck状态补猜在途。
- 训练后原始观测A/B/C方案与脚本见 [trained_intervention_001](../trained_intervention_001/README.md)。

结果：[综合报告](../../../../reports/token_attention_001/business_graph_summary_009/trained_analysis_001/README.md)、[训练与梯度机器结果](../../../../reports/token_attention_001/business_graph_summary_009/trained_analysis_001/training_reward_convergence_gradient.json)、[独评比较机器结果](../../../../reports/token_attention_001/business_graph_summary_009/trained_analysis_001/inference_comparison.json)。

后续学习率、epoch、reward或融合变化仅是建议，未实施，不属于本轮100班推理启动包。
