# 两个Actor学习率的配对比较

按用户“推理结束，对比两个学习率”执行。只读两个已完成u13的100班推理结果及训练汇总，不重新选择Checkpoint、不启动仿真、RPC或优化器。

`analyze.py`按原始共同u0、训练/推理对应seed和固定u13核对比较条件；使用正式物理评价函数，从封存HDF5的事件/观测重建100班物理结果，并与正式最终报告逐项相等核对。按完整班次对进行20,000次bootstrap，分析seed为2026111803；同一班四窗一起重采样，RMSE重新汇总平方误差后开方。成功率另报告配对四格与双侧精确McNemar。Critic使用正式pooled MC指标及目标方差。

[最终结果与解释](../../../../../reports/token_attention_001/business_graph_summary_009/optimization_stability_001/lr_comparison_001/README.md) · [完整数值](../../../../../reports/token_attention_001/business_graph_summary_009/optimization_stability_001/lr_comparison_001/comparison.json) · [100对明细](../../../../../reports/token_attention_001/business_graph_summary_009/optimization_stability_001/lr_comparison_001/pairs.csv)。

物理预读阶段结果physical_comparison.json保留为中间证据，最终结论以comparison.json和正式完成审计为准。原控制器在100班采齐后仍进行校验/评分，最终正常发布报告；沙箱PID不可见的误判及被控制锁拦截的恢复尝试保留在相邻completion_recovery_001，未产生第二控制器、模型评分、正式报告写入或额外物理班次。
