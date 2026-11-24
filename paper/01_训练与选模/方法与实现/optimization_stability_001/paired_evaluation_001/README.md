# 同种子固定新u13独评100班

**已完成100班、零更新，正式汇总与独立完成审计通过。不要重复Run本helper。** [最终配对比较](../../../../../reports/token_attention_001/business_graph_summary_009/optimization_stability_001/lr_comparison_001/README.md) · [完成审计](../../../../../reports/token_attention_001/business_graph_summary_009/optimization_stability_001/paired_evaluation_001/completion_independent_audit.md)。以下为历史启动准备记录。

512班训练全部完成，16次更新已提交；固定使用训练前已指定的新u13进行100班随机采样推理，不根据本次训练曲线重新选Checkpoint。

当前：**最终配置及sealed u13已绑定，预检、模型只读加载和100行Java身份核对通过；尚未启动或登记本次100班。** 操作入口为[手工启动卡](manual_anylogic.md)，替换本次helper，将Number of runs从512改为100，Build后Run一次。

- [最终启动身份与配置SHA](prepared_001/launch_identity.json)
- [全部100班run/seed](prepared_001/identities.csv)
- [Java helper全文](prepared_001/Baseline1CampaignPlan.java)
- [准备验收](../../../../../reports/token_attention_001/business_graph_summary_009/optimization_stability_001/paired_evaluation_001/preparation_verification.json)
- [训练完成及初始状态核对](../../../../../reports/token_attention_001/business_graph_summary_009/optimization_stability_001/training_completion_001/audit.json)

环境/采样seed与旧u13独评逐槽一致；新run/recording/目录均独立。所有100班均计新增消耗，训练、梯度及优化器更新为0。seed_plan.json保留最初未绑定Checkpoint时的历史原件，最终可执行身份以上述prepared_001为准。
