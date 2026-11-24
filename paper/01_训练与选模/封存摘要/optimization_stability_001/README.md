# BG128 相同起点/种子、低学习率：准备证据

最新：[学习率配对比较](lr_comparison_001/README.md)与[100班独评完成审计](paired_evaluation_001/completion_independent_audit.md)通过。建议保留3e-4原基线；本轮512训练＋100推理均完成，不重复Run。以下历史准备记录保留。

当前：512班训练完成、16次更新已提交，见[完成核对](training_completion_001/README.md)；新u13的100班同种子推理包已准备，见[最终准备核验](paired_evaluation_001/preparation_verification.json)。以下保留最初准备阶段记录。

55项离线检查通过。真实原始共同u0导入后，Actor/Critic张量、两个私有RNG逐位相同，两个Adam状态为空，训练计数0；仅Actor LR为1e-4。与原学习器在相同LR下比较，实际更新、Adam及RNG逐位一致，epoch诊断为只读观察。

- [测试输出](paired_pytest.txt)和[命令](paired_test_command.json)
- [实际共同u0核对](common_checkpoint_verification.json)
- [原源码、ALP及历史预算保全](preservation_verification.json)
- [离线通信/数据门禁](runtime_acceptance.json)
- [512行Java编译/访问器、源及配置SHA预检](preparation_verification.json)

新训练配置已冻结；100班推理仅冻结配对seed与身份，新u13的Checkpoint/最终配置SHA尚不可得。当前零新增物理班、未启动RPC/AnyLogic、未登记预算。合成测试中的优化器调用仅用于数值验收，不是历史数据训练或真实实验更新。

当前方案与手工操作见[计划入口](../../../../change_plans/token_attention_001/business_graph_summary_009/optimization_stability_001/README.md)。已有第一次fresh-seed工程检查输出pytest.txt继续保留为历史中间证据；当前以paired_pytest.txt及共同u0/配对协议为准。
