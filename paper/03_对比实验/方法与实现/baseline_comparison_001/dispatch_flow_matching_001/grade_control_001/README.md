# 路径流量匹配品位控制修订

十班闭环使用已核验的修订版及十个新的标称环境 seed。逐班身份、启动步骤和验收口径见 [十班测试手册](manual_anylogic.md)。

冻结 `s01/u32` 的一轮 GAT 加逐节点 MLP 使用上述十槽逐一配对。启动、模型校验与报告口径见 [GAT1 同环境配对手册](gat1_s01_paired_001/manual_anylogic.md)。

五班物理测试全部通过工程验收，矿废产量均达标，完整任务 0/5。20 个破碎窗中 7 个越界，全部高于 28%。第一窗 3 班超限，第二窗 4 班超限。五班的首份 LP 预测四窗均贴住 28.0%。934 次真实派车中，443 次为强制超配，当时的分配规则只比较运输损失。 [五班终验](../../../../../../reports/token_attention_001/business_graph_summary_009/baseline_comparison_001/dispatch_flow_matching_001/smoke5_evaluation.json)

## 算法改动

路径流量 LP 保留车源、资源、仓容和 FIFO 映射。优化顺序为业务品位与产量缺口、26.6%–27.4% 内带缺口、车时。内带用金属吨线性松弛。后续阶段使用前一阶段已验证的最优值约束，达到计算时限便保留已验证的可行解。业务验收沿用 26%–28% 和既有矿废目标。 [品位 LP](grade_flow_plan.py)

每个请求先产生路径匹配或备用动作。原动作派矿时，用同一份已承诺的物料和资源状态，分别插入各合法矿石候选的整车载荷。预测按 FIFO 切到当前窗和下一窗。只有候选保住先前预测的破碎量、两窗合计进料且每窗业务越界不恶化，才比较两窗品位风险。改善判据采用约 0.2 个百分点对应的金属吨，最低为 2 金属吨。内带风险改善时还要求同时靠近 27% 目标。此规则也覆盖超配和计划不可用的矿车动作。矿石与废石类别沿用初始匹配。 [整车 FIFO 复核](grade_guard.py)

最终动作沿用现有掩码与回显台账。发生改派时按最终路径和载荷记账，单列改派来源和原因。复核超时使用先前完成的合法动作。FIFO 复核使用可见已承诺矿批和这一车的名义预测，守住的余量仍需在真实闭环中核验。 [调度策略](grade_policy.py)

## 离线验收

从实验根目录对相同的封存帧运行两版策略。输出位置使用 `reports/`。

```bash
dispatch_grade_dir=change_plans/token_attention_001/business_graph_summary_009/baseline_comparison_001/dispatch_flow_matching_001/grade_control_001
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B "$dispatch_grade_dir/review_grade_control.py"
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B -m pytest -p no:cacheprovider -q "$dispatch_grade_dir/test_grade_control.py"
```

24 帧对照中两版均返回合法动作。四窗初始预测由 28.0% 移到 27.4%，贴近 28% 的名义窗口由 48/60 降为 0/60，平均绝对品位偏差由 0.835 降至 0.355 个百分点。整车品位复核改派 6 帧，新版规划 P95 为 528 ms，零超时。五班旧结果保留为诊断证据。新策略的实际四窗品位及超配表现以独立身份下的 AnyLogic END 验收。 [离线对照报告](../../../../../../reports/token_attention_001/business_graph_summary_009/baseline_comparison_001/dispatch_flow_matching_001/grade_control_001/offline_review_verified.json)
