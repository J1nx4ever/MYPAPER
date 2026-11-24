# BG128 优化稳定性：共同u0与相同种子的受控重跑

最新结果：[两个学习率的100班同种子比较](../../../../reports/token_attention_001/business_graph_summary_009/optimization_stability_001/lr_comparison_001/README.md)已完成；3e-4/1e-4完整任务98%/92%、成本0.32751/0.53710、等待4.990/5.407车辆小时。建议保留3e-4旧u13作为主基线。以下准备阶段说明保留追溯。

后续工作已转入[物料兑现、图/C128与后期更新机制](../next_stage_mechanism_001/README.md)。本目录下“尚未启动”“当前操作入口”均属已完成实验的历史准备文字，不能作为再次启动授权。当前完整预算入口为`business_graph_stability_evaluation_v1.budget.budget_snapshot`，已包含本轮100班推理，合并占用/预留21695班。

2026-11-18，Asia/Shanghai。按用户最新指令执行最新优先级第一项，并采纳“相同种子训练和推理、从原始共同初始化Checkpoint重新开始”的修订。**只把Actor学习率从3e-4降至1e-4。** 512班训练已完成、16次更新已提交；100班同种子推理最终配置已绑定新u13并完成核验，尚未启动。

当前操作入口：[100班推理手工卡](paired_evaluation_001/manual_anylogic.md)；[训练完成审计](../../../../reports/token_attention_001/business_graph_summary_009/optimization_stability_001/training_completion_001/README.md)。已完成的[训练手工卡](manual_anylogic.md)保留追溯，不再次Run；完整身份见[16批启动身份](prepared_001/launch_identity.json)，612个新旧run/seed对应关系见[配对清单](paired_seed_mapping.csv)。原始u0的实际导入和状态一致性见[检查证据](../../../../reports/token_attention_001/business_graph_summary_009/optimization_stability_001/common_checkpoint_verification.json)。

## 固定比较条件

|项目|上次BG128|本次|
|---|---|---|
|起点|原共同u0 `initial.pt`|同一封存文件，Actor/Critic张量、shuffle/采样随机状态逐位导入；计数0，Adam为空|
|Actor学习率|3e-4|1e-4|
|Critic学习率、epoch、minibatch|1e-3、4、32|相同|
|gamma/lambda、clip、entropy、梯度裁剪|1/.95、.2、.01、.5|相同|
|表征与模型|BG128两层GAT、四关系组、C128、独立Actor/单Critic|相同，复用封存模型及目标函数|
|奖励|四窗deadband.004、scale.006、cap2、空窗2；矿/废短缺8/8；等待.2×W/28800|相同；MC和GAE同一联合成本|
|训练|16×32=512班|相同；每班环境/采样seed按旧表对应，新物理run|
|独评|u13、随机采样、100班、零更新|固定新u13，相同100组环境/采样seed，零更新|
|高频日志|关闭|关闭；新增每epoch KL/clip/梯度分布汇总|

该起点不是仅凭同seed重新随机初始化，也不是从训练后u13续训。新命名空间用于保护旧Checkpoint和source pins；导入u0时只允许配置命名空间和Actor LR不同，其他配置不符即拒绝。训练使用与上次实际运行一致的BG128修复后存储层；不重引入早期旧TQ存储兼容问题。

## 规则、预算和解释

[预注册协议](protocol.json)固定16次更新；原有数值、通信/schema、行为/物料完整性错误触发STOP。不增加KL early-stop、自适应LR、回滚或业务权重调节，避免混入第二项优化干预。每epoch报告的KL和clip来自更新后完整batch重放；梯度分布和裁剪命中来自实际minibatch裁剪前范数，二者分母不同。诊断已证明不改变数值更新、Adam和RNG。

用户明确授权本次对应seed复用，例外仅绑定本campaign、原512/100班来源SHA和逐槽清单；乱序、交换seed类型、第三次重跑或旧run/recording复用均拒绝。所有512训练班与100推理班仍是612个新增物理班次，不能因同seed抵扣预算；旧H2预留保持0，UNKNOWN/历史记录保留。

新训练预算扩展使用`business_graph_stability_campaign_v1.budget.budget_snapshot`，合并最新BG128独评及所有早期账本。推理扩展后完整入口为`business_graph_stability_evaluation_v1.budget.budget_snapshot`。旧独评JSON和旧SQLite均保持原字节。512训练班已登记并全部完成，当前合并预算上限21595；100班推理已准备、尚未登记，登记后上限21695。

Checkpoint选择提前固定为新u13，u16只归档及优化诊断，不看新结果挑最佳。推理的100班种子/身份已冻结，新u13和最终配置SHA已绑定，见[最终身份](paired_evaluation_001/prepared_001/launch_identity.json)。最初[推理预备计划](paired_evaluation_001/seed_plan.json)的null作为历史原件保留。两个阶段沿用已授权范围，具体身份和最终配置SHA须在各自启动前告知。

同起点、同种子加强单因素配对可比性；策略分叉后动作和仿真随机调用顺序可能不同，不能声称经历相同轨迹或证明多训练种子泛化。最终报告应逐对比较完整任务、四窗品位、生产、等待、成本及MC残差，保留失败班次，并区分优化稳定性与业务性能。

## 物料兑现主线

用户提出的核心任务已经写入[截止前物料兑现诊断合同](physical_realization_contract.md)：跟踪承诺→运输/服务→卸载→FIFO→实际破碎/交付的时间和Q/M，关注当前决策在截止前真正能兑现的合规产出。现有C128已含FIFO前缀和名义投影；后续第二项核验其利用情况，不能把本轮低学习率当作已修复物理表征或已消除首窗失败。

## 工程结果

[55项离线检查与证据](../../../../reports/token_attention_001/business_graph_summary_009/optimization_stability_001/README.md)覆盖数值等价、实际共同u0、配对例外、完整预算、BG128真实存储及独评零更新守护。启动仍须真实RPC与首32班端到端验收；离线通过不等于已训练或已证明收益。受保护PPO核心SHA仍为`8e4cfa31c622f5e33756e304eea3f7526547db72ae93f7d9f00808845d61ef96`。
