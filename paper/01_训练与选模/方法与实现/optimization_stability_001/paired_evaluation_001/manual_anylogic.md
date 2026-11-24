# 低学习率新u13：100班同种子推理启动卡

**本次100班已完成，零更新及来源审计通过。此卡仅作历史追溯，不重复Run。** [结果比较](../../../../../reports/token_attention_001/business_graph_summary_009/optimization_stability_001/lr_comparison_001/README.md)。

2026-11-18，Asia/Shanghai。512班训练与16次更新已全部完成；当前推理包已冻结并通过预检、Checkpoint只读加载和100行Java编译/身份检查，尚未启动推理、监听或登记100班预算。沿用用户已明确要求的相同训练/推理种子及原共同u0起点，无需再次授权。

## 本次固定内容

使用本轮低Actor学习率训练的 **u13**，不改选u16或事后挑选新最佳。保持随机采样，100班零训练、零梯度/优化器更新。100组环境与Actor采样seed逐槽等于上次旧u13的100班；新run、recording和输出目录独立。高频transition、RPC和rollout详细记录保持关闭，保留核心物料证据、episode统计与完整性计数。

## 手工操作与验证

在当前已完成训练的AnyLogic模型中操作。代理没有编辑`.alp`；运行配置仍由真实RPC逐项校验，无需重新上传或本地审计ALP。

|精确位置|修改前|修改后|原因、影响及验证|
|---|---|---|---|
|Additional Class → `Baseline1CampaignPlan` → 全部Java源码|刚完成512班的[训练helper全文](../prepared_001/Baseline1CampaignPlan.java)|全文替换为[本次100班推理helper](prepared_001/Baseline1CampaignPlan.java)|切换新u13只读推理控制器及本轮master/配置SHA。`jobCount()`及`frozenPlanJobCount()`=100，`resumeOffset()`=0；`runIndex(0)`=9970001，`runIndex(99)`=9970100。|
|`RL_Training → Properties → Number of runs`|`512`|`100`|只采100个固定Checkpoint推理班次，不训练。|
|`RL_Training → Parameters → Freeform`|现有`Baseline1CampaignPlan.*((int) index)`调用|保持，不加offset|自动填写对应seed、新run、recording、事件目录；不手填覆盖旧值。|
|`Main → On startup`|既有`Baseline1CampaignPlan.awaitConfigSha256(...)`及RPC/物料初始化|保持|Main核对新配置SHA；首次调用自动启动唯一Python推理控制器。|
|`RL_Training`运行设置|现有串行，不勾选并行评估和使用重复|保持|按master逐班执行，与配对seed顺序一致。|
|生产、品位、道路、车辆与服务参数|当前训练场景参数|保持|本次比较仅Actor学习率训练产生的策略差异，不能混入新场景或改奖励。|

Save → Build → 核对以下身份 → `RL_Training → Run` **一次**。无需另开Python；本轮控制器当前未启动。旧512班训练helper不能再次Run。若出现STOP、UNKNOWN或未完成claim，保留证据定位，不重复Run补班。

## 启动前独立身份

|字段|冻结值|
|---|---|
|campaign|`business-graph-stability-paired-eval100-20261118-001`|
|recording_id|`business-graph-stability-paired-eval100-20261118-001-s01_BG128_eval0100`|
|配置字节SHA256|`191a2e994ff233eb1982c977b2a8a43819632c23ec97a514bf43cf3c8240d98b`|
|suite字节SHA256|`31d4c016f5a99a16547fc3eeb02499af62715373396c75b4c2a728a6e81d352f`|
|新u13 Checkpoint字节SHA256|`b0d4f50fa6eb292c00f7fe60295b6853d66a3a738f38fbcaf3614edc483d2373`|
|run_index|`9970001—9970100`|
|共同初始化seed|`75597143230906`|
|首班环境seed / Actor采样seed|`93625302859898` / `196394957391198`|
|Python输出|`/home/jinx/Faith/experiments/lagrangian_ppo_v1/runs/business-graph-stability-paired-eval100-20261118-001/evaluation`|
|AnyLogic物料事件目录|`/home/jinx/Faith/experiments/lagrangian_ppo_v1/runs/business-graph-stability-paired-eval100-20261118-001_materials/s01_BG128_eval0100`|

[全部100班run/seed清单](prepared_001/identities.csv) · [完整身份JSON](prepared_001/launch_identity.json) · [最终配置](prepared_001/configs/s01_BG128_eval0100.json) · [预注册固定u13的绑定证据](../../../../../reports/token_attention_001/business_graph_summary_009/optimization_stability_001/paired_evaluation_001/selection_review.json)。配置SHA为实际字节SHA，与suite和Checkpoint SHA不同。

`seed_plan.json`保留最初等待Checkpoint时的原件，其中null不再代表最终配置；当前以本卡链接的`prepared_001/launch_identity.json`及最终配置为准，不改写原预备记录。

## 完成判据与预算

本次100班均计新增物理消耗，不因复用seed抵扣。当前完整合并预算上限21595，本次登记100后为21695；旧UNKNOWN/PLANNED和历史记录保留，H2未来预留仍为0。完整预算入口为`business_graph_stability_evaluation_v1.budget.budget_snapshot`。

完成应为100 COMPLETE、UNKNOWN0、无CLAIMED、独评账本commits为空，输出`suite_complete.json`及`run_status.json: COMPLETE`，全部训练/梯度/优化器更新计数为0。Actor、Critic、Adam、训练账本及私有RNG接受只读守护。

低频进度位于输出根目录的`run_status.json`；控制器日志为本目录`prepared_001/launch/controller.log`。最终应同时报告四窗品位RMSE、四窗联合合规、矿/废达产、完整任务、累计等待、Critic MC MSE和EV，保留所有失败班次，再与旧100班逐对比较。
