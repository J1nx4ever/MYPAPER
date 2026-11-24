# 同Checkpoint、同种子、低Actor学习率：手工启动卡

**训练已完成：本卡保留历史追溯，不再Run512班。当前使用[100班推理启动卡](paired_evaluation_001/manual_anylogic.md)。**

2026-11-18，Asia/Shanghai。当前仅准备，未启动AnyLogic/RPC、未登记预算。用户手动应用以下内容，不直接编辑ALP。完整方案见[README](README.md)。

## 本次修改

从原始共同u0实际Checkpoint恢复Actor、Critic和随机状态，确认训练计数0、两个Adam状态为空。512班训练逐班复用原512班的环境/采样seed；之后新u13的100班推理逐班复用原100班seed。所有recording、run_index和目录为新身份。唯一训练超参数变化是Actor LR `3e-4 → 1e-4`。

## AnyLogic精确操作

对象是你当前运行的模型；不要求重新上传或离线审计ALP。常见模型位置为`anylogic/3d/DRL-PPO-baseline1/DRL-PPO-baseline1.alp`，以当前打开的运行模型为准。

|精确位置|修改前|修改后|原因、影响及验证|
|---|---|---|---|
|Additional Class → `Baseline1CampaignPlan` → Java全文|当前已结束实验的helper；若仍为本次100班独评，则为[旧全文](../checkpoint_evaluation_001/prepared_001/Baseline1CampaignPlan.java)|全文替换为[本轮helper](prepared_001/Baseline1CampaignPlan.java)|切换到新训练控制器、master和配置SHA。`jobCount()`/`frozenPlanJobCount()`=512，`resumeOffset()`=0；`runIndex(0)`=9960001、`runIndex(511)`=9960512。不能只改班数沿用旧helper。|
|`RL_Training → Properties → Number of runs`|上次独评为100；若仍显示其他旧班数，以当前值为准|`512`|本次Run只覆盖512训练班；本阶段没有附加冒烟。|
|`RL_Training → Parameters → Freeform`|现有`Baseline1CampaignPlan.*((int) index)`调用|保持现有调用，不加offset|helper自动分配本轮身份，环境及动作seed与旧表逐槽对应。|
|`Main → On startup`|已有`Baseline1CampaignPlan.awaitConfigSha256(...)`及后续RPC/物料初始化|保持现有代码|首次Main启动本轮固定Python桥接；RPC实际配置、schema、物料证据必须核对。|
|`RL_Training`串行/并行设置|既有串行；不勾选并行评估、使用重复|保持|保证对应种子、班次和更新顺序。|
|`Main`生产/品位/物理参数|16000t矿石、4000t废石目标；6000t停派、6193t上界及现有场景|保持|本轮不改变业务条件或奖励。Actor LR只在已冻结Python learner配置中修改。|

Save → Build → 核对下表 → `RL_Training → Run`一次。helper启动唯一Python控制器；不需要另开Python。当前代理没有预先启动控制器。出现STOP、UNKNOWN或未完成claim时保留证据定位，不重复Run补班。

## 启动前独立身份

|字段|本轮冻结值|
|---|---|
|campaign|`business-graph-stability-lr1e4-seed1-20261118-001`|
|suite字节SHA256|`fbbb1803f3d0099ef1686f5c13384de8021bde49b19e6b118b5ef61b12afa78a`|
|首批recording_id|`business-graph-stability-lr1e4-seed1-20261118-001-s01_BG128_train_u0001`|
|首批配置字节SHA256|`f612dfc7f815e42767fba3305cd5b3b095d57b8570609693809ee57f5fc64acd`|
|训练run_index|`9960001—9960512`|
|共同初始化seed|`75597143230906`|
|首班环境seed|`255314794623997`|
|首班Actor采样seed|`274856137631895`|
|Python输出根目录|`/home/jinx/Faith/experiments/lagrangian_ppo_v1/runs/business-graph-stability-lr1e4-seed1-20261118-001`|
|首批Python输出|`/home/jinx/Faith/experiments/lagrangian_ppo_v1/runs/business-graph-stability-lr1e4-seed1-20261118-001/arms/s01_BG128/train_u0001`|
|首批AnyLogic物料事件目录|`/home/jinx/Faith/experiments/lagrangian_ppo_v1/runs/business-graph-stability-lr1e4-seed1-20261118-001_materials/s01_BG128_train_u0001`|
|原始共同u0文件|`/home/jinx/Faith/experiments/lagrangian_ppo_v1/runs/business-graph-summary-formal-seed1-20261118-001/arms/s01_BG128/initial.pt`|
|原始共同u0字节SHA256|`75bd2d54a78dad799c9f51864def284b915e0246fa2189cd32f5b6bd7a6948d3`|


[16批完整身份和配置SHA](prepared_001/launch_identity.json) · [512班新run/seed清单](prepared_001/identities.csv) · [612班新旧逐项配对清单](paired_seed_mapping.csv)。首批配置SHA是实际文件字节SHA，与suite SHA不同。高频transition/RPC/rollout记录关闭；每次更新仅新增4份epoch汇总。

## 推理阶段

本次训练结束后固定新u13，保持随机采样规则，独立推理100班、零更新。推理身份已冻结为`business-graph-stability-paired-eval100-20261118-001-s01_BG128_eval0100`，run_index为9970001—9970100，Python输出为`/home/jinx/Faith/experiments/lagrangian_ppo_v1/runs/business-graph-stability-paired-eval100-20261118-001/evaluation`，AnyLogic物料事件目录为`/home/jinx/Faith/experiments/lagrangian_ppo_v1/runs/business-graph-stability-paired-eval100-20261118-001_materials/s01_BG128_eval0100`。全部seed见[预备清单](paired_evaluation_001/seed_plan.json)。

新u13尚未产出，当前`checkpoint_sha256`和`final_config_sha256`为null；该预备清单不能启动。训练完整结束后，执行下列准备命令，验证新u13的实际sealed字节并生成最终配置、SHA、100班helper及身份；再告知最终SHA，手工换该helper、Number of runs=100，Run一次。不能用旧u13或根据新结果另选checkpoint。

```bash
cd /home/jinx/Faith/experiments/lagrangian_ppo_v1
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B change_plans/token_attention_001/business_graph_summary_009/optimization_stability_001/paired_evaluation_001/prepare_evaluation.py
```

此命令只准备，不登记预算或启动监听。100班推理最终配置在实际u13出现前无法诚实计算SHA；当前不填造。后续推理启动继续走实际RPC和全部通信一致性检查。

## 完成与验证

训练应为512 COMPLETE、UNKNOWN0、16次COMMITTED及`BUSINESS_GRAPH_STABILITY_TRAINING_COMPLETE_SINGLE_SEED`；训练u16归档，推理候选预注册为u13。低频进度见输出根目录`training_progress.json`，每epoch诊断见`arms/s01_BG128/updates/XXXX/training_summary.json`的`update.epoch_diagnostics`。训练成本不能代替100班固定Checkpoint推理表现。

```bash
cd /home/jinx/Faith/experiments/lagrangian_ppo_v1
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B -m business_graph_stability_campaign_v1.launch status --suite change_plans/token_attention_001/business_graph_summary_009/optimization_stability_001/prepared_001/suite.json
```
