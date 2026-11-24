# BG128 单随机种子正式训练

2026-11-18，Asia/Shanghai。按用户“固定当前BG128架构，使用正式训练数据训练Actor/Critic，先一个随机种子”的指令，沿用现有正式单种子配置：**s01_BG128、16次更新×32班＝512个新训练班次**。不扩展GAT层数、关系组、C128、成本或PPO，不附加第二组、独立冒烟或100班独评。

当前可操作入口为 [AnyLogic手工清单](manual_anylogic.md)：替换 [新Baseline1CampaignPlan全文](prepared_001/Baseline1CampaignPlan.java)，将 `RL_Training → Number of runs` 设为512，Save→Build→Run一次。helper通过固定Python桥接自动启动本轮唯一控制器，无需另开Python。代理没有修改ALP；未Run之前不表示已开始采样或训练。

## 数据与训练范围

从新种子初始化，训练样本由BG128当前策略在正式AnyLogic环境中在线采集。历史TQ轨迹和合成物理分支不作为BG128的PPO行为样本。每批32个完整班次核对观测schema、实际RPC配置、物料日志、行为采样与log-prob、同一成本MC及冻结old-V GAE，然后更新Actor/单标量Critic。

冻结架构：19×64D实体、2层GAT、transport/commitment/service/material四组、25种有向关系、16条道路属性边、每节点21D明确数量、原C128、无最终Query、候选Load/Dump直接共享评分。Actor/Critic独立参数。封存的 `business_graph_summary_v1` 与受保护PPO核心均未修改，新增训练通信入口仅在 `business_graph_summary_campaign_v1`。

成本保持008：四窗品位中心.27、deadband.004、scale.006、cap2、空窗2；矿石16000/废石4000吨缺口各8；等待0.2×车队等待秒/28800。废石6000为停派阈值。PPO保持32班/批、4epochs、Actor/Critic LR=3e-4/1e-3、clip.2、entropy.01、梯度裁剪.5、gamma1、lambda.95。没有利用梯度比动态改权重。

首32班直接计入这512班，完成端到端校验；更新后从检查点恢复并对已采原始观测做逐位一致性检查，通过后提交并继续后续15批。错误触发STOP，保留UNKNOWN/未完成claim；不自动重试或补班。已有离线架构验收和本轮通信/预算/Java检查不能代替尚未发生的真实端到端验收。

## 独立身份（启动前告知）

|字段|冻结值|
|---|---|
|campaign|`business-graph-summary-formal-seed1-20261118-001`|
|初始化seed|`75597143230906`|
|run_index范围|`9900001—9900512`|
|首班环境seed / Actor采样seed|`255314794623997` / `274856137631895`|
|首批recording_id|`business-graph-summary-formal-seed1-20261118-001-s01_BG128_train_u0001`|
|suite字节SHA256|`90ada18aa85cdba4b51b3c55e9a3326729662a1dcc1b53a4b331a029cd7614d3`|
|首批配置字节SHA256|`253639247b3c8c72a3b7b149834b27aa89be1ca54ccafa225a7eabf8abab024f`|
|Python输出根目录|`/home/jinx/Faith/experiments/lagrangian_ppo_v1/runs/business-graph-summary-formal-seed1-20261118-001`|
|首批Python输出|`/home/jinx/Faith/experiments/lagrangian_ppo_v1/runs/business-graph-summary-formal-seed1-20261118-001/arms/s01_BG128/train_u0001`|
|首批AnyLogic物料事件目录|`/home/jinx/Faith/experiments/lagrangian_ppo_v1/runs/business-graph-summary-formal-seed1-20261118-001_materials/s01_BG128_train_u0001`|

[16批recording/configSHA/输出目录](prepared_001/launch_identity.json) · [全部512班run/seed清单](prepared_001/identities.csv) · [套件](prepared_001/suite.json)。本轮身份与全部历史训练、独评及采样比较记录禁重。

现有合并账本上限20471：COMPLETE20461、UNKNOWN1、PLANNED9；本轮登记512后合并上限20983。上限不等于已完成班数。旧H2未来预留为0，旧UNKNOWN和PLANNED保持原样，不回收或转移。本次授权依据是上述用户单种子训练指令；512是沿用正式16×32的执行规格，未声称用户原话直接给出了512。独评尚未纳入本次范围。

## 验证与运行状态

[运行边界6项检查](../../../../reports/token_attention_001/business_graph_summary_009/formal_training_001/runtime_boundary_verification.json) 与 [预算29项临时数据库检查](../../../../reports/token_attention_001/business_graph_summary_009/formal_training_001/budget_integrity_001/budget_integrity.json) 已通过。冻结入口核对396项源文件签名；所有原始BG模型源文件、旧账本和PPO核心保持原样。Java全512行只读身份检查及准确当前状态见 [准备验证](../../../../reports/token_attention_001/business_graph_summary_009/formal_training_001/preparation_verification.json)。

只读状态：

```bash
cd /home/jinx/Faith/experiments/lagrangian_ppo_v1
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B -m business_graph_summary_campaign_v1.launch status --suite change_plans/token_attention_001/business_graph_summary_009/formal_training_001/prepared_001/suite.json
```

进度为输出根目录的 `training_progress.json`；控制器日志为本目录 `prepared_001/launch/controller.log`。训练完成条件：512 COMPLETE、UNKNOWN0、16次COMMITTED、最终u0016检查点、`BUSINESS_GRAPH_SUMMARY_TRAINING_COMPLETE_SINGLE_SEED`。完成与性能结论分开：本轮训练统计不作为固定检查点独评，尚不作多种子稳定性结论。
