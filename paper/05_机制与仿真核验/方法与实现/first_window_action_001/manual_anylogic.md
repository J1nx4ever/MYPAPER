# 4 班首窗反事实启动卡

2026-11-19，Asia/Shanghai。**4 班均已完成。** 当次启动和配置记录保留供追溯。 [结果](../../../../reports/token_attention_001/business_graph_summary_009/first_window_action_001/experiment_result.json) · [控制器恢复记录](recovery_001/README.md)

|位置|修改前|修改后|用途及核对|
|---|---|---|---|
|项目树 → Additional Class → `Baseline1CampaignPlan` → Java 源码|上一轮已经完成的 BG128 helper|全文替换为 [本轮 Java helper](prepared_002/Baseline1CampaignPlan.java)|读取本轮 master 和配置 SHA，自动启动诊断控制器。核对 `jobCount()=4`、`frozenPlanJobCount()=4`、`resumeOffset()=0`。|
|`RL_Training → Properties → Number of runs`|上轮实际独评 100，训练副本可能显示 512|填 `4`|按 index 0–3 采集，首末 run 为 9980001、9980004。|

沿用 `RL_Training` 既有串行设置。Freeform 中 `experimentRunIndex`、`experimentSeed`、`boundaryRecordingId`、`boundaryEventRoot`、`materialEventRoot` 继续使用现有 `Baseline1CampaignPlan.*((int) index)`。Main 启动段使用已有的 `awaitConfigSha256(...)`，RPC 逐决策核对现有 Reward、业务参数和观测契约。

**Save → Build → 在 `RL_Training` 点击一次 Run。** Java 自动启动 Python。控制台出现 `Starting first-window action diagnostic` 后，日志写入本轮 `control/controller.log`。若原动作班未通过复现，控制器写入 `STOP.properties`，对应替代班无法领取 ready。

|字段|值|
|---|---|
|recording_id|`bg128-first-window-action-20261119-001-s01_BG128_first_window`|
|配置文件字节 SHA256|`1f82728da7ee93f66e61fa2da072b1c93733a76c05e1a47a49e22f366f643158`|
|Python 输出|`/home/jinx/Faith/experiments/lagrangian_ppo_v1/runs/bg128-first-window-action-20261119-001/evaluation`|
|AnyLogic 物料事件目录|`/home/jinx/Faith/experiments/lagrangian_ppo_v1/runs/bg128-first-window-action-20261119-001_materials/s01_BG128_first_window`|
|绑定 Checkpoint|`u13`，SHA256 `a9acc888413ce0861240c04c396a239d9c26e48290eddc9f2c14dd9c97e6afd5`|

|run_index|分支|环境 seed|Actor 采样 seed|
|---:|---|---:|---:|
|9980001|9950012 原动作|49007268870950|254738232869928|
|9980002|9950012 Load0 替换|49007268870950|254738232869928|
|9980003|9950029 原动作|111338353016942|171331923119204|
|9980004|9950029 Load0 替换|111338353016942|171331923119204|

[配置](prepared_002/phase_config.json) · [身份 JSON](prepared_002/launch_identity.json) · [run/seed 清单](prepared_002/run_seed.csv)

完成后输出根目录出现 `suite_complete.json`，报告目录生成 `experiment_result.json)。任务保持固定推理，正式训练与优化步均为 0。启动后若出现 STOP 或 UNKNOWN，保留本轮日志、物料记录和账本供排查。
