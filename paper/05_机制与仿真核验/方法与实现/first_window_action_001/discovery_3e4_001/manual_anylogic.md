# 100 班固定 u13 发现批次启动卡

本批已完成 100/100。此卡保留作为启动身份记录。 [完成结果](../../../../../reports/token_attention_001/business_graph_summary_009/first_window_action_001/discovery_3e4_001/README.md)

2026-11-19，Asia/Shanghai。沿用刚完成首窗反事实的 AnyLogic 工程和 BG128 主模型。

|精确位置|修改前|修改后|用途与核对|
|---|---|---|---|
|项目树 → Additional Class → `Baseline1CampaignPlan` → Java 全文|4 班首窗反事实 helper|替换为 [本轮 helper](prepared_001/Baseline1CampaignPlan.java) 全文|连接本轮控制目录和 100 班清单。`jobCount()` = 100，`resumeOffset()` = 0，index 0、99 映射到 9981001、9981100。|
|`RL_Training → Properties → Number of runs`|4|100|一次 Run 顺序完成全部固定策略班次。|

`RL_Training` 保持串行，沿用 Freeform 中的 `Baseline1CampaignPlan.*((int) index)` 和 `Main → On startup` 中的 `awaitConfigSha256(...)`。保存并 Build，再点击 `RL_Training → Run` 一次。Java 会连接已预启动的精准身份控制器，若尚无启动回执则启动本轮控制器。真实 RPC 逐步验证原物理配置和 schema。

|字段|冻结值|
|---|---|
|recording_id|`bg128-u13-discovery100-20261119-001-s01_BG128_discover0100`|
|配置字节 SHA256|`4437ea2299f7bd5a43129a03248a2604ee6c74fe7d3ea1727de6be59736d186b`|
|Python 输出|`/home/jinx/Faith/experiments/lagrangian_ppo_v1/runs/bg128-u13-discovery100-20261119-001/evaluation`|
|AnyLogic 物料事件目录|`/home/jinx/Faith/experiments/lagrangian_ppo_v1/runs/bg128-u13-discovery100-20261119-001_materials/s01_BG128_discover0100`|
|run_index|9981001–9981100|
|首班环境 seed / Actor 采样 seed|`220512009835087` / `100656007734783`|
|Checkpoint SHA256|`a9acc888413ce0861240c04c396a239d9c26e48290eddc9f2c14dd9c97e6afd5`|
|合并账本|登记前 21699，登记后 21799|

[逐班 run 和双种子](prepared_001/run_seed.csv) · [冻结配置](prepared_001/phase_config.json) · [完整启动身份](prepared_001/launch_identity.json)

首次看到 `WAITING_FOR_MANUAL_RUN` 代表 Python 已在等待 Java 声明。真实启动后显示 `JAVA_START_VERIFIED run_index=9981001`。每班严格通过 RPC、紧凑物料事件及完整班准入后推进完成计数。完成时应有 100 个 COMPLETE 槽、`suite_complete.json` 与承接研究的 `discovery_result.json`。运行技术异常保留原运行身份与证据等待排查。
