# BG128 原512班任务恢复：余下511班

2026-11-18，Asia/Shanghai。本页仅用于原任务的审计恢复；不是新实验，不新增班次额度，不改变两层GAT、四组关系、C128、联合成本或PPO设置。

原首班 `9900001` 已正常完成仿真，184个transition、四窗口完整，随后Python派生存储的旧TQ身份检查报错，训练更新为0。先以原始数据、物料journal、原Actor及配置证据完成离线接纳，并将原UNKNOWN通过保留旧数据库/旧行的审计迁移认定为COMPLETE，才允许恢复。原STOP、首班claim、原配置与原启动记录全部保留。证据迁移和恢复预检尚未完成时，不使用本页启动。

当前已完成上述[证据迁移](/home/jinx/Faith/experiments/lagrangian_ppo_v1/reports/token_attention_001/business_graph_summary_009/formal_training_001/recovery_001/reconciliation_applied.json)，账本为COMPLETE=1、PLANNED=511、UNKNOWN=0、COMMITTED=0。恢复包已冻结，通过[完整门禁核验](/home/jinx/Faith/experiments/lagrangian_ppo_v1/reports/token_attention_001/business_graph_summary_009/formal_training_001/recovery_001/preparation_verification.json)与[实际Java helper编译/全部511行映射检查](/home/jinx/Faith/experiments/lagrangian_ppo_v1/reports/token_attention_001/business_graph_summary_009/formal_training_001/recovery_001/frozen_helper_verification.json)，恢复控制器尚未启动。

恢复仍使用原初始化checkpoint与同一Actor，首批为已接纳的run9900001和其后31班，之后继续原16次更新。此次只运行未执行的run9900002—9900512；不重跑run9900001。

## AnyLogic只改两处

操作当前实际运行的模型副本，先结束已报错的旧`RL_Training`运行，再备份模型。本轮不直接编辑`.alp`。

|精确位置|修改前|修改后|理由、影响与验证|
|---|---|---|---|
|Additional Class → `Baseline1CampaignPlan` → 全部Java源码|原512班正式helper全文|全文替换为 [Baseline1CampaignPlan.java](Baseline1CampaignPlan.java)|内部把AnyLogic索引0映射到原master ordinal1；`jobCount()`为511、`frozenPlanJobCount()`为512、`resumeOffset()`为1，`runIndex(0)`为9900002、`runIndex(510)`为9900512。原512行master及SHA继续核对，首班不可再次请求。|
|`RL_Training → Properties → Number of runs`|`512`|`511`|只执行原授权中余下511个PLANNED身份；累计仍512班。|
|`RL_Training → Parameters → Freeform`中的所有helper调用|`Baseline1CampaignPlan.*((int) index)`|保留，不添加`+1`，不手填旧身份|偏移已在helper内部统一完成；再次加1会跳班并越界。|
|`Main → On startup`中的`Baseline1CampaignPlan.awaitConfigSha256(...)`|现有调用与材料/RPC初始化|保留|首次Main调用独立恢复桥，绑定原ordinal1、原配置字节SHA及新恢复控制目录。|
|环境目标与运行模式|矿石16000、废石目标4000、停派6000，上界6193；串行、允许并行评估不勾选、使用重复不勾选|保留|原物理场景、原录制配置及采样顺序不变。|

Save → Build → 核对下方身份及511/512/offset1 → **Run一次**。Java自动启动唯一恢复控制器，不另开Python。运行期间不要再次Run；防重复的拒绝不代表需要另启Python。

原`prepared_001/Baseline1CampaignPlan.java`与原512班入口保留作审计，不能再次Run。新helper读取独立恢复控制目录；不要删除旧STOP、旧claim或改写任何旧配置。出现新STOP或错误时保留现场，不重复Run。

## 首个恢复班次身份

|字段|值|
|---|---|
|campaign|`business-graph-summary-formal-seed1-20261118-001`|
|恢复描述字节SHA256|`4783b9f195ac01f64b30da1e617759a6b00fa532a634a87630492ab1e864efa6`|
|恢复Java helper字节SHA256|`2d86c0ef9332d44b58f735a2014e57057a4bb80e448b1da657d801985831e0eb`|
|AnyLogic索引／原master ordinal|`0 / 1`|
|run_index|`9900002`|
|环境seed|`158151262824415`|
|Actor采样seed|`97434156278002`|
|初始化seed|`75597143230906`|
|recording_id|`business-graph-summary-formal-seed1-20261118-001-s01_BG128_train_u0001`|
|原首批配置字节SHA256|`253639247b3c8c72a3b7b149834b27aa89be1ca54ccafa225a7eabf8abab024f`|
|原512行master字节SHA256|`0b1dd27e450204f6661b0554e1ffa06d8ab87453e85612b4d0a59b6a1dc1044b`|
|Python首批输出|`/home/jinx/Faith/experiments/lagrangian_ppo_v1/runs/business-graph-summary-formal-seed1-20261118-001/arms/s01_BG128/train_u0001`|
|AnyLogic物料事件父目录|`/home/jinx/Faith/experiments/lagrangian_ppo_v1/runs/business-graph-summary-formal-seed1-20261118-001_materials/s01_BG128_train_u0001`|
|新恢复控制目录|`/home/jinx/Faith/experiments/lagrangian_ppo_v1/runs/business-graph-summary-formal-seed1-20261118-001/recovery_001/control`|
|新增班次／剩余班次|`0 / 511`|

完整描述和全部余下身份已冻结为 [recovery.json](recovery.json)、[launch_identity.json](launch_identity.json) 和 [remaining_identities.csv](remaining_identities.csv)。recording和配置仍是原首批；每个run、环境seed、采样seed、物料子目录各自保持原计划，已占用的首班不复用。

恢复成功后仍应累计512班COMPLETE、UNKNOWN=0、16次COMMITTED。旧失败及UNKNOWN迁移过程保留在独立恢复审计中，不作为新增训练，也不删除历史记录。
