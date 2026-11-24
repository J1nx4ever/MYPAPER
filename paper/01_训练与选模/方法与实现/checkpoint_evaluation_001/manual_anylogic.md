# BG128 固定Checkpoint13：100班一键推理

2026-11-18，Asia/Shanghai。正式训练512班/16次更新已完成；本轮新增100班独立推理，零训练更新。选择记录在推理前已冻结。

## 手工修改

在当前实际运行的AnyLogic模型副本中结束已完成的训练运行。由用户手工操作；本次没有直接编辑ALP。

| 精确位置 | 修改前 | 修改后 | 理由、影响和验证 |
| --- | --- | --- | --- |
| Additional Class → `Baseline1CampaignPlan` → 全部Java源码 | 上轮恢复训练helper全文 | 全文替换为 [prepared_001/Baseline1CampaignPlan.java](prepared_001/Baseline1CampaignPlan.java) | 使用本轮独立master和控制目录，100个索引仅映射本轮新身份。`jobCount()`=100、`resumeOffset()`=0、`frozenPlanJobCount()`=100；index0→9950001，index99→9950100。 |
| `RL_Training → Properties → Number of runs` | 上轮恢复训练为511（原正式配置为512） | `100` | 仅运行用户要求的100个新班次；Actor/Critic全程固定在update13。 |
| `RL_Training → Parameters → Freeform` 中的helper调用 | `Baseline1CampaignPlan.*((int) index)` | 保留 | 不加偏移、不手填run/seed；helper内部本轮从0开始。 |
| `Main → On startup` 的 `Baseline1CampaignPlan.awaitConfigSha256(...)` 及材料/RPC初始化 | 现有调用 | 保留 | 首班自动触发固定评估桥，Python先核验配置、模型、预算与schema，再发布ready。 |
| `RL_Training` 串行、重复和并行设置 | 现有串行；重复和并行均不勾选 | 保留 | 单控制器、一次Run顺序执行100班。 |
| 业务环境参数 | 矿石16000、废石目标4000、停派6000、上界6193等原配置 | 保留 | 与训练物理环境及当前008冻结成本一致；以RPC实际传入配置执行一致性检查。 |

**Save → Build → Run一次。** Java自动启动Python，不另开Python。运行期间不再次Run；若出现报错，保留日志和现场后报告。

## 启动前身份

| 字段 | 值 |
| --- | --- |
| campaign | `business-graph-summary-best-checkpoint-eval100-20261118-001` |
| recording_id | `business-graph-summary-best-checkpoint-eval100-20261118-001-s01_BG128_eval0100` |
| 配置字节SHA256 | `9084cb4a98cc9099f5c9876f87253185be9a67ad40fa7aa2c18594c25cdfa5a2` |
| 冻结suite SHA256 | `c52f980a3930f2d87f5f42b63872b3a2dce28451358bb724c70ec492dd6356d5` |
| 固定Checkpoint | `runs/business-graph-summary-formal-seed1-20261118-001/arms/s01_BG128/updates/0013/checkpoint.pt` |
| Checkpoint SHA256 | `a9acc888413ce0861240c04c396a239d9c26e48290eddc9f2c14dd9c97e6afd5` |
| run_index范围 | `9950001..9950100` |
| 首班环境seed | `93625302859898` |
| 首班Actor采样seed | `196394957391198` |
| 父模型初始化seed | `75597143230906`（固定模型身份，环境和采样seed均为新值） |
| Python输出目录 | `/home/jinx/Faith/experiments/lagrangian_ppo_v1/runs/business-graph-summary-best-checkpoint-eval100-20261118-001/evaluation` |
| AnyLogic事件目录 | `/home/jinx/Faith/experiments/lagrangian_ppo_v1/runs/business-graph-summary-best-checkpoint-eval100-20261118-001_materials/s01_BG128_eval0100` |
| 新控制目录 | `/home/jinx/Faith/experiments/lagrangian_ppo_v1/runs/business-graph-summary-best-checkpoint-eval100-20261118-001/control` |

全部100班run/env/sampling seed和路径在 [identities.csv](prepared_001/identities.csv)；[launch_identity.json](prepared_001/launch_identity.json)和[suite.json](prepared_001/suite.json)绑定完整配置与模型。无需手填这些值，helper从冻结master读取。

## 运行验证与交付

运行状态：`runs/business-graph-summary-best-checkpoint-eval100-20261118-001/run_status.json`。

Python启动日志：`change_plans/token_attention_001/business_graph_summary_009/checkpoint_evaluation_001/prepared_001/launch/controller.log`。

成功后该轮独评账本应有100个COMPLETE、无该轮UNKNOWN/CLAIMED、commits为空；`suite_complete.json`绑定固定Checkpoint与零更新证明。结果输出到本轮根目录的 `evaluation_complete.json` 和 `reports/summary.json`、`reports/per_shift.csv`、`reports/per_window.csv`、`reports/README.md`。

以上路径在实际Run之前尚不代表已经开始采样或获得100班结果。
