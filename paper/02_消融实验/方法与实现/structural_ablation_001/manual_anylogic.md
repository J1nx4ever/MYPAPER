# AnyLogic 手工启动 · s01 的第 1、4 组消融

本套件一次 Run 串行执行 Flat-MLP → w/o C128。各组 1,024 个训练班和 700 个固定 u32 评测班，共 3,448 班。Python 由 AnyLogic 首班自动启动。先让同端口的现有活动结束并释放 50051，再接入本套件。请使用运行混合工况的同一版本 [DRL-PPO-baseline1.alp](../../../../anylogic/3d/DRL-PPO-baseline1/DRL-PPO-baseline1.alp)。本次未直接修改 ALP。

## 两处手工变更

| 精确位置 | 修改前 | 修改后与验证 |
| --- | --- | --- |
| 项目树 → Additional Class → `Baseline1CampaignPlan` | 混合训练复现 helper（s02–s05） | 全文替换为[本批合并 Java helper](prepared_001/Baseline1CampaignPlan.java)。`jobCount() == 3448`，`runIndex(0) == 10014001`，`runIndex(1724) == 10016001`。Helper SHA256 为 `2f2c5b60e5b73c0269fee0a91427424af5380a9b6719456f840a476851b32b65`。 |
| `RL_Training` → Properties → Number of runs | 混合复现一键版为 `6896` | 设为 `3448`，使用一个串行参数扫描、每槽一次。 |

确认 `Main → Variables → dispatchInferenceHost` 是 `"127.0.0.1"`，`dispatchInferencePort` 是 `50051`。`RL_Training → Properties → Parameters → Freeform → freeSpeedFactorStd` 使用 `Baseline1CampaignPlan.freeSpeedFactorStd((int) index)`，本批每槽为 `0.05`。保留 `Main` 启动握手、物料事件与场景读取绑定。场景侧表和配置文件由 helper 自动取用。

## 启动身份与逐班清单

| 组别 | AnyLogic index | run_index | 首班环境 seed / Actor 采样 seed | 清单 |
| --- | ---: | ---: | --- | --- |
| 第 1 组 Flat-MLP | 0–1723 | 10014001–10015724 | `71476098037375` / `135079610946182` | [阶段配置 SHA 与输出位置](prepared_001/flat-s01/launch_identity.json) · [逐班 run/seed](prepared_001/flat-s01/identities.csv) |
| 第 4 组 w/o C128 | 1724–3447 | 10016001–10017724 | `181389286896626` / `179449711374749` | [阶段配置 SHA 与输出位置](prepared_001/noc128-s01/launch_identity.json) · [逐班 run/seed](prepared_001/noc128-s01/identities.csv) |

| 组别 | 首班 recording_id | 首班配置文件字节 SHA256 |
| --- | --- | --- |
| Flat-MLP | `bg128-ablation-20261121-flat-s01-s01_BG128_train_u0001` | `a8b379cd67882477d8f445de0e85fe7f63e2a4d3c1ddb1ff8485371fea0d410a` |
| w/o C128 | `bg128-ablation-20261121-noc128-s01-s01_BG128_train_u0001` | `91aefa41238768bebea99da2d7a890291410b535b5bec5ac0c19a0ae64f37845` |

Python 输出位于项目的 `runs/bg128-ablation-20261121-flat-s01/` 和 `runs/bg128-ablation-20261121-noc128-s01/`。AnyLogic 物料事件位于对应的 `runs/bg128-ablation-20261121-flat-s01_materials/` 和 `runs/bg128-ablation-20261121-noc128-s01_materials/`。每一阶段和物理班的完整文件路径以[合并启动清单](prepared_001/launch_identity.json)为准。

## 启动与验收

```bash
cd /home/jinx/Faith/experiments/lagrangian_ppo_v1
PLAN=change_plans/token_attention_001/business_graph_summary_009/structural_ablation_001
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B "$PLAN/ablation_run.py" --check
```

预检正常后在 AnyLogic 执行 **Save → Build → Run**。首班日志在 `runs/bg128-ablation-20261121-flat-s01/launch/controller.log`，看到 `JAVA_START_VERIFIED run_index=10014001` 表示 Java 身份握手已通过。Flat-MLP 完整封存并释放端口后自动接续第 4 组。两组各自出现 `suite_complete.json` 且 `run_status.json` 为 `COMPLETE` 时，可读取[配对评测总表](../../../../reports/token_attention_001/business_graph_summary_009/structural_ablation_001/paired_evaluation_summary.json)。若出现 STOP，保留回执与账本供核对，已领取的物理身份不能重跑。
