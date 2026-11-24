# BG128 六车车队组成配对验证

使用九车训练的冻结 `s02/u32`，测试四种开班即停运三辆车的标准工况。每种方案五班，共 20 个物理班。每个配对槽对四种方案使用相同的新环境种子和独立于环境随机流的 Actor 采样种子。不同槽使用各自的新双种子与物理身份。

| 方案 | 停运 ID | 保留小／中／重型车 | 在役额定载重合计（吨） | 掩码 |
| --- | --- | --- | ---: | ---: |
| 重型全停 | 6、7、8 | 3／3／0 | 716.1 | 448 |
| 每种车型停一辆 | 2、5、8 | 2／2／2 | 863.4 | 292 |
| 小型全停 | 0、1、2 | 0／3／3 | 1,020.0 | 7 |
| 中型全停 | 3、4、5 | 3／0／3 | 854.1 | 56 |

三辆停运车保留原观测槽，放到停车区。六辆在役车保持各自的原始发车位置。因此，小型车全停时，六辆在役车均从破碎站一侧出发。额定载重合计是每辆在役车额定载重之和，整班吞吐量由运输、装卸和配矿结果决定。

## 启动

在独立 AnyLogic 副本按清单接入。使用本活动 Java helper，将 `RL_Training → Number of runs` 设为 `20`，保持 `numTrucks=9`。端口 50051 释放后，保存、构建并点击一次 Run。逐班配置指纹、两类种子和输出目录已冻结。 [AnyLogic 手工清单](manual_anylogic.md) · [Java helper](prepared_001/Baseline1CampaignPlan.java) · [20 班身份清单](prepared_001/run_seed.csv)

```bash
cd /home/jinx/Faith/experiments/lagrangian_ppo_v1
fleet_plan_dir=change_plans/token_attention_001/business_graph_summary_009/fleet_outage_002
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B "$fleet_plan_dir/launch.py" --check
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B "$fleet_plan_dir/launch.py" --status
```

## 验收

逐次校验停运配置、停放观测、实际派车 ID 和动作掩码。所有班使用同样的 28,800 秒班长、四窗品位边界、16,000 吨矿石及 4,000 吨废石目标。报告按方案列出完整任务、品位、矿废吨位、等待、联合成本和六辆在役车的实际派车次数，以均衡停运为参照列出每槽差值。样本量为五个配对槽，采用描述性汇报。
