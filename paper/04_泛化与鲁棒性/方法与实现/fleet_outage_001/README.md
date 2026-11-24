# BG128 九车训练后六车运行验证

固定混合训练 `s02/u32`。本次运行五个独立的 8 小时班，环境保持标称，车辆 2、5、8 从开班起停运，三个车型各停一辆。剩余车辆为 0、1、3、4、6、7。品位窗、矿废目标、四个装载动作和品位及等待结算口径沿用标称对比。

原九辆 Truck 的 ID、车型、载重与 1519 维观测槽均保留。停运车辆停在道路和初始发车位置之外，观测以已有停放相位 11 表示。每次 RPC 验证停运掩码、六辆可用车、停放车观测和在役车的前车引用。每班检查采样、有效动作和正常 28,800 秒物理结算。

这是同一组合三车全班停运的工程验证。报告包含每车派车数、四窗品位、矿废达产、等待及联合成本。样本量为五个独立班，结论限于冻结策略在该停运配置下的表现。

## 启动

在独立 AnyLogic 副本手工接入。待当前 500 班标准工况释放端口 50051，再点击 **Save → Build → Run RL_Training** 一次。使用 `launch.py --check` 核对启动前冻结配置与历史防重，使用 `launch.py --status` 查询进度。 [AnyLogic 修改清单](manual_anylogic.md) · [离线预检记录](../../../../reports/token_attention_001/business_graph_summary_009/fleet_outage_001/preflight_review.json)

```bash
cd /home/jinx/Faith/experiments/lagrangian_ppo_v1
fleet_plan_dir=change_plans/token_attention_001/business_graph_summary_009/fleet_outage_001
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B "$fleet_plan_dir/launch.py" --check
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B "$fleet_plan_dir/launch.py" --status
```

每班身份和参数快照在 [run/seed 清单](prepared_001/run_seed.csv)。其中包含 recording_id、配置字节 SHA256、Python 输出和 AnyLogic 事件目录。正式结果封存在报告目录的 `fleet_outage_evaluation.json`，运行器汇总记录，逐请求和全量观测留存开关保持关闭。
