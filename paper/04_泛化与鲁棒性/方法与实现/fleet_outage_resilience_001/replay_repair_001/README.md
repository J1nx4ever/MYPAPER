# 首次 PPO 更新停止后的恢复

原活动 `bg128-fleet-resilience-20261124-001` 已完成 `10043001–10043032`，32 个班的物理回执记录了 5,124 次决策。控制器在尝试共同热身 u1 的概率复核时停止。已核对 SQLite 状态、Java 的独占领取记录、逐班回执和 u1 的更新意向文件。u0 已封存，u1 尚未执行优化；逐步轨迹当时仅保存在已经退出的控制器内存中。[原控制器日志](../../../../../runs/bg128-fleet-resilience-20261124-001/controller.log)

用户已确认补 32 班以保留 4 次共同热身更新。两分支各 9 次更新，验证和测试各 36 个四方配对槽。从原封存 u0 开始，恢复程序串行执行 992 班，计入此前 32 班后本活动共 1,024 个物理班。

## AnyLogic 手工操作

1. 在 AnyLogic 停止或关闭原来已经报错的 `RL_Training` 实例。打开此前已修正、编译通过的 `DRL-PPO-baseline2.alp`。
2. 项目树 → Java classes → **Baseline1CampaignPlan**。把旧 `prepared_001` 中的类声明整体替换为**本子目录**的 [Baseline1CampaignPlan.java](prepared_001/Baseline1CampaignPlan.java)。保留 IDE 管理的 `fusion` 包。替换前活动名为 `bg128-fleet-resilience-20261124-001`，替换后常量 `CAMPAIGN_ID` 为 `bg128-fleet-resilience-20261124-repair01`，Java 桥指向 `repair_launch.py`。新握手目录单独记录此次恢复；原停止标记和 32 份历史回执保留供核对。
3. 保存并 Build。打开实验 **RL_Training**，Number of runs 设为 **992**，串行，从 **index 0** 启动。主表把新 32 班放在最前面，首班控制台应显示 run `10044001`；到 index 32 时自动转到原计划未领取的 run `10043033`。Main、Truck、DumpSite 使用之前编译通过的挂点。Python 桥由 AnyLogic 自动拉起。
4. 可在本子目录查看恢复活动的状态。首次优化成功的标志为项目 `runs/bg128-fleet-resilience-20261124-repair01/models/warmup/u0001/update.json` 与对应封存的模型。旧活动的 `run_status.json` 继续记录原事故。

```bash
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B repair_launch.py --suite prepared_001/suite.json --status
```

## 运行身份与预算

| 本次索引 | 物理 run | 身份及归属 |
|---|---|---|
| 0–31 | 10044001–10044032 | 新 recording、配置和独立双种子。逐班复用原首批的出勤名单、物料角色和品位设置。 |
| 32–991 | 10043033–10043992 | 使用原活动尚未领取的 960 行。配置原字节、环境种子、采样种子、输出和事件路径逐行核验。 |

完整登记信息见 [run_seed.csv](prepared_001/run_seed.csv)，其中 `python_output`、`event_directory` 均为绝对路径。首班 recording 为 `bg128-fleet-resilience-20261124-repair01-10044001`。环境 seed 为 `139183616299100`，Actor 采样 seed 为 `166733822921685`，[首班配置](prepared_001/configs/run_10044001.json)字节 SHA256 为 `4155469a09d0b4ebe4f806e746d3d4e84d924fe93b30e6df46d2a1e2a9bde4b4`。

恢复模型、控制目录和前 32 个新班的输出位于项目 `runs/bg128-fleet-resilience-20261124-repair01/`。其余 960 班的物理回执写入原配置指定的班次路径。冻结时合并预算已登记 43,014 班；恢复包实际启动时追加 32 班，在该基准下合计 43,046。准备包本身登记 0 班。[恢复账本](recovery_budget.py) · [封存文件](prepared_001/suite.json)

## 错误与验证

旧 Actor 在线逐次处理一个派单请求，训练器入批校验则对 16 帧一起做浮点前向计算。虽然参数和输入相同，两条计算路径的舍入差异足以越过原有阈值。修正版在每个训练班 END 和批次入训时按在线单帧方式重算动作对数概率，保留原有绝对及相对容差各 `1e-6`、动作掩码和策略版本检查，PPO 的梯度计算仍按原批次执行。原事故逐步数据未落盘，具体首个误差帧无法追溯。[修复代码](replay_policy.py) · [七种真实工况的复现记录](../../../../../reports/token_attention_001/business_graph_summary_009/fleet_outage_resilience_001/replay_repair_likelihood.json)

测试覆盖过期权重、被篡改的概率或动作掩码、采样和复核使用不同工作线程、封存和恢复、真实 PPO 优化以及旧新两个账本之间的领取顺序。32 个历史完整班还单独验证了真实动作概率入批、冻结目标与一次打乱决策的梯度计算。以上验证均为离线，不计恢复活动物理班次。[整批历史入训验证](../../../../../reports/token_attention_001/business_graph_summary_009/fleet_outage_resilience_001/replay_repair_full_batch_admission.json)
