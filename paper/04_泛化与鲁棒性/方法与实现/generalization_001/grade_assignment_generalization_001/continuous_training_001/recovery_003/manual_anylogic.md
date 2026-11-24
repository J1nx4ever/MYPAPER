# BG128 混合品位逐班 RPC 恢复

第 2 班 `10000002` 已完整准入。原共享 Python bridge 保留了它的矿源配置，第 3 班 `10000003` 在 START、0 秒和 0 次决策时被拒。它已被 Java 认领，原账本 `UNKNOWN`、停止日志和不完整样本保留。恢复为每个物理班新建严格 RPC 会话，班内仍锁定配置，第一批继续使用已封存 u0 Actor。

**本次 Run 授权登记 1 个替补物理班。** 替补使用第 3 班原定的同一矿废角色和品位参数，分配新的 run、recording 和两类班次种子。待运行 **2,422 班**，包含替补班 `10002425` 和原计划 `10000004` 至 `10002424`。预计全套保留 2,424 个完整班及 1 个历史 UNKNOWN，活动登记上限为 2,425，合并上限为 25,636。两个已完成班仍计入 u1。

| AnyLogic 位置 | 操作 |
| --- | --- |
| Additional Class → `Baseline1CampaignPlan` | 全文替换为[逐班 RPC 恢复 helper](prepared_001/Baseline1CampaignPlan.java)。其 `CONTROL_DIRECTORY` 以 `/recovery_003/control` 结尾。index 0 和 1 分别映射 `10002425`、`10000004`。 |
| `RL_Training` → Properties → Number of runs | 改为 **2422**。串行，关闭重复，Freeform 保持 `Baseline1CampaignPlan.*((int) index)`。 |

四个 LoadSite 初始化、Main 启动握手和物料审计继续使用已经接好的混合设置。关闭原报错实验后 **Save → Build → Run** 一次。

[替补班 recording、配置 SHA256、Python 输出和 AnyLogic 事件目录](prepared_001/launch_identity.json) · [全部待运行 run 和双种子](prepared_001/remaining_identities.csv)

后续预算与 run/seed 禁重使用[合并账本入口](accounting/budget.py)，保留原 UNKNOWN 并纳入获批后新登记的替补槽。

```bash
cd /home/jinx/Faith/experiments/lagrangian_ppo_v1
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B change_plans/token_attention_001/business_graph_summary_009/generalization_001/grade_assignment_generalization_001/continuous_training_001/recovery_003/recovery_bridge.py --check
```

`--check` 只读复验前两班及未用身份，启动后用 `--status` 查看进展。新日志在 `runs/bg128-continuous-mixed-20261119-001/recovery_003/launch/controller.log`。先出现 `JAVA_START_VERIFIED run_index=10002425`，接着应出现下一物理班 `10000004`。u1 在完成原两班和新 30 班后提交。
