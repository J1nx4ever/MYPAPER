# 原版 BG128 连续品位与矿废角色混合训练

本活动检验原版 BG128 在接受多种矿源工况训练后，能否根据当班可见品位和矿废角色作出有效调度。车队、物理站点、道路连接、1519D RPC、原四类关系、原边属性、C128、奖励与 PPO 算法沿用原版。训练从本活动独立初始化开始，按 32 个完整班收集一批，提交 32 次更新。u32 是固定主评检查点。[原版训练合同](../../../formal_training_001/README.md#L7) · [固定模型换位失败审查](../../../../../../reports/token_attention_001/business_graph_summary_009/generalization_001/grade_swap_state_relation_audit.md)

## 训练与独评

每批四个物理装载站分别承担 8 班废石角色，打散班次顺序。其余三个站按抽样分配高、中、低矿。每班的名义品位从高矿 36%–38%、中矿 29%–30%、低矿 15%–17% 连续抽取，矿石波动半区间从 0.5–1.0 个百分点抽取。角色、均值和半区间在该班开局冻结，班内沿用 500 吨矿块的随机生成机制。废石位置依据当班物料标志，废石交付和 6,000 吨停派继续按实际矿废类别生效。站点 ID、铲斗和位置留在物理站点。工况生成时排除留出的角色排列，并核对名义材料支持覆盖品位目标。实际矿废达产与入窗兑现由闭环评价确认。

训练为 1,024 班。随后旧标称 u32 和新混合 u32 分别在七个工况上各做 100 班固定随机推理，1,400 班独评，共 2,424 个物理班。两模型和七个工况在本活动对应槽内配对新环境与采样种子，物理 run 和 recording 独立。场景清单明确标出标称、纯品位换位、未见品位配置，以及矿废角色互换等工况。独评阶段梯度更新为零。合并登记新增上限为 2,424 班，现有登记上界 23,211 班，启动登记后合计 25,635 班。[本活动合并预算入口](mixed_budget.py) · [冻结身份与配置](prepared_001/launch_identity.json)

旧训练的模型初始化种子为 `75597143230906`，新训练为 `98044477675931`。两组训练各有 1,024 个独立环境种子和 1,024 个独立 Actor 采样种子，对应种子集合无交集。这次比较的对象是两个实际训练所得的固定检查点，差异同时包含训练分布和训练随机性的影响。

| 结果组 | 工况 | 报告口径 |
| --- | --- | --- |
| 共同留出主评 | `held_order_1`、`held_order_2`、`new_numeric`、`joint_shift`、`waste_swap` | 两组各 500 班。五个冻结配置均未进入任一组训练，也未用于上一轮 600 班闭环泛化独评。`new_numeric` 检验已见角色排列内的新品位参数，其余四案留出了角色排列。 |
| 历史故障复验 | `grade_swap` | 两组各 100 班。训练未见该配置，上一轮已用来诊断高低品位互换失败。 |
| 标称基准 | `nominal` | 两组各 100 班。单列原工况表现。 |

主评逐工况报告完整任务、四窗合规与矿废达产。每个工况内用相同物理配置、相同环境种子、相同 Actor 初始采样种子逐槽比较两个模型。三组结果独立报告。

## 交付物与启动

前两班已完成。当前续跑使用[逐班 RPC 恢复清单](recovery_003/manual_anylogic.md)，替补原第三班的中断槽位。[第一次恢复](recovery_001/manual_anylogic.md)与[第二次恢复](recovery_002/manual_anylogic.md)为历史记录。

1. 用[手工接入清单](manual_anylogic.md)把新 helper 和四站初始化片段接入泛化副本。
2. 在本项目环境执行以下只读预检。它读取全部冻结配置和历史账本，核对本批身份是否仍可用。

```bash
cd /home/jinx/Faith/experiments/lagrangian_ppo_v1
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B change_plans/token_attention_001/business_graph_summary_009/generalization_001/grade_assignment_generalization_001/continuous_training_001/mixed_bootstrap.py --check
```

3. 在 AnyLogic 保存、构建并启动一次 `RL_Training`，实际启动由冻结首班身份触发。将上述命令的 `--check` 换成 `--status` 可读进度。首批 32 班计入正式训练。训练的全部 32 班同属一个冻结行为 Actor，经 RPC、物料、动作、采样随机数和旧对数概率验收后才更新。出现 STOP 时保留现场，原 run 与种子由账本锁定。

[冻结 Java helper](prepared_001/Baseline1CampaignPlan.java) · [Python 活动入口](mixed_bootstrap.py) · [训练与独评控制器](mixed_runtime.py) · [全部运行身份](prepared_001/identities.csv)

离线预检已通过。Java 编译并核对全部 2,424 条工况，Python 通过新 u0 初始化与恢复、账本防重与 UNKNOWN 保留，以及历史两班共 340 次决策的精确重放。实际 AnyLogic 联调与闭环结果待接入后验证。[离线验收记录](../../../../../../reports/token_attention_001/business_graph_summary_009/generalization_001/continuous_training_001/offline_verification.json)

## 怎样判断原状态与边是否够用

同一固定矿山场景下，已有旧 u32 在标称固定独评的 100/100 完整任务成立，支持该模型在该配置和评价场景内具备调度能力。品位换位的 0/100 同时限制其跨配置主张。两组评价的共同任务口径是四个 2 小时窗全部品位合规、矿石至少 16,000 吨、废石至少 4,000 吨。[标称封存结果](../../../../../../runs/bg128-u32-generalization-20261119-001/evaluation/nominal/reports/README.md#L3) · [矿源品位换位结果](../../../../../../runs/bg128-u32-generalization-20261119-001/evaluation/grade_swap/reports/README.md#L3)

新混合 u32 如能在标称和留出工况同时保持完整任务，原图和 C128 就有了进一步保留的实证依据。若只在某一类留出场景失败，按品位偏差方向、实际矿废达产、等待和候选窗口兑现追因。明确出现待装材料或服务状态辨识缺口时，再做相同训练分布下的单项边属性对照。明确出现当窗净兑现方向错误时，再做候选摘要对照。历史旧模型与新模型配对比较的是两个固定检查点，单训练种子的结果仍限于本批学习轨迹和评估场景。
