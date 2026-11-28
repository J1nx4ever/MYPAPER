# 变拓扑训练的班次可行性门

场景和业务目标合在一起判定。只有完整可执行的解通过复核，才可把对应场景放进 `RL_Training` 的新准备包。定义任务后冻结目标，未通过的布局可以调整路网、在役车队或装卸资源，再建立独立配置复核。

## 运力初筛

`screen_capacity.py` 为每种车型、空车起站、装载点及对应卸点建立车次变量。每趟占用车辆的时长包括空程、装车、重程和卸车。约束同时检查车型组车时、装卸资源时长、源库存、粗破小时能力、矿废吨量和全班平均品位。

本轮用 8 小时、矿石实际破碎 16,000 t、废石实卸 4,000 t 作为同口径示例，沿用四装点各一铲、粗破三卸位、废石两卸位及粗破 2,500 t/h、缓冲区 500 t 的模型参数。初筛使用满载和可分车次，无排队，空重车分别以 6.5 和 5.0 m/s 为上限并服从道路限速。装卸时间按模型的名义公式，品位用装点均值。根据这些假设，A 和 B 可同时达到约 1.25 倍矿废目标，受粗破名义产能约束。这个结果是运力筛查，无法证明四个窗口和 FIFO 的时序可实现，也无法证明拥堵、故障或随机料块下仍达标。

示例的 6–9 车布局保留当前编译器的连通性、路段唯一性和角色约束。单边 10 km 是观测缩放上限。是否能达产由整个空重车循环和资源分配决定。

## 可执行解的证据

下述已封存的两班见证使用固定的九车矿废目标。七车按设计运力缩放后，废石停派线也随之下降。核验该工况时要使用缩放后的任务与回执，并逐窗检查品位。 [B 组缩放任务](../frozen_policy_trial_001/capacity_scaled_001/README.md)

1. 用预先冻结的布局、出勤和车型、铲和卸点能力、限速、启动库存、调度规则与环境实现，跑一个完整原生见证班。执行合法派单，保留四窗、实际装卸和 FIFO 边界事件。一次正常完整的合格班即可证明该固定输入下存在一条可行轨迹。启发式失败或规划器超时，记为尚无合格见证。
2. `admit_layouts.py` 复用 `python.acceptance_events.replay_event_bundle`，重算实际矿石破碎、废石卸达和金属守恒。四个两小时窗都要有实际破碎产量且品位在 26%–28%。矿废同时达标，废石保留当前任务的单车超量上限，舍入守恒容差沿同一任务。
3. 见证配置字节、布局、任务、原生实际场景、模型、run、recording、seed 和事件封存逐项绑定。实际场景必须从 AnyLogic 构建后的对象提取。更换物理配置或外生实现时重新验收。

A 图 9 车和 B 图 7 车的确定性编组各通过了一次原生 8 小时见证，两班均已登记完成。将 `require_admission(...)` 接入变拓扑训练准备入口时，缺少合格证据的场景在生成计划前拒绝。运行方法和结果见 [A/B 原生整班可行性见证](full_shift_001/README.md)。

## 调用与见证字段

以项目的 `anylogic-env` 执行 `screen_capacity.py` 运行初筛。执行 `admit_layouts.py --witness-index PATH` 复核完整班，省略索引时直接报告未准入。索引按布局 ID 指向独立见证的以下字段：

- `run_index`、`recording_id`、`environment_seed`。
- `config_path`、`config_sha256`、`event_dir`、`event_manifest_sha256`。
- 见证配置 JSON 使用 `purpose=topology_full_shift_feasibility`，绑定 `layout_sha256`、`task_sha256`、`model_sha256` 和 `expected_runtime_scene_sha256`。任务和实际场景 SHA 按 `admit_layouts.canonical` 编码。
- 封存的原生事件 manifest 在 `config.actual_runtime_scene` 内回传实际节点、边、车辆、装卸设施、库存、外生配置及模型身份，并在根字段记录班末 `fallback_dispatches` 和 `invalid_dispatches`。

[初筛数据](/home/jinx/Faith/experiments/lagrangian_ppo_v1/reports/token_attention_001/business_graph_summary_009/variable_topology_training_001/feasibility_001/capacity_screen.json) · [初筛后的历史准入结果](/home/jinx/Faith/experiments/lagrangian_ppo_v1/reports/token_attention_001/business_graph_summary_009/variable_topology_training_001/feasibility_001/admission_status.json)
