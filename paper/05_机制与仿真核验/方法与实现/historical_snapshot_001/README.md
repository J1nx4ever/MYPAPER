# 历史32班与相邻决策快照离线审计

2026-11-18，Asia/Shanghai。本任务严格限定为历史数据读取、正式BG009构图、确定性数量/C128计算、静态绘图及008成本核验。没有模型实例化或Forward，没有训练或参数更新，没有RPC/AnyLogic/预算登记。

最终交付见 [audit_report.md](../../../../reports/token_attention_001/business_graph_summary_009/historical_snapshot_001/audit_report.md)；全部文件及哈希见 [manifest.json](../../../../reports/token_attention_001/business_graph_summary_009/historical_snapshot_001/manifest.json)。旧有 `verify_offline.py` 是模型工程验收入口，不属于本次历史只读流程；本次只使用本目录脚本。

## 数据与选择

使用历史TQ128 `train_u0016` 完整32班，共5841个有效决策。相邻快照为`run_9810998`原始state61→62，精确时刻9603.707937→9717.867212秒，Ego Truck8→Truck0。选择规则为班中20%—75%范围内、时间间隔(0.01,200)秒、矿石装载阶段变化最多，平分按run/index排序；这是事后展示筛选，不作性能统计。

每个快照以自身1519D原观测独立调用正式`build_graph`和C128函数，禁止另一时刻信息补边。事实图与候选读取索引分离。25种指Schema关系词表，包含road；不是每张图必须包含全部类型。当前19节点、16道路、21维数量均按正式schema校验。

## 脚本职责

- `build_snapshots.py`：检查32班原始字段/时间/地图；独立构图；输出选择、两时刻原始字节值、边CSV、全部1/2边walk、物理节点字段、数量oracle及C128差异。
- `audit_rewards.py`：按已封印物料事件与RPC原记录重建当前008成本，核对冻结MC/逐步目标，并区分当时已知与后验结算。原RPC reward与008成本分别列出。
- `render_snapshots.py`：Pillow静态绘图，固定节点布局，输出两张快照及差异PNG。正/逆消息可共用双箭头曲线，完整有向多重边始终保留在CSV。图中变化着色、事后业务说明和Reward旁注是审计展示，不进入事实图或C128。
- `finalize_audit.py`：核对完成的独立审查、源SHA、完整交付，生成报告与manifest。独立审查结论来自另行只读核对，不通过原构图器自证替代。

## 复现核心数据和图

在仓库根目录执行下列命令，使用新的输出目录保留本次证据。默认历史来源固定到上述32班；不使用新run/seed，不触发预算或通信。

```bash
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B \
  experiments/lagrangian_ppo_v1/change_plans/token_attention_001/business_graph_summary_009/historical_snapshot_001/build_snapshots.py \
  --output /tmp/bg009-historical-snapshots

/home/jinx/miniconda3/envs/anylogic-env/bin/python -B \
  experiments/lagrangian_ppo_v1/change_plans/token_attention_001/business_graph_summary_009/historical_snapshot_001/audit_rewards.py \
  --selection /tmp/bg009-historical-snapshots/selection.json \
  --output /tmp/bg009-historical-snapshots

/home/jinx/miniconda3/envs/anylogic-env/bin/python -B \
  experiments/lagrangian_ppo_v1/change_plans/token_attention_001/business_graph_summary_009/historical_snapshot_001/render_snapshots.py \
  --output /tmp/bg009-historical-snapshots
```

最终独立审查及`finalize_audit.py`的封存结果以当前报告目录为准。重新输出到其他目录时，不复制旧审查记录冒充一次新审查。

## 解释边界

两层路径只核对拓扑/特征通道，不声称已学习关系。`node_features.csv`为原始物理字段及确定性描述，不生成随机64D神经特征。已载车清空的历史Load来源、未知载荷品位、未来真实ETA和未结算窗口结果不补猜。候选未来去向不写成已派发边。

当前Reward并不按每次装完吨位即时结算品位或达产。所选区间487吨装载完成、Truck0卸91.7吨、实际破碎79.277吨属于不同阶段；该区间008成本只产生等待0.001585545477，结果在T1才已知。完整后验结果仅用于审计。
