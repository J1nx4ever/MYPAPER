# BG128 Checkpoint13 原始观测干预

使用已冻结的正式训练 Checkpoint13，重做历史 `physical_intervention_001` 同一 T1、同一 A/B/C 的原始观测分支。分支从共同 BASE 独立构造，逐字节比对旧冻结输入，重新构建事实图、21维数量、H0/H1/H2 和原始 C128。旧随机初始化结果不作为已学习语义的证据。

- A：Truck1 完成 Load0 的已知 91.7 吨装载，释放服务和预留，成为目标 Dump0 的已载车辆。仍为9辆车。这同时改变装载完成、服务、承诺和物料状态，不能归因于单一拥堵因素。
- B：Truck8 将193吨未开始服务的预留从 Load0 改为 Load1，同步账本与未来路线尾，保留当前位置和历史。
- C：仅将原始 `dump.0.crusherRate` 从2500改为1250吨/小时，不改变 busy、FIFO、库存或过去累计量。此输入确实表达当前处理能力降低。

正式训练512班的该处理速率配置始终为2500吨/小时，C的1250属于参数外推；图支路响应弱不能单独判为两层GAT失效。C128工程投影会显式按当前速率重新计算，与网络通过变化速率的训练样本学习是两件事。

主比较均走正式 Actor/Critic 完整前向，输出四动作合法性、吨位、Logit、减去合法动作平均值后的 Logit、概率和单联合成本 Critic。层间消息按 transport、commitment、service、material 分组保存。

另列机制探针：删除指定层的消息边但保留完整事实图构建出的数量、Token输入和 C128，以及图/C128 在读出接口的四组合比较。它们不代表物理状态，不作为仿真收益或唯一因果归因；数量只按事实车辆计数一次，多种编码路径不等于把吨位重复加总。

运行（不会训练、启动RPC、启动AnyLogic或登记预算）：

```bash
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B experiments/lagrangian_ppo_v1/change_plans/token_attention_001/business_graph_summary_009/trained_intervention_001/analyze.py
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B experiments/lagrangian_ppo_v1/change_plans/token_attention_001/business_graph_summary_009/trained_intervention_001/review.py
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B experiments/lagrangian_ppo_v1/change_plans/token_attention_001/business_graph_summary_009/trained_intervention_001/audit_training_support.py
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B experiments/lagrangian_ppo_v1/change_plans/token_attention_001/business_graph_summary_009/trained_intervention_001/render_report.py
```

证据位于 [trained_intervention_001](../../../../reports/token_attention_001/business_graph_summary_009/trained_intervention_001/)。历史原始文件、旧干预数据、模型、架构和已冻结源码保持原字节。主脚本拒绝覆盖既有 `summary.json`，以保留本轮证据。
