# S02 训练预算与梯度方向

同一物理采样预算下，S02 PPO 后期的完整达标率更高。每组从同一初始 Actor 出发，在同一混合工况日程采集 1,024 个完整班，训练种子实现与动作轨迹独立。两者使用相同的 32 个批次编号。每批封存一次 Actor。

| 训练实测量 | Full S02 PPO | 整班 REINFORCE S02 |
| --- | ---: | ---: |
| 采样决策数 | 188,386 | 189,239 |
| Actor 梯度计算使用的决策行次数 | 753,544 | 189,239 |
| 每批 Actor Adam 步数 | 720–744 | 1 |
| 全程 Actor Adam 步数 | 23,612 | 32 |
| 全程 Critic Adam 步数 | 23,612 | 0 |

PPO 每批采集 32 班，按 32 个决策切小批、重复四轮。REINFORCE 将各班的梯度累加后执行一次 Adam。前者的 Actor 步数为后者的 737.875 倍，参与 Actor 反传的决策行次数为约 3.98 倍。运行和验证中的额外前向计算不计入行次数。两者的总计算量还受 Critic、轨迹验证和实现开销影响。[逐批优化预算核验](../../../../../reports/token_attention_001/business_graph_summary_009/reinforce_comparison_001/gradient_audit_001/actor_budget.json)

## 优先控制 Actor 步数

保持原始 Tang 式（19）S02 作低步数参照。另设一个使用 S02 PPO 优势估计的 Actor 步数消融组，从相同 u0 出发，继续采 32 批、每批 32 个完整新班。每批在冻结的采样策略上计算原 GAE 和优势标准化，使用原 PPO 的概率比、熵项和学习率。按所有决策的总数加权、分块累加 Actor 梯度，然后统一裁剪并执行一次 Actor Adam。每个分块执行反向传播，整批参数只改变一次。Critic 独立拟合并单列优化预算，比较 PPO 自身的高低 Actor 步数组时优先维持原 Critic 的迭代规则。

此组能和原版 REINFORCE 对齐物理班数、Actor 步数以及单次数据遍历。与 Full PPO 的比较揭示减少 Actor 步数和数据复用后的综合影响。由于每批只在采样参数上求一次梯度，初始概率比约为 1，PPO 裁剪在该步通常不起作用。此组应标为“单 Actor 步的 GAE 对照”。实际采用增加 REINFORCE Actor 步数的实验设计。[新活动](../ppo_steps_matched_001/README.md)

已在内存回放 Full S02 首批 32 班的 5,844 次决策，从封存 u0 出发，使用原批次 GAE、旧动作概率、熵项和 Adam 配置，验证所有 Actor 参数的 Adam 计数均只增加 1。原 PPO 在这批执行了 732 个 Actor 步。此次只验证 Actor 更新路径，新增物理班和已保存模型均为零。[单步代码](matched_actor_step.py) · [真实首批单步验收](../../../../../reports/token_attention_001/business_graph_summary_009/reinforce_comparison_001/gradient_audit_001/matched_actor_step.json)

若问题专门针对“在原 PPO 更新规则下，整班回报与 GAE 有多大差别”，可做独立的高步数优势估计消融。沿用行为 logp、概率比裁剪、熵项、梯度范数裁剪和 4 轮数据复用，仅替换冻结的优势输入。为严格匹配 23,612 个 Actor 步，按已存的每批步数将新轨迹的所有决策划为对应数量的小批，每轮覆盖全部决策。用原 PPO 的逐批优势标准化时，同批共享的标量 EMA 会在中心化时抵消，此试验能够比较整班回报与 GAE。它属于 PPO 更新规则下的消融，和原版整班 REINFORCE 各自报告。

## 已验算的梯度

两组真实 REINFORCE 更新（u1、u32）分别用 32 班重算物理成本，在所有 Actor 参数上独立求奖励目标梯度，和 Adam 一阶矩还原的损失梯度比较，方向符合损失最小化约定。三批的评分头解析梯度和 32 次全网络 Adam 日志也吻合。该验证针对固定采样轨迹上的估计量。仿真期望回报对参数的真实导数未由这一回放直接观测。[全参数方向和有限差分](../../../../../reports/token_attention_001/business_graph_summary_009/reinforce_comparison_001/gradient_audit_001/direction_audit.json) · [Adam 更新](../../../../../reports/token_attention_001/business_graph_summary_009/reinforce_comparison_001/gradient_audit_001/adam_audit.json)

全部训练决策中，72.57% 发生在首个品位窗结算后。历史整班权重仍包含已结算的成本。另做的离线信用分配诊断，对每个动作只计从该动作起的实际剩余品位、欠产和等待成本，并从此前批次按已结算窗口数估计剩余回报基线。回报和折扣保持原业务口径。u1/u16/u32 共 96 班上，评分头末层 128 个权重的班间梯度离散度（样本协方差迹）依次从 12.70、6.71、196.47 变为 5.73、3.73、85.39。仅替换剩余回报、继续用原整班标量基线时，u16 的离散度反而升至 13.58。此诊断度量单个参数层在三批封存数据上的波动，尚无对应新策略的闭环性能结果。步数消融可独立进行。[逐班回报复核](../../../../../reports/token_attention_001/business_graph_summary_009/reinforce_comparison_001/gradient_audit_001/signal_audit.json) · [信用分配回放](../../../../../reports/token_attention_001/business_graph_summary_009/reinforce_comparison_001/gradient_audit_001/credit_replay.json) · [基于后续回报的策略梯度](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#don-t-let-the-past-distract-you)
