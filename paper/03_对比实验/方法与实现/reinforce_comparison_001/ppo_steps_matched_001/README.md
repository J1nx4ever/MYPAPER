# 与 Full S02 PPO 对齐 Actor 步数的整班 REINFORCE

训练已封存 1,024 班和 23,612 个实测 Actor 步。[最终 u32 的七工况配对评测](paired_inference_001/README.md)

新组使用 Full S02 封存的初始 BG128 Actor、32 批共 1,024 个完整班的工况日程和相同的物理成本。按 Full S02 已提交的逐批实际步数，每批复用轨迹四轮，累计执行 23,612 次 Actor Adam。首批 732 步，每批目标在 720–744 步之间。每轮完整覆盖该批的全部决策。新轨迹长度由策略和仿真决定，实际小批大小会随决策数调整。训练组使用新物理 run、环境和动作种子。[历史梯度步数核验](../../../../../reports/token_attention_001/business_graph_summary_009/reinforce_comparison_001/gradient_audit_001/actor_budget.json)

### 更新规则

每个完整班用四窗品位、班末矿废欠产和车队累计等待计算原成本 $J_b$。班内所有决策共用权重 $A_b=-J_b-\beta$。采样时记录的合法动作对数概率是该批四轮优化的固定分母。对每个决策计算概率比 $r=\exp(\log\pi_\theta(a\mid s)-\log\pi_{\mathrm{old}}(a\mid s))$，Actor 最小化以下小批等权损失。

$$
L=-\operatorname{mean}\left[\min(rA_b,\operatorname{clip}(r,0.8,1.2)A_b)+0.01\,H(\pi_\theta)\right]
$$

使用 Full S02 的 Adam 学习率 `3e-4`、`eps=1e-5` 和全局梯度范数上限 0.5。四轮使用同一旧基线，全部小批成功提交后，旧基线和本批班均回报各取一半。原 PPO 使用批内标准化的 GAE，整班回报版保留 EMA 给出的原始权重。对整班权重做批内去均值会抵消共享的 EMA。Critic 优化步数为 0。该组是使用 PPO 式裁剪和数据复用的整班 REINFORCE 变体。原始 Tang 式（19）基线见上级的历史 S02 训练和独评。[PPO 原论文](https://arxiv.org/abs/1707.06347)

### 准备与现场验证

启动身份和分批配置哈希见[冻结清单](prepared_001/launch_identity.json)，逐班 run 和双种子见[身份表](prepared_001/identities.csv)。[AnyLogic 接入](manual_anylogic.md)包含模型对象、参数变更和控制器入口。

离线预检核验了 32 份 PPO 训练记录、1,024 个工况槽、Java helper、双种子和隔离账本的领取及提交。预检时合并账本为 52,305 班，本活动登记 0 班。[预检证据](../../../../../reports/token_attention_001/business_graph_summary_009/reinforce_comparison_001/ppo_steps_matched_001/prepared_verification.json)

每批核对实际 Actor 步数、四轮决策覆盖数、当前与累计优化步数、班均成本和 EMA。正式训练的结果应在完成 32 批后报告同样的 23,612 个 Actor 步，并保留全部物理回执。和 Full S02 比较时，Critic 步数及实际参与 Actor 反传的决策行数单列。

[首批离线回放程序](verify_replay.py)走正式更新路径，重放 32 个归档班的 5,844 次决策。观察到 732 次 Adam 调用和四轮累计 23,376 行。模拟检查点封存后、账本提交前中断，恢复时新增 Adam 调用为 0，恢复后的 Actor 可导出。[实测回执](../../../../../reports/token_attention_001/business_graph_summary_009/reinforce_comparison_001/ppo_steps_matched_001/real_first_batch_replay.json)
