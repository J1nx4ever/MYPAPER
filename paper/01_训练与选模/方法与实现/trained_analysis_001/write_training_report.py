"""Publish interpretation of the completed, read-only numerical diagnostics."""
from pathlib import Path
import hashlib,json

ROOT=Path('/home/jinx/Faith/experiments/lagrangian_ppo_v1')
PLAN=ROOT/'change_plans/token_attention_001/business_graph_summary_009/trained_analysis_001'
REPORT=ROOT/'reports/token_attention_001/business_graph_summary_009/trained_analysis_001'
def ref(p):p=Path(p).resolve();return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
p=REPORT/'training_reward_convergence_gradient.json';x=json.loads(p.read_bytes())
assert x['status']=='PASS_OFFLINE_ANALYSIS_NO_UPDATES' and len(x['batches'])==16 and len(x['gradient_diagnostics'])==5
rs=x['batches'];ds=x['gradient_diagnostics'];best=rs[13];first=rs[0]
x['interpretation']=dict(
    convergence='Strong improvement into a low-cost region; insufficient evidence of stable optimization convergence. The frozen update13 generalizes to100 new shifts; later training policies have noisier higher cost.',
    reward_implementation='All16 batches rebuilt from canonical material events and recorded physical waiting/production agree with original costs, MC, raw GAE and single joint standardization. No Actor/Critic objective mismatch found.',
    reward_signal='The grade deadband sparsifies direct realized grade cost but does not eliminate delayed grade GAE or frozen Critic bootstrap information. Direct waiting policy gradients remain smaller and sometimes oppose joint policy gradient.',
    critic='Late raw MSE is much lower while target variance also shrinks. Independent pooled EV remains0.16849; the value model is useful but has substantial residual error.',
    actor_updates='Mean preclip total gradient norms above0.5 establish some clipping, not its frequency. Later net checkpoint parameter changes remain about5% of prior Actor parameter L2.',
    adam='Cost-specific Adam attribution is unavailable. Autograd norms, cosines and signed projections diagnose one frozen full-batch surrogate only; actual checkpoint-to-checkpoint delta is the net joint four-epoch Adam result.')
x['gradient_method']=dict(
    decomposition='A_joint=A_grade(zeroV)+A_production(zeroV)+A_wait(zeroV)+A_value(zeroCost,oldV); center each term by its own full-batch mean and divide every term by the same original joint std+1e-8.',
    old_value_term='sum_l lambda^l*(V_{t+l+1}-V_{t+l}); at lambda=.95 this includes future-value bootstrapping and is not only a current-state subtractive baseline. Actor gradients do not propagate into the frozen Critic.',
    gradient='For each component use mean(ratio*its_joint_scaled_advantage) at original pre-update weights; separately entropy=-.01*mean(H). Joint gradient is recomputed through original max(r*A,clip(r)*A) surrogate.',
    clipping='All replay ratios differ from1 by at most1.20e-6; zero rows outside[.8,1.2]. Decomposition therefore applies to the pre-update unclipped point and is not a decomposition of four epochs of clipped Adam updates.',
    centering='Independent centering is algebraically additive because the common denominator is unchanged. It is a defined decomposition, not unique causal credit assignment.',
    aggregation='All original32 shifts and all decisions, equally weighted by decision count, recomputed in bounded16-row chunks; no optimizer step, no input intervention, no new simulation.',
    diagnostics='l2_norm, dot, cosine and signed projection against joint policy gradient; projections can be negative or exceed1 and are not Adam shares. Pairwise cosines and module squared-gradient fractions are diagnostic only.',
    error_bounds=dict(max_raw_gae_abs=max(d['closure']['raw_gae_max_abs_error'] for d in ds),max_normalized_adv_abs=max(d['closure']['normalized_advantage_max_abs_error'] for d in ds),max_gradient_relative=max(d['closure']['gradient_plus_entropy_relative_error'] for d in ds)),
    missing_actual_logs=['per-minibatch cost gradients','per-minibatch preclip norm distribution or clipping frequency','per-step Adam parameter delta by objective','causal allocation of the single joint Critic bootstrap to individual business costs'])
x['plots']={name:ref(REPORT/name) for name in ('training_reward_convergence.png','gradient_contributions.png')}
x['report_writer']=ref(__file__)
p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
batchtable='|实际采样策略|成本均值±SE|品位/生产/等待成本|零品位成本窗/128|RMSE|完整任务率|前/后MC MSE|前/后EV|\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
for r in rs:
    c=r['cost_components'];batchtable+=f"|u{r['behavior_checkpoint_update']}|{r['cost_mean']:.4f} ± {r['cost_se']:.4f}|{c['grade']['mean_cost_per_shift']:.4f}/{c['production']['mean_cost_per_shift']:.4f}/{c['wait']['mean_cost_per_shift']:.4f}|{r['zero_grade_cost_window_count']}|{r['grade_rmse']:.5f}|{r['complete_task_rate']:.1%}|{r['critic_before_mse']:.4f}/{r['critic_after_in_sample_mse']:.4f}|{r['critic_before_ev']:.3f}/{r['critic_after_in_sample_ev']:.3f}|\n"
gradtable='|冻结策略|品位PG范数|生产PG范数|等待PG范数|价值TD/bootstrap范数|熵项范数|联合policy范数|品位/等待范数比|等待与联合cos|\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
for d in ds:
    g=d['gradients'];gradtable+=f"|u{d['behavior_checkpoint_update']}|{g['grade']['l2_norm']:.6f}|{g['production']['l2_norm']:.6f}|{g['wait']['l2_norm']:.6f}|{g['old_value_baseline']['l2_norm']:.6f}|{g['entropy']['l2_norm']:.6f}|{g['joint_policy']['l2_norm']:.6f}|{g['grade']['l2_norm']/g['wait']['l2_norm']:.2f}|{g['wait']['cosine_with_joint_policy']:.3f}|\n"
text=fr'''# BG128 训练 Reward、收敛性与梯度贡献分析

**训练已经学到有效策略，所选 u13 在100个新班次上保持表现；现有证据还不能判定优化过程已稳定收敛。** 初始行为策略 u0 的平均联合成本为4.52180，u13为0.32235，下降{1-best['cost_mean']/first['cost_mean']:.1%}。独评固定 u13 后成本0.32751、完整任务率98%，与训练期间的低成本水平接近。后续 u14、u15 的32班成本升到0.45234、0.49178；各批seed不同且只有32班，均值回升是风险信号，不足以单独证明统计显著退化，更不能据此判断未采样的最终 u16 的性能。

## 1. 同一个训练目标已核对

本分析复用了原始512班的 sealed HDF5事件、等待增量、生产统计及全部16份 `training_targets.h5`。逐班重建成本、MC和GAE，与训练时保存的标签逐项核对；再验证各项GAE之和及统一标准化后的优势之和。没有发现 Actor/Critic 使用了不同目标或奖励尺度混用。

训练采用成本最小化，对应 reward 可记为 **R=-C**：

\[
C=\sum_{{w=1}}^4\min\left(\frac{{(|G_w-.27|-.004)_+}}{{.006}},2\right)
+8s_{{ore}}+8s_{{waste}}+0.2W/28800.
\]

空窗成本2；矿石目标16000、废石目标4000；6000只作为废石停派阈值。W是**全车队累计等待车辆秒**，会累加不同车辆同时等待，分母28800是归一化基准，不能把W误当仿真墙钟时间。旧RPC的 `grade_reward_raw` 是物理证据，训练标签按当前deadband成本重建；本报告没有直接把旧RPC reward之和当新训练目标。

## 2. Reward改善来自哪里

初始平均品位成本4.37652，占总成本96.8%；等待成本0.14490，占3.2%。u13的品位/生产/等待成本变为 **0.19346/0/0.12889**，占比 **60.0%/0%/40.0%**。成本下降主要来自品位改善；等待有改善，但未随品位成本同步下降一个数量级。这些是标量目标的占比，不能解释成梯度或Adam更新占比。

u13的128个窗口中，105个窗口的品位成本为0，23个仍超出[.266,.274]死区；四窗业务合规用[.26,.28]判定，因此可以同时出现“32班四窗均合规”和“仍存在品位成本”。RMSE约0.002992不会自动意味着每个窗口都落在deadband内。

deadband内不奖励更靠近中心，是当前有意冻结的业务取舍。它让直接品位结算变稀疏，但**某一窗口即时品位成本为0，不代表该处Actor没有品位相关信号**：后续超带窗口的GAE仍可回传；全批中心化会改变逐行分项优势；冻结价值的TD/bootstrapping项也包含联合目标的预测。

生产成本是末端缺口惩罚。u10、u13所采32班没有实际生产缺口，直接生产cost/PG均为0；这不能证明生产约束已成为硬保证。u7阶段废石达产只有17/32，u15阶段再次出现一个废石缺口班次；失败班次完整保留。

等待权重的量纲值得关注：1个单位grade成本相当于144000车辆秒（40车辆小时）等待；超出deadband再偏离0.001产生约0.1667成本，相当于24000车辆秒等待。在当前权重下，规避少量品位外带尾部仍可能比削减等待更有价值。独评废石平均约6029.5吨，超过4000目标的部分没有直接生产奖励或过量惩罚，相关机会成本只通过等待和其他结果间接体现；这符合现有任务定义，不能在本次事后换目标重新评分。

{batchtable}

表中“前”是该行为策略的冻结Critic在随后新采样32班上的误差；“后”是更新一次后在同一批训练数据上的拟合误差。采样策略uK对应 `train_u(K+1)`，同批更新产出checkpoint u(K+1)，两者不可混用。

## 3. 是否收敛

业务上已进入低成本区域，但还没有稳定平台与优化停止的证据。u13以后可见均值回升、MC目标方差增加、Actor仍有明显移动。u13→u14、u14→u15的全批重放近似KL为0.01963、0.02602，clip fraction为28.8%、38.2%；它们是4轮PPO更新后的诊断，不是逐minibatch的实际clip操作频率。

同期记录的Actor平均**裁剪前**梯度范数为0.735、0.711，高于阈值0.5，说明至少部分minibatch触发梯度裁剪；现有日志只有均值，不能据此求出裁剪频率、最大值或声称每个minibatch都被裁剪。平均梯度不趋零也不能单独当作“不收敛”的证明，因为这是有噪声且每批分布变化的PPO。

原始joint GAE标准差从0.48792降到u13的0.07972，相同统一标准化公式中的倒数尺度提高约6.12倍。即使实际成本减少，剩余尾部差异仍被标准化到可观的更新尺度；这是数值事实，并非已证明梯度增长由该因素单独导致。

相邻sealed checkpoint还可以直接量到实际**联合Adam训练后的净参数变化**：u13→u14、u14→u15、u15→u16的Actor净Δθ L2分别1.9178、1.8938、1.8774，约占更新前参数L2的5.09%、5.02%、4.97%。它们包含4epoch全部minibatch、动量、预条件和裁剪后的净结果；没有趋近0。净参数移动也不等价于策略函数同幅度变化，不能用它替代KL或性能评价。

Critic绝对MSE显著下降，但MC目标方差也从3.8428降至u13的0.04077，因此不能只用MSE下降宣称已准确理解长期业务。u13更新前MSE/EV为0.02954/0.3007；100班独评 pooled MSE/EV为0.04178/0.1685，说明仍有相当残差。独评单班EV的简单平均还会受各班目标方差影响，不能与pooled EV混算。

## 4. 直接policy梯度的离线实测

固定原始u0/u6/u10/u13/u15，每个点使用其原32班全部决策和原冻结标签重算autograd，不调用 `optimizer.step()`。Actor目标按原代价PPO实现取MAX surrogate。定义：

\[
A=A_g+A_p+A_w+A_V,\quad
A_j=\operatorname{{GAE}}(c_j,V=0),\quad
A_V=\operatorname{{GAE}}(c=0,V_{{old}}).
\]

每项减去各自的**整批均值**后，都除以**原joint优势的同一个标准差+1e-8**，其和严格还原原标准化优势；没有分别标准化三种业务成本。

必须注意价值项：

\[
A^V_t=\sum_l\lambda^l(V_{{t+l+1}}-V_{{t+l}})
=-V_t+(1-\lambda)\sum_{{k\ge1}}\lambda^{{k-1}}V_{{t+k}},\quad V_T=0.
\]

λ=.95时，这项包含未来价值bootstrap，**不仅是当前状态减baseline**。Critic预测在形成GAE前冻结，不会通过Actor loss反传更新Critic；但它所提供的优势信息属于整个joint目标，没有三个价值头可将其再按业务唯一分配。

{gradtable}

u13直接品位梯度范数约为等待的12倍；u15约22.7倍。u13的等待方向与joint policy梯度cos=-0.222，u15=-0.309，表示在这些冻结样本上存在方向冲突。u13价值TD/bootstrapping项范数0.16431，超过直接品位的0.06454；这提示要重视价值估计和后期优势质量，但不能据此证明Critic导致后续性能回退。

各分项向量相加后，与原MAX surrogate直接求得的joint梯度相对误差最大约1.06e-6；加上熵项也闭合。所有重放ratio偏离1不超过1.20e-6，没有进入[.8,1.2]之外，因此这是**更新前未clip点的完整batch诊断**。它既不是原逐minibatch梯度分布，也不是4epoch更新全过程的分项归因；ratio进入裁剪区后，不能把每个分项独立clip再声称仍是原联合PPO。

图中的norm和cos是向量诊断，生产项为0时cos未定义，在对数图中不画0点；连线只是离散checkpoint之间的视觉引导。JSON还提供dot、signed projection、分项两两cos和参数模块的梯度平方范数分布。负投影与相互抵消意味着不能将norm归一为一组“贡献百分比”；模块梯度分布也不等于模型已学会正确业务语义。

## 5. 能回答与不能回答的梯度贡献

可以回答：本轮cost如何分配、原始GAE信号如何分解、五个冻结点的直接policy梯度强度/方向、训练记录的联合总梯度均值、相邻checkpoint的实际净Δθ。

不能回答：grade/wait/production各占实际Adam参数更新多少。原因是没有保存逐步分项梯度和更新，Adam还含历史一阶/二阶矩、联合梯度裁剪、epsilon与4epoch交互。不能用成本比、范数比、投影比或一次全批梯度比冒充实际Adam贡献；单一joint Critic的bootstrap信息也没有唯一分项归属。

## 6. 下一步改进优先级（本次未执行）

1. **先稳定优化，再改网络。** 保留本轮u13与100班结果；下一次在同一架构、同一cost下预先固定更保守的Actor更新预算或学习率，并记录每epoch KL、clip比例及梯度分布，采用预注册停止/选择规则。现有后期KL、净参数移动和成本回升支持优先验证此项，但尚不能断言某个学习率就是唯一原因。
2. **提升单Critic与优势诊断。** 增补按窗口/阶段/是否缺口分组的MC残差、bias、target variance与EV，检查早班尾部和生产缺口预测，仍保持单joint价值目标。不要将低MSE单独作为收敛或最佳Actor标准。
3. **等待权重只做独立固定权重实验。** 若业务要进一步降低累计等待，可预先冻结一个新的等待权重对照，Actor和Critic同步用完全相同成本，独立报告品位尾部、生产和等待。不能依据12倍梯度范数直接决定应将等待权重乘12，也不采用逐minibatch自适应梯度比例。deadband中心精度是否还应优化、废石是否应接近4000而非允许6000，先作为业务目标选择，不能事后改本轮评价。
4. **最后做表征干预与多seed验证。** 保持两层GAT/四组关系/C128冻结，先对已训练模型做真实状态自洽的边/实体/摘要分组干预和logit/价值误差分析，再判断是否存在表征遗漏。当前cost曲线和梯度范数不能单独证明图结构有问题，也不能证明C128已取代图。

## 7. 证据与运行边界

- [完整数值JSON](training_reward_convergence_gradient.json)
- [训练reward与收敛图](training_reward_convergence.png)
- [冻结点梯度强度与方向图](gradient_contributions.png)
- [100班独评原始汇总]({ROOT/'runs/business-graph-summary-best-checkpoint-eval100-20261118-001/reports/summary.json'})

本次新建分析文件与图片；只读512班历史数据和sealed checkpoint。五个点均通过模型参数、优化器、训练账本与RNG的只读守护；零新班次、零RPC、零训练更新。受保护 `python/episode_budget_ppo.py` SHA保持 `{x['protected_ppo_sha_after']}`，全部训练source pins未变。初次执行的数值脚本原字节保留为 `analyze_training_numerical_001.py`；随后只调整图例名称，内部key保持 `old_value_baseline`，图中显示 `Frozen value TD/bootstrap`。
'''
(REPORT/'training_reward_convergence_gradient.md').write_text(text)
manifest=dict(status='PASS_OFFLINE_ANALYSIS_NO_UPDATES',artifacts=[ref(REPORT/n) for n in ('training_reward_convergence_gradient.json','training_reward_convergence_gradient.md','training_reward_convergence.png','gradient_contributions.png')],numerical_script=x['analysis_script'],plotting_script=x['plotting_script'],report_writer=ref(__file__),physical_shifts=0,optimizer_updates=0)
(REPORT/'training_analysis_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(manifest,ensure_ascii=False,indent=2))
