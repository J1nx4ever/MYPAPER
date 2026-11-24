"""Render conclusions from sealed offline evidence; no model execution."""
from collections import defaultdict
import numpy as np
from common import *

def csv_read(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def strong_cancellation(rows):
    result=[];examples=[]
    for scene in 'ABC':
        groups=defaultdict(list)
        for r in rows:
            if r['role']=='actor' and r['scene']==scene:groups[r['lr'],r['run_index'],r['state_index']].append(r)
        retained=[];strong=0
        for key,rs in groups.items():
            g=np.array([float(r['graph_shapley']) for r in rs]);c=np.array([float(r['c128_shapley']) for r in rs])
            denom=np.abs(g).sum()+np.abs(c).sum()
            retained.append(float(np.abs(g+c).sum()/denom) if denom else 1.)
            severe=[r for r in rs if float(r['graph_shapley'])*float(r['c128_shapley'])<0 and
                min(abs(float(r['graph_shapley'])),abs(float(r['c128_shapley'])))>=.001 and float(r['retained_fraction'])<=.5]
            strong+=bool(severe)
            examples.extend(severe)
        result.append(dict(scene=scene,eligible_states=len(groups),strong_cancellation_states=strong,
            mean_vector_L1_retained_fraction=float(np.mean(retained)),
            p25_vector_L1_retained_fraction=float(np.quantile(retained,.25))))
    return result,sorted(examples,key=lambda r:float(r['retained_fraction']))[:5]

def main():
    replay=read(OUT/'replay.json');inter=read(OUT/'interventions.json');late=read(OUT/'late_updates.json')
    chain=read(OUT/'dispatch_realization.json')
    joint=read(OUT/'joint_support.json')
    validation=read(OUT/'validation.json');assert validation['status']=='PASS'
    recon=OUT/'minibatch_replay_002';complete=read(recon/'complete.json')
    assert complete['status']=='PASS_EXACT_RECONSTRUCTION'
    cancellations,examples=strong_cancellation(csv_read(OUT/'readout_decomposition.csv'))
    mrows=csv_read(recon/'minibatches.csv');epochs=csv_read(recon/'epochs.csv')
    mini=[]
    for n in (14,15,16):
        rs=[r for r in mrows if int(r['update'])==n];w=np.array([int(r['minibatch_size']) for r in rs])
        kl=np.array([float(r['sampled_reverse_approx_kl_before_actor_step']) for r in rs])
        cl=np.array([float(r['clip_fraction_before_actor_step']) for r in rs])
        mini.append(dict(update=n,minibatches=len(rs),sample_weighted_KL=float(np.average(kl,weights=w)),
            minibatch_KL_p95=float(np.quantile(kl,.95)),minibatch_KL_max=float(kl.max()),
            sample_weighted_clip_fraction=float(np.average(cl,weights=w)),
            minibatch_clip_fraction_p95=float(np.quantile(cl,.95))))
    failures=[r for r in replay['trajectories'] if not r['compliant']]
    pairs=defaultdict(list)
    for r in replay['trajectories']:pairs[r['ordinal'],r['window']].append(r)
    assert all(len(v)==2 and sum(r['compliant'] for r in v)==1 for v in pairs.values())
    lines=['# 下一阶段机制研究：首轮离线证据', '',
        '保留 **3e-4/u13** 作为当前主要基线。现有证据支持把重点前移到“承诺在截止前实际兑现多少物料”，并继续排查图/C128相抵和后期更新；尚未证明具体早期替代动作必然能修复失败，也尚未证明后期 Checkpoint 的独立性能下降。','',
        '本轮新增物理班次、RPC、正式训练更新、模型文件写入、预算写入均为 **0**。原更新14–16仅在内存副本中做数值重构，重构的Actor/ Critic各自优化步另计；最终状态必须与原封存 Checkpoint 完全一致。ALP、原学习器与受保护 PPO 核心未修改。','',
        '## 物料兑现：已有 FIFO 会锁定尾段结果','',
        '旧2个失败、新8个失败及其同种子成功对照共20条真实轨迹、987个决策前缀。10个失败中8个在首窗，2个在末窗。本批两个学习率的矿废达产率均为100%，这里排查的是窗口品位兑现。所有前缀的窗口Q、M与FIFO均对账，最大原始观测差异为 '+f"{max(r['accounting_max_abs_error'] for r in replay['trajectories']):.6f} t"+'。10个成功对照均未触发排除；全部已观察结果位于乐观前缀范围内。','',
        '下表的证书保持已有FIFO、固定2500 tph work-conserving破碎器；后续新料即使瞬时到达、品位任取[0,1]，也不能越过FIFO。宽误差检验额外向外放宽0.1 t，所以会保守地漏掉极近边界的例子。表中时间是第一个**已观测到**证据的决策，不是精确失效时刻。','',
        '|失败run|窗口|实际处理品位|宽误差证书时刻(s)|距截止(s)|截止FIFO(t)|',
        '|---|---:|---:|---:|---:|---:|']
    for r in failures:
        f=r['earliest_observed_exclusion']
        lines.append(f"|{r['run_index']}|{r['window']}|{r['processed']['grade']:.9f}|"+
            (f"{f['time']:.2f}|{f['seconds_remaining']:.2f}" if f else '未排除|—')+f"|{r['end_fifo']['tons']:.3f}|")
    near=next(r for r in failures if r['run_index']==9970001);v=near['earliest_visible_support_low_exclusion']
    lines += ['',
        '9例在宽误差检验发现证书时，已有FIFO已覆盖全部剩余产能，所有后续新卸载都无法在本窗FIFO顺序中插入。旧9950012/9950029截止FIFO分别留有高品位矿，说明“已经卸下高品位料”仍不等于“已经处理成合格物料”。末窗高品位失败则可能留下低品位尾料；方向相反，兑现顺序问题相同。','',
        f"9970001没有被宽误差检验排除，但既有严格向外取整的float32 FIFO风险检验在{v['time']:.2f}s给出上界 **{v['visible_support_upper']:.9f}<0.26**；此时已有FIFO{v['fifo_tons']:.3f}t大于剩余处理能力{v['remaining_capacity']:.3f}t，不需要任何未来品位来解释该尾段。它仍是失败，不能扩大合规epsilon来改判。",'',
        '进一步按事件序号把987个真实派车决策关联到卸载批次及截止前PROCESS，所有矿石卸载均找到此前该车最近一个同源决策，当前窗/前窗派车处理量之和与实际处理总量闭合。关联仍依赖现有每车决策顺序，日志没有独立行程ID，不能把它扩张为任意改派场景的因果匹配。废石明确列为矿石事件流之外。未来事件只作为兑现结果标签。','',
        '|同种子对照run|首窗结果|派往Load0候选吨位|Load0已处理(t)|Load2已处理(t)|',
        '|---|---|---:|---:|---:|']
    for rid in (9950012,9970012,9950029,9970029):
        r=next(r for r in chain['trajectories'] if r['run_index']==rid)
        lines.append(f"|{rid}|{'合格' if r['compliant'] else '失败'}|{r['per_source']['0']['candidate_tons']:.1f}|{r['per_source']['0']['processed_in_window']:.1f}|{r['per_source']['2']['processed_in_window']:.1f}|")
    lines += ['',
        '9950012与9970012派往高品位Load0的累计候选吨位接近，真正进入首窗处理结果的Load0吨位相差约284t；另一配对中，成功轨迹处理的Load0更少，同时大幅减少了Load2进入首窗的量。因此只比较派车次数、高品位预留总量或卸载混合品位，无法解释窗口结果。这里是轨迹间已实现组成对照，尚不构成单一动作的因果效果；仍需结合逐批品位、金属量与处理时间定位差异来自哪段链路。','',
        '**尚未证明策略漏选。** 证书之前仍需核对具体合法替代动作的运输、排队、装载、卸载和破碎兑现时间。原日志缺LOAD_COMPLETE；观测到的dispatch→unload时长也不是所有替代动作的物理下界。同种子成功对照是机制线索，不是保持同一状态与外生过程的动作反事实。','',
        '来源：[replay.json](replay.json)、[全部决策前缀](decision_prefixes.csv)、[派车—卸载关联](dispatch_unload_joins.csv)、[兑现结果](dispatch_realization.json)、[逐决策兑现](dispatch_realization.csv)。','',
        '## 图与 C128：相抵已跨多个真实基态复现','',
        '从上述轨迹按固定决策序位及证书邻近状态抽取118个真实基态，以原3e-4/u13前向。A适用53、B适用104、C适用118，共275个外生分支，均通过原物理/账本/路径一致性检查。低学习率轨迹基态对这个模型属于离策略输入。A是内部装载完成瞬态；B是环境未暴露给PPO的重定向；C是破碎能力减半，均未被仿真实际执行。','',
        '下表“较强相抵”是探索性描述口径：至少一个动作的图与C128贡献反向，双方各≥0.1个百分点，净变化仅保留双方绝对贡献和的≤50%。分母是适用状态数，各动作输出相互依赖，不当作独立统计样本。向量保留率为四动作净变化L1范数除以两支贡献L1范数之和。','',
        '|干预|适用状态|较强相抵状态|平均向量保留率|',
        '|---|---:|---:|---:|']
    for r in cancellations:lines.append(f"|{r['scene']}|{r['eligible_states']}|{r['strong_cancellation_states']}|{r['mean_vector_L1_retained_fraction']:.1%}|")
    example=next(r for r in examples if r['scene']=='A') if any(r['scene']=='A' for r in examples) else examples[0]
    lines += ['',
        f"例如run {example['run_index']}、决策{example['state_index']}的{example['scene']}分支，对Load{example['action']}：图贡献{100*float(example['graph_shapley']):+.3f}个百分点，C128贡献{100*float(example['c128_shapley']):+.3f}个百分点，正常完整前向净变化{100*float(example['full_delta']):+.3f}个百分点。",'',
        '轨迹内状态相关，各干预的资格集合不同，不作发生率推断或直接比较。C的零计数只表示未达到这组较强相抵门槛；图贡献较小也可能使其无法越过门槛，不能写成C没有相抵或信息利用问题。','',
        f"训练支持扫描覆盖原512班的{inter['training_decisions_scanned']:,}个决策。所检查十组业务量中，只有C的破碎率1250 tph超出已观察边际范围；原训练恒为2500 tph。A/B处于这些边际范围内，不能由此推断联合分布覆盖充分。当前结果支持进一步做匹配覆盖与通道消融，**不能证明图无效，也不能确定相抵中的哪一支更正确**。",'',
        f"进一步检查与干预直接相关的离散联合模式：训练决策中的“已载矿且OTHER阶段”观测为{joint['loaded_ore_other_phase_truck_observations']}，所以A的53个分支都包含未见模式；B的“未载TO_LOAD阶段＋目标Load＋当前segment＋有序剩余route”在57/104分支中未见。真实基态与构造一致性通过，仍不足以保证干预位于训练支持内。其余B分支只说明这个有限投影出现过，不能证明完整联合状态覆盖充分或可达。",'',
        '这使A/B响应更适合用来检查通路与外推边界。下一轮应优先补充决策时刻可达、且有训练覆盖的干预，再用匹配实验判断覆盖不足是否造成实际性能问题。','',
        '来源：[interventions.json](interventions.json)、[Shapley分解](readout_decomposition.csv)、[适用性](intervention_eligibility.csv)、[边际支持](training_support.csv)、[离散联合模式](joint_support.json)、[分支变更与检查](branches.json)。','',
        '## PPO：后期存在较大策略移动，独立影响仍待验证','',
        '原3e-4运行没有保存Minibatch KL/Clip或epoch KL。本轮从原封存前态、原批次、原Adam和私有shuffle RNG重构14、15、16，终点Actor/Critic/两套Adam/目标/RNG均严格匹配原封存下一Checkpoint，更新回执也一致。它们是验收后的**数值重构结果**，并非补写成原运行实时日志。','',
        '|重构更新|Minibatch数|样本加权Minibatch KL|KL P95 / 最大|样本加权Clip|',
        '|---|---:|---:|---:|---:|']
    for r in mini:lines.append(f"|u{r['update']-1}→u{r['update']}|{r['minibatches']}|{r['sample_weighted_KL']:.5f}|{r['minibatch_KL_p95']:.5f} / {r['minibatch_KL_max']:.5f}|{r['sample_weighted_clip_fraction']:.1%}|")
    lines += ['',
        'Minibatch KL为Actor Adam步之前、相对冻结行为动作log-prob的非负采样反向KL近似，Clip是该Minibatch中ratio越过0.2区间的比例。Critic Minibatch EV也在其Adam步之前计算，完整批次EV另在每epoch末固定参数下计算。二者不混作同一指标。三个原批次的MC目标方差分别为0.040769、0.090898、0.153726，EV与MSE需要结合这些不同尺度解释。','',
        '|更新|epoch|完整原批次KL|Clip|Critic EV|Critic MSE|',
        '|---|---:|---:|---:|---:|---:|']
    for r in epochs:lines.append(f"|u{int(r['update'])-1}→u{r['update']}|{r['epoch']}|{float(r['approx_kl']):.5f}|{float(r['clip_fraction']):.1%}|{float(r['explained_variance']):.3f}|{float(r['critic_mc_mse']):.5f}|")
    lines += ['',
        'u15→u16的第2个epoch末KL约0.02891、Clip约38.3%，最终降至0.01063、14.0%。末值会隐藏这类中途移动；这些变化值得结合完整更新路径分析，仍不能单凭它们判定独评退化。','',
        '独立于这些训练批次，固定旧u13独评100班、每窗25%/75%决策序位，共800状态。在相同状态/Mask上计算精确分类KL(π_before || π_after)，对800状态等权平均；评估的是策略移动，不是各Checkpoint的新轨迹表现。','',
        '|状态银行比较|精确KL均值|平均TV|最高概率动作改变|',
        '|---|---:|---:|---:|']
    for r in late['policy_changes']:
        if r['window']=='all':lines.append(f"|u{r['before']}→u{r['after']}|{r['exact_categorical_kl_mean']:.5f}|{r['total_variation_mean']:.1%}|{r['argmax_changed_fraction']:.2%}|")
    lines += ['',
        '固定状态银行上，Critic对同一u13回报标签的EV：u13约0.179、u14约0.207、u15约0.130、u16约0.096；对非u13 Checkpoint，这是离策略拟合诊断。原训练批次的目标方差也随阶段变化，不能仅看MSE或EV归因。','',
        '**独立性能结论仍开放。** u14/u15的策略移动、Minibatch峰值和Clip值得继续检查；但只有u13已有选定100班独评。要判断更新是否真的降低完整任务率或增加兑现失败，仍须为后期固定Checkpoint准备独立身份、合并预算及同种子配对推理。当前不以这次诊断新增班次，也不因KL值自行改学习率、early-stop或PPO数值逻辑。','',
        '来源：[固定状态分析](late_updates.json)、[策略变化](late_policy_changes.csv)、[训练批次](original_training_batches.csv)、[数值重构验收](minibatch_replay_002/complete.json)、[Minibatch明细](minibatch_replay_002/minibatches.csv)、[epoch明细](minibatch_replay_002/epochs.csv)。','',
        '## 后续实施顺序','',
        '1. 优先验证证书之前的具体替代动作及全链兑现时间；先解决回放状态恢复/外生随机量配对，避免把成功对照当作反事实。','2. 用匹配的训练覆盖与图/C128通道消融判断相抵原因；继续保留正常完整前向作为主要结果。','3. 冻结后期Checkpoint的独评方案，以独立结果判断是否需要调整更新机制；本轮保留3e-4。','',
        f"验证：[validation.json](validation.json)，核验{validation['unique_source_files_verified']}个唯一来源文件，4项边界测试通过，3个原更新精确重构；通过验收的内存重构Actor/Critic各{complete['offline_actor_optimizer_steps']}步。首轮误用1线程（原运行2线程）被终点检查拒绝，保留[拒绝记录](minibatch_replay_001/review_resolution.json)；包括该次尝试，Actor/Critic内存优化步各{validation['total_including_rejected_offline_actor_optimizer_steps']}，正式模型/训练账本均未写入。受保护PPO SHA256：`{CORE_SHA}`。",'',
        '本轮脚本、配置与证据文件封印见 [artifact_manifest.json](artifact_manifest.json)。','']
    with (OUT/'README.md').open('x') as f:f.write('\n'.join(lines))
    save('research_findings.json',dict(status='PASS_BOUNDED_OFFLINE_RESEARCH',failed_windows=10,paired_trajectories=20,
        wide_envelope_excluded_failures=9,near_boundary_float32_certificate_run=9970001,
        causal_policy_misses_proven=0,cancellations=cancellations,minibatch_summary=mini,
        unseen_discrete_branch_patterns=joint['unseen_branch_patterns'],
        independent_performance_of_late_checkpoints_established=False,primary_actor_lr=3e-4,
        new_physical_shifts=0,official_training_updates=0,offline_actor_optimizer_steps=complete['offline_actor_optimizer_steps'],
        offline_critic_optimizer_steps=complete['offline_critic_optimizer_steps'],
        total_including_rejected_offline_actor_optimizer_steps=validation['total_including_rejected_offline_actor_optimizer_steps'],
        report_implementation=ref(Path(__file__)),validation=ref(OUT/'validation.json')))
    print(json.dumps(dict(report=str(OUT/'README.md'),minibatch_summary=mini)))

if __name__=='__main__':main()
