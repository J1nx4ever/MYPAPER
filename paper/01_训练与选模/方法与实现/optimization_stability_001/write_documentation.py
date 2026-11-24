"""Render handoff from actual frozen bytes, not copied historical identities."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))


def main():
    from business_graph_stability_campaign_v1.plan_io import PLAN,REPORT,reference,load_suite
    suite_path=PLAN/'prepared_001/suite.json'
    suite,_=load_suite(suite_path)
    identity=json.loads((PLAN/'prepared_001/launch_identity.json').read_bytes())
    first=identity['configurations'][0];job=suite['jobs'][0]
    pending=json.loads((PLAN/'paired_evaluation_001/seed_plan.json').read_bytes())
    source=suite['pairing']['common_initial_checkpoint']
    rows=[('campaign',suite['campaign_id']),('suite字节SHA256',reference(suite_path)['sha256']),
        ('首批recording_id',first['recording_id']),('首批配置字节SHA256',first['config']['sha256']),
        ('训练run_index','9960001—9960512'),('共同初始化seed',job['actor_seed']),
        ('首班环境seed',job['experiment_seed']),('首班Actor采样seed',job['actor_sampling_seed']),
        ('Python输出根目录',suite['python_output']),('首批Python输出',first['python_output']),
        ('首批AnyLogic物料事件目录',first['anylogic_event_directory']),
        ('原始共同u0文件',source['path']),('原始共同u0字节SHA256',source['sha256'])]
    table='|字段|本轮冻结值|\n|---|---|\n'+'\n'.join(f'|{k}|`{v}`|' for k,v in rows)+'\n'
    manual=f'''# 同Checkpoint、同种子、低Actor学习率：手工启动卡

2026-11-18，Asia/Shanghai。当前仅准备，未启动AnyLogic/RPC、未登记预算。用户手动应用以下内容，不直接编辑ALP。完整方案见[README](README.md)。

## 本次修改

从原始共同u0实际Checkpoint恢复Actor、Critic和随机状态，确认训练计数0、两个Adam状态为空。512班训练逐班复用原512班的环境/采样seed；之后新u13的100班推理逐班复用原100班seed。所有recording、run_index和目录为新身份。唯一训练超参数变化是Actor LR `3e-4 → 1e-4`。

## AnyLogic精确操作

对象是你当前运行的模型；不要求重新上传或离线审计ALP。常见模型位置为`anylogic/3d/DRL-PPO-baseline1/DRL-PPO-baseline1.alp`，以当前打开的运行模型为准。

|精确位置|修改前|修改后|原因、影响及验证|
|---|---|---|---|
|Additional Class → `Baseline1CampaignPlan` → Java全文|当前已结束实验的helper；若仍为本次100班独评，则为[旧全文](../checkpoint_evaluation_001/prepared_001/Baseline1CampaignPlan.java)|全文替换为[本轮helper](prepared_001/Baseline1CampaignPlan.java)|切换到新训练控制器、master和配置SHA。`jobCount()`/`frozenPlanJobCount()`=512，`resumeOffset()`=0；`runIndex(0)`=9960001、`runIndex(511)`=9960512。不能只改班数沿用旧helper。|
|`RL_Training → Properties → Number of runs`|上次独评为100；若仍显示其他旧班数，以当前值为准|`512`|本次Run只覆盖512训练班；本阶段没有附加冒烟。|
|`RL_Training → Parameters → Freeform`|现有`Baseline1CampaignPlan.*((int) index)`调用|保持现有调用，不加offset|helper自动分配本轮身份，环境及动作seed与旧表逐槽对应。|
|`Main → On startup`|已有`Baseline1CampaignPlan.awaitConfigSha256(...)`及后续RPC/物料初始化|保持现有代码|首次Main启动本轮固定Python桥接；RPC实际配置、schema、物料证据必须核对。|
|`RL_Training`串行/并行设置|既有串行；不勾选并行评估、使用重复|保持|保证对应种子、班次和更新顺序。|
|`Main`生产/品位/物理参数|16000t矿石、4000t废石目标；6000t停派、6193t上界及现有场景|保持|本轮不改变业务条件或奖励。Actor LR只在已冻结Python learner配置中修改。|

Save → Build → 核对下表 → `RL_Training → Run`一次。helper启动唯一Python控制器；不需要另开Python。当前代理没有预先启动控制器。出现STOP、UNKNOWN或未完成claim时保留证据定位，不重复Run补班。

## 启动前独立身份

{table}

[16批完整身份和配置SHA](prepared_001/launch_identity.json) · [512班新run/seed清单](prepared_001/identities.csv) · [612班新旧逐项配对清单](paired_seed_mapping.csv)。首批配置SHA是实际文件字节SHA，与suite SHA不同。高频transition/RPC/rollout记录关闭；每次更新仅新增4份epoch汇总。

## 推理阶段

本次训练结束后固定新u13，保持随机采样规则，独立推理100班、零更新。推理身份已冻结为`{pending['recording_id']}`，run_index为9970001—9970100，Python输出为`{pending['python_output']}`，AnyLogic物料事件目录为`{pending['anylogic_event_directory']}`。全部seed见[预备清单](paired_evaluation_001/seed_plan.json)。

新u13尚未产出，当前`checkpoint_sha256`和`final_config_sha256`为null；该预备清单不能启动。训练完整结束后，执行下列准备命令，验证新u13的实际sealed字节并生成最终配置、SHA、100班helper及身份；再告知最终SHA，手工换该helper、Number of runs=100，Run一次。不能用旧u13或根据新结果另选checkpoint。

```bash
cd /home/jinx/Faith/experiments/lagrangian_ppo_v1
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B change_plans/token_attention_001/business_graph_summary_009/optimization_stability_001/paired_evaluation_001/prepare_evaluation.py
```

此命令只准备，不登记预算或启动监听。100班推理最终配置在实际u13出现前无法诚实计算SHA；当前不填造。后续推理启动继续走实际RPC和全部通信一致性检查。

## 完成与验证

训练应为512 COMPLETE、UNKNOWN0、16次COMMITTED及`BUSINESS_GRAPH_STABILITY_TRAINING_COMPLETE_SINGLE_SEED`；训练u16归档，推理候选预注册为u13。低频进度见输出根目录`training_progress.json`，每epoch诊断见`arms/s01_BG128/updates/XXXX/training_summary.json`的`update.epoch_diagnostics`。训练成本不能代替100班固定Checkpoint推理表现。

```bash
cd /home/jinx/Faith/experiments/lagrangian_ppo_v1
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B -m business_graph_stability_campaign_v1.launch status --suite change_plans/token_attention_001/business_graph_summary_009/optimization_stability_001/prepared_001/suite.json
```
'''
    (PLAN/'manual_anylogic.md').write_text(manual)
    readme=f'''# BG128 优化稳定性：共同u0与相同种子的受控重跑

2026-11-18，Asia/Shanghai。按用户最新指令执行最新优先级第一项，并采纳“相同种子训练和推理、从原始共同初始化Checkpoint重新开始”的修订。**只把Actor学习率从3e-4降至1e-4。** 已完成离线工程与启动准备，真实训练/推理均未启动。

当前操作入口：[AnyLogic手工卡](manual_anylogic.md)；完整身份见[16批启动身份](prepared_001/launch_identity.json)，612个新旧run/seed对应关系见[配对清单](paired_seed_mapping.csv)。原始u0的实际导入和状态一致性见[检查证据](../../../../reports/token_attention_001/business_graph_summary_009/optimization_stability_001/common_checkpoint_verification.json)。

## 固定比较条件

|项目|上次BG128|本次|
|---|---|---|
|起点|原共同u0 `initial.pt`|同一封存文件，Actor/Critic张量、shuffle/采样随机状态逐位导入；计数0，Adam为空|
|Actor学习率|3e-4|1e-4|
|Critic学习率、epoch、minibatch|1e-3、4、32|相同|
|gamma/lambda、clip、entropy、梯度裁剪|1/.95、.2、.01、.5|相同|
|表征与模型|BG128两层GAT、四关系组、C128、独立Actor/单Critic|相同，复用封存模型及目标函数|
|奖励|四窗deadband.004、scale.006、cap2、空窗2；矿/废短缺8/8；等待.2×W/28800|相同；MC和GAE同一联合成本|
|训练|16×32=512班|相同；每班环境/采样seed按旧表对应，新物理run|
|独评|u13、随机采样、100班、零更新|固定新u13，相同100组环境/采样seed，零更新|
|高频日志|关闭|关闭；新增每epoch KL/clip/梯度分布汇总|

该起点不是仅凭同seed重新随机初始化，也不是从训练后u13续训。新命名空间用于保护旧Checkpoint和source pins；导入u0时只允许配置命名空间和Actor LR不同，其他配置不符即拒绝。训练使用与上次实际运行一致的BG128修复后存储层；不重引入早期旧TQ存储兼容问题。

## 规则、预算和解释

[预注册协议](protocol.json)固定16次更新；原有数值、通信/schema、行为/物料完整性错误触发STOP。不增加KL early-stop、自适应LR、回滚或业务权重调节，避免混入第二项优化干预。每epoch报告的KL和clip来自更新后完整batch重放；梯度分布和裁剪命中来自实际minibatch裁剪前范数，二者分母不同。诊断已证明不改变数值更新、Adam和RNG。

用户明确授权本次对应seed复用，例外仅绑定本campaign、原512/100班来源SHA和逐槽清单；乱序、交换seed类型、第三次重跑或旧run/recording复用均拒绝。所有512训练班与100推理班仍是612个新增物理班次，不能因同seed抵扣预算；旧H2预留保持0，UNKNOWN/历史记录保留。

新训练预算扩展使用`business_graph_stability_campaign_v1.budget.budget_snapshot`，合并最新BG128独评及所有早期账本。推理扩展后完整入口为`business_graph_stability_evaluation_v1.budget.budget_snapshot`。旧独评JSON和旧SQLite均保持原字节。准备包训练登记上限为{suite['maximum_total_if_approved']}，100班推理按届时完整账本另登记；当前二者均未登记。

Checkpoint选择提前固定为新u13，u16只归档及优化诊断，不看新结果挑最佳。推理的100班种子/身份已冻结；新u13尚未产生，最终Checkpoint/配置SHA保留null，待训练完成后一次性绑定。详见[推理预备计划](paired_evaluation_001/seed_plan.json)。两个阶段沿用已授权范围，具体身份和最终配置SHA须在各自启动前告知。

同起点、同种子加强单因素配对可比性；策略分叉后动作和仿真随机调用顺序可能不同，不能声称经历相同轨迹或证明多训练种子泛化。最终报告应逐对比较完整任务、四窗品位、生产、等待、成本及MC残差，保留失败班次，并区分优化稳定性与业务性能。

## 物料兑现主线

用户提出的核心任务已经写入[截止前物料兑现诊断合同](physical_realization_contract.md)：跟踪承诺→运输/服务→卸载→FIFO→实际破碎/交付的时间和Q/M，关注当前决策在截止前真正能兑现的合规产出。现有C128已含FIFO前缀和名义投影；后续第二项核验其利用情况，不能把本轮低学习率当作已修复物理表征或已消除首窗失败。

## 工程结果

[55项离线检查与证据](../../../../reports/token_attention_001/business_graph_summary_009/optimization_stability_001/README.md)覆盖数值等价、实际共同u0、配对例外、完整预算、BG128真实存储及独评零更新守护。启动仍须真实RPC与首32班端到端验收；离线通过不等于已训练或已证明收益。受保护PPO核心SHA仍为`8e4cfa31c622f5e33756e304eea3f7526547db72ae93f7d9f00808845d61ef96`。
'''
    (PLAN/'README.md').write_text(readme)
    report='''# BG128 相同起点/种子、低学习率：准备证据

55项离线检查通过。真实原始共同u0导入后，Actor/Critic张量、两个私有RNG逐位相同，两个Adam状态为空，训练计数0；仅Actor LR为1e-4。与原学习器在相同LR下比较，实际更新、Adam及RNG逐位一致，epoch诊断为只读观察。

- [测试输出](paired_pytest.txt)和[命令](paired_test_command.json)
- [实际共同u0核对](common_checkpoint_verification.json)
- [原源码、ALP及历史预算保全](preservation_verification.json)
- [离线通信/数据门禁](runtime_acceptance.json)
- [512行Java编译/访问器、源及配置SHA预检](preparation_verification.json)

新训练配置已冻结；100班推理仅冻结配对seed与身份，新u13的Checkpoint/最终配置SHA尚不可得。当前零新增物理班、未启动RPC/AnyLogic、未登记预算。合成测试中的优化器调用仅用于数值验收，不是历史数据训练或真实实验更新。

当前方案与手工操作见[计划入口](../../../../change_plans/token_attention_001/business_graph_summary_009/optimization_stability_001/README.md)。已有第一次fresh-seed工程检查输出pytest.txt继续保留为历史中间证据；当前以paired_pytest.txt及共同u0/配对协议为准。
'''
    (REPORT/'README.md').write_text(report)
    print(json.dumps(dict(suite=reference(suite_path),first_config=first['config'],recording_id=first['recording_id'],
        run_indices=[suite['jobs'][0]['run_index'],suite['jobs'][-1]['run_index']],
        seed=job['actor_seed'],python_output=first['python_output'],anylogic_event_directory=first['anylogic_event_directory'])))


if __name__=='__main__':main()
