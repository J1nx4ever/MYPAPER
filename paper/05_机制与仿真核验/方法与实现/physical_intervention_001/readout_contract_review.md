# 原始观测干预的 C128 与正式 Actor 接口

这是只读接口与物理口径审查。正式源码、历史观测、ALP 均未修改。复现脚本 `readout_contract_probe.py` 仅建立固定初始化 Actor，在内存副本上完成当前状态读取、C128 重算、正式 Actor Forward 和证书拒绝检查；无优化、仿真或 RPC。

## 必须按修改后的原始观测重建证书

```python
import numpy as np
import torch
from business_graph_summary_v1.models import Actor, pack_summary_prefixes

actor = Actor(pair_seed=2026111801).eval()
modified = np.asarray(modified_raw, dtype=np.float32).copy()  # [1519]
prefix = pack_summary_prefixes(modified[None], feature_config=actor.feature_config)
with torch.no_grad():
    logits, details = actor(
        torch.tensor(modified[None].copy(), dtype=torch.float32),
        prefix, diagnostic=True)
```

`pack_summary_prefixes` 的证书绑定原始状态字节、C128 字节、查询索引与特征配置 SHA。不得以旧 prefix 配合修改后的 raw；实测会报 `numerical-cache current raw observation differs`。不能只传 ndarray 摘要代替 `SummarySequence`。一次只读当前快照即可，C128 不需要历史序列或未来状态。

BG009 的 `feature_config_for('s01_BG128')` 明确启用 C128；008 的 `build_features(raw)` 默认配置却是 disabled，会返回零向量。因此手工核对 C 时必须显式传 `actor.feature_config` 或 BG 的 `feature_config_for()`。

## 273D 候选到 4 个分数已接通

|中间量|形状|来源|
|---|---|---|
|current_summary|1×128|修改后 raw 重算的认证摘要|
|task_summary|1×32|C128 前88D|
|action_summary|1×4×16|每动作后10D，各自共享投影|
|context_input|1×500|四类实体均值256＋Ego64＋Global64＋四类数量和84＋task C32|
|context|1×64|公共上下文|
|candidate_input|1×4×273|Ego/Load/Dump 各64，共192；三者数量各21，共63；mask1＋tons1＋action C16|
|candidates|1×4×64|共享候选编码器|
|score_input|1×4×192|候选64＋上下文64＋逐元素乘积64|
|unmasked_logits|1×4|共享评分器＋乘积 dot/8|
|masked_logits|1×4|原始动作 mask 硬约束|

`Actor.forward(..., diagnostic=True)` 返回的 logits 与 `details['masked_logits']` 逐元素相等。正式 candidate/readout 路径无需新增代码或额外 Query。

## 最小的 C 场景：同一时刻正值破碎速率阶跃

建议：以历史 T0 原始观测作底，单独令 Dump0 `crusherRate` 的原始索引1304从1改为0.5，对应2500→1250吨/小时。它是**同一时刻新增的外生处理能力设定**，用于敏感性测试，不是历史中真实发生的变化，也不是把整段历史假设为1250吨/小时后得到的反事实轨迹。

只改这个字段时，累计已处理量、FIFO 内容、库存、预留、现有服务/等待列表、道路、车辆当前状态与历史即时 reward 均保持不变。它们没有在同一瞬间经历新的时间积分。global19 和动作 mask/tons 的观测公式也不直接依赖该速率，故无需虚构同步变化。

依据：

- 当前本地 ALP 中，DumpSite `crusherRateTph` 是可变 public double（`SaveInSnapshot=true, Constant=false`，约19382行）。
- 初始化要求接受矿石的卸点 `crusherRateTph > 0`、缓冲容量至少193吨（约19807行）。
- `advanceCrusherTo` 以 `crusherRateTph/3600` 积分，要求严格正有限值（约20026行）。因此首版不用0表示停机。
- `appendDispatchObservation` 将 `acceptsOre ? crusherRateTph/2500 : 0` 写入 `dump.0.crusherRate`（约20288行），与冻结 observation_schema 索引1304完全一致。
- 全局19字段为时钟、当前/累计实绩、目标和开关；action mask/tons 来自 Load 的 `getDispatchableTons(ego)`，没有额外的破碎速率副本需要覆盖。

本地 ALP 仅提供这些字段公式和状态规则的源码证据，不能据此宣称它就是历史采样时完整运行身份的逐字节 ALP。历史原始运行配置仍为2500吨/小时，本离线分支必须显式记录覆盖值。它不经过正式 RPC/runtime-scene 验收，也不能写回旧运行配置。

### C128 与 ETA 同步规则

必须由正式摘要函数重新生成完整128维，不手改派生项。在本次 T0 探针上，实际有7项改变：

- `crusher.rate`
- `current.capacity`
- `current.free_capacity`
- `current.required_grade_lower`
- `current.required_grade_upper`
- `next.capacity`
- `next.free_capacity`

四个动作10D块本次完全相同。原因是冻结 `_action_eta` 使用道路名义时间、装载队列/服务均值及卸载队列/服务均值，不读取 crusherRate；不能为了“同步”而私自修改这项定义。

已有载货/预留车辆的 `_arrival` 则包含按当前速率计算的 buffer delay，窗口 `_project` 也使用该速率，均会在完整 C128 重算时自动处理。摘要中的 current/next 是基于当前观测与固定假设推导的名义界限，不是读取未来真实结果。

### 不建议作为最小 C 的修改

- 只改 `freeBufferTons`：必须满足 `free=capacity-stored-reserved`，不能把独立数值改成互相矛盾的库存。
- 直接改 `bufferTons`：须同步FIFO吨位/品位、有质量来源与累计收料/处理关系，单一字段不足。
- 改 `occupied/dumping`：须与服务列表、资源占用、车辆阶段一致。
- 改 `available`：是站点可用标志，不等于处理速率；改为不可用而保留业务调度语义，需要另行明确实际停用合同。
- 改 `acceptsOre` 或把矿车目标改为 Dump1：违反 BG009 冻结矿石/废石角色与物料兼容校验，不是有效参数扰动。

## 结果解释边界

改变破碎速率可同时经过三条路径进入最终分数：Dump 自身表示经业务图进入 Load、候选直接读取 Dump H2、C128 task_summary 进入 context。原始观测干预只给出总数值响应，不能独占归因为两跳传播；分路解释需要另做明确标记的内部张量补丁。

固定初始化下的 logit/概率变化只说明数值依赖通道存在，不代表学到了业务偏好、正确选择或因果收益。探针的全部 shape、证书、C128差异、参数摘要与历史/ALP/PPO SHA 前后检查保存在 `reports/token_attention_001/business_graph_summary_009/physical_intervention_001/readout_contract_review.json`。
