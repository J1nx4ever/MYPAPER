# BG128 编码器快照独立核验

状态：**PASS**；146 项检查；固定初始化 seed=2026111801。

未加载已训练 BG checkpoint。只执行无梯度前向和内部张量替换，无优化器、参数更新或仿真。

## 核验内容

- 保存 H0/H1/H2 与正式 `GraphStateEncoder.scene` 逐元素一致。
- 正式 diagnostic 与普通 scene 一致；两层 Global 均保持旁路。
- 每条边的加权 value 经对应关系组输出块投影，逐目标求和后加 presence 独立项，恢复 LayerNorm 前 projected。
- 三因素 Shapley 由独立八格混合重算，向量和恢复真实 T1−T0。
- 两种角色全部参数和 buffer 前后摘要一致。

## 路径补丁边界

`Dump0_H1_T0_into_T1` 只把 Dump0 的第1层表示替换为 T0；第二层图、其余节点 H1 与候选 Load 的目标 H1 固定为 T1。
`Truck*_H0_via_Dump0_H1` 先在 G_T1 上替换所列 Truck 的 H0，再只将所得 Dump0 H1 放入第二层；不改变新 loaded_dump 边或其他节点 H1。
结果保存在 `path_interventions.csv/json/npz`。这是初始化网络内部通道测试，不是可实现原始状态干预，也不是业务因果效应。

## 解释边界

- Attention alpha is only a normalized score, not output contribution.
- Additive projected edge vectors are exact before LayerNorm; norms are not additive and LayerNorm is nonlinear.
- H0 already contains referenced site/truck/route/quantity information; two graph layers add two relation-composition steps.
- Own H0 attribution is not purely raw local-state attribution because H0 embeds referenced neighbor content and deterministic quantities.
- The G factor covers edge topology, relation labels and observed edge attributes jointly.
- Shapley vectors close additively; their norms do not. Signed projections may be negative or exceed one.
- Random initialization diagnostics establish numerical channels and implementation behavior, not learned scheduling use.
- C128 enters final Actor/Critic readouts, not this graph encoder H0/H1/H2.
