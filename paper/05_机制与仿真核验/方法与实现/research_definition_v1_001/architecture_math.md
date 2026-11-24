# BG128 数学架构定义 V1.0

版本锚点为原 Actor 学习率 `3e-4` 的 `s01_BG128`，检查点 `u13`。原始观测 schema 为 `dispatch.physical.v3`。训练与发现批次沿用同一 `business_graph_summary_v1` 模型与学习器。检查点 SHA256 为 `a9acc888413ce0861240c04c396a239d9c26e48290eddc9f2c14dd9c97e6afd5`。 [架构合同](../../../../business_graph_summary_v1/contract.py#L23) · [独评加载入口](../../../../business_graph_summary_evaluation_v1/runtime.py#L64) · [发现批次固定模型与采样](../first_window_action_001/discovery_3e4_001/discovery_common.py#L81)

## 1. 观测与决策

设 $t$ 为 AnyLogic 发出派车请求的序号，$o_t\in\mathbb R^{1519}$ 为当时可见的快照，$m_t=o_t[19:23]$ 为四个装载候选的合法掩码，$d_t=o_t[23:27]$ 为归一化候选派车吨位。中央 Actor 为当前请求的矿卡从四个装载站中选一个合法动作。候选卸载站由装载站的矿废属性确定，作为候选读取的输入。索引采用从零开始的半开区间。 [候选读取](../../../../business_graph_summary_v1/models.py#L178) · [矿废映射](../../../../business_graph_summary_v1/graph.py#L160)

| 观测区间 | 字段 | 形状 |
|---|---|---:|
| `[0:19]` | 全局字段 | 19 |
| `[19:27]` | 合法掩码与候选派车吨位 | 4 + 4 |
| `[27:1107]` | 矿卡各自状态 | 9 × 120 |
| `[1107:1295]` | 装载站 | 4 × 47 |
| `[1295:1391]` | 卸载站 | 2 × 48 |
| `[1391:1439]` | 破碎前 FIFO | 16 × 3 |
| `[1439:1519]` | 有向道路 | 16 × 5 |

状态编码使用 **19 个 64 维 Token**，按 9 辆矿卡、4 个装载站、2 个卸载站、3 个路口和 1 个 Global 排列。道路的局部投影用于车辆路线、位置引用和地理端点特征。图中有 16 条预定义有向道路边。车辆数和道路端点固定，实例化的业务边随观测改变。 [原始观测与 Token](../../../../business_graph_summary_v1/encoder.py#L81) · [节点和道路注册](../../../../business_graph_summary_v1/graph.py#L22)

## 2. 图实例化

定义有向多重图 $G_t=(V,E_t)$。每条实际边 $e$ 包含源 $i$、目的 $j$、关系类型 $r_e$、业务组 $g_e$ 和 14 维属性 $b_e$。四组对应 25 个方向类型，具体计数如下。反向边使用独立的 `_reverse` 类型，复制该实例属性。道路按实际方向登记，两个方向都存在时来自各自的道路记录。 [关系注册表](../../../../business_graph_summary_v1/graph.py#L28) · [边实例化](../../../../business_graph_summary_v1/graph.py#L169)

| 业务组 | 类型数 | 正向语义与实例化条件 |
|---|---:|---|
| `transport` 运输 | 9 | `road` 为端点→端点。`truck_at_node` 为车→当前节点，存在当前路段时优先建 `truck_segment_from`、`truck_segment_to` 两条车→端点边。`truck_predecessor` 为车→观测前车。后四种均配反向类型。 |
| `commitment` 承诺 | 4 | `reserved_load` 为已知装载目标、有正预留量、尚未载货的非 Ego 车→装载站。`loaded_dump` 为已载货且卸载目标可见的车→卸载站。两种均配反向类型。 |
| `service` 服务 | 10 | `queue`、`seize`、`capacity_wait`、`service`、`admission` 为站→车，均配反向类型。前三种列表记录槽位顺序，其中 `capacity_wait` 只读卸载站。`service` 使用保留重复实例的无序多重集。`admission` 来自卸载站当前接纳位。 |
| `material` 物料兼容 | 2 | `compatible_material` 为装载站→对应的矿石或废石卸载站，并配反向类型。它表示固定的物料兼容性。某候选是否合法、执行后是否形成承诺，由掩码或下一状态另行体现。 |

边属性按实现顺序为道路长度/10000、限速/10、封闭状态、观测路段占用车数/9、道路有效位、当前路段剩余距离/10000、剩余距离有效位、吨位/193、吨位有效位、矿石标志、载荷品位、品位已知位、服务列表槽位/9、槽位有效位。各关系仅填写可观测且适用的字段。道路占用数来自有当前路段引用的车。GraphBuilder 另外生成每节点的 $q_j\in\mathbb R^{21}$，包括预留、在途已载、已知品位质量与金属量、排队与服务计数以及道路与车辆计数。吨位和金属吨按 1737 归一化，入出道路数按 16，其他计数按 9。预留和已载统计逐车计入一次，服务记录保留唯一车数与实例数。Global 无物理边。 [属性与尺度](../../../../business_graph_summary_v1/graph.py#L38) · [事实图与数量构造](../../../../business_graph_summary_v1/graph.py#L179)

## 3. 初始 Token

令 $P_x(u)=\operatorname{ReLU}(W_xu+b_x)$。对物理节点 $j$，初始化为

$$
h_j^0=\operatorname{LN}_{\tau(j)}\!\left(P_{\kappa(j)}(x_j)+P_q(q_j)+e_{\tau(j)}\right).
$$

其中 $\tau$ 标识 Truck、Load、Dump、Junction、Global 五个类型嵌入，$\kappa$ 标识最终投影。Load 和 Dump 共用 `site_final` 投影和站点 LayerNorm。Global 为 $\operatorname{LN}_{G}(P_G(o_t[0:19])+e_G)$。原始数值、有效位和有序引用进入各类型固定的 $x_j$。引用由目标基础内容、目标类型内身份 one-hot 及存在位组成。缺失引用整组为零。 [引用与投影](../../../../business_graph_summary_v1/encoder.py#L24) · [归一化与拼接](../../../../business_graph_summary_v1/encoder.py#L197)

| 路径 | 投影维数与输入 |
|---|---|
| 基础内容 | Truck `72→16` 取门控后的局部字段。Site `64→16` 取站点类型、局部字段及字段有效布局。Road `4→16` 取长度、限速、封闭状态和观测占用。地理端点 `54→16` 汇聚入路、出路、到达车辆三个 16 维均值，以及归一化计数与存在位。 |
| Truck `223→64` | 局部 72 维、9 维车辆身份、装载站/卸载站/地理节点/当前路段/前车引用，以及按路线次序排列的道路编码及存在位。路线投影 `544→16`。 |
| Site `176→64` | 站点基础内容、身份、排队/占用资源/容量等待序列、无序服务聚合、接纳引用、地理端点及 FIFO 内容。三类有序车列表共用 `234→16` 投影，服务集合 `26→16`，FIFO `48→32` 只附于矿石卸载站。 |
| Junction `26→64`，Global `19→64` | 路口使用地理端点引用，全局使用原始全局字段。 |

数量投影为 `21→64`。品位按 grade-known 有效位门控，车辆运动历史按有效位/前车存在位门控。站点有序列表保留位置，服务聚合保留数量与身份构成。 [基础内容与门控](../../../../business_graph_summary_v1/encoder.py#L59) · [车辆、站点与 FIFO 路径](../../../../business_graph_summary_v1/encoder.py#L137)

## 4. 组内关系注意力

两层同步更新，层间参数独立。每层四个头，每头 16 维。记 $x_e=[\operatorname{onehot}_{25}(r_e);b_e]\in\mathbb R^{39}$，$[\cdot]_k$ 为第 $k$ 个 16 维切片，$L$ 为负斜率 0.2 的 LeakyReLU。第 $\ell$ 层的参数下标省略。对实例 $e:i\to j$，每头的分数与消息为

$$
s_{e,k}=a_k^\top\!\left[L(W_s h_i^\ell+W_d h_j^\ell+W_e x_e)\right]_k,
\qquad
v_{e,k}=\left[L(W_v h_i^\ell+b_v+W_{ev}x_e)\right]_k.
$$

分数投影和 $W_{ev}$ 无偏置。关系 one-hot 与连续属性共同进入打分和值分支，关系间共用这些矩阵。设 $E_{jg}$ 为同一目的节点 $j$ 的业务组 $g$ 入边，四个头分别在该集合内归一化。若只有一条入边，其注意力权重为 1，连续属性仍通过 $v_{e,k}$ 影响传递值。

$$
\alpha_{e,k}=\frac{\exp(s_{e,k})}{\sum_{e'\in E_{jg}}\exp(s_{e',k})},
\qquad
u_{jg}=\mathop{\Vert}_{k=1}^{4}\sum_{e\in E_{jg}}\alpha_{e,k}v_{e,k},
\qquad p_{jg}=\mathbf 1[E_{jg}\ne\varnothing].
$$

空组 $u_{jg}=0$。分母包含组内所有方向关系实例，同组内不同关系类型可竞争。按 `transport, commitment, service, material` 顺序拼接四个 64 维组向量，然后拼接四个存在位。对物理节点及 Global 分别有

$$
h_j^{\ell+1}=\operatorname{LN}\!\left(h_j^\ell+W_o[u_{j1};u_{j2};u_{j3};u_{j4};p_{j1};p_{j2};p_{j3};p_{j4}]\right),
\qquad h_G^{\ell+1}=h_G^\ell.
$$

$W_o$ 为无偏置 `260→64`，LayerNorm 的 $\epsilon=10^{-5}$。同一次前向的两层复用该次实例图，残差保留本节点状态。 [逐层参数与计算](../../../../business_graph_summary_v1/graph_layers.py#L34) · [两层调用](../../../../business_graph_summary_v1/models.py#L78)

## 5. Actor 与 Critic 读取

两分支均按 Truck、Load、Dump、Junction 对最终 $h^2$ 作四组均值池化，并对原始 $q$ 作四组求和池化。拼接神经均值池化、Ego、Global 和数量和池化得到 $z_\mathrm{graph}\in\mathbb R^{468}$，维数为 $4\times64+64+64+4\times21$。这里的读取函数结构相同，参数分开。 [读取维数与索引](../../../../business_graph_summary_v1/models.py#L99)

C128 为 $C_t=\phi(o_t)$，由同一时刻的观测确定性派生。它含已兑现进度、料仓 FIFO、已载/预留承诺、当前和下一窗的名义处理投影、缺失与不确定性，以及候选装载的名义时间、吨位和品位字段。已入仓 FIFO 与预测到达车辆的含义不同。实际未来破碎、后续交通和后续派车不进入 $C_t$。封存的原始观测保留原字节，C128 私有投影在校验后处理有效但零吨的 FIFO 项。 [C128 物理语义](../../../../grade_commitment_ppo_v1/features.py#L1) · [C128 字段](../../../../grade_commitment_ppo_v1/features.py#L88) · [当前态证书](../../../../single_critic_task_query_v1/c_features.py#L74)

**Actor。** $C_t$ 拆为 88 维任务块和四个 10 维动作块，分别使用 `88→32` 和跨候选共享的 `10→16` ReLU 投影。任务投影与 $z_\mathrm{graph}$ 拼接，经 `500→128→64` MLP 得 $u$。候选 $a$ 的输入依次是 Ego/Load/Dump 的三个 64 维 Token、三者各 21 维数量、合法标志、候选派车吨位和 16 维候选摘要，经共享 `273→128→64` MLP 得 $v_a$。评分为

$$
\ell_a=\operatorname{MLP}_{192\to128\to1}([v_a;u;v_a\odot u])+
\frac{\sum_i v_{a,i}u_i}{8}.
$$

这些两层 MLP 使用 ReLU 隐层和线性末层。非法动作 logit 为 $-\infty$，合法动作构成分类分布。原主基线的线上策略用私有 RNG 从分布采样，并保存所选动作的旧 log-prob。 [Actor 结构与掩码](../../../../business_graph_summary_v1/models.py#L151) · [原训练采样入口](../../../../business_graph_summary_recovery_v1/collection.py#L75)

**Critic。** 使用独立编码器计算其 $z_\mathrm{graph}$。完整 C128 经 `128→32` ReLU 投影后与该向量拼接，经 `500→128→1` MLP 输出一个联合剩余成本估计 $V_\psi$。真实班末的输出严格为零。 [Critic 结构](../../../../business_graph_summary_v1/models.py#L209)

| 可训练参数 | Actor | Critic |
|---|---:|---:|
| Token 投影及类型嵌入 | 48,560 | 48,560 |
| 两层关系注意力，各 34,176 | 68,352 | 68,352 |
| C128 投影 | 3,024 | 4,128 |
| 场景上下文 MLP | 72,384 | — |
| 候选 MLP 和共享评分 MLP | 68,161 | — |
| 标量价值融合 MLP | — | 64,257 |
| 合计 | **260,481** | **185,297** |

总计 445,778。单侧图编码器 116,912。Actor 的跨候选权重共享仅发生在 Actor 内部。两分支参数存储及 Adam 状态各自独立，使用相同观测及相同确定性构图规则。 [模块构造](../../../../business_graph_summary_v1/models.py#L138) · [独立优化器](../../../../business_graph_summary_v1/learner.py#L257) · [参数量与离线验收](../../../../reports/token_attention_001/business_graph_summary_009/README.md)

## 6. 物料结算与 PPO 梯度路径

在 28,800 秒班次内设四个连续 7,200 秒窗口。矿石卸入 FIFO，实际破碎事件消耗队首批次。窗口 $w$ 的 $Q_w$ 和 $M_w$ 分别累计实际处理矿石和金属吨，$g_w=M_w/Q_w$ 在有有效品位时定义。矿石班次产量 $Q_o$ 按实际破碎量，废石班次产量 $Q_b$ 按实际卸载量。原版学习器最小化固定权重的标量联合成本。 [FIFO 与结算核验](../../../../grade_event_memory_v1/events.py#L271) · [班末产量来源](../../../../business_graph_summary_recovery_v1/collection.py#L275)

$$
c_g(w)=\begin{cases}
\min\bigl(\max(|g_w-0.27|-0.004,0)/0.006,\;2\bigr), & g_w\text{ 有效},\\
2, & \text{空窗或品位无效}.
\end{cases}
$$
$$
c_t=\sum_{w\text{ 在该转移结算}}c_g(w)
+\mathbf 1_{\mathrm{true\ end}}\!\left(8[1-Q_o/16000]_+ +8[1-Q_b/4000]_+\right)
+0.2\,\Delta W_t/28800.
$$

$W$ 为车队累计等待车辆秒。每个窗口的成本按规范物料事件序号计入相应转移一次，结算序号严格位于该转移起止观测标记之间。班末产量成本只结算一次。原始 RPC 回报参与一致性核对，随后由物料事件与运行统计构造上述训练成本。6000 吨是废石停派阈值，废石达产下限是 4000 吨。 [成本定义与结算](../../../../single_critic_task_query_v1/targets.py#L35)

每次更新用冻结的旧策略 log-prob 和 `no_grad` 算出的旧 $V$。设 $T$ 为真实终止，$V^{\mathrm{old}}_T=A_T=0$。更新以完整班为轨迹，$\gamma=1$，$\lambda=0.95$，窗口结算后 GAE 继续递推。批内 Actor 优势用总体标准差加 $10^{-8}$ 标准化。

$$
\delta_t=c_t+V^{\mathrm{old}}_{t+1}-V^{\mathrm{old}}_t,\qquad
A_t=\delta_t+0.95A_{t+1},\qquad G_t^{\mathrm{MC}}=\sum_{i=t}^{T-1}c_i.
$$
$$
r_t=\exp\!\left(\log\pi_\theta(a_t\mid o_t)-\log\pi_{\mathrm{old}}(a_t\mid o_t)\right),\qquad
\mathcal L_\pi=\operatorname{mean}_t\!\left[\max\!\left(r_t\widehat A_t,\operatorname{clip}(r_t,0.8,1.2)\widehat A_t\right)-0.01\,\mathcal H(\pi_\theta(\cdot\mid o_t))\right],
$$
$$
\mathcal L_V=\operatorname{mean}_t\!\left(V_\psi(o_t)-G_t^{\mathrm{MC}}\right)^2.
$$

`max` 对应成本最小化。Critic 回归完整 Monte Carlo 剩余成本。旧值和优势在两侧更新前冻结，Actor 梯度沿当前 log-prob 与熵经过评分、C 投影、图层、Token 投影，Critic 梯度沿当前值误差经过自己的融合、C 投影、图层、Token 投影。每个逻辑 minibatch 按轨迹计算分组梯度、用同一 minibatch 样本数归一化，各分支梯度范数裁剪到 0.5 后独立做 Adam，顺序 Actor 后 Critic。原版每次更新 32 个完整班，4 个 shuffle epoch，minibatch 32 个决策，Actor/Critic LR 分别为 `3e-4` 和 `1e-3`，Adam `eps=1e-5`。 [GAE 与 MC](../../../../grade_event_memory_v1/targets.py#L125) · [冻结超参数](../../../../business_graph_summary_v1/learner.py#L131) · [损失与更新](../../../../business_graph_summary_v1/learner.py#L375)

运行时仿真、动作采样、离散边索引与类型选择、C128 工程特征计算、结算与旧目标处于训练的可微边界之外。图中部分连续属性和数量从原张量构建，代码保留这些张量的数值梯度路径。实际优化变量为各分支的网络参数。 [连续边属性与数量](../../../../business_graph_summary_v1/graph.py#L143) · [C128 构建](../../../../single_critic_task_query_v1/c_features.py#L92) · [分支梯度测试](../../../../business_graph_summary_v1/test_models.py#L203)
