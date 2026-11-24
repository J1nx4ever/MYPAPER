# BG128 结构消融（structural_ablation_001）

2026-11-20，Asia/Shanghai。**状态：方案（未启动）**。本目录只放设计、预注册判定规则与身份/预算分配；不写 AnyLogic，不登记预算，不启动训练。

## 1. 要回答的三个归因问题

| # | 问题 | 若答案为"无贡献"对论文的影响 |
|---|---|---|
| Q1 | 两层关系图卷积（GAT）在 C128 + 站点锚定之上是否还有边际贡献？ | 架构叙述必须从"图强化学习"降级为"C128 + 结构化 MLP"，方法学卖点从图转到训练机制 |
| Q2 | `semantic_type_embedding`（truck/load/dump/junction/global 5 类锚定）是否是策略区分装载点与卸载点的必要条件？ | 说明"站点身份"可由 C128 的候选通道隐式恢复，架构可简化 |
| Q3 | 关系分组 softmax（transport/commitment/service/material 四组内归一化再掩码拼接）是否优于全局 attention？ | 四组业务语义的物理先验未被利用，架构 justification 需重写 |

**消融方向统一为"从 Full 往下拆"（single-factor removal）**，不做加法式拼装，避免与已冻结 BG128 的证据链冲突。

## 2. 臂定义

| 臂 | candidate_id | 唯一的改动（相对 s01_BG128） | 设计字段变化 |
|---|---|---|---|
| **A0** Full-512 | `s01_BG128` | 无（同现有架构），但**训练预算匹配为 16 updates × 32 班 = 512 班** | 无 |
| **A1** −GAT | `s02_BG128_NOGAT` | `state_encoder.graph_layers` 置空，token 表示仅经 `TokenEncoder` + typed pool 直接读出 | `graph_layers=0`，`graph_weighting='none_tokenwise_readout'` |
| **A2** −类型锚定 | `s03_BG128_NOTYPE` | `encoder.py:71-72` 的 `semantic_type_embedding` 恒置零（保留参数量与索引，保证 forward 形状不变） | `semantic_anchoring=False` |
| **A3** −分组 softmax | `s04_BG128_NOGROUP` | `graph_layers.py` 的 `graph.edge_group` 全部置 0，分组数 4→1：全 19 token 单一 attention 分母、单一融合通道 | `graph_weighting='global_attention_all_tokens'`，`relation_group_ids` 全 0 |
| **A4**（可选）−C128 | — | **不建议做**：`models._common` 有 `require(feature_config['enabled'] and design['summary_enabled'])` 硬门禁，且 Critic 吃完整 C128。C128 的必留结论已由 11-18 表征路线决策固定，做反向验证的成本/收益不划算 | 需放宽受保护 require，不列入首批 |

A0 必须做：现有 u32 checkpoint 是 32 updates（1024 班），与消融臂的 16 updates 不匹配，直接用会混淆"训练量"与"结构"两个因子。

## 3. 必须保持不变量（改动边界）

1. `python/episode_budget_ppo.py` 逐字节不变（受保护核心，SHA 见 §6 门禁 G0）。
2. 观测契约不变：输入仍为 `(B,35)` 原始观测 + `prefixes` C128；`(B,19:23)` 动作掩码语义不变；候选恒为 4（Load0-3，含废石）。
3. 训练 reward、成本口径、4×2 小时窗品位约束、episode 终止条件全部沿用 009 现行契约。
4. 参数初始化仍走 `initialize(pair_seed=..., role=...)`，同一 `pair_seed` 在所有臂上一致，保证 init 不是混杂因子。
5. 每个臂独立 `design_sha256`（`ablation_contract` 扩写），checkpoint 的 `ablation_contract` 必须能反查臂身份。

## 4. 实现清单（G0 阶段要落的代码）

| 文件 | 改动 | 验证 |
|---|---|---|
| `business_graph_summary_v1/contract.py` | `CANDIDATES` 增补 s02/s03/s04；`ablation_contract(candidate)` 按 candidate 输出对应 design 字段与 SHA | `python -c` 打印三臂 design_sha256 互不相同 |
| `business_graph_summary_v1/models.py` | `ACTOR_SPEC`/`CRITIC_SPEC` 参数化（按 candidate 选择 spec 变体）；`GraphStateEncoder` 支持 `graph_layers=0`；`_common` 的 C128 require 保持 | `test_models.py` 增补三臂构造 + 参数量断言 |
| `business_graph_summary_v1/encoder.py` | `semantic_anchoring=False` 时 `semantic_type_embedding` 恒零（**不改形状**） | A2 vs A0 在随机权重下 forward 形状一致、数值不同 |
| `business_graph_summary_v1/graph_layers.py` | 分组数参数化（4 → 1） | A3 的 `group_messages` 维度为 `(B,19,1,64)` |
| `campaign`/`learner` 校验 | 确认 candidate 枚举不是硬编码单值；若硬编码则改为按 suite 声明 | 三臂各跑 1 个 update 的空转（无 AnyLogic）通过 |

**回归铁律（G1）**：A0 分支在 100 条历史冻结观测上 forward，输出 logits 必须与现网 s01_BG128 **逐位相同**（`torch.equal`）。不通过则不得进入 G2 —— 这条是防止"顺手改坏 Full 臂"的唯一硬闸。

## 5. 预算、身份与记账

| 项 | 值 |
|---|---|
| 每臂训练 | 16 updates × 32 班 = **512 班** |
| 每臂独评 | 固定 checkpoint，**50 班**（同一批 50 个评估种子，跨臂逐槽配对） |
| 每臂合计 | **562 班** |
| 首批 5 臂（A0–A3 实际 4 臂 + 余量） | A0/A1/A2/A3 = 4 × 562 = **2248 班** |
| 若独评加到 100 班/臂（推荐） | 4 × 612 = **2448 班** |
| 当前登记量 | 25,636 班（`speed_budget.py` 入口；速度扰动 40 班尚未 run，run 后为 25,676） |
| 本批登记后 | 25,676 + 2448 = **28,124 班**（含速度扰动） |
| 新记账入口 | `structural_ablation_001/ablation_budget.py::budget_snapshot`，必须合并 speed 入口 + 训练/独评两本旧账 |
| run_index 段 | **10011001–10013500**（四种子复现 s02–s05 占 10003001–10010724，已核；G3 前再复核） |
| recording_id | `bg009-ablation-<臂>-r01`（A0: `ab-r01`、A1: `ab-gat`、A2: `ab-type`、A3: `ab-group`），Python 输出目录 `runs/bg128-ablation-<臂>-2026MMDD-001/` |
| 配置 SHA256 | 由 `prepare` 阶段生成，写入各臂 `launch_identity.json`；**构建后再补，不使用占位值启动** |

**独评 50 vs 100**：50 班配对 sign test 在 α=0.05 下只能稳定检出中等以上效应（≈0.6 SD）。若要把"未检出差异"写成结论（而不是"证据不足"），必须 100 班。建议直接 100。

## 6. 预注册指标与判定规则（不可事后修改）

- **主指标**：完整任务率（4 窗全合规 且 矿石/废石达产）
- **次指标**：累计等待（车辆秒/班）、每班 `grade_cost`、`|grade−0.27|`、每班 ore 吨
- **统计**：对同一评估种子逐班配对，计算差值 → exact sign test（α=0.05）+ bootstrap 10,000 次重采样 95% CI
- **判定口径**：
  - 只有 95% CI 排除 0 且方向为"消融更差"，才写"该组件有贡献"；
  - CI 跨 0 → 写"未检出差异（power / 组件冗余）"，**不得**写"该组件不重要"；
  - 若消融臂在完整任务率持平但等待显著更差 → 写"该组件影响效率而非可行性"，这是可发表的分层结论。
- **不做**：跨臂挑最好的种子、按结果增删指标、事后换 α。

## 7. 执行门禁与工时

| 门禁 | 内容 | 工时 | 通过条件 |
|---|---|---|---|
| G0 | 代码 fork + 单元测试 + 参数量/design_sha256 | 0.5–1 天 | `test_models/test_graph/test_learner` 全绿 |
| G1 | A0 逐位回归（上述铁律） | 2 h | `torch.equal` 全 True |
| G2 | 每臂 AnyLogic smoke 2 班 | 1 h | RPC/schema/掩码一致，无 UNKNOWN |
| G3 | 正式：4 臂 × (512 + 100) = 2448 班 | ≈ 17 h 墙钟（~150 班/h） | 预算登记 + 身份禁重复核 |
| G4 | 报告：分臂表 + CI + 归因结论 | 半天 | 与预注册判据一致 |

## 8. 与论文的关系（不要混淆优先级）

**环境内建的随机性与拥堵（训练与独评全程开启，非 OOD）**：`freeSpeedFactorMin/Max=0.85/1.10`、`freeSpeedFactorStd=0.05`、`speedTau=1.8s`、`speedDt=0.2s`、`brakeTau=0.8s`、`followRange=70m`、`followSpeedMargin=0.3m/s`、`followTau=1.5s`、`useLocalDynamicSpeed=1`。即策略训练与评估时一直面对逐车自由速度扰动与内生跟驰/制动拥堵；这部分属于 in-distribution 鲁棒性，**不是**"未测速度扰动"。真正未测的只有**幅度外推**（σ=0.15 / 0.25 两档，40 班已准备未 run）。

四个支柱的当前状态见 `research_definition_v1_001/`。**消融不是最大缺口**：

| 支柱 | 状态 | 缺口 |
|---|---|---|
| ① 消融（本文件） | 未做 | 4 臂 × 612 班 |
| ② 训练机制创新（非拉格朗日） | 未定 | 需定一个明确的机制 claim 并给出对照 |
| ③ 仿真 + 4×2h 窗品位约束 | 已具备 | 无（seed1 7 工况 + seed2 复现中） |
| ④ 传统配矿基线 | **未跑（最大缺口）** | 规格已定：**滚动优化 + 纯反应式调度**，见 `../baseline_comparison_001/README.md` |

**结论：消融可以并行启动，但④必须先定规格并跑完，否则审稿人第一个问题答不上。④的规格于 11-20 已定（滚动优化 / 纯反应式），下一步是 S1 打通策略适配器链路。**
