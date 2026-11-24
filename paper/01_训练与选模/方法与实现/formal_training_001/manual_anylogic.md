# BG128 单随机种子正式训练：AnyLogic 手工清单

2026-11-18，Asia/Shanghai。本轮仅 `s01_BG128`，32班/批、16次更新，共512个新训练班次；不增加独立冒烟、独立评价或第二组。第1批32班就是正式训练数据，完成RPC、物料账本、观测构图、联合成本、Actor/Critic更新及checkpoint提交检查后，继续第2—16批。错误触发STOP，禁止静默跳过或补班。

架构冻结为19个64D实体、两层GAT、transport/commitment/service/material四组关系、25种有向关系、16条有向道路属性边、每节点21维确定性数量及原C128。Actor/Critic各自独立编码。正式模型从新初始化seed `75597143230906` 开始，不继承随机传播试验、物理干预、旧008或其他checkpoint。

## 训练目标与数据

数据来自本轮AnyLogic在线交互；历史32班仅用于既有离线检查，不作PPO回放训练。Actor与单标量Critic严格使用同一008冻结联合成本：四窗品位deadband=.004、scale=.006、cap=2、空窗成本2；矿石16000吨和废石4000吨缺口各权重8；车队累计等待权重0.2/28800。Critic拟合同成本MC，Actor用更新前冻结V计算同成本GAE并统一标准化联合优势。RPC中的旧形态奖励仅作物理证据核验，不直接作为新训练标签。

PPO保持原正式设置：Actor学习率3e-4、Critic学习率1e-3、4epochs、clip=.2、entropy=.01、最大梯度范数.5、GAE lambda=.95、gamma=1。一次实验内不改业务权重。不启用逐transition、RPC JSONL、rollout快照等高频记录；保留HDF5训练数据、episode汇总、异常及完整性计数。

## 只需更换两处

操作对象为当前运行的新环境模型，原模型常见位置是 `anylogic/3d/DRL-PPO-baseline1/DRL-PPO-baseline1.alp`。本轮不直接编辑`.alp`；用户先备份模型，然后手工执行下表。

|精确位置|修改前|修改后|理由、影响与验证|
|---|---|---|---|
|Additional Class → `Baseline1CampaignPlan` → 全部Java源码|当前已结束的旧008或其他campaign helper全文|全文替换为 [prepared_001/Baseline1CampaignPlan.java](prepared_001/Baseline1CampaignPlan.java)|绑定本轮新master、配置SHA及BG128控制器；禁止只改班数或沿用旧身份。核对`jobCount()`和`frozenPlanJobCount()`均为512，`resumeOffset()`为0，`runIndex(0)`为9900001，`runIndex(511)`为9900512。|
|`RL_Training → Properties → Number of runs`|当前helper对应的旧班数，例如1224|`512`|单组512班在线训练；没有额外评价或冒烟班次。|
|`RL_Training → Parameters → Freeform`中的helper调用|现有`Baseline1CampaignPlan.*((int) index)`表达式|保留，不加offset|helper自动提供run/环境seed/采样seed/recording与事件根目录。不得手填覆盖旧值。|
|`Main → On startup`中的`Baseline1CampaignPlan.awaitConfigSha256(...)`|现有调用及后续材料/RPC初始化|保留|首个Main核对并启动唯一Python控制器，等对应配置ready、绑定实际配置字节SHA，再发RPC。|
|`RL_Training`运行设置|串行、允许并行评估不勾选、使用重复不勾选|保留|严格按master逐班、逐批推进；批间无需再Run。|
|`Main`矿石/废石目标与停派参数|矿石16000、废石生产目标4000、废石停派6000、上界6193、末车193|保留|RPC v2逐项核对；6000是停派阈值，不是训练生产目标。|

Save → Build → 核对下面的新身份 → `RL_Training → Run` **一次**。helper首次运行自动调用固定的`anylogic-env` Python桥接，不需要另开第二个Python。若代理已经启动这份suite的控制器，helper将核对suite、PID启动身份与固定命令后复用，不重复创建。

出现STOP、UNKNOWN、未完成claim、端口被其他控制器占用或Build/Run错误时，保留现有日志与账本，先定位问题；不要重复Run，不删除claim，不复用旧run补班。

## 本轮独立身份

完整配置逐批身份见 [launch_identity.json](prepared_001/launch_identity.json)，512班run/环境seed/采样seed/初始化seed清单见 [identities.csv](prepared_001/identities.csv)。配置SHA是文件实际字节SHA，suite SHA与每批配置SHA不同。

|字段|值|
|---|---|
|campaign|`business-graph-summary-formal-seed1-20261118-001`|
|candidate|`s01_BG128`|
|suite字节SHA256|`90ada18aa85cdba4b51b3c55e9a3326729662a1dcc1b53a4b331a029cd7614d3`|
|首批配置字节SHA256|`253639247b3c8c72a3b7b149834b27aa89be1ca54ccafa225a7eabf8abab024f`|
|run范围|`9900001—9900512`|
|初始化seed|`75597143230906`|
|首班环境seed|`255314794623997`|
|首班Actor采样seed|`274856137631895`|
|首recording_id|`business-graph-summary-formal-seed1-20261118-001-s01_BG128_train_u0001`|
|首批Python输出|`/home/jinx/Faith/experiments/lagrangian_ppo_v1/runs/business-graph-summary-formal-seed1-20261118-001/arms/s01_BG128/train_u0001`|
|首批AnyLogic物料事件目录|`/home/jinx/Faith/experiments/lagrangian_ppo_v1/runs/business-graph-summary-formal-seed1-20261118-001_materials/s01_BG128_train_u0001`|

不可变包已生成。身份、源文件和配置字节SHA、全部512行Java只读访问器经准备核验后，首次Run仍会再次执行启动门禁。首批配置文件为 [s01_BG128_train_u0001.json](prepared_001/configs/s01_BG128_train_u0001.json)，其余15批见完整身份JSON。

## 输出与完成判据

控制器低频进度：`runs/business-graph-summary-formal-seed1-20261118-001/training_progress.json`；控制器日志：本计划`prepared_001/launch/controller.log`。只读状态检查使用新包：

```bash
cd /home/jinx/Faith/experiments/lagrangian_ppo_v1
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B -m business_graph_summary_campaign_v1.launch status --suite change_plans/token_attention_001/business_graph_summary_009/formal_training_001/prepared_001/suite.json
```

完成应满足512班COMPLETE、UNKNOWN=0、16次COMMITTED、唯一BG128组u0016检查点及suite完成记录；所有训练批都有同一成本定义、MC/GAE冻结证据、参数变化/梯度与采集完整性计数。正式训练统计可以描述学习过程，不能充当固定checkpoint的独立评价；八项独评指标仍按既有定义保留，本轮未另采独评班次。
