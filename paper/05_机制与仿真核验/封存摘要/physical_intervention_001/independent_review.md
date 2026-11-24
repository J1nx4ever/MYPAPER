# BG128 原始观测干预独立核验

状态：**PASS**；426 项检查。

从当前 scenarios.json 独立读取 BASE/A/B/C，未调用构造脚本；从原始HDF5重新核验共同T1基准。

- 原始修改索引严格符合独立白名单：A20项、B12项、C1项；A可用吨位保持原始float32位模式，避免1ULP伪变化。
- 独立重算21维数量、服务/预留/物料守恒、路线连续性与长度、候选mask/tons。
- 从各自原始输入重建图、H0/H1/H2、C128与证书，逐元素核对保存的候选输入、表示、logits、概率和价值。
- Actor普通与diagnostic前向一致；显式重建候选Ego/Load/Dump及乘性交互评分公式；参数/buffer前后摘要不变。

## C 的跨物料内容传播

C没有修改事实图或21维数量，但Dump0率通过初始dump_ref使Truck2/3/5/6/7的H0改变。Load0..3的H0均未变。
到Load3存在7条两跳消息通道：Truck3→Junction3→Load3；Truck3→Truck4→Load3（三条不同入边）；Truck5→Truck4→Load3（三条不同入边）。
具体方向、关系和边属性见 C_Load3_content_paths.json 与 cross_material_paths_C.csv。它们解释无拓扑变化也可有Load3 H2响应；不意味着矿石变为废石或业务因果正确。
`truck_predecessor` 是首次派车前驱关系，并非当前道路运动前车。7条路径共享节点、非线性与归一化，不能将各条路径范数相加来分摊总响应。
两种角色的H0变化来源及L2：{"actor": [{"node": "Truck2", "H0_delta_l2": 0.022356629371643066}, {"node": "Truck3", "H0_delta_l2": 0.019399873912334442}, {"node": "Truck5", "H0_delta_l2": 0.019489869475364685}, {"node": "Truck6", "H0_delta_l2": 0.021342942491173744}, {"node": "Truck7", "H0_delta_l2": 0.02022131346166134}, {"node": "Dump0", "H0_delta_l2": 0.03238443657755852}], "critic": [{"node": "Truck2", "H0_delta_l2": 0.01743808574974537}, {"node": "Truck3", "H0_delta_l2": 0.017666613683104515}, {"node": "Truck5", "H0_delta_l2": 0.018782760947942734}, {"node": "Truck6", "H0_delta_l2": 0.015202028676867485}, {"node": "Truck7", "H0_delta_l2": 0.015551965683698654}, {"node": "Dump0", "H0_delta_l2": 0.024169890210032463}]}

## 边界

- A is source-compatible after-load-completion/before-travel microstep, not proof production RPC can pause there.
- A uses a grade already completely inferred at BASE; no T0/T1 future result is inserted.
- B is an exogenous reservation/route-tail reassignment, not a normal exposed PPO action.
- C is an exogenous same-time positive processing-rate step, outside original runtime configuration.
- Raw fixtures satisfy checked observable constraints, not complete hidden AnyLogic-state reachability.
- H0 already embeds references and deterministic quantities; no exclusive two-hop mediation claim is made.
- The seven C-to-Load3 paths are overlapping information channels; individual path contributions cannot be summed to the total H2 response.
- truck_predecessor denotes the initial dispatch predecessor, not the currently leading truck in road motion.
- Full Actor/Critic and C128 fusion were run this time; all outputs remain untrained numeric sensitivities.
