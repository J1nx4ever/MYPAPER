# 共享代码

以下入口对应图构造、图层、模型、PPO 更新及仿真约束。实验共用模块随本地源码中可静态识别的 Python 导入收录。协议定义及相关 Java 源文件也在本目录。要运行原实验，请从对应冻结方案进入，并使用项目约定的 Conda 环境。

- [业务图关系和输入](实验共用模块/business_graph_summary_v1/graph.py)
- [分组图传播](实验共用模块/business_graph_summary_v1/graph_layers.py)
- [Actor 与 Critic](实验共用模块/business_graph_summary_v1/models.py)
- [学习器](实验共用模块/business_graph_summary_v1/learner.py)
- [原 PPO 核心的只读参考副本](仓库公共实现/python/episode_budget_ppo.py)
- [AnyLogic 主模型原件](../../experiments/lagrangian_ppo_v1/anylogic/3d/DRL-PPO-baseline1/DRL-PPO-baseline1.alp)
