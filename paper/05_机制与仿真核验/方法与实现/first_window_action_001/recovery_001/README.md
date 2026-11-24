2026-11-19，控制器 PID 848493 在发布首班 ready 前退出。原脚本使用通配符导入，导致下划线开头的 `_publish` 在账本写入与异常 STOP 路径均未绑定。原控制器日志和启动回执保留。

核对宿主机确认旧 PID 已退出，Java PID 848188 仍在 `Baseline1CampaignPlan.awaitConfigSha256` 等待。四槽磁盘状态为 PLANNED，零 ready、零 Java claim、零 RPC 轨迹和零物料批次目录。原配置与控制器初始化保存的副本 SHA 一致。

恢复入口保留原封存配置、suite、脚本和 Java helper，显式绑定同一个发布函数，并加载既有四槽账本。临时账本测试通过持久化 claim、拒绝恢复已领取槽位及 STOP 原子发布。只恢复已等待的 Run，复用了已登记实验配额。

新控制器为 PID 850209，已核对宿主机存活、首班 ready、Java 独占声明和物料 episode 目录。 [恢复身份与来源](../../../../../runs/bg128-first-window-action-20261119-001/control/controller_recovery_001/identity.json) · [恢复控制器日志](../../../../../runs/bg128-first-window-action-20261119-001/control/controller_recovery_001/controller.log)
