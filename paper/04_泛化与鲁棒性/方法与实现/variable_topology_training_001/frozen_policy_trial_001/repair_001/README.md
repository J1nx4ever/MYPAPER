# B 图冻结策略独立重试

`loop_B` 保留原验证过的路网、7 辆出勤车、装卸设施和确定性仿真设置。班次为 8 小时，矿石破碎目标 16,000 t、废石卸达目标 4,000 t，四个两小时窗按 26%–28% 结算。派车仍用动态出勤 `u9` 在合法装载点中取最大概率动作，车位置换种子为 0。

本次独立运行编号为 **10054043**，环境种子为 **279598031114873**，记录 ID 为 `site-layout-frozen-policy-trial-20261127-repair01-10054043`。配置字节 SHA256 为 `fa50e5cbdf910e3bd249a8748cfb0ada7b0267c64a3b63c5926952e48848ad27`。Python 输出和 AnyLogic 事件分别写入 `runs/site-layout-frozen-policy-trial-20261127-repair01-B` 下的 `python` 和 `anylogic_events`。 [启动身份](launch_identity.json)

## 固定九车目标下的历史通信验收

本页的 10054043 班使用九车目标验证了 B 图的通信。按七车运力评估策略时，请使用换入正确目标的独立班 10054046。 [B 组运力缩放试跑](../capacity_scaled_001/README.md)

修复班在第 57 次派车向车 0 实际下发了 3 号装载点，前 56 次与原 B 的车号、时间、动作和观测哈希逐项一致。班末在 28,800 秒正常结算，156 次推理均与 AnyLogic 实际派车相符，非法和兜底派车均为零。运行身份 10054043 已封存，班后数据可直接复核。 [完整 B 班核验](/home/jinx/Faith/experiments/lagrangian_ppo_v1/reports/token_attention_001/business_graph_summary_009/variable_topology_training_001/frozen_policy_trial_001/repair_001/trial_evaluation_B.json)

| 冻结动态 u9 | 破碎矿石（t） | 卸达废石（t） | 合规品位窗 | 矿石目标（16,000 t） |
| --- | ---: | ---: | ---: | --- |
| 树形 A，9 辆 | 19,406.4 | 6,118.0 | 4/4 | 达标 |
| 环形 B，7 辆 | 13,778.5 | 6,051.1 | 4/4 | 欠 2,221.5 t |

B 的四窗品位依次为 27.56%、27.23%、27.88%、27.64%。同一 B 工况已有固定规则完整班实现 17,894.7 t 矿石、4,446.5 t 废石和四窗合规，确认该目标可达。 [B 工况固定规则回执](/home/jinx/Faith/experiments/lagrangian_ppo_v1/runs/site-layout-feasibility-witness-20261127-001-B/python/receipt.json)

## 原 B 中断的复现边界

原 B 的 `10054042` 以 `UNKNOWN` 封存。首次诊断 `10054044` 仅留下 `t=0` 的原生启动记录。第二次诊断 `10054045` 在 AnyLogic 重放了 56 个已执行动作，在 `8624.941626628355 s` 复现最后一次卸载，并截获车 0 的下一条派车请求 `decision_sequence=56`。诊断在应用动作前主动抛出 `CapturedDecision` 停机。冻结 `u9` 离线对该请求返回合法动作 3。原中断的异常栈未保留，根因仍待确认。 [原生诊断与离线复核](/home/jinx/Faith/experiments/lagrangian_ppo_v1/reports/token_attention_001/business_graph_summary_009/variable_topology_training_001/frozen_policy_trial_001/repair_001/diagnostic_10054045)

## AnyLogic 接线

打开独立的 B 重试 ALP，手工复核按下表定位。需要重建副本时，用生成器读取封存的原试跑 ALP。 [B 重试 ALP](/home/jinx/Models/DRL-PPO-baseline-topologydemo/DRL-PPO-baseline-topologydemo-frozen-policy-trial-b-repair.alp) · [生成器](prepare_repair_alp.py) · [封存的原试跑 ALP](/home/jinx/Faith/experiments/lagrangian_ppo_v1/reports/token_attention_001/business_graph_summary_009/variable_topology_training_001/frozen_policy_trial_001/DRL-PPO-baseline-topologydemo-frozen-policy-trial.alp)

| AnyLogic 位置 | 修改前 | 修改后及作用 |
| --- | --- | --- |
| 项目树 → `Topology_Frozen_Policy_B` → Properties → Parameters | B 试跑用 Freeform `topologyWitnessCase = 2` | 独立实验名为 `Topology_Frozen_Policy_B_Repair`，表达式仍为 `2`，Number of runs 为 `1`，Allow parallel evaluations 关闭。Model time 为 `0` 到 `28801 s`，业务结算在 `28800 s` |
| 项目树 → Java Classes → `TopologyWitness` → `DIRECTORY`、`start()` | 从原试跑 `configs` 读取 `B.json` 和 `B.sha256`。调用 `new TopologyPolicyDispatch(main, topology, configSha, text("checkpoint_sha256"))` | 从本目录 `repair_001/configs` 读取对应文件。构造派车器时追加 `Path.of(text("event_directory")).getParent()`，首错记录写在该班运行目录的 `native-error.log` |
| Java Classes → `TopologyPolicyDispatch` → 构造器、`choose()`、`post()` | 类内创建 `HttpClient`，调用一次 `post(body)` | 构造器接收 `Path output`，执行 `this.transport = new TopologyRetryTransport(output);`。删除原类内 `post()`，在 `choose()` 改用 `JsonObject reply = transport.post(body);`。回包身份、动作掩码和预留校验继续在原派车器中执行 |
| Java Classes → 新增 `TopologyRetryTransport` | 原 HTTP 客户端遇到 I/O 错误即中止 | 接入 HTTP/1.1。单次超时 15 秒，仅 `IOException` 重发同一请求，最多三次尝试。非 200 响应立即带状态码和服务端详情报错。首异常用独占写入保存。 [完整类源码](TopologyRetryTransport.java) |
| Java Classes → `TopologyWitness` → `dispatch()`、`applied()`、`departing()`、`arrived()`、`finish()` | 各方法直接执行原方法体 | 原方法体置于 `try` 中，捕获 `RuntimeException \| Error` 后调用 `recordFailure(phase, truck, error)` 并抛回。对应 phase 为 `dispatch`、`native_reservation_commit`、`road_departure`、`road_arrival`、`native_shift_close`，末项使用 `null` |
| `DumpSite` → 事件流 `exitDump`（Exit）→ Properties → Actions → On exit | 原生卸载完成、同步并发出 `restart` | 按下方代码包住卸载尾部，保持调用顺序。若该阶段出错，保留原堆栈和车辆、派车序号、仿真时刻 |

`exitDump → On exit` 原尾部

```java
t.onDumpingFinished(this);
main.fleetResilience.finishDump(t.truckId, time(SECOND));
main.fleetSync();
t.send("restart", t);
```

替换后的尾部

```java
try {
    t.onDumpingFinished(this);
    main.fleetResilience.finishDump(t.truckId, time(SECOND));
    main.fleetSync();
    t.send("restart", t);
} catch (RuntimeException | Error error) {
    if (main.topologyWitness != null)
        main.topologyWitness.recordFailure("dump_completion", t, error);
    throw error;
}
```

## 启动和复核

在本目录用 `anylogic-env` 执行第一条命令可只读核对封存的班次。第二条是本次启动时使用的命令，供具备独立运行身份的新班参考。本次启动时，在服务输出 `Policy ready on 127.0.0.1:50871 for B` 后运行原生实验。对应的手工操作是在 AnyLogic 按 **F7** 构建，右键 **Topology_Frozen_Policy_B_Repair → Run**。

```bash
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B collect_retry.py
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B repair_service.py
```

原生实验结束且 Python 封存推理清单后，执行下列命令。回执分别记录 8 小时物理完成、全量实际派车与 Python 推理一致性、四窗和矿废是否达标，并与 B 图的可行性见证比较。

```bash
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B collect_retry.py --collect
```

若实验已中止，先保存班目录的 `native-error.log` 或 Python 目录的 `first_inference_error.json`（如有），再用 `collect_retry.py --collect --case B --finalize-incomplete` 封存已启动但未完成的记录。
