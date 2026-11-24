# 第一次环境验收：用户手动操作

本轮已完成：服务正常退出，真实单班次结果与奖励/物料复核见 [实际结果报告](reports/environment_validation_001/actual_result_review.md)。当前没有继续监听，不要直接再次Run。下文保留本次启动步骤；后续运行使用独立新输出目录。保存到工作区的ALP仍未体现本次手工设置，待整理运行副本；正式训练门禁保持关闭。

用户已批准本次临时参数并要求开始环境验证。本次只运行一个当前场景的完整八小时班次，核查 RPC、奖励与窗口账本；不生成正式离线训练数据、不执行 MC 或 PPO。实际 AnyLogic 由用户启动。固定策略为 `first_legal_rule`，用于接口与账本检查；欠产、窗口违规按实际结果记录，不作为环境实现错误的充分依据。

## 1. 已准备的服务与配置

- 临时验收配置：[config/acceptance_environment_001.json](config/acceptance_environment_001.json)，`acceptance_run_approved=true`，只适用于本次环境验收。
- 一班次清单：[config/acceptance_seeds_001.json](config/acceptance_seeds_001.json)，run_index=1、seed=202610280401，沿用现有 RL_Training 第一项；场景标签为当前 ALP 基准模型，不代表已构建新的矿山场景。
- RPC：`127.0.0.1:50051`；服务已打印 LISTENING。
- 实际结果目录：`runs/environment_acceptance_001_retry2/`。首次沙箱绑定失败的 `runs/environment_acceptance_001/` 仅保留配置，不是实际班次结果，不覆盖它。
- 服务等待超时：3600 秒墙钟时间；仿真班次仍为 28800 秒。服务超时/中断需要新输出目录重启，不能复用已有目录。
- 默认不保存全量观测和逐转移详细日志；保留班次结果、配置、四窗数据与异常。针对性事件顺序和等待兜底证据后续另行安排。

正式模板 `config/unconfirmed.json` 仍为 `approved=false`，κ 等正式参数仍为 null；`acceptance_unconfirmed.json` 仍保持 false。这次临时选择不冻结正式训练设置，也不批准正式数据采集或训练。

## 2. 在实验 ALP 手工完成三项

打开 `/home/jinx/Faith/experiments/lagrangian_ppo_v1/anylogic/DRL-PPO.alp`。不要修改原件或 baseline 快照。以下项目由用户手工完成，助手没有编辑 ALP。

### A. Main → Variables → rewardWaitKappa → Initial value

修改前：

```java
Double.NaN
```

修改后：

```java
1.0 / 25000.0
```

理由：使用用户刚批准的临时验收尺度，单位 reward/(车·秒)。影响：只改变本次验收等待奖励尺度；保留 Startup 非有限值/正值检查。验证：首个 RPC 的 `config.wait_kappa` 必须等于验收 JSON 的 `0.00004`，不一致会拒绝。

### B. RL_Training → Properties → Parameters → Number of runs

修改前：`55`。

修改后：`1`。

理由：服务只接受一班次清单，避免误启动旧 55 班设置。其余必要参数逐项核对：

| GUI 位置 | 本次值 |
|---|---|
| Use freeform parameters | true |
| Freeform → currentDispatchPolicy | 3；如为空继承 Main，确认 Main 实际为 3，或显式填 `3` |
| Freeform → experimentRunIndex | `(long) index + 1L`；单次 index=0，对应 run_index=1 |
| Freeform → experimentSeed | `202610280401L + (long) index`；单次对应清单 seed |
| Freeform → usePhysicalObservation | true |
| Freeform → bcCollectEnabled | false；BC 输出目录无需修改 |
| Freeform → diagEvalEnabled | true，保留班末诊断 |
| Freeform → deterministicCheck | false，保持当前物理场景 |
| Allow parallel evaluations | false |
| Replications → Use replication | false |
| Stop time / Final time | 28800 秒 |
| Simultaneous events | LIFO |

验证：只出现一个完整 episode；收到不同 seed/run_index、缺窗、截断或错误班末时服务应拒绝，不补造正常结果。

### C. Truck → TravelToDumpSite → ArriveSegmentEnd1 → Entry action

本次具体测试发现该处仍缺日志开关，当前 ALP 文本约第 15336 行。注意选择带 `1` 的 `ArriveSegmentEnd1`，不是装载路段中的 `ArriveSegmentEnd`。

修改前：

```java
traceln("Truck " + truckId + " arrived node: " + currentNode);
```

修改后：

```java
if (main.experimentVerboseTrace) traceln("Truck " + truckId + " arrived node: " + currentNode);
```

理由：满足默认关闭高频到站日志；当前这是唯一未通过的具体日志开关测试项。影响：仅控制打印，不更改节点更新或状态转移。验证：保持 `Main.experimentVerboseTrace=false`，该到站打印默认关闭；原业务动作仍执行。现有完整面板代码也可参照 [manual_alp/14_Truck_ArriveSegmentEnd1_EntryAction.after.txt](manual_alp/14_Truck_ArriveSegmentEnd1_EntryAction.after.txt)，已有其他自定义代码时仅合并这一行。

## 3. 核对临时数值并 Build

以下值应与本次批准的配置一致；原值已一致时不修改：

| Main 对象 | 本次临时值 |
|---|---:|
| wProduction / wWasteShortfall | 4 / 4 |
| gradeLowerBound / gradeUpperBound | 0.26 / 0.28 |
| cpGradeEpsilon | 1e-12 |
| cpTonnageEpsilon | 1e-6 t |
| cpMassBalanceEpsilon | 1e-5 t |
| dispatchInferenceHost / dispatchInferencePort | `"127.0.0.1"` / 50051 |
| enableOnnxDispatch | true，历史命名，实际启用 RPC |
| enableDispatchFallback | false |
| experimentVerboseTrace | false |

保存实验副本，执行 AnyLogic Build。Build 失败时保留完整错误文本；Build 成功后才点击 `RL_Training` 的 Run。无需再次修改品位范围、FIFO、装卸或窗口拆分代码。

## 4. 观察结果并停止

- AnyLogic 应报告四个窗口与班末物料审计；业务违规/欠产仍保留真实数值。
- Python 完整班次结果写入 `runs/environment_acceptance_001_retry2/episodes.json`；汇总写入 `summary.json`。
- `summary.json.status=observed_complete_requires_review` 仅表示取得正常完整班次，不能视为所有环境验收项目 passed。
- 错误、断连、截断应拒绝；服务异常写 `ABORTED.json`。若绑定发生在异常处理前失败，可能只有配置文件，不能当作完整班次。
- 本次结束后停止并审查奖励账本、四窗结果和运行配置。等待兜底可达性、同戳业务全序、专门空窗/跨窗案例及全量观测时序尚需独立证据。

## 5. 服务重启命令（当前已监听时不要重复启动）

先预检，确认端口没有另一个本服务实例；重启必须把 `--output` 改为未存在的新目录：

```bash
cd /home/jinx/Faith/experiments/lagrangian_ppo_v1
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B -m python.acceptance_service preflight --config config/acceptance_environment_001.json
/home/jinx/miniconda3/envs/anylogic-env/bin/python -B -u -m python.acceptance_service run --config config/acceptance_environment_001.json --seed-manifest config/acceptance_seeds_001.json --output runs/environment_acceptance_001_retry3 --callback-timeout 3600 --host 127.0.0.1 --port 50051
```

沙箱中本地端口绑定受限；当前实例是在获批沙箱外启动。用户自己的终端可按其本地权限执行。此命令不会启动 AnyLogic，也不会执行网络优化。
