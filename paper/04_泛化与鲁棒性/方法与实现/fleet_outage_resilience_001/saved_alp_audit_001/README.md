# baseline2 保存版本的复核与手工修正

复核对象为 `/home/jinx/Models/DRL-PPO-baseline2/DRL-PPO-baseline2.alp`，保存文件的 SHA256 为 `ebb6c2d129d4ea8a24b877be44c9e3a199d5898adfcc46db562daf2ea05f7522`。以下行号指这一版。当前生成的 29 个 Java 源文件可用本机 SDK 编译，单独的账本与同刻事件用例通过。要在 IDE 启动 992 班，先完成以下手工修正。

## 四处影响正式运行的设置

| IDE 位置 | 当前内容 | 修改与原因 |
|---|---|---|
| Main → Agent actions → On startup，第 200–206 行 | 先取 `getDumpSiteById(0)` 校验空仓，再执行 `init1();` | 将**已有**的 `init1();` 移至 `DumpSite fleetCrusher = getDumpSiteById(0);` 的正上方。保留前面创建 `fleetResilience` 的位置与后面的空仓校验、定时器登记。班初 `dumpSites` 为空，正是 `init1` 创建 0 号破碎站；整段只调用一次 `init1`。 |
| DumpSite → delayDump → Delay time，第 21297–21299 行 | 表达式调用 `startDump(…, 60.0 * dumpServiceBoundaryMinutes(…))`，属性单位为分钟 | 保留表达式，在同一属性的单位下拉框中选**秒**。两分钟卸载在状态中登记为 120 秒，现在流程块会实际等待 120 分钟。改后两者都为 120 秒。 |
| RL_Training → Parameters → Freeform → `wasteDispatchStopTonsShift`，第 26058–26060 行 | Freeform 为空，继承 Main 的默认值 6000 | 填入 `Baseline1CampaignPlan.wasteStop((int) index)`。这样停派线按当班出勤运力缩放。默认 6000 只匹配 296 个九车班，另 696 班的停派线和上限都与冻结配置不同。 |
| Main → Functions → `buildRewardConfigComponents`，第 9455–9456 行 | 对 `config.waste_task_overrun_tons` 写入 `getWasteTaskUpperTons() - wasteDispatchStopTonsShift` | 将这一个字段的两行替换为下面的代码，直接取单车最大额定载重。前面的停派线及上限继续使用各自的原有表达式。此处是原方案漏掉的浮点兼容问题。 |

```java
double maxPayloadTons = 0.0;
for (Truck t : trucks) {
    maxPayloadTons = Math.max(maxPayloadTons, t.capacity);
}
c.put("config.waste_task_overrun_tons", enableWasteTaskLimit ? maxPayloadTons : 0.0);
```

第 2 班 `10043002` 是第一个六车班，冻结配置的停派线为 `4000.0000000000005`，上限为 `4193.0`，单车容差为 `193.0` 吨。用上限减停派线反算得到 `192.99999999999955`，严格校验会拒收，影响全部 104 个六车班。直接取最大额定载重后，992 班的这三个字段都与封包逐值一致。

## 两处速度循环和一个参数默认值

- Main → Events → `evUpdateSpeedObservation` 第 3789 行和 `evApplySpeedActions` 第 3831 行。在各自第一个 `for (Truck t : trucks) {` 紧后插入 `if (isFleetTruckDisabled(t.truckId)) continue;`。前者从零时刻起每秒执行，后者当前触发条件为 `false`。这个判断使用 `inSystemMask`，在网退场中的车仍被更新，实际退场和班初缺勤的车会被跳过。
- Main → Parameters → `fleetFaultTruckId` 第 3576 行，将空白默认值设成 `-1`，与健康班的 `fleetFaultTriggerSeconds = -1.0` 成对。当前生成的 Java 默认值是 0。`RL_Training` 已显式读取本班车号，默认值影响直接使用 Main 默认参数的入口。

## 已确认的挂点

用户选中的 `buildRewardConfigComponents` 第 9429–9435 行位置正确，实时双掩码和废石承诺对账用于每次 RPC，终端以真实入仓、破碎、FIFO 三方量核对矿石和金属。破碎积分的退场窗切片位于更新游标之前。一次性故障事件、派单入口、卸载记账后退场和 `restart` 完成最后行程的顺序均已接入。实验为 992 次串行运行，运行身份、车型、吨位目标、故障车与时刻来自本轮 CampaignPlan。

在 IDE 保存并 Build 后，可从生成的 `Main.java` 核对开班先 `init1()` 再取破碎站，检查 overrun 直接取最大载重；在 `DumpSite.java` 核对 `delayDump` 的延时单位是 `SECOND`；在 `RL_Training.java` 核对 `setupRootParameters` 对停派线调用 `wasteStop((int) index)`。这些检查均可在正式启动之前进行。[本次只读复核记录](../../../../../reports/token_attention_001/business_graph_summary_009/fleet_outage_resilience_001/saved_alp_ebb6c2d1.json)
