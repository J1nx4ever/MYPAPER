"""Read-only BG009 historical snapshot reward audit; no model construction.

Reuses the sealed 008 cost implementation and validates raw RPC ledgers. All
future settlements remain audit labels, never graph/C128 inputs. Run using the
project anylogic-env Python. The selection.json is created by the graph audit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import h5py
import numpy as np

PROJECT = Path(__file__).resolve().parents[4]
REPO = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT))
from python.constrained_batch import load_dataset
from python.reward_config import configuration_from_components
from python.reward_contract import validate_reward_accounting
from single_critic_task_query_v1.storage import read_episode_hdf5
from single_critic_task_query_v1.targets import build_composite_targets, composite_contract, grade_cost

PHASE = PROJECT / 'runs/single-critic-task-query-formal-seed1-20261117-002/arms/s01_TQ128/train_u0016'
REPORT = PROJECT / 'reports/token_attention_001/business_graph_summary_009/historical_snapshot_001'
PROTECTED = REPO / 'python/episode_budget_ppo.py'
EXPECTED_CORE = '8e4cfa31c622f5e33756e304eea3f7526547db72ae93f7d9f00808845d61ef96'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, default=REPORT / 'selection.json')
    parser.add_argument('--output', type=Path, default=REPORT)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    run_selected = int(selection['run_index'])
    indices = {label: int(selection[label]['state_index']) for label in ('T0', 'T1')}
    assert indices['T1'] == indices['T0'] + 1, 'snapshots must be adjacent raw decision rows'
    frozen_path = PHASE.parent / 'updates/0016/training_targets.h5'
    phase = json.loads((PHASE / 'phase_config.json').read_text())
    common_path = Path(phase['common_reference']['path'])
    assert sha(common_path) == phase['common_reference']['sha256']
    inputs = {PROTECTED, args.selection, PHASE / 'phase_config.json', frozen_path,
              PROJECT / 'single_critic_task_query_v1/targets.py',
              PROJECT / 'python/reward_contract.py', common_path}
    episode_dirs = sorted((PHASE / 'episodes').glob('run_*'), key=lambda p: int(p.name[4:]))
    assert len(episode_dirs) == 32, 'requires the complete historical 32-shift phase'
    for directory in episode_dirs:
        for name in ('samples.h5', 'episodes.json', 'manifest.json', 'receipt.json', 'material_receipt.json'):
            inputs.add(directory / name)
        material = PHASE / 'material' / (directory.name + '.h5')
        inputs.update((material, Path(str(material) + '.manifest.json'), Path(str(material) + '.seal.json')))
    before = {str(path): sha(path) for path in sorted(inputs)}
    assert before[str(PROTECTED)] == EXPECTED_CORE, 'protected PPO core differs'
    with h5py.File(frozen_path, 'r') as source:
        frozen = {dataset.attrs['array_name']: dataset[...] for dataset in source['arrays'].values()}
        frozen_meta = json.loads(source['metadata_json'][...].tobytes())
    assert frozen_meta['transaction']['learning_contract']['reward_contract'] == composite_contract()
    assert frozen_meta['transaction']['run_indices'] == [int(p.name[4:]) for p in episode_dirs]
    assert json.loads(common_path.read_text())['composite_contract'] == composite_contract()
    csv_rows, run_audits, selected = [], [], None
    costs_all, mc_all = [], []
    max_raw_error = max_wait_error = max_state_error = 0.0
    for directory in episode_dirs:
        run = int(directory.name[4:])
        episode = read_episode_hdf5(PHASE / 'material' / (directory.name + '.h5'), validator=lambda e: None)
        rows, ledgers, _, _, raw_manifest = load_dataset(directory, require_real=True)
        try:
            assert len(ledgers) == 1 and raw_manifest['source_kind'] == 'real_anylogic_rpc'
            ledger = ledgers[0]
            infos = [ledger['start_info']] + [row['settlement_info'] for row in rows]
            memory = episode['memory']
            markers = [event for event in memory.events if event.kind in ('DECISION', 'END')]
            assert len(infos) == len(markers) == len(rows) + 1
            assert all((event.request_id, event.decision_sequence, event.time) ==
                       (info['request_id'], info['decision_sequence'], info['sim_time_seconds'])
                       for event, info in zip(markers, infos))
            assert len(memory.settlements) == len(ledger['windows']) == 4
            cfg = configuration_from_components(infos[0]['reward_components'])
            prior_wait, actual_rewards, rpc_components, raw_wait_deltas = 0.0, [], [], []
            state_error = float(np.max(np.abs(episode['states'][0] - rows[0]['observation'])))
            for index, row in enumerate(rows):
                info, terminal = row['settlement_info'], bool(row['terminated'])
                components = info['reward_components']
                validate_reward_accounting(info['transition_reward_raw'], components,
                                           info['closed_windows'], config=cfg, terminal=terminal)
                assert row['reward'] == info['transition_reward_raw']
                assert row['action_time_seconds'] == infos[index]['sim_time_seconds']
                error = abs(info['transition_reward_raw'] - math.fsum(components[k] for k in ('wait', 'production', 'waste')))
                max_raw_error = max(max_raw_error, error)
                delta = components['delta_wait_vehicle_seconds']
                max_wait_error = max(max_wait_error, abs(components['wait_vehicle_seconds'] - prior_wait - delta))
                prior_wait = components['wait_vehicle_seconds']
                raw_wait_deltas.append(delta)
                actual_rewards.append(float(row['reward']))
                rpc_components.append({k: float(components[k]) for k in
                                       ('wait', 'production', 'waste', 'grade_2h', 'grade_action', 'grade_shift')})
                state_error = max(state_error, float(np.max(np.abs(episode['states'][index+1] - row['next_observation']))))
            max_state_error = max(max_state_error, state_error)
            assert state_error == 0.0 and np.array_equal(episode['wait_deltas'], raw_wait_deltas)
            terminal = infos[-1]['reward_components']
            rebuilt = build_composite_targets(memory, wait_deltas=raw_wait_deltas,
                ore_tons=terminal['ore_tons_shift'], waste_tons=terminal['waste_tons_shift'],
                total_wait_seconds=terminal['wait_vehicle_seconds'])
            assert rebuilt == episode['targets']
            for settlement, window in zip(memory.settlements, ledger['windows']):
                assert settlement.window == window['window_index']
                assert math.isclose(settlement.tons, window['ore_tons'], rel_tol=1e-10, abs_tol=1e-8)
                assert math.isclose(settlement.metal, window['metal_tons'], rel_tol=1e-10, abs_tol=1e-8)
                assert math.isclose(settlement.grade_reward_raw, window['grade_reward_raw'], rel_tol=1e-10, abs_tol=1e-8)
            total_grade = math.fsum(grade_cost(s.grade if s.grade_valid else None, empty=not s.grade_valid)
                                    for s in memory.settlements)
            total_production = 8*max(0., 1-terminal['ore_tons_shift']/16000) + 8*max(0., 1-terminal['waste_tons_shift']/4000)
            total_wait = .2*terminal['wait_vehicle_seconds']/28800
            physical_cost = math.fsum((total_grade, total_production, total_wait))
            combined = math.fsum(rebuilt.step_costs)
            assert abs(combined - physical_cost) < 1e-10
            costs_all.extend(rebuilt.step_costs)
            mc_all.extend(rebuilt.mc_returns)
            audit = dict(run_index=run, environment_seed=episode['identity']['environment_seed'],
                transitions=len(rows), grade_cost=total_grade, production_cost=total_production,
                wait_cost=total_wait, joint_cost=combined, physical_formula_cost=physical_cost,
                physical_formula_abs_error=abs(combined-physical_cost), rpc_reward_sum=math.fsum(actual_rewards),
                ore_tons=terminal['ore_tons_shift'], waste_tons=terminal['waste_tons_shift'],
                fleet_wait_seconds=terminal['wait_vehicle_seconds'], missing_reward_components=[],
                raw_vs_material_states_max_error=state_error)
            run_audits.append(audit)
            csv_rows.append(dict(snapshot='ALL_32', scope='completed_shift_posthoc_audit', run_index=run,
                component='joint_cost', value=combined, current008_cost=combined,
                raw_rpc_reward=math.fsum(actual_rewards), available_at_snapshot=False,
                available_at_time=28800., entered_snapshot_input=False, data_status='observed_and_rebuilt',
                note='Full-shift physical audit only; terminal production and future windows do not enter either snapshot.'))
            if run == run_selected:
                selected = dict(episode=episode, rows=list(rows), ledger=ledger, infos=infos,
                                markers=markers, targets=rebuilt, audit=audit, raw_directory=str(directory))
        finally:
            rows.close()
    assert selected is not None
    costs_all, mc_all = np.asarray(costs_all), np.asarray(mc_all)
    assert np.array_equal(costs_all, frozen['costs']), 'reconstructed step cost differs from frozen training targets'
    assert np.array_equal(mc_all, frozen['mc_returns']), 'reconstructed MC differs from frozen training targets'
    episode, rows, infos, markers, targets = (selected[k] for k in ('episode','rows','infos','markers','targets'))
    snapshot_details = {}

    def add(label, scope, component, value=None, **kwargs):
        row = dict(snapshot=label, scope=scope, run_index=run_selected, component=component,
                   value=value, entered_snapshot_input=False)
        row.update(kwargs)
        csv_rows.append(row)

    for label, index in indices.items():
        info, marker = infos[index], markers[index]
        assert index < len(rows) and marker.kind == 'DECISION'
        assert info['sim_time_seconds'] == selection[label]['time_seconds']
        components = info['reward_components']
        incoming = index-1
        details = dict(state_index=index, decision_sequence=info['decision_sequence'],
            material_event_seq=marker.seq, time_seconds=marker.time,
            encoded_time_seconds=float(episode['states'][index,0])*28800.,
            ego_id=rows[index]['truck_id'], historical_action=rows[index]['action'],
            incoming_transition_index=incoming, incoming_reward_interval=(infos[incoming]['sim_time_seconds'], marker.time),
            received_rpc_reward=info['transition_reward_raw'], received_rpc_components={k: components[k] for k in
                ('wait','production','waste','grade_2h','grade_action','grade_shift')},
            incoming_current008_cost=dict(joint=targets.step_costs[incoming], grade=targets.grade_costs[incoming],
                production=targets.production_costs[incoming], wait=targets.wait_costs[incoming]),
            accrued_fleet_wait_seconds=components['wait_vehicle_seconds'],
            closed_windows_before=[s.window for s in episode['memory'].settlements if s.seq < marker.seq],
            unsettled_windows=[s.window for s in episode['memory'].settlements if s.seq >= marker.seq],
            cumulative_settled_grade_cost=math.fsum(grade_cost(s.grade if s.grade_valid else None, empty=not s.grade_valid)
                for s in episode['memory'].settlements if s.seq < marker.seq),
            cumulative_wait_cost=.2*components['wait_vehicle_seconds']/28800,
            production_cost_status='not_settled_until_true_END',
            current_observed_processed_ore_tons_float32=float(episode['states'][index,7])*20000.,
            current_observed_delivered_waste_tons_float32=float(episode['states'][index,9])*4000.)
        snapshot_details[label] = details
        common = dict(transition_index=incoming, decision_sequence=info['decision_sequence'],
            interval_start=infos[incoming]['sim_time_seconds'], interval_end=marker.time,
            available_at_snapshot=True, available_at_time=marker.time,
            data_status='actual_RPC_record', source_reference=f"{selected['raw_directory']}/samples.h5 row {incoming} settlement_info")
        add(label,'received_previous_action_reward','RPC_transition_reward_raw',info['transition_reward_raw'],
            raw_rpc_reward=info['transition_reward_raw'], **common,
            note='Already received at this decision; belongs to the previous action interval, not the action selected now.')
        for component in ('wait','production','waste','grade_2h','grade_action','grade_shift','delta_wait_vehicle_seconds','wait_vehicle_seconds'):
            add(label,'received_previous_action_reward','RPC.'+component,components[component], **common)
        for component, values in (('joint_cost', targets.step_costs),('grade_cost',targets.grade_costs),
                                  ('production_cost',targets.production_costs),('wait_cost',targets.wait_costs)):
            add(label,'received_previous_action_current008_rebuild',component,values[incoming],
                current008_cost=values[incoming], **{**common,'data_status':'frozen008_physical_rebuild'},
                note='Current training cost reconstructed from the recorded completed interval; negative is cost-equivalent reward, not actual RPC reward.')
        # Window status uses the canonical event sequence, including same-time ordering.
        for settlement in episode['memory'].settlements:
            known = settlement.seq < marker.seq
            add(label,'known_settled_window' if known else 'unsettled_window','window_status',
                'settled' if known else 'unsettled', window=settlement.window,
                available_at_snapshot=known, available_at_time=settlement.time,
                data_status='known_settlement' if known else 'final_grade_unknown_at_snapshot',
                note='Window closure is known only after its canonical WINDOW_CLOSE sequence. Unsettled final cost is intentionally blank.')
            cost = grade_cost(settlement.grade if settlement.grade_valid else None, empty=not settlement.grade_valid)
            add(label,'known_settled_window' if known else 'future_actual_settlement_audit_only',
                'settled_window_grade',settlement.grade if settlement.grade_valid else None,
                window=settlement.window, window_ore_tons=settlement.tons, window_metal_tons=settlement.metal,
                raw_window_grade_reward=settlement.grade_reward_raw, current008_cost=cost,
                available_at_snapshot=known, available_at_time=settlement.time,
                data_status='recorded_WINDOW_CLOSE',
                note='Future actual outcome is audit-only; neither future grade nor future cost enters graph, quantities or C128.')
        add(label,'production_not_yet_settled','production_cost_status','not_settled_until_END',
            available_at_snapshot=True, available_at_time=marker.time, data_status='contract_fact',
            note='Reserved or carried tons are not processed ore/delivered waste. No intermediate shortfall charge is added.')
        add(label,'cumulative_known_cost','settled_grade_cost',details['cumulative_settled_grade_cost'],
            available_at_snapshot=True, available_at_time=marker.time, data_status='past_settlements_only')
        add(label,'cumulative_known_cost','waiting_cost',details['cumulative_wait_cost'],
            available_at_snapshot=True, available_at_time=marker.time, data_status='observed_fleet_wait_only')

    # The T0 action result is delivered at T1. It is unknown at T0, known at T1.
    t0_index = indices['T0']
    interval_info = infos[t0_index+1]
    interval_events = [event.to_dict() for event in episode['memory'].events
                       if markers[t0_index].seq < event.seq < markers[t0_index+1].seq]
    for label in ('T0','T1'):
        add(label,'T0_action_to_T1_result','RPC_transition_reward_raw',rows[t0_index]['reward'],
            transition_index=t0_index, decision_sequence=interval_info['decision_sequence'],
            interval_start=markers[t0_index].time, interval_end=markers[t0_index+1].time,
            raw_rpc_reward=rows[t0_index]['reward'], current008_cost=targets.step_costs[t0_index],
            available_at_snapshot=label=='T1', available_at_time=markers[t0_index+1].time,
            data_status='actual_RPC_record_and_frozen008_rebuild',
            note='Same recorded interval, shown from each snapshot visibility boundary. No per-truck causal attribution is implied.')
    t1_index = indices['T1']
    add('T1','future_T1_action_result_audit_only','RPC_transition_reward_raw',rows[t1_index]['reward'],
        transition_index=t1_index, interval_start=markers[t1_index].time, interval_end=markers[t1_index+1].time,
        raw_rpc_reward=rows[t1_index]['reward'], current008_cost=targets.step_costs[t1_index],
        available_at_snapshot=False, available_at_time=markers[t1_index+1].time,
        data_status='future_record_excluded_from_inputs', note='Not yet received when T1 action is selected.')
    current_transition = dict(transition_index=t0_index, action=rows[t0_index]['action'], truck_id=rows[t0_index]['truck_id'],
        from_time=markers[t0_index].time, to_time=markers[t0_index+1].time,
        raw_rpc_reward=rows[t0_index]['reward'], delta_wait_vehicle_seconds=interval_info['reward_components']['delta_wait_vehicle_seconds'],
        current008_cost=dict(joint=targets.step_costs[t0_index],grade=targets.grade_costs[t0_index],
            production=targets.production_costs[t0_index],wait=targets.wait_costs[t0_index]),
        closed_windows=interval_info['closed_windows'], material_events=interval_events)
    after = {path: sha(path) for path in before}
    assert before == after, 'read-only input changed during audit'
    details = dict(schema='bg009-historical-snapshot-reward-audit-v1',
        execution=dict(new_physical_shifts=0,optimizer_steps=0,model_forwards=0,rpc_starts=0,budget_registrations=0),
        frozen008_contract=composite_contract(), snapshot_selection=str(args.selection),
        raw_rpc_contract='dispatch-production-wait-shortfall-v2: wait+production+waste; grade appears in separate window ledger',
        snapshots=snapshot_details, T0_to_T1=current_transition, historical_32_shifts=run_audits,
        validation=dict(shifts=len(run_audits),transitions=len(costs_all),
            step_costs_exactly_equal_frozen008=True,mc_returns_exactly_equal_frozen008=True,
            max_step_cost_error=float(np.max(np.abs(costs_all-frozen['costs']))),
            max_mc_error=float(np.max(np.abs(mc_all-frozen['mc_returns']))),
            max_raw_rpc_additive_error=max_raw_error,max_wait_conservation_error=max_wait_error,
            max_raw_material_state_error=max_state_error,all_input_sha256_unchanged=before==after,
            protected_core_sha256=after[str(PROTECTED)]),
        missing_fields=[], limitations=[
            'No truck-level decomposition or causal attribution exists for aggregate fleet waiting or window grade cost.',
            'A pair of decision states does not reveal exact loading completion times; only interval bounds are available unless an event was recorded.',
            'Unsettled window final grade/cost and end-of-shift production were unknown at T0/T1 and are audit-only outcomes.',
            'Original observations are float32; displayed decoded tons/time can differ slightly from float64 RPC metadata.',
            'Material journal records UNLOAD/PROCESS/WINDOW_CLOSE/DECISION/END, not exact future ETA or a causal reward allocation to reservation edges.'],
        source_sha256_before=before,source_sha256_after=after)
    args.output.mkdir(parents=True,exist_ok=True)
    all_fields = sorted({key for row in csv_rows for key in row})
    with (args.output/'reward_audit.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=all_fields);writer.writeheader();writer.writerows(csv_rows)
    (args.output/'reward_details.json').write_text(json.dumps(details,ensure_ascii=False,indent=2,allow_nan=False,default=json_default)+'\n')
    a,b=snapshot_details['T0'],snapshot_details['T1']
    future=[s for s in episode['memory'].settlements if s.seq>=markers[indices['T1']].seq]
    report=f'''# 历史相邻决策 Reward 与当前 008 成本审计

本脚本只读原始 RPC HDF5、sealed material HDF5 和已经冻结的训练目标；模型 forward、optimizer 更新、RPC、AnyLogic 启动、预算登记均为 0。输入 SHA256 前后全部一致，受保护 PPO 核心为 `{after[str(PROTECTED)]}`。

## 数据与时间归属

使用 TQ128 `train_u0016` 完整 32 班、{len(costs_all)} 个 transition。选中同班 `run_{run_selected}` 原始状态 {indices['T0']}→{indices['T1']}，精确 RPC 时间 {a['time_seconds']:.12f}→{b['time_seconds']:.12f} 秒；对应 float32 观测解码时间分别为 {a['encoded_time_seconds']:.12f}、{b['encoded_time_seconds']:.12f} 秒。时间排序和即时奖励归属使用 RPC 与 canonical material event seq，原始 float32 输入不作修正。

|项目|T0|T1|
|---|---:|---:|
|当时收到的上一动作 RPC reward|{a['received_rpc_reward']:.12g}|{b['received_rpc_reward']:.12g}|
|该已完成区间的当前 008 成本|{a['incoming_current008_cost']['joint']:.12g}|{b['incoming_current008_cost']['joint']:.12g}|
|累计已知 fleet 等待秒|{a['accrued_fleet_wait_seconds']:.9f}|{b['accrued_fleet_wait_seconds']:.9f}|
|已结算品位成本累计|{a['cumulative_settled_grade_cost']:.12g}|{b['cumulative_settled_grade_cost']:.12g}|
|已结算窗口（0 起）|{a['closed_windows_before']}|{b['closed_windows_before']}|
|未结算窗口（0 起）|{a['unsettled_windows']}|{b['unsettled_windows']}|

T0 所选历史动作是 Truck{current_transition['truck_id']}→Load{current_transition['action']}。T0→T1 区间新增 fleet 等待 {current_transition['delta_wait_vehicle_seconds']:.12g} 秒，实际 RPC reward 为 **{current_transition['raw_rpc_reward']:.12g}**，当前 008 联合成本为 **{current_transition['current008_cost']['joint']:.12g}**（品位 {current_transition['current008_cost']['grade']:.12g}、生产 {current_transition['current008_cost']['production']:.12g}、等待 {current_transition['current008_cost']['wait']:.12g}）。该结果在 T0 未知，在 T1 收到。T1 自己所选动作的后续结果另列为 future audit-only，不能与当时已经收到的 reward 混用。

## 原始 Reward 与训练成本是两种已有口径

实际 RPC reward 由 `wait + production + waste` 相加，其中 `wait=-config.wait_kappa*delta_wait_vehicle_seconds`；本批 `wait_kappa=4e-5`，终态矿/废缺口权重是 4/4。`grade_2h/grade_action/grade_shift` 在该 RPC production contract 中严格为 0；实际窗口品位、矿量、金属量与旧 `grade_reward_raw` 保存在独立 window ledger 和 material WINDOW_CLOSE。这些字段全部存在并已核验，不能把原 RPC reward 直接取负称为当前 008 成本。

当前 008 冻结成本严格复用 `single_critic_task_query_v1.targets.build_composite_targets`：四窗口 `min(max(abs(G-.27)-.004,0)/.006,2)`，空窗 2；班末 `8*(1-Qore/16000)+ +8*(1-Qwaste/4000)+`；每区间 `.2*delta_fleet_wait/28800`。6000 吨仅是废石停派阈值，不是成本达产目标。32 班逐步成本与 MC 回报均与历史 `updates/0016/training_targets.h5` **逐元素完全相同（误差 0）**；原始 RPC 与 material 的 N+1 状态也逐元素相同。无奖励分项缺失，无新定义、无静默补值。

## 与业务承诺变化的对应范围

Truck 从预留待装转成已载，改变的是承诺阶段和在途物料事实；它本身不产生独立的品位或达产即时收费。当前品位按真实破碎后窗口结算，矿石达产按已破碎量，废石达产按实际已交付量，生产缺口只在真实 END 一次结算。预留吨位、已载吨位、卸到矿石缓冲区的吨位都不能直接充当完成矿石产量。

该相邻区间关闭窗口数为 {len(current_transition['closed_windows'])}，所以两图中明显的装载/卸载阶段变化与即时品位成本可以不同时出现。等待是全车队累计量，只能核验整个区间的等待成本，不能仅凭这对快照把它因果分摊给 Truck3/5/7 或某条业务边。CSV 对每行标记 `scope`、`available_at_snapshot`、`available_at_time`、`entered_snapshot_input=False`，尚未结算窗口的当时最终成本留空，后续实际结果在独立 `future_actual_settlement_audit_only` 行显示。

## 完整班次后验结果（不进入 T0/T1 输入）

选中班次实际品位成本 {selected['audit']['grade_cost']:.12g}、生产缺口成本 {selected['audit']['production_cost']:.12g}、等待成本 {selected['audit']['wait_cost']:.12g}，联合成本 **{selected['audit']['joint_cost']:.12g}**；矿石 {selected['audit']['ore_tons']:.9f} 吨、废石 {selected['audit']['waste_tons']:.9f} 吨、fleet 等待 {selected['audit']['fleet_wait_seconds']:.9f} 秒。未结算窗口共有 {len(future)} 个，其后续结算值仅用于成本核验。

## 无法从历史快照恢复的内容

- 单车、单条承诺边对未来窗口成本的因果贡献；原数据只记录聚合 reward/窗口成本。
- 两次决策之间没有逐项记录的精确装载完成时刻；快照阶段变化只能定位到区间内。
- T0/T1 当时尚未结算窗口的未来最终品位、终态产量缺口与实际未来 ETA。这些即使在完整历史中可见，也没有传入图、数量特征或 C128。
- 原始 float64 状态字段的完整精度；采样观测保留的是原始 float32 字节，精确事件时间另由 RPC 元数据提供。

逐项结果见 `reward_audit.csv`，32 班核验与完整输入哈希见 `reward_details.json`。本报告证明奖励记账及时间边界一致，不证明任何网络已经学会使用这些业务关系。
'''
    (args.output/'reward_findings.md').write_text(report)
    print(json.dumps(dict(output=str(args.output),validation=details['validation'],T0_to_T1={k:v for k,v in current_transition.items() if k!='material_events'}),ensure_ascii=False,default=json_default))


if __name__ == '__main__':
    main()
