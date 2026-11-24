"""Compact publications of the unchanged TQ008/BG009 evaluation metrics.

No model, simulator, optimizer or alternate reward is created here. The input
is the source-bound complete-shift evaluator's report, including failures.
"""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from business_graph_summary_v1.contract import ablation_contract, require
from business_graph_summary_v1.targets import composite_contract
from business_graph_summary_recovery_v1.plan_io import reference
from online_v1_1.operations import _publish as _atomic_publish


SCHEMA = 'business-graph-summary-independent-evaluation-summary-v1'
HEADLINE = (
    'grade_rmse', 'joint_four_window_compliance_rate',
    'ore_target_met_rate', 'waste_target_met_rate', 'complete_task_rate',
    'mean_total_wait_seconds', 'critic_mc_mse_pooled',
    'critic_mc_explained_variance_pooled',
)


def build_report(report, *, expected_complete_shifts=100):
    require(report.get('schema') == 'single-critic-task-query-joint-evaluation-v1',
            'frozen evaluation metric schema differs')
    require(report.get('role') == 'independent_evaluation' and
            report.get('training_state_unchanged') is True and
            report.get('composite_contract') == composite_contract(),
            'read-only same-cost evaluation report required')
    require(all(report.get(name) == 0 and type(report.get(name)) is int for name in
                ('optimizer_updates', 'gradient_updates', 'actor_updates', 'critic_updates')),
            'independent evaluation cannot optimize')
    require(report.get('representation', {}).get('graph_contract') == ablation_contract(),
            'frozen BG128 representation report required')
    records = report.get('per_episode', [])
    require(len(records) == report.get('n_shifts') == expected_complete_shifts and
            report.get('episodes') == records, 'complete-shift report inventory differs')
    require(all(row['identity']['split'] == 'validation' for row in records),
            'independent validation partition required')
    for key in ('run_index', 'environment_seed', 'actor_sampling_seed'):
        require(len({row['identity'][key] for row in records}) == len(records),
                'duplicate evaluation ' + key)
    require(all(len(row['physical']['windows']) == 4 for row in records),
            'all four windows, including failures, are required')
    head = report['headline_metrics']
    require(all(name in head for name in HEADLINE), 'eight declared metrics required')
    value = dict(schema=SCHEMA, n_shifts=len(records), n_states=report['n_states'],
        checkpoint_provenance=report.get('checkpoint_provenance'),
        headline_metrics=head, metric_names=list(HEADLINE),
        critic_mc=report['critic_mc'], composite_contract=composite_contract(),
        graph_contract=ablation_contract(),
        evaluation_thresholds=report['metrics']['evaluation_thresholds'],
        training_updates=0, optimizer_updates=0, gradient_updates=0,
        action_selection='stochastic_private_per_shift_generator',
        failed_shifts_retained=True,
        grade_rmse_denominator='all nonempty windows; empty windows separately counted',
        compliance_denominator='all four windows per complete shift; empty windows fail',
        critic_primary_weighting='all decision states pooled',
        uncertainty_scope='one preselected checkpoint from one training seed; no training-seed robustness claim')
    json.dumps(value, allow_nan=False)
    return value


def _publish(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == data, 'immutable evaluation publication differs: ' + str(path))
    else:
        _atomic_publish(path, data)
    return reference(path)


def _json(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2,
                       allow_nan=False) + '\n').encode()


def _csv(rows):
    require(bool(rows), 'nonempty evaluation CSV required')
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def write_reports(summary, report, folder):
    folder = Path(folder)
    shifts, windows = [], []
    for record in report['per_episode']:
        identity, p = record['identity'], record['physical']
        common = {name: identity[name] for name in
                  ('run_index', 'environment_seed', 'actor_sampling_seed')}
        row = dict(common, transitions=record['transitions'])
        row.update({name: p[name] for name in (
            'ore_tons', 'waste_tons', 'total_wait_seconds', 'grade_cost',
            'production_cost', 'waiting_cost', 'combined_cost',
            'all_four_windows_compliant', 'ore_target_met', 'waste_target_met',
            'complete_task')})
        row.update(critic_mc_mse=record['critic_mc']['mse'],
                   critic_mc_explained_variance=record['critic_mc']['explained_variance'])
        shifts.append(row)
        windows.extend(dict(common, **window) for window in p['windows'])
    head = summary['headline_metrics']
    lines = ['# BG128 固定 Checkpoint 独立评价', '',
        f"完整班次：{summary['n_shifts']}；决策数：{summary['n_states']}。",
        '采用固定策略的随机采样；训练、梯度和优化器更新均为 0。', '',
        '| 指标 | 数值 |', '| --- | --- |']
    labels = dict(grade_rmse='四窗口品位 RMSE',
        joint_four_window_compliance_rate='四窗口联合合规率',
        ore_target_met_rate='矿石达产率', waste_target_met_rate='废石达产率',
        complete_task_rate='完整任务完成率', mean_total_wait_seconds='平均每班累计等待秒数',
        critic_mc_mse_pooled='Critic MC MSE（决策等权）',
        critic_mc_explained_variance_pooled='Explained Variance（决策等权）')
    for name in HEADLINE:
        value = head[name]
        lines.append(f"| {labels[name]} | {'未定义' if value is None else format(value, '.8g')} |")
    lines += ['', '品位 RMSE 使用非空窗口；空窗单列并计为不合规。',
        '达产阈值为矿石 16000 吨、废石 4000 吨；6000 吨仅为废石停派阈值。',
        '保留所有完整班次，包括品位或生产失败班次；本轮独评不用于重新选择 Checkpoint。',
        '该结果只评价一个训练随机种子的固定策略。', '']
    return dict(summary=_publish(folder/'summary.json', _json(summary)),
        per_shift=_publish(folder/'per_shift.csv', _csv(shifts)),
        per_window=_publish(folder/'per_window.csv', _csv(windows)),
        readme=_publish(folder/'README.md', '\n'.join(lines).encode()))
