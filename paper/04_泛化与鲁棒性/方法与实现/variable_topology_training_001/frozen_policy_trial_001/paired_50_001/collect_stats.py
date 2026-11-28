"""Replay the 100 independent native shifts and report fixed-denominator success rates."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import NormalDist, fmean, stdev
import sys
from types import SimpleNamespace

from prepare_stats import CASES, HERE, PROJECT, ledger_entry, load_case, settlement_session, witness

import collect_trial
from opening_targets import scaled_task
from resilience_runtime import atomic, record


REPORTS = PROJECT / 'reports' / HERE.relative_to(PROJECT / 'change_plans')
REFERENCE_TARGETS = dict(ore_target_tons=16000, waste_target_tons=4000,
                         waste_dispatch_stop_tons=6000, waste_upper_tons=6193)
RATE_NAMES = dict(joint='全部业务目标', four_windows='四窗品位均合规', ore='矿石达标', waste='废石达标',
                  **{f'window_{i}': f'第 {i} 窗品位合规' for i in range(1, 5)})


def opening_targets(group, config):
    scene = config['expected_runtime_scene']
    fleet = scene['opening_fleet']
    task = config['task']
    native = scene['exogenous_config']
    expected = scaled_task(REFERENCE_TARGETS, fleet)
    expected_count = 9 if group == 'A' else 7
    witness.require(sum(row[1] for row in fleet) == expected_count
                    and scene['layout_id'] == {'A': 'tree_A', 'B': 'loop_B'}[group],
                    'wrong topology or opening fleet for the group')
    witness.require(all(task[key] == value == native[key] for key, value in expected.items())
                    and native['ore_target_tons_2h'] == task['ore_target_tons'] / 4,
                    'frozen targets differ from the opening rated-payload scaling rule')
    witness.require(task['grade_lower'] == .26 and task['grade_upper'] == .28
                    and task['window_seconds'] == 7200, 'unexpected grade band or window length')
    witness.require(config['action_mode'] == 'argmax' and config['actor_sampling_seed'] is None
                    and native['deterministic'] is False and native['fault_truck_id'] == -1
                    and native['endogenous_congestion'] is False,
                    'wrong environment randomness or action mode')
    return dict(active_trucks=expected_count, ore_target_tons_2h=native['ore_target_tons_2h'], **expected)


def assess_one(argument):
    label, config, job, final = argument
    targets = opening_targets(label[0], config)
    actions = []

    def stream_with_actions(path, manifest):
        for event in collect_trial.iter_event_bundle(path, manifest):
            if event['event_type'] == 'ACTION_APPLIED':
                actions.append(event)
            yield event

    # Preserve the full FIFO, event-hash, wait and material replay. Retain only
    # dispatch records for the second cross-check, so the large stream is read once.
    replay = witness.bind(witness.replay_event_bundle, iter_event_bundle=stream_with_actions)
    native_tools = SimpleNamespace(native_manifest=witness.native_manifest, reference=witness.reference,
                                   replay_event_bundle=replay)
    assess = witness.bind(collect_trial.assess, witness=native_tools,
                          source_comparison=lambda label, outcomes, task: {})
    physical, native = assess(label, config, job, final)
    physical['opening_targets'] = targets
    policy_audit = witness.bind(collect_trial.policy_audit, iter_event_bundle=lambda path, manifest: actions)
    verify_policy = witness.bind(collect_trial.verify_policy, policy_audit=policy_audit)
    policy = verify_policy(config, job, native)
    if native is None:
        return physical, policy

    _, manifest, evidence = native
    material = evidence['material_totals']
    physical['fifo_balance_error_tons'] = {
        'ore': material['enqueued_tons'] - material['ore_tons'] - material['fifo_tons'],
        'metal': material['enqueued_metal_tons'] - material['metal_tons'] - material['fifo_metal_tons']}
    tolerance = config['task']['mass_balance_epsilon']
    physical['material_balance_verified'] = all(abs(error) <= tolerance
                                               for error in physical['fifo_balance_error_tons'].values())
    physical['invalid_dispatches'] = manifest['invalid_dispatches']
    physical['fallback_dispatches'] = manifest['fallback_dispatches']
    physical['event_count'] = evidence['event_count']
    if policy['status'] != 'VERIFIED':
        return physical, policy
    applied = [[event['data']['truck_id'], event['data']['action']] for event in actions]
    timed = [[event['sim_time_seconds'], *action] for event, action in zip(actions, applied)]
    policy['truck_action_sequence_sha256'] = sha256(witness.canonical(applied)).hexdigest()
    policy['timed_trajectory_sha256'] = sha256(witness.canonical(timed)).hexdigest()
    return physical, policy


def success_rate(k, n):
    z = NormalDist().inv_cdf(0.975)
    p = k / n
    den = 1 + z*z/n
    midpoint = (p + z*z/(2*n)) / den
    radius = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return dict(successes=k, planned=n, fraction=p,
                wilson_95=[0. if k == 0 else midpoint - radius, 1. if k == n else midpoint + radius])


def distribution(values):
    return dict(n=len(values), mean=fmean(values) if values else None,
                sample_sd=stdev(values) if len(values) > 1 else None,
                min=min(values) if values else None, max=max(values) if values else None)


def group_metrics(rows):
    witness.require(len(rows) == 50 and len({r['slot'] for r in rows}) == 50, 'expected 50 distinct group slots')
    statuses = Counter(r['physical_status'] for r in rows)
    complete = [r for r in rows if r['physical_status'] == 'COMPLETE']
    policy_matched = [r for r in complete if r['inference_audit']['status'] == 'VERIFIED']
    verified = [r for r in policy_matched if r['material_balance_verified']]
    targets = rows[0]['opening_targets']
    witness.require(all(r['opening_targets'] == targets for r in rows), 'group opening targets differ')
    passed = Counter()
    grades = [[] for _ in range(4)]
    for row in verified:
        passed['joint'] += row['task_status'] == 'MET'
        passed['four_windows'] += row['compliant_windows'] == 4
        passed['ore'] += row['ore_target_met']
        passed['waste'] += row['waste_target_met']
        for i, window in enumerate(row['windows']):
            if window['ore_tons'] > 0:
                grades[i].append(100 * window['metal_tons'] / window['ore_tons'])
                passed[f'window_{i+1}'] += window['cost'] == 0
    trajectories = {row['inference_audit']['timed_trajectory_sha256'] for row in verified}
    sequences = {row['inference_audit']['truck_action_sequence_sha256'] for row in verified}
    return dict(planned=50, physical_complete=statuses['COMPLETE'], verified=len(verified),
                unknown=statuses['UNKNOWN'], not_started=statuses['NOT_STARTED'],
                unverified_physical=statuses['UNVERIFIED'],
                complete_with_unverified_policy=len(complete)-len(policy_matched),
                policy_matched_with_failed_mass_balance=len(policy_matched)-len(verified),
                ledger_counts=dict(Counter(r['ledger_status'] for r in rows)), targets=targets,
                rates={key: success_rate(passed[key], 50) for key in RATE_NAMES},
                ore_tons=distribution([r['ore_tons'] for r in verified]),
                waste_tons=distribution([r['waste_tons'] for r in verified]),
                grade_percent_by_window=[distribution(values) for values in grades],
                distinct_timed_trajectories=len(trajectories), distinct_truck_action_sequences=len(sequences),
                observed_trajectory_variation=(len(trajectories) > 1 if len(verified) > 1 else None),
                verified_actions=sum(r['inference_audit']['decision_count'] for r in verified),
                fallback_dispatches=sum(r['fallback_dispatches'] for r in verified),
                invalid_dispatches=sum(r['invalid_dispatches'] for r in verified))


def human_report(summary):
    groups = summary['groups']
    lines = ['# 动态出勤 u9 在 A、B 两图的独立验证', '',
             '每图 50 个独立环境种子。策略固定取概率最高的合法动作。随机变化来自原生矿块和装卸作业。', '',
             '| 图 | 物理完整班 | 核验通过班次 | 物理状态待核实 | 尚未启动 | 核验派车数 | 不同派车轨迹 |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name, g in groups.items():
        lines.append(f"| {name} | {g['physical_complete']} | {g['verified']} | "
                     f"{g['unknown']+g['unverified_physical']} | {g['not_started']} | {g['verified_actions']} | "
                     f"{g['distinct_timed_trajectories']} |")
    lines += ['', '## 业务结果', '',
              '成功率的分母固定为每图 50 个计划班。分子仅计入物料回放和策略核验均通过的成功班。括号内为 Wilson 95% 区间。', '',
              '| 指标 | A 图 | B 图 |', '| --- | --- | --- |']
    for key, label in RATE_NAMES.items():
        cells = [label]
        for g in groups.values():
            r = g['rates'][key]
            low, high = r['wilson_95']
            cells.append(f"{r['successes']}/50，{100*r['fraction']:.1f}%（{100*low:.1f}%–{100*high:.1f}%）")
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines += ['', '## 产量与品位', '',
              '下表为有效观测的平均值 ± 样本标准差。各窗品位只统计当窗有矿石产出的班次。', '',
              '| 指标 | A 图 | B 图 |', '| --- | --- | --- |']
    for key, label in [('ore_tons','破碎矿石（t）'), ('waste_tons','卸达废石（t）'),
                       *[(i, f'第 {i+1} 窗品位（%）') for i in range(4)]]:
        cells = [label]
        for g in groups.values():
            d = g['grade_percent_by_window'][key] if isinstance(key, int) else g[key]
            if d['n'] == 0:
                cells.append('无有效班次')
            elif d['n'] == 1:
                cells.append(f"{d['mean']:.2f}（1 班）")
            else:
                cells.append(f"{d['mean']:.2f} ± {d['sample_sd']:.2f}（{d['n']} 班）")
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines += ['', '## 开班目标', '']
    for name, g in groups.items():
        t = g['targets']
        lines.append(f"- {name} 图出勤 {t['active_trucks']} 辆，矿石目标 {t['ore_target_tons']:,.2f} t，"
                     f"废石目标 {t['waste_target_tons']:,.2f} t，停派线 {t['waste_dispatch_stop_tons']:,.2f} t，"
                     f"单车容差计入后的上限 {t['waste_upper_tons']:,.2f} t。")
    lines += ['', '[逐班结果](cases.json) · [完整统计](summary.json)', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collect', action='store_true', help='settle physical evidence and publish reports')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--case', choices=CASES, help='inspect or settle only one case')
    parser.add_argument('--finalize-incomplete', action='store_true', help='settle an explicitly stopped partial case')
    args = parser.parse_args()
    if args.workers < 1 or args.finalize_incomplete and not (args.collect and args.case):
        parser.error('use positive workers; finalizing an incomplete case requires --collect --case')
    labels = [args.case] if args.case else CASES
    inputs = [(label, *load_case(label)) for label in labels]
    work = [(label, config, job, args.finalize_incomplete) for label, config, suite, job in inputs]
    rows = []
    parallel = args.workers > 1 and len(work) > 1
    with (ProcessPoolExecutor(max_workers=args.workers) if parallel else nullcontext()) as pool:
        results = pool.map(assess_one, work) if parallel else map(assess_one, work)
        with (settlement_session() if args.collect else nullcontext()) as settle:
            for (label, config, suite, job), (physical, audit) in zip(inputs, results, strict=True):
                if args.collect and physical['physical_status'] in ('COMPLETE', 'UNKNOWN'):
                    status = settle(suite, job, physical, collect=True)
                else:
                    status, _ = ledger_entry(suite, job)
                if status == 'UNKNOWN':
                    physical['physical_status'] = 'UNKNOWN'
                    audit = dict(status='UNVERIFIED', reason='physical run is sealed UNKNOWN')
                row = dict(physical, case=label, group=label[0], slot=int(label.split('_')[1]),
                           seed=job['experiment_seed'], ledger_status=status, inference_audit=audit)
                rows.append(row)
                if args.collect and status == 'COMPLETE':
                    record(Path(job['python_output'])/'policy_verification.json', audit, replace=True)
    if args.case:
        print(json.dumps(rows[0], ensure_ascii=False, indent=2))
    else:
        witness.require(len({r['seed'] for r in rows}) == len(rows) == 100, 'expected 100 unique environment seeds')
        groups = {name: group_metrics([r for r in rows if r['group'] == name]) for name in ('A', 'B')}
        summary = dict(planned=100, groups=groups, grade_interval_percent=[26, 28],
                       confidence_interval='Wilson 95%', unit='independent shift',
                       checkpoint_sha256=inputs[0][1]['checkpoint_sha256'])
        if args.collect:
            record(REPORTS/'cases.json', rows, replace=True)
            record(REPORTS/'summary.json', summary, replace=True)
            atomic(REPORTS/'summary.md', human_report(summary).encode(), replace=True)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if all(r['physical_status'] == 'COMPLETE' and r['inference_audit']['status'] == 'VERIFIED'
                    and r.get('material_balance_verified') is True
                    for r in rows) else 2


if __name__ == '__main__':
    sys.exit(main())
