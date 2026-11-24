"""Physical and decision-sampled comparison of the two fixed speed-variation arms."""

import math
from pathlib import Path

import numpy as np

from business_graph_summary_campaign_v1.plan_io import encoded, reference
from business_graph_summary_v1.contract import require
from online_v1_1.operations import _publish


SCENES = ('speed_std_015', 'speed_std_025')
STDS = (.15, .25)
PAIRS = 20
BOOTSTRAPS = 10000
BOOTSTRAP_SEED = 20261120
ENV_PREFIXES = ('config.runtime_scene.env.', 'config.env.')
TARGETS = ('complete_task', 'all_four_windows_compliant', 'ore_target_met', 'waste_target_met')
METRICS = ('total_wait_seconds', 'ore_tons', 'waste_tons',
           'resource_queue_mean_trucks', 'dump_capacity_queue_mean_trucks',
           'follow_or_brake_share', 'brake_gap_share', 'mean_sampled_moving_speed_mps',
           'mean_sampled_free_speed_ratio', 'sampled_free_speed_ratio_p90_p10')


def _env(rpc, name):
    values = {rpc[prefix + name] for prefix in ENV_PREFIXES if prefix + name in rpc}
    require(len(values) == 1, f'expected one consistent RPC value for {name}')
    return float(next(iter(values)))


def _known_mean(values):
    return float(np.mean(values)) if len(values) else None


def congestion_evidence(episodes, *, speed_tick_seconds=.2, follow_range_m=70., safe_gap_m=50.):
    """Reduce existing RPC observations to small per-shift traffic summaries.

    Use the newest eligible speed-tick observation for each moving vehicle at
    a dispatch decision. These are decision-sampled proxies for traffic
    exposure. No high-frequency event stream or interpolation is used.
    """
    require(0 < speed_tick_seconds and 0 < safe_gap_m < follow_range_m, 'invalid motion thresholds')
    rows = []
    for episode in episodes:
        states = np.asarray(episode['states'])
        require(states.dtype == np.float32 and states.shape == (len(episode['actions']) + 1, 1519)
                and len(states) > 1 and np.all(states[:-1, 17] == 0) and states[-1, 17] == 1,
                'traffic summary requires a complete physical observation sequence')
        decision_states = states[:-1]
        trucks = decision_states[:, 27:1107].reshape(-1, 9, 120)
        moving = (trucks[..., 45:61] == 1).any(axis=-1) & (trucks[..., 0] == 0)
        history = trucks[..., 92:120].reshape(-1, 9, 4, 7)
        ages = np.where(history[..., 0] == 1, history[..., 1] * 28800., np.inf)
        newest = np.take_along_axis(history, ages.argmin(axis=-1)[..., None, None], axis=2).squeeze(axis=2)
        fresh = moving & (ages.min(axis=-1) <= speed_tick_seconds + 1e-5)
        front_visible = fresh & (newest[..., 4] == 1) & (newest[..., 5] >= 0.)
        front_gap = newest[..., 5][front_visible].astype(np.float64) * 10000.
        denominator = int(fresh.sum())
        follow = int((front_gap < follow_range_m).sum())
        brake = int((front_gap < safe_gap_m).sum())
        free_ratio = trucks[..., 90][moving & (trucks[..., 89] > 0)].astype(np.float64)
        free_spread = float(np.quantile(free_ratio, .90) - np.quantile(free_ratio, .10)) if len(free_ratio) else None
        dumps = decision_states[:, 1295:1391].reshape(-1, 2, 48)
        capacity = np.count_nonzero(dumps[..., 30:39], axis=(1, 2))
        row = dict(
            run_index=episode['identity']['run_index'], decisions=len(decision_states),
            moving_truck_observations=int(moving.sum()), fresh_motion_observations=denominator,
            front_observations=int(front_visible.sum()), follow_or_brake_observations=follow,
            brake_gap_observations=brake,
            follow_or_brake_share=follow / denominator if denominator else None,
            brake_gap_share=brake / denominator if denominator else None,
            front_gap_p10_m=float(np.quantile(front_gap, .10)) if len(front_gap) else None,
            mean_sampled_moving_speed_mps=_known_mean(newest[..., 2][fresh].astype(np.float64) * 10.),
            sampled_free_speed_ratio_observations=len(free_ratio), mean_sampled_free_speed_ratio=_known_mean(free_ratio),
            sampled_free_speed_ratio_p90_p10=free_spread,
            dump_capacity_queue_mean_trucks=float(capacity.mean()),
        )
        rows.append(row)
    require(len({row['run_index'] for row in rows}) == len(rows), 'duplicate traffic evidence runs')
    return dict(schema='bg128-speed-decision-sampled-congestion-v1', per_episode=rows,
                speed_tick_seconds=speed_tick_seconds, follow_range_m=follow_range_m, safe_gap_m=safe_gap_m,
                sampling='at dispatch decisions, for each moving non-ego truck use its newest valid motion sample'
                         ' no more than one speed tick old',
                denominator='fresh moving-truck observations; adjacent dispatches may observe the same tick',
                time_weighted=False)


def _truncated_factor(sigma, lower, upper):
    a, b = (lower - 1.) / sigma, (upper - 1.) / sigma
    cdf_delta = (math.erf(b / math.sqrt(2.)) - math.erf(a / math.sqrt(2.))) / 2.
    density_a, density_b = (math.exp(-x * x / 2.) / math.sqrt(2. * math.pi) for x in (a, b))
    offset = (density_a - density_b) / cdf_delta
    variance = 1. + (a * density_a - b * density_b) / cdf_delta - offset ** 2
    return dict(input_std=sigma, lower=lower, upper=upper,
                theoretical_accepted_mean=1. + sigma * offset,
                theoretical_accepted_std=sigma * math.sqrt(variance))


def _index_phase(cfg, report, scene):
    require(cfg['scenario_id'] == report['scenario_id'] == scene, 'report phase order differs')
    require(report['optimizer_updates'] == report['gradient_updates'] == 0
            and report['training_state_unchanged'], 'speed evaluation must use a frozen policy')
    jobs = {job['pair_ordinal']: job for job in cfg['jobs']}
    require(len(cfg['jobs']) == len(jobs) == PAIRS and set(jobs) == set(range(PAIRS)),
            'expected all 20 frozen pair ordinals')
    observations = {row['identity']['run_index']: row for row in report['per_episode']}
    runs = {job['run_index'] for job in jobs.values()}
    require(len(report['per_episode']) == PAIRS and set(observations) == runs,
            'incomplete or unmatched evaluation outcomes')
    traffic = report.get('congestion_evidence')
    if traffic is None:
        return jobs, observations, {}
    semantics = {'speed_tick_seconds': 'speedDtSec', 'follow_range_m': 'followRangeM', 'safe_gap_m': 'safeGapM'}
    require(not traffic['time_weighted'] and all(
        traffic[key] == _env(job['expected_rpc_config'], env_name)
        for job in jobs.values() for key, env_name in semantics.items()),
        'traffic proxy thresholds must agree with the frozen RPC contract')
    by_run = {row['run_index']: row for row in traffic['per_episode']}
    require(len(traffic['per_episode']) == PAIRS and set(by_run) == runs, 'traffic evidence run coverage differs')
    return jobs, observations, by_run


def _outcome(report, job, row, traffic):
    identity, physical = row['identity'], row['physical']
    checks = dict(run_index=job['run_index'], environment_seed=job['experiment_seed'],
                  actor_sampling_seed=job['actor_sampling_seed'], recording_id=job['recording_id'],
                  scene_id=job['scene_id'], config_sha256=report['phase_config']['sha256'],
                  policy_sha256=report['actor_artifact']['sha256'])
    require(all(identity[key] == value for key, value in checks.items()), 'paired record identity differs')
    windows = physical['windows']
    require([window['window'] for window in windows] == [0, 1, 2, 3], 'four settled windows required')
    queue = physical['decision_sampled_queues']
    require(not queue['time_weighted'], 'expected decision-sampled resource queues')
    sites = queue['sites']
    require(set(sites) == {f'load.{i}' for i in range(4)} | {f'dump.{i}' for i in range(2)}
            and all(site['n'] == row['transitions'] for site in sites.values()),
            'resource queue site coverage differs')
    return dict(
        run_index=job['run_index'], recording_id=job['recording_id'], transitions=row['transitions'],
        **{key: physical[key] for key in (*METRICS[:3], *TARGETS)},
        windows=[{key: window[key] for key in ('window', 'tons', 'metal_tons', 'grade', 'empty', 'compliant')}
                 for window in windows],
        resource_queue_mean_trucks=math.fsum(site['mean'] for site in sites.values()),
        resource_queue_by_site=sites, **{key: traffic.get(key) for key in METRICS[4:]},
        congestion_evidence=traffic or None,
    )


def _paired_scalar(pairs, metric, rng):
    observations = [(pair['std_015'][metric], pair['std_025'][metric]) for pair in pairs]
    aligned = [(a, b) for a, b in observations if a is not None and b is not None]
    if not aligned:
        return dict(pairs=0)
    a, b = np.asarray(aligned, dtype=np.float64).T
    require(np.isfinite(a).all() and np.isfinite(b).all(), 'nonfinite paired outcome')
    delta = b - a
    draw = delta[rng.integers(0, len(delta), (BOOTSTRAPS, len(delta)))].mean(axis=1)
    result = dict(pairs=len(a), std_015=float(a.mean()), std_025=float(b.mean()),
                  delta_025_minus_015=float(delta.mean()), paired_bootstrap_95ci=np.quantile(draw, [.025, .975]).tolist(),
                  pairs_025_lower=int((delta < 0).sum()), pairs_025_higher=int((delta > 0).sum()),
                  ties=int((delta == 0).sum()))
    if metric in TARGETS:
        result.update(std_015_passed=int(a.sum()), std_025_passed=int(b.sum()),
                      both_pass=int(((a == 1) & (b == 1)).sum()),
                      both_fail=int(((a == 0) & (b == 0)).sum()))
    return result


def _grade_rows(pairs, rng):
    resample = rng.integers(0, len(pairs), (BOOTSTRAPS, len(pairs)))
    grades = np.asarray([[[window['grade'] if window['grade'] is not None else np.nan
                          for window in pair[arm]['windows']] for pair in pairs]
                         for arm in ('std_015', 'std_025')], dtype=np.float64)
    samples = np.where(np.isfinite(grades), (grades - .27) ** 2, 0.)
    counts = np.isfinite(grades)
    total_count = counts.sum(axis=(1, 2))
    rmse = [float(np.sqrt(samples[i].sum() / n)) if n else None for i, n in enumerate(total_count)]
    numerator = samples[:, resample].sum(axis=(2, 3))
    denominator = counts[:, resample].sum(axis=(2, 3))
    estimates = np.sqrt(np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0))
    diff = estimates[1] - estimates[0]
    valid = diff[np.isfinite(diff)]
    per_window = []
    for index in range(4):
        row = dict(window=index)
        for arm in ('std_015', 'std_025'):
            values = [pair[arm]['windows'][index] for pair in pairs]
            row[arm] = dict(compliant=sum(w['compliant'] for w in values),
                            empty=sum(w['empty'] for w in values),
                            lower=sum(w['grade'] is not None and w['grade'] < .26 - 1e-12 for w in values),
                            upper=sum(w['grade'] is not None and w['grade'] > .28 + 1e-12 for w in values))
        per_window.append(row)
    return dict(std_015=rmse[0], std_025=rmse[1], valid_windows=total_count.tolist(),
                delta_025_minus_015=rmse[1] - rmse[0] if None not in rmse else None,
                paired_bootstrap_95ci=np.quantile(valid, [.025, .975]).tolist() if len(valid) else None,
                valid_bootstrap_draws=len(valid), by_window=per_window)


def _markdown(summary):
    lines = ['# 固定 BG128/u32 速度扰动配对试验', '',
             '比较同一冻结 Actor 在两档自由速度因子输入标准差下的端到端结果。',
             '每档 20 个完整班次。逐槽匹配环境种子和 Actor 采样种子。', '',
             '## 主要结果', '',
             '| 指标 | 0.15 | 0.25 | 0.25 − 0.15 | 配对 95% 区间 |',
             '| --- | ---: | ---: | ---: | ---: |']
    labels = [('total_wait_seconds', '平均装卸等待（车·秒/班）'), ('ore_tons', '矿石（吨/班）'),
              ('waste_tons', '废石（吨/班）'), ('resource_queue_mean_trucks', '决策时资源队列车辆数'),
              ('dump_capacity_queue_mean_trucks', '决策时卸点容量队列车辆数'),
              ('follow_or_brake_share', '近距跟驰或制动代理（%）'),
              ('brake_gap_share', '低于安全间距代理（%）'),
              ('mean_sampled_moving_speed_mps', '决策时在途速度（m/s）'),
              ('mean_sampled_free_speed_ratio', '可见自由流速度比均值'),
              ('sampled_free_speed_ratio_p90_p10', '可见自由流速度比 P90−P10')]
    for key, label in labels:
        row = summary['metrics'][key]
        if not row['pairs']:
            continue
        scale = 100 if key.endswith('_share') else 1
        low, high = row['paired_bootstrap_95ci']
        lines.append(f"| {label} | {scale*row['std_015']:.2f} | {scale*row['std_025']:.2f} | "
                     f"{scale*row['delta_025_minus_015']:+.2f} | [{scale*low:+.2f}, {scale*high:+.2f}] |")
    lines += ['', '| 达标口径 | 0.15 | 0.25 | 仅 0.15 达标 | 仅 0.25 达标 |',
              '| --- | ---: | ---: | ---: | ---: |']
    for key, label in zip(TARGETS, ('完整任务', '四窗均合规', '矿石 ≥16,000 吨', '废石 ≥4,000 吨'), strict=True):
        row = summary['metrics'][key]
        lines.append(f"| {label} | {row['std_015_passed']}/20 | {row['std_025_passed']}/20 | "
                     f"{row['pairs_025_lower']} | {row['pairs_025_higher']} |")
    lines += ['', '| 窗口 | 0.15 合规 | 0.25 合规 | 0.15 低/高/空 | 0.25 低/高/空 |',
              '| --- | ---: | ---: | ---: | ---: |']
    for row in summary['grade']['by_window']:
        a, b = row['std_015'], row['std_025']
        violations = [' / '.join(str(value[key]) for key in ('lower', 'upper', 'empty')) for value in (a, b)]
        lines.append(f"| {row['window'] + 1} | {a['compliant']}/20 | {b['compliant']}/20 | {violations[0]} | {violations[1]} |")
    a, b = summary['physical_factors']
    rmse = summary['grade']
    if rmse['std_015'] is not None and rmse['std_025'] is not None:
        lines += ['', f"非空窗品位 RMSE 为 {100 * rmse['std_015']:.3f} 和 {100 * rmse['std_025']:.3f} 个百分点。"]
    lines += ['', '## 口径', '',
              f"0.15 和 0.25 是截断前正态分布的输入标准差。接受区间为 [{a['lower']:.2f}, {a['upper']:.2f}]。"
              f"该区间内接受因子的理论标准差分别为 {a['theoretical_accepted_std']:.4f} 和 {b['theoretical_accepted_std']:.4f}，"
              '实际路速还受车型、路段限速和车辆交互影响。',
              '主指标按班等权统计车辆从进入装卸等待到开始服务的累计时间，计入班末仍在等待的时间。'
              '路上跟驰和制动另用决策时已有观测作代理。'
              '这两项比例以可用在途车观测为分母，组间差以百分点计。可见自由流速度比使用当时车辆已观测的行程均值。',
              '95% 区间以整对班次重采样，每对保留两档及四个窗口。低/高/空窗均属于不合规。'
              '拒绝采样和后续轨迹可改变随机数的消耗顺序。区间描述同一训练种子下的测试不确定性。',
              '历史 0.05 标称独评仅作未配对背景。此次正式比较使用新班次。', '']
    return ('\n'.join(lines)).encode()


def _publish_once(path, data):
    if path.exists():
        require(path.read_bytes() == data, f'immutable speed report changed: {path}')
    else:
        _publish(path, data)


def publish_report(report_dir, cfgs, reports):
    """Publish a frozen 20-pair comparison from the two complete evaluation reports."""
    cfgs, reports = tuple(cfgs), tuple(reports)
    require(len(cfgs) == len(reports) == 2, 'exactly two speed arms required')
    actor_sha = reports[0]['actor_artifact']['sha256']
    require(reports[1]['actor_artifact']['sha256'] == actor_sha, 'speed arms require the same frozen Actor')
    indexes = [_index_phase(cfg, report, scene) for cfg, report, scene in zip(cfgs, reports, SCENES, strict=True)]
    require(bool(indexes[0][2]) == bool(indexes[1][2]), 'traffic evidence must have both arms or neither')
    pairs, physical_factors = [], []
    std_keys = {prefix + 'freeSpeedFactorStd' for prefix in ENV_PREFIXES}
    for slot in range(PAIRS):
        a, b = (index[0][slot] for index in indexes)
        require(a['run_index'] != b['run_index'] and a['recording_id'] != b['recording_id']
                and all(a[key] == b[key] for key in ('experiment_seed', 'actor_sampling_seed')),
                'paired jobs need shared seeds and independent physical identities')
        p, q = (job['expected_rpc_config'] for job in (a, b))
        require(p.keys() == q.keys() and {key for key in p if p[key] != q[key]} == std_keys.intersection(p),
                'RPC arms differ outside the free-speed standard deviation')
        case = []
        for job, rpc, std, index, report in zip((a, b), (p, q), STDS, indexes, reports, strict=True):
            require(_env(rpc, 'freeSpeedFactorStd') == std, 'frozen speed endpoint differs')
            require(_env(rpc, 'useEndogenousCongestion') == _env(rpc, 'useLocalDynamicSpeed') == 1.,
                    'both arms require the endogenous motion model')
            require((_env(rpc, 'freeSpeedFactorMin'), _env(rpc, 'freeSpeedFactorMax')) == (.85, 1.10),
                    'frozen free-speed truncation limits differ')
            run = job['run_index']
            case.append(_outcome(report, job, index[1][run], index[2].get(run, {})))
        pairs.append(dict(pair_ordinal=slot, environment_seed=a['experiment_seed'],
                          actor_sampling_seed=a['actor_sampling_seed'], std_015=case[0], std_025=case[1]))
    for std in STDS:
        physical_factors.append(_truncated_factor(std, .85, 1.10))
    require(len({pair['environment_seed'] for pair in pairs}) == PAIRS
            and len({pair['actor_sampling_seed'] for pair in pairs}) == PAIRS
            and len({pair[arm]['run_index'] for pair in pairs for arm in ('std_015', 'std_025')}) == 2 * PAIRS,
            'expected 20 unique paired slots and 40 unique physical runs')
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    summary = dict(schema='bg128-speed-congestion-paired-v1', pairs=pairs, paired_slots=PAIRS, complete_shifts=2*PAIRS,
                   actor_sha256=actor_sha, scenarios=[{key: report[key] for key in
                       ('scenario_id', 'phase_config', 'actor_artifact')} for report in reports],
                   primary_endpoint='total_wait_seconds', contrast='std_025_minus_std_015',
                   bootstrap=dict(unit='complete_shift_pair_with_all_four_windows', seed=BOOTSTRAP_SEED,
                                  replicates=BOOTSTRAPS, interval='unadjusted percentile 95%'),
                   physical_factors=physical_factors,
                   metrics={key: _paired_scalar(pairs, key, rng) for key in (*METRICS, *TARGETS)},
                   grade=_grade_rows(pairs, rng),
                   historical_nominal_std_005=dict(use='descriptive_only_unpaired',
                       report='runs/bg128-continuous-mixed-20261119-001/evaluation/nominal/new/reports/summary.json'))
    directory = Path(report_dir)
    directory.mkdir(parents=True, exist_ok=True)
    json_path, md_path = directory / 'paired_evaluation.json', directory / 'paired_evaluation.md'
    _publish_once(json_path, encoded(summary))
    _publish_once(md_path, _markdown(summary))
    return dict(json=reference(json_path), markdown=reference(md_path))
