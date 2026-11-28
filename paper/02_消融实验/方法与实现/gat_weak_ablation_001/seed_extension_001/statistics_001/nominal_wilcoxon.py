"""Compare the five trained Full policies on their shared 100 nominal RNG slots."""

from collections import Counter
from itertools import combinations
from pathlib import Path
import math
import numpy as np
from scipy.stats import binomtest, rankdata, wilcoxon

from analyze import FROZEN_SHA, OUTPUT, SEEDS, SUMMARY, read_frozen


def load_nominal():
    index = read_frozen(SUMMARY, FROZEN_SHA)
    reference, used = None, set()
    policies = {}
    for seed in SEEDS:
        source = index['sources'][seed]
        completed = read_frozen(**checksum_args(source['full_completion']))
        ref, = [item['report'] for item in completed['phase_results']
                if Path(item['report']['path']).parts[-3:-1] == ('nominal', 'new')]
        report = read_frozen(**checksum_args(ref))
        phase = read_frozen(**checksum_args(report['phase_config']))
        if not (report['scenario_id'] == 'nominal' and report['checkpoint_provenance'] == source['full_checkpoint']
                and completed['training_updates'] == 32 and report['evaluation_optimizer_updates'] == 0
                and report['training_state_unchanged'] and report['n_shifts'] == 100):
            raise ValueError('Incorrect evaluation arm or checkpoint')
        panel, physical = [], []
        for k, (row, job) in enumerate(zip(report['per_episode'], phase['jobs'], strict=True)):
            identity = row['identity']
            pair = identity['environment_seed'], identity['actor_sampling_seed']
            rid = identity['run_index']
            if (k, rid, *pair) != (job['pair_ordinal'], job['run_index'], job['experiment_seed'],
                                    job['actor_sampling_seed']) or rid in used:
                raise ValueError('Physical eval slot mismatch')
            panel.append((k, pair, job['expected_rpc_config']))
            used.add(rid)
            physical.append(row['physical'])
        if len(panel) != len({e[1] for e in panel}) or len(panel) != 100 or (reference and reference != panel):
            raise ValueError('Incomplete or unpaired test set')
        reference = panel
        policies[seed] = physical, ref
    return policies


def checksum_args(ref):
    return dict(path=ref['path'], checksum=ref['sha256'])


def signed_ranks(values):
    nonzero = values[values != 0]
    if not len(nonzero):
        return 0.0, 1.0
    # Twice the midrank is always an integer, including binary outcome ties.
    ranks = (2 * rankdata(np.abs(nonzero), method='average')).astype(int)
    positive = int(ranks[nonzero > 0].sum())
    w = min(positive, int(ranks.sum()) - positive)
    # Exact null lower tail via subset sums. Python integers are needed at n=100.
    ways = [1] + [0] * w
    for rank in ranks:
        for i in range(w, int(rank)-1, -1):
            ways[i] += ways[i-rank]
    p = min(1., 2 * sum(ways) / (2**len(ranks)))
    if len(set(ranks)) == len(ranks):
        # Check the lower tail directly to avoid cancellation in 1 - CDF.
        lower_tail = nonzero if positive == w else -nonzero
        check = wilcoxon(lower_tail, alternative='less', method='exact')
        assert math.isclose(2 * float(check.pvalue), p, rel_tol=1e-10)
    elif len(set(ranks)) == 1:
        assert math.isclose(binomtest(int(np.sum(nonzero > 0)), len(ranks)).pvalue, p)
    return w / 2, p


def holm(rows, key):
    ordered = sorted(rows, key=lambda row: row['p'])
    running = 0.0
    for rank, row in enumerate(ordered):
        running = min(1.0, max(running, row['p'] * (len(rows) - rank)))
        row[key] = running


def pformat(p):
    return f'{p:.4g}'


def main():
    policies = load_nominal()
    lines = ['# 混合训练 Full u32 标准工况百槽配对比较', '',
        '每个检查点在同一标称工况运行 100 班，按原始槽序 0–99、环境种子和 Actor 采样种子配对，'
        '并核对每槽 RPC 工况和 500 个物理运行身份。完整任务要求四窗品位均合规且矿废均达产。'
        '等待为每班全车队累计排队时长，已换算为车时。', '',
        '|种子|完整任务（班/100）|四窗合规（班/100）|矿达产（班/100）|废达产（班/100）|平均等待（车时/班）|矿石（吨/班）|废石（吨/班）|品位 RMSE（百分点）|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    data = {}
    failures = Counter()
    failed_windows = Counter()
    squared_deviations = []
    for seed, (shifts, _) in policies.items():
        def col(name):
            return np.asarray([r[name] for r in shifts], dtype=float)
        task, wait = col('complete_task'), col('total_wait_seconds')
        data[seed] = {'完整任务':task, '累计等待':wait}
        failures.update(np.flatnonzero(task==0).tolist())
        windows = [window for shift in shifts for window in shift['windows']]
        assert len(windows) == 400 and all(not window['empty'] for window in windows)
        deviations = [(100 * window['center_deviation']) ** 2 for window in windows]
        squared_deviations.extend(deviations)
        failed_windows.update(window['window'] + 1 for window in windows if not window['compliant'])
        lines.append(f"|{seed}|{task.sum():.0f}|{col('all_four_windows_compliant').sum():.0f}|"
            f"{col('ore_target_met').sum():.0f}|{col('waste_target_met').sum():.0f}|"
            f"{wait.mean()/3600:.3f}|{col('ore_tons').mean():,.1f}|{col('waste_tons').mean():,.1f}|"
            f"{math.sqrt(np.mean(deviations)):.3f}|")
    lines.extend(['',f'五个模型在 {100-len(failures)} 个槽均成功。单模型失败的槽有 '
                  f'{sum(n == 1 for n in failures.values())} 个，两模型失败的槽有 '
                  f'{sum(n == 2 for n in failures.values())} 个。',
        f'实际破碎窗口品位相对目标中心的总体 RMSE 为 {math.sqrt(np.mean(squared_deviations)):.3f} 个百分点。'
        f'{len(squared_deviations)} 窗中有 {sum(failed_windows.values())} 窗不合规，'
        f'其中第一窗 {failed_windows[1]} 个，第四窗 {failed_windows[4]} 个。','',
        '## 双侧 Wilcoxon 配对符号秩检验','',
        '每对比较的 100 个槽内差值按种子 A 减 B 计算。检验使用未舍入的数据，排除零差，'
        '绝对差并列取平均秩。W 为正、负秩和的较小者。按零假设下符号可交换，精确计算双侧概率。'
        '二元任务在不一致槽上的差值均为 ±1，此时等价于精确 McNemar 检验。'
        '每项指标的十对比较先分别用 Holm 法校正，同时报告两个指标共二十次的 Holm 敏感性分析。'
        '推断条件为这五个已训练的策略和此标称工况下的独立随机槽。'
        '符号秩检验考察配对差是否围绕零对称。[检验定义及并列值说明](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html)'])
    tests = []
    for endpoint in ('累计等待','完整任务'):
        rows = []
        for a,b in combinations(SEEDS,2):
            d = data[a][endpoint] - data[b][endpoint]
            w,p = signed_ranks(d)
            scale = 3600 if endpoint=='累计等待' else .01
            rows.append(dict(pair=f'{a} / {b}',endpoint=endpoint,plus=int(np.sum(d>0)),minus=int(np.sum(d<0)),
                             n=int(np.count_nonzero(d)),w=w,p=p,mean=float(d.mean()/scale),median=float(np.median(d)/scale)))
        holm(rows,'p10')
        tests.extend(rows)
    holm(tests,'p20')
    for endpoint, unit in [('累计等待','车时/班'),('完整任务','百分点')]:
        lines.extend(['',f'### {endpoint}','',
            f'|A / B|A > B 槽数|A < B 槽数|非零槽数|差均值（{unit}）|差中位数（{unit}）|W|原始 p|Holm（10）|Holm（20）|',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|'])
        for v in tests:
            if v['endpoint'] == endpoint:
                lines.append(f"|{v['pair']}|{v['plus']}|{v['minus']}|{v['n']}|{v['mean']:+.3f}|{v['median']:+.3f}|"
                             f"{v['w']:g}|{pformat(v['p'])}|{pformat(v['p10'])}|{pformat(v['p20'])}|")
    lines.extend(['','## 封存的逐班来源',''])
    for seed,(_,ref) in policies.items():
        lines.append(f"- [{seed} 的 nominal/new 固定检查点评测]({ref['path']})，SHA256 `{ref['sha256']}`。")
    OUT = OUTPUT / 'full_nominal_wilcoxon.md'
    OUT.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(OUT)


if __name__ == '__main__':
    main()
