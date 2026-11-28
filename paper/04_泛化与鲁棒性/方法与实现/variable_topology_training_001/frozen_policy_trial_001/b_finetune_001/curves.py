"""Plot and report the audited B-specific fine-tune and matched B50 rerun."""

import math
from pathlib import Path
import subprocess
from xml.etree import ElementTree as ET


NAVY, AMBER, TEAL = '#254e78', '#ae521d', '#227669'
INK, MUTED, GRID = '#172d40', '#465c6b', '#d8e0e5'


def element(parent, tag, **attrs):
    return ET.SubElement(parent, tag, {k.replace('_', '-'): str(v) for k, v in attrs.items()})


def label(parent, x, y, value, *, size=15, color=INK, anchor='start', bold=False):
    item = element(parent, 'text', x=x, y=y, font_size=size, fill=color, text_anchor=anchor,
                   font_weight='bold' if bold else 'normal')
    item.text = str(value)


def line(parent, x1, y1, x2, y2, color, width=1):
    element(parent, 'line', x1=x1, y1=y1, x2=x2, y2=y2, stroke=color, stroke_width=width)


def stroke_series(parent, points, color, hollow=False):
    element(parent, 'polyline', points=' '.join(f'{x:.2f},{y:.2f}' for x, y in points),
            fill='none', stroke=color, stroke_width=2 if hollow else 2.8,
            stroke_dasharray='6 4' if hollow else 'none')
    for x, y in points:
        if hollow:
            element(parent, 'rect', x=x-6, y=y-6, width=12, height=12,
                    fill='none', stroke=color, stroke_width=2)
        else:
            element(parent, 'circle', cx=x, cy=y, r=3.8, fill=color, stroke=color)


def value_ticks(values, *, nonnegative=False):
    if any(not math.isfinite(v) for v in values) or (nonnegative and min(values) < 0):
        raise ValueError('Loss values are invalid')
    low, high = min(0, *values), max(0, *values)
    if low == high:
        return [0, .25, .5, .75, 1] if nonnegative else [-1, -.5, 0, .5, 1]
    raw = (high - low) / 4
    power = 10 ** math.floor(math.log10(raw))
    step = next(s * power for s in (1, 2, 2.5, 5, 10) if s >= raw / power)
    return [i * step for i in range(math.floor(low / step), math.ceil(high / step) + 1)]


def panel(group, left, top, title, subtitle, indices, ticks, series, *, highlight=None):
    label(group, left, top, title, size=20, bold=True)
    label(group, left, top+24, subtitle, size=14, color=MUTED)
    x0, x1, y0, y1 = left+64, left+477, top+48, top+190
    xs = [x0 + 12 + i*(x1-x0-24)/(len(indices)-1) for i in range(len(indices))]

    def ordinate(v):
        return y1 - (v - ticks[0])*(y1-y0)/(ticks[-1]-ticks[0])

    if highlight is not None:
        mid = xs[indices.index(highlight)]
        element(group, 'rect', x=mid-10, y=y0-5, width=20, height=y1-y0+10, fill='#e0eeea')
    for tick in ticks:
        y = ordinate(tick)
        line(group, x0, y, x1, y, '#97aab4' if tick == 0 else GRID)
        label(group, x0-8, y+5, f'{tick:.3g}', size=13, color=MUTED, anchor='end')
    for x, u in zip(xs, indices):
        label(group, x, y1+24, f'u{u}', size=14,
              bold=(u == highlight), color=TEAL if u == highlight else MUTED, anchor='middle')
    for values, color, hollow in series:
        if len(values) != len(xs) or any(not ticks[0] <= v <= ticks[-1] for v in values):
            raise ValueError(f'Incomplete or out-of-range points for {title}')
        stroke_series(group, list(zip(xs, map(ordinate, values))), color, hollow)


def rates(rows, field):
    return [100 * r[field] / r['n'] for r in rows]


def figure(out_dir, train, val, selected):
    svg = ET.Element('svg', xmlns='http://www.w3.org/2000/svg', width='1100', height='627',
                     viewBox='0 0 1100 627', role='img')
    ET.SubElement(svg, 'title').text = 'Topology B fleet-dispatch adaptation'
    ET.SubElement(svg, 'desc').text = ('Training and validation shift pass rates. The shaded column identifies '
                                       'the selected validation checkpoint. Lower panels show the PPO losses.')
    element(svg, 'rect', width=1100, height=627, fill='white')
    g = element(svg, 'g', font_family='Liberation Sans, Arial, sans-serif')
    label(g, 36, 42, 'Topology B fleet-dispatch fine-tuning', size=29, bold=True)
    stroke_series(g, [(39, 72), (60, 72)], NAVY)
    label(g, 73, 77, 'Joint shift pass', size=15)
    stroke_series(g, [(220, 72), (244, 72)], AMBER, True)
    label(g, 258, 77, 'All four grade windows pass', size=15)
    element(g, 'rect', x=505, y=63, width=16, height=16, fill='#e0eeea')
    label(g, 529, 77, f'Selected validation checkpoint (u{selected})', size=15)
    for left, rows, name, note, chosen in [
        (36, train, 'Training shift outcomes', 'Pass rate (%)  ·  32 sampled shifts used for each update', None),
        (564, val, 'Paired validation outcomes', 'Pass rate (%)  ·  16 shared seeds per checkpoint, greedy dispatch', selected)
    ]:
        panel(g, left, 126, name, note, [r['update'] for r in rows], [0, 25, 50, 75, 100],
              [(rates(rows, 'joint'), NAVY, False), (rates(rows, 'four'), AMBER, True)], highlight=chosen)
    for left, field, name, note in [
        (36, 'actor_loss', 'Actor loss', 'PPO policy loss per optimizer update'),
        (564, 'critic_mse', 'Critic MSE', 'Mean squared value error per optimizer update')
    ]:
        values = [float(r[field]) for r in train]
        panel(g, left, 392, name, note, [r['update'] for r in train],
              value_ticks(values, nonnegative=field == 'critic_mse'), [(values, TEAL, False)])
    dest = out_dir / 'finetune_curves.svg'
    ET.ElementTree(svg).write(dest, encoding='utf-8', xml_declaration=True)
    subprocess.run(['ffmpeg', '-nostdin', '-loglevel', 'error', '-y', '-width', '2200',
                    '-i', str(dest), '-frames:v', '1', str(dest.with_suffix('.png'))], check=True)


def count(row, key):
    return f"{row[key]}/{row['n']}（{100*row[key]/row['n']:.1f}%）"


def report(s, train, val):
    sel, base, test, p = (s[k] for k in ('selection', 'baseline', 'test', 'paired'))
    u = sel['selected_update']
    winner = val[u]
    lines = [
        '# B 图微调与同种子复测', '',
        f"从动态出勤 u9 热启动（记为本地 u0），完成 4 次 PPO 更新。按 16 个共享验证槽选出 **u{u}**，"
        f"其联合达标率为 **{count(winner, 'joint')}**。在历史 B50 环境种子的配对复测中，"
        f"联合达标由 **{count(base, 'joint')}** 变为 **{count(test, 'joint')}**，差值为 **{p['delta_pp']:+.1f} 个百分点**。", '',
        '联合达标要求同一班同时满足矿石、废石和四个两小时品位窗的考核。训练沿用 B 图的班初运力缩放目标。', '',
        '## 更新曲线', '', '![B 图训练、验证及 PPO 损失曲线](finetune_curves.png)', '',
        '左上每一点对应一次更新所用的 32 个采样班。例如 u1 处的训练达标率来自 u0 采集的班次。'
        '右上各点是对应权重在同一组 16 个验证种子上的贪心派单结果。下排给出各次 PPO 更新的 Actor 损失与 Critic 均方误差。'
        '折线连接实测点。[矢量原图](finetune_curves.svg)', '',
        '| 更新 | 训练联合达标 | 训练四窗全达标 | 平均成本 | Actor 损失 | Critic MSE |',
        '|---|---:|---:|---:|---:|---:|'
    ]
    for row in train:
        lines.append(f"| u{row['update']} | {count(row, 'joint')} | {count(row, 'four')} |"
                     f" {row['mean_cost']:.4g} | {row['actor_loss']:.4g} | {row['critic_mse']:.4g} |")
    lines += ['', '## 验证选模', '',
              '以联合达标班数优先，依次用合规窗口总数、累计窗品位偏离幅度、较早更新打破平局。'
              '下表各窗单列的数字为该窗合规班数，每列分母均为 16。', '',
              '| 权重 | 联合达标 | 四窗全达标 | 0–2 h | 2–4 h | 4–6 h | 6–8 h | 品位累计偏离（百分点） |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in val:
        wins = ' | '.join(str(v) for v in row['window_pass'])
        lines.append(f"| u{row['update']} | {count(row, 'joint')} | {count(row, 'four')} | {wins} |"
                     f" {row['grade_excess_pp']:.3f} |")
    lines += ['', '## B50 配对复测', '',
              '两组均使用贪心派单，按相同历史环境种子逐槽对齐。每组 50 班，表中区间为各组联合达标率的 Wilson 95% 区间。', '',
              '| 模型 | 联合达标 | 95% 区间 | 四窗全达标 | 矿石达标 | 废石达标 | 矿石均值 ± SD（t） | 废石均值 ± SD（t） |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for title, row in [('历史动态 u9', base), (f'入选 u{u} 复测', test)]:
        lo, hi = row['joint_ci']
        lines.append(f"| {title} | {count(row, 'joint')} | {100*lo:.1f}%–{100*hi:.1f}% | {count(row, 'four')} |"
                     f" {count(row, 'ore')} | {count(row, 'waste')} | {row['ore_mean']:,.1f} ± {row['ore_sd']:,.1f} |"
                     f" {row['waste_mean']:,.1f} ± {row['waste_sd']:,.1f} |")
    lines += ['', f"同槽比较中，{p['both_pass']} 槽两次均达标，{p['improved']} 槽由未达标转为达标，"
              f"{p['regressed']} 槽由达标转为未达标，{p['both_fail']} 槽两次均未达标。"
              f"联合达标的双侧精确 McNemar 检验 p = {p['exact_mcnemar_p']:.4g}。", '',
              f"本轮核验 {s['verified_shifts']} 个物理班、{s['total_actions']:,} 次派单动作。"
              f"入选权重位于 `{sel['checkpoint_path']}`，SHA256 为 `{sel['checkpoint_sha256']}`。", '']
    return '\n'.join(lines)


def write_figures_and_report(out_dir, summary):
    out = Path(out_dir)
    train, val = (sorted(summary[k], key=lambda r: r['update']) for k in ('training', 'validation'))
    selected = summary['selection']['selected_update']
    for rows, expected, n in [(train, range(1, 5), 32), (val, range(5), 16)]:
        if [r['update'] for r in rows] != list(expected) or any(r['n'] != n for r in rows):
            raise ValueError('Incomplete B training or validation cohorts')
    if selected not in range(5) or any(summary[k]['n'] != 50 for k in ('baseline', 'test')):
        raise ValueError('Incomplete selected checkpoint or B50 cohort')
    markdown = report(summary, train, val)
    out.mkdir(parents=True, exist_ok=True)
    figure(out, train, val, selected)
    (out / 'finetune_report.md').write_text(markdown, encoding='utf-8')
