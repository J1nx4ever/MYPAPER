"""Render the settled paired-severity results without rerunning inference or tests."""
import json
import math
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile


POLICIES = {'full': 'BG128 Full', 'flow': '品位修订路径流量匹配', 'flat': 'Flat-PPO'}
DURATIONS = (0, 30, 60)
ENDPOINTS = {'joint': '联合达标率（百分点）', 'wait': '累计等待（车·小时）', 'grade': '品位标准差（百分点）'}
FAMILIES = {'within_joint': '策略内达标率', 'within_wait': '策略内等待', 'within_grade': '策略内品位波动',
            'interaction': '30 至 60 分钟的策略差变化', 'continuous_impacts': '相对对照的增量差'}


def number(value, digits=2, signed=False):
    if value is None:
        return '—'
    if abs(value) < .5 * 10**-digits:
        value = 0.
    return format(value, ('+' if signed else '') + f'.{digits}f')


def effect(row, key='mean'):
    value = row.get(key)
    scale = 100 if row['endpoint'] == 'joint' else 1
    digits = 1 if row['endpoint'] == 'joint' else 3
    if value is None:
        return '—'
    if key == 'ci':
        return '［' + '，'.join(number(v * scale, digits) for v in value) + '］'
    return number(value * scale, digits, signed=True)


def pvalue(p):
    return '—' if p is None else ('<0.0001' if p < .0001 else f'{p:.4f}')


def table(headings, rows):
    return ['| ' + ' | '.join(str(x) for x in row) + ' |' for row in
            (headings, ['---'] * len(headings), *rows)]


def contrast_name(row):
    terms = row['terms']
    first, duration, _ = terms[0]
    if row['family'].startswith('within_'):
        return f'{POLICIES[first]}，{duration} 分钟相对无停工'
    second = terms[1][0]
    reference = terms[2][1]
    return f'Full − {POLICIES[second]}，{duration} 分钟相对 {reference} 分钟'


def markdown(report):
    cells = {(r['strategy'], r['duration_minutes']): r for r in report['descriptive']}
    contrasts = {(r['name'], r['endpoint']): r for r in report['contrasts']}
    primary = [contrasts[f'full-{b}@60', 'joint'] for b in ('flow', 'flat')]
    primary_counts = '、'.join(f'{POLICIES[p]} {cells[p, 60]["joint"]}/30' for p in POLICIES)
    h = ['# A 图高品位装载点短暂停工的配对试验', '',
         f'**60 分钟停工时的联合达标班数为 {primary_counts}。**']
    for row in primary:
        baseline = POLICIES[row['terms'][1][0]]
        significance = ('达到' if row['p_holm'] < .05 else '未达到') + '预设显著性水平'
        p = pvalue(row['p_holm'])
        h[-1] += (f' Full 相对 {baseline} 的配对达标率差为 {effect(row)} 个百分点，'
                  f'Holm 校正后 p {p if p.startswith("<") else "= " + p}，{significance}。')
    h += ['', '## 九组班次表现', '',
          'A 图每班 9 辆车、8 小时。30 个新环境种子各覆盖三种策略与三种工况，'
          '共 270 个独立物理班。0 分钟为同种子无停工对照，30 和 60 分钟均从班内第 4 小时起关闭 0 号高品位装载点。'
          'BG128 Full 冻结 s02/u32，Flat-PPO 冻结 s01/u32，两者均选合法动作中得分最高者。', '']
    runs = []
    for strategy in POLICIES:
        for duration in DURATIONS:
            c = cells[strategy, duration]
            lo, hi = c['joint_ci']
            wait = contrasts.get((f'{strategy}:{duration}-0', 'wait'))
            grade = contrasts.get((f'{strategy}:{duration}-0', 'grade'))
            runs.append([POLICIES[strategy], duration,
                         f'{c["joint"]}/{c["n"]}（{100*c["joint"]/c["n"]:.1f}%）',
                         f'{100*lo:.1f}%–{100*hi:.1f}%', f'{c["four"]}/{c["n"]}',
                         f'{number(c["mean_wait"])}（{effect(wait) if wait else "参照"}）',
                         f'{number(c["mean_grade_sd"], 3)}（{effect(grade) if grade else "参照"}）', c['grade_n']])
    h += table(['策略', '停工（分）', '联合达标', '达标率 95% 区间', '四窗全合规',
                '平均等待车·小时（相对对照）', '平均品位 SD 百分点（相对对照）', '品位有效班'], runs)
    h += ['', '联合达标要求破碎矿石、废石均满足既定业务目标及容差，且四个固定两小时窗口的破碎矿石品位均在 26%–28%。'
          '等待量累计全车队装卸排队时间。品位 SD 先按每窗破碎矿石金属量除以矿石量计算窗口品位，再对四窗求班内样本标准差，最后取跨班平均。'
          '任一窗无矿的班次记为联合未达标，连续品位指标仅用有效班。括号内的变化按两种工况均有效的同种子班次配对计算。', '',
          '![各停工时长下的联合达标率、相对无停工的等待增量和班内品位波动](severity_comparison.svg)', '',
          '## 60 分钟停工的主要比较', '',
          '每个数对来自同一环境种子。“仅 Full 达标”与“仅基线达标”是精确 McNemar 检验使用的不一致对。', '']
    h += table(['比较', '配对数', '仅 Full 达标', '仅基线达标', '达标率差（百分点）', '95% 区间', '原始 p', 'Holm p'],
               [[f'Full − {POLICIES[r["terms"][1][0]]}', r['n'], r['gains'], r['losses'], effect(r),
                 effect(r, 'ci'), pvalue(r['p']), pvalue(r['p_holm'])] for r in primary])
    h += ['', '## 从 30 分钟延长至 60 分钟', '',
          '下表将 60 分钟下的“Full 减基线”减去 30 分钟下的同一差值。达标率差的正值表示相对优势增加，'
          '等待和品位 SD 差的负值表示 Full 相对增加得更少。', '']
    interactions = [r for r in report['contrasts'] if r['family'] == 'interaction']
    h += table(['比较基线', '指标', '配对数', '差值的变化', '95% 区间', 'Holm p'],
               [[POLICIES[r['terms'][1][0]], ENDPOINTS[r['endpoint']], r['n'], effect(r), effect(r, 'ci'),
                 pvalue(r['p_holm'])] for r in interactions])
    confirmed = [r for r in interactions if r.get('p_holm') is not None and r['p_holm'] < .05]
    if not confirmed:
        h += ['', '在预设的六项交互比较中，未检出经 Holm 校正后显著的变化。']
    h += ['', '## 次要检验', '']
    significant = [r for r in report['contrasts'] if r['family'] != 'primary60'
                   and r.get('p_holm') is not None and r['p_holm'] < .05]
    if significant:
        h += ['以下为各预设检验族中 Holm 校正后达到 0.05 水平的所有次要对比。', '']
        h += table(['检验族', '比较', '指标', '配对数', '差值', '95% 区间', 'Holm p'],
                   [[FAMILIES[r['family']], contrast_name(r), ENDPOINTS[r['endpoint']], r['n'],
                     effect(r), effect(r, 'ci'), pvalue(r['p_holm'])] for r in significant])
    else:
        h += ['各预设次要检验族均未检出 Holm 校正后显著的差异。']
    h += ['', '策略内比较对每种指标的六项检验分别做 Holm 校正，等待和品位的策略间增量比较共八项组成一族。'
          '60 分钟主要检验的两项比较与时长交互的六项比较各自组成一族。均为双侧检验。达标率采用 Wilson 区间，'
          '配对二元差值采用对不一致对概率的保守精确区间，其余差值采用 20,000 次按种子配对重采样的均值百分位区间。'
          '连续量采用配对符号翻转检验，非零对数超过 16 时使用 100,000 次固定种子抽样。所有 95% 区间均为逐项区间。'
          '完整的效应方向、有效对数、原始及校正 p 值见[逐项统计结果](summary.json)。', '',
          '## 产量、窗口和停工兑现情况', '']
    support = []
    means_available = all('mean_window_grade_percent' in c for c in cells.values())
    for strategy in POLICIES:
        for duration in DURATIONS:
            c = cells[strategy, duration]
            values = [POLICIES[strategy], duration, f'{c["ore_met"]}/{c["n"]}（{c["mean_ore_t"]:.0f}）',
                      f'{c["waste_met"]}/{c["n"]}（{c["mean_waste_t"]:.0f}）',
                      ' / '.join(map(str, c['window_pass']))]
            if means_available:
                values.append(' / '.join(number(v) for v in c['mean_window_grade_percent']))
            values.extend([c['suspended_active_shifts'], number(c['mean_paused_truck_hours']),
                           f'{c["mean_high_tons_4_to_5"]:.0f}'])
            support.append(values)
    titles = ['策略', '停工（分）', '矿达标班数（均量 t）', '废达标班数（均量 t）', '四窗各自合规班数']
    if means_available:
        titles.append('四窗各自平均品位（%）')
    h += table(titles + ['触发时正在装载的班数', '平均暂停车·小时', '第 4 至 5 小时高品位点平均装出量（t）'], support)
    h += ['', '## 配对与派车核验', '']
    mismatches = report['prefix_mismatch_slots']
    flow_slots = mismatches.get('flow', [])
    h += [f'两种神经策略在停工前三种工况的派单、业务事件与前两窗入账量逐槽一致。'
          f'路径流量匹配策略的停工前轨迹在 {len(flow_slots)}/30 个槽存在差异' +
          ('，槽号为 ' + '、'.join(map(str, sorted(flow_slots))) if flow_slots else '') +
          '。该策略按归档设置使用 800 ms 墙钟求解预算，按原分配保留全部槽参与比较。', '',
          '下表列出路径流量匹配策略内部调用合法备用分支的次数。停工时装载能力变为零，'
          '原求解器可能报告 `invalid_shovel_capacity`。同一次决策可同时进入多项计数。', '']
    diagnostics = []
    for duration in DURATIONS:
        phases = report['flow_diagnostics'][str(duration)]
        selections = [('total', '全班'), ('during_outage', '实际停工区间')] if duration else [('total', '全班')]
        for key, name in selections:
            counts = phases.get(key, {})
            n, fallback = counts.get('decisions', 0), counts.get('fallback_decisions', 0)
            rate = f'{100*fallback/n:.1f}%' if n else '—'
            capacity = counts.get('fallback_reason/invalid_shovel_capacity', 0)
            diagnostics.append([duration, name, n, f'{fallback}（{rate}）', counts.get('timed_out', 0), capacity])
    h += table(['停工（分）', '统计区间', '派车决策', '内部备用分支', '超时', '容量校验错误'], diagnostics)
    h += ['', f'逐班核验了矿石与金属质量守恒、停工屏蔽、复工可用性及策略身份。共核验 {report["verified_actions"]:,} 次决策，'
          f'原生仿真的非法派单为 {report["native_invalid"]} 次，原生替代派单为 {report["native_fallback"]} 次。'
          '结论适用于本次 A 图九车、指定停工时刻与所用冻结权重。[逐班核验记录](cases.json)', '']
    return '\n'.join(h)


def axis_limits(values):
    values = [v for v in values if v is not None] + [0.]
    low, high = min(values), max(values)
    step = (high - low) / 4 if low != high else .25
    unit = 10 ** math.floor(math.log10(step))
    tick = unit * next(s for s in (1, 2, 2.5, 5, 10) if s >= step/unit)
    lower, upper = math.floor(low / tick), math.ceil(high / tick)
    if lower == upper:
        upper = lower + 4
    return [v*tick for v in range(lower, upper+1)]


def vector_figure(report):
    # The host supplies ReportLab as a pure-Python system package on Debian.
    # Prefer the active Conda installation whenever present.
    try:
        import reportlab
    except ModuleNotFoundError:
        system_packages = Path('/usr/lib/python3/dist-packages')
        if not (system_packages/'reportlab').is_dir():
            raise
        sys.path.append(str(system_packages))
    from reportlab.graphics import renderSVG
    from reportlab.graphics.charts.lineplots import LinePlot
    from reportlab.graphics.shapes import Drawing, Line, Rect, String
    from reportlab.graphics.widgets.markers import makeMarker
    from reportlab.lib.colors import HexColor

    c = {(row['strategy'], row['duration_minutes']): row for row in report['descriptive']}
    contrasts = {(r['name'], r['endpoint']): r for r in report['contrasts']}
    drawing = Drawing(1040, 386)
    drawing.add(Rect(0, 0, 1040, 386, fillColor=HexColor('#ffffff'), strokeColor=None))
    dark = HexColor('#182c36')
    muted = HexColor('#415660')
    grid = HexColor('#d8e2e5')
    colors = [HexColor(k) for k in ('#006c74', '#ab4d12', '#6950a0')]
    marks = ['FilledCircle', 'FilledSquare', 'FilledDiamond']
    styles = [None, [6, 3], [2, 3]]

    def label(x, y, text, size=10, bold=False, where='start', color=dark):
        drawing.add(String(x, y, text, fontName='Helvetica-Bold' if bold else 'Helvetica',
                           fontSize=size, fillColor=color, textAnchor=where))

    def marker(i):
        m = makeMarker(marks[i])
        m.size, m.fillColor, m.strokeColor = 6, colors[i], colors[i]
        return m

    label(38, 357, 'Temporary closure of the high-grade loading site', 20, True)
    label(39, 335, 'Map A  /  Nine trucks  /  30 matched environment seeds per cell', 11, color=muted)
    titles = ['A   Joint compliance', 'B   Change in fleet waiting', 'C   Within-shift grade variability']
    units = ['Shifts meeting all targets (%)', 'Mean paired change vs. control (truck-hours)',
             'Mean SD of four window grades (pp)']
    for panel in range(3):
        x, y, w, h = 48 + 344*panel, 109, 262, 152
        label(x-10, 296, titles[panel], 14, True)
        label(x-10, 277, units[panel], 10, color=muted)
        series, whiskers = [], []
        for i, policy in enumerate(POLICIES):
            line, errors = [], []
            for d in DURATIONS:
                cell = c[policy, d]
                if panel == 0:
                    val, bounds = 100*cell['joint']/cell['n'], [100*v for v in cell['joint_ci']]
                elif panel == 1:
                    change = contrasts.get((f'{policy}:{d}-0', 'wait'))
                    val, bounds = (change['mean'], change['ci']) if change else (0., None)
                else:
                    val, bounds = cell['mean_grade_sd'], None
                if val is not None:
                    px = d + (i-1)*1.5
                    line.append((px, val))
                    if bounds:
                        errors.append((px, *bounds))
            series.append(line)
            whiskers.append(errors)
        limits = ([0.,25.,50.,75.,100.] if panel == 0 else axis_limits(
                  [v for line in series for _,v in line] + [v for group in whiskers for _,lo,hi in group for v in (lo,hi)]))
        plot = LinePlot()
        plot.x, plot.y, plot.width, plot.height = x, y, w, h
        segments = []
        for i, line in enumerate(series):
            for j, point in enumerate(line):
                if j == 0 or point[0] - line[j-1][0] > 31:
                    segments.append((i, []))
                segments[-1][1].append(point)
        plot.data = [segment for _, segment in segments]
        plot.xValueAxis.valueMin, plot.xValueAxis.valueMax = -7, 67
        plot.xValueAxis.valueSteps = list(DURATIONS)
        plot.xValueAxis.joinAxisMode = 'bottom'
        plot.yValueAxis.valueMin, plot.yValueAxis.valueMax = limits[0], limits[-1]
        plot.yValueAxis.valueSteps = limits
        plot.yValueAxis.labelTextFormat = lambda v: f'{v:g}'
        for axis in (plot.xValueAxis, plot.yValueAxis):
            axis.strokeColor, axis.strokeWidth = muted, .65
            axis.labels.fontName, axis.labels.fontSize, axis.labels.fillColor = 'Helvetica', 9, muted
        plot.yValueAxis.tickLeft = 3
        plot.xValueAxis.tickDown = 3
        to_y = lambda value: y + (value-limits[0])*h/(limits[-1]-limits[0])
        for level in limits:
            drawing.add(Line(x, to_y(level), x+w, to_y(level), strokeColor=grid, strokeWidth=.65))
        for i, intervals in enumerate(whiskers):
            for dur, lo, hi in intervals:
                pos = x+(dur+7)*w/74
                for x1, y1, x2, y2 in ((pos, to_y(lo), pos, to_y(hi)), (pos-2.4,to_y(lo),pos+2.4,to_y(lo)),
                                        (pos-2.4,to_y(hi),pos+2.4,to_y(hi))):
                    drawing.add(Line(x1, y1, x2, y2, strokeColor=colors[i], strokeWidth=.8))
        for line_index, (i, _) in enumerate(segments):
            plot.lines[line_index].strokeColor, plot.lines[line_index].strokeWidth = colors[i], 1.6
            plot.lines[line_index].strokeDashArray, plot.lines[line_index].symbol = styles[i], marker(i)
        if segments:
            drawing.add(plot)
        else:
            label(x+w/2, y+h/2, 'No evaluable shifts', where='middle', color=muted)
        label(x+w/2, 78, 'Outage duration (min)', where='middle', color=muted)

    for i, (x, text) in enumerate(zip([140, 395, 740],
                                      ['BG128 Full (s02/u32)', 'Grade-revised path-flow matching', 'Flat-PPO (s01/u32)'])):
        drawing.add(Line(x, 50, x+28, 50, strokeColor=colors[i], strokeWidth=1.8, strokeDashArray=styles[i]))
        m = marker(i)
        m.x, m.y = x+14, 50
        drawing.add(m)
        label(x+35, 47, text, 10)
    label(38, 19, 'Bars show pointwise 95% Wilson intervals (A) and paired bootstrap intervals (B). C shows descriptive means. '
          'Points are offset horizontally for visibility.', 9, color=muted)
    svg = renderSVG.drawToString(drawing)
    svg = svg.replace('<title>...</title>', '<title>High-grade loading-site closure on Map A</title>')
    return svg.replace('<desc>...</desc>', '<desc>Joint compliance, paired changes in fleet waiting, '
                       'and within-shift variability in crushed-ore grade during 0-, 30-, or 60-minute closures.</desc>')


def write_report(output_path, report):
    folder = Path(output_path)
    stored = json.loads((folder/'summary.json').read_bytes())
    if stored != json.loads(json.dumps(report)) or stored['new_physical_shifts'] != 270 or stored['environment_seeds'] != 30:
        raise ValueError('report requires the complete 270-shift settled summary')
    for name, content in [('severity_comparison.svg', vector_figure(stored)), ('summary.md', markdown(stored))]:
        path = folder/name
        data = content.encode('utf-8')
        if path.is_file() and path.read_bytes() == data:
            continue
        with NamedTemporaryFile(dir=folder, delete=False) as file:
            file.write(data)
        os.replace(file.name, path)
