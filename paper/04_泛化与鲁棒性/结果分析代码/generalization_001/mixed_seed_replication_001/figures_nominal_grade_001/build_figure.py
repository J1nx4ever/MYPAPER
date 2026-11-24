"""Plot an audited nominal crusher-output grade trajectory, with no new runs."""
from bisect import bisect_right
from hashlib import sha256
from html import escape
import json
import math
from pathlib import Path
import sqlite3
from statistics import mean, median
import subprocess
import sys

HERE = Path(__file__).resolve().parent
PROJECT = next(folder for folder in HERE.parents if (folder / 'grade_event_memory_v1').is_dir())
sys.path.insert(0, str(PROJECT))
from grade_event_memory_v1.events import build_memory
from grade_event_memory_v1.journal import read_journal

RUN = PROJECT / 'runs/bg128-continuous-mixed-20261120-s02'
PHASE = RUN / 'evaluation/nominal/new'
STEM = 'crusher_output_grade'
ROLL_TONS = 1000.
WIDTH, HEIGHT = 1100, 680
LEFT, RIGHT = 94, 1055
INK, MUTED, GRID = '#1a3041', '#526675', '#dbe4e9'
ROLL_COLOR, WINDOW_COLOR, TARGET_COLOR = '#123a68', '#007d70', '#6d6471'
SOURCE_COLORS = {0: '#b65227', 1: '#7186a1', 2: '#16928a'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_bytes())


def reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha256(path.read_bytes()).hexdigest())


def sample():
    completed = read(RUN / 'suite_complete.json')
    report_path = PHASE / 'evaluation_complete.json'
    report_ref = reference(report_path)
    control, = [phase for phase in completed['phase_results'] if phase['report']['path'] == str(report_path)]
    require(completed['status'] == 'COMPLETE' and report_ref == control['report'], 'unsealed nominal report')
    report = read(report_path)
    require(report['scenario_id'] == 'nominal' and report['policy_arm'] == 'new'
            and report['training_state_unchanged'] and report['evaluation_optimizer_updates'] == 0
            and report['checkpoint_provenance'] == control['checkpoint']
            and reference(control['checkpoint']['path']) == control['checkpoint'], 'wrong evaluation contract')
    pool = report['per_episode']
    target = report['composite_contract']['grade_center']
    require(len(pool) == report['n_shifts'] == 100 and all(len(row['physical']['windows']) == 4
                and all(not window['empty'] and window['grade'] is not None
                        for window in row['physical']['windows']) for row in pool),
            'selection requires all 100 shifts and their four nonempty windows')

    def rmse(row):
        return math.sqrt(mean((window['grade'] - target) ** 2 for window in row['physical']['windows']))

    typical = median(map(rmse, pool))
    chosen = min(pool, key=lambda row: (abs(rmse(row) - typical), row['identity']['run_index']))
    identity = chosen['identity']
    index = identity['run_index']
    with sqlite3.connect((RUN / 'budget.sqlite').as_uri() + '?mode=ro', uri=True) as ledger:
        status, = ledger.execute('SELECT status FROM slots WHERE run=?', (index,)).fetchone()
    require(status == 'COMPLETE' and identity['policy_version'] == report['fingerprints']['actor'],
            'selected shift was not completed by the frozen Actor')
    receipt_path = PHASE / 'episodes' / f'run_{index}' / 'material_receipt.json'
    receipt = read(receipt_path)
    journal_ref = receipt['references']['journal']
    require(receipt['normal_completion'] and receipt['identity'] == identity
            and receipt['phase_config'] == report['phase_config']
            and reference(report['phase_config']['path']) == report['phase_config']
            and journal_ref == reference(journal_ref['path']), 'shift receipt or journal binding differs')
    journal = read_journal(journal_ref['path'])
    header = journal.header
    require(header['source_kind'] == 'actual_anylogic' and header['run_index'] == index
            and header['episode_id'] == identity['episode_id'] and header['recording_id'] == identity['recording_id']
            and header['seed'] == identity['environment_seed'] and header['target_grade'] == target
            and header['config_sha256'] == identity['config_sha256']
            and journal.file_sha256 == identity['journal_file_sha256']
            and journal.logical_sha256 == identity['journal_logical_sha256'], 'wrong physical journal identity')
    memory = build_memory(journal.events)
    processes = [event for event in journal.events if event.kind == 'PROCESS']
    windows = chosen['physical']['windows']
    for window, settled in zip(windows, memory.settlements, strict=True):
        selected = [event for event in processes if event.window == window['window']]
        quantity = math.fsum(event.tons for event in selected)
        metal = math.fsum(event.metal for event in selected)
        require(math.isclose(quantity, window['tons'], abs_tol=1e-7)
                and math.isclose(metal, window['metal_tons'], abs_tol=1e-7)
                and math.isclose(metal / quantity, window['grade'], abs_tol=1e-10)
                and math.isclose(settled.grade, window['grade'], abs_tol=1e-10),
                'PROCESS metal balance differs from window settlement')
    shift = chosen['physical']['shift_grade']
    require(math.isclose(shift, math.fsum(event.metal for event in processes)
                         / math.fsum(event.tons for event in processes), abs_tol=1e-10),
            'PROCESS metal balance differs from shift settlement')
    return dict(identity=identity, target=target, grade_band=[header['grade_lower'], header['grade_upper']],
                shift_seconds=header['shift_seconds'], window_seconds=header['window_seconds'],
                shift_grade=shift, windows=windows, complete_task=chosen['physical']['complete_task'],
                selection=dict(criterion='nearest four-window grade RMSE to median; ties by run_index',
                               n_shifts=len(pool), selected_rmse=rmse(chosen), cohort_median_rmse=typical),
                source_refs=dict(report=report_ref, material_receipt=reference(receipt_path),
                                 journal=journal_ref, config=report['phase_config'], checkpoint=control['checkpoint']),
                process_intervals=[dict(source=event.source_id, start=event.interval_start, end=event.interval_end,
                                        tons=event.tons, metal=event.metal, grade=event.metal / event.tons)
                                   for event in processes])


def rolling_paths(intervals, trailing_tons):
    tons, metal = [0.], [0.]
    for event in intervals:
        tons.append(tons[-1] + event['tons'])
        metal.append(metal[-1] + event['metal'])

    def cumulative_metal(quantity):
        position = min(bisect_right(tons, quantity) - 1, len(intervals) - 1)
        return metal[position] + (quantity - tons[position]) * intervals[position]['grade']

    changes = [quantity + trailing_tons for quantity in tons if quantity + trailing_tons <= tons[-1]]
    paths, previous_end = [], -1.
    for position, event in enumerate(intervals):
        start, end = tons[position:position + 2]
        first = max(start, trailing_tons)
        if end <= first:
            continue
        if not paths or event['start'] > previous_end + 1e-7:
            paths.append([])
        quantities = [first, *[value for value in changes if first < value < end], end]
        for quantity in quantities:
            portion = (quantity - start) / (end - start)
            time = event['start'] + portion * (event['end'] - event['start'])
            grade = (cumulative_metal(quantity) - cumulative_metal(quantity - trailing_tons)) / trailing_tons
            paths[-1].append((time, grade))
        previous_end = event['end']
    require(paths and all(len(path) >= 2 for path in paths), 'missing processed-tonnage trace')
    low, high = min(event['grade'] for event in intervals), max(event['grade'] for event in intervals)
    require(all(low - 1e-10 <= grade <= high + 1e-10 for path in paths for _, grade in path),
            'rolling grade outside processed grade range')
    return paths


def render_svg(sample):
    fig = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" role="img">',
           '<title>单班破碎出料品位</title><desc>保持实际批次的波动，另画最近1000吨移动品位和四个两小时结算窗。</desc>',
           f'<rect width="{WIDTH}" height="{HEIGHT}" fill="#ffffff"/>']

    def text(x, y, value, size=16, *, color=INK, weight=400, anchor='start', extra=''):
        fig.append(f'<text x="{x}" y="{y}" font-family="Noto Sans CJK SC, sans-serif" font-size="{size}" '
                   f'font-weight="{weight}" fill="{color}" text-anchor="{anchor}" {extra}>{escape(value)}</text>')

    def line(x1, y1, x2, y2, color, width=1, *, extra=''):
        fig.append(f'<line x1="{x1:.3f}" y1="{y1:.3f}" x2="{x2:.3f}" y2="{y2:.3f}" '
                   f'stroke="{color}" stroke-width="{width}" {extra}/>')

    def rect(x, y, width, height, color):
        fig.append(f'<rect x="{x:.3f}" y="{y:.3f}" width="{width:.3f}" height="{height:.3f}" fill="{color}"/>')

    def time_x(seconds):
        return LEFT + (RIGHT - LEFT) * seconds / sample['shift_seconds']

    def upper_y(grade_percent):
        return 375 - (grade_percent - 13) / 28 * 227

    def lower_y(grade_percent):
        return 581 - (grade_percent - 25.75) / 2.5 * 126

    def guides(y_top, y_bottom, tick_values, ordinate, label_y):
        for grade in tick_values:
            ypos = ordinate(grade)
            line(LEFT, ypos, RIGHT, ypos, GRID)
            text(LEFT - 12, ypos + 5, str(grade), 15, color=MUTED, anchor='end')
        for hour in (2, 4, 6):
            xpos = time_x(hour * 3600)
            line(xpos, y_top, xpos, y_bottom, '#9cabb5', extra='stroke-dasharray="4 5"')
        line(LEFT, y_bottom, RIGHT, y_bottom, INK)
        line(LEFT, y_top, LEFT, y_bottom, INK)
        text(27, label_y, '品位 (%)', 16, color=MUTED, anchor='middle',
             extra=f'transform="rotate(-90 27 {label_y})"')

    text(LEFT, 49, '同一班次的破碎出料品位', 33, weight=700)
    text(LEFT, 82, f"标称工况  /  BG128 s02–u32  /  班次 {sample['identity']['run_index']}", 17, color=MUTED)
    text(RIGHT, 82, f"全班吨位加权品位  {sample['shift_grade']:.3%}", 18, weight=700, anchor='end')
    text(LEFT, 128, 'A  实际处理料批与短时移动品位', 19, weight=700)
    for xpos, label, color in [(451, '高品位批次', SOURCE_COLORS[0]), (579, '中品位批次', SOURCE_COLORS[1]),
                               (707, '低品位批次', SOURCE_COLORS[2]), (835, '最近 1,000 吨加权', ROLL_COLOR)]:
        line(xpos, 121, xpos + 22, 121, color, 2.5)
        text(xpos + 29, 127, label, 15, color=MUTED)
    guides(148, 375, (15, 20, 25, 30, 35, 40), upper_y, 261)
    for event in sample['process_intervals']:
        ypos = upper_y(100 * event['grade'])
        line(time_x(event['start']), ypos, time_x(event['end']), ypos, SOURCE_COLORS[event['source']], 2.6)
    for trace in sample['rolling_paths']:
        points = ' '.join(f'{time_x(time):.3f},{upper_y(100 * grade):.3f}' for time, grade in trace)
        fig.append(f'<polyline points="{points}" fill="none" stroke="{ROLL_COLOR}" stroke-width="2.5" '
                   'stroke-linecap="round" stroke-linejoin="round"/>')
    text(LEFT, 429, 'B  2 小时窗口的实际品位', 19, weight=700)
    rect(788, 415, 19, 13, '#e1f2ed')
    text(814, 428, '窗口合规带 26%–28%', 15, color=MUTED)
    line(976, 422, 995, 422, TARGET_COLOR, 1.3, extra='stroke-dasharray="5 4"')
    text(1001, 428, '27%', 15, color=MUTED)
    lower, upper = (100 * value for value in sample['grade_band'])
    rect(LEFT, lower_y(upper), RIGHT - LEFT, lower_y(lower) - lower_y(upper), '#e1f2ed')
    guides(455, 581, (26, 27, 28), lower_y, 517)
    goal_y = lower_y(100 * sample['target'])
    line(LEFT, goal_y, RIGHT, goal_y, TARGET_COLOR, 1.3, extra='stroke-dasharray="5 4"')
    for window in sample['windows']:
        start = window['window'] * sample['window_seconds']
        end = start + sample['window_seconds']
        xpos = time_x((start + end) / 2)
        ypos = lower_y(window['grade'] * 100)
        line(time_x(start) + 6, ypos, time_x(end) - 6, ypos, WINDOW_COLOR, 2.7)
        fig.append(f'<circle cx="{xpos:.3f}" cy="{ypos:.3f}" r="5" fill="{WINDOW_COLOR}"/>')
        text(xpos, ypos - 11, f"{window['grade']:.3%}", 17, weight=700, anchor='middle')
    for hour in range(9):
        xpos = time_x(hour * 3600)
        line(xpos, 581, xpos, 587, INK)
        text(xpos, 605, str(hour), 15, color=MUTED, anchor='middle')
    text((LEFT + RIGHT) / 2, 630, '班次时间 (h)', 16, anchor='middle')
    text(LEFT, 662, '来源：AnyLogic 破碎 PROCESS 记录。停料处留空；合规带只用于 2 小时结算，不代表瞬时料批准则。',
         14, color=MUTED)
    return ''.join(fig) + '</svg>\n'


def main():
    evidence = sample()
    evidence['rolling_tons'] = ROLL_TONS
    evidence['rolling_paths'] = rolling_paths(evidence['process_intervals'], ROLL_TONS)
    svg = HERE / f'{STEM}.svg'
    svg.write_text(render_svg(evidence), encoding='utf-8')
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-width', str(WIDTH * 2),
                    '-height', str(HEIGHT * 2), '-i', str(svg), '-frames:v', '1',
                    '-threads', '1', str(HERE / f'{STEM}.png')], check=True)
    (HERE / f'{STEM}.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                                     encoding='utf-8')
    print(f"Rendered sealed run {evidence['identity']['run_index']}: {svg}")


if __name__ == '__main__':
    main()
