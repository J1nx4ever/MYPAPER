"""Draw the three sealed training traces and the matched held-out comparison."""
from hashlib import sha256
from html import escape
import json
import math
from pathlib import Path
import sqlite3
from statistics import mean
import subprocess

HERE = Path(__file__).resolve().parent
PROJECT = next(folder for folder in HERE.parents if (folder / 'grade_event_memory_v1').is_dir())
RUNS = {
    's01': 'bg128-continuous-mixed-20261119-001',
    's02': 'bg128-continuous-mixed-20261120-s02',
    's03': 'bg128-continuous-mixed-20261120-s03',
}
HELD = ('held_order_1', 'held_order_2', 'new_numeric', 'joint_shift', 'waste_swap')
COLORS = {'旧标称': '#748290', 's01': '#205fa5', 's02': '#08796a', 's03': '#b1532c'}
DASHES = {'s01': '', 's02': '9 4', 's03': '2 4'}
INK, MUTED, GRID = '#172c3d', '#516674', '#e0e8ec'
WIDTH, HEIGHT = 1280, 770
STEM = 'three_seed_learning_evidence'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def json_file(path):
    raw = Path(path).read_bytes()
    return json.loads(raw), dict(path=str(Path(path).resolve()), sha256=sha256(raw).hexdigest())


def checked_json(ref):
    value, actual = json_file(ref['path'])
    require(ref == actual, 'frozen reference differs: ' + ref['path'])
    return value


def collect():
    training, reports, sources = {}, {}, {}
    for seed, name in RUNS.items():
        root = PROJECT / 'runs' / name
        completed, completion_ref = json_file(root / 'suite_complete.json')
        require(completed['status'] == 'COMPLETE'
                and (completed['training_shifts'], completed['training_updates'],
                     completed['evaluation_optimizer_updates']) == (1024, 32, 0), 'incomplete campaign')
        suite_ref = completed['campaign'] if seed == 's01' else completed['suite']
        require(checked_json(suite_ref)['campaign_id'] == name, 'wrong campaign identity')
        with sqlite3.connect((root / 'budget.sqlite').as_uri() + '?mode=ro', uri=True) as ledger:
            commits = ledger.execute('SELECT update_id FROM commits ORDER BY update_id').fetchall()
        require(commits == [(step,) for step in range(1, 33)], 'incomplete update lineage')
        traces, traces_refs, previous_actor = [], [], None
        for step, checkpoint in enumerate(completed['training_checkpoints'], 1):
            path = root / 'arms/s01_BG128/updates' / f'{step:04d}'
            require(checkpoint['path'] == str(path / 'checkpoint.pt'), 'checkpoint ordering differs')
            summary, summary_ref = json_file(path / 'training_summary.json')
            update = summary['update']
            scenes = summary['scenario_statistics'].values()
            n_shifts = sum(item['shifts'] for item in scenes)
            successes = sum(item['complete_task'] for item in scenes)
            cost = update['metrics']['combined_cost_mean']
            require(summary['transaction']['campaign'] == suite_ref and step == update['update_index']
                    and n_shifts == update['episodes'] == 32 and 0 <= successes <= n_shifts
                    and math.isfinite(cost) and cost >= 0
                    and (previous_actor is None or update['policy_before'] == previous_actor),
                    'training batch does not bind the committed experiment')
            previous_actor = update['policy_after']
            traces.append(dict(batch=step, shifts=n_shifts, complete_tasks=successes,
                               mean_return=-cost, task_rate=successes / n_shifts))
            traces_refs.append(summary_ref)
        pair_ref = completed['comparison']['json'] if seed == 's01' else completed['comparison']
        paired = checked_json(pair_ref)
        held_rows = {(row['scenario_id'], row['pair_ordinal']): row
                     for row in paired['rows'] if row['scenario_id'] in HELD}
        require(set(held_rows) == {(case, slot) for case in HELD for slot in range(100)},
                'held-out evaluation slots differ')
        training[seed] = traces
        reports[seed] = held_rows
        sources[seed] = dict(completion=completion_ref, suite=suite_ref, training_summaries=traces_refs,
                             fixed_evaluation=pair_ref)
    anchor = reports['s01']
    for seed in ('s02', 's03'):
        for key, row in reports[seed].items():
            require(all(row[field] == anchor[key][field] for field in ('environment_seed', 'actor_sampling_seed', 'old'))
                    and row['seed01'] == anchor[key]['new'], 'physical evaluation pairing differs')
    metrics = {}
    for seed in COLORS:
        rows = [row['old' if seed == '旧标称' else 'new']
                for row in reports.get(seed, anchor).values()]
        metrics[seed] = dict(n_shifts=len(rows), complete_tasks=sum(row['complete_task'] for row in rows),
                             mean_wait_vehicle_seconds=mean(row['total_wait_seconds'] for row in rows))
    blocks = {
        seed: [dict(mean_return=mean(row['mean_return'] for row in batch),
                    complete_tasks=sum(row['complete_tasks'] for row in batch),
                    shifts=sum(row['shifts'] for row in batch))
               for batch in (rows[:8], rows[8:16], rows[16:24], rows[24:])]
        for seed, rows in training.items()
    }
    return dict(training=training, fixed_evaluation=metrics, training_blocks_of_eight=blocks,
                heldout_scenarios=list(HELD), sources=sources,
                learning_return='negative mean unnormalized whole-shift composite cost; frozen pre-update rollout',
                training_task='all four windows compliant and both production targets met',
                smoothing='trailing 5 batches; available prefix for first four batches',
                evaluation='fixed u32; paired scene/environment/Actor seeds per slot; 500 shifts per model')


class Chart:
    def __init__(self):
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
            f'viewBox="0 0 {WIDTH} {HEIGHT}" role="img">',
            '<title>三个种子的训练趋势与独立评测</title><desc>回报与任务率贯穿三十二个训练批次，下游比较五个留出工况。</desc>',
            f'<rect width="{WIDTH}" height="{HEIGHT}" fill="#ffffff"/>',
        ]

    def text(self, x, y, content, size=16, *, color=INK, weight=400, anchor='start'):
        self.parts.append(f'<text x="{x:.3f}" y="{y:.3f}" font-family="Noto Sans CJK SC, sans-serif" '
                          f'font-size="{size}" font-weight="{weight}" fill="{color}" '
                          f'text-anchor="{anchor}">{escape(content)}</text>')

    def line(self, x1, y1, x2, y2, color=GRID, width=1, *, dash=''):
        self.parts.append(f'<line x1="{x1:.3f}" y1="{y1:.3f}" x2="{x2:.3f}" y2="{y2:.3f}" '
                          f'stroke="{color}" stroke-width="{width}" stroke-dasharray="{dash or "none"}"/>')

    def trace(self, points, seed, *, raw=False):
        coords = ' '.join(f'{xpos:.3f},{ypos:.3f}' for xpos, ypos in points)
        self.parts.append(f'<polyline points="{coords}" fill="none" stroke="{COLORS[seed]}" '
                          f'stroke-width="{1.4 if raw else 2.5}" opacity="{.32 if raw else 1}" '
                          f'stroke-dasharray="{(DASHES[seed] or "none") if not raw else "none"}" '
                          'stroke-linecap="round" stroke-linejoin="round"/>')
        if raw:
            for xpos, ypos in points:
                self.parts.append(f'<circle cx="{xpos:.3f}" cy="{ypos:.3f}" r="2.1" '
                                  f'fill="{COLORS[seed]}" opacity="0.38"/>')

    def bar(self, x, y, width, height, color):
        self.parts.append(f'<rect x="{x:.3f}" y="{y:.3f}" width="{width:.3f}" height="{height:.3f}" fill="{color}"/>')


def describe(chart, data):
    chart.text(76, 55, '三个训练种子的学习趋势与独立评测', 34, weight=700)
    chart.text(76, 90, 'BG128 完整架构  /  每个训练种子 1,024 班  /  固定评测检查点 u32', 17, color=MUTED)
    for xpos, seed in zip((77, 169, 261), RUNS, strict=True):
        chart.line(xpos, 122, xpos + 28, 122, COLORS[seed], 2.8, dash=DASHES[seed])
        chart.text(xpos + 36, 128, seed, 16, color=MUTED)
    chart.text(365, 128, '浅线与点：原始批次    深线：最近 5 批均值', 16, color=MUTED)
    chart.text(1215, 128, '固定独评：5 个共同留出工况 × 100 个配对班', 16, color=MUTED, anchor='end')

    def training_panel(top, title, subtitle, field, scale, limits, tick_values):
        left, right = 106, 627
        y_top, bottom = top + 56, top + 239

        def x(step):
            return left + (right - left) * (step - 1) / 31

        def y(value):
            return bottom - (bottom - y_top) * (value - limits[0]) / (limits[1] - limits[0])

        chart.text(76, top + 16, title, 21, weight=700)
        chart.text(76, top + 42, subtitle, 15, color=MUTED)
        for tick in tick_values:
            level = y(tick)
            chart.line(left, level, right, level)
            chart.text(left - 11, level + 5, str(tick).replace('-', '−'), 14, color=MUTED, anchor='end')
        for step in (1, 8, 16, 24, 32):
            xpos = x(step)
            chart.line(xpos, y_top, xpos, bottom)
            chart.text(xpos, bottom + 22, str(step), 14, color=MUTED, anchor='middle')
        chart.line(left, bottom, right, bottom, INK)
        chart.line(left, y_top, left, bottom, INK)
        for seed, rows in data['training'].items():
            values = [row[field] * scale for row in rows]
            require(all(limits[0] <= value <= limits[1] for value in values), 'curve outside axis domain')
            chart.trace([(x(index), value_y) for index, value_y in enumerate(map(y, values), 1)], seed, raw=True)
        for seed, rows in data['training'].items():
            values = [row[field] * scale for row in rows]
            smooth = [mean(values[max(0, index - 4):index + 1]) for index in range(len(values))]
            chart.trace([(x(index), value_y) for index, value_y in enumerate(map(y, smooth), 1)], seed)
        chart.text((left + right) / 2, bottom + 46, 'PPO 更新批次', 15, color=MUTED, anchor='middle')

    def evaluation_panel(top, title, subtitle, values, upper, ticks, formatting):
        left, right = 820, 1136
        y_top, bottom = top + 56, top + 239
        chart.text(748, top + 16, title, 21, weight=700)
        chart.text(748, top + 42, subtitle, 15, color=MUTED)
        for tick in ticks:
            xpos = left + (right - left) * tick / upper
            chart.line(xpos, y_top, xpos, bottom)
            chart.text(xpos, bottom + 22, str(tick), 14, color=MUTED, anchor='middle')
        chart.line(left, bottom, right, bottom, INK)
        for position, (seed, value) in enumerate(values.items()):
            center = y_top + 24 + 43 * position
            require(0 <= value <= upper, 'bar outside axis domain')
            width = (right - left) * value / upper
            chart.text(left - 12, center + 6, seed, 16, color=MUTED, anchor='end')
            chart.bar(left, center - 9, width, 18, COLORS[seed])
            chart.text(left + width + 9, center + 6, formatting(value), 16, weight=700)

    start, end = 155, 462
    minimum = min(row['mean_return'] for values in data['training'].values() for row in values)
    limit = math.floor(minimum)
    training_panel(start, 'A  训练回报', '每班平均；综合成本取负，越高越好', 'mean_return', 1,
                   (limit, 0), tuple(range(limit, 1)))
    training_panel(end, 'B  训练任务完成率', '四窗合规且矿石、废石均达产（%）', 'task_rate', 100,
                   (0, 100), (0, 25, 50, 75, 100))
    metrics = data['fixed_evaluation']
    evaluation_panel(start, 'C  留出工况任务完成率', '固定策略独评（%）', {
        seed: 100 * row['complete_tasks'] / row['n_shifts'] for seed, row in metrics.items()
    }, 100, (0, 25, 50, 75, 100), lambda value: f'{value:.1f}%')
    evaluation_panel(end, 'D  留出工况平均车队等待', '千车辆秒 / 班；越低越好', {
        seed: row['mean_wait_vehicle_seconds'] / 1000 for seed, row in metrics.items()
    }, 30, (0, 10, 20, 30), lambda value: f'{value:.2f}')


def main():
    data = collect()
    chart = Chart()
    describe(chart, data)
    svg = HERE / f'{STEM}.svg'
    svg.write_text(''.join(chart.parts) + '</svg>\n', encoding='utf-8')
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-width', str(WIDTH * 2),
                    '-height', str(HEIGHT * 2), '-i', str(svg), '-frames:v', '1',
                    '-threads', '1', str(HERE / f'{STEM}.png')], check=True)
    (HERE / f'{STEM}.json').write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                                     encoding='utf-8')
    print(f'Rendered 96 training batches and the paired evaluation: {svg}')


if __name__ == '__main__':
    main()
