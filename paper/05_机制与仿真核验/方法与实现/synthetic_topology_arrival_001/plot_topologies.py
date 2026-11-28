"""Render the two AnyLogic SDK route samples on a shared spatial scale."""

import argparse
import csv
import json
from pathlib import Path
import subprocess


NODE_LABELS = {
    'start': '起点', 'junction': '中心路口', 'load': '装载点', 'dump': '卸载点',
    'west': '西路口', 'east': '东路口', 'south': '南路口',
}
BLUE, ORANGE, GRAY = '#1172bb', '#c45920', '#b1bfcb'


def line(x1, y1, x2, y2, color, width, dash='none'):
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
            f'stroke-width="{width}" stroke-dasharray="{dash}" stroke-linecap="round"/>')


def text(x, y, content, size, attributes=''):
    return f'<text x="{x}" y="{y}" font-size="{size}" {attributes}>{content}</text>'


def draw_panel(panel_x, layout, rows):
    nodes = layout['nodes']
    roads = layout['roads']
    layout_id = layout['id']
    topology = '分支树' if layout_id == 'A' else '含环路'

    def project(point):
        return panel_x + 90 + 1.97 * point[0], 422 - 1.97 * point[1]

    yield f'<rect x="{panel_x}" y="124" width="650" height="414" rx="16" fill="white" stroke="#dce6ef"/>'
    yield text(panel_x + 24, 163, f'地图 {layout_id}  ·  {topology}', 23, 'font-weight="700"')
    yield text(panel_x + 624, 163, f'{len(nodes)} 节点  ·  {len(roads)} 双向道路', 16,
               'class="secondary" text-anchor="end"')
    for x in range(0, 251, 50):
        screen_x, _ = project((x, 0))
        yield line(screen_x, 189, screen_x, 474, '#e9eff4', 1)
        yield text(screen_x, 498, x, 14, 'class="secondary" text-anchor="middle"')
    for y in range(0, 101, 50):
        _, screen_y = project((0, y))
        yield line(panel_x + 69, screen_y, panel_x + 612, screen_y, '#e9eff4', 1)
        yield text(panel_x + 57, screen_y + 4, y, 14, 'class="secondary" text-anchor="end"')
    yield text(panel_x + 325, 523, 'X（模型单位）', 14, 'class="secondary" text-anchor="middle"')
    yield text(panel_x + 28, 205, 'Y', 14, 'class="secondary"')

    for start, end in roads:
        yield line(*project(nodes[start]), *project(nodes[end]), GRAY, 7)
    for stage, color, dash in (('load', BLUE, 'none'), ('dump', ORANGE, '10 8')):
        samples = (project((float(row['x']), float(row['y']))) for row in rows if row['stage'] == stage)
        points = ' '.join(f'{x:.2f},{y:.2f}' for x, y in samples)
        yield (f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="5" '
               f'stroke-linejoin="round" stroke-dasharray="{dash}"/>')
    for key, (x, y) in nodes.items():
        screen_x, screen_y = project((x, y))
        color = {'load': BLUE, 'dump': ORANGE, 'start': '#20384d'}.get(key, '#62788f')
        radius = 7 if key in ('load', 'dump') else 5.5
        yield (f'<circle cx="{screen_x:.2f}" cy="{screen_y:.2f}" r="{radius}" '
               f'fill="{color}" stroke="white" stroke-width="2"/>')
        below = key in ('start', 'dump', 'junction', 'south')
        label = f'{NODE_LABELS[key]} ({x}, {y})'
        style = 'font-weight="600" paint-order="stroke" stroke="white" stroke-width="4"'
        if key == 'east':
            yield text(screen_x + 17, screen_y + 5, label, 15, style)
        else:
            yield text(screen_x, screen_y + (23 if below else -18), label, 15, style + ' text-anchor="middle"')


def render(config, event_directory):
    yield '<svg xmlns="http://www.w3.org/2000/svg" width="1440" height="612" viewBox="0 0 1440 612">'
    yield '<rect width="1440" height="612" fill="#f4f8fb"/>'
    yield '<style>text{font-family:Noto Sans CJK SC,sans-serif;fill:#20384d}.secondary{fill:#546b81}</style>'
    yield text(54, 58, '两张合成路网的到达坐标验证', 32, 'font-weight="700"')
    yield text(55, 96, 'AnyLogic 原生路径坐标采样  ·  起点 → 装载点 → 卸载点  ·  两图共用坐标比例', 18,
               'class="secondary"')
    arrivals = 0
    for index, layout in enumerate(config['layouts']):
        with (event_directory / f"{layout['id']}.csv").open() as source:
            rows = list(csv.DictReader(source))
        yield from draw_panel(54 + 681 * index, layout, rows)
        arrivals += sum(row['event'] == 'arrival' and
                        float(row['error']) <= config['arrival_tolerance_model_units'] for row in rows)
    for x, color, dash, label in (
        (59, BLUE, 'none', '去装载点'),
        (244, ORANGE, '10 8', '去卸载点'),
        (429, GRAY, 'none', '道路'),
    ):
        yield line(x, 572, x + 43, 572, color, 5, dash)
        yield text(x + 56, 578, label, 16)
    yield text(1382, 578, f'{arrivals} 个目标坐标通过校验', 17, 'font-weight="700" text-anchor="end"')
    yield '</svg>'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--events', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(Path(__file__).with_name('campaign.json').read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    svg = args.output / 'topology_comparison.svg'
    png = args.output / 'topology_comparison.png'
    svg.write_text('\n'.join(render(config, args.events)), encoding='utf-8')
    subprocess.run(['gdk-pixbuf-thumbnailer', '-s', '1800', svg, png], check=True)


if __name__ == '__main__':
    main()
