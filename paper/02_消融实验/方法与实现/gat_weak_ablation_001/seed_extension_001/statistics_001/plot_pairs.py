"""Plot the two primary held-out outcomes for the five matched training schedules."""

import json
from pathlib import Path
import subprocess
from xml.etree import ElementTree as ET


HERE = Path(__file__).resolve().parent
PLANS = next(d for d in HERE.parents if d.name == 'change_plans')
OUTPUT = PLANS.parent / 'reports' / HERE.relative_to(PLANS)
SEEDS = ('s01', 's02', 's03', 's04', 's05')
NAVY = '#23466B'
ORANGE = '#B64B10'
TEXT = '#152A3A'


def label(group, x, y, value, *, anchor='start', size=15, bold=False):
    tag = ET.SubElement(group, 'text', {
        'x': str(x), 'y': str(y), 'text-anchor': anchor,
        'font-size': str(size), 'font-weight': 'bold' if bold else 'normal', 'fill': TEXT,
    })
    tag.text = value


def line(group, x1, y1, x2, y2, color, width=1):
    ET.SubElement(group, 'line', {
        'x1': str(x1), 'y1': str(y1), 'x2': str(x2), 'y2': str(y2),
        'stroke': color, 'stroke-width': str(width),
    })


def marker(group, x, y, arm):
    if arm == 'full':
        ET.SubElement(group, 'circle', {'cx': str(x), 'cy': str(y), 'r': '5.5', 'fill': NAVY})
    else:
        points = [(x, y - 7), (x + 7, y), (x, y + 7), (x - 7, y)]
        ET.SubElement(group, 'polygon', {'points': ' '.join(f'{u},{v}' for u, v in points),
                                         'fill': ORANGE})


def panel(group, left, title, unit, values, scale, limits, ticks):
    width = 346
    low, high = limits

    def xpos(value):
        return left + width * (value - low) / (high - low)

    label(group, left, 31, title, size=20, bold=True)
    label(group, left, 54, unit, size=15)
    for tick in ticks:
        x = xpos(tick)
        line(group, x, 68, x, 239, '#D8DEE4')
        label(group, x, 260, str(tick), anchor='middle')
    for index, seed in enumerate(SEEDS):
        y = 80 + 37 * index
        values_by_arm = values[seed]
        if values_by_arm['paired_n'] != 500:
            raise ValueError(f'Expected 500 held-out shifts for {seed}')
        full, gat = (scale * values_by_arm[key] for key in ('full', 'ablation'))
        if not (low <= full <= high and low <= gat <= high):
            raise ValueError(f'Axis limits exclude an observation for {seed}')
        x_full, x_gat = xpos(full), xpos(gat)
        line(group, x_full, y, x_gat, y, '#778796', width=2)
        marker(group, x_full, y, 'full')
        marker(group, x_gat, y, 'ablation')
    line(group, left, 239, left + width, 239, '#7A8996')


def main():
    source = OUTPUT.parent / 'paired_evaluation_summary.json'
    summary = json.loads(source.read_bytes())
    if (tuple(summary['training_seed_schedules']) != SEEDS
            or summary['evaluation_slots_per_checkpoint'] != 700
            or summary['direction'] != 'gat1_minus_full'):
        raise ValueError('Unexpected fixed-checkpoint summary')
    held_out = summary['summaries']['held_out']
    svg = ET.Element('svg', {'xmlns': 'http://www.w3.org/2000/svg', 'width': '180mm',
                             'height': '62mm', 'viewBox': '0 0 900 310', 'role': 'img'})
    ET.SubElement(svg, 'title').text = 'Full and one-round GAT results on the five held-out conditions'
    ET.SubElement(svg, 'desc').text = (
        'Each row joins two fixed u32 checkpoints for a matched training schedule. '
        'Each point averages 500 shifts across five held-out conditions on the same 100 paired seeds.'
    )
    ET.SubElement(svg, 'rect', {'x': '0', 'y': '0', 'width': '900', 'height': '310', 'fill': 'white'})
    group = ET.SubElement(svg, 'g', {'font-family': 'Liberation Sans, Arial, sans-serif'})
    label(group, 13, 54, 'Seed', size=15, bold=True)
    for index, seed in enumerate(SEEDS):
        label(group, 13, 85 + 37 * index, seed, size=16)
    panel(group, 73, '(a) Complete-task rate', '% of shifts',
          held_out['complete_task']['by_training_seed'], 100,
          (89.6, 100.4), [90, 92, 94, 96, 98, 100])
    panel(group, 506, '(b) Cumulative fleet waiting time', 'Truck-hours per shift',
          held_out['total_wait_seconds']['by_training_seed'], 1 / 3600,
          (3.4, 6.2), [3.5, 4, 4.5, 5, 5.5, 6])
    marker(group, 244, 291, 'full')
    label(group, 259, 296, 'Full (2-round GAT)', size=16)
    marker(group, 455, 291, 'ablation')
    label(group, 470, 296, '1-round GAT + node MLP', size=16)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT / 'paired_seeds.svg'
    ET.ElementTree(svg).write(destination, xml_declaration=True, encoding='utf-8')
    subprocess.run(['ffmpeg', '-nostdin', '-loglevel', 'error', '-y', '-width', '2400',
                    '-i', str(destination), '-frames:v', '1', str(destination.with_suffix('.png'))],
                   check=True)


if __name__ == '__main__':
    main()
