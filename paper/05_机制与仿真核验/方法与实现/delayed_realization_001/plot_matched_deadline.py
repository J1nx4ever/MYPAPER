"""Plot the observed unloading and physical processing intervals of the archived example."""
import json
from math import isclose
from pathlib import Path
from xml.etree import ElementTree as ET

from matched_batches import OUT, sealed, read_journal


def physical_intervals(row, report_ref):
    episode = next(e for e in sealed(report_ref)['per_episode'] if e['identity']['run_index'] == row['run'])
    folder = Path(report_ref['path']).parent / 'episodes' / f"run_{row['run']}"
    receipt = json.loads((folder / 'material_receipt.json').read_bytes())
    ref = receipt['references']['journal']
    journal = read_journal(ref['path'])
    if (episode['identity'] != receipt['identity'] or episode['identity']['journal_file_sha256']
            != journal.file_sha256 or ref['sha256'] != journal.file_sha256):
        raise ValueError('Frozen episode or journal differs')
    intervals = [(e.interval_start, e.interval_end) for e in journal.events
                 if e.kind == 'PROCESS' and e.batch_id == row['batch']]
    if not intervals or not isclose(intervals[0][0], row['first_process_s']):
        raise ValueError('Missing or mismatched crusher intervals')
    return intervals


def node(group, tag, **attrs):
    return ET.SubElement(group, tag, {k.replace('_','-'):str(v) for k,v in attrs.items()})


def text(group, x, y, string, size=17, color='#203344', weight='normal', anchor='start'):
    node(group, 'text', x=x, y=y, font_size=size, fill=color, font_weight=weight, text_anchor=anchor).text = string


def render():
    report = json.loads((OUT / 'matched_batches.json').read_bytes())
    pair = report['illustrative_pair']
    w = pair['window']
    end = pair['full_s04']['boundary_s']
    if w != 3 or end != pair['full_s05']['boundary_s'] or end != 21600:
        raise ValueError('Expected third physical window closure')
    svg = ET.Element('svg', xmlns='http://www.w3.org/2000/svg', width='1070', height='374',
                     viewBox='0 0 1070 374', role='img')
    ET.SubElement(svg,'title').text = 'Physical crusher timing for two matched high-grade deliveries'
    ET.SubElement(svg,'desc').text = 'Two 91.7-tonne batches from source zero arrived before the closing of W3. Full s04 crushed the entire batch in W4; Full s05 crushed the entire batch in W3.'
    body = node(svg, 'g', font_family='DejaVu Sans, sans-serif')
    node(body, 'rect', x=0, y=0, width=1070, height=374, fill='white')
    left, right = 235, 1030
    def pos(t):
        return left + (t + 420) * (right-left) / 720
    blue, amber = '#225775', '#a5470f'
    text(body, 18, 30, 'Matched ore deliveries at the W3 boundary', 22, weight='bold')
    text(body, 18, 55, f"Source {pair['full_s04']['source']}  ·  {pair['full_s04']['tons']:.1f} t per batch  ·  held_order_1, paired slot {pair['ordinal']} (0-based)", 15)
    top, bottom = 91, 281
    node(body,'rect',x=left,y=top,width=pos(0)-left,height=bottom-top,fill='#f2f6f8')
    node(body,'rect',x=pos(0),y=top,width=right-pos(0),height=bottom-top,fill='#faf4ed')
    text(body,left+9,top+20,'W3',14)
    text(body,right-9,top+20,'W4',14,anchor='end')
    node(body,'line',x1=pos(0),y1=top-5,x2=pos(0),y2=bottom,stroke='#354655',stroke_width=1.6,stroke_dasharray='5 4')
    text(body,pos(0),top-8,'W3 closes at 21,600 s',15,weight='bold',anchor='middle')
    for i,(key,seed) in enumerate((('full_s04','s04'),('full_s05','s05'))):
        row = pair[key]
        intervals = physical_intervals(row, report['input_evaluation_reports'][i])
        y = 148 + i*90
        text(body,18,y-12, f"Full {seed}  ·  run {row['run']}",16,weight='bold')
        text(body,18,y+10, f"FIFO ahead: {row['fifo_ahead_t']:.1f} t",16)
        text(body,18,y+30, f"Batch grade: {100*row['grade']:.2f}%",14)
        unload=row['unload_s']-end
        start,finish=intervals[0][0]-end,max(b for _,b in intervals)-end
        node(body,'line',x1=pos(unload),x2=pos(start),y1=y,y2=y,stroke='#708493',stroke_width=1.5,
             stroke_dasharray='4 4')
        for a,b in intervals:
            node(body,'rect',x=pos(a-end),y=y-6,width=pos(b-end)-pos(a-end),height=12,fill=blue)
        xu=pos(unload)
        node(body,'polygon',points=f'{xu},{y-1} {xu-6},{y-14} {xu+6},{y-14}',fill=amber)
        text(body,xu,y-21,f'{unload:+.0f} s',15,color=amber,anchor='middle')
        text(body,(pos(start)+pos(finish))/2,y+26,f'{start:+.0f} to {finish:+.0f} s',14,color=blue,anchor='middle')
    node(body,'line',x1=left,x2=right,y1=bottom,y2=bottom,stroke='#486070')
    for tick in (-400,-300,-200,-100,0,100,200,300):
        x=pos(tick)
        node(body,'line',x1=x,x2=x,y1=bottom,y2=bottom+6,stroke='#486070')
        text(body,x,bottom+23,str(tick),14,anchor='middle')
    text(body,(left+right)/2,bottom+50,'Time relative to W3 closure (s)',16,anchor='middle')
    node(body,'polygon',points='268,354 262,342 274,342',fill=amber)
    text(body,281,354,'Unload',15)
    node(body,'line',x1=390,x2=424,y1=350,y2=350,stroke='#708493',stroke_width=1.7,stroke_dasharray='4 4')
    text(body,434,354,'Waiting for crusher',15)
    node(body,'rect',x=661,y=344,width=28,height=11,fill=blue)
    text(body,700,354,'Actual crushing interval',15)
    ET.indent(svg)
    ET.ElementTree(svg).write(OUT / 'matched_deadline.svg', encoding='utf-8', xml_declaration=True)


if __name__ == '__main__':
    render()
