"""Static plots of frozen-initialization encoder tensors; no model is run."""
from pathlib import Path
from collections import defaultdict
import csv
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT=Path(__file__).resolve().parents[4]
OUT=PROJECT/'reports/token_attention_001/business_graph_summary_009/encoder_snapshot_001'
INPUT=PROJECT/'reports/token_attention_001/business_graph_summary_009/historical_snapshot_001'
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
BOLD='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
INK='#17283d';MUTED='#566b80';BG='#f5f7fb'
COLORS={'own_H0':'#4379be','other_H0':'#2b9c80','graph':'#dc9541'}
NAMES=[f'Truck{i}' for i in range(9)]+[f'Load{i}' for i in range(4)]+['Dump0','Dump1','Junction2','Junction3','Junction4','Global']


def f(n=25,bold=False):return ImageFont.truetype(BOLD if bold else FONT,n)


def text(d,xy,s,n=25,color=INK,bold=False,anchor=None):
    # Noto CJK lacks these mathematical superscript/subscript glyphs.
    label=str(s).translate(str.maketrans({'⁽':'(', '⁾':')', '⁰':'0', '¹':'1', '²':'2', '₂':'2'}))
    d.text(xy,label,font=f(n,bold),fill=color,anchor=anchor)


def csv_rows(name):
    with (OUT/name).open(encoding='utf-8-sig') as handle:return list(csv.DictReader(handle))


def color(value,limit):
    v=float(np.clip(value/limit,-1,1));white=np.array([249,249,247])
    target=np.array([192,57,65]) if v>=0 else np.array([43,103,174])
    return tuple(np.rint(white*(1-abs(v))+target*abs(v)).astype(int))


def heatmaps(role,arrays):
    im=Image.new('RGB',(3450,1630),BG);d=ImageDraw.Draw(im)
    text(d,(65,45),f'{role.title()} · H⁽⁰⁾ / H⁽¹⁾ / H⁽²⁾',58,bold=True)
    text(d,(70,132),'固定随机初始化 seed=2026111801；同一角色T0/T1参数完全相同；通道编号不代表物理语义',28,MUTED)
    limit=max(float(np.abs(arrays[f'{role}_{t}_H{k}']).max()) for t in ('T0','T1') for k in range(3))
    limit=math.ceil(limit*2)/2
    cw,ch=14,25
    for row,t in enumerate(('T0','T1')):
        for k in range(3):
            px=60+k*1125;py=270+row*600
            d.rounded_rectangle((px,py-65,px+1092,py+520),18,fill='white',outline='#d8e1eb',width=2)
            text(d,(px+22,py-53),f'{t} · H⁽{k}⁾  [19 × 64]',34,bold=True)
            arr=arrays[f'{role}_{t}_H{k}'];gx=px+165
            for i in range(19):
                text(d,(gx-15,py+i*ch+ch/2),NAMES[i],23,
                     '#197a58' if 9<=i<13 else INK,9<=i<13,anchor='rm')
                for j in range(64):
                    d.rectangle((gx+j*cw,py+i*ch,gx+(j+1)*cw-1,py+(i+1)*ch-1),fill=color(arr[i,j],limit))
            for j in (0,8,16,24,32,40,48,56,63):text(d,(gx+j*cw+cw/2,py+19*ch+8),j,20,MUTED,anchor='mt')
    x,y=2335,1510
    for i in range(620):d.line((x+i,y,x+i,y+28),fill=color((i/619*2-1)*limit,limit))
    for label,xx in ((f'-{limit:g}',x),('0',x+310),(f'+{limit:g}',x+620)):text(d,(xx,y+35),label,23,MUTED,anchor='mt')
    text(d,(70,1500),'同一张图六面板共用色标；Global在两层GAT中保持旁路。',27,MUTED)
    text(d,(70,1547),'本图仅展示实际数值；不表示学到业务语义，也不表示某个通道具有特定业务含义。',27,MUTED)
    im.save(OUT/f'{role}_H_layers.png')


def changes(summary,changes):
    im=Image.new('RGB',(3200,1880),BG);d=ImageDraw.Draw(im)
    text(d,(65,40),'相邻快照 · 候选Load表示变化与内部路径分解',53,bold=True)
    text(d,(70,123),'只比较固定随机初始化的数值响应；ΔL2 与分解份额均不代表学习效果或业务重要性',29,MUTED)
    stage_colors=['#a9b5c4','#5588c6','#188b79']
    for r,role in enumerate(('actor','critic')):
        px=65+r*1580
        d.rounded_rectangle((px,215,px+1510,970),22,fill='white',outline='#d7e1eb',width=2)
        text(d,(px+28,240),role.title()+' · ||H(T1) − H(T0)||₂',37,bold=True)
        for k,c in enumerate(stage_colors):
            d.rectangle((px+700+k*220,253,px+730+k*220,278),fill=c)
            text(d,(px+744+k*220,249),f'H⁽{k}⁾',28)
        bx,by=px+120,830;plotw,ploth=1300,450
        for tick in (0,.5,1,1.5,2,2.5):
            yy=by-tick/2.5*ploth;d.line((bx,yy,bx+plotw,yy),fill='#e7ecf2',width=2)
            text(d,(bx-15,yy),f'{tick:g}',24,MUTED,anchor='rm')
        for a in range(4):
            gx=bx+100+a*315
            for k,c in enumerate(stage_colors):
                row=next(x for x in changes if x['role']==role and int(x['node_id'])==9+a and int(x['layer'])==k)
                value=float(row['l2']);xx=gx+(k-1)*66
                d.rounded_rectangle((xx-23,by-value/2.5*ploth,xx+23,by),6,fill=c)
                text(d,(xx,by-value/2.5*ploth-12),f'{value:.3f}',22,c,anchor='mb')
            text(d,(gx,by+35),f'Load{a}',31,bold=True,anchor='mt')
        # Signed projection shares use a diverging scale, not vector norm shares.
        d.rounded_rectangle((px,1010,px+1510,1680),22,fill='white',outline='#d7e1eb',width=2)
        text(d,(px+28,1037),role.title()+' · 三因素向量分解',35,bold=True)
        text(d,(px+28,1091),'数值分量沿实际ΔH⁽²⁾方向的带符号投影（%）',25,MUTED)
        zx=px+260;span=1060
        def coord(v):return zx+v/110*span
        for tick in (-10,0,25,50,75,100):
            xx=coord(tick);d.line((xx,1160,xx,1515),fill='#e1e7ef',width=2)
            text(d,(xx,1530),f'{tick}%',23,MUTED,anchor='mt')
        for a in range(4):
            yy=1180+a*82;pos=neg=0
            text(d,(px+30,yy+21),f'Load{a}',29,bold=True)
            rows=[x for x in summary['roles'][role]['attribution'] if x['node_id']==9+a]
            for row in rows:
                v=row['signed_delta_projection_share']*100;factor=row['factor']
                start=pos if v>=0 else neg;end=start+v
                d.rectangle((coord(min(start,end)),yy,coord(max(start,end)),yy+45),fill=COLORS[factor])
                if v>=0:pos=end
                else:neg=end
            label=' / '.join(f"{x['signed_delta_projection_share']*100:.1f}" for x in rows)
            text(d,(px+1450,yy+47),label,20,MUTED,anchor='rt')
        for j,(factor,label) in enumerate([('own_H0','目标自身H⁽⁰⁾'),('other_H0','其余节点H⁽⁰⁾'),('graph','关系图/边属性')]):
            xx=px+35+j*470;d.rectangle((xx,1600,xx+26,1626),fill=COLORS[factor]);text(d,(xx+42,1595),label,25)
    text(d,(70,1725),'own H⁽⁰⁾ 已包含列表、引用、数量，不能解释为纯自身原始状态；G包含拓扑、类型、组存在性和边属性。',28,MUTED)
    text(d,(70,1778),'内部混合张量不要求对应真实可达状态；投影可以为负，向量范数不能直接相加为百分比。',28,MUTED)
    im.save(OUT/'load_changes_and_attribution.png')


def arrow(d,a,b,color,width=4):
    d.line((a,b),fill=color,width=width)
    dx,dy=b[0]-a[0],b[1]-a[1];n=max(math.hypot(dx,dy),1);dx/=n;dy/=n
    s=15;d.polygon([b,(b[0]-s*dx+s*.5*dy,b[1]-s*dy-s*.5*dx),(b[0]-s*dx-s*.5*dy,b[1]-s*dy+s*.5*dx)],fill=color)


def node(d,xy,title,sub,color='#e9f2fb',w=220):
    x,y=xy;d.rounded_rectangle((x-w/2,y-43,x+w/2,y+43),12,fill=color,outline='#92a6bc',width=2)
    text(d,(x,y-22),title,26,bold=True,anchor='mt');text(d,(x,y+14),sub,20,MUTED,anchor='mt')


def flow(evidence):
    im=Image.new('RGB',(3300,2150),BG);d=ImageDraw.Draw(im)
    text(d,(65,40),'Truck → Dump → Load · 两层消息数值通道',55,bold=True)
    text(d,(70,128),'数字＝同源有向边的64D投影消息先相加、再取L2范数；位置在残差/LayerNorm之前。',29,MUTED)
    for ri,role in enumerate(('actor','critic')):
        for ti,t in enumerate(('T0','T1')):
            px=55+ti*1640;py=235+ri*900
            d.rounded_rectangle((px,py,px+1590,py+850),22,fill='white',outline='#d7e1eb',width=2)
            text(d,(px+28,py+25),f'{role.title()} · {t}',38,bold=True)
            text(d,(px+28,py+86),'第1层：直接Truck消息                 第2层：兼容Dump消息',25,MUTED)
            first=[e for e in evidence if e['role']==role and e['snapshot']==t and int(e['layer'])==1]
            grouped=defaultdict(list)
            for e in first:grouped[(e['source'],e['destination'])].append(e)
            trucks=sorted(set(x[0] for x in grouped),key=lambda n:int(n[5:]))
            ys=np.linspace(py+200,py+640,len(trucks))
            dump0=(px+785,py+380);dump1=(px+785,py+700)
            for truck,yy in zip(trucks,ys):
                rows=grouped[(truck,'Dump0')]
                vec=np.array([[float(e[f'additive_{i:02d}']) for i in range(64)] for e in rows]).sum(0)
                norm=float(np.linalg.norm(vec));node(d,(px+170,yy),truck,f'{len(rows)}条消息记录',w=200)
                arrow(d,(px+280,yy),(dump0[0]-115,dump0[1]),'#3d83b8',3)
                xx=(px+280+dump0[0]-115)/2;midy=(yy+dump0[1])/2
                d.rounded_rectangle((xx-52,midy-17,xx+52,midy+22),8,fill='white')
                text(d,(xx,midy),f'{norm:.3f}',23,'#2c6b98',anchor='mm')
            node(d,dump0,'Dump0 H⁽¹⁾',f'{len(trucks)}辆Truck直接传入',color='#fff1db',w=230)
            node(d,dump1,'Dump1 H⁽¹⁾','无Truck直接入边',color='#fff1db',w=230)
            second=[e for e in evidence if e['role']==role and e['snapshot']==t and int(e['layer'])==2]
            for a in range(4):
                dest=f'Load{a}';yy=py+210+a*155
                rows=[e for e in second if e['destination']==dest]
                assert len(rows)==1
                e=rows[0];source=dump0 if e['source']=='Dump0' else dump1
                arrow(d,(source[0]+125,source[1]),(px+1300-120,yy),'#ab713a',5)
                xx=(source[0]+125+px+1300-120)/2;midy=(source[1]+yy)/2
                d.rounded_rectangle((xx-65,midy-18,xx+65,midy+24),8,fill='white')
                text(d,(xx,midy),f"{float(e['projected_additive_norm']):.3f}",24,'#9b642e',anchor='mm')
                node(d,(px+1300,yy),dest,'H⁽²⁾ 接收该消息',color='#e9f5ee',w=230)
            text(d,(px+30,py+774),'此图省略其他入边、presence项及残差；完整数值见edge/group CSV。',23,MUTED)
    text(d,(70,2070),'固定随机初始化；非训练结果。非零消息或向量变化只证明数值通道响应，不能证明正确业务语义。',28,MUTED)
    im.save(OUT/'two_layer_message_flow.png')


def main():
    summary=json.loads((OUT/'summary.json').read_text())
    with np.load(OUT/'representations.npz') as arrays:
        for role in ('actor','critic'):heatmaps(role,arrays)
    changes(summary,csv_rows('representation_changes.csv'))
    flow(csv_rows('message_evidence.csv'))
    print('Rendered actor/critic H layers, load changes/attribution, and two-layer message flow.')


if __name__=='__main__':main()
