"""Render saved physical-intervention responses; never run a model."""
from pathlib import Path
import csv
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT = Path(__file__).resolve().parents[4]
OUT = PROJECT/'reports/token_attention_001/business_graph_summary_009/physical_intervention_001'
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
BOLD='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
INK='#192c40'; MUTED='#586f84'; BG='#f4f7fb'
STAGES=('#aab6c5','#4f86c6','#188c7a')
TITLES={'A':'A · Truck1 完成91.7吨装载', 'B':'B · Truck8 的193吨预留改派', 'C':'C · Dump0 处理能力减半'}


def text(d,xy,s,n=26,color=INK,bold=False,anchor=None):
    d.text(xy,str(s),font=ImageFont.truetype(BOLD if bold else FONT,n),fill=color,anchor=anchor)


def rows(name):
    with (OUT/name).open(encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def panel(d,x,y,w,h):
    d.rounded_rectangle((x,y,x+w,y+h),22,fill='white',outline='#d7e2ed',width=2)


def compact(x):
    return '0' if x == 0 else f'{x:.4g}'


def response_plot():
    records=rows('representation_changes.csv')
    im=Image.new('RGB',(3450,1900),BG); d=ImageDraw.Draw(im)
    text(d,(65,42),'BG128 · 原始观测干预后的 H0 / H1 / H2 响应',54,bold=True)
    text(d,(70,128),'同一历史T1分支；完整重构图、数量和Token；固定随机初始化2026111801；各面板纵轴单独缩放',28,MUTED)
    for k,color in enumerate(STAGES):
        x=2350+k*230; d.rectangle((x,145,x+30,175),fill=color);text(d,(x+42,139),f'H{k}',29)
    for r,role in enumerate(('actor','critic')):
        for col,scene in enumerate(('A','B','C')):
            px=55+col*1135; py=240+r*765
            panel(d,px,py,1080,720)
            text(d,(px+26,py+23),TITLES[scene],33,bold=True)
            text(d,(px+27,py+85),role.title()+' · ||H(干预) − H(BASE)|| L2',28,MUTED)
            values=np.array([[float(next(q['l2'] for q in records if q['scene']==scene and q['role']==role and q['layer']==str(k) and q['node']==f'Load{a}')) for k in range(3)] for a in range(4)])
            cap=float(values.max())*1.25; bx=px+100; by=py+505; ph=300
            for tick in np.linspace(0,cap,5):
                yy=by-tick/cap*ph; d.line((bx,yy,px+1030,yy),fill='#e2e9f1',width=2)
                text(d,(bx-12,yy),compact(tick),23,MUTED,anchor='rm')
            for a in range(4):
                gx=bx+100+a*230
                for k,color in enumerate(STAGES):
                    value=values[a,k]; xx=gx+(k-1)*52; yy=by-value/cap*ph
                    if value>0:d.rectangle((xx-17,yy,xx+17,by),fill=color)
                text(d,(gx,by+20),f'Load{a}',29,bold=True,anchor='mt')
                for k,color in enumerate(STAGES):
                    text(d,(gx,by+72+k*30),f'H{k}  {compact(values[a,k])}',22,color,anchor='mt')
            dump0=float(next(q['l2'] for q in records if q['scene']==scene and q['role']==role and q['layer']=='0' and q['node']=='Dump0'))
            text(d,(px+32,py+675),f'Dump0 的 H0 响应：{compact(dump0)}；完整19节点数值见CSV/NPZ',23,MUTED)
    text(d,(70,1815),'向量变化表示数值敏感性；不表示学习到业务语义、重要性、单调偏好或策略收益。H0已含邻居引用和确定性数量。',27,MUTED)
    im.save(OUT/'encoder_response.png')


def readout_plot():
    candidates=rows('candidate_changes.csv'); logits=rows('actor_logits.csv')
    im=Image.new('RGB',(3450,1830),BG);d=ImageDraw.Draw(im)
    text(d,(65,42),'BG128 Actor · 候选表示与四动作评分已接通',54,bold=True)
    text(d,(70,128),'每个分支重新认证并编码C128；同一Actor参数；本图为随机初始化响应，正负方向不代表正确决策',28,MUTED)
    for col,scene in enumerate(('A','B','C')):
        px=55+col*1135
        panel(d,px,225,1080,635)
        text(d,(px+26,248),TITLES[scene],33,bold=True)
        text(d,(px+27,310),'4 × 64D候选表示：相对BASE的L2变化',27,MUTED)
        values=[float(next(r['l2'] for r in candidates if r['scene']==scene and r['action']==str(a) and r['block']=='candidates')) for a in range(4)]
        cap=max(values)*1.3; by=735; ph=310;bx=px+115
        for tick in np.linspace(0,cap,5):
            yy=by-tick/cap*ph; d.line((bx,yy,px+1025,yy),fill='#e4ebf2',width=2)
            text(d,(bx-14,yy),compact(tick),23,MUTED,anchor='rm')
        for a,value in enumerate(values):
            xx=bx+110+a*220; yy=by-value/cap*ph
            d.rectangle((xx-42,yy,xx+42,by),fill='#258c7c')
            text(d,(xx,yy-13),compact(value),25,'#197967',anchor='mb')
            text(d,(xx,by+25),f'Load{a}',29,bold=True,anchor='mt')
        panel(d,px,900,1080,695)
        text(d,(px+27,924),'四动作 Logit 的变化（单位 × 10^-3）',31,bold=True)
        text(d,(px+27,985),'灰：原始变化；蓝：减去四动作平均变化后',25,MUTED)
        delta=np.array([float(next(r['delta_unmasked_logit'] for r in logits if r['scene']==scene and r['action']==str(a))) for a in range(4)])*1000
        centered=delta-delta.mean();limit=max(np.abs(delta).max(),np.abs(centered).max())*1.35
        zero=1255;ph=155;bx=px+115
        for tick in (-limit,0,limit):
            yy=zero-tick/limit*ph;d.line((bx,yy,px+1025,yy),fill='#d9e2ec',width=3 if tick==0 else 2)
            text(d,(bx-14,yy),f'{tick:.2f}',23,MUTED,anchor='rm')
        for a in range(4):
            xx=bx+110+a*220
            for j,vector in enumerate((delta,centered)):
                value=vector[a];x=xx+(j-.5)*55;yy=zero-value/limit*ph;color='#a7b3c3' if j==0 else '#477fc0'
                d.rectangle((x-19,min(zero,yy),x+19,max(zero,yy)),fill=color)
                text(d,(x,yy+(-12 if value>=0 else 12)),f'{value:+.3f}',21,color,anchor='mb' if value>=0 else 'mt')
            text(d,(xx,1482),f'Load{a}',29,bold=True,anchor='mt')
        text(d,(px+28,1540),'原始/Mask后logits、概率及两动作差值均已保存CSV',23,MUTED)
    text(d,(70,1648),'本例四个动作均合法，Mask与候选吨位保持一致。减均值只去掉Softmax不敏感的公共平移，不改变任意两动作Logit差。',27,MUTED)
    text(d,(70,1700),'总评分同时接收图池化、Ego/Load/Dump直接读出、明确数量和C128；不能将Logit响应全部归于某一条两跳路径。',27,MUTED)
    im.save(OUT/'actor_readout_response.png')


def main():
    response_plot();readout_plot()
    print('Rendered encoder_response.png and actor_readout_response.png from saved data only.')


if __name__=='__main__':main()
