"""Render exact BG009 fact graphs with fixed schematic coordinates using Pillow.

No model is instantiated. Bidirectional message pairs may share a drawn curve;
the exact directed multigraph and all continuous attributes remain in the CSVs.
"""
from pathlib import Path
from collections import defaultdict
import argparse
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W,H = 3600,2500
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
BOLD='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
BG='#f5f7fb'; INK='#17263a'; MUTED='#576a7f'
GROUP={'transport':'#6b91b9','commitment':'#17845f','service':'#8855b2','material':'#bc732b'}
STATUS={'added':'#008b72','removed':'#d14351','changed':'#e3a025','unchanged':'#b8c4d1'}
NAMES=[f'Truck{i}' for i in range(9)]+[f'Load{i}' for i in range(4)]+['Dump0','Dump1','Junction2','Junction3','Junction4','Global']
POS={0:(340,440),1:(810,440),2:(1280,440),3:(1750,440),4:(2220,440),
     5:(450,1950),6:(1010,1950),7:(1570,1950),8:(2130,1950),
     9:(980,820),10:(1810,820),11:(2280,820),12:(1400,820),
     13:(300,1200),14:(1240,1590),15:(720,1200),16:(1240,1200),17:(1940,1200),18:(2380,1580)}
Q_NAMES=('reserved_truck_count','reserved_tons','inbound_loaded_truck_count','inbound_loaded_tons',
 'inbound_known_grade_truck_count','inbound_known_grade_tons','inbound_known_metal_tons',
 'queue_unique_count','queue_instance_count','seize_unique_count','seize_instance_count',
 'capacity_wait_unique_count','capacity_wait_instance_count','service_unique_count','service_instance_count',
 'admission_count','at_node_truck_count','incoming_road_count','outgoing_road_count',
 'incoming_segment_truck_count','outgoing_segment_truck_count')
LABEL={'road':'道路','truck_at_node':'所在地点','truck_segment_from':'路段起点',
 'truck_segment_to':'路段终点','truck_predecessor':'首次派车前驱','reserved_load':'预留装载',
 'loaded_dump':'载货目标','queue':'排队','seize':'资源等待','capacity_wait':'容量等待',
 'service':'正在服务','admission':'准入待处理','compatible_material':'物料兼容'}


def font(n=27,bold=False):return ImageFont.truetype(BOLD if bold else FONT,n)


def text(d,xy,s,n=27,fill=INK,bold=False,anchor=None):
    d.text(xy,str(s),font=font(n,bold),fill=fill,anchor=anchor)


def wrap(d,s,width,n=26):
    result=[];line=''
    for char in str(s):
        if d.textlength(line+char,font=font(n))>width and line:
            result.append(line);line=char
        else:line+=char
    if line:result.append(line)
    return result


def paragraph(d,xy,s,width,n=26,fill=MUTED,gap=10):
    x,y=xy
    for line in wrap(d,s,width,n):
        text(d,(x,y),line,n,fill);y+=n+gap
    return y


def boxsize(i):
    return (250,164) if i<9 else (270,150) if i<15 else (170,100) if i<18 else (280,150)


def crop_center(i,toward):
    x,y=POS[i];tx,ty=toward;vx,vy=tx-x,ty-y
    bw,bh=boxsize(i)
    ratio=1/max(abs(vx)/(bw/2+8),abs(vy)/(bh/2+8),1e-9)
    return (x+vx*ratio,y+vy*ratio)


def curve_points(a,b,offset):
    x,y=POS[a];u,v=POS[b];dx,dy=u-x,v-y;l=max(math.hypot(dx,dy),1)
    ctrl=((x+u)/2-dy/l*offset,(y+v)/2+dx/l*offset)
    start=crop_center(a,ctrl);end=crop_center(b,ctrl)
    pts=[]
    for t in np.linspace(0,1,90):
        pts.append(((1-t)**2*start[0]+2*(1-t)*t*ctrl[0]+t*t*end[0],
                    (1-t)**2*start[1]+2*(1-t)*t*ctrl[1]+t*t*end[1]))
    return pts


def stroke(d,pts,color,width,dashed=False):
    if not dashed:d.line(pts,fill=color,width=width,joint='curve');return
    length=0
    for a,b in zip(pts,pts[1:]):
        if int(length/15)%2==0:d.line([a,b],fill=color,width=width)
        length+=math.dist(a,b)


def arrow(d,tip,prev,color,size=15):
    dx,dy=tip[0]-prev[0],tip[1]-prev[1];z=max(math.hypot(dx,dy),1e-9);dx/=z;dy/=z
    d.polygon([tip,(tip[0]-size*dx+size*.55*dy,tip[1]-size*dy-size*.55*dx),
               (tip[0]-size*dx-size*.55*dy,tip[1]-size*dy+size*.55*dx)],fill=color)


def pill(d,xy,s,color,n=21):
    x,y=xy;tw=d.textlength(s,font=font(n));rect=(x-tw/2-9,y-n/2-8,x+tw/2+9,y+n/2+12)
    d.rounded_rectangle(rect,9,fill='white',outline=color,width=1)
    text(d,(x,y),s,n,color,anchor='mm')


def raw(s):return np.asarray(s.get('raw',s.get('raw_state')),np.float32)


def getstatus(e):
    s=e.get('diff_status',e.get('status','unchanged'))
    return {'new':'added','disappeared':'removed','modified':'changed','same':'unchanged'}.get(s,s)


def bundles(edges):
    merged={}
    for e in edges:
        r=e['relation'];src,dst=int(e['source']),int(e['destination'])
        attr=e['attributes'];rev=r.endswith('_reverse');base=r.removesuffix('_reverse')
        if rev:src,dst=dst,src
        # Preserve two independently observed directed roads as one two-arrow
        # schematic segment only when their endpoints are opposite.
        if base=='road':src,dst=sorted((src,dst))
        key=(src,dst,base,tuple(sorted(attr.items())) if base!='road' else ())
        if key not in merged:
            merged[key]=dict(src=src,dst=dst,relation=base,group=e['group'],attr=attr,
                             records=[],forward=False,reverse=False,status=getstatus(e))
        z=merged[key];z['records'].append(e)
        if r=='road':
            if int(e['source'])==src:z['forward']=True
            else:z['reverse']=True
        elif rev:z['reverse']=True
        else:z['forward']=True
        if getstatus(e)!='unchanged':z['status']=getstatus(e)
    return list(merged.values())


def build_union(a,b):
    # T0/T1 already carry multiplicity-aware statuses from the formal audit.
    out=list(b)
    out.extend(e for e in a if getstatus(e)=='removed')
    return out


def draw_edges(d,edges,mode):
    bb=bundles(edges)
    order={'transport':0,'material':1,'service':2,'commitment':3}
    bb.sort(key=lambda e:(order[e['group']],e['src'],e['dst'],e['relation']))
    pair_counts=defaultdict(int)
    for z in bb:
        a,b=z['src'],z['dst'];base=z['relation'];pair=tuple(sorted((a,b)))
        k=pair_counts[pair];pair_counts[pair]+=1
        offset=0
        if base=='road':offset=0
        elif base=='compatible_material':offset=80+(a-9)*28
        elif base=='reserved_load':offset=-70-(a%3)*35
        elif base=='loaded_dump':offset=65+(a%3)*35
        elif z['group']=='service':offset=130+(a%3)*20+k*30
        elif base=='truck_predecessor':offset=-155
        else:offset=35+k*24
        pts=curve_points(a,b,offset)
        status=z['status'];color=GROUP[z['group']]
        if mode=='diff':
            color=STATUS.get(status,STATUS['unchanged'])
            width=6 if status!='unchanged' else 2
        else:
            width=5 if z['group']!='transport' else 3
            if status!='unchanged':stroke(d,pts,STATUS.get(status,'#e3a025'),width+8)
        dashed=(base=='truck_predecessor' or status=='removed' or base=='compatible_material')
        stroke(d,pts,color,width,dashed)
        if z['forward']:arrow(d,pts[-1],pts[-5],color)
        if z['reverse']:arrow(d,pts[0],pts[4],color)
        # Detailed road and positioning attrs are in the CSV. Keep main graph
        # labels focused on factual business roles, not every parallel channel.
        label=''
        if base in ('reserved_load','loaded_dump'):
            label=f"{LABEL[base]} {z['attr'].get('tons_193',0)*193:.0f}t"
        elif z['group']=='service':
            label=LABEL.get(base,base)
            if z['attr'].get('position_valid',0):label+=f" #{round(z['attr']['position_9']*9)}"
            instances=max(sum(not e['relation'].endswith('_reverse') for e in z['records']),1)
            if instances>1:label+=f' ×{instances}'
        elif mode=='diff' and status!='unchanged':label=LABEL.get(base,base)
        if label:
            index=round(len(pts)*(.38 if base=='reserved_load' else .62 if base=='loaded_dump' else .5))
            pill(d,pts[index],label,color)


def node_lines(s,i):
    x=raw(s);t=x[27:1107].reshape(9,120);q=np.asarray(s['raw_quantities'])
    if i<9:
        v=t[i];phase=int(v[4:16].argmax()) if v[4:16].sum() else -1
        observed=[e for e in s['edges'] if e['source']==i and
                  e['relation'] in ('reserved_load','loaded_dump')]
        target=NAMES[int(observed[0]['destination'])] if observed else '未观测目标'
        tons=(v[19] if v[17] else v[20])*193
        grade=(f'载荷品位 {v[22]:.3f}' if v[17] and v[21] and v[18]
               else '载荷物料：废石' if v[17] and not v[18]
               else '载荷品位：未知' if v[20]>0 else '当前无载荷品位')
        return [f'phase={phase}  '+('已载' if v[17] else '未载'),f'{tons:.1f}t → {target}',grade]
    if i<13:
        v=x[1107+47*(i-9):1154+47*(i-9)]
        return [f"{'矿石' if v[1] else '废石'}  品位 {v[2]:.3f}",
                f'预留 {q[i,0]:.0f}车 / {q[i,1]:.1f}t']
    if i<15:
        return [f'载货目标 {q[i,2]:.0f}车 / {q[i,3]:.1f}t',
                f'容量等待 {q[i,11]:.0f}  服务 {q[i,13]:.0f}']
    if i<18:return [f'在节点车辆 {q[i,16]:.0f}']
    return [f'窗口 {int(round(float(x[2])*4))+1}/4', 'C128 单独保留；无事实边']


def semantic_key(s,i):
    x=raw(s)
    if i<9:
        t=x[27+120*i:147+120*i]
        return tuple(t[np.r_[0,4:32]].round(6))
    if i<15:return tuple(np.asarray(s['raw_quantities'])[i].round(4))
    return ()


def draw_nodes(d,s,other,mode):
    ego=int(s['ego_id'])
    for i in range(19):
        x,y=POS[i];bw,bh=boxsize(i)
        fill='#eaf1fb' if i<9 else '#e8f5ed' if i<13 else '#fff1df' if i<15 else '#edf0f5' if i<18 else '#f1ebfa'
        changed=semantic_key(s,i)!=semantic_key(other,i)
        outline='#e3a025' if changed else '#98aabe'
        if i==ego:outline='#2763c4'
        d.rounded_rectangle((x-bw/2,y-bh/2,x+bw/2,y+bh/2),20,fill=fill,outline=outline,width=5 if changed or i==ego else 2)
        text(d,(x,y-bh/2+21),NAMES[i]+(' · EGO' if i==ego else ''),29,INK,True,anchor='mt')
        for j,line in enumerate(node_lines(s,i)):
            text(d,(x,y-bh/2+(61+29*j if i<9 else 65+34*j)),line,21 if i<9 else 22,MUTED,anchor='mt')
        if i<9 and changed:
            d.ellipse((x+bw/2-15,y-bh/2-15,x+bw/2+15,y-bh/2+15),fill='#e3a025',outline='white',width=3)


def sidepanel(d,s,other,mode):
    left,right=2710,3530;y=330
    d.rounded_rectangle((left,y,right,2320),24,fill='white',outline='#d6dfea',width=2)
    x=left+32;y+=28
    text(d,(x,y),'候选读取索引',36,bold=True);y+=60
    y=paragraph(d,(x,y),'紫色卡片仅表示评分读取；不属于事实边，不增加预留吨位。',right-x-28,25);y+=18
    rawx=raw(s);ego=int(s['ego_id'])
    for a,dump in enumerate(s['candidate_dump_ids']):
        d.rounded_rectangle((x,y,right-30,y+97),14,fill='#f7f3fc',outline='#b399ca',width=2)
        text(d,(x+18,y+12),f'Truck{ego} → Load{a} ＋ {NAMES[int(dump)]}',26,INK,True)
        text(d,(x+18,y+53),f"合法={int(rawx[19+a])}   可派 {rawx[23+a]*193:.1f}t",24,MUTED)
        y+=110
    y+=16;text(d,(x,y),'事实关系分组',33,bold=True);y+=57
    for key,label in [('transport','transport · 道路/定位/初始前驱'),('commitment','commitment · 预留/载货目标'),('service','service · 排队/等待/服务'),('material','material · 物料兼容与反馈')]:
        d.line((x,y+17,x+62,y+17),fill=GROUP[key],width=7)
        text(d,(x+82,y),label,25);y+=49
    y+=19;text(d,(x,y),'相邻快照变化',33,bold=True);y+=57
    for key,label in [('added','新增：仅T1存在'),('removed','消失：仅T0存在'),('changed','变化：同关系属性改变')]:
        d.line((x,y+17,x+62,y+17),fill=STATUS[key],width=9)
        text(d,(x+82,y),label,25);y+=48
    y=paragraph(d,(x,y+16),'快照图：内线色＝关系组，外沿色＝变化；差异图：线色＝变化。双箭头合画正/逆消息对，CSV保留每条有向记录与多重性。',right-x-28,24)
    y+=32;text(d,(x,y),'输入与传播边界',32,bold=True);y+=53
    for note in ['19个节点 · 16条有向道路',
                 '25种关系是Schema词表，非每图25条',
                 'Global旁路；C128不参与事实构图',
                 '线宽不代表注意力；本次无模型Forward',
                 '道路连接为信息通道，不是可通行证明',
                 '初始前驱 ≠ 当前运动前车']:
        y=paragraph(d,(x,y),'• '+note,right-x-28,24);y+=9
    if '_reward' in s:
        reward=s['_reward'];y+=25
        text(d,(x,y),'此刻已收到的上一动作结果',30,bold=True);y+=54
        for note in [f"原RPC reward：{reward['received_rpc_reward']:.9f}",
                     f"同区间008成本：{reward['incoming_current008_cost']['joint']:.9f}",
                     '本区间品位/生产成本均为0，仅等待',
                     '窗口0已结算；窗口1/2/3尚未结算']:
            y=paragraph(d,(x,y),note,right-x-28,24);y+=9
        y=paragraph(d,(x,y+8),'奖励为审计旁注，不进入19节点或C128；原RPC reward与008训练成本定义不同。',right-x-28,23)


def draw_one(data,name,out):
    a,b=data['T0'],data['T1'];mode='diff' if name=='diff' else 'snapshot'
    s=b if name in ('T1','diff') else a;other=a if s is b else b
    image=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(image)
    title='相邻决策关系变化' if name=='diff' else f'{name} · 19节点事实业务图'
    text(d,(80,55),title,58,bold=True)
    selection=data.get('selection',{})
    run=selection.get('run_index',selection.get('run',9810998))
    t0=float(a['time_seconds']);t1=float(b['time_seconds'])
    subtitle=(f'历史 TQ128 · run {run} · T0 {t0:.3f}s → T1 {t1:.3f}s · Δt={t1-t0:.3f}s'
              if name=='diff' else f"历史 TQ128 · run {run} · 原始 state[{s['state_index']}] · t={float(s['time_seconds']):.3f}s · Ego=Truck{s['ego_id']}")
    text(d,(84,145),subtitle,31,MUTED)
    text(d,(84,205),'BG009 / s01_BG128 输入重构审计  ·  固定示意布局（非真实坐标）  ·  无训练、无模型推理',27,MUTED)
    d.rounded_rectangle((60,305,2660,2155),24,fill='white',outline='#d6dfea',width=2)
    edges=build_union(a['edges'],b['edges']) if name=='diff' else s['edges']
    draw_edges(d,edges,mode);draw_nodes(d,s,other,mode);sidepanel(d,s,other,mode)
    y=2200;text(d,(85,y),'业务变化焦点（T0→T1事后对照）',33,bold=True)
    t_a=raw(a)[27:1107].reshape(9,120);t_b=raw(b)[27:1107].reshape(9,120)
    load_events=[];unload_events=[]
    for i in range(9):
        if not t_a[i,17] and t_b[i,17]:load_events.append(f'Truck{i} 装载完成 {t_b[i,19]*193:.1f}t')
        if t_a[i,17] and not t_b[i,17]:unload_events.append(f'Truck{i} 载荷清空')
    y=paragraph(d,(85,y+58),'；'.join(load_events+unload_events)+f'；决策车辆 Truck{a["ego_id"]} → Truck{b["ego_id"]}。',2500,29,INK)
    y=paragraph(d,(85,y+14),'T0、T1各由自身原始观测独立构图；候选卡片为直接读取索引。两跳审计检验 Truck→Dump→Load 的拓扑通道，不声称已学会利用。',2500,26,MUTED)
    text(d,(85,2435),'完整关系与边属性：graph_edges_T0/T1.csv    两层路径：two_hop_paths.csv    奖励时序：reward_audit.csv',24,MUTED)
    path=out/('snapshot_diff.png' if name=='diff' else f'snapshot_{name}.png')
    image.save(path)
    return path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    data=json.loads((args.output/'snapshots.json').read_text())
    reward_path=args.output/'reward_details.json'
    if reward_path.exists():
        rewards=json.loads(reward_path.read_text())
        for name in ('T0','T1'):data[name]['_reward']=rewards['snapshots'][name]
    outputs=[draw_one(data,n,args.output) for n in ('T0','T1','diff')]
    (args.output/'layout.json').write_text(json.dumps(dict(canvas=[W,H],positions={NAMES[k]:v for k,v in POS.items()},
        fixed_across_snapshots=True,schematic_not_physical_coordinates=True,
        directed_multigraph_preserved_in_CSV=True,reverse_message_pairs_bundled_in_png=True),ensure_ascii=False,indent=2)+'\n')
    for path in outputs:print(path)


if __name__=='__main__':main()
