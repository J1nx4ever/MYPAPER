"""Render saved trained-intervention evidence and seal its compact report.

This script never loads a model or performs a neural forward.
"""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import csv
import hashlib
import json
import math

from PIL import Image,ImageDraw,ImageFont
import numpy as np

PROJECT=Path(__file__).resolve().parents[4]
PLAN=Path(__file__).resolve().parent
OUT=PROJECT/'reports/token_attention_001/business_graph_summary_009/trained_intervention_001'
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
BOLD='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
INK='#183047';MUTED='#526c82';BG='#f3f7fb';BLUE='#3981b8';ORANGE='#de9546';GREEN='#168778'
COLORS=('#a6b5c4','#4287bd','#158b79')
TITLES={'A':'A · Truck1 完成已知装载','B':'B · 193 吨预留由 Load0 → Load1','C':'C · Dump0 处理速率减半'}

def read(n):return json.loads((OUT/n).read_bytes())
def rows(n):
    with (OUT/n).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p),bytes=Path(p).stat().st_size)
def text(d,xy,s,size=27,color=INK,bold=False,anchor=None):
    d.text(xy,str(s),font=ImageFont.truetype(BOLD if bold else FONT,size),fill=color,anchor=anchor)
def panel(d,x,y,w,h):d.rounded_rectangle((x,y,x+w,y+h),22,fill='white',outline='#d7e3ee',width=2)
def compact(v):return '0' if v==0 else f'{v:.4g}'


def encoder_figure():
    rr=rows('representation_changes.csv');im=Image.new('RGB',(3300,1900),BG);d=ImageDraw.Draw(im)
    text(d,(55,35),'已训练 BG128 · 候选 Load 的逐层表征响应',51,bold=True)
    text(d,(60,117),'正式 Checkpoint13；同一历史 BASE 的独立物理分支；纵轴为相对 BASE 的 L2，面板各自缩放',28,MUTED)
    for k,c in enumerate(COLORS):
        x=2460+k*240;d.rectangle((x,165,x+28,193),fill=c);text(d,(x+42,156),'H'+str(k),27)
    for ri,role in enumerate(('actor','critic')):
        for ci,scene in enumerate('ABC'):
            x=45+ci*1090;y=230+ri*770;panel(d,x,y,1040,725)
            text(d,(x+24,y+23),TITLES[scene],30,bold=True)
            text(d,(x+25,y+81),role.title()+' · H0 / H1 / H2',28,MUTED)
            vals=np.array([[float(next(r['l2'] for r in rr if r['role']==role and r['scene']==scene and r['node']=='Load'+str(a) and r['layer']==str(k))) for k in range(3)] for a in range(4)])
            cap=max(float(vals.max())*1.22,1e-9);by=y+475;ph=290;bx=x+110
            for tick in np.linspace(0,cap,5):
                yy=by-tick/cap*ph;d.line((bx,yy,x+1005,yy),fill='#e0e8f1',width=2)
                text(d,(bx-14,yy),compact(tick),22,MUTED,anchor='rm')
            for a in range(4):
                gx=bx+105+a*220
                for k,c in enumerate(COLORS):
                    v=vals[a,k];xx=gx+(k-1)*48;yy=by-v/cap*ph
                    if v>0:d.rectangle((xx-15,yy,xx+15,by),fill=c)
                text(d,(gx,by+16),'Load'+str(a),27,bold=True,anchor='mt')
                for k,c in enumerate(COLORS):text(d,(gx,by+67+k*30),f'H{k}  {compact(vals[a,k])}',22,c,anchor='mt')
            text(d,(x+25,y+672),'数值为表征变化，不是收益、重要性比例或语义正确率。',23,MUTED)
    text(d,(60,1813),'A 的 Load1/2、C 的所有 Load：H0 不变而 H1 开始变化，验证邻居消息通道；H0 本身已包含引用和明确数量。',27,MUTED)
    im.save(OUT/'encoder_response.png')


def signed_bars(d,x,y,w,h,vectors,colors,labels):
    values=np.asarray(vectors,dtype=float);limit=max(float(np.abs(values).max())*1.23,1e-9)
    bx=x+100;right=x+w-30;zero=y+h*.50;ph=h*.34
    for tick in (-limit,0,limit):
        yy=zero-tick/limit*ph;d.line((bx,yy,right,yy),fill='#dbe5ef',width=3 if tick==0 else 2)
        text(d,(bx-13,yy),f'{tick:+.3f}' if tick else '0',22,MUTED,anchor='rm')
    for a in range(4):
        gx=bx+100+a*(right-bx-120)/3
        for j,vec in enumerate(values):
            v=float(vec[a]);xx=gx+(j-(len(values)-1)/2)*46;yy=zero-v/limit*ph
            if abs(v)>1e-12:d.rectangle((xx-16,min(yy,zero),xx+16,max(yy,zero)),fill=colors[j])
        text(d,(gx,y+h-16),'Load'+str(a),27,bold=True,anchor='mm')
        if len(values)==1:
            v=values[0,a];yy=zero-v/limit*ph
            text(d,(gx,yy+(-12 if v>=0 else 12)),f'{v:+.3f}',23,colors[0],anchor='mb' if v>=0 else 'mt')


def actor_figure():
    summary=read('summary.json');decomp=rows('readout_decomposition.csv')
    im=Image.new('RGB',(3300,1900),BG);d=ImageDraw.Draw(im)
    text(d,(55,35),'已训练 Actor · 概率变化与 Graph / C128 读取分解',50,bold=True)
    text(d,(60,115),'上：完整物理分支；下：读出接口四组合机制探针。单位均为概率百分点；原始 Logit 的公共平移已另存 CSV。',27,MUTED)
    for ci,scene in enumerate('ABC'):
        x=45+ci*1090;panel(d,x,215,1040,690)
        text(d,(x+25,238),TITLES[scene],30,bold=True)
        text(d,(x+25,295),'完整分支 ΔP（相对共同 BASE）',27,MUTED)
        delta=(np.array(summary['primary_results'][scene]['probabilities'])-summary['primary_results']['BASE']['probabilities'])*100
        signed_bars(d,x,360,1040,465,[delta],[GREEN],['完整分支'])
        panel(d,x,955,1040,730)
        text(d,(x+25,980),'机制探针：两条读取支路的分解',29,bold=True)
        for j,(label,c) in enumerate((('Graph',BLUE),('C128',ORANGE))):
            xx=x+35+j*260;d.rectangle((xx,1043,xx+25,1068),fill=c);text(d,(xx+40,1035),label,26)
        r=[r for r in decomp if r['scene']==scene and r['role']=='actor' and r['target']=='probabilities']
        graph=np.array([float(z['graph_interface_shapley'])*100 for z in r]);c128=np.array([float(z['C128_interface_shapley'])*100 for z in r])
        signed_bars(d,x,1100,1040,445,[graph,c128],[BLUE,ORANGE],['Graph','C128'])
        for a in range(4):
            gx=x+200+a*248
            text(d,(gx,1580),f'G {graph[a]:+.3f}',21,BLUE,anchor='mt')
            text(d,(gx,1615),f'C {c128[a]:+.3f}',21,ORANGE,anchor='mt')
    text(d,(60,1740),'C：方向主要来自 C128，图支路对 Load2 概率变化仅 +0.00521 点。B：图支路抑制 Load1，但摘要支路反向抵消。',28,bold=True)
    text(d,(60,1800),'训练速率始终2500吨/小时，C的1250为参数外推；支路分解是机制探针，不能据此单独判定两层GAT失效。',26,MUTED)
    im.save(OUT/'actor_and_fusion_response.png')


def arrow(d,a,b,c=BLUE,width=7):
    d.line((*a,*b),fill=c,width=width);angle=math.atan2(b[1]-a[1],b[0]-a[0]);r=23
    d.polygon((b,(b[0]-r*math.cos(angle-.4),b[1]-r*math.sin(angle-.4)),(b[0]-r*math.cos(angle+.4),b[1]-r*math.sin(angle+.4))),fill=c)


def paths_figure():
    im=Image.new('RGB',(3000,1600),BG);d=ImageDraw.Draw(im)
    text(d,(60,35),'同一事实的多条编码通道，数量仍只计一次',50,bold=True)
    text(d,(65,117),'A 的局部消息通道：Truck1 完成91.7吨装载，成为目标 Dump0 的已载车。完整19节点和所有边见 graph_edges.csv。',27,MUTED)
    boxes=[(100,330,650,580,'Truck1 H0','已载91.7吨 / 目标Dump0'),
        (1050,330,1600,580,'Dump0 H1','在途车辆5→6；771.7→863.4吨'),
        (2050,330,2850,580,'Load2 H2','自身原始状态、H0不变')]
    for x1,y1,x2,y2,t,s in boxes:
        panel(d,x1,y1,x2-x1,y2-y1);text(d,((x1+x2)/2,y1+55),t,40,bold=True,anchor='mm');text(d,((x1+x2)/2,y1+150),s,27,MUTED,anchor='mm')
    arrow(d,(650,415),(1040,415));text(d,(835,335),'L1 commitment',25,BLUE,anchor='mm')
    text(d,(835,485),'loaded_dump',24,BLUE,anchor='mm')
    arrow(d,(1600,415),(2040,415),GREEN);text(d,(1815,335),'L2 material',25,GREEN,anchor='mm')
    text(d,(1815,485),'Dump → Load',24,GREEN,anchor='mm')
    text(d,(110,630),'另一条更早的通道：唯一在途数量 → Dump0 H0 → 第一层 material → Load2 H1；不是在层内连续执行两跳。',28,GREEN,bold=True)
    panel(d,100,720,2750,700)
    text(d,(135,755),'A → Load2，Actor H2 响应的成对阻断检查',36,bold=True)
    lines=[('完整消息通道','0.0241590'),('仅阻断 Truck1 → Dump0 的第一层 loaded_dump 边','0.0245383'),
        ('两层均去掉 material 组消息','0.000303568'),('两层均去掉全部消息；H0、数量、C128仍保留完整分支','0')]
    for i,(name,value) in enumerate(lines):
        yy=860+i*104;text(d,(145,yy),name,29);text(d,(2675,yy),value,32,BLUE,bold=True,anchor='ra')
    text(d,(140,1308),'单条 loaded_dump 边并非唯一通道：Dump0 的 H0 已通过明确数量得到该事实，再由 material 边传给 Load。',28,bold=True)
    text(d,(65,1480),'阻断后的变化不是独占因果份额。直接读取 Dump、候选 Load 和 C128 是不同编码视图，网络不把这些视图的吨位相加。',27,MUTED)
    im.save(OUT/'message_paths_and_counts.png')


def report():
    s=read('summary.json');review=read('independent_review.json');support=read('training_support_audit.json');r=s['primary_results'];rep=rows('representation_changes.csv');dec=rows('readout_decomposition.csv')
    lines=['# BG128 Checkpoint13：A/B/C 原始观测物理干预', '',
        '**结论：已训练网络对业务变化存在数值响应，但响应来源不同。B的图支路确实抑制新增预留的Load1，摘要支路反向抵消；C的正确局部方向主要由C128承担，GAT对最终评分的影响很弱。但C的1250吨/小时超出训练固定2500吨/小时的参数支持域，不能据此单独判定两层GAT失败。A同时释放装载服务、转移预留和新增已载物料，不能用单因素解释。**', '',
        '本轮使用正式训练 Checkpoint13，未使用旧随机初始化权重；主A/B/C每次都从同一历史T1原始观测独立构建，正式重算19节点事实图、16条道路、25种关系、21维数量、Token、C128和Actor/Critic。四个动作均合法，候选吨位均为91.7吨，没有Mask变化造成的概率重分配。', '',
        '| 分支 | Load0概率 | Load1概率 | Load2概率 | Load3概率 | 联合成本V |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for scene in ('BASE','A','B','C'):
        q=r[scene];lines.append('| '+scene+' | '+' | '.join(f'{p*100:.5f}%' for p in q['probabilities'])+f" | {q['critic_value']:.8f} |")
    lines+=['', '## 分支定义与方向', '',
        '- A：Truck1完成原已知的91.7吨、品位0.3757197装载，Load0服务1→0、预留284.7→193吨；Dump0在途车辆5→6、吨位771.7→863.4。Load0概率0.23983%→0.39541%，V下降0.305224。该分支同时包含释放服务和新增高品位在途，不能把结果解释成“拥堵增加却更偏好Load0”或单独的品位反应。',
        '- B：Truck8的193吨未开始服务预留由Load0转移到Load1，保留当前路段并改未来路线尾。Load1概率5.25437%→3.62499%，与避免新增长期承诺的局部方向一致；但改派同时改变目的地、材料承诺和未来路程，没有反事实真实回报，V下降0.568054不能被单独判为正确或错误。',
        '- C：只改原始dump.0.crusherRate（索引1304），2500→1250吨/小时，busy/FIFO/库存/过去累计量不变，确实是处理能力减半。矿石动作总概率86.71108%→84.81244%，废石Load3上升1.89863百分点，V上升0.617700；这一局部方向与处理能力下降一致。Load0/1的各自概率仍会上升，不能要求所有矿石动作都单调下降。', '',
        '## 训练支持域与C的外推限制', '',
        '独立读取全部16个冻结phase和512个job，`config.runtime_scene.dump.0.crusherRateTph`唯一配置值为2500.0。C仅在离线原始观测将其改为1250，不是正式训练中出现过的处理速率参数。因此此例可以观察外推响应与支路依赖，不能将图支路幅度小直接解释为两层GAT表达能力失败，也不能宣称网络从变化处理速率的训练样本学会降容语义。该核对检查配置，未额外遍历所有训练原始状态列。', '',
        '原始C128的冻结工程实现显式使用 `rate=raw[1304]*2500/3600`，然后按该rate重算FIFO投影、窗口处理capacity、可调整数量和品位需求。本C有8个摘要特征随确定性公式改变；读取这些投影的权重经过训练，但投影本身不是神经网络学出的物理模型。应将“工程投影自然响应rate变化”和“策略学会利用投影”分别表述。A的内部暂态、B的外生改派也均不是实际执行的PPO可选转移。证据见[训练支持域审计](training_support_audit.json)。', '',
        '## 逐层传播', '', '[表征响应图](encoder_response.png) 显示候选Load的H0/H1/H2差异。下表均为Actor的L2变化；Critic完整数据见CSV，两个独立模型的向量范数不可直接当作跨模型重要性比较。', '',
        '| 分支 | 候选 | H0 | H1 | H2 |', '| --- | --- | ---: | ---: | ---: |']
    for scene in 'ABC':
        for a in range(4):
            vals=[float(next(x['l2'] for x in rep if x['role']=='actor' and x['scene']==scene and x['node']=='Load'+str(a) and x['layer']==str(k))) for k in range(3)]
            lines.append('| '+scene+f' | Load{a} | '+' | '.join(f'{v:.8g}' for v in vals)+' |')
    lines+=['',
        'A的Load0在H0已变化，来自本地状态、服务列表和明确数量；Load1/2自身原始字段与H0不变，H1已出现变化，说明邻居消息已传入。B的Load0/1在H0就变化，第一层commitment消息进一步放大；Actor Load1的第一层该组投影消息变化L2=1.009741。C所有Load的H0均不变，H1开始变化，存在Dump/Truck相关邻居通道，且不是靠修改Load自身输入造成。', '',
        '“Truck→Dump→Load”拓扑和实际GAT数值通道存在。但H0本身有引用与明确数量，所以同一事实也能先进入Dump0 H0，再通过一次material消息传到Load H1；不能将所有传播强行归为第二层。精确一跳/两跳方向、原实体和边属性见[incoming_paths.csv](incoming_paths.csv)，四组入边数量及消息投影变化见[group_messages.csv](group_messages.csv)。', '',
        '## 图读取和摘要读取是否一致', '',
        '下面是独立列出的机制探针：固定同一已训练读出参数，比较Graph/C128两接口的四种组合，用二接口Shapley将交互平均分配。Graph包含GNN状态、数量、Ego/Load/Dump、图池化及原始动作边栏；C128包含任务和动作摘要。混合接口不是物理状态，数值不是唯一因果归因或通用重要性比例。', '',
        '| 响应 | 完整变化 | Graph接口 | C128接口 |', '| --- | ---: | ---: | ---: |']
    for scene,a in (('A',2),('B',1),('C',2),('C',3)):
        q=next(x for x in dec if x['role']=='actor' and x['scene']==scene and x['target']=='probabilities' and x['action']==str(a))
        lines.append(f"| {scene}→Load{a}概率（百分点） | {float(q['total_change'])*100:+.6f} | {float(q['graph_interface_shapley'])*100:+.6f} | {float(q['C128_interface_shapley'])*100:+.6f} |")
    for scene in 'ABC':
        q=next(x for x in dec if x['role']=='critic' and x['scene']==scene)
        lines.append(f"| {scene}→联合成本V | {float(q['total_change']):+.8f} | {float(q['graph_interface_shapley']):+.8f} | {float(q['C128_interface_shapley']):+.8f} |")
    lines+=['',
        '**C不能归功于GAT。** C的Load2概率总下降3.369057百分点，图接口反而贡献+0.005214，C128贡献−3.374271；Critic的+0.617700几乎也完全由C128给出，图接口仅−0.00003171。该历史点上，图的响应通道虽然可达，但最终业务方向依赖摘要。', '',
        '**B的图读取有效，但两支路抵消。** Load1概率图接口−2.796094百分点、C128+1.166714，总计−1.629381。这比“网络没关注Load”更准确：图中目标与承诺变化确实影响候选动作，摘要未与图支路保持同方向。', '',
        'A的Load2概率图接口+0.40042百分点、C128−1.54017，总计−1.13975，也存在抵消。A是多因素物理分支，不能进一步拆成唯一“装载完成收益”或“品位惩罚”。[概率与支路图](actor_and_fusion_response.png) 同时呈现完整分支和机制探针。原始Logit及减均值Logit均已保存，概率变化以完整Softmax计算，不根据单个raw Logit符号猜测。', '',
        '## 数量准确计数与神经路径冗余', '',
        '21维数量逐节点通过独立oracle校验。A只把Truck1的91.7吨从Load0未完成承诺转为Dump0已载在途，Load0预留2车→1车、服务1→0；Dump0在途5车→6车。B只移动Truck8的一份193吨预留，Load0+Load1合计预留保持284.7吨。C所有图边和数量均不变。候选动作没有生成事实承诺。未发现重复车辆计数或吨位重复累加。浮点显示约1e−5吨尾差来自原始float32缩放。', '',
        '另一个层次是**神经编码的替代路径**：A中Dump0 H0已经含新增在途数量，再有Truck1→Dump0第一层commitment，以及Dump0→Load的material消息；Actor还直接读取候选Dump和C128。它们表达相同业务事实的不同视图，没有把三份神经向量当作三倍吨位相加。', '',
        '| A→Actor Load2 H2差异 | L2 |', '| --- | ---: |',
        '| 完整两层消息 | 0.02415904 |',
        '| 只阻断Truck1→Dump0第一层loaded_dump边 | 0.02453833 |',
        '| 两层均去掉material组消息，保留完整H0/数量/C128 | 0.000303568 |',
        '| 两层均去掉所有消息，仍保留完整H0/数量/C128 | 0 |', '',
        '单条loaded_dump边阻断后响应没有消失，不能宣称这条边是独占传递通道；甚至略增，说明存在非线性、重归一化和其他通道的抵消。去掉material组后该Load2响应大幅减少，表明此例主要借助物料兼容消息将Dump端已有事实传到Load端。边阻断并未抹去Dump0数量、Token内引用、直接候选Dump读取或C128，所以残余响应符合结构设计。[消息路径与数量图](message_paths_and_counts.png) 给出局部通道。', '',
        '## 对后续改进的直接含义', '',
        '当前证据支持先保留两层结构，检查图和摘要的融合及训练变化覆盖：C的处理能力特征在图端影响很小，不能用“图可达”代替“策略已依赖图”，但C是未训练参数外推，须先检查速率变化样本支持；B存在明确的图/摘要方向抵消。后续若要改进，先在更多真实已采集状态上重复成对干预，统计方向、幅度、支路抵消和输入支持域，再考虑融合门控、特征归一化或摘要路径约束。仅凭本点不直接增加GAT层数，也不宣布删去C128会提升性能。', '',
        '## 审计范围与交付', '',
        f"独立复核通过 **{review['total_checks']}项** 检查：独立原始字段白名单/物理约束/21数量oracle，正式完整前向复现所有主张量；另通过正式模型的GraphLayer和summary encoder接口hook，独立复现机制探针，未调用主分析脚本自写的读出函数。", '',
        '- [H0/H1/H2原始张量与全部数组](analysis_tensors.npz)、[各层变化](representation_changes.csv)、[候选表征变化](candidate_changes.csv)。',
        '- [原始字段变化](raw_changes.csv)、[事实图](graph_edges.csv)、[数量](quantities.csv)、[C128](c128_features.csv)。',
        '- [Logit/概率](actor_logits.csv)、[机制阻断](mechanism_probes.csv)、[读取分解](readout_decomposition.csv)、[独立复核](independent_review.json)。', '',
        '所有结果是一个训练种子、一个历史共同基点、三个外生观测分支的局部敏感性证据。历史BASE来自旧TQ策略，A是装载完成与下一状态之间暂态，B不是当前PPO动作，C不是已部署的新处理速率。未运行未来仿真，不能给反事实MC真值、全局单调性、因果唯一归因或生产收益承诺。未训练、未启动RPC/AnyLogic、未新增物理班次、未登记预算。', '']
    with (OUT/'audit_report.md').open('w') as stream:stream.write('\n'.join(lines))


def main():
    assert read('independent_review.json')['status']=='PASS'
    encoder_figure();actor_figure();paths_figure();report()
    images={}
    for p in OUT.glob('*.png'):
        with Image.open(p) as im:images[p.name]=list(im.size);im.verify()
    s=read('summary.json')
    artifacts={p.name:ref(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='manifest.json'}
    manifest=dict(schema='bg128-trained-raw-interventions-manifest-v1',generated_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        checkpoint_update=13,parent_artifacts=s['parent_artifacts'],parameter_digests=s['parameter_digests'],
        frozen_inputs=s['frozen_inputs'],execution=s['execution'],independent_checks=read('independent_review.json')['total_checks'],
        source_refs=s['source_refs'],script_refs=[ref(p) for p in sorted(PLAN.iterdir()) if p.suffix in ('.py','.md')],
        artifacts=artifacts,images=images,visual_qa='Three figures inspected for readable Chinese text, separated numeric labels, axes, relation direction and agreement with CSV; see visual_qa.json.',interpretation_limits=s['limits'],
        training_support_audit=ref(OUT/'training_support_audit.json'))
    with (OUT/'manifest.json').open('w') as f:json.dump(manifest,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status='PASS',artifacts=len(artifacts),images=images),ensure_ascii=False))


if __name__=='__main__':main()
