"""Compare already completed independent evaluations; no model or simulator."""
from pathlib import Path
import hashlib
import json
import math
import sys
import numpy as np

ROOT=Path('/home/jinx/Faith/experiments/lagrangian_ppo_v1')
REPORT=ROOT/'reports/token_attention_001/business_graph_summary_009/trained_analysis_001'
CURRENT=ROOT/'runs/business-graph-summary-best-checkpoint-eval100-20261118-001/evaluation_complete.json'
PREVIOUS=ROOT/'runs/single-critic-task-query-formal-seed1-20261117-002/arms/s01_TQ128/final_u0016/evaluation_complete.json'


def ref(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def extract(report):
    assert report['n_shifts']==100 and report['training_state_unchanged'] and report['optimizer_updates']==0
    records=report['per_episode'];physical=[r['physical'] for r in records]
    square=np.array([sum((w['grade']-.27)**2 for w in p['windows'] if not w['empty']) for p in physical])
    valid=np.array([sum(not w['empty'] for w in p['windows']) for p in physical])
    return dict(report=report,rows=physical,square=square,valid=valid,
        arrays={k:np.array([p[k] for p in physical],dtype=float) for k in (
            'combined_cost','grade_cost','production_cost','waiting_cost','total_wait_seconds',
            'ore_tons','waste_tons','complete_task','all_four_windows_compliant','ore_target_met','waste_target_met')})


def wilson(k,n):
    z=1.959963984540054;p=k/n;den=1+z*z/n
    mid=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0,mid-half),min(1,mid+half)]


def metrics(x):
    h=x['report']['headline_metrics'];a=x['arrays']
    out=dict(headline_metrics=h,policy_entropy=x['report']['policy_entropy'],critic_mc=x['report']['critic_mc'])
    out['cost_components']={k:dict(mean=float(a[k].mean()),sd=float(a[k].std(ddof=1)),
        p50=float(np.quantile(a[k],.5)),p95=float(np.quantile(a[k],.95))) for k in (
            'combined_cost','grade_cost','production_cost','waiting_cost')}
    out['rate_95_wilson_intervals']={k:wilson(int(a[k].sum()),len(a[k])) for k in (
        'complete_task','all_four_windows_compliant','ore_target_met','waste_target_met')}
    out['failure_runs']=[dict(run_index=r['identity']['run_index'],ore_tons=r['physical']['ore_tons'],
        waste_tons=r['physical']['waste_tons'],window_grades=[w['grade'] for w in r['physical']['windows']],
        wait_seconds=r['physical']['total_wait_seconds'],combined_cost=r['physical']['combined_cost'])
        for r in x['report']['per_episode'] if not r['physical']['complete_task']]
    return out


def main():
    old=extract(json.loads(PREVIOUS.read_bytes()));new=extract(json.loads(CURRENT.read_bytes()))
    assert old['report']['composite_contract']==new['report']['composite_contract']
    overlap=set(r['identity']['environment_seed'] for r in old['report']['per_episode']) & set(
        r['identity']['environment_seed'] for r in new['report']['per_episode'])
    assert not overlap
    rng=np.random.default_rng(2026111809);nboot=20000
    ixold=rng.integers(0,100,(nboot,100));ixnew=rng.integers(0,100,(nboot,100))
    comparison={}
    for key in new['arrays']:
        point=float(new['arrays'][key].mean()-old['arrays'][key].mean())
        draws=new['arrays'][key][ixnew].mean(1)-old['arrays'][key][ixold].mean(1)
        comparison[key]=dict(TQ128_mean=float(old['arrays'][key].mean()),BG128_mean=float(new['arrays'][key].mean()),
            difference_BG_minus_TQ=point,unpaired_shift_bootstrap_95_CI=np.quantile(draws,[.025,.975]).tolist())
    rmse=lambda x:float(np.sqrt(x['square'].sum()/x['valid'].sum()))
    draws=np.sqrt(new['square'][ixnew].sum(1)/new['valid'][ixnew].sum(1))-np.sqrt(
        old['square'][ixold].sum(1)/old['valid'][ixold].sum(1))
    comparison['grade_rmse']=dict(TQ128_mean=rmse(old),BG128_mean=rmse(new),
        difference_BG_minus_TQ=rmse(new)-rmse(old),unpaired_shift_bootstrap_95_CI=np.quantile(draws,[.025,.975]).tolist())
    result=dict(schema='bg128-fixed-checkpoint-independent-evaluation-analysis-v1',status='PASS',
        sources=dict(current=ref(CURRENT),previous=ref(PREVIOUS)),BG128=metrics(new),TQ128=metrics(old),
        descriptive_BG_minus_TQ=comparison,environment_seed_overlap=0,
        bootstrap=dict(repetitions=nboot,seed=2026111809,unit='complete_shift_with_all_four_windows',paired=False),
        interpretation='Different training seeds and evaluation seeds; BG training-selected u13 vs TQ fixed-final u16. Descriptive fixed-policy comparison, not a controlled architecture ablation or training-seed robustness claim.',
        new_physical_shifts=0,optimizer_updates=0,model_forwards=0)
    REPORT.mkdir(parents=True,exist_ok=True)
    (REPORT/'inference_comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    from PIL import Image,ImageDraw,ImageFont
    fontpath='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font=ImageFont.truetype(fontpath,20);titlefont=ImageFont.truetype(fontpath,23)
    heading=ImageFont.truetype(fontpath,30)
    image=Image.new('RGB',(1800,1140),'white');draw=ImageDraw.Draw(image)
    draw.text((40,20),'Independent evaluation: 100 shifts each',font=heading,fill='#142c40')
    draw.text((40,65),'Same joint cost; different evaluation seeds; descriptive fixed-policy comparison',font=font,fill='#46576b')
    plots=[('Grade RMSE','grade_rmse',1),('All-four-window compliance','joint_four_window_compliance_rate',100),
        ('Complete task rate','complete_task_rate',100),('Fleet wait (vehicle-hours / shift)','mean_fleet_wait_hours',1),
        ('Waste target met','waste_target_met_rate',100),('Critic MC MSE (pooled)','critic_mc_mse_pooled',1)]
    for i,(title,key,scale) in enumerate(plots):
        values=[old['report']['headline_metrics'][key]*scale,new['report']['headline_metrics'][key]*scale]
        left=40+(i%3)*590;top=125+(i//3)*490
        draw.text((left,top),title,font=titlefont,fill='#142c40')
        x=left+85;y=top+365;high=100 if scale==100 else max(values)*1.2
        draw.line((x,top+60,x,y,x+420,y),fill='#6f7b88',width=2)
        for tick in range(5):
            value=high*tick/4;yy=y-round(290*tick/4)
            draw.line((x,yy,x+420,yy),fill='#e1e7eb',width=1)
            draw.text((left,yy-12),format(value,'.3g'),font=font,fill='#58687a')
        for b,(value,color,label) in enumerate(zip(values,['#8793a3','#168a87'],['TQ128 final u16','BG128 selected u13'])):
            bx=x+65+b*205;yy=y-round(290*value/high)
            draw.rectangle((bx,yy,bx+105,y),fill=color)
            draw.text((bx,yy-30),format(value,'.5g')+('%' if scale==100 else ''),font=font,fill='#172b3b')
            draw.text((bx-30,y+18),label.split(' ')[0],font=font,fill='#25394c')
            draw.text((bx-30,y+46),' '.join(label.split(' ')[1:]),font=font,fill='#25394c')
    draw.text((40,1100),'BG128 was selected on training data before these 100 shifts; no parameter updates during evaluation.',font=font,fill='#46576b')
    image.save(REPORT/'independent_evaluation_comparison.png')
    print(json.dumps(dict(status='PASS',BG128=result['BG128']['headline_metrics'],costs=result['BG128']['cost_components'])))


if __name__=='__main__':main()
