"""Prespecified paired contrasts, with environment seed as the sampling unit."""
from collections import Counter
from hashlib import sha256
from math import comb
from statistics import fmean

import numpy as np
from scipy.stats import binomtest, norm

STRATEGIES = ('full', 'flow', 'flat')
DURATIONS = (0, 30, 60)
ENDPOINTS = ('joint', 'wait', 'grade')
ALPHA = .05


def wilson(k, n):
    z = norm.ppf(.975)
    radius = z * (k*(n-k)/n + z*z/4)**.5
    center = k + z*z/2
    return [max(0., (center-radius)/(n+z*z)), min(1., (center+radius)/(n+z*z))]


def binary_difference_interval(x):
    # Simultaneous bounds for each discordance probability, with >=95% joint coverage.
    plus = binomtest(sum(x == 1), len(x)).proportion_ci(.975, method='exact')
    minus = binomtest(sum(x == -1), len(x)).proportion_ci(.975, method='exact')
    return [max(-1., plus.low-minus.high), min(1., plus.high-minus.low)]


def rng_for(key):
    key_hash = sha256(key.encode()).digest()
    return np.random.default_rng([20261127, int.from_bytes(key_hash[:4],'big')])


def signflip(values, rng, discrete):
    magnitudes = abs(values[values != 0])
    observed = abs(values.sum())
    if not len(magnitudes) or observed <= 1e-12:
        return 1.
    if discrete:
        sums = Counter({0: 1})
        for magnitude in magnitudes:
            updated = Counter()
            for old, count in sums.items():
                updated[old + int(magnitude)] += count
                updated[old - int(magnitude)] += count
            sums = updated
        return sum(n for t,n in sums.items() if abs(t) >= observed) / 2**len(magnitudes)
    if len(magnitudes) <= 16:
        totals = np.array([0.])
        for magnitude in magnitudes:
            totals = np.concatenate((totals+magnitude, totals-magnitude))
        return float(np.mean(abs(totals) >= observed-1e-12))
    n_draws = 100000
    extreme = 0
    for _ in range(n_draws//2500):
        signs = 2*rng.integers(0,2,size=(2500,len(magnitudes)),dtype=np.int8)-1
        extreme += np.count_nonzero(abs((signs*magnitudes).sum(axis=1)) >= observed-1e-12)
    return (extreme+1)/(n_draws+1)


def contrast(cells, name, endpoint, terms, family, binary=False):
    # All cells use the same sorted 30 slot IDs. Compute the within-slot contrast first.
    x = np.zeros(30)
    for strategy, duration, coefficient in terms:
        vals = [r[endpoint] for r in cells[strategy,duration]]
        x += coefficient*np.array(vals,dtype=float)
    x=x[np.isfinite(x)]
    row=dict(name=name, endpoint=endpoint, family=family, n=len(x), terms=terms)
    if len(x)==0:
        return dict(row, mean=None, ci=None, p=None, p_holm=None)
    rng = rng_for(name+'/'+endpoint)
    if binary:
        gains=int(sum(x==1))
        losses=int(sum(x==-1))
        discordant=gains+losses
        p=binomtest(gains,discordant,.5).pvalue if discordant else 1.
        ci=binary_difference_interval(x)
        row.update(gains=gains,losses=losses)
    else:
        bootstrap=x[rng.integers(0,len(x),size=(20000,len(x)))].mean(axis=1)
        ci=np.quantile(bootstrap,[.025,.975]).tolist()
        p=signflip(x,rng,discrete=endpoint=='joint')
    return dict(row,mean=float(x.mean()),ci=ci,p=float(p))


def holm(rows):
    for family in sorted({r['family'] for r in rows}):
        ordered=sorted((r for r in rows if r['family']==family),key=lambda r:1. if r['p'] is None else r['p'])
        bound=0.
        for rank,row in enumerate(ordered):
            if row['p'] is not None:
                bound=max(bound,min(1.,(len(ordered)-rank)*row['p']))
                row['p_holm']=bound


def summarize(rows):
    cells={}
    for policy in STRATEGIES:
        for duration in DURATIONS:
            group=sorted((r for r in rows if r['strategy']==policy and r['duration_minutes']==duration),key=lambda r:r['slot'])
            if [r['slot'] for r in group] != list(range(1,31)):
                raise ValueError('incomplete policy/severity cell')
            cells[policy,duration]=group
    seeds=[r['environment_seed'] for r in cells['full',0]]
    if len(set(seeds))!=30 or any([r['environment_seed'] for r in g]!=seeds for g in cells.values()):
        raise ValueError('unmatched slot seeds')
    descriptive=[]
    for (policy,duration),group in cells.items():
        grades=[r['grade'] for r in group if r['grade'] is not None]
        per_window=[[r['window_grade_percent'][i] for r in group if r['window_grade_percent'][i] is not None]
                    for i in range(4)]
        joint=sum(r['joint'] for r in group)
        descriptive.append(dict(strategy=policy,duration_minutes=duration,n=30,joint=joint,
                joint_ci=wilson(joint,30),four=sum(r['four'] for r in group),
                ore_met=sum(r['ore_met'] for r in group),waste_met=sum(r['waste_met'] for r in group),
                window_pass=[sum(r['window_pass'][i] for r in group) for i in range(4)],
                mean_window_grade_percent=[fmean(g) if g else None for g in per_window],
                window_grade_n=[len(g) for g in per_window],
                mean_ore_t=fmean(r['ore_t'] for r in group),mean_waste_t=fmean(r['waste_t'] for r in group),
                mean_wait=fmean(r['wait'] for r in group),mean_grade_sd=fmean(grades) if grades else None,
                grade_n=len(grades),suspended_active_shifts=sum(r['paused_in_service']>0 for r in group),
                mean_paused_truck_hours=fmean(r['paused_in_service']*duration/60 for r in group),
                mean_high_tons_4_to_5=fmean(r['high_tons_4_to_5'] for r in group),
                decisions=sum(r['decisions'] for r in group)))
    contrasts=[]
    for baseline in ('flow','flat'):
        contrasts.append(contrast(cells,f'full-{baseline}@60','joint',[('full',60,1),(baseline,60,-1)],
                                  'primary60',binary=True))
    for endpoint in ENDPOINTS:
        for strategy in STRATEGIES:
            for duration in (30,60):
                contrasts.append(contrast(cells,f'{strategy}:{duration}-0',endpoint,
                                          [(strategy,duration,1),(strategy,0,-1)],'within_'+endpoint,endpoint=='joint'))
        for baseline in ('flow','flat'):
            contrasts.append(contrast(cells,f'full-{baseline}:60-30',endpoint,
                                      [('full',60,1),(baseline,60,-1),('full',30,-1),(baseline,30,1)],'interaction'))
            if endpoint!='joint':
                for duration in (30,60):
                    contrasts.append(contrast(cells,f'full-{baseline}:{duration}-0',endpoint,
                                      [('full',duration,1),(baseline,duration,-1),('full',0,-1),(baseline,0,1)],
                                      'continuous_impacts'))
    holm(contrasts)
    return descriptive,contrasts
