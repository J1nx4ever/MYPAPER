"""Narrow discrete joint-pattern checks; not a full distribution estimator."""
from collections import Counter
import h5py
import numpy as np
from common import *

inter=module('expanded_branches_support',PLAN/'interventions.py')

def motion_signature(t):
    phase=inter.builder.onehot(t[4:16]);load=inter.builder.onehot(t[26:30]);seg=inter.builder.onehot(t[45:61])
    ranks=np.rint(t[63:79]*16).astype(int)
    route=tuple(sorted(np.flatnonzero(ranks).tolist(),key=lambda r:ranks[r]))
    return phase,int(t[17]),load,seg,route

def main():
    arm=PROJECT/'runs/business-graph-summary-formal-seed1-20261118-001/arms/s01_BG128'
    patterns=Counter();loaded_other=0;decision_count=0;states_with_loaded_other=0;pins=[]
    for n in range(1,17):
        s=read(arm/f'updates/{n:04}/training_summary.json')
        for refm in s['transaction']['sources']:
            assert ref(refm['path'])==refm
            path=Path(str(refm['path']).removesuffix('.manifest.json'));m=read(refm['path'])
            assert sha(path)==m['hdf5_sha256'];pins.append(ref(path))
            with h5py.File(path) as h:
                count=len(h['transitions/action']);trucks=h['states'][:count,27:1107].reshape(count,9,120)
                active=(trucks[:,:,17]==1)&(trucks[:,:,18]==1)&(trucks[:,:,15]==1)
                loaded_other+=int(active.sum());states_with_loaded_other+=int(active.any(-1).sum());decision_count+=count
                mask=(trucks[:,:,17]==0)&(trucks[:,:,6]==1)
                for t in trucks[mask]:patterns[motion_signature(t)]+=1
    selection=read(OUT/'state_selection.json')['states'];bases={(s['lr'],s['run_index'],s['state_index']):s for s in selection}
    rows=[]
    for b in read(OUT/'branches.json'):
        base=np.asarray(bases[b['lr'],b['run_index'],b['state_index']]['raw'],dtype=np.float32)
        fn=dict(A=inter.branch_a,B=inter.branch_b,C=inter.builder.scenario_c)[b['scene']]
        raw,definition=fn(base);assert inter.builder.raw_sha(raw)==b['raw_sha256'] and definition==b['definition']
        if b['scene']=='A':
            observed=loaded_other;pattern='loaded ore AND OTHER phase (any truck)'
        elif b['scene']=='B':
            t=inter.builder.blocks(raw)[0][definition['truck']]
            observed=patterns[motion_signature(t)];pattern='empty TO_LOAD: target load + current segment + ordered remaining route'
        else:observed=0;pattern='crusher rate 1250 tph (original configuration always 2500)'
        rows.append({k:b[k] for k in ('lr','run_index','window','state_index','scene')}|dict(
            discrete_pattern=pattern,training_pattern_occurrences=observed,pattern_unseen_in_scanned_training=observed==0))
    result=dict(schema='bg128-narrow-discrete-pattern-support-v1',status='PASS_DESCRIPTIVE_ONLY',
        training_decisions=decision_count,loaded_ore_other_phase_truck_observations=loaded_other,
        training_decisions_with_loaded_ore_other_phase=states_with_loaded_other,
        distinct_empty_to_load_motion_signatures=len(patterns),
        unseen_branch_patterns={s:sum(r['scene']==s and r['pattern_unseen_in_scanned_training'] for r in rows) for s in 'ABC'},
        eligible={s:sum(r['scene']==s for r in rows) for s in 'ABC'},
        limitations=['Only the specified discrete projection of a state is tested.',
            'An unseen projection proves that exact projected pattern was absent from these 512 shifts; it does not prove failure cause.',
            'An observed projection does not establish full-state joint support, adequate frequency, or simulator reachability.',
            'A/B are still exogenous branches, not executed policy actions.'],
        physical_shifts=0,optimizer_updates=0,source_pins=pins+[ref(Path(__file__)),ref(inter.__file__),ref(OUT/'branches.json')])
    save('joint_support.json',result);csv_save('joint_support.csv',rows)
    print(json.dumps({k:v for k,v in result.items() if k not in ['source_pins','limitations']}))

if __name__=='__main__':main()
