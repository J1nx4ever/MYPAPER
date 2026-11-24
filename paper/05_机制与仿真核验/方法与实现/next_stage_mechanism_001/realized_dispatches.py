"""Link each observed ore dispatch to unload and processing by its deadline.

Future events are OUTCOME LABELS, never added to observations or risk inputs.
Waste dispatches are outside the ore material tape and are not missing ore.
"""
from collections import defaultdict
import math
import numpy as np
from common import *
from grade_event_memory_v1.journal import read_journal
from python.constrained_batch import load_dataset

def main():
    replay=read(OUT/'replay.json');bounds={(r['lr'],r['run_index']):r for r in replay['trajectories']}
    rows_out=[];summaries=[];pins=[]
    for c in cohort():
        j=c['job'];rid=j['run_index'];w=c['window'];journal=read_journal(j['material_journal'])
        events=journal.events;markers={e.decision_sequence:e for e in events if e.kind=='DECISION'}
        close=next(e for e in events if e.kind=='WINDOW_CLOSE' and e.window==w)
        raw,*_=load_dataset(c['raw'],require_real=True);acts=[]
        for i,r in enumerate(raw):
            d=markers[i];s=np.asarray(r['observation'],dtype=np.float32);a=int(r['action'])
            acts.append(dict(decision_sequence=i,seq=d.seq,time=d.time,window=d.window,truck=int(r['truck_id']),
                action=a,ore=bool(s[1107+a*47+1]),candidate_tons=float(s[23+a])*193,
                visible_source_mean=float(s[1107+a*47+2]),visible_source_half_range=float(s[1107+a*47+3])))
        by_action=defaultdict(list);batch_owner={};unmatched=[]
        for e in events:
            if e.kind!='UNLOAD':continue
            preceding=[a for a in acts if a['truck']==e.truck_id and a['seq']<e.seq]
            a=preceding[-1] if preceding else None
            if a is None or a['action']!=e.source_id or not a['ore']:
                unmatched.append(e.batch_id);continue
            batch_owner[e.batch_id]=a;by_action[a['decision_sequence']].append(e)
        process=defaultdict(list)
        for e in events:
            if e.kind=='PROCESS' and e.window==w:process[e.batch_id].append(e)
        r=bounds[c['lr'],rid];f=r['earliest_observed_exclusion']
        if f is None:f=r['earliest_visible_support_low_exclusion']
        current=[]
        for a in acts:
            if a['window']!=w:continue
            unloads=by_action[a['decision_sequence']];batch_ids=[e.batch_id for e in unloads]
            processed=[e for b in batch_ids for e in process[b]]
            on_time=[e for e in unloads if e.seq<close.seq]
            q=math.fsum(e.tons for e in processed);m=math.fsum(e.metal for e in processed)
            label='waste_outside_ore_tape' if not a['ore'] else (
                'partly_or_fully_processed_by_deadline' if q>1e-6 else
                'unloaded_but_not_processed_by_deadline' if on_time else
                'unloaded_after_deadline' if unloads else 'no_unload_observed_before_shift_end')
            assert not a['ore'] or len(unloads)<=1
            if unloads:assert abs(sum(e.tons for e in unloads)-a['candidate_tons'])<.006
            row=dict(lr=c['lr'],run_index=rid,target_window=w,**a,
                batch_ids=','.join(batch_ids),first_unload_time=unloads[0].time if unloads else None,
                unloaded_tons_whole_shift=math.fsum(e.tons for e in unloads),
                unloaded_tons_by_deadline=math.fsum(e.tons for e in on_time),
                processed_tons_in_target_window=q,processed_metal_in_target_window=m,
                realized_processed_grade=m/q if q else None,
                processed_fraction_of_dispatched_candidate=q/a['candidate_tons'] if a['ore'] else None,
                after_observed_exclusion=bool(f and a['time']>=f['time']),outcome_label=label)
            current.append(row);rows_out.append(row)
        attribution=dict(current_window_dispatch=0.,prior_window_dispatch=0.,unmatched=0.)
        for b,ps in process.items():
            a=batch_owner.get(b);key='unmatched' if a is None else 'current_window_dispatch' if a['window']==w else 'prior_window_dispatch'
            attribution[key]+=math.fsum(e.tons for e in ps)
        assert abs(sum(attribution.values())-r['processed']['tons'])<1e-6
        late=[a for a in current if a['ore'] and a['after_observed_exclusion']]
        summaries.append(dict(lr=c['lr'],run_index=rid,window=w,compliant=r['compliant'],
            ore_dispatches=sum(a['ore'] for a in current),ore_candidate_tons=sum(a['candidate_tons'] for a in current if a['ore']),
            current_window_dispatch_processed_tons=attribution['current_window_dispatch'],
            processed_mass_attribution=attribution,unmatched_unload_batches=unmatched,
            outcome_counts={name:sum(a['outcome_label']==name for a in current) for name in sorted({a['outcome_label'] for a in current})},
            after_observed_exclusion_ore_dispatches=len(late),
            after_observed_exclusion_candidate_tons=sum(a['candidate_tons'] for a in late),
            after_observed_exclusion_processed_tons=sum(a['processed_tons_in_target_window'] for a in late),
            per_source={str(lid):dict(dispatch_count=sum(a['action']==lid for a in current),
                candidate_tons=sum(a['candidate_tons'] for a in current if a['action']==lid),
                unloaded_by_deadline=sum(a['unloaded_tons_by_deadline'] for a in current if a['action']==lid),
                processed_in_window=sum(a['processed_tons_in_target_window'] for a in current if a['action']==lid)) for lid in range(4)}))
        pins.extend([ref(j['material_journal']),ref(c['raw']/'samples.h5')]);raw.close()
    save('dispatch_realization.json',dict(status='PASS_OBSERVED_EVENT_CHAIN_ONLY',trajectories=summaries,
        physical_shifts=0,optimizer_updates=0,source_pins=pins+[ref(Path(__file__)),ref(OUT/'replay.json')],
        limitations=['Future UNLOAD/PROCESS only supplies outcome labels; no observation or certificate is modified.',
            'Observed dispatch candidate tons are a flow at decision time, not outstanding inventory.',
            'Waste dispatches do not enter the ore FIFO tape.',
            'The log lacks loading completion and complete travel/service event boundaries.',
            'Late dispatches may serve later windows; zero current-window realization does not prove they were wrong.',
            'No alternate action or external randomness is replayed.']))
    csv_save('dispatch_realization.csv',rows_out)
    print(json.dumps(dict(trajectories=len(summaries),dispatches=len(rows_out),
        unmatched=sum(len(r['unmatched_unload_batches']) for r in summaries))))

if __name__=='__main__':main()
