"""Event-sequence FIFO replay, including paired successful controls.

The optimistic envelope is a conditional exclusion test, NOT a transport ETA
predictor or a proof that an unexcluded action could recover a window.
"""
import math
import numpy as np
from common import *
from grade_event_memory_v1.journal import read_journal
from python.constrained_batch import load_dataset
from physical_realization_v1.risk import fifo_risk

old=module('old_first_window', PLAN.parent/'checkpoint_evaluation_001/trained_analysis_001/analyze_first_window_failures.py')

def envelope(q0,m0,batches,capacity,gmin=0.,gmax=1.):
    """All existing FIFO must precede unlimited instantly arriving ideal ore.

    Endpoints also allow no future ore. Expanding every mass/metal/capacity by
    a fixed conservative accounting allowance avoids boundary certificates
    from serialization rounding (verified closure must be below this bound).
    """
    left=capacity; fq=fm=0.
    for b in batches:
        take=min(left,b['remaining_tons']); fq+=take; fm+=take*b['grade']; left-=take
    qb,mb=q0+fq,m0+fm
    if qb<=1e-9: return dict(lower=0.,upper=1.,forced_tons=fq,ideal_tail_tons=left,excluded=False)
    lower=min(mb/qb,(mb+gmin*left)/(qb+left))
    upper=max(mb/qb,(mb+gmax*left)/(qb+left))
    # q,m FIFO sums come from double journal values; 0.02 t is deliberately
    # larger than checked raw float32 Q/M/FIFO closure and event precision.
    # Each FIFO truncation is 1-Lipschitz in capacity; its metal slope is <=1.
    # With <=16 batches, 0.1 t on both numerator/denominator exceeds all
    # admitted aggregate/endpoint errors, including the remaining capacity.
    err=.1
    low_safe=max(0.,(lower*qb-err)/(qb+err))
    high_safe=min(1.,(upper*qb+err)/max(qb-err,1e-9))
    return dict(lower=low_safe,upper=high_safe,point_lower=lower,point_upper=upper,
        forced_tons=fq,ideal_tail_tons=left,fifo_covers_capacity=left<1e-6,
        excluded=high_safe<.26-1e-12 or low_safe>.28+1e-12)

def main():
    details=[]; decisions=[]; snapshots=[]; joins=[]
    for selected in cohort():
        j=selected['job']; rid=j['run_index']; window=selected['window']; end=(window+1)*7200.
        journal=read_journal(j['material_journal']); rows,*_=load_dataset(selected['raw'],require_real=True)
        events=journal.events; markers={e.decision_sequence:e for e in events if e.kind=='DECISION'}
        process=[e for e in events if e.kind=='PROCESS' and e.window==window]
        unload=[e for e in events if e.kind=='UNLOAD' and e.window==window]
        close=next(e for e in events if e.kind=='WINDOW_CLOSE' and e.window==window)
        final=old.aggregate(process); terminal=old.fifo_at(events,close.seq)
        assert abs(final['tons']-close.window_q)<1e-6 and abs(final['metal_tons']-close.window_m)<1e-6
        compliant=.26-1e-12<=final['grade']<=.28+1e-12
        trace=[]; state_ids=[]; maxerr=0.
        for i,r in enumerate(rows):
            d=markers[i]
            if d.window!=window or d.time>=end: continue
            s=np.asarray(r['observation'],dtype=np.float32); fifo=old.fifo_at(events,d.seq)
            past=old.aggregate([e for e in process if e.seq<d.seq]); q0=past['tons'];m0=past['metal_tons']
            qerr=abs(float(s[4])*5000-q0); merr=abs(float(s[5])*5000-m0)
            ferr=abs(float(s[1300])*500-fifo['tons'])
            assert max(qerr,merr,ferr)<.02, (rid,i,qerr,merr,ferr)
            maxerr=max(maxerr,qerr,merr,ferr)
            # Config is an authenticated upper capacity bound for this replay.
            rate=j['expected_rpc_config']['config.runtime_scene.dump.0.crusherRateTph']
            assert rate==2500. and abs(float(s[1304])*2500-rate)<1e-6
            cap=(end-d.time)*rate/3600
            bound=envelope(q0,m0,fifo['batches'],cap)
            support=fifo_risk(s).to_dict()
            row=dict(lr=selected['lr'],run_index=rid,ordinal=selected['ordinal'],window=window,
                decision_sequence=i,journal_sequence=d.seq,time=d.time,seconds_remaining=end-d.time,
                action=int(r['action']),processed_tons=q0,processed_metal=m0,
                processed_grade=m0/q0 if q0 else None,fifo_tons=fifo['tons'],fifo_grade=fifo['grade'],
                remaining_capacity=cap,**bound,
                visible_support_low_recovery_excluded=support['recovery_excluded'],
                visible_support_upper=support['grade_upper'],
                legal_actions=','.join(str(a) for a in range(4) if s[19+a]),
                accounting_max_abs_error=max(qerr,merr,ferr))
            trace.append(row);decisions.append(row);state_ids.append(i)
        locked=[r for r in trace if r['excluded']]
        visible_locked=[r for r in trace if r['visible_support_low_recovery_excluded']]
        # A successful observed trajectory is an independent consistency check
        # on an exclusion algorithm, not evidence of counterfactual feasibility.
        assert not compliant or not locked, (rid,locked)
        first=locked[0] if locked else None
        before=[r for r in trace if not r['excluded'] and (first is None or r['time']<first['time'])]
        # Four elapsed-window quartiles plus first observed exclusion/last
        # unexcluded decision and final decision. No output-sensitive selection.
        ids={state_ids[min(len(state_ids)-1,int(len(state_ids)*f))] for f in (.2,.4,.6,.8)}
        ids.add(state_ids[-1])
        if first: ids.add(first['decision_sequence'])
        if before: ids.add(before[-1]['decision_sequence'])
        for i in sorted(ids):
            snapshots.append(dict(lr=selected['lr'],run_index=rid,window=window,ordinal=selected['ordinal'],
                state_index=i,time=float(rows[i]['action_time_seconds']),raw=rows[i]['observation'].tolist()))
        all_actions=[(markers[i],r) for i,r in enumerate(rows)]
        for e in unload:
            preceding=[(d,r) for d,r in all_actions if int(r['truck_id'])==e.truck_id and d.seq<e.seq]
            d,r=preceding[-1] if preceding else (None,None)
            joins.append(dict(lr=selected['lr'],run_index=rid,window=window,batch_id=e.batch_id,
                source_id=e.source_id,truck_id=e.truck_id,tons=e.tons,grade=e.grade,unload_time=e.time,
                dispatch_time=d.time if d else None,dispatch_sequence=d.decision_sequence if d else None,
                source_match=int(r['action'])==e.source_id if r else False,
                dispatch_to_unload_seconds=e.time-d.time if d else None))
        details.append(dict(lr=selected['lr'],run_index=rid,ordinal=selected['ordinal'],window=window,
            seed=j['experiment_seed'],actor_sampling_seed=j['actor_sampling_seed'],compliant=compliant,
            processed=final,unloaded=old.aggregate(unload),end_fifo=terminal,
            earliest_observed_exclusion=first,last_unexcluded_before_first=before[-1] if before else None,
            earliest_visible_support_low_exclusion=visible_locked[0] if visible_locked else None,
            last_decision=trace[-1],decisions=len(trace),accounting_max_abs_error=maxerr,
            causal_policy_miss_proven=False,
            classification='conditional_fifo_capacity_exclusion_observed' if locked else 'not_excluded_by_optimistic_envelope',
            sources=dict(journal=ref(j['material_journal']),raw_manifest=ref(selected['raw']/'manifest.json'),
                         raw_samples=ref(selected['raw']/'samples.h5'))))
        rows.close()
    save('replay.json',dict(schema='bg128-paired-material-realization-replay-v1',trajectories=details,
        physical_shifts=0,optimizer_updates=0,protected_ppo_sha=sha(CORE),
        assumptions=['Fixed work-conserving 2500 tph crusher, immutable FIFO order and no ore bypass.',
            'Future material may arrive instantaneously, unlimited, with any grade in [0,1]; capacity is the only upper bound.',
            'Bounds allow stopping after the forced FIFO; production-target feasibility is deliberately relaxed.',
            '0.1 t outward accounting allowance; every Q/M/FIFO observation closure must be <0.02 t.',
            'Only past event sequence and current observation enter certificates; future tape is used for outcome/accounting only.'],
        limitations=['First observed exclusion is not the exact instant recovery became impossible.',
            'Unexcluded is not feasible; no alternate action trajectory is executed.',
            'No LOAD_COMPLETE event: complete service timing is not exactly observable.',
            'Visible-support lower-grade certificate uses the pre-existing outward float32 interval implementation and stronger support assumptions.',
            'No source/grade is backfilled from future unloads into model input or certificate.',
            'Failure-conditioned seed cohort is descriptive, not a population performance estimate.'],
        source_pins=[ref(p) for p in list(SUITES.values())+list(AUDITS.values())+[Path(__file__),PLAN/'common.py']]))
    csv_save('decision_prefixes.csv',decisions);csv_save('dispatch_unload_joins.csv',joins)
    save('state_selection.json',dict(selection='20/40/60/80 percent decision ranks + last + first exclusion + preceding unexcluded',states=snapshots))
    assert sha(CORE)==CORE_SHA
    print(json.dumps(dict(trajectories=len(details),failed=sum(not r['compliant'] for r in details),
        excluded_failures=sum(not r['compliant'] and r['earliest_observed_exclusion'] is not None for r in details),
        decisions=len(decisions),selected_states=len(snapshots))))

if __name__=='__main__': main()
