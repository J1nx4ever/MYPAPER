"""Exercise paired selection, denominator asymmetry, empty and late windows."""
import json

from reports import summarize


def test_paired_report(tmp_path):
    jobs = []
    for split in ('validation','test'):
        for slot in range(3):
            for arm in ('static','dynamic'):
                for condition in ('outage','healthy'):
                    run = len(jobs)+1
                    path = tmp_path/str(run)
                    path.mkdir()
                    j = dict(run_index=run,recording_id=str(run),experiment_seed=100+slot,
                             actor_sampling_seed=200+slot,config_sha256=str(run),arm=arm,condition=condition,
                             split=split,pair_ordinal=slot,role='eval',python_output=str(path),starting_count=8,
                             fault_truck_id=3 if condition=='outage' else -1,waste_site=2)
                    triggered = condition=='outage'
                    late = triggered and slot==1 and arm=='static'
                    no_exit = triggered and slot==2
                    t = -1. if not triggered or no_exit else 22000. if late else 14000.
                    valid = 0<=t<=21600.
                    rate = .05 if arm=='dynamic' else .1
                    physical = dict(ore_target_tons=100,waste_target_tons=50,ore_tons=100*(1-rate),waste_tons=50,
                                    ore_shortfall=rate,waste_shortfall=0.,all_four_windows_compliant=True,
                                    complete_task=False,total_wait_seconds=0.,grade_cost=0.,production_cost=8*rate,
                                    waiting_cost=0.,combined_cost=8*rate)
                    outage = dict(trigger_seconds=10000. if triggered else -1.,exit_seconds=t,
                                  exit_delay_seconds=t-10000. if t>=0 else None,window_valid=valid,
                                  window_empty=valid and slot==0,window_compliant=valid and slot!=0)
                    (path/'receipt.json').write_text(json.dumps(dict(j,physical=physical,outage=outage)))
                    jobs.append(j)
    suite = dict(jobs=jobs,campaign_id='test',grade_margin=0.,protocol=dict(validation_slots=3,test_slots=3))
    result = summarize(suite)
    assert result['validation_grade_gate'] and result['selection_from_validation']=='dynamic'
    for split in ('validation','test'):
        block = result['results'][split]
        assert block['paired_valid_exit_windows']==1 and block['paired_exit_rate_delta']==0
        assert block['dynamic']['outage']['post_exit_rate']==.5
        assert block['static']['outage']['valid_post_exit_windows']==1
        assert block['static']['outage']['empty_post_exit_windows']==1
        assert block['static']['outage']['late_exits']==block['static']['outage']['still_retiring_at_end']==1
        assert block['dynamic']['outage']['mean_ore_shortfall_tons']==5
        assert block['static']['healthy']['no_fault_shifts']==3

    j = next(j for j in jobs if j['split']=='validation' and j['arm']=='dynamic' and j['condition']=='outage')
    file = tmp_path/str(j['run_index'])/'receipt.json'
    receipt = json.loads(file.read_bytes())
    receipt['physical']['all_four_windows_compliant'] = False
    file.write_text(json.dumps(receipt))
    assert summarize(suite)['selection_from_validation']=='static'
