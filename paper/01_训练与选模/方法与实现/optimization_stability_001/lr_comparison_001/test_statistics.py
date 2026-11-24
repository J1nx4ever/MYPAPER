import importlib.util
from pathlib import Path

p=Path(__file__).with_name('analyze.py');spec=importlib.util.spec_from_file_location('lr_compare',p)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def records(successes):
    return [dict(physical=dict(complete_task=i in successes,all_four_windows_compliant=i in successes,
        ore_target_met=True,waste_target_met=True,total_wait_seconds=10.,combined_cost=.5,grade_cost=.4,
        production_cost=0.,waiting_cost=.1,ore_tons=18000.,waste_tons=6000.,
        windows=[dict(grade=.27,compliant=True)]*4)) for i in range(100)]


def test_identical_pairs_have_exact_zero_differences_and_intervals():
    old=records(set(range(98)));metrics,_=module.paired_statistics(old,old)
    for row in metrics.values():
        assert row['new_minus_old']==0 and row['paired_bootstrap_95ci']==[0.,0.]
    assert metrics['complete_task']['exact_mcnemar_two_sided_p']==1


def test_exact_mcnemar_uses_discordant_pairs():
    metrics,_=module.paired_statistics(records(set(range(98))),records(set(range(100))))
    result=metrics['complete_task']
    assert result['paired_outcomes']['old_failure_new_success']==2
    assert result['paired_outcomes']['old_success_new_failure']==0
    assert result['exact_mcnemar_two_sided_p']==.5


def test_rmse_is_root_of_pooled_squared_errors_instead_of_mean_shift_rmse():
    old=records(set(range(100)));new=records(set(range(100)))
    new[0]['physical']['windows']=[dict(grade=.28,compliant=True)]*4
    metrics,_=module.paired_statistics(old,new)
    assert abs(metrics['grade_rmse']['new']-.001)<1e-12
    assert metrics['grade_rmse']['old']==0
