"""Narrow user-authorized seed replay, without physical identity reuse."""
from copy import deepcopy
import pytest

from .pairing import (training_source,evaluation_source,check_seed_pairing,contract,TARGET_TRAIN,TARGET_EVAL)
from .budget import check_collisions


@pytest.fixture(scope='module')
def paired():
    result={}
    for evaluate,source,campaign,base in [(False,training_source(),TARGET_TRAIN,9960001),
                                        (True,evaluation_source(),TARGET_EVAL,9970001)]:
        jobs=[]
        for ordinal,old in enumerate(source['jobs']):
            job=deepcopy(old)
            job.update(run_index=base+ordinal,recording_id=campaign+'-paired',
                       source_recording_id=campaign+'-paired')
            jobs.append(job)
        result[evaluate]=dict(campaign_id=campaign,pairing=contract(),jobs=jobs)
    return result


@pytest.mark.parametrize('evaluate',[False,True])
def test_exact_corresponding_training_and_evaluation_seeds_pass(paired,evaluate):
    source=check_seed_pairing(paired[evaluate],evaluation=evaluate)
    assert len(source['jobs'])==(100 if evaluate else 512)


@pytest.mark.parametrize('mutation',['swap_rows','swap_roles','outside_seed','missing_row','old_run','old_recording','third_campaign','source_sha'])
def test_narrow_exception_rejects_reordering_expansion_and_old_identity(paired,mutation):
    value=deepcopy(paired[False]);j=value['jobs']
    if mutation=='swap_rows':
        j[0]['experiment_seed'],j[1]['experiment_seed']=j[1]['experiment_seed'],j[0]['experiment_seed']
    elif mutation=='swap_roles':
        j[0]['experiment_seed'],j[0]['actor_sampling_seed']=j[0]['actor_sampling_seed'],j[0]['experiment_seed']
    elif mutation=='outside_seed':j[0]['experiment_seed']=123
    elif mutation=='missing_row':j.pop()
    elif mutation=='old_run':j[0]['run_index']=9900001
    elif mutation=='old_recording':j[0]['recording_id']=training_source()['jobs'][0]['recording_id']
    elif mutation=='third_campaign':value['campaign_id']='another-seed-reuse'
    elif mutation=='source_sha':value['pairing']['training_suite']['sha256']='0'*64
    with pytest.raises(ValueError):check_seed_pairing(value)


def test_valid_pairing_still_counts_new_budget_and_checks_historical_runs(paired):
    source=training_source()
    history=dict(campaigns={'old':{}},jobs={str(j['run_index']):dict(identity=j) for j in source['jobs']})
    check_collisions(paired[False],history)
    history['jobs']['collision']=dict(identity=paired[False]['jobs'][0])
    with pytest.raises(ValueError,match='historical collision'):
        check_collisions(paired[False],history)
