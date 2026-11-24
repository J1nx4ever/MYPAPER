"""Latest independent-evaluation inventory and append-only ledger boundaries."""
from copy import deepcopy
from pathlib import Path

import pytest

from . import budget, gate
from .plan_io import reference, write


def minimal_suite(tmp_path, total, previous):
    job=dict(run_index=9960001,recording_id='synthetic-stability-ledger',
        source_recording_id='synthetic-stability-ledger',experiment_seed=981,
        actor_sampling_seed=982,actor_seed=983,role='train',target_training_update=1)
    phase=tmp_path/'phase.json';write(phase,dict(jobs=[job]))
    suite=dict(campaign_id='synthetic-stability-ledger',suite_path=str(tmp_path/'suite.json'),
        phase_configs=[reference(phase)],new_shift_cap=1,episodes_per_batch=1,
        budget_state_before=previous,maximum_total_if_approved=total+1,jobs=[job])
    write(suite['suite_path'],{k:v for k,v in suite.items() if k!='jobs'})
    return suite


@pytest.fixture(scope='module')
def merged_history():
    from business_graph_summary_evaluation_v1.budget import budget_snapshot
    return budget_snapshot()


def test_latest_merged_evaluation_is_included_exactly_once_and_readonly(tmp_path,merged_history):
    from business_graph_summary_evaluation_v1 import budget as latest
    refs=budget.preceding_reference()
    before={v['path']:reference(v['path']) for v in refs.values() if isinstance(v,dict) and 'path' in v and Path(v['path']).is_file()}
    old=merged_history
    new=budget.budget_snapshot(tmp_path/'absent.sqlite')
    assert new==old
    selected=[j for j in new['jobs'].values() if j.get('campaign_id')=='business-graph-summary-best-checkpoint-eval100-20261118-001']
    assert len(selected)==100 and all(j['status']=='COMPLETE' for j in selected)
    assert refs['bg128_independent_extension_before']==reference(latest.GLOBAL_PATH)
    assert 'business_graph_budget_before' in refs
    assert not (tmp_path/'absent.sqlite').exists()
    assert all(reference(p)==r for p,r in before.items())


@pytest.mark.parametrize('key', ['run_index','experiment_seed','actor_sampling_seed','actor_seed','recording_id'])
def test_historical_eval_identity_or_cross_type_seed_is_rejected(key,merged_history):
    snapshot=merged_history
    row=next(r['identity'] for r in snapshot['jobs'].values() if r.get('campaign_id')=='business-graph-summary-best-checkpoint-eval100-20261118-001')
    job=dict(run_index=9960001,recording_id='fresh',source_recording_id='fresh',
        experiment_seed=731,actor_sampling_seed=732,actor_seed=733)
    job[key]=row[key]
    with pytest.raises(ValueError,match='collision'):
        budget.check_collisions(dict(campaign_id='fresh-stability',jobs=[job]),snapshot)


def test_preceding_byte_change_rejected_before_creating_database(tmp_path,monkeypatch):
    history=dict(campaigns={},jobs={},maximum_total_shifts=100,additional_reserved_shifts=0)
    monkeypatch.setattr(budget,'preceding_snapshot',lambda:deepcopy(history))
    monkeypatch.setattr(budget,'preceding_reference',lambda:dict(hash='changed'))
    monkeypatch.setattr(gate,'check_jobs',lambda value:value)
    monkeypatch.setattr(gate,'check_approval',lambda *args:None)
    suite=minimal_suite(tmp_path,100,dict(hash='frozen'))
    target=tmp_path/'fresh.sqlite'
    with pytest.raises(ValueError,match='preceding budget source changed'):
        budget.BudgetLedger(target,suite,campaign_sha256=reference(suite['suite_path'])['sha256'],
            approval={},approval_sha256='a'*64,test_only=True)
    assert not target.exists()


def test_new_ledger_adds_only_own_slots_and_preserves_unknown(tmp_path,monkeypatch):
    history=dict(campaigns={'old':{}},jobs={'9':dict(status='UNKNOWN',identity=dict(run_index=9))},
        maximum_total_shifts=100,additional_reserved_shifts=0)
    monkeypatch.setattr(budget,'preceding_snapshot',lambda:deepcopy(history))
    monkeypatch.setattr(budget,'preceding_reference',lambda:dict(hash='frozen'))
    monkeypatch.setattr(gate,'check_jobs',lambda value:value)
    monkeypatch.setattr(gate,'check_approval',lambda *args:None)
    suite=minimal_suite(tmp_path,100,dict(hash='frozen'))
    ledger=budget.BudgetLedger(tmp_path/'new.sqlite',suite,
        campaign_sha256=reference(suite['suite_path'])['sha256'],approval={},approval_sha256='a'*64,test_only=True)
    state=budget.budget_snapshot(ledger.path)
    assert state['maximum_total_shifts']==101 and state['jobs']['9']['status']=='UNKNOWN'
    assert len(ledger.snapshot()['jobs'])==1
    job=suite['jobs'][0];ledger.claim(job)
    with pytest.raises(ValueError):
        ledger.claim(job)
    ledger.unknown(job,'synthetic_failure')
    with pytest.raises(ValueError,match='UNKNOWN|CLAIMED'):
        ledger.resume()
    monkeypatch.setattr(budget,'preceding_reference',lambda:dict(hash='changed_after_registration'))
    with pytest.raises(ValueError,match='registered preceding ledger bytes changed'):
        budget.budget_snapshot(ledger.path)
