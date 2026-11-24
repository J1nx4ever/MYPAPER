"""Recovery provenance/ready/run boundaries without listener or ledger writes."""
from contextlib import nullcontext
import copy
import json
from pathlib import Path
from types import SimpleNamespace

from . import runtime
from .plan_io import ROOT, load_suite, reference, write


ORIGINAL=ROOT/'change_plans/token_attention_001/business_graph_summary_009/formal_training_001/prepared_001'


def fixture(tmp_path):
    suite,phases=load_suite(ORIGINAL/'suite.json')
    suite['control_directory']=str(tmp_path/'new_control')
    suite['_original_approval_path']=str(ORIGINAL/'approval.json')
    recovery=tmp_path/'recovery.json';write(recovery,{'isolated_boundary_test':True})
    suite['_recovery_ref']=reference(recovery)
    return suite,phases,recovery


def test_transaction_keeps_original_phase_and_adds_explicit_recovery_reference(tmp_path):
    suite,phases,recovery=fixture(tmp_path)
    intent=runtime._intent(suite,suite['suite_path'],phases[0],[])
    assert intent['recovery']==reference(recovery)
    assert intent['campaign']==reference(ORIGINAL/'suite.json')
    assert intent['run_indices'][0]==9900001 and intent['run_indices'][-1]==9900032
    assert intent['update']==1 and intent['arm_update']==1


def test_ready_uses_new_control_and_original_job_config_master_identity(tmp_path,monkeypatch):
    suite,phases,recovery=fixture(tmp_path)
    job=phases[0]['jobs'][1];assert job['run_index']==9900002
    called=[]
    monkeypatch.setattr(runtime,'check_sources',lambda pins,ref:called.append((pins,ref)))
    ledger=SimpleNamespace(job=lambda observed:dict(status='CLAIMED',identity=observed))
    runtime._publisher(suite,ledger)(job,suite['phase_configs'][0]['path'])
    ready=(Path(suite['control_directory'])/'ready'/'run_9900002.properties').read_text()
    assert 'run_index=9900002\n' in ready
    assert 'config_sha256='+suite['phase_configs'][0]['sha256']+'\n' in ready
    assert 'master_plan_sha256='+suite['master_plan']['sha256']+'\n' in ready
    assert 'recovery_descriptor_sha256='+reference(recovery)['sha256']+'\n' in ready
    assert called==[(suite['source_pins'],reference(recovery))]
    assert not (Path(suite['control_directory'])/'ready'/'run_9900001.properties').exists()


def test_run_reuses_original_ledger_and_approval_without_second_registration(tmp_path,monkeypatch):
    from . import budget
    suite,phases,recovery=fixture(tmp_path)
    approval=json.loads((ORIGINAL/'approval.json').read_bytes())
    calls=[]
    class Ledger:
        def __init__(self,path,observed,**values):
            calls.append((path,observed,values))
        def resume(self):return {'isolated':True}
    monkeypatch.setattr(runtime,'preflight',lambda path:(suite,phases,approval))
    monkeypatch.setattr(budget,'BudgetLedger',Ledger)
    monkeypatch.setattr(runtime,'_controller_lock',lambda path:nullcontext())
    monkeypatch.setattr(runtime,'_trial',lambda path,current,observed,ledger:dict(
        original_suite_path=path,recovery=current['_recovery_ref'],phase_count=len(observed)))
    result=runtime.run(recovery)
    assert result['original_suite_path']==str(ORIGINAL/'suite.json')
    assert result['phase_count']==16
    assert len(calls)==1 and calls[0][0]==budget.GLOBAL_PATH
    assert calls[0][2]['campaign_sha256']==reference(ORIGINAL/'suite.json')['sha256']
    assert calls[0][2]['approval_sha256']==reference(ORIGINAL/'approval.json')['sha256']
    assert calls[0][2]['approval']==approval
