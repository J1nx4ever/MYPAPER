"""Offline tests for repair controller, actual optimizer and receipt boundaries."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import repair_runtime as rt
from replay_policy import FleetLearner
from test_fleet_policy import episode


def test_rejected_likelihood_still_accounts_for_physical_end(tmp_path):
    saved = []
    job = dict(role='train', python_output=str(tmp_path/'physical'))
    shift = SimpleNamespace(episode={}, receipt={'terminal_time_seconds':28800})
    def reject(_):
        raise ValueError('replay differs')
    budget = SimpleNamespace(complete=lambda *args: saved.append(args[-1]))
    with pytest.raises(ValueError,match='replay differs'):
        rt.settle_shift({}, job, shift, SimpleNamespace(verify_collected=reject), budget)
    assert len(saved) == 1
    proof = json.loads(saved[0].read_bytes())
    assert proof['terminal_time_seconds'] == 28800 and not proof['likelihood_replay']['passed']


def test_offline_java_bridge_and_real_32_shift_update(tmp_path, monkeypatch):
    """Complete three short training batches and four paired final slots, without sockets or simulation."""
    parent = rt.ROOT/'runs/bg128-fleet-resilience-20261124-001/models/warmup/u0000/checkpoint.pt'
    seed = FleetLearner.load(parent).config['seed']
    root = tmp_path/'model'
    rows = []
    jobs = []
    for arm in ('warmup','static','dynamic','evaluation'):
        for i in range(32 if arm!='evaluation' else 8):
            ordinal = len(rows)
            run = 1_000_000+ordinal
            role = 'eval' if arm=='evaluation' else 'train'
            policy = arm if role=='train' else ('dynamic' if i%4>=2 else 'static')
            config = tmp_path/f'config_{run}.json'
            config.write_text(str(run))
            job=dict(role=role,arm=policy,update=1,ordinal=ordinal,run_index=run,recording_id=str(run),
                     config_path=str(config),config_sha256=rt.sha(config),actor_sampling_seed=1000+run,
                     experiment_seed=run,python_output=str(tmp_path/'receipts'/str(run)))
            jobs.append(job)
            rows.append(dict(ordinal=ordinal,run_index=run,recording_id=str(run)))
    master=tmp_path/'master.csv'
    with master.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    suite=dict(campaign_id='offline-complete',jobs=jobs,branch_seed=120,learner_seed=seed,
               master_plan=rt.reference(master),source_u0=rt.reference(parent),output=str(root),
               control_directory=str(root/'control'))
    states={j['run_index']:('PLANNED',None) for j in jobs}
    def claim(_,j):
        assert states[j['run_index']]==('PLANNED',None)
        states[j['run_index']]=('CLAIMED',None)
    def complete(_,j,p):
        assert states[j['run_index']]==('CLAIMED',None)
        states[j['run_index']]=('COMPLETE',json.dumps(rt.reference(p)))
    budget=SimpleNamespace(register=lambda s:None,claim=claim,complete=complete,state_view=lambda s:states)
    class Service:
        def __init__(self):
            from threading import RLock
            self.lock=RLock()
            self.finished=self
            self.failure=None
        def arm(self,shift,hand):
            self.shift,self.hand=shift,hand
        def wait(self,_):
            hand=self.hand
            assert hand.ready_file.exists()
            hand.claim_file.parent.mkdir(parents=True,exist_ok=True)
            hand.claim_file.write_bytes(hand.claim_bytes())
            assert hand.claim(final=True)
            return True
    def shift(j,model,verify):
        sample = episode(model,j['run_index']) if j['role']=='train' else None
        return SimpleNamespace(episode=sample, receipt=dict(j,policy_version=model.policy_version))
    server = SimpleNamespace(start=lambda:None, add_insecure_port=lambda address:1,
                             stop=lambda grace:SimpleNamespace(wait=lambda:None))
    reports=[]
    monkeypatch.setattr(rt.grpc,'server',lambda *a,**k:server)
    monkeypatch.setattr(rt.dispatch_pb2_grpc,'add_DispatchServiceServicer_to_server',lambda *a:None)
    monkeypatch.setattr(rt,'LiveRPC',Service)
    monkeypatch.setattr(rt,'Shift',shift)
    monkeypatch.setattr(rt,'make_report',lambda s,b:reports.append(s['campaign_id']))
    rt.serve(suite,budget=budget)
    assert reports==['offline-complete']
    updates={'warmup':1,'static':1,'dynamic':1}
    assert rt.recovered_prefix(suite,budget,rows)==(104,updates)
    assert json.loads((root/'suite_complete.json').read_bytes())['updates']==updates
    warm=FleetLearner.load(root/'models/warmup/u0001/checkpoint.pt')
    assert all(FleetLearner.load(root/f'models/{a}/u0000/checkpoint.pt').policy_version==warm.policy_version
               for a in ('static','dynamic'))
    for j in jobs[-8:]:
        used = json.loads((Path(j['python_output'])/'receipt.json').read_bytes())['policy_version']
        assert used==FleetLearner.load(root/f"models/{j['arm']}/u0001/checkpoint.pt").policy_version
