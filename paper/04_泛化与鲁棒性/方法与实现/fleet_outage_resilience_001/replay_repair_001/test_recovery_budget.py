import json
import sqlite3
from pathlib import Path

import pytest

import recovery_budget as budget
import repair_campaign as campaign


def test_sealed_prefix_and_roster():
    src=campaign.source_suite()
    audit=budget.audit_source(src)
    assert (audit['complete_count'],audit['remaining_count'])==(32,960)
    target=budget.ROOT/'runs'/budget.CAMPAIGN
    jobs=[dict(campaign.replacement(j,target),config_sha256='new')
          if i<32 else dict(j) for i,j in enumerate(src['jobs'])]
    s=dict(src,jobs=jobs,output=str(target),source_suite=budget.source_ref(),
           campaign_id=budget.CAMPAIGN,extra_count=32)
    budget.validate_roster(s,src)
    jobs[32]=dict(src['jobs'][0])
    with pytest.raises(ValueError,match='retained'):
        budget.validate_roster(s,src)


def test_register_settle_and_cross_ledger_order(tmp_path,monkeypatch):
    original=tmp_path/'original'
    output=tmp_path/'repair'
    original.mkdir()
    monkeypatch.setattr(budget,'source_output',lambda:original)
    monkeypatch.setattr(budget.old.previous,'LOCKS',tmp_path/'locks')
    # Exercise all I/O on temporary ledgers. Registration's offline package audit
    # is independently exercised on the immutable original in the test above.
    monkeypatch.setattr(budget,'check_fresh',lambda suite:None)
    def job(run,root):
        return dict(run_index=run,actor_seed=12,experiment_seed=run,
                    actor_sampling_seed=run+20000000,config_sha256=str(run),
                    recording_id=str(run),source_recording_id=str(run),
                    event_directory=str(root/f'events/{run}'),python_output=str(root/f'shifts/{run}'))
    source_jobs=[job(i,original) for i in range(10043001,10043993)]
    extra=[job(i,output) for i in budget.NEW_RUNS]
    old_ledger=budget.old.ledger_path(dict(output=original))
    with sqlite3.connect(old_ledger) as db:
        db.execute('CREATE TABLE campaign(name TEXT,suite TEXT)')
        db.execute('INSERT INTO campaign VALUES (?,?)',('original',json.dumps(budget.source_ref())))
        db.execute('CREATE TABLE slots(run INTEGER PRIMARY KEY,id TEXT,status TEXT,evidence TEXT)')
        db.executemany('INSERT INTO slots VALUES (?,?,?,NULL)',
                       [(j['run_index'],budget.encoded(budget.old.identity(j)).decode(),
                         'COMPLETE' if i<32 else 'PLANNED') for i,j in enumerate(source_jobs)])
    pin=tmp_path/'repair.json'
    pin.write_text('fixture')
    s=dict(jobs=[*extra,*source_jobs[32:]],source_suite=budget.source_ref(),output=str(output),
           suite_path=str(pin),campaign_id=budget.CAMPAIGN)
    budget.register(s)
    assert len(budget.ledger_rows(budget.old.ledger_path(s))[2])==32
    view=budget.state_view(s)
    assert len(view)==992 and all(st=='PLANNED' for st,_ in view.values())
    with pytest.raises(ValueError,match='consumed'):
        budget.claim(s,source_jobs[0])
    with pytest.raises(ValueError,match='batch not complete'):
        budget.claim(s,source_jobs[32])
    with pytest.raises(ValueError,match='earlier run'):
        budget.claim(s,extra[1])
    def complete(j):
        budget.claim(s,j)
        receipt=Path(j['python_output'])/'receipt.json'
        receipt.parent.mkdir(parents=True)
        receipt.write_bytes(budget.encoded(j))
        budget.complete(s,j,receipt)
    for j in extra:
        complete(j)
    with pytest.raises(ValueError,match='earlier run'):
        budget.claim(s,source_jobs[33])
    complete(source_jobs[32])
    with pytest.raises(ValueError,match='invalid transition'):
        budget.claim(s,source_jobs[32])
    budget.claim(s,source_jobs[33])
    budget.unknown(s,source_jobs[33],'fixture failure')
    final=budget.state_view(s)
    assert len(final)==992 and sum(st=='COMPLETE' for st,_ in final.values())==33
    assert final[10043034][0]=='UNKNOWN'
    assert budget.ledger_rows(old_ledger)[2][10043033][1]=='COMPLETE'
    assert budget.ledger_rows(old_ledger)[2][10043001][1:] == ('COMPLETE',None)
