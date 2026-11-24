"""Offline durability checks for the Java-start bridge and an interrupted PPO batch."""
import json
import sqlite3
from types import SimpleNamespace

import pytest

from resilience_runtime import Handshake, record, recovered_prefix, sha


def test_recover_only_sealed_batches(tmp_path):
    root = tmp_path
    jobs = [dict(ordinal=i,run_index=1000+i,recording_id=f'fixture-{i}',experiment_seed=2+i,
                 actor_sampling_seed=500+i,config_sha256='a'*64,arm='warmup',role='train',
                 update=1+i//32,python_output=str(root/'shifts'/str(i))) for i in range(33)]
    rows = [{} for _ in jobs]
    suite = dict(output=str(root),control_directory=str(root/'control'),master_plan={'sha256':'b'*64},
                 campaign_id='offline-fixture',jobs=jobs)
    ledger = root/'fixture.sqlite'
    claims = []
    budget = SimpleNamespace(ledger_path=lambda suite: ledger,claim=lambda suite,j: claims.append(j['run_index']))
    with sqlite3.connect(ledger) as db:
        db.execute('CREATE TABLE slots(run INTEGER PRIMARY KEY,status TEXT,evidence TEXT)')
        db.executemany("INSERT INTO slots VALUES (?,'PLANNED',NULL)", [(j['run_index'],) for j in jobs])
    assert recovered_prefix(suite,budget,rows)[0] == 0
    for j in jobs[:32]:
        hand = Handshake(suite,j,{},budget.claim)
        hand.claim_file.parent.mkdir(parents=True,exist_ok=True)
        content = hand.claim_bytes()
        hand.claim_file.write_bytes(content[:-5])
        assert not hand.claim()
        hand.claim_file.write_bytes(content)
        assert hand.claim(final=True) and hand.claim()
        receipt = root/'shifts'/str(j['ordinal'])/'receipt.json'
        record(receipt,j)
        with sqlite3.connect(ledger) as db:
            db.execute("UPDATE slots SET status='COMPLETE',evidence=? WHERE run=?",
                       (json.dumps(dict(path=str(receipt),sha256=sha(receipt))),j['run_index']))
        if j['ordinal'] == 1:
            with pytest.raises(ValueError,match='partial training batch'):
                recovered_prefix(suite,budget,rows)
    folder = root/'models/warmup/u0001'
    folder.mkdir(parents=True)
    checkpoint = folder/'checkpoint.pt'
    checkpoint.write_bytes(b'synthetic checkpoint bytes for receipt/seal boundary check')
    manifest = folder/'checkpoint.pt.manifest.json'
    record(manifest,dict(update_index=1,bytes=checkpoint.stat().st_size,sha256=sha(checkpoint)))
    record(folder/'checkpoint.pt.seal.json',dict(manifest_sha256=sha(manifest),sha256=sha(checkpoint)))
    key = 'offline-fixture:warmup:1'
    record(folder/'intent.json',dict(batch_id=key,runs=[j['run_index'] for j in jobs[:32]]))
    record(folder/'update.json',dict(checkpoint_sha256=sha(checkpoint),summary={'update_index':1,'batch_id':key}))
    count,updates = recovered_prefix(suite,budget,rows)
    assert count == 32 and updates == dict(warmup=1,static=0,dynamic=0)
    assert len(claims) == len(set(claims)) == 32
    unresolved = Handshake(suite,jobs[-1],{},budget.claim)
    unresolved.claim_file.write_bytes(unresolved.claim_bytes())
    with pytest.raises(ValueError,match='unsettled physical run'):
        recovered_prefix(suite,budget,rows)
