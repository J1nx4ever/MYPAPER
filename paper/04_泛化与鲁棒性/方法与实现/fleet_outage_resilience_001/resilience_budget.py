"""Merge historical shifts, frozen identities and admitted resilience campaigns."""
import json
from pathlib import Path
import sqlite3
import sys

from resilience_campaign import PLAN, ROOT, digest, encoded, reference

STRONG = PLAN.parent/'baseline_comparison_001/or_paper_comparison_001/paired100_strong_001'
sys.path.insert(0,str(ROOT)) if str(ROOT) not in sys.path else None
sys.path.insert(0,str(STRONG)) if str(STRONG) not in sys.path else None
import strong100_campaign as previous
from python.offline_split import JAVA_MASK, java_random_states
from business_graph_stability_campaign_v1.launch import _launch_lock


def ledger_path(suite):
    return Path(suite['output'])/'fleet_resilience_budget.sqlite'


def identity(job):
    return {k:job[k] for k in ('run_index','actor_seed','actor_sampling_seed','experiment_seed','recording_id',
                                'source_recording_id','python_output','event_directory','config_sha256')}


def budget_snapshot():
    result = previous.budget_snapshot()
    for path in sorted((ROOT/'runs').glob('*/fleet_resilience_budget.sqlite')):
        with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True) as db:
            name,pin = db.execute('select name,suite from campaign').fetchone()
            rows = db.execute('select id,status,evidence from slots').fetchall()
        if name in result['campaigns'] or not rows:
            raise ValueError('duplicate or empty resilience ledger: '+name)
        result['campaigns'][name]=dict(campaign=json.loads(pin),cap=len(rows),commits={})
        for id_text,status,evidence in rows:
            item=json.loads(id_text)
            run=str(item['run_index'])
            if run in result['jobs']:
                raise ValueError('overlapping physical run '+run)
            result['jobs'][run]=dict(identity=item,status=status,campaign_id=name)
            if evidence:
                result['jobs'][run]['receipt' if status=='COMPLETE' else 'evidence']=json.loads(evidence)
        result['maximum_total_shifts']+=len(rows)
        result['additional_reserved_shifts']+=len(rows)
    return result


def check_fresh(suite):
    history=budget_snapshot()
    if suite['campaign_id'] in history['campaigns']:
        raise ValueError('already registered campaign')
    # The historical collector scans the graph's suite files and the older sidecar rosters.
    from weak_plan import bind
    prior=bind(previous.paired10._prior_rows, SUITE=Path(suite['suite_path']))(history)
    # Strong100 excludes its own file inside its own gate; here it belongs to the historical inventory.
    prior.extend(row['identity'] for row in history['jobs'].values() if isinstance(row.get('identity'),dict))
    used_runs={j.get('run_index') for j in prior}
    recordings={j.get(k) for j in prior for k in ('recording_id','source_recording_id')}
    keys=('experiment_seed','actor_seed','actor_sampling_seed')
    used_seeds={j[k]&JAVA_MASK for j in prior for k in keys if type(j.get(k)) is int and j[k]>0}
    used_states={v for s in {j.get('experiment_seed') for j in prior if type(j.get('experiment_seed')) is int and j['experiment_seed']>0}
                 for v in java_random_states(s).values()}
    env,actor={},{}
    runs,seens=set(),set()
    for j in suite['jobs']:
        run,rec=j['run_index'],j['recording_id']
        if run in used_runs or run in runs or rec in recordings or rec in seens:
            raise ValueError(f'run/recording collision at {run}')
        runs.add(run)
        seens.add(rec)
        group=(j['split'],j['pair_ordinal']) if j['role']=='eval' else ('train',j['ordinal'])
        for key,seen in (('experiment_seed',env),('actor_sampling_seed',actor)):
            value=j[key]
            if not 0 < value <= JAVA_MASK or value in used_seeds or seen.setdefault(value,group)!=group:
                raise ValueError('unexpected seed collision: '+key)
    new_seeds=set(env)|set(actor)
    init={suite['learner_seed'],suite['branch_seed']}
    states={v for s in env for v in java_random_states(s).values()}
    if (set(env)&set(actor) or init & (new_seeds|used_seeds) or len(init)!=2 or
            len(states)!=5*len(env) or states & used_states):
        raise ValueError('random stream or Java effective state collision')
    return dict(registered_before=history['maximum_total_shifts'],additional_if_started=len(runs),
                total_if_started=history['maximum_total_shifts']+len(runs),new_environment_seeds=len(env),
                new_actor_sampling_seeds=len(actor))


def register(suite):
    path=ledger_path(suite)
    with _launch_lock(previous.LOCKS/'budget'):
        if path.exists():
            with sqlite3.connect(path) as db:
                if db.execute('SELECT name,suite FROM campaign').fetchone() != (
                        suite['campaign_id'],encoded(reference(suite['suite_path'])).decode()):
                    raise ValueError('ledger belongs to another frozen package')
            return
        check_fresh(suite)
        # Publish only a complete inventory so concurrent snapshots cannot see a partial ledger.
        staging=path.with_suffix('.registering')
        if staging.exists():
            raise FileExistsError(staging)
        path.parent.mkdir(parents=True,exist_ok=True)
        with sqlite3.connect(staging) as db:
            db.execute('CREATE TABLE campaign(name TEXT PRIMARY KEY,suite TEXT NOT NULL)')
            db.execute('CREATE TABLE slots(run INTEGER PRIMARY KEY,id TEXT NOT NULL,status TEXT NOT NULL,evidence TEXT)')
            db.execute('INSERT INTO campaign VALUES (?,?)',(suite['campaign_id'],encoded(reference(suite['suite_path'])).decode()))
            db.executemany("INSERT INTO slots VALUES (?,?,'PLANNED',NULL)",
                           [(j['run_index'],encoded(identity(j)).decode()) for j in suite['jobs']])
        staging.rename(path)


def transition(suite,job,before,after,evidence=None):
    path=ledger_path(suite)
    with sqlite3.connect(path.as_uri()+'?mode=rw',uri=True) as db:
        db.execute('BEGIN IMMEDIATE')
        if after=='CLAIMED' and db.execute(
                "SELECT 1 FROM slots WHERE run<? AND status!='COMPLETE' LIMIT 1",(job['run_index'],)).fetchone():
            raise ValueError('earlier run unresolved')
        count=db.execute('UPDATE slots SET status=?,evidence=? WHERE run=? AND id=? AND status=?',
                         (after,encoded(evidence).decode() if evidence else None,job['run_index'],
                          encoded(identity(job)).decode(),before)).rowcount
        if count!=1:
            raise ValueError(f'invalid transition {before}->{after} for {job["run_index"]}')


def claim(suite,job):
    if not ledger_path(suite).exists():
        register(suite)
    transition(suite,job,'PLANNED','CLAIMED')


def complete(suite,job,receipt_path):
    path=Path(receipt_path)
    if path!=Path(job['python_output'])/'receipt.json':
        raise ValueError('unexpected receipt path')
    receipt=json.loads(path.read_bytes())
    for k in ('run_index','recording_id','config_sha256','actor_sampling_seed','experiment_seed'):
        if receipt.get(k)!=job[k]:
            raise ValueError('settlement differs: '+k)
    transition(suite,job,'CLAIMED','COMPLETE',reference(path))


def unknown(suite,job,reason):
    transition(suite,job,'CLAIMED','UNKNOWN',{'reason':str(reason)})
