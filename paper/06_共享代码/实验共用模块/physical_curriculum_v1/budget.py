"""Frozen budget registry plus small atomic SQLite slot transactions.

All previous ledgers are read only. Registry reserves exactly6844 once. Dynamic
state stores only run keys, receipts and commits; immutable jobs live in the
SHA-pinned compact plan. Unknown or claimed slots are never auto-resubmitted.
"""
import copy
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
from types import MappingProxyType
import uuid
from baseline1_v1 import budget as legacy
from grade_multiobjective_ppo_v1 import budget_policy as prior
from u8_wait_efficiency_v1 import budget as preceding
from physical_realization_ppo_v1 import budget as physical_previous
from . import gate
ROOT=legacy.ROOT
GLOBAL_PATH=ROOT/'runs/physical-curriculum-budget-v1.json'
FORMAT='physical-curriculum-budget-v1'
CAP,TRAIN_UPDATES=6844,160
require=legacy.require
_SOURCE_CACHE=None
_SOURCE_TIMING=dict(cold_builds=0,byte_verified_cache_hits=0,cold_build_seconds=0.,cache_byte_verification_seconds=0.)
def _byte_reference(path):
    path=Path(path).resolve()
    checksum=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            checksum.update(chunk)
    return dict(path=str(path),sha256=checksum.hexdigest())


def _freeze(value):
    if isinstance(value,dict):return MappingProxyType({k:_freeze(v) for k,v in value.items()})
    if isinstance(value,list):return tuple(_freeze(v) for v in value)
    return value


def _thaw(value):
    if isinstance(value,(dict,MappingProxyType)):return {k:_thaw(v) for k,v in value.items()}
    if isinstance(value,tuple):return [_thaw(v) for v in value]
    return value


def source_cache_timing():
    """Compact cumulative diagnostics; no printing or per-transition records."""
    return dict(_SOURCE_TIMING)


def _sources():
    """Explicitly validate/merge physical608; never weaken its old unknown-ledger guard."""
    known={'baseline1-budget-v1.json','active-budget-v2.json','independent-evaluation-budget-v1.json',
           'sampling-comparison-budget-v1.json','u8-wait-efficiency-budget-v1.json',
           'physical-realization-ppo-budget-v1.json','physical-wait-tradeoff-budget-v1.json',GLOBAL_PATH.name}
    require(not [p.name for p in (ROOT/'runs').glob('*budget*.json') if p.name not in known],
            'unknown additional ledger must be explicitly integrated')
    require(not (ROOT/'runs/physical-wait-tradeoff-budget-v1.json').exists(),
            'old812 was registered; reconcile scope before unified6844, never double-run')
    global _SOURCE_CACHE
    started=time.monotonic()
    if _SOURCE_CACHE is not None:
        for ref in _SOURCE_CACHE['dependencies']:
            require(_byte_reference(ref['path'])==ref,'immutable preceding budget evidence byte SHA changed')
        _SOURCE_TIMING['byte_verified_cache_hits']+=1
        _SOURCE_TIMING['cache_byte_verification_seconds']+=time.monotonic()-started
        return _SOURCE_CACHE['snapshot'],copy.deepcopy(_SOURCE_CACHE['state'])
    snapshot=preceding.budget_snapshot()
    state=preceding.budget_state_reference()
    require(snapshot['maximum_total_shifts']==4012 and
            snapshot['legacy_reserved_shifts']==snapshot['active_h2_reserved_shifts']==0,
            'complete preceding4012 cancelled-H2 budget required')
    path=physical_previous.GLOBAL_PATH
    require(path.is_file(),'preceding physical608 ledger required')
    data=json.loads(path.read_bytes())
    checksum=data.pop('document_sha256',None)
    require(checksum==legacy._digest(data) and data.get('format')==physical_previous.FORMAT and
            data.get('test_only') is False,'physical608 checksum/scope changed')
    require(all(data.get(key)==value for key,value in state.items()) and
            data.get('preceding_reserved_shifts')==4012 and data.get('physical_ppo_reserved_shifts')==608 and
            data.get('maximum_total_shifts')==data.get('additional_reserved_shifts')==4620 and
            data.get('legacy_reserved_shifts')==data.get('active_h2_reserved_shifts')==0,
            'physical608 preceding references or capacity changed')
    require(len(data['campaigns'])==1 and len(data['jobs'])==608,'physical608 inventory changed')
    for cid,entry in data['campaigns'].items():
        campaign=entry['campaign']
        physical_previous._check_jobs(campaign)
        physical_previous.check_approval(entry['approval'],campaign,entry['campaign_sha256'])
        require(entry['campaign_identity_sha256']==legacy._digest(campaign) and
                entry['approval_identity_sha256']==legacy._digest(entry['approval']),
                'physical608 campaign/approval checksum changed')
        expected={str(job['run_index']):job for job in campaign['jobs']}
        require(set(expected)==set(data['jobs']) and len(entry['commits'])==16,
                'physical608 jobs/commits incomplete')
        for key,row in data['jobs'].items():
            require(row['campaign_id']==cid and row['identity']==expected[key] and
                    row['identity_sha256']==legacy._digest(row['identity']) and row['status']=='COMPLETE' and
                    row.get('claim_sha256')==legacy._digest(row['claim']) and
                    row.get('completion_sha256')==legacy._digest(dict(dataset_ref=row['dataset_ref'],receipt=row['receipt'])),
                    'physical608 state/identity/receipt changed')
            physical_previous._check_receipt(row['identity'],row['receipt'])
        physical_previous._validate_commits(entry,data['jobs'])
    require(set(snapshot['campaigns']).isdisjoint(data['campaigns']) and
            set(snapshot['jobs']).isdisjoint(data['jobs']),'physical608 historical collision')
    snapshot=copy.deepcopy(snapshot)
    snapshot['campaigns'].update(data['campaigns']);snapshot['jobs'].update(data['jobs'])
    for key in ('physical_ppo_reserved_shifts','additional_reserved_shifts','maximum_total_shifts'):
        snapshot[key]=data[key]
    state=dict(state,physical_ppo_budget_before=_byte_reference(path))
    dependencies={}
    def collect(value):
        if isinstance(value,dict):
            if set(value)=={'path','sha256'}:
                require(_byte_reference(value['path'])==value,'preceding static evidence byte SHA differs')
                dependencies[value['path']]=dict(value)
            else:
                for child in value.values():collect(child)
        elif isinstance(value,(list,tuple)):
            for child in value:collect(child)
    collect(state)
    policy=json.loads(Path(state['budget_policy']['path']).read_bytes())
    collect(policy['legacy_budget'])
    collect(policy.get('historical_h2_snapshot',{}).get('evidence',[]))
    collect(prior.H2_IDENTITY_REFERENCES)
    collect(snapshot.get('recovery_history',{}))
    _SOURCE_CACHE=dict(snapshot=_freeze(snapshot),state=copy.deepcopy(state),
                       dependencies=tuple(dependencies.values()))
    _SOURCE_TIMING['cold_builds']+=1
    _SOURCE_TIMING['cold_build_seconds']+=time.monotonic()-started
    return _SOURCE_CACHE['snapshot'],copy.deepcopy(state)



def budget_state_reference(path=None):
    _,state=_sources();path=GLOBAL_PATH if path is None else Path(path)
    return dict(state,curriculum_budget_before=gate.reference(path) if path.is_file() else None)

def _history_rows(snapshot):
    rows=[r['identity'] for r in snapshot.get('jobs',{}).values()]
    rows.extend(snapshot.get('historical_h2_identities',[]))
    for record in snapshot.get('recovery_history',{}).values():
        old=legacy._read_reference(dict(record['old_ledger_snapshot']));checksum=old.pop('document_sha256',None)
        require(checksum==legacy._digest(old),'retired budget checksum changed')
        rows.extend(r['identity'] for r in old['jobs'].values())
    return rows

def check_collisions(suite,snapshot):
    require(suite['campaign_id'] not in snapshot.get('campaigns',{}),'campaign identity already consumed')
    rows=_history_rows(snapshot);jobs=suite['jobs']
    for key in ('run_index','recording_id','source_recording_id'):
        old={r.get(key) for r in rows}-{None}
        require(all(j.get(key) not in old for j in jobs),'historical identity collision: '+key)
    # Three immutable sources explicitly authorize same-partition seed pairing.
    allowed={}
    for source in (gate.physical_suite(),gate.historical_suite(),gate.previous.historical_suite()):
        for j in source['jobs']:allowed[j['run_index']]=j
    fields=('experiment_seed','actor_sampling_seed','actor_seed')
    requested={f:{} for f in fields}
    for job in jobs:
        for f in fields:requested[f].setdefault(job[f],[]).append(job)
    for row in rows:
        for field in fields:
            value=row.get(field)
            require(not any(value in requested[other] for other in fields if other!=field),'cross-type historical seed collision')
            matches=requested[field].get(value,[])
            if not matches:continue
            source=allowed.get(row.get('run_index'))
            require(source is not None and source['recording_id']==row.get('recording_id') and source.get(field)==value,'seed reused outside frozen source inventory')
            if field!='actor_seed':
                require(all(j['partition']==source['partition'] and j['eval_kind'] in ('train','benchmark') for j in matches),'new monitoring/blind seed or split overlaps history')

def _check_jobs(suite): return gate.check_jobs(suite)
def check_approval(approval,suite,campaign_sha256): return gate.check_approval(approval,suite,campaign_sha256)

def _record_checksum(value):return gate.digest(value)
def _dump(value):return gate.encoded(value).decode()

def _checked(text,checksum,label):
    value=json.loads(text);require(_record_checksum(value)==checksum,label+' checksum changed');return value

def _registry(path):
    data=json.loads(Path(path).read_bytes());checksum=data.pop('document_sha256',None)
    require(checksum==_record_checksum(data) and data.get('format')==FORMAT,'curriculum registry checksum/format changed')
    return data

def _atomic_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    value=dict(data,document_sha256=_record_checksum(data));temp=path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
    try:
        with temp.open('xb') as f:f.write(gate.encoded(value));f.flush();os.fsync(f.fileno())
        os.replace(temp,path)
        fd=os.open(path.parent,os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
    finally:
        if temp.exists():temp.unlink()

@contextmanager
def _locked(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+b') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        try:yield
        finally:fcntl.flock(lock.fileno(),fcntl.LOCK_UN)

def _connect(path,readonly=False):
    require(Path(path).is_file(),'registered SQLite state missing; never reinitialize')
    con=sqlite3.connect(Path(path).as_uri()+'?mode=ro' if readonly else str(path),uri=readonly,timeout=30)
    con.row_factory=sqlite3.Row
    if not readonly:con.execute('PRAGMA synchronous=FULL')
    return con

def _head(con):
    row=con.execute('SELECT record,checksum FROM metadata WHERE name="head"').fetchone()
    require(row is not None,'durable state head missing')
    result=_checked(row['record'],row['checksum'],'state head')
    require(set(result)=={'claimed_slots','complete','unknown','commits','revision'} and
        all(type(v)is int and v>=0 for v in result.values()),'state head malformed')
    return result

def _advance(con,**increments):
    head=_head(con)
    for key,value in increments.items():head[key]+=value
    head['revision']+=1
    con.execute('UPDATE metadata SET record=?,checksum=? WHERE name="head"',(_dump(head),gate.digest(head)))

def _read_rows(con,suite):
    require(con.execute('PRAGMA quick_check').fetchone()[0]=='ok','SQLite integrity check failed')
    jobs={str(j['run_index']):dict(campaign_id=suite['campaign_id'],identity=j,identity_sha256=gate.digest(j),status='PLANNED') for j in suite['jobs']}
    rows=con.execute('SELECT run_index,record,checksum FROM slots ORDER BY run_index').fetchall()
    head=_head(con)
    require(head['claimed_slots']==len(rows) and [r['run_index'] for r in rows]==[j['run_index'] for j in suite['jobs'][:len(rows)]],
        'durable claimed inventory deleted or discontinuous')
    for r in rows:
        key=str(r['run_index']);require(key in jobs,'SQLite slot absent from frozen plan')
        payload=_checked(r['record'],r['checksum'],'slot')
        require(payload.get('status') in ('CLAIMED','COMPLETE','UNKNOWN') and payload.get('identity_sha256')==jobs[key]['identity_sha256'],'slot status/identity differs')
        require(payload.get('claim_sha256')==gate.digest(payload.get('claim')),'claim receipt differs')
        if payload['status']=='COMPLETE':
            _check_receipt(jobs[key]['identity'],payload['receipt'])
            require(payload.get('completion_sha256')==gate.digest(dict(dataset_ref=payload['dataset_ref'],receipt=payload['receipt'])),'completion receipt differs')
        jobs[key].update(payload)
    commits={}
    for r in con.execute('SELECT update_no,record,checksum FROM commits ORDER BY update_no'):
        item=_checked(r['record'],r['checksum'],'commit');commits[str(r['update_no'])]=item
    require(list(map(int,commits))==list(range(1,len(commits)+1)) and len(commits)<=TRAIN_UPDATES,'commit sequence broken')
    for key,item in commits.items():
        expected=[j for j in suite['jobs'] if j['role']=='train' and j['target_training_update']==int(key)]
        require(len(expected)==32 and set(item['job_ids'])=={str(j['run_index']) for j in expected},'commit physical batch differs')
        require(item.get('commit_sha256')==gate.digest({k:v for k,v in item.items() if k!='commit_sha256'}),'commit digest differs')
        for job in expected:
            row=jobs[str(job['run_index'])]
            require(row['status']=='COMPLETE' and row.get('committed_update')==int(key) and item['receipt_sha256'].get(str(job['run_index']))==row['completion_sha256'],'committed slot differs')
    require(head['commits']==len(commits) and head['complete']==sum(r['status']=='COMPLETE' for r in jobs.values()) and
        head['unknown']==sum(r['status']=='UNKNOWN' for r in jobs.values()),'durable completion/commit counters differ')
    used={str(j) for c in commits.values() for j in c['job_ids']}
    require(all(('committed_update' in r)==(k in used) for k,r in jobs.items()),'unregistered committed row')
    return jobs,commits

def _state_from_registry(registry,suite):
    entry=registry['campaign'];require(entry['campaign_sha256']==gate.sha(suite['_suite_path']) and entry['campaign_id']==suite['campaign_id'],'registry suite binding differs')
    require(entry.get('cap')==CAP and registry.get('maximum_total_shifts')==suite['maximum_total_if_approved'] and
        registry.get('maximum_total_shifts')==registry.get('preceding_reserved_shifts')+CAP and
        entry.get('approval_identity_sha256')==gate.digest(entry.get('approval')),'registry approved capacity or approval identity differs')
    require(entry.get('suite_reference')==gate.reference(suite['_suite_path']) and
        entry.get('campaign_identity_sha256')==gate.digest({k:v for k,v in suite.items() if k not in ('jobs','_suite_path')}),
        'registry immutable compact campaign identity differs')
    check_approval(entry['approval'],suite,entry['campaign_sha256'])
    with _connect(registry['state_database'],readonly=True) as con:
        meta=_checked(*con.execute('SELECT record,checksum FROM metadata WHERE name="registration"').fetchone(),'metadata')
        require(meta==entry,'SQLite registration binding differs')
        jobs,commits=_read_rows(con,suite)
    campaign=dict(entry,campaign=suite,commits=commits)
    return dict(format=FORMAT,campaigns={suite['campaign_id']:campaign},jobs=jobs,
        maximum_total_shifts=registry['maximum_total_shifts'],additional_reserved_shifts=registry['maximum_total_shifts'],
        curriculum_reserved_shifts=CAP,active_h2_reserved_shifts=0,legacy_reserved_shifts=0)

def read_campaign_state(suite_path):
    """Status-only read, no directory/DB/sidecar creation or history preflight."""
    suite=gate.load_suite(suite_path);path=Path(suite['budget_path'])
    return dict(campaigns={},jobs={}) if not path.is_file() else _state_from_registry(_registry(path),suite)

def budget_snapshot(path=None,*,test_only=False):
    source,historical_state=_sources();result=_thaw(source);path=GLOBAL_PATH if path is None else Path(path)
    if path.exists():
        reg=_registry(path);require(reg['test_only'] is test_only,'budget test/production scope differs')
        require(reg['preceding_state']==historical_state and reg['preceding_reserved_shifts']==source['maximum_total_shifts'],
            'registered historical budget references/capacity changed')
        suite=gate.load_suite(reg['campaign']['suite_reference']['path'])
        state=_state_from_registry(reg,suite)
        require(set(result['jobs']).isdisjoint(state['jobs']) and set(result['campaigns']).isdisjoint(state['campaigns']),'curriculum/history identity collision')
        result['jobs'].update(state['jobs']);result['campaigns'].update(state['campaigns'])
        for key in ('maximum_total_shifts','additional_reserved_shifts','curriculum_reserved_shifts'):result[key]=state[key]
    return result

class BudgetLedger:
    """Durable atomic slot transitions; transactions contain one slot or one32 batch."""
    def __init__(self,path,suite,*,campaign_sha256,approval,approval_sha256,test_only=False):
        _check_jobs(suite);check_approval(approval,suite,campaign_sha256);gate.check_sources(suite['source_pins'])
        self.path=Path(path).resolve();self.test_only=test_only;self.suite=suite;self.campaign_id=suite['campaign_id']
        require(self.path!=GLOBAL_PATH.resolve() if test_only else self.path==GLOBAL_PATH.resolve(),'test/production ledger path differs')
        require(gate.sha(suite['_suite_path'])==campaign_sha256,'suite bytes changed before registration')
        self._by_run={str(j['run_index']):j for j in suite['jobs']};self._specs={s['stage_id']+'_'+s['arm_phase_id']:s for s in gate.phase_specs()}
        self._batches={update:[] for update in range(1,TRAIN_UPDATES+1)}
        for job in suite['jobs']:
            if job['role']=='train':self._batches[job['target_training_update']].append(job)
        require(all(len(rows)==32 for rows in self._batches.values()),'frozen train batch inventory differs')
        source,state=_sources();require(all(suite['budget_state_before'].get(k)==v for k,v in state.items()),'preceding historical snapshot differs')
        db=(self.path.parent/(self.path.stem+'.sqlite')) if test_only else Path(suite['control_directory'])/'budget.sqlite'
        suite_ref=gate.reference(suite['_suite_path'])
        entry=dict(campaign_id=self.campaign_id,campaign_sha256=campaign_sha256,suite_reference=suite_ref,
            campaign_identity_sha256=gate.digest({k:v for k,v in suite.items() if k not in ('jobs','_suite_path')}),
            approval=approval,approval_sha256=approval_sha256,approval_identity_sha256=gate.digest(approval),cap=CAP)
        with _locked(self.path.with_name(self.path.name+'.lock')):
            if self.path.exists():
                reg=_registry(self.path);require(reg['campaign']==entry and reg['test_only'] is test_only,'existing registration differs')
                self.db=Path(reg['state_database']);self._registration=entry;self.resume();return
            require(not db.exists(),'orphan state database requires review; no automatic reinitialization')
            require(suite['budget_state_before'].get('curriculum_budget_before') is None and
                suite['maximum_total_if_approved']==source['maximum_total_shifts']+CAP,'exact6844 new capacity differs')
            check_collisions(suite,source)
            db.parent.mkdir(parents=True,exist_ok=True);con=sqlite3.connect(db)
            try:
                con.execute('PRAGMA synchronous=FULL')
                con.executescript('CREATE TABLE metadata(name TEXT PRIMARY KEY, record TEXT NOT NULL, checksum TEXT NOT NULL); CREATE TABLE slots(run_index INTEGER PRIMARY KEY, record TEXT NOT NULL, checksum TEXT NOT NULL); CREATE TABLE commits(update_no INTEGER PRIMARY KEY, record TEXT NOT NULL, checksum TEXT NOT NULL);')
                con.execute('INSERT INTO metadata VALUES(?,?,?)',('registration',_dump(entry),gate.digest(entry)))
                head=dict(claimed_slots=0,complete=0,unknown=0,commits=0,revision=0)
                con.execute('INSERT INTO metadata VALUES(?,?,?)',('head',_dump(head),gate.digest(head)));con.commit()
            finally:con.close()
            reg=dict(format=FORMAT,test_only=test_only,campaign=entry,state_database=str(db),preceding_state=state,
                preceding_reserved_shifts=source['maximum_total_shifts'],maximum_total_shifts=source['maximum_total_shifts']+CAP)
            _atomic_json(self.path,reg)
            self.db=db;self._registration=entry
        self.resume()

    @contextmanager
    def _transaction(self):
        require(_registry(self.path)['campaign']==self._registration,'registry changed during execution')
        con=_connect(self.db)
        try:
            con.execute('BEGIN IMMEDIATE');yield con;con.commit()
        except BaseException:con.rollback();raise
        finally:con.close()

    def _job(self,job):
        require(isinstance(job,dict) and self._by_run.get(str(job.get('run_index')))==job,'job differs from frozen plan')
        return str(job['run_index'])

    def _row(self,con,key):
        row=con.execute('SELECT record,checksum FROM slots WHERE run_index=?',(int(key),)).fetchone()
        return None if row is None else _checked(row['record'],row['checksum'],'slot')

    def _put(self,con,key,row):
        con.execute('INSERT OR REPLACE INTO slots VALUES(?,?,?)',(int(key),_dump(row),gate.digest(row)))

    def snapshot(self):return _state_from_registry(_registry(self.path),self.suite)
    def resume(self):return self.snapshot()

    @contextmanager
    def _read_transaction(self):
        """One consistent, read-only SQLite view bound to the frozen registry."""
        registry=_registry(self.path)
        require(registry['campaign']==self._registration and registry['state_database']==str(self.db) and
            registry['test_only'] is self.test_only and
            gate.sha(self.suite['_suite_path'])==self._registration['campaign_sha256'],
            'budget registration or frozen suite changed during execution')
        con=_connect(self.db,readonly=True)
        try:
            con.execute('BEGIN')
            registered=con.execute('SELECT record,checksum FROM metadata WHERE name="registration"').fetchone()
            require(registered is not None and _checked(registered['record'],registered['checksum'],'metadata')==self._registration,
                'SQLite registration binding differs')
            head=_head(con)
            slots=con.execute('SELECT COUNT(*) AS n, MIN(run_index) AS first, MAX(run_index) AS last FROM slots').fetchone()
            count=head['claimed_slots'];first=self.suite['jobs'][0]['run_index']
            require(slots['n']==count and (not count or (slots['first']==first and slots['last']==first+count-1)) and
                (count or slots['first'] is slots['last'] is None),
                'durable claimed inventory deleted or discontinuous')
            commits=con.execute('SELECT COUNT(*) AS n, MIN(update_no) AS first, MAX(update_no) AS last FROM commits').fetchone()
            require(commits['n']==head['commits'] and head['commits']<=TRAIN_UPDATES and
                (not head['commits'] or (commits['first']==1 and commits['last']==head['commits'])) and
                (head['commits'] or commits['first'] is commits['last'] is None) and
                head['complete']+head['unknown']<=count,
                'durable commit/completion counters differ')
            yield con,head
        finally:con.close()

    def _run_key(self,job_or_run_index):
        if isinstance(job_or_run_index,dict):return self._job(job_or_run_index)
        require(type(job_or_run_index) in (int,str) and str(job_or_run_index).isdigit() and
            str(int(job_or_run_index))==str(job_or_run_index),'run_index must be a canonical positive integer')
        key=str(job_or_run_index)
        require(key in self._by_run,'run_index absent from frozen plan')
        return key

    def _checked_job_row(self,con,key,head):
        job=self._by_run[key]
        row=self._row(con,key)
        if job['ordinal']>=head['claimed_slots']:
            require(row is None,'unclaimed slot unexpectedly present')
            return dict(campaign_id=self.campaign_id,identity=job,identity_sha256=gate.digest(job),status='PLANNED')
        require(row is not None,'claimed slot missing')
        require(row.get('status') in ('CLAIMED','COMPLETE','UNKNOWN') and
            row.get('identity_sha256')==gate.digest(job) and
            row.get('claim_sha256')==gate.digest(row.get('claim')),
            'slot status/identity/claim differs')
        if row['status']=='COMPLETE':
            _check_receipt(job,row['receipt'])
            require(row.get('completion_sha256')==gate.digest(dict(dataset_ref=row['dataset_ref'],receipt=row['receipt'])),
                'completion receipt differs')
        result=dict(campaign_id=self.campaign_id,identity=job,identity_sha256=gate.digest(job))
        result.update(row)
        return result

    def get_job(self,job_or_run_index):
        """Authenticated one-slot status without expanding all6844 planned jobs.

        The phase-boundary ``resume`` remains the exhaustive audit of every row.
        """
        key=self._run_key(job_or_run_index)
        with self._read_transaction() as (con,head):
            return self._checked_job_row(con,key,head)

    def committed_count(self):
        """Authenticated global PPO commit cursor; no full campaign snapshot."""
        with self._read_transaction() as (_,head):return head['commits']

    def get_commit(self,update):
        """Authenticate one committed 32-shift batch and its referenced slots."""
        require(type(update)is int and 1<=update<=TRAIN_UPDATES,'commit update outside frozen plan')
        with self._read_transaction() as (con,head):
            if update>head['commits']:
                require(con.execute('SELECT 1 FROM commits WHERE update_no=?',(update,)).fetchone() is None,
                    'uncommitted update unexpectedly present')
                return None
            stored=con.execute('SELECT record,checksum FROM commits WHERE update_no=?',(update,)).fetchone()
            require(stored is not None,'committed update missing')
            item=_checked(stored['record'],stored['checksum'],'commit')
            expected=self._batches[update]
            require(len(expected)==32 and item.get('update')==update and
                item.get('stage_id')==expected[0]['stage_id'] and
                item.get('arm_update')==expected[0]['arm_update'] and
                item.get('phase_id')==expected[0]['phase_id'] and
                set(item.get('job_ids',()))=={str(j['run_index']) for j in expected} and
                item.get('commit_sha256')==gate.digest({k:v for k,v in item.items() if k!='commit_sha256'}),
                'commit digest/physical batch differs')
            for job in expected:
                key=str(job['run_index'])
                row=self._checked_job_row(con,key,head)
                require(row['status']=='COMPLETE' and row.get('committed_update')==update and
                    item['receipt_sha256'].get(key)==row['completion_sha256'],
                    'committed slot differs')
            return item

    def registration(self):
        return dict(protocol='physical-curriculum-budget-registration-v1',campaign_id=self.campaign_id,
            **{k:self._registration[k] for k in ('campaign_sha256','campaign_identity_sha256','approval_sha256','approval_identity_sha256','cap')},
            preceding_reserved_shifts=_registry(self.path)['preceding_reserved_shifts'],maximum_total_shifts=self.suite['maximum_total_if_approved'])

    def claim(self,job):
        key=self._job(job)
        with self._transaction() as con:
            require(self._row(con,key) is None,'slot claimed/complete/unknown; never automatically resubmit')
            require(_head(con)['claimed_slots']==job['ordinal'],'claim cursor differs; never resubmit missing state')
            if job['ordinal']:
                previous=self._row(con,str(job['run_index']-1));require(previous and previous['status']=='COMPLETE','preceding physical job incomplete')
            commits=con.execute('SELECT COUNT(*) FROM commits').fetchone()[0]
            required=self._specs[job['phase_id']]['required_committed_updates']
            require(commits==(required-1 if job['role']=='train' else required),'phase optimizer barrier differs')
            claim=dict(token=uuid.uuid4().hex,claimed_at=legacy._now(),campaign_id=self.campaign_id,identity_sha256=gate.digest(job))
            row=dict(status='CLAIMED',identity_sha256=gate.digest(job),claim=claim,claim_sha256=gate.digest(claim));self._put(con,key,row)
            _advance(con,claimed_slots=1)
        return dict(campaign_id=self.campaign_id,identity=job,**row)

    def complete(self,job,dataset_ref,receipt):
        key=self._job(job);_check_receipt(job,receipt);legacy._reference(dataset_ref)
        with self._transaction() as con:
            row=self._row(con,key);require(row and row['status']=='CLAIMED','only claimed slot can complete')
            row.update(status='COMPLETE',dataset_ref=dataset_ref,receipt=receipt,
                completion_sha256=gate.digest(dict(dataset_ref=dataset_ref,receipt=receipt)))
            self._put(con,key,row)
            _advance(con,complete=1)
        return dict(campaign_id=self.campaign_id,identity=job,**row)

    def unknown(self,job,error):
        key=self._job(job)
        with self._transaction() as con:
            row=self._row(con,key);require(row and row['status']=='CLAIMED','only claimed slot becomes UNKNOWN')
            row.update(status='UNKNOWN',error=str(error),unknown_at=legacy._now());self._put(con,key,row)
            _advance(con,unknown=1)
        return row

    def commit(self,update,jobs,checkpoint_ref,batch_id):
        require(type(update)is int and 1<=update<=TRAIN_UPDATES and isinstance(batch_id,str) and batch_id,'exact next train update required')
        expected=self._batches[update]
        require(len(jobs)==32 and jobs==expected,'entire ordered32 train phase required; evaluation excluded')
        legacy._reference(checkpoint_ref)
        with self._transaction() as con:
            require(con.execute('SELECT COUNT(*) FROM commits').fetchone()[0]+1==update,'commit must be next uncommitted update')
            existing=[json.loads(r[0]) for r in con.execute('SELECT record FROM commits')]
            require(all(c['batch_id']!=batch_id for c in existing),'batch already committed')
            rows={self._job(j):self._row(con,self._job(j)) for j in jobs}
            require(all(r and r['status']=='COMPLETE' and 'committed_update' not in r for r in rows.values()),'batch incomplete/previously used')
            record=dict(update=update,stage_id=jobs[0]['stage_id'],arm_update=jobs[0]['arm_update'],phase_id=jobs[0]['phase_id'],
                job_ids=sorted(rows),checkpoint_ref=checkpoint_ref,batch_id=batch_id,receipt_sha256={k:r['completion_sha256'] for k,r in rows.items()},committed_at=legacy._now())
            record['commit_sha256']=gate.digest(record)
            con.execute('INSERT INTO commits VALUES(?,?,?)',(update,_dump(record),gate.digest(record)))
            for key,row in rows.items():row['committed_update']=update;self._put(con,key,row)
            _advance(con,commits=1)
        return record

    @classmethod
    def rebind_zero_action_campaign(cls,*args,**kwargs):raise ValueError('UNKNOWN/CLAIMED requires explicit reconciliation')

def _check_receipt(job,receipt):
    require(isinstance(receipt,dict) and receipt.get('normal_completion') is True and receipt.get('source_kind')=='real_anylogic_rpc' and
        not any(receipt.get(k,False) for k in ('truncated','abnormal_interruption','sampler_truncated')),'normal complete real RPC receipt required')
    for key in ('run_index','experiment_seed','role','partition','stage_id','phase_id'):
        require(receipt.get(key)==job.get(key),'completion identity differs: '+key)
    evaluation=job['role']!='train'
    require(receipt.get('evaluation_only',False) is evaluation and receipt.get('ppo_trainable',True) is (not evaluation),
        'evaluation/training receipt flags differ')
