"""BG128-only SQLite slots; every preceding ledger remains read-only.

The state machine is kept compatible with the checked 008 campaign API.  The
contract, compact suite reader and production database are all local to BG009.
"""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sqlite3
from .plan_io import ROOT, encoded, sha, reference, load_suite
from .contract import require

GLOBAL_PATH = ROOT / 'runs/business-graph-stability-budget-v1.sqlite'

def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()

def preceding_snapshot():
    # Latest entry includes BG009 training and its 100-shift fixed evaluation.
    from business_graph_summary_evaluation_v1.budget import budget_snapshot
    return budget_snapshot()

def preceding_reference():
    from business_graph_summary_evaluation_v1.budget import preceding_reference as inherited, GLOBAL_PATH as eval_path
    state = inherited()
    state['bg128_independent_extension_before'] = reference(eval_path) if eval_path.is_file() else None
    return state

def check_collisions(suite, snapshot):
    from physical_curriculum_v1.budget import _history_rows
    require(suite['campaign_id'] not in snapshot['campaigns'], 'campaign identity already occupied')
    rows = _history_rows(snapshot)
    for key in ('run_index', 'recording_id', 'source_recording_id'):
        previous = {row.get(key) for row in rows} - {None}
        require(all(j[key] not in previous for j in suite['jobs']), 'historical collision: ' + key)
    old_seeds = {r.get(k) for r in rows for k in ('experiment_seed', 'actor_sampling_seed', 'actor_seed')} - {None}
    if 'pairing' in suite:
        from .pairing import check_seed_pairing
        check_seed_pairing(suite)
    else:
        require(all(j[k] not in old_seeds for j in suite['jobs'] for k in
                    ('experiment_seed', 'actor_sampling_seed', 'actor_seed')), 'historical/cross-type seed collision')

def _payload(row):
    value = json.loads(row[0])
    require(digest(value) == row[1], 'SQLite payload checksum differs')
    return value

@contextmanager
def connection(path, *, readonly=False):
    path = Path(path).resolve()
    con = sqlite3.connect(f'file:{path}?mode=ro' if readonly else str(path), uri=readonly, timeout=30)
    try:
        if not readonly:
            con.execute('PRAGMA synchronous=FULL')
            con.execute('BEGIN IMMEDIATE')
        yield con
        if not readonly:
            con.commit()
    except BaseException:
        if not readonly:
            con.rollback()
        raise
    finally:
        con.close()

def _state(con, suite, entry, hashes=None):
    cid = suite['campaign_id']
    expected = {j['run_index']: j for j in suite['jobs']}
    jobs = {}
    for run, status, text, checksum in con.execute('SELECT run,status,payload,checksum FROM slots WHERE campaign=?', (cid,)):
        record = _payload((text, checksum))
        require(run in expected and record['identity_sha256'] == (hashes[run] if hashes is not None else digest(expected[run]))
                and status == record['status'] and status in ('PLANNED', 'CLAIMED', 'COMPLETE', 'UNKNOWN'),
                'slot status or canonical identity differs')
        if status=='COMPLETE':
            require(record['completion_sha256']==digest(dict(dataset_ref=record['dataset_ref'],receipt=record['receipt']))
                    and record['receipt']['run_index']==run
                    and record['receipt']['experiment_seed']==expected[run]['experiment_seed'],
                    'completion receipt changed')
        jobs[str(run)] = dict(record, campaign_id=cid, identity=expected[run])
    require(len(jobs) == len(expected), 'incomplete budget inventory')
    commits = {}
    for update, text, checksum in con.execute('SELECT update_id,payload,checksum FROM commits WHERE campaign=?', (cid,)):
        commit = _payload((text, checksum))
        require(update == commit['update'], 'commit update identity differs')
        for run in commit['job_ids']:
            require(jobs[run]['status'] == 'COMPLETE' and jobs[run].get('committed_update') == update,
                    'commit contains incomplete/rebound slot')
            require(jobs[run]['identity']['role']=='train' and jobs[run]['identity']['target_training_update']==update,
                    'committed training phase changed')
        commits[str(update)] = commit
    require(sorted(map(int, commits)) == list(range(1, len(commits)+1)), 'noncontiguous commits')
    return dict(campaigns={cid: dict(entry, campaign=suite, commits=commits)}, jobs=jobs)

def budget_snapshot(path=None):
    result = preceding_snapshot()
    path = GLOBAL_PATH if path is None else Path(path)
    if not path.exists():
        return result
    with connection(path, readonly=True) as con:
        for row in con.execute('SELECT payload,checksum FROM campaigns'):
            entry = _payload(row)
            require(reference(entry['suite_reference']['path']) == entry['suite_reference'], 'registered suite bytes changed')
            suite, _ = load_suite(entry['suite_reference']['path'])
            require(suite['budget_state_before']==preceding_reference(), 'registered preceding ledger bytes changed')
            state = _state(con, suite, entry)
            require(set(result['jobs']).isdisjoint(state['jobs']) and
                    set(result['campaigns']).isdisjoint(state['campaigns']), 'registry/history collision')
            result['jobs'].update(state['jobs']); result['campaigns'].update(state['campaigns'])
            result['maximum_total_shifts'] += entry['cap']
            result['additional_reserved_shifts'] += entry['cap']
    return result

class BudgetLedger:
    def __init__(self, path, suite, *, campaign_sha256, approval, approval_sha256, test_only=False):
        from .gate import check_approval, check_jobs
        check_jobs(suite)
        check_approval(approval, suite, campaign_sha256)  # BEFORE any mkdir/database
        self.path = Path(path).resolve(); self.suite = suite; self.campaign_id = suite['campaign_id']
        require(self.path != GLOBAL_PATH.resolve() if test_only else self.path == GLOBAL_PATH.resolve(),
                'test/production budget location differs')
        self.jobs = {j['run_index']: j for j in suite['jobs']}
        self.hashes = {run:digest(job) for run,job in self.jobs.items()}
        self.update_jobs = {}
        for job in suite['jobs']:
            if job['role'] == 'train':
                self.update_jobs.setdefault(job['target_training_update'], []).append(job)
        entry = dict(suite_reference=dict(path=suite['suite_path'], sha256=campaign_sha256),
                     campaign_sha256=campaign_sha256, approval=approval, approval_sha256=approval_sha256,
                     cap=suite['new_shift_cap'], test_only=test_only)
        require(reference(suite['suite_path']) == entry['suite_reference'], 'registration suite bytes differ')
        # Full merged read/collision check before registration; never allocate an
        # old H2 reservation or omit evaluation campaigns.
        before = budget_snapshot(self.path)
        if self.campaign_id not in before['campaigns']:
            check_collisions(suite, before)
            require(before['maximum_total_shifts'] + entry['cap'] == suite['maximum_total_if_approved'],
                    'capacity changed; review new exact budget before registration')
            require(preceding_reference() == suite['budget_state_before'], 'preceding budget source changed')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with connection(self.path) as con:
            con.execute('CREATE TABLE IF NOT EXISTS campaigns (id TEXT PRIMARY KEY,payload TEXT,checksum TEXT)')
            con.execute('CREATE TABLE IF NOT EXISTS slots (run INTEGER PRIMARY KEY,campaign TEXT,status TEXT,payload TEXT,checksum TEXT)')
            con.execute('CREATE TABLE IF NOT EXISTS commits (campaign TEXT,update_id INTEGER,payload TEXT,checksum TEXT,PRIMARY KEY(campaign,update_id))')
            old = con.execute('SELECT payload,checksum FROM campaigns WHERE id=?', (self.campaign_id,)).fetchone()
            if old:
                require(_payload(old) == entry, 'existing registration or approval differs')
            else:
                registered = [_payload(r) for r in con.execute('SELECT payload,checksum FROM campaigns')]
                require(preceding_snapshot()['maximum_total_shifts'] + sum(r['cap'] for r in registered) + entry['cap']
                        == suite['maximum_total_if_approved'], 'concurrent capacity registration changed')
                require(preceding_reference() == suite['budget_state_before'], 'preceding budget changed during registration')
                check_collisions(suite, preceding_snapshot())
                # Check identities/seeds again under the writer lock.
                for previous in registered:
                    other, _ = load_suite(previous['suite_reference']['path'])
                    check_collisions(suite, dict(campaigns={other['campaign_id']: previous},
                        jobs={str(j['run_index']): dict(identity=j) for j in other['jobs']}))
                con.execute('INSERT INTO campaigns VALUES (?,?,?)', (self.campaign_id, encoded(entry).decode(), digest(entry)))
                for run, job in self.jobs.items():
                    record = dict(status='PLANNED', identity_sha256=digest(job))
                    con.execute('INSERT INTO slots VALUES (?,?,?,?,?)', (run, self.campaign_id, 'PLANNED', encoded(record).decode(), digest(record)))
        self.entry = entry

    def snapshot(self):
        with connection(self.path, readonly=True) as con:
            row = con.execute('SELECT payload,checksum FROM campaigns WHERE id=?', (self.campaign_id,)).fetchone()
            require(row is not None and _payload(row) == self.entry, 'registration changed')
            return _state(con, self.suite, self.entry,self.hashes)

    def job(self, job):
        """Validate one canonical slot without scanning the campaign inventory.

        This is a read-only status accessor. Full snapshot/resume validation is
        retained for campaign reconciliation and checkpoint admission.
        """
        run = job['run_index']
        require(self.jobs.get(run) == job, 'noncanonical budget job')
        with connection(self.path, readonly=True) as con:
            row = con.execute('SELECT status,payload,checksum FROM slots WHERE run=? AND campaign=?',
                              (run, self.campaign_id)).fetchone()
            require(row is not None, 'missing budget slot')
            record = _payload((row[1], row[2]))
            require(record['identity_sha256'] == self.hashes[run] and row[0] == record['status']
                    and record['status'] in ('PLANNED', 'CLAIMED', 'COMPLETE', 'UNKNOWN'),
                    'slot status or canonical identity differs')
            if record['status'] == 'COMPLETE':
                require(record['completion_sha256'] == digest(dict(dataset_ref=record['dataset_ref'],
                            receipt=record['receipt']))
                        and record['receipt']['run_index'] == run
                        and record['receipt']['experiment_seed'] == job['experiment_seed'],
                        'completion receipt changed')
            return dict(record, campaign_id=self.campaign_id, identity=dict(job))

    def commit_record(self, update):
        """Read one checkpoint binding and its profile-sized batch, independent of cap.

        Checkpoint bytes are verified by the checkpoint loader. This accessor
        checks the registry binding and completion integrity without rehashing
        HDF5 or scanning unrelated training/evaluation slots.
        """
        require(type(update) is int and update in self.update_jobs, 'invalid committed update')
        jobs = self.update_jobs[update]
        require(len(jobs) == self.suite['episodes_per_batch'], 'canonical committed batch differs')
        with connection(self.path, readonly=True) as con:
            row = con.execute('SELECT payload,checksum FROM commits WHERE campaign=? AND update_id=?',
                              (self.campaign_id, update)).fetchone()
            require(row is not None, 'committed update absent')
            record = _payload(row)
            require(record.get('update') == update and
                    record.get('job_ids') == [str(j['run_index']) for j in jobs]
                    and isinstance(record.get('batch_id'), str) and bool(record['batch_id']),
                    'commit canonical update or batch differs')
            ref = record.get('checkpoint_ref')
            require(isinstance(ref, dict) and set(ref) == {'path', 'sha256'}
                    and isinstance(ref['path'], str) and Path(ref['path']).is_absolute()
                    and isinstance(ref['sha256'], str) and len(ref['sha256']) == 64
                    and all(c in '0123456789abcdef' for c in ref['sha256']) and ref['sha256'] != '0'*64,
                    'invalid committed checkpoint reference')
            marks = ','.join('?' for _ in jobs)
            slots = con.execute('SELECT run,status,payload,checksum FROM slots WHERE campaign=? AND run IN ('+
                                marks+')', (self.campaign_id, *(j['run_index'] for j in jobs))).fetchall()
            require(len(slots) == self.suite['episodes_per_batch'], 'committed batch slots absent')
            for run, status, text, checksum in slots:
                slot = _payload((text, checksum))
                require(status == slot['status'] == 'COMPLETE' and slot.get('committed_update') == update
                        and slot['identity_sha256'] == self.hashes[run],
                        'commit contains incomplete or rebound slot')
                require(slot['completion_sha256'] == digest(dict(dataset_ref=slot['dataset_ref'], receipt=slot['receipt']))
                        and slot['receipt']['run_index'] == run
                        and slot['receipt']['experiment_seed'] == self.jobs[run]['experiment_seed'],
                        'committed completion receipt changed')
            return record

    def resume(self):
        state = self.snapshot()
        require(all(r['status'] not in ('UNKNOWN', 'CLAIMED') for r in state['jobs'].values()),
                'UNKNOWN/CLAIMED preserved; reconcile without resampling')
        return state

    def _row(self, con, job):
        require(self.jobs.get(job['run_index']) == job, 'noncanonical budget job')
        row = con.execute('SELECT payload,checksum FROM slots WHERE run=? AND campaign=?',
                          (job['run_index'], self.campaign_id)).fetchone()
        require(row is not None, 'missing budget slot')
        record = _payload(row)
        require(record['identity_sha256'] == digest(job), 'slot identity changed')
        return record

    def _put(self, con, job, record):
        con.execute('UPDATE slots SET status=?,payload=?,checksum=? WHERE run=? AND campaign=?',
            (record['status'], encoded(record).decode(), digest(record), job['run_index'], self.campaign_id))

    def claim(self, job):
        with connection(self.path) as con:
            record = self._row(con, job)
            require(record['status'] == 'PLANNED', 'slot cannot be claimed twice')
            record.update(status='CLAIMED'); self._put(con, job, record)

    def complete(self, job, dataset_ref, receipt):
        require(reference(dataset_ref['path']) == dataset_ref and receipt.get('run_index') == job['run_index']
                and receipt.get('experiment_seed') == job['experiment_seed'] and receipt.get('normal_completion') is True,
                'completion source/identity invalid')
        with connection(self.path) as con:
            record = self._row(con, job)
            require(record['status'] == 'CLAIMED', 'completion needs exactly one claim')
            record.update(status='COMPLETE', dataset_ref=dataset_ref, receipt=receipt,
                completion_sha256=digest(dict(dataset_ref=dataset_ref, receipt=receipt)))
            self._put(con, job, record)

    def unknown(self, job, error):
        with connection(self.path) as con:
            record = self._row(con, job)
            require(record['status'] == 'CLAIMED' and bool(error), 'UNKNOWN needs claimed slot and reason')
            record.update(status='UNKNOWN', error=str(error)); self._put(con, job, record)

    def commit(self, update, jobs, checkpoint_ref, batch_id):
        require(reference(checkpoint_ref['path']) == checkpoint_ref and bool(batch_id), 'invalid checkpoint/batch')
        expected = self.update_jobs.get(update, [])
        require(jobs == expected and len(jobs)==self.suite['episodes_per_batch'], 'commit must contain one complete frozen batch')
        with connection(self.path) as con:
            count = con.execute('SELECT count(*) FROM commits WHERE campaign=?', (self.campaign_id,)).fetchone()[0]
            require(update == count+1, 'update already committed or out of order')
            require(all(_payload(row)['batch_id']!=batch_id for row in con.execute('SELECT payload,checksum FROM commits')),
                    'batch already committed')
            records = [self._row(con, job) for job in jobs]
            require(all(r['status']=='COMPLETE' and 'committed_update' not in r for r in records), 'batch slots not fresh and complete')
            for job, record in zip(jobs, records):
                require(reference(record['dataset_ref']['path']) == record['dataset_ref'], 'batch source changed')
                record['committed_update']=update; self._put(con, job, record)
            value = dict(update=update, job_ids=[str(j['run_index']) for j in jobs], checkpoint_ref=checkpoint_ref, batch_id=batch_id)
            con.execute('INSERT INTO commits VALUES (?,?,?,?)', (self.campaign_id, update, encoded(value).decode(), digest(value)))
