"""Fixed-checkpoint BG128 extension of the independent-evaluation ledger.

The original independent-evaluation JSON is an immutable, SHA-pinned prefix of
later ledgers.  This SQLite sidecar adds independent slots without rewriting
that prefix.  ``budget_snapshot`` includes every old ledger through BG009 and
this extension.  Evaluation can never commit a PPO update.
"""
from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path
import sqlite3

from business_graph_stability_campaign_v1 import budget as previous
from business_graph_stability_campaign_v1.plan_io import ROOT, encoded, reference
from business_graph_stability_v1.contract import require

GLOBAL_PATH = ROOT / 'runs/independent-evaluation-budget-v1.bg128-stability.sqlite'
FORMAT = 'independent-evaluation-budget-bg128-extension-v1'
SUITE_CONTRACT = 'business-graph-stability-independent-evaluation-v1'
CAP = 100


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def preceding_snapshot():
    """Read all prior training, fixed-policy and sampling ledgers."""
    return previous.budget_snapshot()


def preceding_reference():
    state = previous.preceding_reference()
    state['stability_training_budget_before'] = reference(previous.GLOBAL_PATH)
    state['legacy_independent_evaluation_budget'] = reference(
        ROOT / 'runs/independent-evaluation-budget-v1.json')
    state['sampling_comparison_budget'] = reference(
        ROOT / 'runs/sampling-comparison-budget-v1.json')
    return state


def _scope(suite):
    require(suite.get('contract') == SUITE_CONTRACT and
            suite.get('kind') == 'independent_evaluation' and
            suite.get('new_shift_cap') == suite.get('evaluation_shifts') == CAP and
            len(suite.get('jobs', [])) == CAP and
            suite.get('training_shifts') == suite.get('training_updates') ==
            suite.get('optimizer_steps') == 0,
            'exact100 fixed-checkpoint slots and zero training required')
    update, actor_seed = suite.get('checkpoint_update'), suite.get('actor_seed')
    require(type(update) is int and update > 0 and type(actor_seed) is int and actor_seed > 0,
            'fixed parent checkpoint update and initialization identity required')
    jobs = suite['jobs']
    require(all(job.get('role') == 'final' and job.get('partition') == 'validation' and
                job.get('target_training_update') == 0 and
                job.get('checkpoint_update') == update and
                job.get('actor_seed') == actor_seed for job in jobs),
            'evaluation slots cannot change checkpoint or request optimizer updates')
    require(len({j['run_index'] for j in jobs}) == CAP,
            'independent physical run identity repeated')
    streams = [set(j[key] for j in jobs) for key in
               ('experiment_seed', 'actor_sampling_seed')]
    require(all(len(values) == CAP and all(type(v) is int and v > 0 for v in values)
                for values in streams) and streams[0].isdisjoint(streams[1]) and
            actor_seed not in streams[0] | streams[1],
            'independent environment and action seed streams required')


def check_collisions(suite, snapshot):
    """Only the explicitly paired corresponding historical seed rows may repeat."""
    _scope(suite)
    from physical_curriculum_v1.budget import _history_rows
    require(suite['campaign_id'] not in snapshot['campaigns'],
            'campaign identity already occupied')
    rows = _history_rows(snapshot)
    for key in ('run_index', 'recording_id', 'source_recording_id'):
        old = {row.get(key) for row in rows} - {None}
        require(all(job[key] not in old for job in suite['jobs']),
                'historical identity collision: ' + key)
    from business_graph_stability_campaign_v1.pairing import check_seed_pairing
    check_seed_pairing(suite,evaluation=True)


@contextmanager
def connection(path, *, readonly=False):
    path = Path(path).resolve()
    con = sqlite3.connect(f'file:{path}?mode=ro' if readonly else str(path),
                          uri=readonly, timeout=30)
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


def _payload(row):
    value = json.loads(row[0])
    require(digest(value) == row[1], 'SQLite evaluation payload checksum differs')
    return value


def _suite(entry):
    ref = entry['suite_reference']
    require(reference(ref['path']) == ref, 'registered evaluation suite bytes changed')
    suite = json.loads(Path(ref['path']).read_bytes())
    _scope(suite)
    require(suite.get('suite_path') == ref['path'] and entry['cap'] == CAP and
            entry['campaign_sha256'] == ref['sha256'] and
            entry['approval_identity_sha256'] == digest(entry['approval']),
            'registered independent evaluation capacity differs')
    from .gate import check_approval, check_jobs
    check_jobs(suite)
    check_approval(entry['approval'], suite, entry['campaign_sha256'])
    return suite


def _metadata(con, test_only):
    rows = dict(con.execute('SELECT key,value FROM metadata'))
    require(rows == dict(format=FORMAT, test_only=json.dumps(test_only)),
            'independent evaluation database scope differs')


def _state(con, suite, entry):
    jobs, expected = {}, {job['run_index']: job for job in suite['jobs']}
    for run, status, text, checksum in con.execute(
            'SELECT run,status,payload,checksum FROM slots WHERE campaign=?',
            (suite['campaign_id'],)):
        row = _payload((text, checksum))
        require(run in expected and row['identity_sha256'] == digest(expected[run]) and
                row['status'] == status and status in ('PLANNED', 'CLAIMED', 'COMPLETE', 'UNKNOWN') and
                'committed_update' not in row,
                'evaluation slot identity, state or zero-update contract differs')
        if status == 'COMPLETE':
            require(row['completion_sha256'] == digest(dict(dataset_ref=row['dataset_ref'],
                        receipt=row['receipt'])) and
                    row['receipt']['run_index'] == run and
                    row['receipt']['experiment_seed'] == expected[run]['experiment_seed'] and
                    row['receipt']['normal_completion'] is True,
                    'evaluation completion receipt changed')
        jobs[str(run)] = dict(row, campaign_id=suite['campaign_id'], identity=expected[run])
    require(set(map(int, jobs)) == set(expected), 'evaluation slot inventory incomplete')
    return dict(campaigns={suite['campaign_id']: dict(entry, campaign=suite, commits={})}, jobs=jobs)


def budget_snapshot(path=None, *, test_only=False):
    """Read-only merged inventory; inspection never creates a ledger or lock."""
    result = copy.deepcopy(preceding_snapshot())
    path = GLOBAL_PATH if path is None else Path(path)
    if not path.is_file():
        return result
    cap = 0
    with connection(path, readonly=True) as con:
        _metadata(con, test_only)
        for cid, text, checksum in con.execute('SELECT id,payload,checksum FROM campaigns'):
            entry = _payload((text, checksum))
            require(entry['test_only'] is test_only, 'registered evaluation scope differs')
            suite = _suite(entry)
            require(cid == suite['campaign_id'], 'evaluation campaign table identity differs')
            state = _state(con, suite, entry)
            require(set(result['jobs']).isdisjoint(state['jobs']) and
                    set(result['campaigns']).isdisjoint(state['campaigns']),
                    'evaluation extension/history identity collision')
            result['jobs'].update(state['jobs'])
            result['campaigns'].update(state['campaigns'])
            cap += entry['cap']
        require(con.execute('SELECT count(*) FROM slots').fetchone()[0] == cap,
                'unregistered evaluation slots found')
    result['independent_bg128_reserved_shifts'] = cap
    result['maximum_total_shifts'] += cap
    result['additional_reserved_shifts'] += cap
    return result


class BudgetLedger:
    def __init__(self, path, suite, *, campaign_sha256, approval,
                 approval_sha256, test_only=False):
        from .gate import check_approval, check_jobs, check_sources
        _scope(suite)
        check_jobs(suite)
        check_sources(suite['source_pins'])
        check_approval(approval, suite, campaign_sha256)
        require(type(test_only) is bool, 'explicit boolean test_only required')
        self.path = Path(path).resolve()
        require(self.path != GLOBAL_PATH.resolve() if test_only else
                self.path == GLOBAL_PATH.resolve(), 'production/test evaluation ledger location differs')
        self.suite, self.campaign_id = suite, suite['campaign_id']
        self.jobs = {job['run_index']: job for job in suite['jobs']}
        self.test_only = test_only
        require(reference(suite['suite_path'])['sha256'] == campaign_sha256,
                'registration suite bytes differ')
        require(isinstance(approval_sha256, str) and len(approval_sha256) == 64 and
                all(c in '0123456789abcdef' for c in approval_sha256),
                'approval byte SHA required')
        entry = dict(suite_reference=dict(path=suite['suite_path'], sha256=campaign_sha256),
                     campaign_sha256=campaign_sha256, approval=copy.deepcopy(approval),
                     approval_sha256=approval_sha256, approval_identity_sha256=digest(approval),
                     cap=CAP, test_only=test_only)
        before = budget_snapshot(self.path, test_only=test_only)
        if self.campaign_id not in before['campaigns']:
            check_collisions(suite, before)
            require(before['maximum_total_shifts'] + CAP == suite['maximum_total_if_approved'] and
                    suite['budget_state_before'] == preceding_reference() and
                    suite.get('evaluation_budget_before') ==
                    (reference(self.path) if self.path.exists() else None),
                    'merged budget or independent extension changed since freeze')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with connection(self.path) as con:
            con.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY,value TEXT)')
            if not con.execute('SELECT count(*) FROM metadata').fetchone()[0]:
                con.executemany('INSERT INTO metadata VALUES (?,?)',
                                [('format', FORMAT), ('test_only', json.dumps(test_only))])
            _metadata(con, test_only)
            con.execute('CREATE TABLE IF NOT EXISTS campaigns (id TEXT PRIMARY KEY,payload TEXT,checksum TEXT)')
            con.execute('CREATE TABLE IF NOT EXISTS slots (run INTEGER PRIMARY KEY,campaign TEXT,status TEXT,payload TEXT,checksum TEXT)')
            old = con.execute('SELECT payload,checksum FROM campaigns WHERE id=?',
                              (self.campaign_id,)).fetchone()
            if old is not None:
                require(_payload(old) == entry, 'registered evaluation approval or suite differs')
            else:
                require(suite['budget_state_before'] == preceding_reference(),
                        'preceding merged budget changed during registration')
                source = preceding_snapshot()
                registered = [_payload(row) for row in con.execute('SELECT payload,checksum FROM campaigns')]
                require(source['maximum_total_shifts'] + sum(row['cap'] for row in registered) + CAP ==
                        suite['maximum_total_if_approved'], 'concurrent evaluation capacity registration changed')
                check_collisions(suite, source)
                for previous_entry in registered:
                    other = _suite(previous_entry)
                    check_collisions(suite, dict(campaigns={other['campaign_id']: previous_entry},
                        jobs={str(job['run_index']): dict(identity=job) for job in other['jobs']}))
                con.execute('INSERT INTO campaigns VALUES (?,?,?)',
                            (self.campaign_id, encoded(entry).decode(), digest(entry)))
                for job in suite['jobs']:
                    row = dict(status='PLANNED', identity_sha256=digest(job))
                    con.execute('INSERT INTO slots VALUES (?,?,?,?,?)',
                        (job['run_index'], self.campaign_id, 'PLANNED', encoded(row).decode(), digest(row)))
        self.entry = entry

    def snapshot(self):
        with connection(self.path, readonly=True) as con:
            _metadata(con, self.test_only)
            row = con.execute('SELECT payload,checksum FROM campaigns WHERE id=?',
                              (self.campaign_id,)).fetchone()
            require(row is not None and _payload(row) == self.entry, 'evaluation registration changed')
            return _state(con, self.suite, self.entry)

    def resume(self):
        state = self.snapshot()
        require(all(row['status'] not in ('UNKNOWN', 'CLAIMED') for row in state['jobs'].values()),
                'UNKNOWN/CLAIMED preserved; reconcile without resampling')
        return state

    def _row(self, con, job):
        require(self.jobs.get(job['run_index']) == job, 'noncanonical evaluation job')
        raw = con.execute('SELECT status,payload,checksum FROM slots WHERE run=? AND campaign=?',
                          (job['run_index'], self.campaign_id)).fetchone()
        require(raw is not None, 'missing evaluation slot')
        row = _payload(raw[1:])
        require(row['identity_sha256'] == digest(job) and raw[0] == row['status'] and
                row['status'] in ('PLANNED', 'CLAIMED', 'COMPLETE', 'UNKNOWN') and
                'committed_update' not in row,
                'evaluation slot identity or status changed')
        return row

    def job(self, job):
        with connection(self.path, readonly=True) as con:
            row = self._row(con, job)
        if row['status'] == 'COMPLETE':
            require(row['completion_sha256'] == digest(dict(dataset_ref=row['dataset_ref'],
                        receipt=row['receipt'])) and
                    row['receipt']['run_index'] == job['run_index'] and
                    row['receipt']['experiment_seed'] == job['experiment_seed'] and
                    row['receipt']['normal_completion'] is True,
                    'evaluation completion receipt changed')
        return dict(row, campaign_id=self.campaign_id, identity=copy.deepcopy(job))

    def _put(self, con, job, row):
        con.execute('UPDATE slots SET status=?,payload=?,checksum=? WHERE run=? AND campaign=?',
            (row['status'], encoded(row).decode(), digest(row), job['run_index'], self.campaign_id))

    def claim(self, job):
        with connection(self.path) as con:
            row = self._row(con, job)
            require(row['status'] == 'PLANNED', 'evaluation slot cannot be claimed twice')
            row['status'] = 'CLAIMED'
            self._put(con, job, row)

    def complete(self, job, dataset_ref, receipt):
        require(reference(dataset_ref['path']) == dataset_ref and
                receipt.get('run_index') == job['run_index'] and
                receipt.get('experiment_seed') == job['experiment_seed'] and
                receipt.get('normal_completion') is True,
                'evaluation completion source or identity invalid')
        with connection(self.path) as con:
            row = self._row(con, job)
            require(row['status'] == 'CLAIMED', 'evaluation completion needs exactly one claim')
            row.update(status='COMPLETE', dataset_ref=copy.deepcopy(dataset_ref),
                receipt=copy.deepcopy(receipt),
                completion_sha256=digest(dict(dataset_ref=dataset_ref, receipt=receipt)))
            self._put(con, job, row)

    def unknown(self, job, error):
        with connection(self.path) as con:
            row = self._row(con, job)
            require(row['status'] == 'CLAIMED' and bool(error),
                    'evaluation UNKNOWN needs a claimed slot and reason')
            row.update(status='UNKNOWN', error=str(error))
            self._put(con, job, row)

    def commit(self, *args, **kwargs):
        raise ValueError('fixed-checkpoint independent evaluation cannot commit PPO updates')

    @classmethod
    def rebind_zero_action_campaign(cls, *args, **kwargs):
        raise ValueError('UNKNOWN/CLAIMED independent slots require audited recovery, never automatic rebind')
