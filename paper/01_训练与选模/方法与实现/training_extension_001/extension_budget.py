"""Append one fixed BG128 extension without rewriting the historical ledgers."""

from importlib import import_module
import json
from pathlib import Path
import sys

PLAN = Path(__file__).resolve().parent
sys.path.insert(0, str(PLAN.parents[3]))

from business_graph_summary_campaign_v1.budget import (
    BudgetLedger, _payload, connection, digest,
)
from business_graph_summary_campaign_v1.contract import require
from business_graph_summary_campaign_v1.plan_io import (
    ROOT, encoded, load_phase, load_reference, reference,
)

CAMPAIGN = 'bg128-u16-to-u32-20261119-001'
LEDGER = ROOT / 'runs' / CAMPAIGN / 'extension_ledger.sqlite'
FORMAT = 'bg128-training-extension-budget-v1'
PREVIOUS_CAP = 21799
EXTENSION_CAP = 812
NEW_CAP = PREVIOUS_CAP + EXTENSION_CAP
FIRST_RUN = 9990001
BATCH_SIZE = 32
FIRST_UPDATE = 17
LAST_UPDATE = 32
PHASE_ORDER = (
    [('eval', 16)]
    + [('train', update) for update in range(17, 25)]
    + [('eval', 24)]
    + [('train', update) for update in range(25, 33)]
    + [('eval', 32)]
)
PAIRING_SOURCE = dict(
    path=str(PLAN.parent / 'checkpoint_evaluation_001/prepared_001/suite.json'),
    sha256='c52f980a3930f2d87f5f42b63872b3a2dce28451358bb724c70ec492dd6356d5',
)
PRIOR_DISCOVERY = dict(
    path=str(ROOT / 'runs/bg128-u13-discovery100-20261119-001/independent_discovery_ledger.json'),
    sha256='1aab4e4a863450c5b8fe29f9dec09f58966c70ca95644e25ce3e3ada741f7348',
)
SEED_FIELDS = ('experiment_seed', 'actor_sampling_seed', 'actor_seed')
MAX_JAVA_SEED = (1 << 48) - 1


def preceding_snapshot():
    require(reference(PRIOR_DISCOVERY['path']) == PRIOR_DISCOVERY,
            'completed discovery ledger changed')
    discovery = PLAN.parent / 'first_window_action_001/discovery_3e4_001'
    sys.path.insert(0, str(discovery))
    try:
        return import_module('discovery_budget').budget_snapshot()
    finally:
        sys.path.remove(str(discovery))


def _phase_inventory(suite):
    require(suite.get('schema') == 'bg128-training-extension-v1'
            and suite.get('campaign_id') == CAMPAIGN
            and suite.get('budget_path') == str(LEDGER)
            and suite.get('paired_seed_source') == PAIRING_SOURCE
            and suite.get('new_shift_cap') == EXTENSION_CAP
            and suite.get('training_shifts') == 512 and suite.get('evaluation_shifts') == 300,
            'extension suite scope or production ledger path differs')
    phases = suite['phases']
    require([(row['role'], row['update_index']) for row in phases] == PHASE_ORDER,
            'extension phases must be eval16, train17–24, eval24, train25–32, eval32')
    jobs, configs = [], {}
    for phase in phases:
        ref = phase['phase_config']
        require(reference(ref['path']) == ref, 'frozen extension phase bytes differ')
        cfg = load_phase(ref['path'])
        evaluation = phase['role'] == 'eval'
        update = phase['update_index']
        checkpoint = update if evaluation else update - 1
        role, partition = ('final', 'validation') if evaluation else ('train', 'train')
        require(cfg['phase_id'] == phase['phase_id'] and cfg['campaign_id'] == CAMPAIGN
                and cfg['python_output'] == phase['python_output']
                and phase['checkpoint_update'] == checkpoint
                and phase['phase_id'] not in configs and cfg['jobs']
                and len(cfg['jobs']) == (100 if evaluation else BATCH_SIZE),
                'extension phase inventory or checkpoint differs')
        for job in cfg['jobs']:
            require(job['ordinal'] == len(jobs) and job['run_index'] == FIRST_RUN + len(jobs)
                    and job['phase_id'] == phase['phase_id']
                    and job['python_output'] == phase['python_output']
                    and job['checkpoint_update'] == checkpoint
                    and job['target_training_update'] == (0 if evaluation else update)
                    and job['role'] == role and job['partition'] == partition
                    and job['candidate_id'] == 's01_BG128'
                    and job['recording_id'] == job['source_recording_id'] == f"{CAMPAIGN}-{phase['phase_id']}",
                    'extension job identity, order or role differs')
            jobs.append(job)
        configs[phase['phase_id']] = ref
    require(len(jobs) == EXTENSION_CAP and ('jobs' not in suite or suite['jobs'] == jobs),
            'the expanded job inventory differs from the sealed phases')
    return jobs, configs


def _approval(suite_ref, approval_ref):
    approval = load_reference(approval_ref)
    require(approval.get('schema') == 'bg128-training-extension-approval-v1'
            and approval.get('campaign_id') == CAMPAIGN
            and approval.get('suite_sha256') == suite_ref['sha256']
            and approval.get('approved_by') == 'user' and approval.get('decision') == 'approved'
            and isinstance(approval.get('source_user_instruction'), str)
            and bool(approval['source_user_instruction'])
            and approval.get('approved_new_shifts') == EXTENSION_CAP
            and approval.get('previous_approved_cap') == PREVIOUS_CAP
            and approval.get('new_approved_cap') == NEW_CAP
            and approval.get('new_training_shifts') == 512
            and approval.get('new_evaluation_shifts') == 300
            and approval.get('first_training_update') == FIRST_UPDATE
            and approval.get('last_training_update') == LAST_UPDATE
            and approval.get('paired_evaluation_updates') == [16, 24, 32],
            'the exact 812-shift extension authorization differs')


def check_collisions(suite, snapshot=None):
    """Allow old evaluation seed pairs, and forbid any other new stream collision."""
    from physical_curriculum_v1.budget import _history_rows
    from python.offline_split import java_random_states

    jobs, _ = _phase_inventory(suite)
    previous = preceding_snapshot() if snapshot is None else snapshot
    require(previous['maximum_total_shifts'] == PREVIOUS_CAP
            and previous['active_h2_reserved_shifts'] == 0
            and CAMPAIGN not in previous['campaigns'],
            'the previous merged budget changed')
    historical = _history_rows(previous)
    for key in ('run_index', 'recording_id', 'source_recording_id'):
        used = {row.get(key) for row in historical}
        require(all(job[key] not in used for job in jobs), 'historical identity collision: ' + key)

    original = load_reference(PAIRING_SOURCE)
    source_rows = original['jobs']
    require(original['checkpoint_update'] == 13 and len(source_rows) == 100
            and original['sampling'] == 'stochastic_private_rng'
            and all(previous['jobs'][str(row['run_index'])]['status'] == 'COMPLETE'
                    and all(previous['jobs'][str(row['run_index'])]['identity'][field] == row[field]
                            for field in SEED_FIELDS) for row in source_rows),
            'the 100 paired environment and sampling seeds require completed source evaluation rows')
    eval_jobs = [job for job in jobs if job['role'] == 'final']
    for new, old in zip(eval_jobs, source_rows * 3):
        require(all(new[field] == old[field] for field in SEED_FIELDS),
                'seed replay requires the corresponding original evaluation row')

    train_jobs = [job for job in jobs if job['role'] == 'train']
    training_fields = ('experiment_seed', 'actor_sampling_seed')
    require(all(job['actor_seed'] == original['actor_seed'] for job in train_jobs),
            'continuation must keep the original actor initialization identity')
    seeds = [job[field] for job in train_jobs for field in training_fields]
    used_seeds = {row[field] & MAX_JAVA_SEED for row in historical
                  for field in SEED_FIELDS if type(row.get(field)) is int}
    require(len(seeds) == len(set(seeds)) == 2 * 512
            and all(type(seed) is int and 0 < seed <= MAX_JAVA_SEED for seed in seeds)
            and used_seeds.isdisjoint(seeds),
            'a new training environment or Actor stream repeats a historical or new seed')
    prior_java = {state for row in historical if type(row.get('experiment_seed')) is int
                  for state in java_random_states(row['experiment_seed']).values()}
    new_java = [state for job in train_jobs
                for state in java_random_states(job['experiment_seed']).values()]
    require(len(new_java) == len(set(new_java)) == 512 * 5 and prior_java.isdisjoint(new_java),
            'new training engine/site Java random state repeats history or this extension')


def _complete_receipt(job, phase_config, dataset_ref, receipt):
    require(receipt['phase_config'] == phase_config
            and receipt['phase_id'] == job['phase_id']
            and receipt['role'] == job['role']
            and receipt['run_index'] == job['run_index']
            and receipt['experiment_seed'] == job['experiment_seed']
            and receipt['normal_completion'] is True and receipt['four_windows'] == 4
            and receipt['references']['episode_manifest'] == dataset_ref,
            'physical receipt does not bind the expected phase, seed and episode manifest')


def budget_snapshot(path=None):
    """Merge every slot, including PLANNED and unresolved consumption."""
    result = preceding_snapshot()
    path = LEDGER if path is None else Path(path)
    if not path.is_file():
        return result
    with connection(path, readonly=True) as con:
        rows = con.execute('SELECT payload,checksum FROM campaigns').fetchall()
        require(len(rows) == 1, 'extension ledger registration inventory differs')
        entry = _payload(rows[0])
    suite_ref, approval_ref = entry['suite_reference'], entry['approval_reference']
    suite = load_reference(suite_ref)
    ledger = ExtensionLedger(
        suite, suite_path=suite_ref['path'], approval_path=approval_ref['path'],
        path=None if path.resolve() == LEDGER.resolve() else path,
    )
    check_collisions(suite, result)
    state = ledger.snapshot()
    result['jobs'].update(state['jobs'])
    result['campaigns'].update(state['campaigns'])
    result['maximum_total_shifts'] += EXTENSION_CAP
    result['additional_reserved_shifts'] += EXTENSION_CAP
    return result


class ExtensionLedger(BudgetLedger):
    """Keep the collection API while starting the commit sequence at update 17."""

    def __init__(self, suite, *, suite_path, approval_path, path=None):
        suite_ref, approval_ref = reference(suite_path), reference(approval_path)
        _approval(suite_ref, approval_ref)
        sealed = load_reference(suite_ref)
        require('jobs' not in sealed, 'the sealed suite must have a single phase inventory')
        jobs, configs = _phase_inventory(sealed)
        require(suite['campaign_id'] == CAMPAIGN and suite['phases'] == sealed['phases']
                and suite.get('jobs', jobs) == jobs, 'in-memory campaign differs from sealed phases')
        self.path = LEDGER if path is None else Path(path).resolve()
        require(path is None or self.path != LEDGER.resolve(),
                'temporary ledger override cannot point at production')
        self.suite = dict(sealed, jobs=jobs, episodes_per_batch=BATCH_SIZE)
        self.campaign_id = CAMPAIGN
        self.jobs = {job['run_index']: job for job in jobs}
        self.hashes = {run: digest(job) for run, job in self.jobs.items()}
        self.configs = configs
        self.update_jobs = {update: [] for update in range(FIRST_UPDATE, LAST_UPDATE + 1)}
        for job in jobs:
            if job['role'] == 'train':
                self.update_jobs[job['target_training_update']].append(job)
        self.entry = dict(format=FORMAT, suite_reference=suite_ref, approval_reference=approval_ref,
                          cap=EXTENSION_CAP, test_only=path is not None)

    def register(self):
        """Reserve the cap only during the authorized controller launch."""
        check_collisions(self.suite)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with connection(self.path) as con:
            con.execute('CREATE TABLE IF NOT EXISTS campaigns (id TEXT PRIMARY KEY,payload TEXT,checksum TEXT)')
            con.execute('CREATE TABLE IF NOT EXISTS slots (run INTEGER PRIMARY KEY,campaign TEXT,status TEXT,payload TEXT,checksum TEXT)')
            con.execute('CREATE TABLE IF NOT EXISTS commits (campaign TEXT,update_id INTEGER,payload TEXT,checksum TEXT,PRIMARY KEY(campaign,update_id))')
            registrations = con.execute('SELECT id,payload,checksum FROM campaigns').fetchall()
            if registrations:
                require(len(registrations) == 1 and registrations[0][0] == CAMPAIGN
                        and _payload(registrations[0][1:]) == self.entry, 'existing registration differs')
                return
            con.execute('INSERT INTO campaigns VALUES (?,?,?)',
                        (CAMPAIGN, encoded(self.entry).decode(), digest(self.entry)))
            records = []
            for run, identity in self.hashes.items():
                record = dict(status='PLANNED', identity_sha256=identity)
                records.append((run, CAMPAIGN, 'PLANNED', encoded(record).decode(), digest(record)))
            con.executemany('INSERT INTO slots VALUES (?,?,?,?,?)', records)

    def snapshot(self):
        with connection(self.path, readonly=True) as con:
            entry = con.execute('SELECT id,payload,checksum FROM campaigns').fetchall()
            require(len(entry) == 1 and entry[0][0] == CAMPAIGN
                    and _payload(entry[0][1:]) == self.entry, 'extension registration changed')
            jobs = {}
            future = False
            for run, campaign, status, data, checksum in con.execute(
                    'SELECT run,campaign,status,payload,checksum FROM slots ORDER BY run'):
                record = _payload((data, checksum))
                job = self.jobs.get(run)
                require(job is not None and campaign == CAMPAIGN and status == record['status']
                        and self.hashes[run] == record['identity_sha256']
                        and status in ('PLANNED', 'CLAIMED', 'COMPLETE', 'UNKNOWN'),
                        'extension slot inventory or legal state differs')
                require(not future or status == 'PLANNED', 'physical shifts did not advance in plan order')
                future = future or status != 'COMPLETE'
                if status == 'COMPLETE':
                    ref, receipt = record['dataset_ref'], record['receipt']
                    require(record['completion_sha256'] == digest(dict(dataset_ref=ref, receipt=receipt)),
                            'physical completion checksum differs')
                    _complete_receipt(job, self.configs[job['phase_id']], ref, receipt)
                jobs[str(run)] = dict(record, campaign_id=CAMPAIGN, identity=job)
            require(len(jobs) == EXTENSION_CAP, 'extension slot inventory incomplete')
            commits = {}
            for campaign, update, data, checksum in con.execute(
                    'SELECT campaign,update_id,payload,checksum FROM commits ORDER BY update_id'):
                record = _payload((data, checksum))
                require(campaign == CAMPAIGN and update == record['update'] == FIRST_UPDATE + len(commits)
                        and update in self.update_jobs
                        and record['job_ids'] == [str(job['run_index']) for job in self.update_jobs[update]]
                        and all(jobs[run]['status'] == 'COMPLETE'
                                and jobs[run].get('committed_update') == update for run in record['job_ids']),
                        'extension commit order or training batch differs')
                commits[str(update)] = record
            for key, row in jobs.items():
                if 'committed_update' in row:
                    commit = commits[str(row['committed_update'])]
                    require(key in commit['job_ids'], 'slot has no matching checkpoint commit')
                if row['status'] != 'PLANNED' and row['identity']['checkpoint_update'] >= FIRST_UPDATE:
                    require(str(row['identity']['checkpoint_update']) in commits,
                            'physical slot started before its checkpoint was committed')
        return dict(jobs=jobs, campaigns={
            CAMPAIGN: dict(self.entry, campaign=self.suite, commits=commits),
        })

    def claim(self, job):
        run = job['run_index']
        with connection(self.path) as con:
            record = self._row(con, job)
            require(record['status'] == 'PLANNED', 'slot cannot be claimed twice')
            earlier = con.execute("SELECT 1 FROM slots WHERE run < ? AND status != 'COMPLETE' LIMIT 1",
                                  (run,)).fetchone()
            require(earlier is None, 'an earlier physical slot is incomplete')
            checkpoint = job['checkpoint_update']
            if checkpoint >= FIRST_UPDATE:
                require(con.execute('SELECT 1 FROM commits WHERE campaign=? AND update_id=?',
                                    (CAMPAIGN, checkpoint)).fetchone() is not None,
                        'checkpoint must be committed before a physical slot starts')
            record['status'] = 'CLAIMED'
            self._put(con, job, record)

    def complete(self, job, dataset_ref, receipt):
        _complete_receipt(job, self.configs[job['phase_id']], dataset_ref, receipt)
        super().complete(job, dataset_ref, receipt)

    def commit_record(self, update):
        require(update in self.update_jobs, 'invalid extension update')
        with connection(self.path, readonly=True) as con:
            exists = con.execute('SELECT 1 FROM commits WHERE campaign=? AND update_id=?',
                                 (CAMPAIGN, update)).fetchone()
        return super().commit_record(update) if exists else None

    def commit(self, update, jobs, checkpoint_ref, batch_id):
        require(type(update) is int and update in self.update_jobs
                and jobs == self.update_jobs[update] and len(jobs) == BATCH_SIZE
                and reference(checkpoint_ref['path']) == checkpoint_ref
                and isinstance(batch_id, str) and bool(batch_id),
                'commit must bind an exact extension training batch and checkpoint')
        with connection(self.path) as con:
            prior = [_payload(row) for row in con.execute(
                'SELECT payload,checksum FROM commits WHERE campaign=? ORDER BY update_id', (CAMPAIGN,))]
            require(update == FIRST_UPDATE + len(prior) and
                    all(record['batch_id'] != batch_id for record in prior),
                    'update out of order or batch already committed')
            records = [self._row(con, job) for job in jobs]
            for job, record in zip(jobs, records):
                require(record['status'] == 'COMPLETE' and 'committed_update' not in record
                        and reference(record['dataset_ref']['path']) == record['dataset_ref'],
                        'commit sources changed or were already used')
                record['committed_update'] = update
                self._put(con, job, record)
            data = dict(update=update, job_ids=[str(job['run_index']) for job in jobs],
                        checkpoint_ref=checkpoint_ref, batch_id=batch_id)
            con.execute('INSERT INTO commits VALUES (?,?,?,?)',
                        (CAMPAIGN, update, encoded(data).decode(), digest(data)))


if __name__ == '__main__':
    from collections import Counter

    merged = budget_snapshot()
    state = Counter(row['status'] for row in merged['jobs'].values() if row['campaign_id'] == CAMPAIGN)
    print(json.dumps(dict(maximum_total_shifts=merged['maximum_total_shifts'],
                          extension_registered=CAMPAIGN in merged['campaigns'],
                          extension_slots=state)))
