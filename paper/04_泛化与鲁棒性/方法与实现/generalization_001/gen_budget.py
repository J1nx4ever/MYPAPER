"""Dedicated, append-only evaluation budget for the single-seed u32 experiment."""

from pathlib import Path
import sys

PLAN = Path(__file__).resolve().parent
sys.path.insert(0, str(PLAN.parents[3]))

from business_graph_summary_campaign_v1.budget import (
    BudgetLedger, _payload, _state, connection, digest,
)
from business_graph_summary_campaign_v1.contract import require
from business_graph_summary_campaign_v1.plan_io import (
    ROOT, encoded, load_phase, load_reference, reference,
)
from change_plans.token_attention_001.business_graph_summary_009.training_extension_001 import (
    extension_budget as prior,
)

CAMPAIGN = 'bg128-u32-generalization-20261119-001'
LEDGER = ROOT / 'runs' / CAMPAIGN / 'generalization_ledger.sqlite'
FORMAT = 'bg128-u32-generalization-budget-v1'
AUTHORITY_SCHEMA = 'bg128-u32-generalization-launch-authority-v1'
PREVIOUS_CAP = 22611
FIRST_RUN = 9991001
SHIFTS_PER_SCENARIO = 100
SCENARIOS = (
    'nominal', 'grade_swap', 'speed_18kph', 'speed_15kph',
    'grade_mean_shift', 'grade_variability',
)
CAP = len(SCENARIOS) * SHIFTS_PER_SCENARIO
CHECKPOINT_SHA = '5d42df7d8e33b5f11424094268a4da894a36454e46d86889fc0218ecce6b2914'
SEED_FIELDS = ('experiment_seed', 'actor_sampling_seed')
JAVA_MASK = (1 << 48) - 1


def preceding_snapshot():
    """Carry forward all previously registered slots and the sealed u32 commitment."""
    snapshot = prior.budget_snapshot()
    extension = snapshot['campaigns'].get(prior.CAMPAIGN, {})
    commits = extension.get('commits', {})
    slots = [slot for slot in snapshot['jobs'].values() if slot.get('campaign_id') == prior.CAMPAIGN]
    require(snapshot['maximum_total_shifts'] == PREVIOUS_CAP
            and snapshot['active_h2_reserved_shifts'] == 0
            and CAMPAIGN not in snapshot['campaigns']
            and len(slots) == prior.EXTENSION_CAP
            and all(slot['status'] == 'COMPLETE' for slot in slots)
            and set(commits) == {str(update) for update in range(17, 33)}
            and commits['32']['checkpoint_ref']['sha256'] == CHECKPOINT_SHA,
            'completed u16–u32 extension or prior budget differs')
    return snapshot


def _phase_inventory(suite):
    require(suite.get('campaign_id') == CAMPAIGN and suite.get('budget_path') == str(LEDGER)
            and suite.get('checkpoint_sha256') == CHECKPOINT_SHA,
            'generalization scope, checkpoint, or ledger path differs')
    phases = suite['phases']
    require(tuple(phase['scenario_id'] for phase in phases) == SCENARIOS,
            'scenario order or inventory differs')
    jobs, refs, recordings = [], {}, set()
    for phase in phases:
        phase_id = phase['phase_id']
        config_ref = phase['phase_config']
        require(phase_id not in refs and phase['checkpoint_update'] == 32
                and reference(config_ref['path']) == config_ref,
                'frozen phase or checkpoint identity differs')
        config = load_phase(config_ref['path'])
        recording = f'{CAMPAIGN}-{phase_id}'
        require(config['campaign_id'] == CAMPAIGN and config['phase_id'] == phase_id
                and config['python_output'] == phase['python_output']
                and config['recording_id'] == recording and recording not in recordings
                and config['evaluation_only'] is True and config['optimizer_steps'] == 0
                and len(config['jobs']) == SHIFTS_PER_SCENARIO,
                'frozen fixed-evaluation phase scope differs')
        recordings.add(recording)
        for job in config['jobs']:
            ordinal = len(jobs)
            require(job['ordinal'] == ordinal and job['run_index'] == FIRST_RUN + ordinal
                    and job['phase_id'] == phase_id and job['python_output'] == phase['python_output']
                    and job['recording_id'] == job['source_recording_id'] == recording
                    and job['candidate_id'] == 's01_BG128' and job['role'] == 'final'
                    and job['partition'] == 'validation'
                    and job['checkpoint_update'] == 32 and job['target_training_update'] == 0
                    and job.get('scenario_id', phase['scenario_id']) == phase['scenario_id'],
                    'frozen generalization slot identity or role differs')
            jobs.append(job)
        refs[phase_id] = config_ref
    require(len(jobs) == CAP and ('jobs' not in suite or suite['jobs'] == jobs),
            'expanded generalization slot inventory differs')
    return jobs, refs


def check_collisions(suite, snapshot=None):
    """Allow same-slot seed reuse within this comparison, nowhere in its history."""
    from physical_curriculum_v1.budget import _history_rows
    from python.offline_split import java_random_states

    jobs, _ = _phase_inventory(suite)
    previous = preceding_snapshot() if snapshot is None else snapshot
    require(previous['maximum_total_shifts'] == PREVIOUS_CAP and CAMPAIGN not in previous['campaigns'],
            'preceding capacity or campaign identity changed')
    historical = _history_rows(previous)
    for key in ('run_index', 'recording_id', 'source_recording_id'):
        used = {row.get(key) for row in historical}
        require(all(job[key] not in used for job in jobs), 'historical identity collision: ' + key)

    def pair(job):
        return tuple(job[field] for field in SEED_FIELDS)

    nominal = [pair(job) for job in jobs[:SHIFTS_PER_SCENARIO]]
    require([pair(job) for job in jobs] == nominal * len(SCENARIOS),
            'new scenario slots must share the corresponding nominal environment and Actor seed')
    new_seeds = [seed for seeds in nominal for seed in seeds]
    used_seeds = {row[field] & JAVA_MASK for row in historical
                  for field in (*SEED_FIELDS, 'actor_seed') if type(row.get(field)) is int}
    require(len(set(new_seeds)) == len(new_seeds)
            and all(type(seed) is int and 0 < seed <= JAVA_MASK for seed in new_seeds)
            and used_seeds.isdisjoint(new_seeds),
            'generalization environment or Actor seed repeats another slot or history')
    # The actor_seed identifies the fixed checkpoint's original initialization.
    original_actor = {row['identity']['actor_seed'] for row in previous['jobs'].values()
                      if row.get('campaign_id') == prior.CAMPAIGN}
    require(len(original_actor) == 1
            and all(job['actor_seed'] in original_actor for job in jobs)
            and original_actor.isdisjoint(new_seeds), 'frozen Actor initialization seed differs')
    previous_states = {state for row in historical if type(row.get('experiment_seed')) is int
                       for state in java_random_states(row['experiment_seed']).values()}
    new_states = [state for env, _ in nominal for state in java_random_states(env).values()]
    require(len(new_states) == len(set(new_states)) == SHIFTS_PER_SCENARIO * 5
            and previous_states.isdisjoint(new_states),
            'Java engine or site random states repeat another slot or history')


def _authorization(suite_ref, approval_ref):
    approval = load_reference(approval_ref)
    require(approval.get('schema') == AUTHORITY_SCHEMA and approval.get('campaign_id') == CAMPAIGN
            and approval.get('suite_sha256') == suite_ref['sha256']
            and approval.get('checkpoint_sha256') == CHECKPOINT_SHA
            and approval.get('max_new_shifts') == CAP and approval.get('training_shifts') == 0
            and approval.get('optimizer_updates') == 0
            and approval.get('authorized_by_anylogic_run') is True,
            '600-shift fixed-checkpoint AnyLogic launch authority differs')


class GenLedger(BudgetLedger):
    """Retain collection-compatible slots and prohibit optimizer commitments."""

    def __init__(self, suite, *, suite_path, approval_path, path=None):
        suite_ref, approval_ref = reference(suite_path), reference(approval_path)
        _authorization(suite_ref, approval_ref)
        sealed = load_reference(suite_ref)
        require('jobs' not in sealed, 'sealed suite must use only the phase inventory')
        jobs, refs = _phase_inventory(sealed)
        require({key: value for key, value in suite.items() if key != 'jobs'} == sealed
                and suite.get('jobs', jobs) == jobs, 'in-memory generalization suite differs')
        self.path = LEDGER if path is None else Path(path).resolve()
        require(path is None or self.path != LEDGER.resolve(),
                'temporary ledger override cannot point at production')
        self.suite = dict(sealed, jobs=jobs)
        self.campaign_id = CAMPAIGN
        self.jobs = {job['run_index']: job for job in jobs}
        self.hashes = {run: digest(job) for run, job in self.jobs.items()}
        self.configs = refs
        self.update_jobs = {}
        self.entry = dict(format=FORMAT, suite_reference=suite_ref, approval_reference=approval_ref,
                          cap=CAP, test_only=path is not None)

    def register(self):
        """Register the fixed batch after its user-operated launch, before any claim."""
        check_collisions(self.suite)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with connection(self.path) as con:
            con.execute('CREATE TABLE IF NOT EXISTS campaigns (id TEXT PRIMARY KEY,payload TEXT,checksum TEXT)')
            con.execute('CREATE TABLE IF NOT EXISTS slots (run INTEGER PRIMARY KEY,campaign TEXT,status TEXT,payload TEXT,checksum TEXT)')
            con.execute('CREATE TABLE IF NOT EXISTS commits (campaign TEXT,update_id INTEGER,payload TEXT,checksum TEXT,PRIMARY KEY(campaign,update_id))')
            registered = con.execute('SELECT id,payload,checksum FROM campaigns').fetchall()
            if registered:
                require(len(registered) == 1 and registered[0][0] == CAMPAIGN
                        and _payload(registered[0][1:]) == self.entry,
                        'existing generalization registration differs')
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
            registered = con.execute('SELECT id,payload,checksum FROM campaigns').fetchall()
            require(len(registered) == 1 and registered[0][0] == CAMPAIGN
                    and _payload(registered[0][1:]) == self.entry
                    and con.execute('SELECT COUNT(*) FROM slots').fetchone()[0] == CAP
                    and con.execute('SELECT 1 FROM commits LIMIT 1').fetchone() is None,
                    'generalization registration, inventory, or zero-optimization contract changed')
            state = _state(con, self.suite, self.entry, self.hashes)
        future = False
        for run in self.jobs:
            record = state['jobs'][str(run)]
            status = record['status']
            require(not future or status == 'PLANNED', 'shifts did not advance in frozen plan order')
            require('committed_update' not in record, 'fixed-policy evaluation slot was marked as training')
            future = future or status != 'COMPLETE'
            if status == 'COMPLETE':
                job = record['identity']
                prior._complete_receipt(job, self.configs[job['phase_id']], record['dataset_ref'], record['receipt'])
        return state

    def claim(self, job):
        with connection(self.path) as con:
            record = self._row(con, job)
            require(record['status'] == 'PLANNED'
                    and con.execute('SELECT 1 FROM slots WHERE run < ? AND status != ? LIMIT 1',
                                    (job['run_index'], 'COMPLETE')).fetchone() is None,
                    'slot cannot be claimed twice or before a prior slot completes')
            record['status'] = 'CLAIMED'
            self._put(con, job, record)

    def complete(self, job, dataset_ref, receipt):
        prior._complete_receipt(job, self.configs[job['phase_id']], dataset_ref, receipt)
        super().complete(job, dataset_ref, receipt)

    def commit(self, update, jobs, checkpoint_ref, batch_id):
        require(False, 'the frozen generalization budget cannot commit a training update')


def budget_snapshot(path=None):
    """Read the entire merged budget including PLANNED and unresolved generalization slots."""
    snapshot = preceding_snapshot()
    path = LEDGER if path is None else Path(path).resolve()
    if not path.is_file():
        return snapshot
    with connection(path, readonly=True) as con:
        registered = con.execute('SELECT payload,checksum FROM campaigns').fetchall()
        require(len(registered) == 1, 'generalization registration inventory differs')
        entry = _payload(registered[0])
    suite_ref, approval_ref = entry['suite_reference'], entry['approval_reference']
    suite = load_reference(suite_ref)
    ledger = GenLedger(suite, suite_path=suite_ref['path'], approval_path=approval_ref['path'],
                       path=None if path == LEDGER else path)
    check_collisions(suite, snapshot)
    state = ledger.snapshot()
    snapshot['jobs'].update(state['jobs'])
    snapshot['campaigns'].update(state['campaigns'])
    snapshot['maximum_total_shifts'] += CAP
    snapshot['additional_reserved_shifts'] += CAP
    return snapshot
