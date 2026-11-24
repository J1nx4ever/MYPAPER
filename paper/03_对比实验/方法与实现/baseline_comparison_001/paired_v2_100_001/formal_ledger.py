"""Register and admit 500 physical V2 comparison shifts."""
from pathlib import Path
import sys

from formal_plan import (PLAN, CAMPAIGN, COUNT, OUTPUT, SUITE, encoded, load_reference, load_suite, reference,
                         require, validate_jobs)
import weak_budget as weak
from weak_budget import LOCKS, budget_snapshot as preceding_snapshot

# Upstream budget importers prepend older plan directories. Resolve the new
# paired_runtime and java_adapter through this plan directory when launched.
if str(PLAN) in sys.path:
    sys.path.remove(str(PLAN))
sys.path.insert(0, str(PLAN))
from business_graph_summary_campaign_v1.budget import connection, digest, _payload
from business_graph_summary_campaign_v1.launch import _launch_lock
from physical_curriculum_v1.budget import _history_rows
from python.offline_split import JAVA_MASK, java_random_states


SCHEMA = 'bg128-traditional-paired100-v2-budget-v1'
LEDGER = OUTPUT/'budget.sqlite'
AUTHORITY = OUTPUT/'launch/run_authority.json'
STATUSES = ('COMPLETE', 'CLAIMED', 'UNKNOWN', 'PLANNED')
SEED_FIELDS = ('experiment_seed', 'actor_sampling_seed', 'actor_seed')


def _prepared_rows(historical):
    rows = weak._prepared_rows(historical)
    for unit in weak.plan.UNITS:
        location = weak.plan.locations(unit)
        if location['campaign_id'] not in historical['campaigns'] and Path(location['suite_path']).is_file():
            suite, _ = weak.plan.load_package(unit)
            rows.extend(suite['jobs'])
    return rows


def check_fresh(suite, historical):
    require(CAMPAIGN not in historical['campaigns'] and historical['active_h2_reserved_shifts'] == 0,
            'paired campaign is already registered or previous budget scope differs')
    jobs = suite['jobs']
    validate_jobs(jobs)
    history = _history_rows(historical)
    jobs_unregistered = _prepared_rows(historical)
    occupied_rows = [*history, *jobs_unregistered]
    live_runs = {row.get('run_index') for row in occupied_rows}
    live_recordings = {row.get(key) for row in occupied_rows for key in ('recording_id', 'source_recording_id')}
    require(all(job['run_index'] not in live_runs and job['recording_id'] not in live_recordings
                and job['source_recording_id'] not in live_recordings for job in jobs),
            'physical run or recording overlaps historical or prepared identity')
    previous = {row[key] & JAVA_MASK for row in occupied_rows for key in SEED_FIELDS
                if type(row.get(key)) is int and row[key] > 0}
    environment = {job['experiment_seed'] for job in jobs}
    sampling = {job['actor_sampling_seed'] for job in jobs if job['actor_sampling_seed_consumed']}
    training_lineage = {job['actor_seed'] & JAVA_MASK for job in jobs if job['actor_seed'] > 0}
    require((environment|sampling).isdisjoint(previous|training_lineage),
            'new environment or private Actor sampling seed overlaps history')
    previous_java = {state for row in occupied_rows if type(row.get('experiment_seed')) is int and row['experiment_seed'] > 0
                     for state in java_random_states(row['experiment_seed']).values()}
    new_java = [state for seed in environment for state in java_random_states(seed).values()]
    require(len(set(new_java)) == len(new_java) and previous_java.isdisjoint(new_java),
            'effective Java engine or geology RNG state overlaps another slot or history')
    return dict(occupied_before=historical['maximum_total_shifts'],
                maximum_total_if_registered=historical['maximum_total_shifts']+COUNT,
                historical_identities_checked=len(history), prepared_identities_checked=len(jobs_unregistered),
                new_physical_jobs=COUNT,
                new_environment_seeds=len(environment), new_actor_sampling_seeds=len(sampling),
                new_java_rng_states=len(new_java))


def authority_for(suite):
    first = suite['jobs'][0]
    return dict(schema=SCHEMA+'-authority', campaign_id=CAMPAIGN, suite=reference(SUITE),
                authorized_by_matching_anylogic_run=True, maximum_new_shifts=COUNT,
                first_run={key: first[key] for key in ('run_index', 'experiment_seed', 'recording_id')})


class Ledger:
    """Serial admission with immutable identities and atomic status transitions."""

    def __init__(self, suite):
        self.suite = suite
        self.jobs = {job['run_index']:job for job in suite['jobs']}
        self.hashes = {run:digest(job) for run, job in self.jobs.items()}

    def _entry(self):
        return dict(schema=SCHEMA, campaign_id=CAMPAIGN, suite=reference(SUITE), authority=reference(AUTHORITY), cap=COUNT)

    def register(self):
        with _launch_lock(LOCKS/'budget'):
            require(not LEDGER.exists() and load_reference(reference(AUTHORITY)) == authority_for(self.suite),
                    'existing ledger or mismatched AnyLogic launch authority')
            occupied = check_fresh(self.suite, preceding_snapshot())['occupied_before']
            entry = dict(self._entry(), occupied_before=occupied)
            with connection(LEDGER) as con:
                con.execute('CREATE TABLE campaign (id TEXT PRIMARY KEY, payload TEXT, checksum TEXT)')
                con.execute('CREATE TABLE slots (run INTEGER PRIMARY KEY, status TEXT, payload TEXT, checksum TEXT)')
                con.execute('INSERT INTO campaign VALUES (?,?,?)', (CAMPAIGN, encoded(entry).decode(), digest(entry)))
                con.executemany('INSERT INTO slots VALUES (?,?,?,?)',
                                (self._row(run, 'PLANNED') for run in self.jobs))

    def _row(self, run, status, **fields):
        slot = dict(status=status, identity_sha256=self.hashes[run], **fields)
        return run, status, encoded(slot).decode(), digest(slot)

    def _decode(self, run, status, payload, checksum):
        slot = _payload((payload, checksum))
        require(run in self.jobs and slot['identity_sha256'] == self.hashes[run]
                and status == slot['status'] and status in STATUSES, 'paired physical slot identity differs')
        return slot

    def snapshot(self):
        with connection(LEDGER, readonly=True) as con:
            (cid, text, checksum), = con.execute('SELECT id,payload,checksum FROM campaign')
            entry = _payload((text, checksum))
            require(cid == CAMPAIGN and entry == dict(self._entry(), occupied_before=entry['occupied_before'])
                    and load_reference(entry['authority']) == authority_for(self.suite),
                    'paired ledger registration or authority differs')
            slots = {str(run):dict(self._decode(run, status, payload, checksum), identity=self.jobs[run])
                     for run, status, payload, checksum in con.execute(
                         'SELECT run,status,payload,checksum FROM slots ORDER BY run')}
        require(len(slots) == COUNT, 'incomplete physical slot inventory')
        statuses = [row['status'] for row in slots.values()]
        require(statuses == sorted(statuses, key=STATUSES.index)
                and sum(status in ('CLAIMED', 'UNKNOWN') for status in statuses) <= 1,
                'paired ledger physical order differs')
        return dict(entry=entry, slots=slots)

    def _transition(self, job, before, after, **details):
        run = job['run_index']
        require(self.jobs.get(run) == job, 'noncanonical paired job')
        with connection(LEDGER) as con:
            row = con.execute('SELECT status,payload,checksum FROM slots WHERE run=?', (run,)).fetchone()
            require(row is not None and self._decode(run, *row)['status'] == before,
                    'paired slot is out of order or already used')
            if after == 'CLAIMED':
                earlier = con.execute("SELECT 1 FROM slots WHERE run < ? AND status != 'COMPLETE' LIMIT 1", (run,)).fetchone()
                require(earlier is None, 'an earlier physical job is incomplete or unresolved')
            _, status, text, checksum = self._row(run, after, **details)
            con.execute('UPDATE slots SET status=?,payload=?,checksum=? WHERE run=?', (status, text, checksum, run))

    def claim(self, job):
        self._transition(job, 'PLANNED', 'CLAIMED')

    def complete(self, job, receipt):
        require(receipt['path'] == str(Path(job['python_output'])/'evaluation_receipt.json'),
                'receipt must match the frozen physical output path')
        payload = load_reference(receipt)
        for key in ('run_index', 'recording_id', 'experiment_seed', 'actor_sampling_seed', 'policy_kind',
                    'phase_config', 'pair_ordinal'):
            require(payload[key] == job[key], 'physical receipt differs from launched job: ' + key)
        require(payload['status'] == 'COMPLETE' and payload['normal_completion'] is True
                and payload['completed_shifts'] == 1 and payload['optimizer_updates'] == 0,
                'complete evaluation-only physical receipt required')
        self._transition(job, 'CLAIMED', 'COMPLETE', receipt=receipt)

    def mark_unknown(self, job, error):
        self._transition(job, 'CLAIMED', 'UNKNOWN', reason=str(error))


def budget_snapshot():
    """Merge the GAT weak ablation and prior ledgers plus this registered V2 campaign."""
    result = preceding_snapshot()
    if not LEDGER.is_file():
        return result
    suite = load_suite(check_sources=False)
    check_fresh(suite, result)
    state = Ledger(suite).snapshot()
    result['jobs'].update({run:dict(row, campaign_id=CAMPAIGN) for run, row in state['slots'].items()})
    entry = state['entry']
    result['campaigns'][CAMPAIGN] = dict(campaign=entry['suite'], authority=entry['authority'], commits={}, cap=COUNT)
    result['maximum_total_shifts'] += COUNT
    result['additional_reserved_shifts'] += COUNT
    return result
