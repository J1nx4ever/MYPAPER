"""Append one approved replacement while retaining the consumed UNKNOWN slot."""
from collections import Counter
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from business_graph_summary_campaign_v1.budget import connection, digest, _payload, _state
from business_graph_summary_campaign_v1.plan_io import encoded, load_phase
from mixed_budget import MixedLedger
from mixed_plan import CAMPAIGN, LEDGER, TOTAL, UPDATES, read, reference, require

FAILED = 10000003
REPLACEMENT = 10002425
FAILED_REASON = "ProtocolError('AnyLogic reward configuration changed during this run')"
AUTHORITY_SCHEMA = 'bg128-continuous-mixed-replacement-authority-v1'
TABLE = 'replacement_scope'


def _registration(database):
    present = database.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (TABLE,)).fetchone()
    if not present:
        return None
    rows = database.execute(f'SELECT campaign,payload,checksum FROM {TABLE}').fetchall()
    require(len(rows) == 1 and rows[0][0] == CAMPAIGN, 'replacement authority registry differs')
    return _payload(rows[0][1:])


def replacement_registration(path=LEDGER):
    """Read the durable supplemental registration without attaching or writing a slot."""
    if not Path(path).is_file():
        return None
    with connection(path, readonly=True) as database:
        return _registration(database)


def authority_for(suite, replacement_ref):
    return dict(schema=AUTHORITY_SCHEMA, campaign_id=CAMPAIGN, original_suite=reference(suite['suite_path']),
                replacement_ref=replacement_ref, replacement_run_index=REPLACEMENT, failed_run_index=FAILED,
                additional_shifts=1, authorized_by_anylogic_run=True)


class ReplacementLedger(MixedLedger):
    """Keep canonical phase receipts and commit using the frozen logical job order."""
    def __init__(self, suite, configs, replacement_config, replacement_ref, *, approval=None, path=None):
        super().__init__(suite, configs)
        if path is not None:
            self.path = Path(path).resolve()
            require(self.path != LEDGER and self.path.is_file(), 'test registry needs an isolated existing database copy')
        self.original_suite = suite
        self.original_hashes = dict(self.hashes)
        self.failed_job = self.jobs[FAILED]
        require(reference(replacement_ref['path']) == replacement_ref
                and load_phase(replacement_ref['path']) == replacement_config,
                'supplemental phase bytes differ')
        require(len(replacement_config['jobs']) == 1 and replacement_config['campaign_id'] == CAMPAIGN,
                'supplemental phase must contain exactly one training shift')
        job = replacement_config['jobs'][0]
        old = self.failed_job
        require(job['run_index'] == REPLACEMENT and old['ordinal'] == job['ordinal'] == 2
                and job['run_index'] not in self.jobs and job['phase_id'] == replacement_config['phase_id']
                and job['phase_id'] not in self.configs and job['role'] == job['partition'] == 'train'
                and job['recording_id'] == job['source_recording_id'] == replacement_config['recording_id'],
                'supplemental training timeline or recording differs')
        require(all(job[key] == old[key] for key in ('expected_rpc_config', 'runtime_scene_sha256', 'scene_id',
                    'candidate_id', 'arm_update', 'checkpoint_update', 'target_training_update', 'actor_seed'))
                and job['target_training_update'] == 1 and job['checkpoint_update'] == 0,
                'replacement must preserve the failed slot material scene and u0 behavior policy')
        known_recordings = {record[key] for record in self.jobs.values() for key in ('recording_id','source_recording_id')}
        known_seeds = {record[key] for record in self.jobs.values() for key in ('actor_seed','experiment_seed','actor_sampling_seed')}
        new_seeds = {job['experiment_seed'], job['actor_sampling_seed']}
        require(job['recording_id'] not in known_recordings and len(new_seeds) == 2
                and all(type(seed) is int and 0 < seed < 2**48 for seed in new_seeds)
                and known_seeds.isdisjoint(new_seeds), 'replacement reuses an original recording or seed')
        self.replacement = job
        self.replacement_ref = replacement_ref
        self.approval = approval
        self.configs[job['phase_id']] = replacement_ref
        self.logical_runs = tuple(REPLACEMENT if run == FAILED else run for run in self.jobs)
        self.logical_position = {run: position for position, run in enumerate(self.logical_runs)}
        self.jobs[REPLACEMENT] = job
        self.hashes[REPLACEMENT] = digest(job)
        self.update_jobs[1] = [job if previous['run_index'] == FAILED else previous for previous in self.update_jobs[1]]
        self.suite = dict(suite, jobs=[*suite['jobs'], job], new_shift_cap=TOTAL + 1)
        self.failed_record = dict(status='UNKNOWN', error=FAILED_REASON, identity_sha256=self.hashes[FAILED])

    def _assert_original_registration(self, database):
        rows = database.execute('SELECT id,payload,checksum FROM campaigns').fetchall()
        require(len(rows) == 1 and rows[0][0] == CAMPAIGN and _payload(rows[0][1:]) == self.entry,
                'original campaign authority changed')

    def _authority(self, ref):
        require(isinstance(ref, dict) and reference(ref['path']) == ref
                and read(ref['path']) == authority_for(self.original_suite, self.replacement_ref),
                'explicit operator-click supplemental authority differs')
        return dict(schema='bg128-mixed-replacement-registration-v1', original_suite=self.entry['suite_reference'],
                    approval_reference=ref, replacement_phase=self.replacement_ref,
                    replacement_run_index=REPLACEMENT, failed_run_index=FAILED,
                    added_shifts=1, identity_sha256=self.hashes[REPLACEMENT])

    def _verified_registration(self, database):
        registration = _registration(database)
        if registration is not None:
            require(registration == self._authority(registration['approval_reference']),
                    'registered replacement authority differs')
            if self.approval is not None:
                require(registration['approval_reference'] == self.approval, 'replacement approval was rebound')
        return registration

    def _assert_failed(self, database):
        row = self._row(database, self.failed_job)
        require(row == self.failed_record, 'consumed failed shift must remain the original UNKNOWN')

    def _before_state(self, database):
        original = _state(database, self.original_suite, self.entry, self.original_hashes)
        require(not original['campaigns'][CAMPAIGN]['commits'] and
                Counter(row['status'] for row in original['jobs'].values()) == {
                    'COMPLETE': 2, 'UNKNOWN': 1, 'PLANNED': TOTAL - 3,
                }, 'unexpected original physical or optimizer inventory')
        require(all(original['jobs'][str(run)]['status'] == 'COMPLETE' for run in self.logical_runs[:2]),
                'first two original completed shifts differ')
        return original

    def register(self):
        """Add one supplemental slot transactionally, only with the exact operator authority."""
        require(self.approval is not None, 'supplemental registration requires explicit operator authority')
        entry = self._authority(self.approval)
        with connection(self.path) as database:
            self._assert_original_registration(database)
            self._assert_failed(database)
            current = self._verified_registration(database)
            if current is not None:
                require(current == entry, 'existing replacement scope differs')
                return
            self._before_state(database)
            database.execute(f'CREATE TABLE {TABLE} (campaign TEXT PRIMARY KEY,payload TEXT,checksum TEXT)')
            database.execute(f'INSERT INTO {TABLE} VALUES (?,?,?)', (CAMPAIGN, encoded(entry).decode(), digest(entry)))
            planned = dict(status='PLANNED', identity_sha256=self.hashes[REPLACEMENT])
            database.execute('INSERT INTO slots VALUES (?,?,?,?,?)',
                             (REPLACEMENT, CAMPAIGN, 'PLANNED', encoded(planned).decode(), digest(planned)))

    def snapshot(self):
        with connection(self.path, readonly=True) as database:
            self._assert_original_registration(database)
            self._assert_failed(database)
            entry = self._verified_registration(database)
            if entry is None:
                return self._before_state(database)
            require(database.execute('SELECT count(*) FROM slots').fetchone()[0] == TOTAL + 1,
                    'supplemental physical inventory differs')
            state = _state(database, self.suite, self.entry, self.hashes)
        campaign = state['campaigns'][CAMPAIGN]
        campaign.update(cap=TOTAL + 1, replacement_registration=entry)
        incomplete = False
        for run in self.logical_runs:
            row = state['jobs'][str(run)]
            require(not incomplete or row['status'] == 'PLANNED', 'logical physical execution order differs')
            incomplete = incomplete or row['status'] != 'COMPLETE'
        for update, record in campaign['commits'].items():
            require(record['job_ids'] == [str(job['run_index']) for job in self.update_jobs[int(update)]],
                    'commit differs from the supplemented training phase')
        return state

    def resume(self):
        state = self.snapshot()
        require('replacement_registration' in state['campaigns'][CAMPAIGN], 'supplemental shift is not registered')
        del state['jobs'][str(FAILED)]
        require(all(row['status'] not in ('CLAIMED', 'UNKNOWN') for row in state['jobs'].values()),
                'unexpected CLAIMED or UNKNOWN preserved; reconcile without resampling')
        return state

    def claim(self, job):
        run = job['run_index']
        require(run in self.logical_position, 'consumed UNKNOWN is outside logical execution')
        with connection(self.path) as database:
            require(self._verified_registration(database) is not None, 'supplemental shift is not registered')
            self._assert_failed(database)
            row = self._row(database, job)
            statuses = dict(database.execute('SELECT run,status FROM slots WHERE campaign=?', (CAMPAIGN,)))
            position = self.logical_position[run]
            require(row['status'] == 'PLANNED' and
                    all(statuses.get(previous) == 'COMPLETE' for previous in self.logical_runs[:position]) and
                    all(statuses.get(later) == 'PLANNED' for later in self.logical_runs[position + 1:]),
                    'slot is already used or outside logical physical order')
            actual = database.execute('SELECT count(*) FROM commits WHERE campaign=?', (CAMPAIGN,)).fetchone()[0]
            needed = job['target_training_update'] - 1 if job['role'] == 'train' else UPDATES
            require(actual == needed, 'physical shift would use an uncommitted training checkpoint')
            row['status'] = 'CLAIMED'
            self._put(database, job, row)

    def commit_record(self, update):
        require(type(update) is int and update in self.update_jobs, 'invalid mixed training update')
        with connection(self.path, readonly=True) as database:
            present = database.execute('SELECT 1 FROM commits WHERE campaign=? AND update_id=?',
                                       (CAMPAIGN, update)).fetchone()
        return super().commit_record(update) if present is not None else None
