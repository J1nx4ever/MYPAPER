"""Keep replica ledgers independent and merge them with all registered history."""

from pathlib import Path

from replica_plan import CAP, REPLICAS, TRAIN, locations, parent, load_package, anchored_source
from business_graph_summary_campaign_v1.budget import BudgetLedger, _payload, _state, connection, digest
from business_graph_summary_campaign_v1.launch import _launch_lock
from business_graph_summary_campaign_v1.plan_io import encoded, load_reference, reference
from change_plans.token_attention_001.business_graph_summary_009.generalization_001.speed_congestion_001 import speed_budget
from change_plans.token_attention_001.business_graph_summary_009.training_extension_001.extension_budget import _complete_receipt
from physical_curriculum_v1.budget import _history_rows
from python.offline_split import java_random_states

ROOT = parent.ROOT
LOCKS = ROOT / 'runs/.bg128-mixed-replication-locks'
SEEDS = ('experiment_seed', 'actor_sampling_seed', 'actor_seed')


def preceding_snapshot():
    return speed_budget.budget_snapshot()


def assert_available(suites, previous):
    history = _history_rows(previous)
    identities = {key: {row.get(key) for row in history}
                  for key in ('run_index', 'recording_id', 'source_recording_id')}
    old_seeds = {row[key] & parent.MAX_JAVA_SEED for row in history for key in SEEDS
                 if type(row.get(key)) is int}
    old_states = {value for row in history if type(row.get('experiment_seed')) is int
                  for value in java_random_states(row['experiment_seed']).values()}
    runs, seeds, states = 0, 0, 0
    held_configs = set()
    for suite in suites:
        jobs = suite['jobs']
        parent.require(suite['campaign_id'] not in previous['campaigns'] and len(jobs) == CAP,
                       'replication already registered or its scope differs')
        for key, used in identities.items():
            candidate = {job[key] for job in jobs}
            parent.require(len(candidate) == (CAP if key == 'run_index' else 39)
                           and used.isdisjoint(candidate), 'historical collision: ' + key)
            used.update(candidate)
        train = jobs[:TRAIN]
        fresh = [suite['actor_seed'], *[job[key] for job in train for key in SEEDS[:2]]]
        train_states = [state for job in train for state in java_random_states(job['experiment_seed']).values()]
        config_hashes = {job['runtime_scene_sha256'] for job in train}
        parent.require(len(set(fresh)) == len(fresh) and all(0 < value < 2**48 for value in fresh)
                       and old_seeds.isdisjoint(fresh), 'training or initialization seed collision')
        parent.require(len(set(train_states)) == len(train_states) and old_states.isdisjoint(train_states),
                       'training Java engine or geology RNG collision')
        parent.require(held_configs.isdisjoint(config_hashes), 'replicas reuse a sampled training configuration')
        old_seeds.update(fresh)
        old_states.update(train_states)
        held_configs.update(config_hashes)
        runs += len(jobs)
        seeds += len(fresh)
        states += len(train_states)
    parent.require(previous['active_h2_reserved_shifts'] == 0, 'cancelled H2 reserve changed')
    return dict(occupied_before=previous['maximum_total_shifts'], new_runs_checked=runs,
                new_initialization_and_training_seeds_checked=seeds, new_java_rng_states_checked=states,
                projected_upper_bound=previous['maximum_total_shifts'] + runs,
                historical_identities_checked=len(history))


def authority_path(suite):
    return Path(suite['python_output']) / 'launch/run_authority.json'


def authorization(suite):
    return dict(schema='bg128-mixed-replica-authority-v1', suite=reference(suite['suite_path']),
                campaign_id=suite['campaign_id'], training_shifts=TRAIN, evaluation_shifts=CAP - TRAIN,
                maximum_new_shifts=CAP, source_evaluation=suite['source_completion'],
                rpc_port=suite['rpc_port'], authorized_by_matching_anylogic_run=True)


class ReplicaLedger(BudgetLedger):
    def __init__(self, suite, phases, *, path=None, approval=None):
        self.suite = suite
        self.campaign_id = suite['campaign_id']
        live_path = Path(suite['budget_path'])
        live_approval = authority_path(suite)
        self.path = live_path if path is None else Path(path).resolve()
        approval = live_approval if approval is None else Path(approval).resolve()
        parent.require((path is None and approval == live_approval)
                       or (path is not None and self.path != live_path and approval != live_approval),
                       'test and live ledger/authority must be isolated')
        auth_ref = reference(approval)
        parent.require(load_reference(auth_ref) == authorization(suite), 'launch authority differs')
        self.entry = dict(format='bg128-mixed-replica-v1', cap=CAP, test_only=path is not None,
                          suite_reference=reference(suite['suite_path']), approval_reference=auth_ref)
        self.jobs = {job['run_index']: job for job in suite['jobs']}
        self.hashes = {run: digest(job) for run, job in self.jobs.items()}
        self.update_jobs = {cfg['arm_update']: cfg['jobs'] for cfg in phases if cfg['policy_arm'] == 'train'}
        self.configs = {cfg['phase_id']: ref for cfg, ref in zip(phases, suite['phase_configs'], strict=True)}

    def register(self, *, test_history=None):
        if self.entry['test_only']:
            parent.require(test_history is not None, 'isolated registration requires a test history')
            self._register(test_history)
        else:
            parent.require(test_history is None, 'live registration must use the current merged history')
            with _launch_lock(LOCKS / 'budget'):
                self._register(budget_snapshot())

    def _register(self, history):
        assert_available([self.suite], history)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with connection(self.path) as con:
            con.execute('CREATE TABLE campaigns (id TEXT PRIMARY KEY,payload TEXT,checksum TEXT)')
            con.execute('CREATE TABLE slots (run INTEGER PRIMARY KEY,campaign TEXT,status TEXT,payload TEXT,checksum TEXT)')
            con.execute('CREATE TABLE commits (campaign TEXT,update_id INTEGER,payload TEXT,checksum TEXT,PRIMARY KEY(campaign,update_id))')
            con.execute('INSERT INTO campaigns VALUES (?,?,?)',
                        (self.campaign_id, encoded(self.entry).decode(), digest(self.entry)))
            for run, identity in self.hashes.items():
                record = dict(status='PLANNED', identity_sha256=identity)
                con.execute('INSERT INTO slots VALUES (?,?,?,?,?)',
                            (run, self.campaign_id, 'PLANNED', encoded(record).decode(), digest(record)))

    def snapshot(self):
        with connection(self.path, readonly=True) as con:
            rows = con.execute('SELECT id,payload,checksum FROM campaigns').fetchall()
            parent.require(len(rows) == 1 and rows[0][0] == self.campaign_id and _payload(rows[0][1:]) == self.entry
                           and con.execute('SELECT count(*) FROM slots').fetchone()[0] == CAP,
                           'replica ledger registration or inventory changed')
            return _state(con, self.suite, self.entry, self.hashes)

    def commit_record(self, update):
        with connection(self.path, readonly=True) as con:
            if con.execute('SELECT 1 FROM commits WHERE campaign=? AND update_id=?',
                           (self.campaign_id, update)).fetchone() is None:
                return None
        return super().commit_record(update)

    def claim(self, job):
        with connection(self.path) as con:
            record = self._row(con, job)
            previous = con.execute("SELECT 1 FROM slots WHERE run < ? AND status != 'COMPLETE' LIMIT 1",
                                   (job['run_index'],)).fetchone()
            updates = con.execute('SELECT count(*) FROM commits').fetchone()[0]
            parent.require(record['status'] == 'PLANNED' and previous is None
                           and updates == job['checkpoint_update'], 'out-of-order, duplicate or wrong-policy shift')
            self._put(con, job, dict(record, status='CLAIMED'))

    def complete(self, job, dataset_ref, receipt):
        _complete_receipt(job, self.configs[job['phase_id']], dataset_ref, receipt)
        super().complete(job, dataset_ref, receipt)


def budget_snapshot():
    """Include the speed campaign when registered, plus every replica state."""
    result = preceding_snapshot()
    source = None
    for replica in REPLICAS:
        if not Path(locations(replica)['budget_path']).is_file():
            continue
        source = source or anchored_source()
        suite, phases = load_package(replica, source)
        assert_available([suite], result)
        current = ReplicaLedger(suite, phases).snapshot()
        result['jobs'].update(current['jobs'])
        result['campaigns'].update(current['campaigns'])
        result['maximum_total_shifts'] += CAP
        result['additional_reserved_shifts'] += CAP
    return result
