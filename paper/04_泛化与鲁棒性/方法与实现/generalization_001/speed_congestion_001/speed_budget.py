"""Fixed-policy speed-congestion slots and the merged budget after registration."""

from importlib import import_module
from pathlib import Path
import sys


PLAN = Path(__file__).resolve().parent
ROOT = PLAN.parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from business_graph_summary_campaign_v1.budget import (  # noqa: E402
    BudgetLedger, _payload, _state, connection, digest,
)
from business_graph_summary_campaign_v1.contract import require  # noqa: E402
from business_graph_summary_campaign_v1.plan_io import (  # noqa: E402
    encoded, load_phase, load_reference, reference,
)
from change_plans.token_attention_001.business_graph_summary_009.training_extension_001.extension_budget import (  # noqa: E402
    _complete_receipt,
)
from physical_curriculum_v1.budget import _history_rows  # noqa: E402
from python.offline_split import java_random_states  # noqa: E402


CAMPAIGN = 'bg128-speed-congestion-20261120-001'
FORMAT = 'bg128-speed-congestion-budget-v1'
PREVIOUS_CAP = 25636
FIRST_RUN = 10002426
PAIR_COUNT = 20
SPEED_STDS = (.15, .25)
CAP = PAIR_COUNT * len(SPEED_STDS)
DEFAULT_LEDGER = ROOT / 'runs' / CAMPAIGN / 'budget.sqlite'
JAVA_MASK = (1 << 48) - 1
SEED_FIELDS = ('experiment_seed', 'actor_sampling_seed')
SPEED_FIELD = 'config.runtime_scene.env.freeSpeedFactorStd'
PARENT_CAMPAIGN = 'bg128-continuous-mixed-20261119-001'
PARENT_CHECKPOINT_SHA = 'b8b81c0821535cdfdff3c9352c11d69bc91d14fbe97694ab5f45c08fb97c3c9a'
PARENT_ACTOR_SEED = 98044477675931


def _plan():
    return import_module('.speed_plan', __package__) if __package__ else import_module('speed_plan')


def preceding_snapshot():
    from change_plans.token_attention_001.business_graph_summary_009.generalization_001.grade_assignment_generalization_001.continuous_training_001.recovery_003.accounting.budget import (  # noqa: E501
        budget_snapshot as previous_budget_snapshot,
    )

    return previous_budget_snapshot()


def _jobs(suite):
    jobs = suite['jobs']
    require(suite['campaign_id'] == CAMPAIGN and len(jobs) == CAP,
            'speed evaluation requires exactly forty frozen physical slots')
    phase_ids, recordings = set(), set()
    for phase, speed_std in enumerate(SPEED_STDS):
        block = jobs[phase * PAIR_COUNT:(phase + 1) * PAIR_COUNT]
        phase_id, recording = block[0]['phase_id'], block[0]['recording_id']
        require(isinstance(phase_id, str) and bool(phase_id) and phase_id not in phase_ids
                and isinstance(recording, str) and bool(recording) and recording not in recordings,
                'each speed phase requires a distinct phase and recording identity')
        phase_ids.add(phase_id)
        recordings.add(recording)
        for pair_ordinal, job in enumerate(block):
            ordinal = phase * PAIR_COUNT + pair_ordinal
            require(all(type(job[key]) is int for key in ('ordinal', 'pair_ordinal', 'run_index'))
                    and (job['ordinal'], job['pair_ordinal'], job['run_index']) ==
                    (ordinal, pair_ordinal, FIRST_RUN + ordinal)
                    and job['phase_id'] == phase_id
                    and job['recording_id'] == job['source_recording_id'] == recording
                    and job['candidate_id'] == 's01_BG128'
                    and job['role'] == 'final' and job['partition'] == 'validation'
                    and job['checkpoint_update'] == 32 and job['target_training_update'] == 0
                    and job['actor_seed'] == PARENT_ACTOR_SEED
                    and job['expected_rpc_config'].get(SPEED_FIELD) == speed_std,
                    'speed slot, checkpoint, or condition differs from the frozen evaluation scope')
    return jobs


def check_collisions(suite, before=None):
    """Return the prior merged view after validating all new slot identities."""
    jobs = _jobs(suite)
    previous = preceding_snapshot() if before is None else before
    parent = previous['campaigns'].get(PARENT_CAMPAIGN, {})
    require(previous['maximum_total_shifts'] == PREVIOUS_CAP
            and previous['active_h2_reserved_shifts'] == 0
            and CAMPAIGN not in previous['campaigns']
            and parent.get('commits', {}).get('32', {}).get('checkpoint_ref', {}).get('sha256')
            == PARENT_CHECKPOINT_SHA,
            'completed mixed u32 or preceding merged capacity differs')
    history = _history_rows(previous)
    for key in ('run_index', 'recording_id', 'source_recording_id'):
        occupied = {row.get(key) for row in history}
        require(all(job[key] not in occupied for job in jobs), 'historical identity collision: ' + key)

    pairs = [tuple(job[key] for key in SEED_FIELDS) for job in jobs]
    require(pairs[:PAIR_COUNT] == pairs[PAIR_COUNT:],
            'the two speed conditions must share both seeds of the corresponding slot')
    new_seeds = [value for pair in pairs[:PAIR_COUNT] for value in pair]
    used_seeds = {row[key] & JAVA_MASK for row in history for key in (*SEED_FIELDS, 'actor_seed')
                  if type(row.get(key)) is int}
    require(all(type(value) is int and 0 < value <= JAVA_MASK for value in new_seeds)
            and len(set(new_seeds)) == 2 * PAIR_COUNT
            and PARENT_ACTOR_SEED not in new_seeds and used_seeds.isdisjoint(new_seeds),
            'new environment or Actor sampling seed repeats another seed or history')

    used_java = {value for row in history if type(row.get('experiment_seed')) is int
                 for value in java_random_states(row['experiment_seed']).values()}
    new_java = [value for env, _ in pairs[:PAIR_COUNT] for value in java_random_states(env).values()]
    require(len(set(new_java)) == len(new_java) == 5 * PAIR_COUNT and used_java.isdisjoint(new_java),
            'new Java engine or site RNG state repeats history or another slot')
    return previous


def _ledger_path(plan):
    path = Path(getattr(plan, 'LEDGER', Path(plan.OUTPUT) / DEFAULT_LEDGER.name)).resolve()
    require(path.parent == Path(plan.OUTPUT).resolve(), 'speed ledger must be under its own output directory')
    return path


class SpeedLedger(BudgetLedger):
    """Register and advance the serial forty-slot evaluation without training commits."""

    def __init__(self, suite, configs, authority_path, *, path=None):
        plan = _plan()
        live_path = _ledger_path(plan)
        authority_path = Path(authority_path).resolve()
        canonical_authority = Path(plan.AUTHORITY).resolve()
        self.path = live_path if path is None else Path(path).resolve()
        require((path is None and authority_path == canonical_authority)
                or (path is not None and self.path != live_path and authority_path != canonical_authority),
                'test and production speed authority or ledger paths overlap')
        if 'budget_path' in suite:
            require(Path(suite['budget_path']).resolve() == live_path, 'frozen speed budget path differs')

        jobs = _jobs(suite)
        suite_ref = reference(plan.SUITE)
        sealed = load_reference(suite_ref)
        require({key: value for key, value in suite.items() if key != 'jobs'}
                == {key: value for key, value in sealed.items() if key != 'jobs'}
                and sealed.get('jobs', jobs) == jobs,
                'in-memory speed evaluation suite differs from frozen bytes')
        config_refs = suite['phase_configs']
        require(len(configs) == len(config_refs) == len(SPEED_STDS)
                and [job for cfg in configs for job in cfg['jobs']] == jobs,
                'speed phase configs disagree with the registered jobs')
        self.configs = {}
        for cfg, ref in zip(configs, config_refs, strict=True):
            require(reference(ref['path']) == ref and load_phase(ref['path']) == cfg
                    and cfg['campaign_id'] == CAMPAIGN and len(cfg['jobs']) == PAIR_COUNT
                    and all(job['phase_id'] == cfg['phase_id'] and job['recording_id'] == cfg['recording_id']
                            for job in cfg['jobs']), 'speed phase byte or recording identity differs')
            self.configs[cfg['phase_id']] = ref
        require(len(self.configs) == len(SPEED_STDS), 'speed phase identities are not distinct')
        approval_ref = reference(authority_path)
        require(load_reference(approval_ref) == plan.authorization_for(suite), 'speed launch authority differs')

        self.suite = suite
        self.campaign_id = CAMPAIGN
        self.jobs = {job['run_index']: job for job in jobs}
        self.hashes = {run: digest(job) for run, job in self.jobs.items()}
        self.update_jobs = {}
        self.entry = dict(format=FORMAT, cap=CAP, test_only=path is not None,
                          suite_reference=suite_ref, approval_reference=approval_ref)

    def _validated_state(self, con):
        registered = con.execute('SELECT id,payload,checksum FROM campaigns').fetchall()
        require(len(registered) == 1 and registered[0][0] == CAMPAIGN
                and _payload(registered[0][1:]) == self.entry
                and con.execute('SELECT COUNT(*) FROM slots').fetchone()[0] == CAP
                and con.execute('SELECT 1 FROM commits LIMIT 1').fetchone() is None,
                'speed registration, slot inventory, or zero-optimization contract changed')
        state = _state(con, self.suite, self.entry, self.hashes)
        waiting = False
        for run, job in self.jobs.items():
            record = state['jobs'][str(run)]
            status = record['status']
            require((not waiting or status == 'PLANNED') and 'committed_update' not in record,
                    'speed slots advanced out of order or contain a training commit')
            if status == 'COMPLETE':
                self._verify_completion(job, record['dataset_ref'], record['receipt'])
            waiting = waiting or status != 'COMPLETE'
        return state

    def register(self):
        """Preserve an existing registration and allocate only the authorized slots."""
        for name in ('suite_reference', 'approval_reference'):
            ref = self.entry[name]
            require(reference(ref['path']) == ref, 'speed registration bytes changed: ' + name)
        check_collisions(self.suite)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with connection(self.path) as con:
            con.execute('CREATE TABLE IF NOT EXISTS campaigns (id TEXT PRIMARY KEY,payload TEXT,checksum TEXT)')
            con.execute('CREATE TABLE IF NOT EXISTS slots (run INTEGER PRIMARY KEY,campaign TEXT,status TEXT,payload TEXT,checksum TEXT)')
            con.execute('CREATE TABLE IF NOT EXISTS commits (campaign TEXT,update_id INTEGER,payload TEXT,checksum TEXT,PRIMARY KEY(campaign,update_id))')
            if con.execute('SELECT 1 FROM campaigns LIMIT 1').fetchone() is not None:
                self._validated_state(con)
                return
            require(con.execute('SELECT 1 FROM slots LIMIT 1').fetchone() is None
                    and con.execute('SELECT 1 FROM commits LIMIT 1').fetchone() is None,
                    'orphan speed slots or commits precede registration')
            con.execute('INSERT INTO campaigns VALUES (?,?,?)',
                        (CAMPAIGN, encoded(self.entry).decode(), digest(self.entry)))
            rows = []
            for run, identity_sha256 in self.hashes.items():
                record = dict(status='PLANNED', identity_sha256=identity_sha256)
                rows.append((run, CAMPAIGN, 'PLANNED', encoded(record).decode(), digest(record)))
            con.executemany('INSERT INTO slots VALUES (?,?,?,?,?)', rows)

    def snapshot(self):
        with connection(self.path, readonly=True) as con:
            con.execute('BEGIN')
            return self._validated_state(con)

    def claim(self, job):
        with connection(self.path) as con:
            record = self._row(con, job)
            pending = con.execute(
                "SELECT 1 FROM slots WHERE campaign=? AND run < ? AND status != 'COMPLETE' LIMIT 1",
                (CAMPAIGN, job['run_index']),
            ).fetchone()
            require(record['status'] == 'PLANNED' and pending is None,
                    'slot cannot be claimed twice or before an earlier slot completes')
            record['status'] = 'CLAIMED'
            self._put(con, job, record)

    def _verify_completion(self, job, dataset_ref, receipt):
        _complete_receipt(job, self.configs[job['phase_id']], dataset_ref, receipt)
        identity = receipt['identity']
        require(identity['actor_sampling_seed'] == job['actor_sampling_seed']
                and identity['environment_seed'] == job['experiment_seed']
                and identity['run_index'] == job['run_index']
                and identity['recording_id'] == job['recording_id'],
                'speed completion receipt does not bind both seeds and its recording')

    def complete(self, job, dataset_ref, receipt):
        self._verify_completion(job, dataset_ref, receipt)
        super().complete(job, dataset_ref, receipt)

    def commit(self, update, jobs, checkpoint_ref, batch_id):
        require(False, 'fixed speed evaluation cannot commit a training update')


def budget_snapshot():
    """Read the latest historical budgets plus every registered speed slot."""
    previous = preceding_snapshot()
    plan = _plan()
    path = _ledger_path(plan)
    if not path.is_file():
        return previous
    suite, configs = plan.load_package()
    check_collisions(suite, previous)
    current = SpeedLedger(suite, configs, plan.AUTHORITY).snapshot()
    previous['jobs'].update(current['jobs'])
    previous['campaigns'].update(current['campaigns'])
    previous['maximum_total_shifts'] += CAP
    previous['additional_reserved_shifts'] += CAP
    return previous
