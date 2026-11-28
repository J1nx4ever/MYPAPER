"""Account for matched, physical u32 inference without authorizing optimizer steps."""
from pathlib import Path

from . import prepare as plan
from .. import budget as previous

from business_graph_summary_campaign_v1.budget import connection, digest
from business_graph_summary_campaign_v1.launch import _launch_lock
from business_graph_summary_campaign_v1.plan_io import load_reference, reference
from replica_budget import ReplicaLedger
from weak_plan import bind

require = plan.require
LOCKS = previous.LOCKS
preceding_snapshot = previous.budget_snapshot
authority_path = previous.authority_path
rows_for_collision = bind(previous._rows, plan=plan, _prior_rows=bind(previous._prior_rows, SUITE=plan.SUITE))


def assert_available(suite, history):
    require(suite['campaign_id'] == plan.CAMPAIGN and plan.CAMPAIGN not in history['campaigns']
            and history['active_h2_reserved_shifts'] == 0 and len(suite['jobs']) == plan.COUNT
            and plan.training.CAMPAIGN in history['campaigns'], 'paired inference or preceding budget differs')
    _, phases, _ = plan.historical()
    originals = [job for phase in phases for job in phase['jobs']]
    jobs, recordings = suite['jobs'], set()
    for i, (job, old) in enumerate(zip(jobs, originals, strict=True)):
        prior = history['jobs'][str(old['run_index'])]
        require(prior['status'] == 'COMPLETE' and prior['identity'] == old
                and prior['receipt']['identity']['actor_sampling_seed'] == job['actor_sampling_seed']
                and prior['receipt']['identity']['environment_seed'] == job['experiment_seed']
                and all(old[key] == job[key] for key in plan.PAIR_KEYS),
                'source physical slot, scene, role or paired seed differs')
        require(job['ordinal'] == i and job['run_index'] == plan.FIRST_RUN+i
                and job['actor_seed'] == suite['actor_seed'] and job['role'] == 'final'
                and job['recording_id'] == job['source_recording_id'] == plan.CAMPAIGN + '-' + job['phase_id'],
                'new physical identity differs')
        recordings.add(job['recording_id'])
    require(len(recordings) == len(phases) and len({job['run_index'] for job in jobs}) == plan.COUNT,
            'new run or phase recordings repeat')
    pairs = [(job['experiment_seed'], job['actor_sampling_seed']) for job in jobs[:plan.PER_CASE]]
    require(len(set(pairs)) == len({x[0] for x in pairs}) == len({x[1] for x in pairs}) == plan.PER_CASE
            and len({seed for pair in pairs for seed in pair}) == 2 * plan.PER_CASE
            and all((job['experiment_seed'],job['actor_sampling_seed']) == pairs[job['pair_ordinal']] for job in jobs),
            'pairing must reproduce the common historical block of 100 stochastic slots')
    rows = rows_for_collision(history)
    runs = {row.get('run_index') for row in rows}
    prior_recordings = {row.get(key) for row in rows for key in ('recording_id', 'source_recording_id')}
    require(runs.isdisjoint(job['run_index'] for job in jobs) and recordings.isdisjoint(prior_recordings),
            'new physical run or recording already exists')
    before = history['maximum_total_shifts']
    return dict(registered_before=before, additional_if_started=plan.COUNT, total_if_started=before+plan.COUNT,
                actual_and_prepared_rows_checked=len(rows), physically_verified_s02_slots=len(originals),
                reused_slot_seed_pairs=len(pairs), new_recordings=len(recordings))


def authorization(suite):
    return dict(schema='bg128-frozen-reinforce-s02-paired-evaluation-authority-v1',
                suite=reference(suite['suite_path']), campaign_id=plan.CAMPAIGN, frozen_actor=suite['frozen_actor'],
                training_shifts=0, evaluation_shifts=plan.COUNT, maximum_new_shifts=plan.COUNT,
                rpc_port=suite['rpc_port'])


class InferenceLedger(previous.ReinforceLedger):
    def __init__(self, suite, phases):
        self.suite = suite
        self.path = Path(suite['budget_path'])
        self.campaign_id = suite['campaign_id']
        authority = authorization(suite)
        approval = reference(authority_path(suite))
        require(load_reference(approval) == authority, 'inference authority differs')
        self.entry = dict(format='bg128-reinforce-fixed-paired-inference-budget-v1', cap=plan.COUNT,
                          test_only=False, suite_reference=authority['suite'], approval_reference=approval)
        self.jobs = {job['run_index']: job for job in suite['jobs']}
        self.hashes = {run: digest(job) for run, job in self.jobs.items()}
        self.update_jobs = {}
        self.configs = {cfg['phase_id']: ref for cfg, ref in zip(phases, suite['phase_configs'], strict=True)}

    def register(self):
        with _launch_lock(previous.SHARED_BUDGET_LOCK):
            require(not self.path.exists(), 'inference has already registered')
            for key in ('approval_reference', 'suite_reference'):
                ref = self.entry[key]
                require(reference(ref['path']) == ref, 'pre-registration bytes differ')
            register = bind(ReplicaLedger._register, assert_available=lambda suites, state: assert_available(suites[0], state))
            register(self, preceding_snapshot())

    snapshot = bind(previous.ReinforceLedger.snapshot, plan=plan)

    def claim(self, job):
        with connection(self.path) as con:
            row = self._row(con, job)
            earlier = con.execute("SELECT 1 FROM slots WHERE run<? AND status!='COMPLETE' LIMIT 1",
                                  (job['run_index'],)).fetchone()
            require(row['status'] == 'PLANNED' and earlier is None
                    and con.execute('SELECT COUNT(*) FROM commits').fetchone()[0] == 0,
                    'duplicate, out-of-order or non-frozen inference claim')
            self._put(con, job, dict(row, status='CLAIMED'))

    def complete(self, job, dataset_ref, receipt):
        require(receipt['identity']['policy_sha256'] == self.suite['frozen_actor']['sha256'],
                'inference receipt belongs to another Actor')
        super().complete(job, dataset_ref, receipt)

    def commit(self, *args, **kwargs):
        raise ValueError('evaluation-only ledger permits zero updates')


def budget_snapshot():
    snapshot = preceding_snapshot()
    if plan.LEDGER.is_file():
        suite, phases = plan.load_package()
        assert_available(suite, snapshot)
        state = InferenceLedger(suite, phases).snapshot()
        snapshot['campaigns'].update(state['campaigns'])
        snapshot['jobs'].update(state['jobs'])
        snapshot['maximum_total_shifts'] += plan.COUNT
        snapshot['additional_reserved_shifts'] += plan.COUNT
    return snapshot
