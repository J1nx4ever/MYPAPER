"""Merge structural ablation ledgers and enforce scoped reuse of evaluation seeds."""

from pathlib import Path

if __package__:
    from . import ablation_plan as plan
else:
    import ablation_plan as plan
from replica_budget import LOCKS, ReplicaLedger, budget_snapshot as preceding_snapshot
from business_graph_summary_campaign_v1.budget import connection, digest
from business_graph_summary_campaign_v1.launch import _launch_lock
from business_graph_summary_campaign_v1.plan_io import encoded, load_reference, load_suite, reference
from physical_curriculum_v1.budget import _history_rows
from python.offline_split import java_random_states


FORMAT = 'bg128-structural-ablation-budget-v1'
SEED_FIELDS = ('experiment_seed', 'actor_sampling_seed')
SCENE_FIELDS = ('scene_id', 'runtime_scene_sha256', 'expected_rpc_config', 'pair_ordinal')
UPDATES = plan.parent.UPDATES
PHASES = UPDATES + len(plan.parent.EVAL_CASES)


def _source_jobs(phases):
    return [job for phase in phases for job in phase['jobs']]


def _check_slots(suite, historical):
    unit = suite['unit']
    plan.parent.require(unit in plan.UNITS and suite['baseline_seed'] == unit.rsplit('-', 1)[1],
                        'structural ablation unit or matched baseline seed differs')
    layout = plan.locations(unit)
    jobs = suite['jobs']
    plan.parent.require(suite['campaign_id'] == layout['campaign_id'] and len(jobs) == plan.CAP,
                        'structural ablation scope differs')
    train, evaluate = jobs[:plan.TRAIN], jobs[plan.TRAIN:]
    source = historical[suite['baseline_seed']]
    train_source = _source_jobs(source['training'])
    eval_source = _source_jobs(historical['s01']['evaluation'])
    plan.parent.require(len(train_source) == plan.TRAIN and len(eval_source) == plan.EVAL,
                        'the frozen baseline has a different training or evaluation inventory')

    for ordinal, (job, original) in enumerate(zip(jobs, train_source + eval_source, strict=True)):
        training = ordinal < plan.TRAIN
        update = ordinal // plan.parent.BATCH_SIZE + 1 if training else UPDATES
        plan.parent.require(
            job['run_index'] == layout['run_index_base'] + ordinal and job['ordinal'] == ordinal
            and job['actor_seed'] == suite['actor_seed']
            and job['role'] == ('train' if training else 'final')
            and job['partition'] == ('train' if training else 'validation')
            and job['target_training_update'] == (update if training else 0)
            and job['checkpoint_update'] == (update - 1 if training else UPDATES)
            and all(job[key] == original[key] for key in SCENE_FIELDS),
            'ablation slot order, fresh policy identity or matched RPC scene differs')
    plan.parent.require(all(all(job[key] == original[key] for key in SEED_FIELDS)
                            for job, original in zip(evaluate, eval_source, strict=True)),
                        'evaluation seed reuse must match both frozen seeds of the exact original slot')

    phase_recordings = {job['phase_id']: job['recording_id'] for job in jobs}
    recordings = set(phase_recordings.values())
    plan.parent.require(len(phase_recordings) == len(recordings) == PHASES
                        and all(isinstance(recording, str) and recording for recording in recordings)
                        and all(job['recording_id'] == job['source_recording_id']
                                == phase_recordings[job['phase_id']] for job in jobs),
                        'every ablation phase requires an independent recording')
    return train, recordings


def assert_available(suites, previous):
    """Audit fresh physical identities and permit only explicitly paired evaluation seeds."""
    require = plan.parent.require
    require(previous['active_h2_reserved_shifts'] == 0, 'cancelled H2 reserve changed')
    history = _history_rows(previous)
    historical = plan.historical_sources()
    runs = {job.get('run_index') for job in history}
    recordings = {job.get(key) for job in history for key in ('recording_id', 'source_recording_id')}
    frozen_evaluation = _source_jobs(historical['s01']['evaluation'])
    unavailable = history + frozen_evaluation
    seeds = {job[key] & plan.parent.MAX_JAVA_SEED for job in unavailable
             for key in (*SEED_FIELDS, 'actor_seed') if type(job.get(key)) is int}
    states = {state for job in unavailable if type(job.get('experiment_seed')) is int
              for state in java_random_states(job['experiment_seed']).values()}
    campaigns = set(previous['campaigns'])
    run_count = seed_count = state_count = 0
    for suite in suites:
        train, new_recordings = _check_slots(suite, historical)
        campaign = suite['campaign_id']
        new_runs = {job['run_index'] for job in suite['jobs']}
        fresh = [suite['actor_seed'], *(job[key] for job in train for key in SEED_FIELDS)]
        require(campaign not in campaigns and len(new_runs) == plan.CAP and runs.isdisjoint(new_runs)
                and recordings.isdisjoint(new_recordings), 'ablation campaign, run or recording already occupied')
        require(len(set(fresh)) == len(fresh) and seeds.isdisjoint(fresh)
                and all(type(seed) is int and 0 < seed <= plan.parent.MAX_JAVA_SEED for seed in fresh),
                'ablation initialization or training seed collides with current or historical seeds')
        fresh_states = [state for job in train for state in java_random_states(job['experiment_seed']).values()]
        require(len(set(fresh_states)) == len(fresh_states) and states.isdisjoint(fresh_states),
                'ablation training Java engine or geology RNG state repeats')
        campaigns.add(campaign)
        runs.update(new_runs)
        recordings.update(new_recordings)
        seeds.update(fresh)
        states.update(fresh_states)
        run_count += len(new_runs)
        seed_count += len(fresh)
        state_count += len(fresh_states)
    return dict(occupied_before=previous['maximum_total_shifts'], new_runs_checked=run_count,
                new_initialization_and_training_seeds_checked=seed_count, new_java_rng_states_checked=state_count,
                projected_upper_bound=previous['maximum_total_shifts'] + run_count,
                historical_identities_checked=len(history))


def authority_path(suite):
    return Path(suite['python_output']) / 'launch/run_authority.json'


def authorization(suite):
    return dict(schema='bg128-structural-ablation-authority-v1', suite=reference(suite['suite_path']),
                campaign_id=suite['campaign_id'], unit=suite['unit'], baseline_seed=suite['baseline_seed'],
                training_shifts=plan.TRAIN, evaluation_shifts=plan.EVAL, maximum_new_shifts=plan.CAP,
                rpc_port=suite['rpc_port'], authorized_by_matching_anylogic_run=True)


class AblationLedger(ReplicaLedger):
    """Keep the serial collection contract and register against the complete merged history."""

    def __init__(self, suite, phases, *, path=None, approval=None):
        self.suite = suite
        self.campaign_id = suite['campaign_id']
        live_path = Path(suite['budget_path']).resolve()
        live_approval = authority_path(suite).resolve()
        self.path = live_path if path is None else Path(path).resolve()
        approval = live_approval if approval is None else Path(approval).resolve()
        plan.parent.require((path is None and approval == live_approval)
                            or (path is not None and self.path != live_path and approval != live_approval),
                            'test and production ablation ledger or authority paths overlap')
        auth_ref = reference(approval)
        authority = authorization(suite)
        plan.parent.require(load_reference(auth_ref) == authority, 'ablation launch authority differs')
        sealed, configs = load_suite(authority['suite']['path'])
        plan.parent.require(sealed == suite and configs == phases and reference(suite['suite_path']) == authority['suite'],
                            'ablation suite or phases do not match frozen bytes')

        self.entry = dict(format=FORMAT, cap=plan.CAP, test_only=path is not None,
                          suite_reference=authority['suite'], approval_reference=auth_ref)
        self.jobs = {job['run_index']: job for job in suite['jobs']}
        self.hashes = {run: digest(job) for run, job in self.jobs.items()}
        self.update_jobs = {cfg['arm_update']: cfg['jobs'] for cfg in phases if cfg['policy_arm'] == 'train'}
        self.configs = {cfg['phase_id']: ref for cfg, ref in zip(phases, suite['phase_configs'], strict=True)}

    def register(self, *, test_history=None):
        require = plan.parent.require
        if self.entry['test_only']:
            require(test_history is not None, 'isolated ablation registration requires test history')
            self._register(test_history)
        else:
            require(test_history is None, 'ablation registration must use current merged history')
            with _launch_lock(LOCKS / 'budget'):
                self._register(budget_snapshot())

    def _register(self, history):
        for key in ('suite_reference', 'approval_reference'):
            ref = self.entry[key]
            plan.parent.require(reference(ref['path']) == ref, 'ablation registration bytes changed: ' + key)
        assert_available([self.suite], history)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with connection(self.path) as con:
            con.execute('CREATE TABLE campaigns (id TEXT PRIMARY KEY,payload TEXT,checksum TEXT)')
            con.execute('CREATE TABLE slots (run INTEGER PRIMARY KEY,campaign TEXT,status TEXT,payload TEXT,checksum TEXT)')
            con.execute('CREATE TABLE commits (campaign TEXT,update_id INTEGER,payload TEXT,checksum TEXT,PRIMARY KEY(campaign,update_id))')
            con.execute('INSERT INTO campaigns VALUES (?,?,?)',
                        (self.campaign_id, encoded(self.entry).decode(), digest(self.entry)))
            for run, identity_sha256 in self.hashes.items():
                record = dict(status='PLANNED', identity_sha256=identity_sha256)
                con.execute('INSERT INTO slots VALUES (?,?,?,?,?)',
                            (run, self.campaign_id, 'PLANNED', encoded(record).decode(), digest(record)))

    def complete(self, job, dataset_ref, receipt):
        identity = receipt['identity']
        plan.parent.require(identity['actor_sampling_seed'] == job['actor_sampling_seed']
                            and identity['environment_seed'] == job['experiment_seed']
                            and identity['recording_id'] == job['recording_id']
                            and identity['run_index'] == job['run_index'],
                            'ablation receipt differs from its physical run, recording or seed pair')
        super().complete(job, dataset_ref, receipt)


def budget_snapshot():
    """Return all prior campaigns plus every registered ablation slot in any state."""
    result = preceding_snapshot()
    for unit in plan.UNITS:
        if not Path(plan.locations(unit)['budget_path']).is_file():
            continue
        suite, phases = plan.load_package(unit)
        assert_available([suite], result)
        current = AblationLedger(suite, phases).snapshot()
        result['jobs'].update(current['jobs'])
        result['campaigns'].update(current['campaigns'])
        result['maximum_total_shifts'] += plan.CAP
        result['additional_reserved_shifts'] += plan.CAP
    return result
