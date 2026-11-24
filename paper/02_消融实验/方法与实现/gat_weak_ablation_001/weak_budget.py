"""Keep fresh physical identities while reusing the frozen mixed evaluation slots."""

from pathlib import Path
import sys

PLAN = Path(__file__).resolve().parent
for directory in (PLAN.parents[3], PLAN.parent / 'structural_ablation_001',
                  PLAN.parent / 'baseline_comparison_001/all_recheck_001'):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

if __package__:
    from . import weak_plan as plan
else:
    import weak_plan as plan

from ablation_budget import (AblationLedger, LOCKS, SCENE_FIELDS, SEED_FIELDS, _source_jobs,
                             authority_path, authorization)
from business_graph_summary_campaign_v1.budget import connection, digest
from business_graph_summary_campaign_v1.contract import require
from business_graph_summary_campaign_v1.launch import _launch_lock
from business_graph_summary_campaign_v1.plan_io import encoded, reference
from change_plans.token_attention_001.business_graph_summary_009.generalization_001.speed_congestion_001 import speed_plan
from physical_curriculum_v1.budget import _history_rows
from python.offline_split import JAVA_MASK, java_random_states
from recheck_budget import budget_snapshot as preceding_snapshot
import paired_plan
import replica_plan


FORMAT = 'bg128-gat-weak-ablation-budget-v1'
PROFILE_FIELDS = ('role', 'partition', 'target_training_update', 'checkpoint_update', 'phase_id')


def _prepared_rows(previous):
    """Protect frozen identities that are available to start but have no ledger yet."""
    campaigns = previous['campaigns']
    rows = []
    if paired_plan.CAMPAIGN not in campaigns and paired_plan.SUITE.is_file():
        rows.extend(paired_plan.load_suite(check_sources=False)['jobs'])
    if speed_plan.CAMPAIGN not in campaigns and speed_plan.SUITE.is_file():
        suite, _ = speed_plan.load_package()
        rows.extend(suite['jobs'])
    for replica in replica_plan.REPLICAS:
        layout = replica_plan.locations(replica)
        if layout['campaign_id'] not in campaigns and Path(layout['suite_path']).is_file():
            suite, _ = replica_plan.load_package(replica)
            rows.extend(suite['jobs'])
    return rows


def _check_slots(suite, training, evaluation):
    unit = suite['unit']
    require(unit in plan.UNITS and suite['baseline_seed'] == 's01', 'GAT weak ablation unit or baseline differs')
    layout = plan.locations(unit)
    jobs = suite['jobs']
    require(suite['campaign_id'] == layout['campaign_id'] and len(jobs) == plan.CAP
            and len(training) == plan.TRAIN and len(evaluation) == plan.EVAL,
            'GAT weak ablation inventory differs')
    for ordinal, (job, source) in enumerate(zip(jobs, training + evaluation, strict=True)):
        require(job['run_index'] == layout['run_index_base'] + ordinal and job['ordinal'] == ordinal
                and job['actor_seed'] == suite['actor_seed']
                and all(job[key] == source[key] for key in (*PROFILE_FIELDS, *SCENE_FIELDS)),
                'GAT weak ablation slot order, policy identity or matched RPC scene differs')
        if ordinal >= plan.TRAIN:
            require(all(job[key] == source[key] for key in SEED_FIELDS),
                    'evaluation seed reuse must match both seeds of the exact original slot')

    phase_recordings = {job['phase_id']: job['recording_id'] for job in jobs}
    recordings = set(phase_recordings.values())
    require(len(phase_recordings) == len(recordings) == plan.parent.UPDATES + len(plan.parent.EVAL_CASES)
            and all(isinstance(recording, str) and recording for recording in recordings)
            and all(job['recording_id'] == job['source_recording_id'] == phase_recordings[job['phase_id']]
                    for job in jobs), 'every GAT weak ablation phase requires an independent recording')
    return recordings


def assert_available(suites, previous):
    """Validate run, recording, initialization and training seeds against the full history."""
    require(previous['active_h2_reserved_shifts'] == 0, 'cancelled H2 reserve changed')
    source = plan.historical_sources()['s01']
    training = _source_jobs(source['training'])
    evaluation = _source_jobs(source['evaluation'])
    history = _history_rows(previous)
    prepared = _prepared_rows(previous)
    tracked = history + prepared
    runs = {job.get('run_index') for job in tracked}
    recordings = {job.get(key) for job in tracked for key in ('recording_id', 'source_recording_id')}
    unavailable = tracked + evaluation
    seeds = {job[key] & JAVA_MASK for job in unavailable
             for key in (*SEED_FIELDS, 'actor_seed') if type(job.get(key)) is int and job[key] > 0}
    states = {state for job in unavailable if type(job.get('experiment_seed')) is int and job['experiment_seed'] > 0
              for state in java_random_states(job['experiment_seed']).values()}
    campaigns = set(previous['campaigns'])
    run_count = seed_count = state_count = 0

    for suite in suites:
        new_recordings = _check_slots(suite, training, evaluation)
        campaign = suite['campaign_id']
        jobs = suite['jobs']
        new_runs = {job['run_index'] for job in jobs}
        require(campaign not in campaigns and len(new_runs) == plan.CAP and runs.isdisjoint(new_runs)
                and recordings.isdisjoint(new_recordings), 'GAT weak ablation campaign, run or recording already occupied')
        train = jobs[:plan.TRAIN]
        fresh = [suite['actor_seed'], *(job[key] for job in train for key in SEED_FIELDS)]
        require(all(type(seed) is int and 0 < seed <= JAVA_MASK for seed in fresh)
                and len(set(fresh)) == len(fresh) and seeds.isdisjoint(fresh),
                'GAT weak ablation initialization or training seed collides')
        fresh_states = [state for job in train for state in java_random_states(job['experiment_seed']).values()]
        require(len(set(fresh_states)) == len(fresh_states) and states.isdisjoint(fresh_states),
                'GAT weak ablation training Java engine or geology RNG state repeats')
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
                historical_identities_checked=len(history), prepared_identities_checked=len(prepared))


class WeakLedger(AblationLedger):
    """Register this batch against the latest history using the ablation slot state machine."""

    def __init__(self, suite, phases, *, path=None, approval=None):
        super().__init__(suite, phases, path=path, approval=approval)
        self.entry['format'] = FORMAT

    def register(self, *, test_history=None):
        if self.entry['test_only']:
            require(test_history is not None, 'isolated GAT weak ablation registration requires test history')
            self._register(test_history)
        else:
            require(test_history is None, 'GAT weak ablation registration must use current merged history')
            with _launch_lock(LOCKS / 'budget'):
                self._register(budget_snapshot())

    def _register(self, history):
        for key in ('suite_reference', 'approval_reference'):
            ref = self.entry[key]
            require(reference(ref['path']) == ref, 'GAT weak ablation registration bytes changed: ' + key)
        assert_available([self.suite], history)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with connection(self.path) as con:
            con.execute('CREATE TABLE campaigns (id TEXT PRIMARY KEY,payload TEXT,checksum TEXT)')
            con.execute('CREATE TABLE slots (run INTEGER PRIMARY KEY,campaign TEXT,status TEXT,payload TEXT,checksum TEXT)')
            con.execute('CREATE TABLE commits (campaign TEXT,update_id INTEGER,payload TEXT,checksum TEXT,PRIMARY KEY(campaign,update_id))')
            con.execute('INSERT INTO campaigns VALUES (?,?,?)',
                        (self.campaign_id, encoded(self.entry).decode(), digest(self.entry)))
            for run, identity_sha256 in self.hashes.items():
                slot = dict(status='PLANNED', identity_sha256=identity_sha256)
                con.execute('INSERT INTO slots VALUES (?,?,?,?,?)',
                            (run, self.campaign_id, 'PLANNED', encoded(slot).decode(), digest(slot)))


def budget_snapshot():
    """Merge every traditional baseline and structural ablation plus registered GAT weak slots."""
    result = preceding_snapshot()
    for unit in plan.UNITS:
        if not Path(plan.locations(unit)['budget_path']).is_file():
            continue
        suite, phases = plan.load_package(unit)
        assert_available([suite], result)
        current = WeakLedger(suite, phases).snapshot()
        result['jobs'].update(current['jobs'])
        result['campaigns'].update(current['campaigns'])
        result['maximum_total_shifts'] += plan.CAP
        result['additional_reserved_shifts'] += plan.CAP
    return result
