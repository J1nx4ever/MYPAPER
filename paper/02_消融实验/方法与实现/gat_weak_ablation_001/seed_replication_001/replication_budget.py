"""Budget GAT1 seed replications against all actual and prepared prior identities."""

import json
from pathlib import Path
import sys

PLAN = Path(__file__).resolve().parent
for directory in (PLAN.parents[4], PLAN.parent,
                  PLAN.parents[1] / 'baseline_comparison_001/dispatch_flow_matching_001'):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

if __package__:
    from . import replication_plan as plan
else:
    import replication_plan as plan

import flow_matching_campaign as flow
from weak_budget import (LOCKS, PROFILE_FIELDS, SCENE_FIELDS, SEED_FIELDS, WeakLedger,
                         _source_jobs, authority_path, authorization)
from weak_plan import bind, historical_sources as original_sources
from business_graph_summary_campaign_v1.contract import require
from physical_curriculum_v1.budget import _history_rows
from python.offline_split import JAVA_MASK, java_random_states


FORMAT = 'bg128-gat1-seed-replication-budget-v1'
IDENTITY_FIELDS = ('run_index', 'recording_id', 'source_recording_id', *SEED_FIELDS, 'actor_seed')


def _identity(job):
    return tuple(job.get(key) for key in IDENTITY_FIELDS)


def _prepared_rows(previous, history):
    """Reserve preassigned runs and seeds without adding them to actual consumption."""
    inherited = flow.v3_ledger._prepared_rows(previous)
    # Include superseded and still unregistered traditional packages, including
    # both flow-matching drafts. Those suites embed their whole job lists.
    for path in flow.PLAN.parent.rglob('suite.json'):
        document = json.loads(path.read_bytes())
        inherited.extend(document.get('jobs', []))
    used = {_identity(job) for job in history}
    unique = {_identity(job): job for job in inherited}
    return [job for identity, job in unique.items() if identity not in used]


def _check_slots(suite, source, evaluation):
    unit, baseline = suite['unit'], suite['baseline_seed']
    require(unit in plan.UNITS and unit == f'gat1-{baseline}',
            'GAT1 replication unit or matched training seed differs')
    layout = plan.locations(unit)
    jobs = suite['jobs']
    training = _source_jobs(source[baseline]['training'])
    require(suite['campaign_id'] == layout['campaign_id'] and len(jobs) == plan.CAP
            and len(training) == plan.TRAIN and len(evaluation) == plan.EVAL,
            'GAT1 replication inventory differs')
    for ordinal, (job, original) in enumerate(zip(jobs, training + evaluation, strict=True)):
        require(job['run_index'] == layout['run_index_base'] + ordinal and job['ordinal'] == ordinal
                and job['actor_seed'] == suite['actor_seed']
                and all(job[key] == original[key] for key in (*PROFILE_FIELDS, *SCENE_FIELDS)),
                'GAT1 replication slot order, policy identity or matched RPC scene differs')
        if ordinal >= plan.TRAIN:
            require(all(job[key] == original[key] for key in SEED_FIELDS),
                    'evaluation seed reuse must match both seeds of the exact original slot')
    phase_recordings = {job['phase_id']: job['recording_id'] for job in jobs}
    recordings = set(phase_recordings.values())
    require(len(phase_recordings) == len(recordings) == plan.parent.UPDATES + len(plan.parent.EVAL_CASES)
            and all(isinstance(recording, str) and recording for recording in recordings)
            and all(job['recording_id'] == job['source_recording_id'] == phase_recordings[job['phase_id']]
                    for job in jobs), 'every GAT1 replication phase requires an independent recording')
    return recordings


def assert_available(suites, previous):
    """Allow original evaluation pairs and require fresh training seeds and physical identity."""
    require(previous['active_h2_reserved_shifts'] == 0, 'cancelled H2 reserve changed')
    history = _history_rows(previous)
    prepared = _prepared_rows(previous, history)
    available_sources = plan.historical_sources()
    evaluation = _source_jobs(original_sources()['s01']['evaluation'])
    occupied = history + prepared
    runs = {job.get('run_index') for job in occupied}
    recordings = {job.get(key) for job in occupied for key in ('recording_id', 'source_recording_id')}
    unavailable = occupied + evaluation
    seeds = {job[key] & JAVA_MASK for job in unavailable
             for key in (*SEED_FIELDS, 'actor_seed') if type(job.get(key)) is int and job[key] > 0}
    states = {state for job in unavailable if type(job.get('experiment_seed')) is int and job['experiment_seed'] > 0
              for state in java_random_states(job['experiment_seed']).values()}
    campaigns = set(previous['campaigns'])
    run_count = seed_count = state_count = 0
    for suite in suites:
        new_recordings = _check_slots(suite, available_sources, evaluation)
        campaign = suite['campaign_id']
        jobs = suite['jobs']
        new_runs = {job['run_index'] for job in jobs}
        require(campaign not in campaigns and len(new_runs) == plan.CAP and runs.isdisjoint(new_runs)
                and recordings.isdisjoint(new_recordings), 'GAT1 replication campaign, run or recording already occupied')
        training = jobs[:plan.TRAIN]
        fresh = [suite['actor_seed'], *(job[key] for job in training for key in SEED_FIELDS)]
        require(all(type(seed) is int and 0 < seed <= JAVA_MASK for seed in fresh)
                and len(set(fresh)) == len(fresh) and seeds.isdisjoint(fresh),
                'GAT1 replication initialization or training seed collides')
        fresh_states = [state for job in training for state in java_random_states(job['experiment_seed']).values()]
        require(len(set(fresh_states)) == len(fresh_states) and states.isdisjoint(fresh_states),
                'GAT1 replication training Java engine or geology RNG state repeats')
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


def budget_snapshot():
    """Include all registered replication slots and every preceding physical budget."""
    result = flow.budget_snapshot()
    for unit in plan.UNITS:
        if not Path(plan.locations(unit)['budget_path']).is_file():
            continue
        suite, phases = plan.load_package(unit)
        assert_available([suite], result)
        current = ReplicationLedger(suite, phases).snapshot()
        result['jobs'].update(current['jobs'])
        result['campaigns'].update(current['campaigns'])
        result['maximum_total_shifts'] += plan.CAP
        result['additional_reserved_shifts'] += plan.CAP
    return result


class ReplicationLedger(WeakLedger):
    """Keep the same serial transactions and validate registration against this batch."""

    def __init__(self, suite, phases, *, path=None, approval=None):
        super().__init__(suite, phases, path=path, approval=approval)
        self.entry['format'] = FORMAT

    register = bind(WeakLedger.register, budget_snapshot=budget_snapshot)
    _register = bind(WeakLedger._register, assert_available=assert_available)
