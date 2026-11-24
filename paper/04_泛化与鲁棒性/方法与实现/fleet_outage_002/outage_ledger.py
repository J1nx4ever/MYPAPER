"""Append the twenty paired fleet outage evaluations to the merged ledger."""
import json
from pathlib import Path
import sys

from outage_plan import (PRIOR, CAMPAIGN, COUNT, SLOTS, OUTPUT, SUITE, encoded, load_reference,
                         load_suite, previous, reference, require, validate_jobs)

sys.path.insert(0, str(PRIOR))
try:
    import fleet_ledger as previous_ledger
finally:
    sys.path.remove(str(PRIOR))

import formal_ledger
from business_graph_summary_campaign_v1.launch import _launch_lock
from online_v1_1.operations import _publish
from physical_curriculum_v1.budget import _history_rows
from python.offline_split import JAVA_MASK, java_random_states


SCHEMA = 'bg128-fleet-outage-comparison-budget-v1'
LOCKS = previous_ledger.LOCKS
preceding_snapshot = previous_ledger.budget_snapshot
LEDGER = OUTPUT/'budget.json'
AUTHORITY = OUTPUT/'launch/run_authority.json'
STATUSES = ('COMPLETE', 'CLAIMED', 'UNKNOWN', 'PLANNED')
SEED_FIELDS = ('experiment_seed', 'actor_sampling_seed', 'actor_seed')


def check_fresh(suite, historical):
    require(CAMPAIGN not in historical['campaigns'] and historical['active_h2_reserved_shifts'] == 0,
            'revised validation campaign already registered or previous budget scope differs')
    jobs = suite['jobs']
    validate_jobs(jobs)
    history = _history_rows(historical)
    pending = formal_ledger._prepared_rows(historical)
    if previous.nominal.CAMPAIGN not in historical['campaigns'] and previous.nominal.SUITE.is_file():
        pending += previous.nominal.load_suite(check_sources=False)['jobs']
    if previous.CAMPAIGN not in historical['campaigns'] and previous.SUITE.is_file():
        pending += previous.load_suite(check_sources=False)['jobs']
    rows = [*history, *pending]
    runs = {row.get('run_index') for row in rows}
    recordings = {row.get(field) for row in rows for field in ('recording_id', 'source_recording_id')}
    require(all(job['run_index'] not in runs and job['recording_id'] not in recordings for job in jobs),
            'physical run or recording overlaps historical or already prepared identity')
    seeds = {row[key] & JAVA_MASK for row in rows for key in SEED_FIELDS if type(row.get(key)) is int and row[key] > 0}
    environments = {job['experiment_seed'] for job in jobs}
    sampling = {job['actor_sampling_seed'] for job in jobs}
    lineage = {job['actor_seed'] & JAVA_MASK for job in jobs}
    require((environments|sampling).isdisjoint(seeds|lineage),
            'new environment or private Actor sampling seed repeats an earlier seed')
    prior_java = {state for row in rows if type(row.get('experiment_seed')) is int and row['experiment_seed'] > 0
                  for state in java_random_states(row['experiment_seed']).values()}
    states = {value for environment in environments for value in java_random_states(environment).values()}
    require(len(states) == SLOTS*5 and prior_java.isdisjoint(states), 'new effective Java RNG state overlaps an earlier state')
    occupied = historical['maximum_total_shifts']
    return dict(occupied_before=occupied, maximum_total_if_registered=occupied+COUNT,
                historical_identities_checked=len(history), prepared_identities_checked=len(pending),
                prior_single_roster_pending=previous.COUNT if previous.CAMPAIGN not in historical['campaigns'] else 0,
                new_physical_jobs=COUNT, new_environment_seeds=len(environments), new_actor_sampling_seeds=len(sampling),
                new_java_rng_states=len(states))


def authority_for(suite):
    job = suite['jobs'][0]
    return dict(schema=SCHEMA+'-authority', campaign_id=CAMPAIGN, suite=reference(SUITE),
                authorized_by_matching_anylogic_run=True, maximum_new_shifts=COUNT,
                first_run={key: job[key] for key in ('run_index', 'experiment_seed', 'recording_id')})


class Ledger:
    """Write the reviewed jobs in order, preserving any unresolved physical start."""

    def __init__(self, suite):
        self.suite = suite
        self.jobs = {str(job['run_index']):job for job in suite['jobs']}

    def register(self):
        with _launch_lock(LOCKS/'budget'):
            require(not LEDGER.exists() and load_reference(reference(AUTHORITY)) == authority_for(self.suite),
                    'existing revised ledger or mismatched launch authority')
            occupied = check_fresh(self.suite, preceding_snapshot())['occupied_before']
            state = dict(schema=SCHEMA, suite=reference(SUITE), authority=reference(AUTHORITY), occupied_before=occupied,
                         slots={run:dict(identity=job, status='PLANNED') for run, job in self.jobs.items()})
            _publish(LEDGER, encoded(state))

    def snapshot(self):
        raw = LEDGER.read_bytes()
        state = json.loads(raw)
        require(raw == encoded(state) and state['schema'] == SCHEMA and state['suite'] == reference(SUITE)
                and state['authority'] == reference(AUTHORITY)
                and load_reference(state['authority']) == authority_for(self.suite)
                and set(state['slots']) == set(self.jobs), 'revised ledger registration or authority differs')
        statuses = []
        for run, job in self.jobs.items():
            slot = state['slots'][run]
            require(slot['identity'] == job and slot['status'] in STATUSES, 'revised ledger slot identity differs')
            if slot['status'] == 'COMPLETE':
                require(reference(slot['receipt']['path']) == slot['receipt'], 'revised completion receipt bytes differ')
            statuses.append(slot['status'])
        require(statuses == sorted(statuses, key=STATUSES.index)
                and sum(status in ('CLAIMED', 'UNKNOWN') for status in statuses) <= 1,
                'revised ledger physical order differs')
        return state

    def _transition(self, job, before, after, **evidence):
        state = self.snapshot()
        run = str(job['run_index'])
        require(self.jobs.get(run) == job and state['slots'][run]['status'] == before,
                'revised physical slot already used or out of order')
        if after == 'CLAIMED':
            earlier = self.suite['jobs'][:job['ordinal']]
            require(all(state['slots'][str(row['run_index'])]['status'] == 'COMPLETE' for row in earlier),
                    'an earlier physical job is incomplete or unresolved')
        state['slots'][run] = dict(identity=job, status=after, **evidence)
        _publish(LEDGER, encoded(state), replace=True)

    def claim(self, job):
        self._transition(job, 'PLANNED', 'CLAIMED')

    def complete(self, job, receipt):
        require(receipt['path'] == str(Path(job['python_output'])/'evaluation_receipt.json'),
                'receipt must match the frozen physical output path')
        payload = load_reference(receipt)
        for key in ('run_index', 'recording_id', 'experiment_seed', 'actor_sampling_seed', 'policy_kind',
                    'phase_config', 'pair_ordinal', 'scene_id'):
            require(payload[key] == job[key], 'receipt differs from launched validation job: ' + key)
        require(payload['status'] == 'COMPLETE' and payload['normal_completion'] is True
                and payload['completed_shifts'] == 1 and payload['optimizer_updates'] == 0,
                'complete evaluation-only physical receipt required')
        self._transition(job, 'CLAIMED', 'COMPLETE', receipt=receipt)

    def mark_unknown(self, job, error):
        self._transition(job, 'CLAIMED', 'UNKNOWN', reason=str(error))


def budget_snapshot():
    """Merge both fleet campaigns and all earlier registered evaluations and training."""
    result = preceding_snapshot()
    if not LEDGER.is_file():
        return result
    suite = load_suite(check_sources=False)
    check_fresh(suite, result)
    state = Ledger(suite).snapshot()
    result['jobs'].update({run:dict(row, campaign_id=CAMPAIGN) for run, row in state['slots'].items()})
    result['campaigns'][CAMPAIGN] = dict(campaign=state['suite'], authority=state['authority'], commits={}, cap=COUNT)
    result['maximum_total_shifts'] += COUNT
    result['additional_reserved_shifts'] += COUNT
    return result
