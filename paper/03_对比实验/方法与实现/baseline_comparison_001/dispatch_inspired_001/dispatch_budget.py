"""Append the single deterministic validation identity to the full budget."""
import json
from pathlib import Path

from dispatch_plan import (CAMPAIGN, COUNT, OUTPUT, SUITE, encoded, load_reference, load_suite,
                           reference, require, validate_jobs)
from paired_budget import LOCKS, budget_snapshot as preceding_snapshot
from paired_plan import CAMPAIGN as PAIRED_CAMPAIGN, SUITE as PAIRED_SUITE, load_suite as load_paired
from business_graph_summary_campaign_v1.launch import _launch_lock
from online_v1_1.operations import _publish
from physical_curriculum_v1.budget import _history_rows
from python.offline_split import JAVA_MASK, java_random_states


SCHEMA = 'bg128-dispatch-inspired-validation-budget-v1'
LEDGER = OUTPUT/'budget.json'
AUTHORITY = OUTPUT/'launch/run_authority.json'
STATUSES = ('PLANNED', 'CLAIMED', 'COMPLETE', 'UNKNOWN')
SEED_FIELDS = ('experiment_seed', 'actor_sampling_seed', 'actor_seed')


def check_fresh(suite, historical):
    require(CAMPAIGN not in historical['campaigns'] and historical['active_h2_reserved_shifts'] == 0,
            'DISPATCH campaign already registered or previous budget scope differs')
    validate_jobs(suite['jobs'])
    job = suite['jobs'][0]
    history = _history_rows(historical)
    # The earlier 400-shift package may be prepared while its physical slots
    # remain unregistered. Preserve its preassigned identities without charging it.
    pending = (load_paired(check_sources=False)['jobs']
               if PAIRED_CAMPAIGN not in historical['campaigns'] and PAIRED_SUITE.is_file() else [])
    rows = [*history, *pending]
    require(not any(row.get('run_index') == job['run_index']
                    or job['recording_id'] in (row.get('recording_id'), row.get('source_recording_id')) for row in rows),
            'physical run or recording overlaps historical or already prepared identity')
    previous = {row[key] & JAVA_MASK for row in rows for key in SEED_FIELDS
                if type(row.get(key)) is int and row[key] > 0}
    environment = job['experiment_seed']
    require(environment not in previous, 'new environment seed overlaps an earlier seed')
    previous_java = {state for row in rows if type(row.get('experiment_seed')) is int and row['experiment_seed'] > 0
                     for state in java_random_states(row['experiment_seed']).values()}
    states = set(java_random_states(environment).values())
    require(len(states) == 5 and previous_java.isdisjoint(states), 'new effective engine or geology RNG state overlaps history')
    return dict(occupied_before=historical['maximum_total_shifts'],
                maximum_total_if_registered=historical['maximum_total_shifts']+COUNT,
                historical_identities_checked=len(history), prepared_identities_checked=len(pending),
                new_physical_jobs=COUNT, new_environment_seeds=1, new_actor_sampling_seeds=0,
                new_java_rng_states=len(states))


def authority_for(suite):
    job = suite['jobs'][0]
    return dict(schema=SCHEMA+'-authority', campaign_id=CAMPAIGN, suite=reference(SUITE),
                authorized_by_matching_anylogic_run=True, maximum_new_shifts=COUNT,
                first_run={key: job[key] for key in ('run_index', 'experiment_seed', 'recording_id')})


class Ledger:
    """Keep one exclusive physical start and its immutable receipt."""

    def __init__(self, suite):
        self.suite = suite
        self.job = suite['jobs'][0]
        self.run = str(self.job['run_index'])

    def register(self):
        with _launch_lock(LOCKS/'budget'):
            require(not LEDGER.exists() and load_reference(reference(AUTHORITY)) == authority_for(self.suite),
                    'existing DISPATCH ledger or mismatched launch authority')
            occupied = check_fresh(self.suite, preceding_snapshot())['occupied_before']
            state = dict(schema=SCHEMA, suite=reference(SUITE), authority=reference(AUTHORITY), occupied_before=occupied,
                         slots={self.run:dict(identity=self.job, status='PLANNED')})
            _publish(LEDGER, encoded(state))

    def snapshot(self):
        data = LEDGER.read_bytes()
        state = json.loads(data)
        require(data == encoded(state) and state['schema'] == SCHEMA and state['suite'] == reference(SUITE)
                and state['authority'] == reference(AUTHORITY)
                and load_reference(state['authority']) == authority_for(self.suite)
                and set(state['slots']) == {self.run}, 'DISPATCH ledger registration or authority differs')
        slot = state['slots'][self.run]
        require(slot['identity'] == self.job and slot['status'] in STATUSES, 'DISPATCH ledger slot identity differs')
        if slot['status'] == 'COMPLETE':
            require(reference(slot['receipt']['path']) == slot['receipt'], 'DISPATCH completion receipt bytes differ')
        return state

    def _transition(self, job, before, after, **evidence):
        state = self.snapshot()
        slot = state['slots'][self.run]
        require(job == self.job and slot['status'] == before, 'DISPATCH physical slot already used or out of order')
        state['slots'][self.run] = dict(identity=job, status=after, **evidence)
        _publish(LEDGER, encoded(state), replace=True)

    def claim(self, job):
        self._transition(job, 'PLANNED', 'CLAIMED')

    def complete(self, job, receipt):
        require(receipt['path'] == str(Path(job['python_output'])/'evaluation_receipt.json'),
                'receipt must match the frozen physical output path')
        payload = load_reference(receipt)
        for key in ('run_index', 'recording_id', 'experiment_seed', 'actor_sampling_seed', 'policy_kind', 'phase_config'):
            require(payload[key] == job[key], 'DISPATCH receipt differs from launched job: ' + key)
        require(payload['status'] == 'COMPLETE' and payload['normal_completion'] is True
                and payload['completed_shifts'] == 1 and payload['optimizer_updates'] == 0,
                'complete evaluation-only DISPATCH receipt required')
        self._transition(job, 'CLAIMED', 'COMPLETE', receipt=receipt)

    def mark_unknown(self, job, error):
        self._transition(job, 'CLAIMED', 'UNKNOWN', reason=str(error))


def budget_snapshot():
    """Merge paired100 and all preceding ledgers plus registered DISPATCH validation."""
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
