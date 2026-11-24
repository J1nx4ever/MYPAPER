"""Register the four physical validation jobs under the complete budget."""
import json
from pathlib import Path

from smoke_plan import CAMPAIGN, OUTPUT, SUITE, STRATEGIES, encoded, load_reference, load_suite, reference, require, validate_jobs

from ablation_budget import LOCKS, budget_snapshot as preceding_snapshot
from business_graph_summary_campaign_v1.launch import _launch_lock
from online_v1_1.operations import _publish
from physical_curriculum_v1.budget import _history_rows
from python.offline_split import JAVA_MASK, java_random_states


SCHEMA = 'bg128-traditional-smoke-budget-v1'
LEDGER = OUTPUT / 'budget.json'
AUTHORITY = OUTPUT / 'launch/run_authority.json'
COUNT = len(STRATEGIES)
STATUSES = ('PLANNED', 'CLAIMED', 'COMPLETE', 'UNKNOWN')
SEED_FIELDS = ('experiment_seed', 'actor_sampling_seed', 'actor_seed')


def check_fresh(suite, historical):
    require(CAMPAIGN not in historical['campaigns'], 'campaign already registered')
    require(historical['active_h2_reserved_shifts'] == 0, 'unexpected active H2 reservation')
    rows = _history_rows(historical)
    jobs = suite['jobs']
    validate_jobs(jobs)
    old_runs = {row.get('run_index') for row in rows}
    old_recordings = {row.get(key) for row in rows for key in ('recording_id', 'source_recording_id')}
    require(all(j['run_index'] not in old_runs and j['recording_id'] not in old_recordings for j in jobs),
            'physical run or recording overlaps a historical ledger')
    used_seeds = {row[key] & JAVA_MASK for row in rows for key in SEED_FIELDS
                  if type(row.get(key)) is int and row[key] > 0}
    environment, sampling = jobs[0]['experiment_seed'], jobs[-1]['actor_sampling_seed']
    require(used_seeds.isdisjoint((environment, sampling)), 'new environment or sampling seed overlaps history')
    require(environment != jobs[-1]['actor_seed'] and sampling != jobs[-1]['actor_seed'],
            'new seed overlaps the inherited checkpoint training seed')
    prior_states = {state for row in rows if type(row.get('experiment_seed')) is int
                    and row['experiment_seed'] > 0
                    for state in java_random_states(row['experiment_seed']).values()}
    proposed_states = java_random_states(environment)
    require(len(set(proposed_states.values())) == len(proposed_states)
            and prior_states.isdisjoint(proposed_states.values()), 'effective Java environment RNG state overlaps history')
    return dict(occupied_before=historical['maximum_total_shifts'],
                maximum_total_if_registered=historical['maximum_total_shifts']+COUNT,
                historical_identities_checked=len(rows), new_physical_jobs=COUNT)


def authority_for(suite):
    first = suite['jobs'][0]
    return dict(schema=SCHEMA+'-authority', campaign_id=CAMPAIGN, suite=reference(SUITE),
                authorized_by_matching_anylogic_run=True, maximum_new_shifts=COUNT,
                first_run={key: first[key] for key in ('run_index', 'experiment_seed', 'recording_id')})


def _validate_state(state, suite):
    require(state['schema'] == SCHEMA and state['suite'] == reference(SUITE), 'smoke ledger suite identity differs')
    require(state['authority'] == reference(AUTHORITY) and load_reference(state['authority']) == authority_for(suite),
            'smoke ledger launch authority differs')
    require(set(state['slots']) == {str(j['run_index']) for j in suite['jobs']}, 'smoke ledger inventory differs')
    statuses = []
    for job in suite['jobs']:
        slot = state['slots'][str(job['run_index'])]
        require(slot['status'] in STATUSES and slot['identity'] == job, 'smoke ledger slot identity or status differs')
        if slot['status'] == 'COMPLETE':
            require(reference(slot['receipt']['path']) == slot['receipt'], 'smoke completion receipt bytes differ')
        statuses.append(slot['status'])
    require(sum(status in ('CLAIMED', 'UNKNOWN') for status in statuses) <= 1
            and sorted(statuses, key=('COMPLETE', 'CLAIMED', 'UNKNOWN', 'PLANNED').index) == statuses,
            'smoke ledger physical order differs')
    return state


def _read_state(suite):
    data = LEDGER.read_bytes()
    state = json.loads(data)
    require(data == encoded(state), 'noncanonical smoke ledger bytes')
    return _validate_state(state, suite)


def budget_snapshot():
    """Include any registered four-strategy smoke slots in all future budgets."""
    result = preceding_snapshot()
    if not LEDGER.is_file():
        return result
    suite = load_suite(check_sources=False)
    check_fresh(suite, result)
    state = _read_state(suite)
    result['jobs'].update({key: dict(row, campaign_id=CAMPAIGN) for key, row in state['slots'].items()})
    result['campaigns'][CAMPAIGN] = dict(campaign=state['suite'], authority=state['authority'], commits={}, cap=COUNT)
    result['maximum_total_shifts'] += COUNT
    result['additional_reserved_shifts'] += COUNT
    return result


class Ledger:
    """Write the exact four sequential physical slots of the launched suite."""

    def __init__(self, suite):
        self.suite = suite

    def register(self):
        with _launch_lock(LOCKS / 'budget'):
            require(not LEDGER.exists() and load_reference(reference(AUTHORITY)) == authority_for(self.suite),
                    'existing ledger or invalid matching AnyLogic launch authority')
            before = preceding_snapshot()
            check_fresh(self.suite, before)
            state = dict(schema=SCHEMA, suite=reference(SUITE), authority=reference(AUTHORITY),
                         occupied_before=before['maximum_total_shifts'],
                         slots={str(j['run_index']): dict(status='PLANNED', identity=j) for j in self.suite['jobs']})
            _publish(LEDGER, encoded(state))

    def snapshot(self):
        return _read_state(self.suite)

    def _transition(self, job, before, after, **evidence):
        state = self.snapshot()
        key = str(job['run_index'])
        row = state['slots'][key]
        require(row['identity'] == job and row['status'] == before, 'physical smoke slot is out of order or already used')
        if after == 'CLAIMED':
            earlier = self.suite['jobs'][:job['ordinal']]
            require(all(state['slots'][str(j['run_index'])]['status'] == 'COMPLETE' for j in earlier),
                    'all preceding physical jobs must complete before the next claim')
        state['slots'][key] = dict(identity=job, status=after, **evidence)
        _publish(LEDGER, encoded(state), replace=True)

    def claim(self, job):
        self._transition(job, 'PLANNED', 'CLAIMED')

    def complete(self, job, receipt):
        require(receipt['path'] == str(Path(job['python_output'])/'evaluation_receipt.json'),
                'receipt must be the sole per-strategy evaluation record')
        payload = load_reference(receipt)
        for key in ('run_index', 'recording_id', 'experiment_seed', 'actor_sampling_seed', 'policy_kind', 'phase_config'):
            require(payload[key] == job[key], 'physical receipt differs from the launched job: ' + key)
        require(payload['status'] == 'COMPLETE' and payload['normal_completion'] is True
                and payload['completed_shifts'] == 1 and payload['optimizer_updates'] == 0,
                'only a complete evaluation-only physical receipt is admissible')
        self._transition(job, 'CLAIMED', 'COMPLETE', receipt=receipt)

    def mark_unknown(self, job, error):
        self._transition(job, 'CLAIMED', 'UNKNOWN', reason=str(error))
