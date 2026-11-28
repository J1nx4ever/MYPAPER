"""Admit the REINFORCE physical roster against registered and prepared identities."""

import csv
import json
from pathlib import Path

from . import prepare as plan

from business_graph_summary_campaign_v1.budget import _payload, _state, connection, digest
from business_graph_summary_campaign_v1.launch import _launch_lock
from business_graph_summary_campaign_v1.plan_io import encoded, load_reference, reference
from change_plans.token_attention_001.business_graph_summary_009.gat_weak_ablation_001.seed_extension_001 import (
    extension_budget as predecessor,
)
from replica_budget import LOCKS, ReplicaLedger
from weak_plan import bind
from python.offline_split import JAVA_MASK, java_random_states


FORMAT = 'bg128-reinforce-full-shift-budget-v1'
SEED_KEYS = ('experiment_seed', 'actor_sampling_seed', 'actor_seed')
GRAPH = plan.PLAN.parent
_prior_rows = bind(predecessor.predecessor.ledger.previous.paired10._prior_rows, SUITE=plan.SUITE)
SHARED_BUDGET_LOCK = predecessor.predecessor.ledger.previous.LOCKS / 'budget'
require = plan.original.require


def _manifest_rows(value):
    if isinstance(value, dict):
        if type(value.get('run_index')) is int:
            yield value
        for child in value.values():
            yield from _manifest_rows(child)
    elif isinstance(value, list):
        for child in value:
            yield from _manifest_rows(child)


def _rows(history):
    rows = _prior_rows(history)
    for path in sorted(GRAPH.rglob('master_plan.csv')):
        if path.is_relative_to(plan.PLAN):
            continue
        with path.open(encoding='utf-8', newline='') as stream:
            roster = csv.DictReader(stream)
            if 'run_index' not in (roster.fieldnames or ()):
                continue
            for row in roster:
                keys = ('run_index', *SEED_KEYS, 'environment_seed')
                item = {key: int(row[key]) for key in keys if row.get(key)}
                item.update({key: row[key] for key in ('recording_id', 'source_recording_id') if row.get(key)})
                rows.append(item)
    for path in sorted(GRAPH.rglob('launch_identity.json')):
        if not path.is_relative_to(plan.PLAN):
            rows.extend(_manifest_rows(json.loads(path.read_bytes())))
    return rows


def assert_available(suite, history):
    jobs = suite['jobs']
    require(suite['campaign_id'] == plan.CAMPAIGN and plan.CAMPAIGN not in history['campaigns']
            and len(jobs) == plan.COUNT and history['active_h2_reserved_shifts'] == 0,
            'REINFORCE campaign scope or historical budget differs')
    original_suite = plan.historical()[0]
    require(suite['actor_seed'] == original_suite['actor_seed']
            and suite['initial_actor'] == plan.INITIAL_ACTOR,
            'historical initial Actor identity differs')
    record_by_phase = {job['phase_id']: job['recording_id'] for job in jobs}
    recordings = set(record_by_phase.values())
    require(len(recordings) == len(record_by_phase) == plan.UPDATES and all(
        job['actor_seed'] == suite['actor_seed'] and job['run_index'] == plan.FIRST_RUN + index
        and job['ordinal'] == index and job['recording_id'] == job['source_recording_id']
        == record_by_phase[job['phase_id']] for index, job in enumerate(jobs)),
        'REINFORCE physical run or recording inventory differs')

    rows = _rows(history)
    runs = {row.get('run_index') for row in rows}
    old_recordings = {row.get(key) for row in rows for key in ('recording_id', 'source_recording_id')}
    require(runs.isdisjoint(job['run_index'] for job in jobs) and recordings.isdisjoint(old_recordings),
            'REINFORCE run or recording overlaps a registered or prepared identity')
    old_seeds = {row[key] & JAVA_MASK for row in rows for key in (*SEED_KEYS, 'environment_seed')
                 if type(row.get(key)) is int and row[key] > 0}
    new_seeds = [job[key] for job in jobs for key in SEED_KEYS[:2]]
    require(len(set(new_seeds)) == 2 * plan.COUNT and all(0 < seed <= JAVA_MASK for seed in new_seeds)
            and old_seeds.isdisjoint(new_seeds), 'REINFORCE environment or action seed collides')
    old_env = {row[key] for row in rows for key in ('experiment_seed', 'environment_seed')
               if type(row.get(key)) is int and row[key] > 0}
    old_states = {state for seed in old_env for state in java_random_states(seed).values()}
    new_states = {state for job in jobs for state in java_random_states(job['experiment_seed']).values()}
    require(len(new_states) == 5 * plan.COUNT and new_states.isdisjoint(old_states),
            'REINFORCE derived Java environment streams collide')
    before = history['maximum_total_shifts']
    return dict(registered_before=before, additional_if_started=plan.COUNT, total_if_started=before + plan.COUNT,
                checked_registered_and_prepared_rows=len(rows), new_environment_seeds=plan.COUNT,
                new_actor_sampling_seeds=plan.COUNT, new_java_rng_states=len(new_states))


def authority_path(suite):
    return Path(suite['python_output']) / 'launch/run_authority.json'


def authorization(suite):
    return dict(schema='bg128-reinforce-full-shift-authority-v1', suite=reference(suite['suite_path']),
                campaign_id=suite['campaign_id'], training_shifts=plan.COUNT, evaluation_shifts=0,
                maximum_new_shifts=plan.COUNT, rpc_port=suite['rpc_port'])


class ReinforceLedger(ReplicaLedger):
    """Use serial claims and complete-batch commits from the established mixed ledger."""

    def __init__(self, suite, phases):
        self.suite = suite
        self.path = Path(suite['budget_path'])
        self.campaign_id = suite['campaign_id']
        auth_ref = reference(authority_path(suite))
        authority = authorization(suite)
        require(load_reference(auth_ref) == authority, 'REINFORCE launch authority differs')
        self.entry = dict(format=FORMAT, cap=plan.COUNT, test_only=False,
                          suite_reference=authority['suite'], approval_reference=auth_ref)
        self.jobs = {job['run_index']: job for job in suite['jobs']}
        self.hashes = {run: digest(job) for run, job in self.jobs.items()}
        self.update_jobs = {cfg['arm_update']: cfg['jobs'] for cfg in phases}
        self.configs = {cfg['phase_id']: ref for cfg, ref in zip(phases, suite['phase_configs'], strict=True)}

    def register(self):
        with _launch_lock(SHARED_BUDGET_LOCK):
            require(not self.path.exists(), 'REINFORCE campaign is already registered')
            for key in ('suite_reference', 'approval_reference'):
                artifact = self.entry[key]
                require(reference(artifact['path']) == artifact, 'registration bytes differ: ' + key)
            # The inherited transaction writes the established table/slot shape.
            register = bind(ReplicaLedger._register, assert_available=lambda suites, history: assert_available(suites[0], history))
            register(self, predecessor.budget_snapshot())

    def snapshot(self):
        with connection(self.path, readonly=True) as db:
            rows = db.execute('SELECT id,payload,checksum FROM campaigns').fetchall()
            require(len(rows) == 1 and rows[0][0] == self.campaign_id and _payload(rows[0][1:]) == self.entry
                    and db.execute('SELECT count(*) FROM slots').fetchone()[0] == plan.COUNT,
                    'REINFORCE registration or inventory differs')
            return _state(db, self.suite, self.entry, self.hashes)

    def complete(self, job, dataset_ref, receipt):
        identity = receipt['identity']
        require(identity['actor_sampling_seed'] == job['actor_sampling_seed']
                and identity['environment_seed'] == job['experiment_seed']
                and identity['recording_id'] == job['recording_id'] and identity['run_index'] == job['run_index'],
                'REINFORCE receipt differs from its run, recording or seed pair')
        super().complete(job, dataset_ref, receipt)


def budget_snapshot():
    """Return the complete accounting history, plus REINFORCE only when registered."""
    state = predecessor.budget_snapshot()
    if not plan.LEDGER.is_file():
        return state
    suite, phases = plan.load_package()
    assert_available(suite, state)
    registered = ReinforceLedger(suite, phases).snapshot()
    state['jobs'].update(registered['jobs'])
    state['campaigns'].update(registered['campaigns'])
    state['maximum_total_shifts'] += plan.COUNT
    state['additional_reserved_shifts'] += plan.COUNT
    return state
