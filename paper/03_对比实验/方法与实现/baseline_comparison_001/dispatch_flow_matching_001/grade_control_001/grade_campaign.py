"""Freeze a ten-shift standard-condition validation of grade-controlled dispatch."""
import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import re
import sys


PLAN = Path(__file__).resolve().parent
ROOT = PLAN.parents[5]
REPLICATION = PLAN.parents[2] / 'gat_weak_ablation_001' / 'seed_replication_001'
for directory in (ROOT, PLAN.parent, REPLICATION):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import flow_matching_campaign as flow
import replication_budget
from flow_matching_campaign import (
    CORE, CORE_SHA256, COLUMNS, EXECUTION, LOCKS, PYTHON, _checked_reference, _csv_bytes,
    _launch_lock, _publish_bytes, _publish_json, _replace_once, _require, connection, encoded,
    episode_id, load_reference, reference, scenario_table_bytes, verify_java,
)
from physical_curriculum_v1.budget import _history_rows
from python.offline_split import JAVA_MASK, java_random_states


CAMPAIGN = 'bg128-dispatch-flow-grade-control10-20261123-001'
SCHEMA = 'bg128-dispatch-flow-grade-control10-v1'
POLICY = 'dispatch_flow_grade_control'
COUNT = 10
FIRST_RUN = 10036521
METHOD = dict(flow.METHOD)
PREPARED = PLAN / 'prepared_001'
SUITE = PREPARED / 'suite.json'
OUTPUT = ROOT / 'runs' / CAMPAIGN
CONTROL = OUTPUT / 'control'
LEDGER = OUTPUT / 'budget.sqlite'
REPORT = flow.REPORT.parent / PLAN.name / 'grade_control10_evaluation.json'
OFFLINE = REPORT.with_name('offline_review_verified.json')
LAUNCHER = PLAN / 'grade_launch.py'


def preceding_snapshot():
    return replication_budget.budget_snapshot()


def make_job(ordinal, anchor):
    run = FIRST_RUN + ordinal
    seed = int.from_bytes(sha256(f'{CAMPAIGN}:environment:{ordinal}'.encode('ascii')).digest()[:6], 'big')
    recording = f'{CAMPAIGN}-{ordinal:03d}'
    events = OUTPUT / 'anylogic_events' / f'run_{run}'
    return dict(ordinal=ordinal, pair_ordinal=ordinal, policy_kind=POLICY, group='BASELINE', role='final',
                run_index=run, experiment_seed=seed, recording_id=recording, source_recording_id=recording,
                actor_seed=0, actor_sampling_seed=0, actor_sampling_seed_consumed=False,
                target_training_update=0, checkpoint_update=0, event_directory=str(events),
                event_bundle=str(events / episode_id(run, seed)), python_output=str(OUTPUT / 'shifts' / f'run_{run}'),
                scene_id=anchor['scene_id'], expected_rpc_config=anchor['expected_rpc_config'],
                runtime_scene_sha256=anchor['runtime_scene_sha256'])


def check_fresh(jobs, history):
    _require(CAMPAIGN not in history['campaigns'] and history['active_h2_reserved_shifts'] == 0,
             'grade trial is already registered or budget policy differs')
    rows = _history_rows(history) + flow.v3_ledger._prepared_rows(history)
    # Reserve every embedded roster and the sidecar-only GAT1 replications.
    for path in PLAN.parents[2].rglob('suite.json'):
        if path != SUITE:
            rows.extend(row for row in json.loads(path.read_bytes()).get('jobs', []) if isinstance(row, dict))
    for unit in replication_budget.plan.UNITS:
        suite_path = Path(replication_budget.plan.locations(unit)['suite_path'])
        if suite_path.is_file():
            with suite_path.with_name('master_plan.csv').open(newline='') as stream:
                roster = list(csv.DictReader(stream))
            _require(len(roster) == replication_budget.plan.CAP, 'prepared GAT1 roster size differs')
            for row in roster:
                rows.append(dict(row, **{key: int(row[key]) for key in ('run_index', *flow.SEED_FIELDS)}))
    prior_runs = {row.get('run_index') for row in rows}
    prior_recordings = {row.get(key) for row in rows for key in ('recording_id', 'source_recording_id')}
    prior_seeds = {row[key] & JAVA_MASK for row in rows for key in flow.SEED_FIELDS
                   if type(row.get(key)) is int and row[key] > 0}
    prior_environment = {row['experiment_seed'] for row in rows if type(row.get('experiment_seed')) is int
                         and row['experiment_seed'] > 0}
    prior_java = {state for seed in prior_environment for state in java_random_states(seed).values()}
    runs = {job['run_index'] for job in jobs}
    recordings = {job['recording_id'] for job in jobs}
    seeds = {job['experiment_seed'] for job in jobs}
    states = {state for seed in seeds for state in java_random_states(seed).values()}
    _require(len(jobs) == len(runs) == len(recordings) == len(seeds) == COUNT
             and all(0 < seed <= JAVA_MASK for seed in seeds) and len(states) == 5 * COUNT,
             'grade roster must have ten independent identities and Java streams')
    _require(runs.isdisjoint(prior_runs) and recordings.isdisjoint(prior_recordings)
             and seeds.isdisjoint(prior_seeds) and states.isdisjoint(prior_java),
             'grade roster overlaps a historical or prepared run, recording or random stream')
    return dict(occupied_before=history['maximum_total_shifts'],
                maximum_total_if_registered=history['maximum_total_shifts'] + COUNT,
                new_physical_shifts=COUNT, new_environment_seeds=COUNT, new_actor_sampling_seeds=0,
                new_java_rng_states=len(states), historical_and_prepared_rows_checked=len(rows))


def _render_java(old, master, scenarios):
    path = Path(old['java_helper']['path'])
    _checked_reference(old['java_helper'])
    source = path.read_text()
    values = dict(MASTER_PLAN_PATH=str(master), MASTER_PLAN_SHA256=reference(master)['sha256'],
                  SCENARIO_TABLE_PATH=str(scenarios), SCENARIO_TABLE_SHA256=reference(scenarios)['sha256'],
                  CONTROL_DIRECTORY=str(CONTROL), CAMPAIGN_ID=CAMPAIGN)
    for name, value in values.items():
        source, replacements = re.subn(f'(private static final String {name} = )"[^"]+";',
                                       lambda found: found[1] + json.dumps(value) + ';', source)
        _require(replacements == 1, f'Java template constant differs: {name}')
    source = _replace_once(source, str(flow.LAUNCHER), str(LAUNCHER))
    source = _replace_once(source, reference(flow.LAUNCHER)['sha256'], reference(LAUNCHER)['sha256'])
    source = _replace_once(source, str(flow.SUITE), str(SUITE))
    source = _replace_once(source, 'private static final int JOB_COUNT = 5;',
                           f'private static final int JOB_COUNT = {COUNT};')
    return _replace_once(source, f'return "{flow.POLICY}";', f'return "{POLICY}";')


def _phase(job, source_pins, protected_core):
    return dict(schema=SCHEMA + '-phase', campaign_id=CAMPAIGN, policy_kind=POLICY,
                method=METHOD, execution=EXECUTION, source_pins=source_pins, protected_core=protected_core, job=job)


def prepare():
    _require(not any(path.exists() for path in (PREPARED, OUTPUT, REPORT)),
             'grade preparation or physical evidence already exists')
    old = flow.load_suite()
    old_complete = reference(flow.OUTPUT / 'suite_complete.json')
    complete = load_reference(old_complete)
    _require(complete['status'] == 'COMPLETE' and complete['physical_shifts'] == flow.COUNT
             and complete['suite'] == reference(flow.SUITE), 'prior physical shift identities are not complete')
    _checked_reference(complete['evaluation'])
    offline = reference(OFFLINE)
    evidence = load_reference(offline)
    names = ('grade_flow_plan.py', 'grade_guard.py', 'grade_policy.py')
    reviewed_pins = [reference(PLAN / name) for name in names]
    _require(evidence['physical_shifts_started'] == 0
             and all(pin in evidence['source_references']['grade_control'] for pin in reviewed_pins),
             'grade runtime does not match the reviewed offline strategy bytes')
    pins = [*old['source_pins'], *reviewed_pins, reference(__file__), reference(LAUNCHER)]
    protected = old['protected_core']
    jobs = [make_job(ordinal, old['jobs'][0]) for ordinal in range(COUNT)]
    budget = check_fresh(jobs, preceding_snapshot())
    for job in jobs:
        path = PREPARED / 'configs' / f"run_{job['run_index']}.json"
        job['phase_config'] = _publish_json(path, _phase(job.copy(), pins, protected))
    master, scenarios = PREPARED / 'master_plan.csv', PREPARED / 'scenario_table.csv'
    master_ref = _publish_bytes(master, _csv_bytes(jobs, COLUMNS))
    scenario_ref = _publish_bytes(scenarios, scenario_table_bytes(jobs))
    helper = PREPARED / 'Baseline1CampaignPlan.java'
    java_ref = _publish_bytes(helper, _render_java(old, master, scenarios).encode('utf-8'))
    proof = verify_java(helper, scenarios)
    _require(proof['rows_checked'] == COUNT, 'Java did not verify the ten reserved rows')
    suite = dict(schema=SCHEMA, campaign_id=CAMPAIGN, policy_kind=POLICY, method=METHOD, execution=EXECUTION,
                 protected_core=protected, source_pins=pins, master_plan=master_ref, scenario_table=scenario_ref,
                 java_helper=java_ref, java_verification=proof, jobs=jobs, baseline=old_complete,
                 offline_review=offline, control_directory=str(CONTROL), python_output=str(OUTPUT),
                 report=str(REPORT), budget=budget)
    suite_ref = _publish_json(SUITE, suite)
    columns = ('run_index', 'experiment_seed', 'recording_id', 'config_sha256', 'python_output', 'event_directory')
    roster = _publish_bytes(PREPARED / 'run_seed.csv', _csv_bytes(
        [dict(job, config_sha256=job['phase_config']['sha256']) for job in jobs], columns))
    return _publish_json(PREPARED / 'launch_identity.json', dict(campaign_id=CAMPAIGN, suite=suite_ref,
        run_seed=roster, java_helper=java_ref, budget=budget, physical_shifts=COUNT, optimizer_updates=0, jobs=jobs))


def load_suite(path=SUITE, *, check_sources=True):
    _require(Path(path).resolve() == SUITE, 'unexpected grade manifest path')
    suite = json.loads(SUITE.read_bytes())
    _require(suite['schema'] == SCHEMA and suite['campaign_id'] == CAMPAIGN and suite['policy_kind'] == POLICY
             and suite['method'] == METHOD and suite['execution'] == EXECUTION and len(suite['jobs']) == COUNT
             and suite['protected_core'] == dict(path=str(CORE), sha256=CORE_SHA256)
             and suite['control_directory'] == str(CONTROL) and suite['report'] == str(REPORT),
             'ten-shift grade contract differs')
    if check_sources:
        for pin in [suite['protected_core'], suite['master_plan'], suite['scenario_table'], suite['java_helper'],
                    suite['baseline'], suite['offline_review'], *suite['source_pins']]:
            _checked_reference(pin)
    anchor = flow.load_suite(check_sources=False)['jobs'][0]
    for ordinal, row in enumerate(suite['jobs']):
        phase = load_reference(row['phase_config'])
        job = {key: value for key, value in row.items() if key != 'phase_config'}
        _require(phase == _phase(job, suite['source_pins'], suite['protected_core'])
                 and job == make_job(ordinal, anchor), 'phase configuration and frozen grade run differ')
    return suite


class Ledger:
    """Serialize physical admission across pending, claimed and settled runs."""

    def __init__(self, suite):
        self.suite = suite
        self.jobs = {job['run_index']: job for job in suite['jobs']}

    def register(self, authority):
        with _launch_lock(LOCKS / 'budget'):
            _require(not LEDGER.exists(), 'grade budget is already registered')
            budget = check_fresh(self.suite['jobs'], preceding_snapshot())
            with connection(LEDGER) as con:
                con.execute('CREATE TABLE campaign (id TEXT PRIMARY KEY, payload TEXT NOT NULL)')
                con.execute('CREATE TABLE slots (run INTEGER PRIMARY KEY, identity TEXT NOT NULL, '
                            'status TEXT NOT NULL, evidence TEXT)')
                entry = dict(suite=reference(SUITE), authority=authority, cap=COUNT,
                             occupied_before=budget['occupied_before'])
                con.execute('INSERT INTO campaign VALUES (?,?)', (CAMPAIGN, encoded(entry).decode()))
                con.executemany("INSERT INTO slots VALUES (?,?,'PLANNED',NULL)",
                                [(run, encoded(job).decode()) for run, job in self.jobs.items()])

    def snapshot(self):
        with connection(LEDGER, readonly=True) as con:
            (campaign, payload), = con.execute('SELECT id,payload FROM campaign')
            entry = json.loads(payload)
            _require(campaign == CAMPAIGN and entry['suite'] == reference(SUITE) and entry['cap'] == COUNT,
                     'ledger and frozen grade campaign differ')
            _checked_reference(entry['authority'])
            slots = {str(run): dict(identity=json.loads(identity), status=state,
                                    evidence=json.loads(evidence) if evidence else None)
                     for run, identity, state, evidence in con.execute(
                         'SELECT run,identity,status,evidence FROM slots ORDER BY run')}
        _require({int(run): row['identity'] for run, row in slots.items()} == self.jobs,
                 'grade ledger identities differ')
        return entry, slots

    def transition(self, job, before, after, evidence=None):
        run = job['run_index']
        _require(self.jobs.get(run) == job and (before, after) in
                 {('PLANNED', 'CLAIMED'), ('CLAIMED', 'COMPLETE'), ('CLAIMED', 'UNKNOWN')},
                 'invalid physical transition')
        if after == 'COMPLETE':
            record = load_reference(evidence)
            _require(evidence['path'] == str(Path(job['python_output']) / 'evaluation_receipt.json')
                     and record['normal_completion'] is True and record['optimizer_updates'] == 0
                     and all(record[key] == job[key] for key in
                             ('run_index', 'experiment_seed', 'recording_id', 'policy_kind', 'phase_config')),
                     'physical receipt and claimed grade identity differ')
        with connection(LEDGER) as con:
            if after == 'CLAIMED':
                earlier = con.execute("SELECT 1 FROM slots WHERE run < ? AND status != 'COMPLETE'", (run,)).fetchone()
                _require(earlier is None, 'earlier grade shift remains unresolved')
            updated = con.execute('UPDATE slots SET status=?,evidence=? WHERE run=? AND status=? AND identity=?',
                                  (after, encoded(evidence).decode() if evidence else None, run, before,
                                   encoded(job).decode()))
            _require(updated.rowcount == 1, 'grade slot was already consumed or has the wrong state')


def budget_snapshot():
    result = preceding_snapshot()
    if LEDGER.is_file():
        entry, slots = Ledger(load_suite(check_sources=False)).snapshot()
        for run, row in slots.items():
            item = dict(identity=row['identity'], status=row['status'], campaign_id=CAMPAIGN)
            if row['status'] == 'COMPLETE':
                item['receipt'] = row['evidence']
            elif row['status'] == 'UNKNOWN':
                item.update(row['evidence'])
            result['jobs'][run] = item
        result['campaigns'][CAMPAIGN] = dict(campaign=entry['suite'], authority=entry['authority'], commits={}, cap=COUNT)
        result['maximum_total_shifts'] += COUNT
        result['additional_reserved_shifts'] += COUNT
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', required=True, action='store_true')
    parser.parse_args()
    print(json.dumps(prepare(), ensure_ascii=False))
