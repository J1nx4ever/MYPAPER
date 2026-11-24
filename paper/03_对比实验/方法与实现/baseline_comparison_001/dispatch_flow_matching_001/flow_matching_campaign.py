"""Freeze and account for five fresh standard-condition AnyLogic shifts."""
import argparse
import csv
from hashlib import sha256
import io
import json
from pathlib import Path
import re
import sys


PLAN = Path(__file__).resolve().parent
ROOT = PLAN.parents[4]
V3 = PLAN.parent / 'dispatch_fifo_v3_001'
for directory in (ROOT, V3):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import v3_ledger
import v3_plan
from baseline1_v1.java_plan import COLUMNS, _replace_once
from business_graph_summary_campaign_v1.budget import connection
from business_graph_summary_campaign_v1.launch import _launch_lock
from business_graph_summary_campaign_v1.plan_io import encoded, load_reference, reference
from java_helper import scenario_table_bytes, verify_java
from physical_curriculum_v1.budget import _history_rows
from python.offline_split import JAVA_MASK, episode_id, java_random_states
from traditional_dispatch_v1.runtime import _checked_reference, _publish_bytes, _publish_json, _require


CAMPAIGN = 'bg128-dispatch-flow-matching-smoke5-20261123-002'
POLICY = 'dispatch_flow_matching'
SCHEMA = 'bg128-dispatch-flow-matching-smoke5-v1'
COUNT = 5
FIRST_RUN = 10036511
METHOD = dict(planning_ms=800, replan_seconds=900, match_trucks=3, soft_replan_seconds=120, response_target_ms=1000)
EXECUTION = dict(v3_plan.EXECUTION)
PYTHON = v3_plan.PYTHON
CORE = v3_plan.CORE
CORE_SHA256 = '8e4cfa31c622f5e33756e304eea3f7526547db72ae93f7d9f00808845d61ef96'
PREPARED = PLAN / 'prepared_002'
SUITE = PREPARED / 'suite.json'
OUTPUT = ROOT / 'runs' / CAMPAIGN
CONTROL = OUTPUT / 'control'
LEDGER = OUTPUT / 'budget.sqlite'
REPORT = v3_plan.REPORT.parent / PLAN.name / 'smoke5_evaluation.json'
LAUNCHER = PLAN / 'flow_matching_launch.py'
JAVA_TEMPLATE = V3 / 'prepared_002' / 'Baseline1CampaignPlan.java'
SEED_FIELDS = ('experiment_seed', 'actor_seed', 'actor_sampling_seed')
LOCKS = v3_ledger.LOCKS


def make_job(ordinal, config_anchor):
    run = FIRST_RUN + ordinal
    seed = int.from_bytes(sha256(f'{CAMPAIGN}:environment:{ordinal}'.encode('ascii')).digest()[:6], 'big')
    recording = f'{CAMPAIGN}-{ordinal:03d}'
    leaf = f'run_{run}'
    events = OUTPUT / 'anylogic_events' / leaf
    return dict(ordinal=ordinal, pair_ordinal=ordinal, policy_kind=POLICY, actor_seed=0, group='BASELINE',
                role='final', source_recording_id=recording, recording_id=recording, run_index=run,
                experiment_seed=seed, actor_sampling_seed=0, actor_sampling_seed_consumed=False,
                event_directory=str(events), event_bundle=str(events / episode_id(run, seed)),
                python_output=str(OUTPUT / 'shifts' / leaf), target_training_update=0, checkpoint_update=0,
                scene_id=config_anchor['scene_id'], expected_rpc_config=config_anchor['expected_rpc_config'],
                runtime_scene_sha256=config_anchor['runtime_scene_sha256'])


def check_fresh(jobs, history):
    _require(CAMPAIGN not in history['campaigns'] and history['active_h2_reserved_shifts'] == 0,
             'campaign already registered or budget policy differs')
    rows = _history_rows(history) + v3_ledger._prepared_rows(history)
    # This also reserves unlaunched drafts and packages staged after the V3 report.
    for path in PLAN.parents[1].rglob('suite.json'):
        if path == SUITE:
            continue
        draft = json.loads(path.read_bytes())
        rows.extend(row for row in draft.get('jobs', []) if isinstance(row, dict))
    occupied_runs = {row.get('run_index') for row in rows}
    occupied_recordings = {row.get(key) for row in rows for key in ('recording_id', 'source_recording_id')}
    occupied_seeds = {row[key] & JAVA_MASK for row in rows for key in SEED_FIELDS
                      if type(row.get(key)) is int and row[key] > 0}
    old_states = {state for row in rows if type(row.get('experiment_seed')) is int and row['experiment_seed'] > 0
                  for state in java_random_states(row['experiment_seed']).values()}
    seeds = {job['experiment_seed'] for job in jobs}
    new_states = {state for seed in seeds for state in java_random_states(seed).values()}
    _require(len(jobs) == len(seeds) == COUNT and min(seeds) > 0 and max(seeds) <= JAVA_MASK
             and len(new_states) == 5 * COUNT and new_states.isdisjoint(old_states) and seeds.isdisjoint(occupied_seeds),
             'fresh environment seeds or effective Java states overlap history')
    _require(len({job['run_index'] for job in jobs}) == len({job['recording_id'] for job in jobs}) == COUNT
             and all(job['run_index'] not in occupied_runs and job['recording_id'] not in occupied_recordings
                     for job in jobs), 'fresh run or recording overlaps historical or prepared work')
    return dict(occupied_before=history['maximum_total_shifts'], new_physical_shifts=COUNT,
                maximum_total_if_registered=history['maximum_total_shifts'] + COUNT,
                existing_identity_rows_checked=len(rows), new_environment_seeds=len(seeds),
                new_actor_sampling_seeds=0)


def _csv_bytes(rows, columns):
    stream = io.StringIO(newline='')
    table = csv.DictWriter(stream, fieldnames=columns, lineterminator='\n')
    table.writeheader()
    table.writerows({key: row[key] for key in columns} for row in rows)
    return stream.getvalue().encode('ascii')


def _render_java(master, scenarios):
    source = JAVA_TEMPLATE.read_text()
    strings = dict(MASTER_PLAN_PATH=str(master), MASTER_PLAN_SHA256=reference(master)['sha256'],
                   SCENARIO_TABLE_PATH=str(scenarios), SCENARIO_TABLE_SHA256=reference(scenarios)['sha256'],
                   CONTROL_DIRECTORY=str(CONTROL), CAMPAIGN_ID=CAMPAIGN)
    for name, val in strings.items():
        pattern = f'(private static final String {name} = )"[^"]+";'
        source, replacements = re.subn(pattern, lambda found: found[1] + json.dumps(val) + ';', source)
        _require(replacements == 1, f'Java template constant differs: {name}')
    source = _replace_once(source, 'private static final int JOB_COUNT = 100;',
                           f'private static final int JOB_COUNT = {COUNT};')
    source = _replace_once(source, str(V3 / 'launch.py'), str(LAUNCHER))
    source = _replace_once(source, reference(V3 / 'launch.py')['sha256'], reference(LAUNCHER)['sha256'])
    source = _replace_once(source, str(v3_plan.SUITE), str(SUITE))
    return _replace_once(source, 'return "dispatch_fifo_v3";', f'return "{POLICY}";')


def prepare():
    _require(not PREPARED.exists() and not OUTPUT.exists() and not REPORT.exists(),
             'prepared or physical smoke artifacts already exist')
    protected = reference(CORE)
    _require(protected['sha256'] == CORE_SHA256, 'protected PPO core bytes differ')
    previous = v3_plan.load_suite(check_sources=False)
    _require(reference(JAVA_TEMPLATE) == previous['java_helper'], 'audited Java template differs')
    jobs = [make_job(ordinal, previous['jobs'][0]) for ordinal in range(COUNT)]
    budget = check_fresh(jobs, v3_ledger.budget_snapshot())
    pins = [reference(PLAN / name) for name in ('flow_matching_campaign.py', 'flow_matching_launch.py',
                                               'flow_matching_policy.py', 'flow_matching_plan.py')]
    # Reuse the audited RPC and model dependencies without changing any historical pins.
    shared_paths = {pin['path'] for pin in previous['source_pins']}
    pins.extend(reference(path) for path in sorted(shared_paths) if not Path(path).is_relative_to(V3))
    pins.append(reference(JAVA_TEMPLATE))
    method = dict(METHOD)
    for job in jobs:
        path = PREPARED / 'configs' / f"run_{job['run_index']}.json"
        job['phase_config'] = _publish_json(path, dict(schema=SCHEMA + '-phase', campaign_id=CAMPAIGN,
            policy_kind=POLICY, method=method, execution=EXECUTION, protected_core=protected, source_pins=pins, job=job.copy()))
    master, scenarios = PREPARED / 'master_plan.csv', PREPARED / 'scenario_table.csv'
    master_ref = _publish_bytes(master, _csv_bytes(jobs, COLUMNS))
    scenario_ref = _publish_bytes(scenarios, scenario_table_bytes(jobs))
    helper = PREPARED / 'Baseline1CampaignPlan.java'
    helper_ref = _publish_bytes(helper, _render_java(master, scenarios).encode('utf-8'))
    java_proof = verify_java(helper, scenarios)
    _require(java_proof['rows_checked'] == COUNT, 'Java did not verify the full five-shift roster')
    suite = dict(schema=SCHEMA, campaign_id=CAMPAIGN, policy_kind=POLICY, method=method, execution=EXECUTION,
                 protected_core=protected, source_pins=pins, master_plan=master_ref, scenario_table=scenario_ref,
                 java_helper=helper_ref, java_verification=java_proof, jobs=jobs,
                 control_directory=str(CONTROL), python_output=str(OUTPUT), report=str(REPORT), budget=budget)
    suite_ref = _publish_json(SUITE, suite)
    columns = ('run_index', 'experiment_seed', 'recording_id', 'config_sha256', 'python_output', 'event_directory')
    roster = _publish_bytes(PREPARED / 'run_seed.csv', _csv_bytes(
        [dict(job, config_sha256=job['phase_config']['sha256']) for job in jobs], columns))
    return _publish_json(PREPARED / 'launch_identity.json', dict(campaign_id=CAMPAIGN, suite=suite_ref,
        run_seed=roster, java_helper=helper_ref, budget=budget, physical_shifts=COUNT, optimizer_updates=0, jobs=jobs))


def load_suite(path=SUITE, *, check_sources=True):
    _require(Path(path).resolve() == SUITE, 'unexpected launch manifest path')
    suite = json.loads(SUITE.read_bytes())
    _require(suite['schema'] == SCHEMA and suite['campaign_id'] == CAMPAIGN and suite['policy_kind'] == POLICY
             and suite['method'] == METHOD and suite['execution'] == EXECUTION and len(suite['jobs']) == COUNT
             and suite['protected_core'] == dict(path=str(CORE), sha256=CORE_SHA256)
             and suite['control_directory'] == str(CONTROL) and suite['report'] == str(REPORT),
             'five-shift launch contract differs')
    if check_sources:
        for pin in [suite['protected_core'], suite['master_plan'], suite['scenario_table'], suite['java_helper'],
                    *suite['source_pins']]:
            _checked_reference(pin)
    for ordinal, row in enumerate(suite['jobs']):
        phase = load_reference(row['phase_config'])
        job = {key: value for key, value in row.items() if key != 'phase_config'}
        _require(phase == dict(schema=SCHEMA + '-phase', campaign_id=CAMPAIGN, policy_kind=POLICY,
                  method=METHOD, execution=EXECUTION, protected_core=suite['protected_core'],
                  source_pins=suite['source_pins'], job=job)
                 and job == make_job(ordinal, job), 'phase config and fresh job identity differ')
    return suite


class Ledger:
    def __init__(self, suite):
        self.suite = suite
        self.jobs = {job['run_index']: job for job in suite['jobs']}

    def register(self, authority):
        with _launch_lock(LOCKS / 'budget'):
            _require(not LEDGER.exists(), 'physical smoke budget is already registered')
            budget = check_fresh(self.suite['jobs'], v3_ledger.budget_snapshot())
            with connection(LEDGER) as con:
                con.execute('CREATE TABLE campaign (id TEXT PRIMARY KEY, payload TEXT NOT NULL)')
                con.execute('CREATE TABLE slots (run INTEGER PRIMARY KEY, identity TEXT NOT NULL, '
                            'status TEXT NOT NULL, evidence TEXT)')
                con.execute('INSERT INTO campaign VALUES (?,?)', (CAMPAIGN, encoded(dict(
                    suite=reference(SUITE), authority=authority, cap=COUNT, occupied_before=budget['occupied_before'])).decode()))
                con.executemany("INSERT INTO slots VALUES (?,?,'PLANNED',NULL)",
                                [(run, encoded(job).decode()) for run, job in self.jobs.items()])

    def snapshot(self):
        with connection(LEDGER, readonly=True) as con:
            (campaign, payload), = con.execute('SELECT id,payload FROM campaign')
            entry = json.loads(payload)
            _require(campaign == CAMPAIGN and entry['suite'] == reference(SUITE) and entry['cap'] == COUNT,
                     'ledger and frozen smoke campaign differ')
            _checked_reference(entry['authority'])
            slots = {str(run): dict(identity=json.loads(identity), status=status,
                                    evidence=json.loads(evidence) if evidence else None)
                     for run, identity, status, evidence in con.execute(
                         'SELECT run,identity,status,evidence FROM slots ORDER BY run')}
        _require({int(run): row['identity'] for run, row in slots.items()} == self.jobs,
                 'ledger inventory differs from the frozen identity')
        return entry, slots

    def transition(self, job, before, after, evidence=None):
        run = job['run_index']
        _require(self.jobs.get(run) == job, 'unfrozen physical job')
        with connection(LEDGER) as con:
            if after == 'CLAIMED':
                earlier = con.execute("SELECT 1 FROM slots WHERE run < ? AND status != 'COMPLETE'", (run,)).fetchone()
                _require(earlier is None, 'earlier shift has not been sealed')
            updated = con.execute('UPDATE slots SET status=?,evidence=? WHERE run=? AND status=? AND identity=?',
                                  (after, encoded(evidence).decode() if evidence else None, run, before,
                                   encoded(job).decode()))
            _require(updated.rowcount == 1, 'physical slot has the wrong state')


def budget_snapshot():
    result = v3_ledger.budget_snapshot()
    if LEDGER.is_file():
        entry, slots = Ledger(load_suite(check_sources=False)).snapshot()
        for run, row in slots.items():
            metadata = dict(identity=row['identity'], status=row['status'], campaign_id=CAMPAIGN)
            if row['status'] == 'COMPLETE':
                metadata['receipt'] = row['evidence']
            elif row['status'] == 'UNKNOWN':
                metadata.update(row['evidence'])
            result['jobs'][run] = metadata
        result['campaigns'][CAMPAIGN] = dict(campaign=entry['suite'], authority=entry['authority'], commits={}, cap=COUNT)
        result['maximum_total_shifts'] += COUNT
        result['additional_reserved_shifts'] += COUNT
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', required=True, action='store_true')
    parser.parse_args()
    print(json.dumps(prepare(), ensure_ascii=False))
