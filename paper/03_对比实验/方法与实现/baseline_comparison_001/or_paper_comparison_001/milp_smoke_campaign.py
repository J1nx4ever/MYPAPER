"""Freeze a matched two-shift AnyLogic smoke test for the online MILP baselines."""

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory
import subprocess


PLAN = Path(__file__).resolve().parent
ROOT = PLAN.parents[4]
GRAPH = PLAN.parents[1]
GRADE = PLAN.parent / 'dispatch_flow_matching_001' / 'grade_control_001'
PAIRED = GRADE / 'gat1_s01_paired_001'
WEAK = GRAPH / 'gat_weak_ablation_001'
for directory in (ROOT, GRADE, PAIRED, WEAK):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import grade_campaign as grade
import paired_campaign as paired
from grade_campaign import (
    CORE, CORE_SHA256, COLUMNS, EXECUTION, LOCKS, PYTHON, _checked_reference, _csv_bytes,
    _publish_bytes, _publish_json, _replace_once, _require, episode_id, load_reference,
    reference, scenario_table_bytes, verify_java,
)
from physical_curriculum_v1.budget import _history_rows
from python.offline_split import JAVA_MASK, java_random_states
from weak_plan import bind


CAMPAIGN = 'bg128-or-milp-smoke-20261124-001'
SCHEMA = 'bg128-or-milp-smoke-v1'
POLICIES = ('rolling_quota', 'trip_fifo')
POLICY = 'or_milp_paired'
COUNT = len(POLICIES)
FIRST_RUN = 10036541
ENVIRONMENT_SEED = int.from_bytes(sha256(f'{CAMPAIGN}:environment:0'.encode('ascii')).digest()[:6], 'big')
METHOD = dict(planning_ms=800, response_target_ms=1000, policy_parameters={
    'rolling_quota': dict(replan_seconds=900, rolling=True, bootstrap_replan=False),
    'trip_fifo': dict(horizon_seconds=1800, max_trips=4, slot_seconds=90),
})
PREPARED = PLAN / 'prepared_smoke_001'
SUITE = PREPARED / 'suite.json'
OUTPUT = ROOT / 'runs' / CAMPAIGN
CONTROL = OUTPUT / 'control'
LEDGER = OUTPUT / 'budget.sqlite'
REPORT = grade.REPORT.parents[2] / PLAN.name / 'milp_smoke_evaluation.json'
OFFLINE = REPORT.with_name('milp_offline_review_001.json')
LAUNCHER = PLAN / 'milp_smoke_launch.py'
POLICY_FILES = (PLAN / 'rolling_quota_policy.py', PLAN / 'trip_fifo_policy.py')


def preceding_snapshot():
    return paired.budget_snapshot()


def make_job(ordinal, anchor):
    policy = POLICIES[ordinal]
    run = FIRST_RUN + ordinal
    recording = f'{CAMPAIGN}-{policy.replace("_", "-")}'
    leaf = f'run_{run}'
    events = OUTPUT / 'anylogic_events' / leaf
    return dict(ordinal=ordinal, pair_ordinal=0, policy_kind=policy, group='BASELINE', role='final',
                run_index=run, experiment_seed=ENVIRONMENT_SEED, recording_id=recording,
                source_recording_id=recording, actor_seed=0, actor_sampling_seed=0,
                actor_sampling_seed_consumed=False, target_training_update=0, checkpoint_update=0,
                event_directory=str(events), event_bundle=str(events / episode_id(run, ENVIRONMENT_SEED)),
                python_output=str(OUTPUT / 'shifts' / leaf), scene_id=anchor['scene_id'],
                expected_rpc_config=anchor['expected_rpc_config'],
                runtime_scene_sha256=anchor['runtime_scene_sha256'])


def check_fresh(jobs, historical):
    _require(CAMPAIGN not in historical['campaigns'] and historical['active_h2_reserved_shifts'] == 0,
             'MILP smoke already registered or historical reservation scope differs')
    rows = _history_rows(historical) + grade.flow.v3_ledger._prepared_rows(historical)
    for path in GRAPH.rglob('suite.json'):
        if path != SUITE:
            rows.extend(row for row in json.loads(path.read_bytes()).get('jobs', []) if isinstance(row, dict))
    for unit in grade.replication_budget.plan.UNITS:
        path = Path(grade.replication_budget.plan.locations(unit)['suite_path'])
        if path.is_file():
            with path.with_name('master_plan.csv').open(newline='') as stream:
                roster = list(csv.DictReader(stream))
            _require(len(roster) == grade.replication_budget.plan.CAP, 'prepared GAT1 sidecar row count differs')
            fields = ('run_index', *grade.flow.SEED_FIELDS)
            rows.extend(dict(row, **{key: int(row[key]) for key in fields}) for row in roster)
    runs = {row.get('run_index') for row in rows}
    recordings = {row.get(key) for row in rows for key in ('recording_id', 'source_recording_id')}
    seeds = {row[key] & JAVA_MASK for row in rows for key in grade.flow.SEED_FIELDS
             if type(row.get(key)) is int and row[key] > 0}
    prior_environment = {row['experiment_seed'] for row in rows
                         if type(row.get('experiment_seed')) is int and row['experiment_seed'] > 0}
    states = set(java_random_states(ENVIRONMENT_SEED).values())
    prior_states = {state for seed in prior_environment for state in java_random_states(seed).values()}
    _require(len(jobs) == COUNT and tuple(job['policy_kind'] for job in jobs) == POLICIES
             and {job['pair_ordinal'] for job in jobs} == {0}
             and {job['experiment_seed'] for job in jobs} == {ENVIRONMENT_SEED}
             and len({job['run_index'] for job in jobs}) == len({job['recording_id'] for job in jobs}) == COUNT
             and all(not job['actor_seed'] and not job['actor_sampling_seed'] for job in jobs),
             'MILP smoke requires two physical identities and one common fresh environment seed')
    _require(all(job['run_index'] not in runs and job['recording_id'] not in recordings for job in jobs)
             and 0 < ENVIRONMENT_SEED <= JAVA_MASK and ENVIRONMENT_SEED not in seeds
             and len(states) == 5 and states.isdisjoint(prior_states),
             'MILP smoke overlaps a historical or prepared physical identity or random stream')
    return dict(occupied_before=historical['maximum_total_shifts'],
                maximum_total_if_registered=historical['maximum_total_shifts'] + COUNT,
                new_physical_shifts=COUNT, new_environment_seeds=1, new_actor_sampling_seeds=0,
                new_java_rng_states=len(states), historical_and_prepared_rows_checked=len(rows))


def _render_java(old, master, scenarios):
    template = old['java_helper']
    _checked_reference(template)
    source = Path(template['path']).read_text(encoding='utf-8')
    values = dict(MASTER_PLAN_PATH=str(master), MASTER_PLAN_SHA256=reference(master)['sha256'],
                  SCENARIO_TABLE_PATH=str(scenarios), SCENARIO_TABLE_SHA256=reference(scenarios)['sha256'],
                  CONTROL_DIRECTORY=str(CONTROL), CAMPAIGN_ID=CAMPAIGN)
    for name, value in values.items():
        source, count = re.subn(f'(private static final String {name} = )"[^"]+";',
                               lambda match: match[1] + json.dumps(value) + ';', source)
        _require(count == 1, f'Java binding differs: {name}')
    replacements = (
        (str(grade.LAUNCHER), str(LAUNCHER)),
        (reference(grade.LAUNCHER)['sha256'], reference(LAUNCHER)['sha256']),
        (str(grade.SUITE), str(SUITE)),
        (f'private static final int JOB_COUNT = {grade.COUNT};', f'private static final int JOB_COUNT = {COUNT};'),
        ('                    require(environmentSeeds.add(row.seed), "environment seed repeats a FIFO DISPATCH slot");',
         '                    require(row.seed == ' + str(ENVIRONMENT_SEED) + 'L, "MILP smoke pair seed differs");'),
        ('                java.util.Set<Long> environmentSeeds = new java.util.HashSet<>();\n', ''),
        (f'row(index); return "{grade.POLICY}";',
         f'row(index); return index == 0 ? "{POLICIES[0]}" : "{POLICIES[1]}";'),
    )
    for before, after in replacements:
        source = _replace_once(source, before, after)
    return source


def _verify_java(helper, scenarios, jobs):
    proof = verify_java(helper, scenarios)
    values = ', '.join(f'"{job["recording_id"]}"' for job in jobs)
    policies = ', '.join(f'"{policy}"' for policy in POLICIES)
    source = f"""public final class MilpSmokeProbe {{
        public static void main(String[] args) {{
            String[] recordings = {{{values}}};
            String[] policies = {{{policies}}};
            if (Baseline1CampaignPlan.jobCount() != {COUNT}) throw new AssertionError();
            for (int i = 0; i < {COUNT}; i++) {{
                if (Baseline1CampaignPlan.runIndex(i) != {FIRST_RUN}L + i
                    || Baseline1CampaignPlan.environmentSeed(i) != {ENVIRONMENT_SEED}L
                    || Baseline1CampaignPlan.actorSamplingSeed(i) != 0L
                    || !Baseline1CampaignPlan.recordingId(i).equals(recordings[i])
                    || !Baseline1CampaignPlan.phaseId(i).equals(policies[i])
                    || Baseline1CampaignPlan.dispatchPolicy(i) != 3
                    || Baseline1CampaignPlan.eventRecordingEnabled(i)
                    || Baseline1CampaignPlan.materialEnabled(i)
                    || Baseline1CampaignPlan.offlineEnabled(i)) throw new AssertionError();
            }}
        }}
    }}
"""
    with TemporaryDirectory(prefix='or-milp-java-') as directory:
        probe = Path(directory) / 'MilpSmokeProbe.java'
        probe.write_text(source, encoding='ascii')
        subprocess.run(['javac', '-encoding', 'UTF-8', '-d', directory, str(helper), str(probe)], check=True)
        subprocess.run(['java', '-cp', directory, 'MilpSmokeProbe'], check=True)
    _require(proof['rows_checked'] == COUNT, 'nominal Java roster size differs')
    return dict(proof, pair_identities_checked=COUNT)


def _phase(job, source_pins, protected_core):
    return dict(schema=SCHEMA + '-phase', campaign_id=CAMPAIGN, policy_kind=job['policy_kind'],
                method=METHOD, execution=EXECUTION, source_pins=source_pins, protected_core=protected_core, job=job)


def _verify_offline(pin, policy_pins, protected_core):
    review = load_reference(pin)
    strategy_shas = {Path(row['path']).name: row['sha256'] for row in policy_pins}
    _require(review['schema'] == 'or-paper-milps-sealed-frame-review-v1'
             and review['physical_shifts_started'] == 0
             and review['protected_core_sha256'] == protected_core['sha256']
             and all(review['policy_sources'].get(name) == digest for name, digest in strategy_shas.items())
             and all(review['groups'][name]['decision_kinds'].get('planned', 0) > 0 for name in POLICIES),
             'sealed offline replay belongs to different strategy bytes or lacks planned decisions')


def prepare():
    _require(not any(path.exists() for path in (PREPARED, OUTPUT, REPORT)),
             'MILP smoke preparation or physical evidence already exists')
    old = grade.load_suite(check_sources=False)
    protected = reference(CORE)
    _require(protected == old['protected_core'] and protected['sha256'] == CORE_SHA256,
             'protected PPO core bytes differ')
    _checked_reference(old['java_helper'])
    review = reference(OFFLINE)
    _verify_offline(review, [reference(path) for path in POLICY_FILES], protected)
    paths = {Path(pin['path']) for pin in old['source_pins']}
    paths.update((*POLICY_FILES, Path(__file__).resolve(), LAUNCHER, PLAN / 'milp_smoke_runtime.py',
                  GRADE / 'grade_launch.py', PAIRED / 'paired_campaign.py', WEAK / 'weak_plan.py'))
    pins = [reference(path) for path in sorted(paths)]
    jobs = [make_job(ordinal, old['jobs'][0]) for ordinal in range(COUNT)]
    budget = check_fresh(jobs, preceding_snapshot())
    for job in jobs:
        phase_path = PREPARED / 'configs' / f"run_{job['run_index']}.json"
        job['phase_config'] = _publish_json(phase_path, _phase(job.copy(), pins, protected))
    master = PREPARED / 'master_plan.csv'
    scenarios = PREPARED / 'scenario_table.csv'
    master_ref = _publish_bytes(master, _csv_bytes(jobs, COLUMNS))
    scenario_ref = _publish_bytes(scenarios, scenario_table_bytes(jobs))
    helper = PREPARED / 'Baseline1CampaignPlan.java'
    java_ref = _publish_bytes(helper, _render_java(old, master, scenarios).encode('utf-8'))
    proof = _verify_java(helper, scenarios, jobs)
    suite = dict(schema=SCHEMA, campaign_id=CAMPAIGN, policy_kind=POLICY, method=METHOD, execution=EXECUTION,
                 protected_core=protected, source_pins=pins, master_plan=master_ref, scenario_table=scenario_ref,
                 java_helper=java_ref, java_verification=proof, jobs=jobs, grade_nominal_contract=reference(grade.SUITE),
                 offline_review=review,
                 control_directory=str(CONTROL), python_output=str(OUTPUT), report=str(REPORT), budget=budget)
    suite_ref = _publish_json(SUITE, suite)
    columns = ('policy_kind', 'run_index', 'experiment_seed', 'recording_id', 'config_sha256', 'python_output',
               'event_directory')
    rows = [dict(job, config_sha256=job['phase_config']['sha256']) for job in jobs]
    roster = _publish_bytes(PREPARED / 'run_seed.csv', _csv_bytes(rows, columns))
    return _publish_json(PREPARED / 'launch_identity.json', dict(
        campaign_id=CAMPAIGN, suite=suite_ref, run_seed=roster, java_helper=java_ref,
        budget=budget, physical_shifts=COUNT, optimizer_updates=0, jobs=jobs))


def load_suite(path=SUITE, *, check_sources=True):
    _require(Path(path).resolve() == SUITE, 'unexpected MILP smoke manifest path')
    suite = json.loads(SUITE.read_bytes())
    _require(suite['schema'] == SCHEMA and suite['campaign_id'] == CAMPAIGN and suite['policy_kind'] == POLICY
             and suite['method'] == METHOD and suite['execution'] == EXECUTION and len(suite['jobs']) == COUNT
             and suite['protected_core'] == dict(path=str(CORE), sha256=CORE_SHA256)
             and suite['control_directory'] == str(CONTROL) and suite['python_output'] == str(OUTPUT)
             and suite['report'] == str(REPORT), 'MILP smoke contract differs')
    if check_sources:
        for pin in [suite['protected_core'], suite['master_plan'], suite['scenario_table'], suite['java_helper'],
                    suite['grade_nominal_contract'], *suite['source_pins']]:
            _checked_reference(pin)
        _verify_offline(suite['offline_review'], [reference(path) for path in POLICY_FILES], suite['protected_core'])
    anchor = load_reference(suite['grade_nominal_contract'])['jobs'][0]
    for ordinal, row in enumerate(suite['jobs']):
        phase = load_reference(row['phase_config'])
        job = {key: value for key, value in row.items() if key != 'phase_config'}
        _require(phase == _phase(job, suite['source_pins'], suite['protected_core'])
                 and job == make_job(ordinal, anchor), 'phase configuration and frozen MILP run differ')
    return suite


class Ledger(grade.Ledger):
    register = bind(grade.Ledger.register, LEDGER=LEDGER, CAMPAIGN=CAMPAIGN, SUITE=SUITE, COUNT=COUNT,
                    check_fresh=check_fresh, preceding_snapshot=preceding_snapshot)
    snapshot = bind(grade.Ledger.snapshot, LEDGER=LEDGER, CAMPAIGN=CAMPAIGN, SUITE=SUITE, COUNT=COUNT)
    transition = bind(grade.Ledger.transition, LEDGER=LEDGER)


budget_snapshot = bind(grade.budget_snapshot, LEDGER=LEDGER, COUNT=COUNT, CAMPAIGN=CAMPAIGN,
                       Ledger=Ledger, load_suite=load_suite, preceding_snapshot=preceding_snapshot)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', required=True, action='store_true')
    parser.parse_args()
    print(json.dumps(prepare(), ensure_ascii=False))
