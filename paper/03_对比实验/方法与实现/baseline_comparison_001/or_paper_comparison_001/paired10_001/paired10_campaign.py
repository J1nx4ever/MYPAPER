"""Freeze ten nominal environment slots for both deterministic online MILPs."""

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

PLAN = Path(__file__).resolve().parent
if str(PLAN.parent) not in sys.path:
    sys.path.insert(0, str(PLAN.parent))

import milp_smoke_campaign as smoke
from milp_smoke_campaign import (
    CORE, CORE_SHA256, COLUMNS, EXECUTION, GRAPH, JAVA_MASK, LOCKS, METHOD, POLICY,
    POLICIES, POLICY_FILES, PYTHON, ROOT, _checked_reference, _csv_bytes, _history_rows,
    _publish_bytes, _publish_json, _replace_once, _require, bind, episode_id, grade,
    java_random_states, load_reference, reference, scenario_table_bytes, verify_java,
)


CAMPAIGN = 'bg128-or-milp-paired10-20261124-001'
SCHEMA = 'bg128-or-milp-paired10-v1'
PAIR_COUNT = 10
COUNT = PAIR_COUNT * len(POLICIES)
FIRST_RUN = 10036601
ENVIRONMENT_SEEDS = tuple(
    int.from_bytes(sha256(f'{CAMPAIGN}:environment:{slot}'.encode('ascii')).digest()[:6], 'big')
    for slot in range(PAIR_COUNT)
)
PREPARED = PLAN / 'prepared_001'
SUITE = PREPARED / 'suite.json'
OUTPUT = ROOT / 'runs' / CAMPAIGN
CONTROL = OUTPUT / 'control'
LEDGER = OUTPUT / 'budget.sqlite'
REPORT = smoke.REPORT.parent / PLAN.name / 'milp_paired10_evaluation.json'
LAUNCHER = PLAN / 'paired10_launch.py'
OFFLINE = smoke.OFFLINE


preceding_snapshot = smoke.budget_snapshot


def make_job(ordinal, anchor):
    slot, arm = divmod(ordinal, len(POLICIES))
    policy = POLICIES[arm]
    run = FIRST_RUN + ordinal
    seed = ENVIRONMENT_SEEDS[slot]
    recording = f'{CAMPAIGN}-{slot:02d}-{policy.replace("_", "-")}'
    leaf = f'run_{run}'
    events = OUTPUT / 'anylogic_events' / leaf
    return dict(ordinal=ordinal, pair_ordinal=slot, policy_kind=policy, group='BASELINE', role='final',
                run_index=run, experiment_seed=seed, recording_id=recording, source_recording_id=recording,
                actor_seed=0, actor_sampling_seed=0, actor_sampling_seed_consumed=False,
                target_training_update=0, checkpoint_update=0, event_directory=str(events),
                event_bundle=str(events / episode_id(run, seed)), python_output=str(OUTPUT / 'shifts' / leaf),
                scene_id=anchor['scene_id'], expected_rpc_config=anchor['expected_rpc_config'],
                runtime_scene_sha256=anchor['runtime_scene_sha256'])


def _prior_rows(historical):
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
    return rows


def check_fresh(jobs, historical):
    _require(CAMPAIGN not in historical['campaigns'] and historical['active_h2_reserved_shifts'] == 0,
             'paired MILP campaign is already registered or historical budget policy differs')
    _require(len(jobs) == COUNT, 'paired MILP campaign requires exactly twenty physical jobs')
    for ordinal, job in enumerate(jobs):
        expected = make_job(ordinal, jobs[0])
        _require({key: value for key, value in job.items() if key != 'phase_config'} == expected,
                 f'paired MILP identity differs at ordinal {ordinal}')
    rows = _prior_rows(historical)
    runs = {row.get('run_index') for row in rows}
    recordings = {row.get(key) for row in rows for key in ('recording_id', 'source_recording_id')}
    prior_seeds = {row[key] & JAVA_MASK for row in rows for key in grade.flow.SEED_FIELDS
                   if type(row.get(key)) is int and row[key] > 0}
    prior_environment = {row['experiment_seed'] for row in rows
                         if type(row.get('experiment_seed')) is int and row['experiment_seed'] > 0}
    prior_states = {state for seed in prior_environment for state in java_random_states(seed).values()}
    seeds = set(ENVIRONMENT_SEEDS)
    states = {state for seed in seeds for state in java_random_states(seed).values()}
    _require(len(seeds) == PAIR_COUNT and all(0 < seed <= JAVA_MASK for seed in seeds)
             and len(states) == 5 * PAIR_COUNT, 'paired MILP slots must have distinct valid Java seeds and states')
    _require(all(job['run_index'] not in runs and job['recording_id'] not in recordings for job in jobs)
             and seeds.isdisjoint(prior_seeds) and states.isdisjoint(prior_states),
             'paired MILP roster overlaps a historical or prepared physical identity or random state')
    return dict(occupied_before=historical['maximum_total_shifts'],
                maximum_total_if_registered=historical['maximum_total_shifts'] + COUNT,
                new_physical_shifts=COUNT, new_environment_seeds=PAIR_COUNT, new_actor_sampling_seeds=0,
                new_java_rng_states=len(states), historical_and_prepared_rows_checked=len(rows))


def _render_java(old, master, scenarios):
    _checked_reference(old['java_helper'])
    source = Path(old['java_helper']['path']).read_text(encoding='utf-8')
    for before, after in ((old['master_plan'], reference(master)),
                          (old['scenario_table'], reference(scenarios))):
        for key in ('path', 'sha256'):
            source = _replace_once(source, json.dumps(before[key]), json.dumps(after[key]))
    seeds = ', '.join(f'{seed}L' for seed in ENVIRONMENT_SEEDS)
    replacements = (
        (json.dumps(str(smoke.CONTROL)), json.dumps(str(CONTROL))),
        (json.dumps(smoke.CAMPAIGN), json.dumps(CAMPAIGN)),
        (json.dumps(str(smoke.LAUNCHER)), json.dumps(str(LAUNCHER))),
        (json.dumps(reference(smoke.LAUNCHER)['sha256']), json.dumps(reference(LAUNCHER)['sha256'])),
        (json.dumps(str(smoke.SUITE)), json.dumps(str(SUITE))),
        (f'private static final int JOB_COUNT = {smoke.COUNT};',
         f'private static final int JOB_COUNT = {COUNT};\n'
         f'    private static final long[] PAIR_SEEDS = {{{seeds}}};'),
        (f'require(row.seed == {smoke.ENVIRONMENT_SEED}L, "MILP smoke pair seed differs");',
         'require(row.seed == PAIR_SEEDS[i / 2], "MILP paired slot seed differs");'),
        ('row(index); return index == 0 ? "rolling_quota" : "trip_fifo";',
         'row(index); return index % 2 == 0 ? "rolling_quota" : "trip_fifo";'),
    )
    for before, after in replacements:
        source = _replace_once(source, before, after)
    return source


def _verify_java(helper, scenarios, jobs):
    proof = verify_java(helper, scenarios)
    recordings = ', '.join(json.dumps(job['recording_id']) for job in jobs)
    seeds = ', '.join(f'{job["experiment_seed"]}L' for job in jobs)
    source = f"""public final class MilpPaired10Probe {{
        public static void main(String[] args) {{
            String[] recordings = {{{recordings}}};
            long[] seeds = {{{seeds}}};
            if (Baseline1CampaignPlan.jobCount() != {COUNT}) throw new AssertionError();
            for (int i = 0; i < {COUNT}; i++) {{
                String policy = i % 2 == 0 ? "rolling_quota" : "trip_fifo";
                if (Baseline1CampaignPlan.runIndex(i) != {FIRST_RUN}L + i
                    || Baseline1CampaignPlan.environmentSeed(i) != seeds[i]
                    || Baseline1CampaignPlan.actorSamplingSeed(i) != 0L
                    || !Baseline1CampaignPlan.recordingId(i).equals(recordings[i])
                    || !Baseline1CampaignPlan.phaseId(i).equals(policy)
                    || Baseline1CampaignPlan.dispatchPolicy(i) != 3
                    || Baseline1CampaignPlan.eventRecordingEnabled(i)
                    || Baseline1CampaignPlan.materialEnabled(i)
                    || Baseline1CampaignPlan.offlineEnabled(i)) throw new AssertionError();
            }}
        }}
    }}
"""
    with TemporaryDirectory(prefix='or-milp-paired10-java-') as directory:
        probe = Path(directory) / 'MilpPaired10Probe.java'
        probe.write_text(source, encoding='ascii')
        subprocess.run(['javac', '-encoding', 'UTF-8', '-d', directory, str(helper), str(probe)], check=True)
        subprocess.run(['java', '-cp', directory, 'MilpPaired10Probe'], check=True)
    _require(proof['rows_checked'] == COUNT, 'nominal Java roster size differs')
    return dict(proof, pair_identities_checked=COUNT)


_phase = bind(smoke._phase, CAMPAIGN=CAMPAIGN, SCHEMA=SCHEMA)


def prepare():
    _require(not any(path.exists() for path in (PREPARED, OUTPUT, REPORT)),
             'paired MILP preparation or physical evidence already exists')
    old = smoke.load_suite()
    protected = old['protected_core']
    nominal = old['grade_nominal_contract']
    anchor = load_reference(nominal)['jobs'][0]
    review = old['offline_review']
    paths = {Path(pin['path']) for pin in old['source_pins']}
    paths.update((Path(__file__).resolve(), LAUNCHER, PLAN / 'paired10_runtime.py',
                  PLAN / 'paired10_statistics.py',
                  PLAN.parents[1] / 'paired_v2_100_001' / 'formal_report.py',
                  PLAN.parents[1] / 'paired_100_001' / 'paired_report.py'))
    pins = [reference(path) for path in sorted(paths)]
    jobs = [make_job(ordinal, anchor) for ordinal in range(COUNT)]
    budget = check_fresh(jobs, preceding_snapshot())
    for job in jobs:
        path = PREPARED / 'configs' / f"run_{job['run_index']}.json"
        job['phase_config'] = _publish_json(path, _phase(job.copy(), pins, protected))
    master = PREPARED / 'master_plan.csv'
    scenarios = PREPARED / 'scenario_table.csv'
    master_ref = _publish_bytes(master, _csv_bytes(jobs, COLUMNS))
    scenario_ref = _publish_bytes(scenarios, scenario_table_bytes(jobs))
    helper = PREPARED / 'Baseline1CampaignPlan.java'
    java_ref = _publish_bytes(helper, _render_java(old, master, scenarios).encode('utf-8'))
    proof = _verify_java(helper, scenarios, jobs)
    suite = dict(schema=SCHEMA, campaign_id=CAMPAIGN, policy_kind=POLICY, method=METHOD,
                 execution=EXECUTION, evaluation_slots=PAIR_COUNT, protected_core=protected, source_pins=pins,
                 master_plan=master_ref, scenario_table=scenario_ref, java_helper=java_ref,
                 java_verification=proof, jobs=jobs, grade_nominal_contract=nominal,
                 smoke_campaign=reference(smoke.SUITE), offline_review=review, control_directory=str(CONTROL),
                 python_output=str(OUTPUT), report=str(REPORT), budget=budget)
    suite_ref = _publish_json(SUITE, suite)
    columns = ('pair_ordinal', 'policy_kind', 'run_index', 'experiment_seed', 'recording_id',
               'config_sha256', 'python_output', 'event_directory')
    roster = _publish_bytes(PREPARED / 'run_seed.csv', _csv_bytes(
        [dict(job, config_sha256=job['phase_config']['sha256']) for job in jobs], columns))
    return _publish_json(PREPARED / 'launch_identity.json', dict(
        status='PREPARED_NOT_STARTED', campaign_id=CAMPAIGN, suite=suite_ref, run_seed=roster,
        java_helper=java_ref, budget=budget, evaluation_slots=PAIR_COUNT, physical_shifts=COUNT,
        registered_shifts=0, completed_shifts=0, optimizer_updates=0, jobs=jobs))


_load_frozen_suite = bind(smoke.load_suite, SUITE=SUITE, SCHEMA=SCHEMA, CAMPAIGN=CAMPAIGN, COUNT=COUNT,
                          CONTROL=CONTROL, OUTPUT=OUTPUT, REPORT=REPORT, make_job=make_job, _phase=_phase)


def load_suite(path=SUITE, *, check_sources=True):
    suite = _load_frozen_suite(path, check_sources=check_sources)
    _require(suite['evaluation_slots'] == PAIR_COUNT and suite['python_output'] == str(OUTPUT),
             'paired MILP suite slot count or output directory differs')
    if check_sources:
        old = load_reference(suite['smoke_campaign'])
        _require(suite['smoke_campaign']['path'] == str(smoke.SUITE)
                 and suite['grade_nominal_contract'] == old['grade_nominal_contract']
                 and suite['offline_review'] == old['offline_review']
                 and all(pin in suite['source_pins'] for pin in old['source_pins']),
                 'paired MILP suite differs from the reviewed smoke configuration')
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
