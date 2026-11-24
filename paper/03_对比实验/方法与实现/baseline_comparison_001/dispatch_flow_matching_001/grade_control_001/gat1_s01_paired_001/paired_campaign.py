"""Reserve ten GAT1 s01/u32 evaluation slots paired to completed grade-control shifts."""
import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import re
import sys


PLAN = Path(__file__).resolve().parent
ROOT = PLAN.parents[6]
GRAPH = PLAN.parents[3]
WEAK = GRAPH / 'gat_weak_ablation_001'
for directory in (ROOT, PLAN.parent, WEAK):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import grade_campaign as grade
from grade_campaign import (
    CORE, CORE_SHA256, COLUMNS, EXECUTION, LOCKS, PYTHON, _checked_reference, _csv_bytes,
    _publish_bytes, _publish_json, _replace_once, _require, encoded, episode_id, load_reference,
    reference, scenario_table_bytes, verify_java,
)
from physical_curriculum_v1.budget import _history_rows
from python.offline_split import JAVA_MASK, java_random_states
from weak_plan import bind


CAMPAIGN = 'bg128-gat1-s01-u32-grade-paired10-20261123-001'
SCHEMA = 'bg128-gat1-s01-u32-grade-paired10-v1'
POLICY = 'gat1_s01_u32'
COUNT = 10
FIRST_RUN = 10036531
METHOD = dict(response_target_ms=1000, actor_sampling='torch_multinomial_cpu')
PREPARED = PLAN / 'prepared_001'
SUITE = PREPARED / 'suite.json'
OUTPUT = ROOT / 'runs' / CAMPAIGN
CONTROL = OUTPUT / 'control'
LEDGER = OUTPUT / 'budget.sqlite'
REPORT = grade.REPORT.parent / PLAN.name / 'paired_evaluation.json'
LAUNCHER = PLAN / 'paired_launch.py'
GRADE_COMPLETE = grade.OUTPUT / 'suite_complete.json'
GAT_COMPLETE = ROOT / 'runs/bg128-gat-weak-20261122-gat1-s01/suite_complete.json'
ACTOR_SHA256 = '49e9c62d4851527ff3beaef3309b30112b76438b6b6b6175f93c40c94f955172'
CHECKPOINT_SHA256 = 'ac0ecd0443f847f2acbacf08768031802d195a009b804d4c130591f564bdcf00'
ACTOR_DIGEST = 'bdb3a64f8ff6fb5e50bc0340f0a73cea3103394fa78ece847c773e31b3782f34'


def preceding_snapshot():
    return grade.budget_snapshot()


def _anchors():
    old = grade.load_suite(check_sources=False)
    completion = reference(GRADE_COMPLETE)
    complete = load_reference(completion)
    _require(complete['status'] == 'COMPLETE' and complete['physical_shifts'] == COUNT
             and complete['campaign_id'] == grade.CAMPAIGN and complete['suite'] == reference(grade.SUITE)
             and complete['optimizer_updates'] == 0, 'grade ten is not sealed and complete')
    report = load_reference(complete['evaluation'])
    _, slots = grade.Ledger(old).snapshot()
    refs, rows = report['receipts'], report['shifts']
    _require(report['status'] == 'COMPLETE' and len(refs) == len(rows) == len(slots) == COUNT,
             'grade evaluation does not cover the ten original slots')
    for job, pin, row in zip(old['jobs'], refs, rows, strict=True):
        slot = slots[str(job['run_index'])]
        _require(slot == dict(identity=job, status='COMPLETE', evidence=pin) and load_reference(pin) == row
                 and pin['path'] == str(Path(job['python_output']) / 'evaluation_receipt.json')
                 and row['normal_completion'] and row['actor_sampling_seed'] == 0
                 and all(row[key] == job[key] for key in
                         ('run_index', 'recording_id', 'experiment_seed', 'phase_config', 'scene_id', 'policy_kind'))
                 and all(row['actual_rpc_configuration'].get(key) == value
                         for key, value in job['expected_rpc_config'].items()),
                 'grade slot receipt, actual RPC configuration and ledger differ')
    return old, completion, complete['evaluation'], refs, rows


def load_anchors():
    return _anchors()[-1]


def _actor_source():
    from weak_plan import load_package
    from weak_runtime import source_files

    original, _ = load_package('gat1-s01')
    complete_ref = reference(GAT_COMPLETE)
    complete = load_reference(complete_ref)
    original_ref = reference(original['suite_path'])
    _require(complete['status'] == 'COMPLETE' and complete['suite'] == original_ref
             and complete['training_updates'] == 32 and complete['evaluation_optimizer_updates'] == 0,
             'the s01 weak GAT activity lacks a completed u32 freeze')
    nominal_ref = complete['phase_results'][0]['report']
    nominal = load_reference(nominal_ref)
    actor, checkpoint = nominal['actor_artifact'], complete['new_checkpoint']
    _require(nominal['scenario_id'] == 'nominal' and nominal['training_state_unchanged']
             and nominal['evaluation_optimizer_updates'] == 0 and nominal['checkpoint_provenance'] == checkpoint
             and checkpoint['sha256'] == CHECKPOINT_SHA256 and actor['sha256'] == ACTOR_SHA256
             and nominal['fingerprints']['actor'] == ACTOR_DIGEST,
             's01 nominal Actor does not derive from the specified one-round GAT u32 checkpoint')
    manifest_ref = reference(actor['path'] + '.manifest.json')
    manifest = load_reference(manifest_ref)
    _require(manifest['candidate_id'] == 's01_BG128' and manifest['update_index'] == 32
             and manifest['actor_digest'] == ACTOR_DIGEST and manifest['sha256'] == ACTOR_SHA256,
             's01 nominal Actor manifest differs')
    source = dict(actor=actor, actor_digest=ACTOR_DIGEST, actor_manifest=manifest_ref, checkpoint=checkpoint,
                  source_suite=original_ref, source_complete=complete_ref, source_nominal=nominal_ref)
    for pin in (actor, checkpoint):
        _checked_reference(pin)
    return original, source, tuple(source_files())


def load_actor(suite):
    import torch
    from weak_runtime import runtime_scope

    torch.set_num_threads(2)
    original, source, _ = _actor_source()
    _require(all(suite[key] == value for key, value in source.items()), 'paired Actor freeze differs')
    with runtime_scope(original) as runtime:
        actor, digest = runtime.frozen_actor(source['actor']['path'])
    _require(digest == ACTOR_DIGEST and not actor.training and all(not p.requires_grad for p in actor.parameters()),
             's01 actor is neither frozen nor the expected state')
    return actor, digest


def make_job(ordinal, anchor_job, anchor_ref):
    run = FIRST_RUN + ordinal
    seed = anchor_job['experiment_seed']
    sample = int.from_bytes(sha256(f'{CAMPAIGN}:actor-sampling:{ordinal}'.encode('ascii')).digest()[:6], 'big')
    recording = f'{CAMPAIGN}-{ordinal:03d}'
    events = OUTPUT / 'anylogic_events' / f'run_{run}'
    return dict(ordinal=ordinal, pair_ordinal=ordinal, policy_kind=POLICY, group='BASELINE', role='final',
                run_index=run, experiment_seed=seed, recording_id=recording, source_recording_id=recording,
                actor_seed=0, actor_sampling_seed=sample, actor_sampling_seed_consumed=True,
                target_training_update=0, checkpoint_update=32, event_directory=str(events),
                event_bundle=str(events / episode_id(run, seed)), python_output=str(OUTPUT / 'shifts' / f'run_{run}'),
                scene_id=anchor_job['scene_id'], expected_rpc_config=anchor_job['expected_rpc_config'],
                runtime_scene_sha256=anchor_job['runtime_scene_sha256'], paired_anchor=anchor_ref,
                paired_reference_run_index=anchor_job['run_index'])


def check_fresh(jobs, history):
    source, _, _, refs, _ = _anchors()
    original = {job['run_index']: job for job in source['jobs']}
    requested = [{key: value for key, value in job.items() if key != 'phase_config'} for job in jobs]
    canonical = [make_job(i, job, pin) for i, (job, pin) in enumerate(zip(source['jobs'], refs, strict=True))]
    _require(CAMPAIGN not in history['campaigns'] and history['active_h2_reserved_shifts'] == 0
             and all(history['jobs'].get(str(run), {}).get('identity') == job
                     and history['jobs'][str(run)].get('campaign_id') == grade.CAMPAIGN
                     and history['jobs'][str(run)].get('status') == 'COMPLETE' for run, job in original.items())
             and requested == canonical,
             'paired job or its registered grade source differs')
    rows = _history_rows(history) + grade.flow.v3_ledger._prepared_rows(history)
    for path in GRAPH.rglob('suite.json'):
        if path != SUITE:
            rows.extend(row for row in json.loads(path.read_bytes()).get('jobs', []) if isinstance(row, dict))
    for unit in grade.replication_budget.plan.UNITS:
        path = Path(grade.replication_budget.plan.locations(unit)['suite_path'])
        if path.is_file():
            with path.with_name('master_plan.csv').open(newline='') as stream:
                plan_rows = list(csv.DictReader(stream))
            _require(len(plan_rows) == grade.replication_budget.plan.CAP, 'GAT1 prepared sidecar row count differs')
            rows.extend(dict(row, **{key: int(row[key]) for key in ('run_index', *grade.flow.SEED_FIELDS)})
                        for row in plan_rows)
    foreign = [row for row in rows if row != original.get(row.get('run_index'))]
    used_runs = {row.get('run_index') for row in rows}
    used_records = {row.get(key) for row in rows for key in ('recording_id', 'source_recording_id')}
    used_seeds = {row[key] & JAVA_MASK for row in rows for key in grade.flow.SEED_FIELDS
                  if type(row.get(key)) is int and row[key] > 0}
    foreign_seeds = {row[key] & JAVA_MASK for row in foreign for key in grade.flow.SEED_FIELDS
                     if type(row.get(key)) is int and row[key] > 0}
    foreign_environment = {row['experiment_seed'] for row in foreign
                           if type(row.get('experiment_seed')) is int and row['experiment_seed'] > 0}
    foreign_java = {state for seed in foreign_environment for state in java_random_states(seed).values()}
    environment = {job['experiment_seed'] for job in jobs}
    samples = {job['actor_sampling_seed'] for job in jobs}
    paired_java = {state for seed in environment for state in java_random_states(seed).values()}
    _require(len(jobs) == len(environment) == len(samples) == COUNT and min(samples) > 0
             and samples.isdisjoint(used_seeds | environment)
             and environment.isdisjoint(foreign_seeds) and len(paired_java) == 5 * COUNT
             and paired_java.isdisjoint(foreign_java)
             and all(job['run_index'] not in used_runs and job['recording_id'] not in used_records for job in jobs),
             'reuse extends outside the ten grade slots or a fresh physical or actor identity conflicts')
    return dict(occupied_before=history['maximum_total_shifts'],
                maximum_total_if_registered=history['maximum_total_shifts'] + COUNT,
                new_physical_shifts=COUNT, reused_grade_environment_seeds=COUNT,
                new_actor_sampling_seeds=COUNT, paired_java_rng_states=len(paired_java),
                historical_and_prepared_rows_checked=len(rows))


def _render_java(old, master, scenarios):
    pin = old['java_helper']
    _checked_reference(pin)
    source = Path(pin['path']).read_text()
    values = dict(MASTER_PLAN_PATH=str(master), MASTER_PLAN_SHA256=reference(master)['sha256'],
                  SCENARIO_TABLE_PATH=str(scenarios), SCENARIO_TABLE_SHA256=reference(scenarios)['sha256'],
                  CONTROL_DIRECTORY=str(CONTROL), CAMPAIGN_ID=CAMPAIGN)
    for name, value in values.items():
        source, found = re.subn(f'(private static final String {name} = )"[^"]+";',
                                lambda match: match[1] + json.dumps(value) + ';', source)
        _require(found == 1, f'Java binding differs: {name}')
    for original, replacement in (
        (str(grade.LAUNCHER), str(LAUNCHER)),
        (reference(grade.LAUNCHER)['sha256'], reference(LAUNCHER)['sha256']),
        (str(grade.SUITE), str(SUITE)),
        (f'return "{grade.POLICY}";', f'return "{POLICY}";'),
        ('require(values[8].equals("0"), "classical policy must not consume a sampling seed");',
         'positive(values[8], "actor_sampling_seed");'),
    ):
        source = _replace_once(source, original, replacement)
    return source


def _phase(job, suite):
    return dict(schema=SCHEMA + '-phase', campaign_id=CAMPAIGN, policy_kind=POLICY, method=METHOD,
                execution=EXECUTION, source_pins=suite['source_pins'], protected_core=suite['protected_core'],
                actor=suite['actor'], actor_digest=suite['actor_digest'], checkpoint=suite['checkpoint'], job=job)


def prepare():
    _require(not any(path.exists() for path in (PREPARED, OUTPUT, REPORT)), 'paired package or physical evidence exists')
    old, grade_completion, grade_report, refs, _ = _anchors()
    _, actor, sources = _actor_source()
    jobs = [make_job(i, job, anchor) for i, (job, anchor) in enumerate(zip(old['jobs'], refs, strict=True))]
    budget = check_fresh(jobs, preceding_snapshot())
    load_actor(actor)
    paths = {pin['path'] for pin in old['source_pins']}
    paths.update(map(str, (*sources, Path(__file__).resolve(), LAUNCHER, WEAK / 'weak_plan.py')))
    suite = dict(schema=SCHEMA, campaign_id=CAMPAIGN, policy_kind=POLICY, method=METHOD, execution=EXECUTION,
                 protected_core=old['protected_core'], source_pins=[reference(path) for path in sorted(paths)],
                 grade_completion=grade_completion, grade_report=grade_report, jobs=jobs, budget=budget,
                 control_directory=str(CONTROL), python_output=str(OUTPUT), report=str(REPORT), **actor)
    _require(reference(CORE) == suite['protected_core'], 'protected PPO core bytes differ')
    for job in jobs:
        path = PREPARED / 'configs' / f"run_{job['run_index']}.json"
        job['phase_config'] = _publish_json(path, _phase(job.copy(), suite))
    master, scenarios = PREPARED / 'master_plan.csv', PREPARED / 'scenario_table.csv'
    suite['master_plan'] = _publish_bytes(master, _csv_bytes(jobs, COLUMNS))
    suite['scenario_table'] = _publish_bytes(scenarios, scenario_table_bytes(jobs))
    helper = PREPARED / 'Baseline1CampaignPlan.java'
    suite['java_helper'] = _publish_bytes(helper, _render_java(old, master, scenarios).encode('utf-8'))
    suite['java_verification'] = verify_java(helper, scenarios)
    _require(suite['java_verification']['rows_checked'] == COUNT, 'Java did not verify the ten slots')
    suite_ref = _publish_json(SUITE, suite)
    columns = ('pair_ordinal', 'paired_reference_run_index', 'run_index', 'experiment_seed', 'actor_sampling_seed',
               'recording_id', 'config_sha256', 'python_output', 'event_directory')
    roster = _publish_bytes(PREPARED / 'run_seed.csv', _csv_bytes(
        [dict(job, config_sha256=job['phase_config']['sha256']) for job in jobs], columns))
    return _publish_json(PREPARED / 'launch_identity.json', dict(campaign_id=CAMPAIGN, suite=suite_ref,
        run_seed=roster, java_helper=suite['java_helper'], budget=budget, physical_shifts=COUNT,
        optimizer_updates=0, checkpoint=suite['checkpoint'], jobs=jobs))


def load_suite(path=SUITE, *, check_sources=True):
    _require(Path(path).resolve() == SUITE, 'unexpected paired manifest path')
    suite = json.loads(SUITE.read_bytes())
    _require(suite['schema'] == SCHEMA and suite['campaign_id'] == CAMPAIGN and suite['policy_kind'] == POLICY
             and suite['method'] == METHOD and suite['execution'] == EXECUTION and len(suite['jobs']) == COUNT
             and suite['protected_core'] == dict(path=str(CORE), sha256=CORE_SHA256)
             and suite['control_directory'] == str(CONTROL) and suite['report'] == str(REPORT),
             'paired evaluation contract differs')
    if check_sources:
        fields = ('protected_core', 'master_plan', 'scenario_table', 'java_helper', 'actor', 'actor_manifest',
                  'checkpoint', 'source_suite', 'source_complete', 'source_nominal', 'grade_completion', 'grade_report')
        for pin in [*(suite[key] for key in fields), *suite['source_pins']]:
            _checked_reference(pin)
    old, complete, report, refs, _ = _anchors()
    _require(suite['grade_completion'] == complete and suite['grade_report'] == report, 'grade anchors changed')
    for ordinal, (row, anchor_job, pin) in enumerate(zip(suite['jobs'], old['jobs'], refs, strict=True)):
        job = {key: value for key, value in row.items() if key != 'phase_config'}
        _require(job == make_job(ordinal, anchor_job, pin)
                 and load_reference(row['phase_config']) == _phase(job, suite), 'paired phase configuration differs')
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
