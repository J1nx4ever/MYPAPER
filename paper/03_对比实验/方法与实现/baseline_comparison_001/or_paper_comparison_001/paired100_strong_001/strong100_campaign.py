"""Freeze 100 historical nominal slots for two new deterministic controllers."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


PLAN = Path(__file__).resolve().parent
OR_PLAN = PLAN.parent
BASELINES = OR_PLAN.parent
GRAPH = BASELINES.parent
for directory in (OR_PLAN, OR_PLAN / 'paired10_001', BASELINES / 'paired_v2_100_001'):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import milp_smoke_campaign as smoke
import paired10_campaign as paired10
import formal_plan as formal
import v3_plan as v3
from milp_smoke_campaign import (
    CORE, CORE_SHA256, COLUMNS, EXECUTION, JAVA_MASK, LOCKS, PYTHON, ROOT, _checked_reference,
    _csv_bytes, _publish_bytes, _publish_json, _replace_once, _require, bind, episode_id, grade,
    java_random_states, load_reference, reference, scenario_table_bytes, verify_java,
)
from change_plans.token_attention_001.business_graph_summary_009.fleet_outage_003 import (
    campaign as recent_budget,
)


CAMPAIGN = 'bg128-or-strong-paired100-20261124-001'
SCHEMA = 'bg128-or-strong-paired100-v1'
PAIR_COUNT = SLOTS = 100
POLICIES = ('dispatch_grade', 'rolling_quota')
POLICY = 'or_strong_paired'
COUNT = SLOTS * len(POLICIES)
FIRST_RUN = 10036701
METHOD = dict(planning_ms=800, response_target_ms=1000, policy_parameters={
    'dispatch_grade': {name: grade.METHOD[name]
                       for name in ('replan_seconds', 'match_trucks', 'soft_replan_seconds')},
    'rolling_quota': dict(paired10.METHOD['policy_parameters']['rolling_quota']),
})
PREPARED = PLAN / 'prepared_001'
SUITE = PREPARED / 'suite.json'
OUTPUT = ROOT / 'runs' / CAMPAIGN
CONTROL = OUTPUT / 'control'
LEDGER = OUTPUT / 'budget.sqlite'
REPORT = smoke.REPORT.parent / PLAN.name / 'strong100_evaluation.json'
COMPUTE_PROBE = REPORT.with_name('same_host_compute_probe.json')
LAUNCHER = PLAN / 'strong100_launch.py'
BASELINE_ALP = ROOT / 'baseline' / 'DRL-PPO.alp'
FULL_ACTOR_SHA256 = 'e1575fec566c647a882c63934ec57c908cef28d43f076431ef9103cc9b73e0a9'
PAIRING = dict(scope='sealed_formal_v2_nominal_slots', slots=SLOTS, share='experiment_seed',
               policies=list(POLICIES), order='slot_then_policy', anchor_policy='bg128')


preceding_snapshot = recent_budget.budget_snapshot


def _completed_inventory(plan):
    """Read a finalized campaign and authenticate all physical receipt bytes."""
    sources = dict(
        suite=reference(plan.SUITE),
        completion=reference(plan.OUTPUT / 'suite_complete.json'),
        report=reference(plan.REPORT / plan.REPORT_FILENAME),
    )
    completion = load_reference(sources['completion'])
    suite = plan.load_suite(check_sources=False)
    report = load_reference(sources['report'])
    if plan is formal:
        _require(suite['actor']['sha256'] == FULL_ACTOR_SHA256, 'historical Full s02/u32 Actor differs')
        _checked_reference(suite['actor'])
    _require(completion['status'] == report['status'] == 'COMPLETE'
             and completion['suite'] == sources['suite'] and completion['evaluation'] == sources['report']
             and completion['campaign_id'] == report['campaign_id'] == plan.CAMPAIGN
             and completion['physical_shifts'] == report['physical_shifts'] == report['completed_shifts'] == plan.COUNT
             and completion['evaluation_slots'] == report['evaluation_slots'] == SLOTS
             and completion['optimizer_updates'] == completion['training_shifts'] == report['optimizer_updates'] == 0
             and completion['protected_core'] == suite['protected_core']
             == dict(path=str(CORE), sha256=CORE_SHA256),
             'historical completion seal differs from the frozen physical campaign')
    _require(len(report['shifts']) == len(report['receipts']) == len(suite['jobs']) == plan.COUNT,
             'historical campaign has incomplete physical evidence')
    by_slot_policy = {}
    fields = ('run_index', 'recording_id', 'experiment_seed', 'actor_sampling_seed', 'policy_kind',
              'phase_config', 'pair_ordinal', 'scene_id')
    for job, row, receipt in zip(suite['jobs'], report['shifts'], report['receipts'], strict=True):
        key = (job['pair_ordinal'], job['policy_kind'])
        _require(key not in by_slot_policy
                 and receipt['path'] == str(Path(job['python_output']) / 'evaluation_receipt.json')
                 and row == load_reference(receipt)
                 and all(row[name] == job[name] for name in fields)
                 and row['status'] == 'COMPLETE' and row['normal_completion'] is True
                 and row['completed_shifts'] == 1 and row['optimizer_updates'] == 0
                 and row['scene_id'] == 'nominal',
                 'historical receipt differs from its sealed nominal slot')
        by_slot_policy[key] = (job, row, receipt)
    return sources, by_slot_policy


def historical_inventory():
    v2_refs, v2_slots = _completed_inventory(formal)
    v3_refs, v3_slots = _completed_inventory(v3)
    expected = set(range(SLOTS))
    _require(set(v2_slots) == {(slot, policy) for slot in expected for policy in formal.STRATEGIES}
             and set(v3_slots) == {(slot, v3.POLICY) for slot in expected},
             'historical policy or slot roster differs')
    common = ('experiment_seed', 'actual_rpc_configuration', 'initial_state')
    anchors = []
    for slot in range(SLOTS):
        anchor = v2_slots[slot, 'bg128']
        _require(anchor[0]['checkpoint_update'] == 32 and anchor[0]['actor_sampling_seed_consumed'] is True,
                 'historical anchor must be the frozen Full u32 policy')
        for other in [v2_slots[slot, name] for name in formal.STRATEGIES] + [v3_slots[slot, v3.POLICY]]:
            _require(all(other[1][name] == anchor[1][name] for name in common),
                     'historical seed, RPC or initial state differs within slot')
        _require(v3_slots[slot, v3.POLICY][1]['v2_anchor'] == v2_slots[slot, v3.PREVIOUS_POLICY][2],
                 'historical FIFO V3 points to a different V2 slot')
        anchors.append(anchor)
    _require(len({anchor[0]['experiment_seed'] for anchor in anchors}) == SLOTS,
             'historical environment reused across different slots')
    return v2_refs, v3_refs, anchors, v2_slots, v3_slots


def make_job(ordinal, anchor):
    slot, arm = divmod(ordinal, len(POLICIES))
    source, _, source_receipt = anchor
    policy, run, seed = POLICIES[arm], FIRST_RUN + ordinal, source['experiment_seed']
    recording = f'{CAMPAIGN}-{slot:03d}-{policy.replace("_", "-")}'
    leaf = f'run_{run}'
    events = OUTPUT / 'anylogic_events' / leaf
    return dict(ordinal=ordinal, pair_ordinal=slot, policy_kind=policy, group='BASELINE', role='final',
                run_index=run, experiment_seed=seed, recording_id=recording, source_recording_id=recording,
                actor_seed=0, actor_sampling_seed=0, actor_sampling_seed_consumed=False,
                target_training_update=0, checkpoint_update=0, event_directory=str(events),
                event_bundle=str(events / episode_id(run, seed)), python_output=str(OUTPUT / 'shifts' / leaf),
                scene_id=source['scene_id'], expected_rpc_config=source['expected_rpc_config'],
                runtime_scene_sha256=source['runtime_scene_sha256'],
                anchor_run_index=source['run_index'], anchor_receipt=source_receipt)


def _validate_jobs(jobs, anchors):
    _require(len(jobs) == COUNT and len(anchors) == SLOTS, 'two complete 100-slot arms are required')
    for ordinal, job in enumerate(jobs):
        bare = {key: value for key, value in job.items() if key != 'phase_config'}
        _require(bare == make_job(ordinal, anchors[ordinal // len(POLICIES)]),
                 'new physical or paired anchor identity differs')


def _check_fresh(jobs, historical, inventory):
    _, _, anchors, v2_slots, v3_slots = inventory
    _require(CAMPAIGN not in historical['campaigns'] and historical['active_h2_reserved_shifts'] == 0,
             'strong baseline campaign already registered or historical budget policy differs')
    _validate_jobs(jobs, anchors)
    permitted = {}
    for job, _, pin in (*v2_slots.values(), *v3_slots.values()):
        state = historical['jobs'][str(job['run_index'])]
        _require(state['status'] == 'COMPLETE' and state['identity'] == job and state['receipt'] == pin,
                 'authorized matched seed roster must be registered and finalized')
        permitted[job['run_index']] = job

    retired = load_reference(v3.RETIRED_SUITE)
    _require(not Path(retired['python_output']).exists() and retired['campaign_id'] not in historical['campaigns']
             and len(retired['jobs']) == SLOTS, 'superseded V3 draft has runtime evidence or differs')
    for slot, job in enumerate(retired['jobs']):
        _require(job['pair_ordinal'] == slot and job['experiment_seed'] == anchors[slot][0]['experiment_seed']
                 and job['run_index'] not in permitted and str(job['run_index']) not in historical['jobs']
                 and not job['actor_seed'] and not job['actor_sampling_seed'], 'superseded V3 seed roster differs')
        permitted[job['run_index']] = job

    rows = _prior_rows(historical)
    existing_runs = {row.get('run_index') for row in rows}
    recordings = {row.get(key) for row in rows for key in ('recording_id', 'source_recording_id')}
    _require(all(job['run_index'] not in existing_runs and job['recording_id'] not in recordings for job in jobs),
             'new run or recording overlaps existing or prepared work')
    foreign = [row for row in rows if row != permitted.get(row.get('run_index'))]
    foreign_environment = {row['experiment_seed'] for row in foreign
                           if type(row.get('experiment_seed')) is int and row['experiment_seed'] > 0}
    forbidden_seeds = {row[key] & JAVA_MASK for row in foreign for key in grade.flow.SEED_FIELDS
                       if type(row.get(key)) is int and row[key] > 0}
    forbidden_seeds.update(row[key] & JAVA_MASK for row in rows for key in ('actor_seed', 'actor_sampling_seed')
                           if type(row.get(key)) is int and row[key] > 0)
    foreign_states = {state for seed in foreign_environment for state in java_random_states(seed).values()}
    seeds = {anchor[0]['experiment_seed'] for anchor in anchors}
    states = {state for seed in seeds for state in java_random_states(seed).values()}
    _require(all(0 < seed <= JAVA_MASK for seed in seeds) and len(states) == 5 * SLOTS
             and seeds.isdisjoint(forbidden_seeds) and states.isdisjoint(foreign_states),
             'historical environment seed or Java state overlaps an unauthorized stream')
    return dict(occupied_before=historical['maximum_total_shifts'],
                maximum_total_if_registered=historical['maximum_total_shifts'] + COUNT,
                new_physical_shifts=COUNT, new_environment_seeds=0, reused_environment_seeds=len(seeds),
                new_actor_sampling_seeds=0, reused_java_rng_states=len(states),
                prior_same_slot_complete_shifts=len(v2_slots) + len(v3_slots),
                superseded_unlaunched_same_slot_identities=len(retired['jobs']),
                historical_and_prepared_rows_checked=len(rows))


_prior_rows = bind(paired10._prior_rows, SUITE=SUITE)


def check_fresh(jobs, historical):
    return _check_fresh(jobs, historical, historical_inventory())


def _render_java(old, master, scenarios, jobs):
    _checked_reference(old['java_helper'])
    source = Path(old['java_helper']['path']).read_text(encoding='utf-8')
    for before, after in ((old['master_plan'], reference(master)),
                          (old['scenario_table'], reference(scenarios))):
        for key in ('path', 'sha256'):
            source = _replace_once(source, json.dumps(before[key]), json.dumps(after[key]))
    first = jobs[0]['experiment_seed']
    seeds = ', '.join(f"{job['experiment_seed']}L" for job in jobs[::2])
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
         'require(row.seed == PAIR_SEEDS[i / 2], "strong baseline paired slot seed differs");'),
        ('row(index); return index == 0 ? "rolling_quota" : "trip_fifo";',
         'row(index); return index % 2 == 0 ? "dispatch_grade" : "rolling_quota";'),
    )
    for before, after in replacements:
        source = _replace_once(source, before, after)
    _require(f'{first}L' in source, 'Java helper does not contain the first frozen seed')
    return source


def _verify_java(helper, scenarios, jobs):
    proof = verify_java(helper, scenarios)
    recordings = ', '.join(json.dumps(job['recording_id']) for job in jobs)
    seeds = ', '.join(f'{job["experiment_seed"]}L' for job in jobs)
    source = f"""public final class Strong100Probe {{
        public static void main(String[] args) {{
            String[] recordings = {{{recordings}}};
            long[] seeds = {{{seeds}}};
            if (Baseline1CampaignPlan.jobCount() != {COUNT}) throw new AssertionError();
            for (int i = 0; i < {COUNT}; i++) {{
                String policy = i % 2 == 0 ? "dispatch_grade" : "rolling_quota";
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
    with TemporaryDirectory(prefix='strong100-java-') as directory:
        probe = Path(directory) / 'Strong100Probe.java'
        probe.write_text(source, encoding='ascii')
        subprocess.run(['javac', '-encoding', 'UTF-8', '-d', directory, str(helper), str(probe)], check=True)
        subprocess.run(['java', '-cp', directory, 'Strong100Probe'], check=True)
    _require(proof['rows_checked'] == COUNT, 'nominal Java roster size differs')
    return dict(proof, pair_identities_checked=COUNT)


def _phase(job, source_pins, protected_core, historical_v2, baseline_alp):
    return dict(schema=SCHEMA + '-phase', campaign_id=CAMPAIGN, policy_kind=job['policy_kind'],
                method=METHOD, execution=EXECUTION, source_pins=source_pins, protected_core=protected_core,
                historical_v2=historical_v2, baseline_alp=baseline_alp, job=job)


def _source_pins(old):
    paths = {Path(pin['path']) for pin in old['source_pins']}
    paths.update((Path(__file__).resolve(), LAUNCHER, PLAN / 'strong100_runtime.py',
                  PLAN / 'strong100_report.py', PLAN / 'strong100_compute_probe.py',
                  OR_PLAN / 'milp_smoke_runtime.py', Path(paired10.__file__),
                  BASELINES / 'dispatch_inspired_001' / 'dispatch_runtime.py',
                  Path(recent_budget.__file__), BASELINES / 'paired_v2_100_001' / 'formal_report.py',
                  BASELINES / 'paired_100_001' / 'paired_report.py'))
    return [reference(path) for path in sorted(paths)]


def prepare():
    _require(not any(path.exists() for path in (PREPARED, OUTPUT, REPORT)),
             'strong baseline preparation or physical evidence already exists')
    old = smoke.load_suite()
    protected = old['protected_core']
    alp = reference(BASELINE_ALP)
    inventory = historical_inventory()
    v2_refs, v3_refs, anchors, _, _ = inventory
    pins = _source_pins(old)
    jobs = [make_job(ordinal, anchors[ordinal // len(POLICIES)]) for ordinal in range(COUNT)]
    budget = _check_fresh(jobs, preceding_snapshot(), inventory)
    for job in jobs:
        target = PREPARED / 'configs' / f"run_{job['run_index']}.json"
        job['phase_config'] = _publish_json(target, _phase(job.copy(), pins, protected, v2_refs, alp))
    master, scenarios = PREPARED / 'master_plan.csv', PREPARED / 'scenario_table.csv'
    master_ref = _publish_bytes(master, _csv_bytes(jobs, COLUMNS))
    scenario_ref = _publish_bytes(scenarios, scenario_table_bytes(jobs))
    helper = PREPARED / 'Baseline1CampaignPlan.java'
    helper_ref = _publish_bytes(helper, _render_java(old, master, scenarios, jobs).encode('utf-8'))
    proof = _verify_java(helper, scenarios, jobs)
    columns = ('pair_ordinal', 'policy_kind', 'run_index', 'experiment_seed', 'anchor_run_index', 'recording_id',
               'config_sha256', 'python_output', 'event_directory')
    roster = _publish_bytes(PREPARED / 'run_seed.csv', _csv_bytes(
        [dict(job, config_sha256=job['phase_config']['sha256']) for job in jobs], columns))
    suite = dict(schema=SCHEMA, campaign_id=CAMPAIGN, policy_kind=POLICY, method=METHOD,
                 execution=EXECUTION, evaluation_slots=SLOTS, protected_core=protected, source_pins=pins,
                 historical_v2=v2_refs, historical_v3=v3_refs, baseline_alp=alp,
                 master_plan=master_ref, scenario_table=scenario_ref, java_helper=helper_ref,
                 java_verification=proof, jobs=jobs, smoke_campaign=reference(smoke.SUITE),
                 pairing=PAIRING, run_seed=roster, control_directory=str(CONTROL),
                 python_output=str(OUTPUT), report=str(REPORT), budget=budget)
    if COMPUTE_PROBE.is_file():
        suite['compute_probe'] = reference(COMPUTE_PROBE)
    suite_ref = _publish_json(SUITE, suite)
    return _publish_json(PREPARED / 'launch_identity.json', dict(
        status='PREPARED_NOT_STARTED', campaign_id=CAMPAIGN, suite=suite_ref, run_seed=roster,
        java_helper=helper_ref, budget=budget, evaluation_slots=SLOTS, physical_shifts=COUNT,
        registered_shifts=0, completed_shifts=0, optimizer_updates=0, jobs=jobs))


def load_suite(path=SUITE, *, check_sources=True):
    _require(Path(path).resolve() == SUITE, 'unexpected strong baseline manifest path')
    suite = load_reference(reference(SUITE))
    _require(suite['schema'] == SCHEMA and suite['campaign_id'] == CAMPAIGN and suite['policy_kind'] == POLICY
             and suite['method'] == METHOD and suite['execution'] == EXECUTION and suite['evaluation_slots'] == SLOTS
             and suite['pairing'] == PAIRING and len(suite['jobs']) == COUNT
             and suite['protected_core'] == dict(path=str(CORE), sha256=CORE_SHA256)
             and suite['control_directory'] == str(CONTROL) and suite['python_output'] == str(OUTPUT)
             and suite['report'] == str(REPORT) and suite['baseline_alp']['path'] == str(BASELINE_ALP),
             'strong baseline contract differs')
    if check_sources:
        inventory = historical_inventory()
        v2_refs, v3_refs, anchors, _, _ = inventory
        _require(suite['historical_v2'] == v2_refs and suite['historical_v3'] == v3_refs
                 and suite['smoke_campaign'] == reference(smoke.SUITE), 'sealed historical source differs')
        _validate_jobs(suite['jobs'], anchors)
        for pin in [suite[key] for key in ('protected_core', 'baseline_alp', 'master_plan', 'scenario_table',
                                          'java_helper', 'run_seed')] + suite['source_pins']:
            _checked_reference(pin)
        if 'compute_probe' in suite:
            _require(suite['compute_probe']['path'] == str(COMPUTE_PROBE), 'unexpected same-host compute proof path')
            _checked_reference(suite['compute_probe'])
        for job in suite['jobs']:
            bare = {key: value for key, value in job.items() if key != 'phase_config'}
            _require(load_reference(job['phase_config']) ==
                     _phase(bare, suite['source_pins'], suite['protected_core'], v2_refs, suite['baseline_alp']),
                     'phase configuration bytes differ from frozen job')
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
