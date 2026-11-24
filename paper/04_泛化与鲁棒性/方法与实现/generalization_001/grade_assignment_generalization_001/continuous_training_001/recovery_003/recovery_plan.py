"""Freeze the unused original shifts and one scoped replacement for the aborted START."""
from collections import Counter
import csv
from pathlib import Path
import sys

PLAN = Path(__file__).resolve().parent
sys.path.insert(0, str(PLAN.parent))
from mixed_plan import (
    CAMPAIGN, CONTROL, CORE, LEDGER, MATERIALS, MAX_JAVA_SEED, OUTPUT, REPORT, ROOT, SUITE,
    TOTAL, check_collisions, load_package, read, reference, require, seed,
)
from mixed_budget import MixedLedger
from mixed_java import scenario_table_bytes
from baseline1_v1.java_plan import COLUMNS
from business_graph_summary_campaign_v1.gate import check_phase
from business_graph_summary_campaign_v1.health import _process_identity
from business_graph_summary_campaign_v1.plan_io import load_phase, write
from physical_curriculum_v1.budget import _history_rows
from python.offline_split import java_random_states

OFFSET = 2
FAILED = 10000003
REPLACEMENT = 10002425
REMAINING = TOTAL - OFFSET
PREVIOUS = OUTPUT / 'recovery_002'
RECOVERY_OUTPUT = OUTPUT / PLAN.name
RECOVERY_CONTROL = RECOVERY_OUTPUT / 'control'
RECOVERY_REPORT = REPORT / PLAN.name
PREPARED = PLAN / 'prepared_001'
RECOVERY = PREPARED / 'recovery.json'
REPLACEMENT_PHASE = PREPARED / 'slot_10002425.json'
HELPER = PREPARED / 'Baseline1CampaignPlan.java'
MASTER = PREPARED / 'master_plan.csv'
SCENARIOS = PREPARED / 'scenarios.csv'
IDENTITIES = PREPARED / 'remaining_identities.csv'
BRIDGE = PLAN / 'recovery_bridge.py'
ERROR = "ProtocolError('AnyLogic reward configuration changed during this run')"


def replacement_config(first):
    failed = first['jobs'][OFFSET]
    require(failed['run_index'] == FAILED and failed['ordinal'] == OFFSET, 'aborted frozen identity differs')
    pid = first['phase_id'] + '_replacement'
    recording = CAMPAIGN + '-' + pid
    output = RECOVERY_OUTPUT / 'replacement'
    events = RECOVERY_OUTPUT / 'legacy_unused' / pid
    material = MATERIALS / pid
    env_seed = seed(CAMPAIGN + '-replacement', 'environment', OFFSET)
    sampling_seed = seed(CAMPAIGN + '-replacement', 'sampling', OFFSET)
    leaf = f'dispatch-production-wait-shortfall-v2-run-{REPLACEMENT}-seed-{env_seed}'
    group = failed['scene_group'] + '-replacement'
    job = dict(failed, run_index=REPLACEMENT, experiment_seed=env_seed, actor_sampling_seed=sampling_seed,
               recording_id=recording, source_recording_id=recording,
               phase_id=pid, scene_group=group, realization_group_id=group,
               event_directory=str(events), event_bundle=str(events / leaf), python_output=str(output),
               material_event_directory=str(material), material_event_bundle=str(material / leaf),
               material_journal=str(material / leaf / 'material_events.bin'))
    return check_phase(dict(first, phase_id=pid, recording_id=recording, event_directory=str(events),
                            python_output=str(output), jobs=[job]))


def _jobs(suite, cfg):
    return suite['jobs'][:OFFSET] + cfg['jobs'] + suite['jobs'][OFFSET + 1:]


def _check_identity(suite, replacement):
    prior = check_collisions(suite)
    historical = [*_history_rows(prior), *suite['jobs']]
    for key in ('run_index', 'recording_id', 'source_recording_id'):
        require(replacement[key] not in {job.get(key) for job in historical}, 'replacement identity collides: ' + key)
    seed_fields = ('experiment_seed', 'actor_sampling_seed', 'actor_seed')
    new = {replacement['experiment_seed'], replacement['actor_sampling_seed']}
    require(len(new) == 2 and all(0 < value <= MAX_JAVA_SEED for value in new)
            and new.isdisjoint(job[key] & MAX_JAVA_SEED for job in historical for key in seed_fields
                               if type(job.get(key)) is int), 'replacement seed collides')
    java = set(java_random_states(replacement['experiment_seed']).values())
    require(len(java) == len(java_random_states(replacement['experiment_seed']))
            and java.isdisjoint(value for job in historical if type(job.get('experiment_seed')) is int
                                for value in java_random_states(job['experiment_seed']).values()),
            'replacement Java random state collides')


def _anchors(suite):
    old = suite['jobs'][OFFSET]
    raw = Path(old['python_output']) / 'episodes' / f'run_{FAILED}'
    return {
        'original_stop': reference(CONTROL / 'STOP.properties'),
        'storage_stop': reference(OUTPUT / 'recovery_001/control/STOP.properties'),
        'previous_stop': reference(PREVIOUS / 'control/STOP.properties'),
        'previous_claim': reference(PREVIOUS / f'control/java_claims/run_{FAILED}.once.claim'),
        'previous_ready': reference(PREVIOUS / f'control/ready/run_{FAILED}.properties'),
        'previous_status': reference(PREVIOUS / 'run_status.json'),
        'previous_log': reference(PREVIOUS / 'launch/controller.log'),
        'previous_launch': reference(PREVIOUS / 'launch/launch_receipt.json'),
        'aborted': reference(raw / 'ABORTED.json'),
        'partial_samples': reference(raw / '.samples.partial.h5'),
        'partial_material': reference(old['material_journal'] + '.partial'),
        'first_reconciliation': reference(REPORT / 'recovery_002/admission/reconciliation.json'),
        'protected_core': reference(CORE),
    }


def _shared(suite, cfg):
    job = cfg['jobs'][0]
    storage = ROOT / 'business_graph_summary_storage_v1'
    return dict(schema='bg128-mixed-cross-shift-recovery-v1', campaign_id=CAMPAIGN, original_suite=reference(SUITE),
                sources={path.name: reference(path) for path in sorted(PLAN.glob('*.py'))},
                storage=reference(storage / 'episode.py'),
                storage_scope=reference(PLAN.parent / 'recovery_002/storage_scope.py'),
                previous_artifacts=_anchors(suite), replacement_config=reference(REPLACEMENT_PHASE),
                replacement=job, failed_run_index=FAILED, resume_offset=OFFSET, remaining_shifts=REMAINING,
                scheduled_complete_shifts=TOTAL, additional_slots=1, post_registration_cap=25636,
                recovery_control=str(RECOVERY_CONTROL), recovery_output=str(RECOVERY_OUTPUT))


def check_start(manifest, suite, configs, cfg):
    """Read-only gate before registration. UNKNOWN is never re-run or promoted."""
    receipt = read(PREVIOUS / 'launch/launch_receipt.json')
    process = _process_identity(receipt['pid'])
    state = read(PREVIOUS / 'run_status.json')
    abort = read(manifest['previous_artifacts']['aborted']['path'])
    require(not process.get('inspection_denied')
            and (process.get('state') == 'Z' or process.get('start_ticks') != receipt['process_start_ticks'])
            and state['status'] == 'STOP' and state['reason'] == ERROR
            and (PREVIOUS / 'control/STOP.properties').read_text() == f'reason={ERROR}\n'
            and abort['completed_episodes'] == 0 and abort['transitions'] == 0,
            'third shift might be live or completed; exact offline review required')
    require(manifest['ledger_before'] == reference(LEDGER)
            and not any(LEDGER.with_name(LEDGER.name + s).exists() for s in ('-journal','-wal')),
            'original ledger bytes changed before replacement registration')
    records = MixedLedger(suite, configs).snapshot()
    require(Counter(row['status'] for row in records['jobs'].values()) ==
            {'COMPLETE': OFFSET, 'UNKNOWN': 1, 'PLANNED': TOTAL-OFFSET-1}
            and records['jobs'][str(FAILED)]['error'] == ERROR
            and not records['campaigns'][CAMPAIGN]['commits'], 'frozen continuation point changed')
    for j in _jobs(suite, cfg)[OFFSET:]:
        require(not (Path(j['python_output']) / 'episodes' / f"run_{j['run_index']}").exists()
                and not Path(j['material_event_bundle']).exists(), 'remaining physical identity has artifacts')
    _check_identity(suite, cfg['jobs'][0])
    return records


def _write_csv(path, fields, rows):
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)


def load_recovery():
    suite, configs = load_package()
    cfg = load_phase(REPLACEMENT_PHASE)
    require(cfg == replacement_config(configs[0]), 'replacement phase or source scene differs')
    manifest = read(RECOVERY)
    require(all(manifest[key] == value for key, value in _shared(suite, cfg).items())
            and all(manifest[k] == reference(p) for k, p in
                    (('helper', HELPER), ('master', MASTER), ('scenarios', SCENARIOS), ('identities', IDENTITIES))),
            'frozen recovery byte binding changed')
    return manifest, suite, configs, cfg


def prepare():
    from recovery_java import verify_java, write_java

    require(not PREPARED.exists() and not RECOVERY_OUTPUT.exists(), 'new offline recovery location required')
    suite, configs = load_package()
    cfg = replacement_config(configs[0])
    initial = dict(previous_artifacts=_anchors(suite), ledger_before=reference(LEDGER))
    check_start(initial, suite, configs, cfg)
    jobs = _jobs(suite, cfg)
    write(REPLACEMENT_PHASE, cfg)
    ref = reference(REPLACEMENT_PHASE)
    _write_csv(MASTER, COLUMNS, jobs)
    with SCENARIOS.open('xb') as stream:
        stream.write(scenario_table_bytes(jobs))
    fields = ('ordinal','run_index','experiment_seed','actor_sampling_seed','recording_id','config_sha256',
              'python_output','material_event_directory')
    phase_sha = {p['phase_id']: r['sha256'] for p, r in zip(configs, suite['phase_configs'], strict=True)}
    phase_sha[cfg['phase_id']] = ref['sha256']
    _write_csv(IDENTITIES, fields, (dict(job, config_sha256=phase_sha[job['phase_id']]) for job in jobs[OFFSET:]))
    write_java(suite['java_helper']['path'], HELPER, CONTROL, RECOVERY_CONTROL,
               suite['one_click_entry']['path'], BRIDGE, SUITE, RECOVERY, MASTER, SCENARIOS)
    verify = verify_java(HELPER, SCENARIOS)
    require(verify['compiled'] and verify['remaining_rows_checked'] == REMAINING
            and not verify['simulation_started'], 'recovery helper compilation or mapping not verified')
    write(RECOVERY, dict(_shared(suite, cfg), **{'ledger_before': initial['ledger_before']},
                         helper=reference(HELPER), master=reference(MASTER), scenarios=reference(SCENARIOS),
                         identities=reference(IDENTITIES), java_verification=verify))
    job = cfg['jobs'][0]
    write(PREPARED / 'launch_identity.json', dict(
        recovery=reference(RECOVERY), replacement_phase=ref,
        **{field: job[field] for field in fields if field != 'config_sha256'}, config_sha256=ref['sha256'],
        remaining_identities=reference(IDENTITIES), previous_failed=FAILED,
        remaining_runs=REMAINING, original_output=str(OUTPUT), anylogic_event_root=str(MATERIALS),
        recovery_control=str(RECOVERY_CONTROL), planned_additional_slots=1))
    checked, suite, configs, cfg = load_recovery()
    check_start(checked, suite, configs, cfg)
    print(f'THIRD_RECOVERY_PREPARED_ONLY {RECOVERY}')


if __name__ == '__main__':
    prepare()
