"""Freeze a 511-job continuation of the original 512-shift BG128 authorization."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path('/home/jinx/Faith/experiments/lagrangian_ppo_v1')
PLAN = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def render_java(original_source, *, old_suite, recovery_path, control_directory, bridge_path):
    from baseline1_v1.java_plan import _replace_once
    from business_graph_summary_campaign_v1.plan_io import sha
    source = original_source
    source = _replace_once(source,
        'private static final String CONTROL_DIRECTORY = ' + json.dumps(old_suite['control_directory']) + ';',
        'private static final String CONTROL_DIRECTORY = ' + json.dumps(str(control_directory)) + ';')
    source = _replace_once(source, '    private static boolean controllerStarted = false;',
        '    private static final int RESUME_OFFSET = 1;\n'
        '    private static final int REMAINING_JOB_COUNT = JOB_COUNT - RESUME_OFFSET;\n'
        '    private static boolean controllerStarted = false;')
    source = _replace_once(source,
        'require(row.ordinal == 0, "one-click startup requires index 0; recovery must be reviewed");',
        'require(row.ordinal == RESUME_OFFSET, "recovery startup requires original ordinal 1");')
    source = _replace_once(source,
        json.dumps(old_suite['one_click_entry']['path']), json.dumps(str(bridge_path)))
    source = _replace_once(source,
        json.dumps(old_suite['one_click_entry']['sha256']), json.dumps(sha(bridge_path)))
    source = _replace_once(source,
        '"--suite", ' + json.dumps(old_suite['suite_path']),
        '"--recovery", ' + json.dumps(str(recovery_path)))
    source = _replace_once(source,
        'require(index >= 0 && index < JOB_COUNT, "Freeform index outside frozen plan");\n'
        '        return plan().rows[index];',
        'require(index >= 0 && index < REMAINING_JOB_COUNT, "Freeform index outside remaining plan");\n'
        '        return plan().rows[index + RESUME_OFFSET];')
    source = _replace_once(source,
        'public static int jobCount() { return JOB_COUNT; }',
        'public static int jobCount() { return REMAINING_JOB_COUNT; }')
    source = _replace_once(source,
        'public static int resumeOffset() { return 0; }',
        'public static int resumeOffset() { return RESUME_OFFSET; }')
    source = _replace_once(source,
        '        require(row != null, "run_index absent from frozen master");',
        '        require(row != null, "run_index absent from frozen master");\n'
        '        require(row.ordinal >= RESUME_OFFSET, "already captured original job must not be run again");')
    return source


def prepare(reconciliation_path):
    from business_graph_summary_campaign_v1.plan_io import reference, write, load_suite
    from business_graph_summary_campaign_v1.gate import check_sources, PROTECTED_SHA
    from business_graph_summary_recovery_v1.gate import source_pins
    old_path = PLAN.parent / 'prepared_001/suite.json'
    old_suite, _ = load_suite(old_path)
    check_sources(old_suite['source_pins'])
    assert reference(old_suite['java_helper']['path']) == old_suite['java_helper']
    assert reference(old_suite['master_plan']['path']) == old_suite['master_plan']
    reconciliation_path = Path(reconciliation_path).resolve()
    assert reconciliation_path.is_file()
    reconciliation = json.loads(reconciliation_path.read_bytes())
    assert reconciliation['schema'] == 'bg128-budget-reconciliation-applied-v1'
    assert reconciliation['status'] == 'APPLIED'
    assert reconciliation['old_suite'] == reference(old_path)
    assert reconciliation['campaign_id'] == old_suite['campaign_id']
    assert reconciliation['run_index'] == 9900001
    assert reconciliation['original_status'] == 'UNKNOWN' and reconciliation['new_status'] == 'COMPLETE'
    assert reconciliation['new_shift_cap'] == 0 and reconciliation['unchanged_planned_slots'] == 511
    assert reconciliation['unchanged_commits'] == 0
    recovery_path = PLAN / 'recovery.json'
    helper_path = PLAN / 'Baseline1CampaignPlan.java'
    identity_path = PLAN / 'launch_identity.json'
    csv_path = PLAN / 'remaining_identities.csv'
    assert all(not path.exists() for path in (recovery_path, helper_path, identity_path, csv_path))
    output = Path(old_suite['python_output'])
    control = output / 'recovery_001/control'
    assert not control.exists()
    assert len(old_suite['jobs']) == 512 and len(old_suite['phase_configs']) == 16
    remaining = old_suite['jobs'][1:]
    assert remaining[0]['run_index'] == 9900002 and remaining[-1]['run_index'] == 9900512
    arm = output / 'arms/s01_BG128'
    initial_checkpoint = reference(arm / 'initial.pt')
    checkpoint_record = json.loads((arm / 'initial_checkpoint.json').read_bytes())
    assert checkpoint_record == initial_checkpoint
    initial_actor = reference(arm / 'train_u0001/actor.pt')
    bridge = PLAN / 'bootstrap.py'
    helper_path.write_text(render_java(Path(old_suite['java_helper']['path']).read_text(),
        old_suite=old_suite, recovery_path=recovery_path, control_directory=control, bridge_path=bridge))
    descriptor = dict(schema='business-graph-summary-recovery-v1',
        recovery_path=str(recovery_path), campaign_id=old_suite['campaign_id'],
        old_suite=reference(old_path), control_directory=str(control), python_output=str(output),
        new_source_pins=source_pins(), reconciliation=reference(reconciliation_path),
        resume_offset=1, job_count=511, frozen_job_count=512,
        new_physical_shift_cap=0, remaining_physical_shifts=511,
        remaining_jobs=remaining, java_helper=reference(helper_path), master_plan=old_suite['master_plan'],
        bridge=reference(bridge), initial_checkpoint=initial_checkpoint, initial_actor=initial_actor,
        initial_checkpoint_record=reference(arm / 'initial_checkpoint.json'),
        initial_artifact_references=[reference(path) for path in (
            arm / 'initial.pt.manifest.json', arm / 'initial.pt.seal.json',
            arm / 'train_u0001/actor.pt.manifest.json', arm / 'train_u0001/actor.pt.seal.json')],
        original_stop=reference(Path(old_suite['control_directory']) / 'STOP.properties'),
        original_first_claim=reference(Path(old_suite['control_directory']) / 'java_claims/run_9900001.once.claim'),
        original_launch=reference(PLAN.parent / 'prepared_001/launch/launch_receipt.json'),
        original_approval=reference(PLAN.parent / 'prepared_001/approval.json'),
        protected_ppo_sha256=PROTECTED_SHA,
        resume_policy='same_u0_actor_first_batch_original_run1_then_31_remaining_jobs',
        additional_budget_registration=False, original_config_bytes_changed=False,
        high_frequency_diagnostics=False)
    write(recovery_path, descriptor)
    first, last = remaining[0], remaining[-1]
    write(identity_path, dict(status='RECOVERY_PREPARED_NOT_STARTED',
        recovery=reference(recovery_path), original_suite=reference(old_path),
        new_physical_shift_cap=0, remaining_physical_shifts=511, first_job=first, last_job=last,
        original_configurations=old_suite['phase_configs'],
        original_initialization_seed=old_suite['arms'][0]['actor_seed'],
        control_directory=str(control), old_control_preserved=True))
    columns = ('ordinal', 'run_index', 'experiment_seed', 'actor_sampling_seed', 'actor_seed',
               'recording_id', 'python_output', 'material_event_directory')
    with csv_path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=('anylogic_index',) + columns, lineterminator='\n')
        writer.writeheader()
        for index, job in enumerate(remaining):
            writer.writerow(dict(anylogic_index=index, **{key: job[key] for key in columns}))
    return reference(recovery_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reconciliation', required=True,
                        help='Immutable APPLIED reconciliation receipt; not a pending proposal.')
    parser.add_argument('--prepare-only', action='store_true',
                        help='This program only prepares; it never reconciles, registers or launches.')
    args = parser.parse_args()
    print(json.dumps(prepare(args.reconciliation), ensure_ascii=False))


if __name__ == '__main__':
    main()
