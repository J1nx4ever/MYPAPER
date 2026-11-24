"""Generate and compile-check the one-click helper for the untouched 499 shifts."""
import csv

from recovery_common import (
    APPROVAL, CAMPAIGN, CONTROL_DIR, DESCRIPTOR, FIRST_REMAINING, LAST_REMAINING, LAUNCH_DIR,
    MATERIALS, OFFSET, OUTPUT, PLAN, PREPARED_DIR, RECONCILIATION, REMAINING, REPORT_DIR, SUITE,
    TOTAL, RegisteredLedger, anchor_files, assert_start_state, checked, original_campaign, preflight,
    reference, require, source_pins,
)
from business_graph_summary_campaign_v1.plan_io import write
from recovery_java import render_java, verify_java


def prepare():
    require(not any(path.exists() for path in (PREPARED_DIR, CONTROL_DIR, LAUNCH_DIR)),
            'fresh recovery preparation and startup identity required')
    suite, _ = original_campaign()
    ledger = RegisteredLedger(suite, suite_path=SUITE, approval_path=APPROVAL)
    assert_start_state(suite, ledger)
    require(RECONCILIATION.is_file(), 'applied reconciliation proof required')
    helper = PREPARED_DIR / 'Baseline1CampaignPlan.java'
    PREPARED_DIR.mkdir(parents=True)
    source = checked(suite['java_helper']).read_text(encoding='utf-8')
    helper.write_text(render_java(source, old_suite=suite, recovery_path=DESCRIPTOR,
                                   control_directory=CONTROL_DIR, bridge_path=PLAN / 'bootstrap_recovery.py'),
                      encoding='utf-8')
    java = verify_java(helper, checked(suite['master_plan']), checked(suite['scenario_table']))
    phases = {p['phase_id']: p for p in suite['phases']}
    rows = []
    for index, job in enumerate(suite['jobs'][OFFSET:]):
        row = {key: job[key] for key in (
            'ordinal', 'pair_ordinal', 'scene_id', 'run_index', 'experiment_seed', 'actor_sampling_seed',
            'actor_seed', 'recording_id', 'python_output', 'material_event_directory',
        )}
        rows.append(dict(anylogic_index=index, **row, config_sha256=phases[job['phase_id']]['phase_config']['sha256']))
    require(len(rows) == REMAINING and rows[0]['run_index'] == FIRST_REMAINING
            and rows[-1]['run_index'] == LAST_REMAINING, 'original remaining range differs')
    inventory = PREPARED_DIR / 'remaining_identities.csv'
    with inventory.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    data = dict(schema='bg128-generalization-recovery-v1', campaign_id=CAMPAIGN,
                original_suite=reference(SUITE), source_pins=source_pins(), original_anchors=anchor_files(),
                control_directory=str(CONTROL_DIR), resume_offset=OFFSET, remaining_shifts=REMAINING,
                additional_shifts=0, frozen_shifts=TOTAL, reconciliation=reference(RECONCILIATION),
                master_plan=suite['master_plan'], scenario_table=suite['scenario_table'],
                checkpoint=suite['parent_artifacts']['checkpoint'], java_helper=reference(helper),
                remaining_identities=reference(inventory))
    write(DESCRIPTOR, data)
    write(PREPARED_DIR / 'launch_identity.json', dict(
        recovery=reference(DESCRIPTOR), original_suite=data['original_suite'], java_helper=data['java_helper'],
        first=rows[0], last=rows[-1], checkpoint=data['checkpoint'], remaining_shifts=REMAINING,
        additional_shifts=0, configurations=suite['phases'][1:],
        python_output=str(OUTPUT), anylogic_event_root=str(MATERIALS), control_directory=str(CONTROL_DIR),
    ))
    preflight()
    write(REPORT_DIR / 'preparation_validation.json', dict(
        status='PASS_OFFLINE_RECOVERY', recovery=reference(DESCRIPTOR), java=java,
        recovered_completed_shifts=OFFSET, remaining_shifts=REMAINING, optimizer_updates=0, simulator_started=False,
    ))
    print(str(DESCRIPTOR))


if __name__ == '__main__':
    prepare()
