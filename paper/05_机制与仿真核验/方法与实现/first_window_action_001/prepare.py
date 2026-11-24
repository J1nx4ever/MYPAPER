"""Freeze the four-run configuration and the matching one-click Java helper."""
import copy
import csv
import io
import json

from common import *
from budget import budget_snapshot, check_new_slots
from baseline1_v1.java_plan import COLUMNS, _replace_once


def csv_bytes(columns, rows):
    data = io.StringIO(newline='')
    writer = csv.DictWriter(data, fieldnames=columns, lineterminator='\n', extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)
    return data.getvalue().encode('ascii')


def helper(master, old, low):
    """Rebind the audited helper using the locally verified paired seed inventory."""
    low_helper = checked(low['java_helper'])
    source = low_helper.read_text()
    replacements = (
        (low['master_plan']['path'], str(master)),
        (low['master_plan']['sha256'], sha(master)),
        (low['control_directory'], str(RUN/'control')),
        ('private static final String CAMPAIGN_ID = '+json.dumps(low['campaign_id']), 
         'private static final String CAMPAIGN_ID = '+json.dumps(CID)),
        ('private static final int JOB_COUNT = 100;', 'private static final int JOB_COUNT = 4;'),
        (str(Path(low['python_output']).with_name(low['campaign_id']+'_materials')), str(MATERIALS.parent)),
        (low['one_click_entry']['path'], str(PLAN/'bootstrap.py')),
        (low['one_click_entry']['sha256'], sha(PLAN/'bootstrap.py')),
        (low['suite_path'], str(PREPARED/'suite.json')),
    )
    for before, after in replacements:
        source = _replace_once(source, before, after)
    return source.encode()


def main():
    (old, _), (low, _) = source_records()
    case_ref = reference(REPORT/'candidate_selection.json')
    case_data = read(case_ref['path'])
    cases = {str(r['historical_run']): r for r in case_data['candidates']}
    actor_ref = case_data['actor']
    checkpoint = old['parent_artifacts']['checkpoint']
    require(checkpoint['sha256'] == CHECKPOINT_SHA and sha(CORE) == CORE_SHA, 'fixed parent differs')
    require(not PREPARED.exists() and not RUN.exists() and not MATERIALS.exists(), 'experiment identity already exists')
    phase = read(checked(old['phase_config']))
    require(not phase.get('common_reference') and phase['capture_physical_events'] is False,
            'expected full independent evaluation phase')
    before = budget_snapshot()
    phase.update(campaign_id=CID, phase_id=PHASE, arm_phase_id='first_window',
        recording_id=RECORDING, python_output=str(RUN/'evaluation'),
        event_directory=str(RUN/'control/legacy_unused'), optimizer_steps=0, evaluation_only=True,
        kind='single_action_material_diagnostic', actor=actor_ref, checkpoint=checkpoint, cases=cases,
        occupied_before=before['maximum_total_shifts'], jobs=[])
    for historic_run in CASES:
        original = next(j for j in old['jobs'] if j['run_index']==historic_run)
        for branch in ('original','alternative'):
            ordinal = len(phase['jobs'])
            run_index = RUN_IDS[ordinal]
            episode = f'dispatch-production-wait-shortfall-v2-run-{run_index}-seed-{original["experiment_seed"]}'
            job = copy.deepcopy(original)
            job.update(ordinal=ordinal, run_index=run_index, historical_run=historic_run, branch=branch,
                scene_group=f'{CID}-{historic_run}', realization_group_id=f'{CID}-{historic_run}',
                phase_id=PHASE, arm_phase_id='first_window', recording_id=RECORDING, source_recording_id=RECORDING,
                python_output=phase['python_output'], event_directory=phase['event_directory'],
                event_bundle=str(Path(phase['event_directory'])/episode), material_event_directory=str(MATERIALS),
                material_event_bundle=str(MATERIALS/episode), material_journal=str(MATERIALS/episode/'material_events.bin'))
            phase['jobs'].append(job)
    check_new_slots(phase, before)
    PREPARED.mkdir()
    config_ref = save(PREPARED/'phase_config.json', phase)
    master = PREPARED/'master_plan.csv'
    master.write_bytes(csv_bytes(COLUMNS, phase['jobs']))
    java = PREPARED/'Baseline1CampaignPlan.java'
    java.write_bytes(helper(master, old, low))
    pinned = {name: reference(PLAN/name) for name in ('common.py','select_actions.py','budget.py','runtime.py','bootstrap.py','evaluate.py')}
    pinned.update(java=reference(java), master=reference(master), actor=actor_ref,
                  checkpoint=checkpoint, protected_core=reference(CORE),
                  checkpoint_manifest=old['parent_artifacts']['manifest'], checkpoint_seal=old['parent_artifacts']['seal'])
    for ref in pinned.values():
        checked(ref)
    suite_ref = save(PREPARED/'suite.json', dict(schema='bg128-first-window-action-v1', campaign_id=CID,
        config=config_ref, candidate_selection=case_ref, pinned=pinned, master_plan=reference(master),
        new_physical_shift_cap=4, historical_seeds_reused_for_pairing=True))
    columns = ('ordinal','run_index','historical_run','branch','experiment_seed','actor_sampling_seed','recording_id')
    (PREPARED/'run_seed.csv').write_bytes(csv_bytes(columns, phase['jobs']))
    identity = dict(status='PREPARED_NOT_STARTED', recording_id=RECORDING, config=config_ref, suite=suite_ref,
        python_output=phase['python_output'], anylogic_event_directory=str(MATERIALS),
        run_seed=reference(PREPARED/'run_seed.csv'), budget_before=phase['occupied_before'],
        budget_after_registration=phase['occupied_before']+4,
        checkpoint=checkpoint, java_helper=reference(java))
    save(PREPARED/'launch_identity.json', identity)
    print(json.dumps(identity))


if __name__ == '__main__':
    main()
