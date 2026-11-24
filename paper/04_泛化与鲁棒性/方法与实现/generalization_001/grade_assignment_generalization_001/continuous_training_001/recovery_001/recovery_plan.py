"""Prepare a one-time recovery from the pre-READY BG128 ledger lookup error."""
from collections import Counter
from pathlib import Path
import sys

PLAN = Path(__file__).resolve().parent
sys.path.insert(0, str(PLAN.parent))

from mixed_plan import (
    APPROVAL, CAMPAIGN, CONTROL, CORE, LAUNCH, LEDGER, MATERIALS, OUTPUT, REPORT,
    ROOT, SUITE, TOTAL, authorization_for, check_collisions, load_package, read,
    reference, require,
)
from mixed_bootstrap import _intent
from mixed_budget import MixedLedger
from business_graph_summary_campaign_v1.health import _process_identity
from business_graph_summary_campaign_v1.plan_io import write

SCHEMA = 'bg128-continuous-mixed-pre-ready-recovery-v1'
RECOVERY_OUTPUT = OUTPUT / PLAN.name
RECOVERY_CONTROL = RECOVERY_OUTPUT / 'control'
RECOVERY_REPORT = REPORT / PLAN.name
PREPARED = PLAN / 'prepared_001'
RECOVERY = PREPARED / 'recovery.json'
HELPER = PREPARED / 'Baseline1CampaignPlan.java'
BRIDGE = PLAN / 'recovery_bridge.py'
RENDERER = PLAN / 'recovery_java.py'
PREVIOUS_STATUS = OUTPUT / 'run_status.json'
PREVIOUS_ERROR = "ValueError('committed update absent')"


def _sources():
    return {path.name: reference(path) for path in sorted(PLAN.glob('*.py'))}


def _anchors(suite):
    arm = Path(suite['arms'][0]['python_output'])
    initial = ('initial.pt', 'initial.pt.manifest.json', 'initial.pt.seal.json',
               'initial_checkpoint.json', 'initialization_comparison.json')
    return dict(
        stop=reference(CONTROL / 'STOP.properties'),
        status=reference(PREVIOUS_STATUS),
        launch_log=reference(LAUNCH / 'controller.log'),
        launch_receipt=reference(LAUNCH / 'launch_receipt.json'),
        launch_intent=reference(LAUNCH / 'launch_intent.json'),
        authority=reference(APPROVAL), protected_core=reference(CORE),
        **{name: reference(arm / name) for name in initial},
    )


def _first(suite):
    row = suite['jobs'][0]
    return dict(run_index=row['run_index'], experiment_seed=row['experiment_seed'],
                actor_sampling_seed=row['actor_sampling_seed'], recording_id=row['recording_id'],
                config_sha256=suite['phase_configs'][0]['sha256'])


def _shared(suite):
    return dict(
        schema=SCHEMA, campaign_id=CAMPAIGN, suite=reference(SUITE),
        original_artifacts=_anchors(suite), source_pins=_sources(),
        bridge=reference(BRIDGE), renderer=reference(RENDERER),
        first=_first(suite), recovery_control=str(RECOVERY_CONTROL), output=str(RECOVERY_OUTPUT),
        original_python_output=str(OUTPUT), material_root=str(MATERIALS),
        master_plan=suite['master_plan'], scenario_table=suite['scenario_table'],
        identities=suite['identities'], resume_offset=0, original_shifts=TOTAL,
        additional_shifts=0, previous_physical_shifts=0,
    )


def check_previous_stop(manifest, suite, configs):
    """Verify that the original first identity has never received READY or run."""
    previous_status = read(PREVIOUS_STATUS)
    receipt = read(LAUNCH / 'launch_receipt.json')
    intent = read(LAUNCH / 'launch_intent.json')
    stop = (CONTROL / 'STOP.properties').read_text()
    trace = (LAUNCH / 'controller.log').read_text()
    require(previous_status['campaign_id'] == CAMPAIGN and previous_status['status'] == 'STOP'
            and previous_status['reason'] == PREVIOUS_ERROR
            and stop == f'reason={PREVIOUS_ERROR}\n'
            and 'record = ledger.commit_record(update)' in trace
            and trace.rstrip().endswith('ValueError: committed update absent'),
            'original controller did not stop at the empty commit lookup')
    require(intent == _intent() and receipt['suite'] == reference(SUITE)
            and read(APPROVAL) == authorization_for(suite), 'original execution scope differs')
    process = _process_identity(receipt['pid'])
    require(not process.get('inspection_denied')
            and (process.get('state') == 'Z' or process.get('start_ticks') != receipt['process_start_ticks']),
            'original controller might still be active')
    require(set(CONTROL.iterdir()) == {CONTROL / 'STOP.properties'}
            and not MATERIALS.exists() and not (OUTPUT / 'training_progress.json').exists()
            and not (OUTPUT / 'evaluation').exists(), 'original simulator handoff or collection artifacts exist')
    arm = Path(suite['arms'][0]['python_output'])
    allowed = {ref['path'] for name, ref in manifest['original_artifacts'].items()
               if name.startswith('initial')}
    require({str(item) for item in arm.iterdir()} == allowed, 'unexpected files beside initial checkpoint')
    require(read(arm / 'initial_checkpoint.json') == reference(arm / 'initial.pt'),
            'original u0 checkpoint reference changed')
    require(manifest['original_ledger'] == reference(LEDGER)
            and not any(LEDGER.with_name(LEDGER.name + suffix).exists()
                        for suffix in ('-journal', '-wal')), 'original registry bytes changed')
    state = MixedLedger(suite, configs).snapshot()
    require(Counter(row['status'] for row in state['jobs'].values()) == {'PLANNED': TOTAL}
            and not state['campaigns'][CAMPAIGN]['commits'], 'first frozen shift may have been consumed')
    check_collisions(suite)
    return state


def load_recovery():
    """Read the immutable descriptor, including the original frozen source pins."""
    suite, configs = load_package()
    data = read(RECOVERY)
    expected = _shared(suite)
    require(all(data[key] == value for key, value in expected.items())
            and data['helper'] == reference(HELPER) and data['original_ledger']['path'] == str(LEDGER)
            and data['java_verification']['rows_checked'] == TOTAL
            and data['java_verification']['simulation_started'] is False,
            'recovery or original frozen bytes differ')
    return data, suite, configs


def prepare():
    """Freeze matching recovery Java and launch identities without starting it."""
    from recovery_java import verify_java, write_java

    require(not PREPARED.exists() and not RECOVERY_OUTPUT.exists(), 'new recovery directories required')
    suite, configs = load_package()
    data = dict(_shared(suite), original_ledger=reference(LEDGER))
    check_previous_stop(data, suite, configs)
    PREPARED.mkdir()
    write_java(suite['java_helper']['path'], HELPER, CONTROL, RECOVERY_CONTROL,
               suite['one_click_entry']['path'], BRIDGE, SUITE, RECOVERY)
    result = verify_java(HELPER, suite['scenario_table']['path'])
    require(result['compiled'] and result['rows_checked'] == TOTAL
            and not result['simulation_started'], 'offline recovery Java verification incomplete')
    data.update(helper=reference(HELPER), java_verification=result)
    write(RECOVERY, data)
    write(PREPARED / 'launch_identity.json', dict(
        recovery=reference(RECOVERY), **_first(suite),
        phase_config=suite['phase_configs'][0], python_output=str(OUTPUT),
        first_phase_python_output=configs[0]['python_output'], recovery_output=str(RECOVERY_OUTPUT),
        anylogic_event_root=str(MATERIALS),
        first_phase_event_directory=suite['jobs'][0]['material_event_directory'],
        identities=suite['identities'], resume_offset=0, additional_shifts=0,
    ))
    loaded, suite, configs = load_recovery()
    check_previous_stop(loaded, suite, configs)
    require(not RECOVERY_OUTPUT.exists(), 'offline recovery preparation started the live campaign')
    print(f'RECOVERY_PREPARED_ONLY {RECOVERY}')


if __name__ == '__main__':
    prepare()
