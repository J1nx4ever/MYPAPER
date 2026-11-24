"""Freeze continuation after re-admitting the completed first BG128 shift."""
from collections import Counter
from pathlib import Path
import sys

PLAN = Path(__file__).resolve().parent
sys.path.insert(0, str(PLAN.parent))

from mixed_plan import (
    CAMPAIGN, CONTROL, CORE, LEDGER, MATERIALS, OUTPUT, REPORT, ROOT, SUITE, TOTAL,
    check_collisions, load_package, read, reference, require,
)
from mixed_budget import MixedLedger
from business_graph_summary_campaign_v1.health import _process_identity
from business_graph_summary_campaign_v1.plan_io import write

OFFSET = 1
COMPLETED_RUN = 10000001
FIRST_REMAINING = 10000002
REMAINING = TOTAL - OFFSET
PREVIOUS = OUTPUT / 'recovery_001'
RECOVERY_OUTPUT = OUTPUT / PLAN.name
RECOVERY_CONTROL = RECOVERY_OUTPUT / 'control'
RECONCILIATION = REPORT / PLAN.name / 'admission' / 'reconciliation.json'
PREPARED = PLAN / 'prepared_001'
RECOVERY = PREPARED / 'recovery.json'
HELPER = PREPARED / 'Baseline1CampaignPlan.java'
BRIDGE = PLAN / 'recovery_bridge.py'
SCHEMA = 'bg128-continuous-mixed-post-first-shift-recovery-v1'
PREVIOUS_ERROR = "ValueError('unknown task-query candidate')"


def _first(suite):
    job = suite['jobs'][OFFSET]
    require(job['run_index'] == FIRST_REMAINING and job['ordinal'] == OFFSET, 'unexpected resume identity')
    return dict(run_index=job['run_index'], experiment_seed=job['experiment_seed'],
                actor_sampling_seed=job['actor_sampling_seed'], recording_id=job['recording_id'],
                config_sha256=suite['phase_configs'][0]['sha256'])


def _anchors(suite):
    arm = Path(suite['arms'][0]['python_output'])
    first = suite['jobs'][0]
    phase = Path(first['python_output'])
    raw = phase / 'episodes' / f'run_{COMPLETED_RUN}'
    return {
        'original_stop': reference(CONTROL / 'STOP.properties'),
        'previous_stop': reference(PREVIOUS / 'control' / 'STOP.properties'),
        'previous_status': reference(PREVIOUS / 'run_status.json'),
        'previous_launch': reference(PREVIOUS / 'launch' / 'launch_receipt.json'),
        'previous_intent': reference(PREVIOUS / 'launch' / 'launch_intent.json'),
        'previous_authority': reference(PREVIOUS / 'launch' / 'run_authority.json'),
        'previous_log': reference(PREVIOUS / 'launch' / 'controller.log'),
        'first_ready': reference(PREVIOUS / 'control' / 'ready' / f'run_{COMPLETED_RUN}.properties'),
        'first_claim': reference(PREVIOUS / 'control' / 'java_claims' / f'run_{COMPLETED_RUN}.once.claim'),
        'first_raw': reference(raw / 'manifest.json'),
        'first_raw_receipt': reference(raw / 'receipt.json'),
        'first_journal': reference(first['material_journal']),
        'first_actor': reference(phase / 'actor.pt'),
        'initial_checkpoint': reference(arm / 'initial_checkpoint.json'),
        'protected_core': reference(CORE),
    }


def _shared(suite):
    storage = ROOT / 'business_graph_summary_storage_v1'
    return dict(schema=SCHEMA, campaign_id=CAMPAIGN, suite=reference(SUITE),
                sources={path.name: reference(path) for path in sorted(PLAN.glob('*.py'))},
                bg128_storage={name: reference(storage / name) for name in ('__init__.py', 'episode.py')},
                previous_artifacts=_anchors(suite), reconciliation=reference(RECONCILIATION),
                first=_first(suite), prior_completed_run=COMPLETED_RUN, resume_offset=OFFSET,
                remaining_shifts=REMAINING, original_shifts=TOTAL, additional_shifts=0,
                original_python_output=str(OUTPUT), recovery_output=str(RECOVERY_OUTPUT),
                recovery_control=str(RECOVERY_CONTROL), material_root=str(MATERIALS),
                master_plan=suite['master_plan'], scenario_table=suite['scenario_table'],
                identities=suite['identities'])


def check_start(manifest, suite, configs):
    """Require the first exact completion and no use of any remaining identity."""
    status = read(PREVIOUS / 'run_status.json')
    previous_stop = PREVIOUS / 'control' / 'STOP.properties'
    receipt = read(PREVIOUS / 'launch' / 'launch_receipt.json')
    process = _process_identity(receipt['pid'])
    require(not process.get('inspection_denied')
            and (process.get('state') == 'Z' or process.get('start_ticks') != receipt['process_start_ticks'])
            and status['status'] == 'STOP' and status['reason'] == PREVIOUS_ERROR
            and previous_stop.read_text() == f'reason={PREVIOUS_ERROR}\n',
            'previous controller still active or stopped for a different reason')
    require(manifest['reconciled_registry'] == reference(LEDGER)
            and not any(LEDGER.with_name(LEDGER.name + suffix).exists() for suffix in ('-journal', '-wal')),
            'reconciled ledger bytes changed')
    state = MixedLedger(suite, configs).snapshot()
    records = state['jobs']
    require(Counter(row['status'] for row in records.values()) == {'COMPLETE': OFFSET, 'PLANNED': REMAINING}
            and not state['campaigns'][CAMPAIGN]['commits'], 'expected one completed shift and no optimizer commits')
    proof = read(RECONCILIATION)
    recovered = records[str(COMPLETED_RUN)]
    require(proof['status'] == 'APPLIED' and proof['run_index'] == COMPLETED_RUN
            and recovered['identity'] == suite['jobs'][0] and recovered['status'] == 'COMPLETE'
            and recovered['reconciled_from'] == 'UNKNOWN'
            and recovered['dataset_ref'] == proof['dataset_ref']
            and reference(proof['dataset_ref']['path']) == proof['dataset_ref'],
            'first completed shift does not bind the reconciled dataset')
    for name in ('ready', 'java_claims'):
        patterns = list((PREVIOUS / 'control' / name).glob('run_*'))
        require(len(patterns) == 1 and patterns[0].name.startswith(f'run_{COMPLETED_RUN}.'),
                'previous handoff reached a remaining identity')
        require(not any((CONTROL / name).glob('run_*')), 'original handoff reached a physical slot')
    for job in suite['jobs'][OFFSET:]:
        require(not (Path(job['python_output']) / 'episodes' / f"run_{job['run_index']}").exists()
                and not Path(job['material_event_bundle']).exists(),
                'pending run already has physical artifacts')
    check_collisions(suite)
    return state


def load_recovery():
    suite, configs = load_package()
    data = read(RECOVERY)
    shared = _shared(suite)
    require(all(data[key] == value for key, value in shared.items())
            and data['helper'] == reference(HELPER) and data['reconciled_registry']['path'] == str(LEDGER)
            and data['java_verification']['remaining_rows_checked'] == REMAINING
            and data['java_verification']['simulation_started'] is False,
            'continuation or frozen ancestor bytes differ')
    return data, suite, configs


def prepare():
    """Seal the recovery helper without acquiring a live listener or a job."""
    from recovery_java import verify_java, write_java

    require(not PREPARED.exists() and not RECOVERY_CONTROL.exists()
            and not (RECOVERY_OUTPUT / 'launch').exists(), 'recovery already prepared or started')
    suite, configs = load_package()
    data = dict(_shared(suite), reconciled_registry=reference(LEDGER))
    check_start(data, suite, configs)
    PREPARED.mkdir()
    write_java(suite['java_helper']['path'], HELPER, CONTROL, RECOVERY_CONTROL,
               suite['one_click_entry']['path'], BRIDGE, SUITE, RECOVERY)
    result = verify_java(HELPER, suite['scenario_table']['path'])
    require(result['compiled'] and result['remaining_rows_checked'] == REMAINING
            and result['prior_run_excluded'] and not result['simulation_started'],
            'resume-offset Java verification incomplete')
    data.update(helper=reference(HELPER), java_verification=result)
    write(RECOVERY, data)
    write(PREPARED / 'launch_identity.json', dict(
        recovery=reference(RECOVERY), **_first(suite), first_remaining=FIRST_REMAINING,
        last_remaining=suite['jobs'][-1]['run_index'], phase_config=suite['phase_configs'][0],
        python_output=str(OUTPUT), first_phase_python_output=configs[0]['python_output'],
        recovery_output=str(RECOVERY_OUTPUT), anylogic_event_root=str(MATERIALS),
        first_phase_event_directory=suite['jobs'][OFFSET]['material_event_directory'],
        identities=suite['identities'], resume_offset=OFFSET, remaining_shifts=REMAINING, additional_shifts=0,
    ))
    checked, suite, configs = load_recovery()
    check_start(checked, suite, configs)
    require(not RECOVERY_CONTROL.exists() and not (RECOVERY_OUTPUT / 'launch').exists(),
            'offline preparation started a controller')
    print(f'SECOND_RECOVERY_PREPARED_ONLY {RECOVERY}')


if __name__ == '__main__':
    prepare()
