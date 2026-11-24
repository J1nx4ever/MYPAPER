"""Bind this recovery to the original 600-shift campaign and closed first failure."""
from collections import Counter
from pathlib import Path
import sys

PLAN = Path(__file__).resolve().parent
sys.path.insert(0, str(PLAN.parent))

from gen_common import (
    APPROVAL, CAMPAIGN, CONTROL, CORE, LAUNCH, MATERIALS, OUTPUT, REPORT, ROOT, SUITE, TOTAL,
    authorization_for, checked, load_package, read, reference, require,
)
from gen_budget import GenLedger

RECOVERY = 'recovery_001'
SUSPENDED_RUN = 9991101
FIRST_REMAINING = 9991102
LAST_REMAINING = 9991600
OFFSET = 101
REMAINING = TOTAL - OFFSET
OUTPUT_DIR = OUTPUT / RECOVERY
CONTROL_DIR = OUTPUT_DIR / 'control'
LAUNCH_DIR = OUTPUT_DIR / 'launch'
REPORT_DIR = REPORT / RECOVERY
PREPARED_DIR = PLAN / 'prepared_001'
DESCRIPTOR = PREPARED_DIR / 'recovery.json'
LEDGER = OUTPUT / 'generalization_ledger.sqlite'
RECONCILIATION = REPORT_DIR / 'admission/reconciliation.json'


class RegisteredLedger(GenLedger):
    """The shared evaluator enters an already registered campaign."""

    def register(self):
        self.resume()


def source_pins():
    return {path.name: reference(path) for path in sorted(PLAN.glob('*.py'))
            if not path.name.startswith('test_')}


def anchor_files():
    return {
        'original_stop': reference(CONTROL / 'STOP.properties'),
        'original_launch': reference(LAUNCH / 'launch_receipt.json'),
        'original_authorization': reference(APPROVAL),
        'first_failed_java_claim': reference(CONTROL / 'java_claims' / f'run_{SUSPENDED_RUN}.once.claim'),
        'core': reference(CORE),
    }


def original_campaign():
    suite, phases = load_package()
    authority = read(APPROVAL)
    intent = read(LAUNCH / 'launch_intent.json')
    require(authority == authorization_for(suite)
            and intent['suite'] == reference(SUITE)
            and intent['authorization'] == reference(APPROVAL)
            and intent['shifts'] == TOTAL, 'original scoped launch differs')
    return suite, phases


def assert_start_state(suite, ledger):
    state = ledger.resume()
    slots = state['jobs']
    require(Counter(row['status'] for row in slots.values()) == {'COMPLETE': OFFSET, 'PLANNED': REMAINING}
            and state['campaigns'][CAMPAIGN]['commits'] == {},
            'recovery must open at exactly 101 completed plus 499 original planned slots')
    require(all(slots[str(job['run_index'])]['status'] == ('COMPLETE' if i < OFFSET else 'PLANNED')
                for i, job in enumerate(suite['jobs'])), 'recovery slots differ from original order')
    require(suite['jobs'][OFFSET]['run_index'] == FIRST_REMAINING and
            suite['jobs'][-1]['run_index'] == LAST_REMAINING, 'remaining identity range differs')
    require(not any((CONTROL / 'java_claims' / f"run_{job['run_index']}.once.claim").exists()
                for job in suite['jobs'][OFFSET:]),
            'original Java has already used a remaining identity')
    require(not any((Path(job['python_output']) / 'episodes' / f"run_{job['run_index']}").exists()
                    or Path(job['material_event_bundle']).exists() for job in suite['jobs'][OFFSET:]),
            'an original planned shift already has physical artifacts')
    return slots[str(SUSPENDED_RUN)]


def preflight(path=DESCRIPTOR):
    require(Path(path).resolve() == DESCRIPTOR, 'fixed recovery descriptor location required')
    data = read(DESCRIPTOR)
    suite, phases = original_campaign()
    require(data['schema'] == 'bg128-generalization-recovery-v1'
            and data['original_suite'] == reference(SUITE)
            and data['source_pins'] == source_pins()
            and data['original_anchors'] == anchor_files()
            and data['control_directory'] == str(CONTROL_DIR)
            and data['campaign_id'] == CAMPAIGN
            and data['resume_offset'] == OFFSET and data['remaining_shifts'] == REMAINING
            and data['additional_shifts'] == 0 and data['frozen_shifts'] == TOTAL
            and data['master_plan'] == suite['master_plan']
            and data['scenario_table'] == suite['scenario_table']
            and data['checkpoint'] == suite['parent_artifacts']['checkpoint']
            and data['reconciliation'] == reference(RECONCILIATION),
            'recovery descriptor or original plan bytes differ')
    require(checked(data['java_helper']) == PREPARED_DIR / 'Baseline1CampaignPlan.java',
            'recovery helper differs')
    checked(data['remaining_identities'])
    ledger = RegisteredLedger(suite, suite_path=SUITE, approval_path=APPROVAL)
    repaired = assert_start_state(suite, ledger)
    proof = read(RECONCILIATION)
    require(proof['status'] == 'APPLIED' and proof['run_index'] == SUSPENDED_RUN
            and proof['original_status'] == 'UNKNOWN' and proof['new_status'] == 'COMPLETE'
            and proof['new_reserved_shifts'] == proof['new_physical_shifts'] == 0
            and proof['completed_after'] == OFFSET and proof['planned_after'] == REMAINING
            and repaired.get('reconciled_from') == 'UNKNOWN' and proof['dataset_ref'] == repaired['dataset_ref'],
            'reconciliation does not bind the original failed slot')
    for key in ('before_database', 'before_unknown_row', 'dataset_ref'):
        checked(proof[key])
    return data, suite, phases, ledger
