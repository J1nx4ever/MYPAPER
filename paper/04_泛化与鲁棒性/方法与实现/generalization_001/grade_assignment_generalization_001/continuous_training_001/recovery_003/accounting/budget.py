"""Merge the original mixed campaign and its supplemental slot into history."""
from ..recovery_plan import CAMPAIGN, TOTAL, check_collisions, load_recovery
from ..recovery_ledger import ReplacementLedger


def budget_snapshot():
    manifest, suite, phases, replacement = load_recovery()
    historical = check_collisions(suite)
    current = ReplacementLedger(suite, phases, replacement, manifest['replacement_config']).snapshot()
    cap = current['campaigns'][CAMPAIGN]['cap']
    assert cap in (TOTAL, TOTAL + 1) and len(current['jobs']) == cap
    historical['jobs'].update(current['jobs'])
    historical['campaigns'].update(current['campaigns'])
    historical['maximum_total_shifts'] += cap
    historical['additional_reserved_shifts'] += cap
    return historical
