"""Analyze the nominal comparison on the frozen environment-seed slots."""
from pathlib import Path
import sys

import numpy as np

from traditional_dispatch_v2.runtime import POLICIES, _verify_anchor
from traditional_dispatch_v1.runtime import _require


PREVIOUS = Path(__file__).resolve().parents[1] / 'paired_100_001'
if str(PREVIOUS) not in sys.path:
    sys.path.append(str(PREVIOUS))
from change_plans.token_attention_001.business_graph_summary_009.baseline_comparison_001.paired_100_001 import paired_report


SCHEMA = 'bg128-traditional-paired100-comparison-v2'
CONTRASTS = (
    ('bg128', 'rolling'),
    ('bg128', 'dispatch_inspired'),
    ('bg128', 'fifo_greedy'),
    ('bg128', 'unload_greedy'),
    ('rolling', 'fifo_greedy'),
    ('fifo_greedy', 'unload_greedy'),
)


def _summarize_arm(records):
    output = paired_report._summarize_arm(records)
    engineering = [row['engineering'] for row in records]
    # The earlier aggregate sums every counter. These two V2 fields are maxima.
    for field in ('max_scored_depth', 'max_matched_trucks'):
        if field in output['action_counts']:
            output['action_counts'][field] = max(row['action_counts'].get(field, 0) for row in engineering)
    telemetry = ('callbacks_with_both_flags', 'total_callbacks')
    output['native_fallback_telemetry'] = {
        key: sum(row['native_fallback_telemetry'][key] for row in engineering) for key in telemetry}
    return output


def summarize(shifts, receipts, *, suite):
    """Keep every completed physical shift and use paired rather than marginal draws."""
    slots = suite['evaluation_slots']
    _require(len(shifts) == len(receipts) == len(suite['jobs']) == slots * len(POLICIES),
             'all five arms of every frozen environment slot are required')
    fields = ('run_index', 'recording_id', 'scene_id', 'experiment_seed', 'actor_sampling_seed',
              'policy_kind', 'phase_config', 'pair_ordinal')
    groups = {policy: [] for policy in POLICIES}
    environment_seeds = set()
    for slot in range(slots):
        offset = slot * len(POLICIES)
        anchor = shifts[offset]
        _require(anchor['experiment_seed'] not in environment_seeds, 'environment seed reused between slots')
        environment_seeds.add(anchor['experiment_seed'])
        for index, policy in enumerate(POLICIES, offset):
            row, job = shifts[index], suite['jobs'][index]
            _require((row['pair_ordinal'], row['policy_kind']) == (slot, policy)
                     and all(row[key] == job[key] for key in fields), 'receipt does not match its frozen roster position')
            _require(row['status'] == 'COMPLETE' and row['normal_completion'] is True
                     and row['completed_shifts'] == 1 and row['optimizer_updates'] == 0,
                     'incomplete or non-evaluation shift in the matched analysis')
            _verify_anchor(row, anchor)
            groups[policy].append(row)

    rng = np.random.default_rng(paired_report.ANALYSIS_SEED)
    draws = rng.integers(0, slots, size=(paired_report.BOOTSTRAP_REPLICATES, slots))
    engineering = {}
    for name, rows in groups.items():
        passed = sum(row['engineering_acceptance']['passed'] for row in rows)
        engineering[name] = dict(passed=passed, failed=len(rows) - passed)
    return dict(schema=SCHEMA, status='COMPLETE', campaign_id=suite['campaign_id'], evaluation_slots=slots,
                completed_shifts=len(shifts), physical_shifts=len(shifts), optimizer_updates=0,
                method=suite['method'], pairing=suite['pairing'], shifts=shifts, receipts=receipts,
                primary_outcome='mean_paired_combined_cost_bg128_minus_rolling',
                intervals=dict(method='percentile_bootstrap_over_environment_seed_slots',
                               replicates=paired_report.BOOTSTRAP_REPLICATES, seed=paired_report.ANALYSIS_SEED,
                               level=.95, secondary_contrasts='descriptive_intervals_without_multiplicity_adjustment'),
                groups={name: _summarize_arm(rows) for name, rows in groups.items()},
                comparisons=[paired_report._matched_comparison(groups, first, second, draws)
                             for first, second in CONTRASTS], engineering_acceptance=engineering)
