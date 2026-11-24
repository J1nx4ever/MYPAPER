"""Read-only verification of the ten sealed six-truck evaluation receipts."""
import json
from pathlib import Path
import sys

import campaign as plan

sys.path.insert(0, str(plan.PLAN))

from fleet_runtime import template as fleet
from traditional_dispatch_v2.runtime import RECEIPT_SCHEMA
from python.runtime_scenario import PREFIX, fingerprint
from change_plans.token_attention_001.business_graph_summary_009.baseline_comparison_001.all_recheck_001.verify_results import (
    verify_shift,
)


def _verify_receipt(suite, job, row, pin, slot):
    plan.require(slot == dict(identity=job, status='COMPLETE', receipt=pin)
                 and row == plan.load_reference(pin)
                 and Path(pin['path']) == Path(job['python_output'])/'evaluation_receipt.json',
                 'receipt or its budget slot differs')
    keys = ('run_index', 'recording_id', 'experiment_seed', 'actor_sampling_seed',
            'policy_kind', 'scene_id', 'pair_ordinal', 'phase_config')
    plan.require(all(row[key] == job[key] for key in keys)
                 and row['schema'] == RECEIPT_SCHEMA and row['completed_shifts'] == 1
                 and all(row[key] == suite[key] for key in ('actor', 'checkpoint'))
                 and row['actor_digest'] == plan.load_reference(suite['actor_manifest'])['actor_digest'],
                 'evaluated checkpoint or physical identity differs')
    claim = row['java_start_claim']
    claim_path = plan.CONTROL/'java_claims'/f"run_{job['run_index']}.once.claim"
    plan.require(claim == plan.reference(claim_path), 'Java claim differs')
    pairs = [line.split('=', 1) for line in claim_path.read_text().splitlines()]
    values = dict(pairs)
    sha = job['phase_config']['sha256']
    plan.require(len(pairs) == len(values) and values == dict(protocol='baseline1-one-click-java-start-v1',
        campaign_id=plan.CAMPAIGN, master_plan_sha256=suite['master_plan']['sha256'], ordinal=str(job['ordinal']),
        run_index=str(job['run_index']), snapshot_sha256=sha, config_sha256=sha), 'Java START identity differs')
    rpc = row['actual_rpc_configuration']
    plan.require(all(rpc.get(key) == value for key, value in job['expected_rpc_config'].items())
                 and fingerprint({k: v for k, v in rpc.items() if k.startswith(PREFIX)}) == job['runtime_scene_sha256']
                 and row['episode_id'] == Path(job['event_bundle']).name, 'actual RPC configuration or episode differs')
    e = row['engineering']
    outage = e['fleet_outage']
    dispatches = outage['active_dispatches']
    decisions = e['decisions']
    plan.require(outage['disabled_ids'] == list(fleet.DISABLED) and outage['active_ids'] == list(fleet.ACTIVE)
                 and set(dispatches) == set(map(str, fleet.ACTIVE))
                 and all(type(count) is int and count >= 0 for count in dispatches.values())
                 and sum(dispatches.values()) == decisions and outage['validated_callbacks'] == decisions + 1
                 and row['terminal_request_id'] == f"{row['episode_id']}-{decisions}",
                 'fleet callback or dispatch accounting differs')
    common = dict(rpc_and_physical_completion=True, met_response_target=e['rpc_responses_over_target'] == 0)
    expected = dict(common, fleet_outage_validated=True, all_six_active_trucks_dispatched=all(dispatches.values()))
    plan.require(row['engineering_acceptance'] == dict(expected, passed=all(expected.values())),
                 'fleet engineering gate differs')
    verify_shift(dict(row, engineering_acceptance=dict(common, passed=all(common.values()))))


def verify():
    seal_path = plan.OUTPUT/'suite_complete.json'
    plan.require(seal_path.is_file(), 'the six-truck activity has no physical COMPLETE seal')
    suite = plan.load_suite()
    seal = json.loads(seal_path.read_bytes())
    report_pin = plan.reference(plan.REPORT/plan.REPORT_FILENAME)
    report = plan.load_reference(report_pin)
    plan.require(seal['status'] == report['status'] == 'COMPLETE'
                 and seal['campaign_id'] == report['campaign_id'] == plan.CAMPAIGN
                 and seal['suite'] == plan.reference(plan.SUITE) and seal['evaluation'] == report_pin
                 and seal['protected_core'] == suite['protected_core']
                 and seal['training_shifts'] == seal['optimizer_updates'] == report['optimizer_updates'] == 0
                 and seal['physical_shifts'] == report['evaluation_slots'] == report['completed_shifts']
                 == report['physical_shifts'] == plan.COUNT
                 and report['outage_configuration'] == fleet.OUTAGE_FIELDS
                 and report['method'] == suite['method'], 'campaign completion or report scope differs')
    slots = plan.Ledger(suite).snapshot()['slots']
    for job, row, pin in zip(suite['jobs'], report['shifts'], report['receipts'], strict=True):
        _verify_receipt(suite, job, row, pin, slots[str(job['run_index'])])
    plan.require(report['summary'] == fleet._summary(report['shifts']), 'sealed fleet summary differs')
    return dict(status='VERIFIED_COMPLETE', campaign_id=plan.CAMPAIGN, physical_shifts=plan.COUNT,
                optimizer_updates=0, summary=report['summary'])


if __name__ == '__main__':
    try:
        print(json.dumps(verify(), ensure_ascii=False, allow_nan=False))
    except (OSError, ValueError, KeyError) as error:
        sys.exit(f'Verification failed: {error}')
