"""Mutation tests for independent, read-only FIFO release verification."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('fifo_release_verify', HERE/'verify_results.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
POLICY = 'dispatch_fifo_v3'
SMOKE = audit.PROJECT/'runs/bg128-baseline-all-recheck-20261122-001'


def _row(slot=10025004):
    path = SMOKE/f'run_{slot}'/'evaluation_receipt.json'
    return json.loads(path.read_bytes())


def _fifo_row():
    row = _row()
    row['policy_kind'] = POLICY
    eng = row['engineering']
    live = eng['first_live_multitruck_match']
    matched = eng['action_counts']['live_multitruck_matches']
    eng['action_counts'].update(fifo_candidates=4*eng['decisions'], fifo_roots_completed=4*eng['decisions'],
        fifo_trials_completed=matched, fifo_trials_compared=matched, fifo_trials_nominated=matched,
        fifo_trials_rejected=0, fifo_requests_with_compared_trial=matched, fifo_requests_with_completed_trial=matched,
        fifo_model_error_requests=0, fifo_guidance_error_requests=0, fifo_selected_from_cohort_requests=0)
    first = dict(request_ordinal=live['decision_sequence'], action=live['action'],
        selected_actions=[live['action']], selected_truck_ids=[live['truck_id']], roots_completed=4,
        trials_completed=1, trials_compared=1, trials_nominated=1, selected_from_cohort=False,
        selected_feasible=True, selected_cost=0.1)
    eng.update(first_completed_fifo_trial=first, first_compared_fifo_trial=dict(first), fifo_planning_errors=[])
    row['engineering_acceptance'].update(fifo_trial_observed=True, fifo_executable_comparison_observed=True,
                                         fifo_models_valid=True)
    return row


def test_checked_physical_cost_and_executable_matching():
    row = _fifo_row()
    audit._check_fifo(row)
    altered = deepcopy(row)
    altered['physical']['windows'][0]['metal_tons'] += 100
    with pytest.raises(ValueError, match='settlement'):
        audit._check_fifo(altered)
    altered = deepcopy(row)
    altered['engineering']['action_counts']['live_multitruck_matches'] = 0
    with pytest.raises(ValueError, match='executable FIFO'):
        audit._check_fifo(altered)
    altered = deepcopy(row)
    altered['engineering']['first_compared_fifo_trial']['action'] = -1
    with pytest.raises(ValueError, match='invalid acknowledged'):
        audit._check_fifo(altered)
    row['engineering']['fifo_planning_errors'] = [dict(stage='model', reason='projection_error', requests=1)]
    row['engineering']['action_counts']['fifo_model_error_requests'] = 1
    with pytest.raises(ValueError, match='engineering acceptance'):
        audit._check_fifo(row)
    row['engineering_acceptance'].update(passed=False, fifo_models_valid=False)
    audit._check_fifo(row)


def test_synthetic_100_slot_runtime_report_and_independent_statistics(monkeypatch, tmp_path):
    # These are explicitly synthetic paired report inputs, never physical run results.
    monkeypatch.syspath_prepend(str(audit.DEFAULT_PLAN))
    import v3_plan
    import v3_runtime

    old, bg = _row(), _row(10025005)
    groups = {arm: [] for arm in (POLICY, 'dispatch_inspired', 'bg128')}
    anchors = []
    for index in range(100):
        fields = dict(pair_ordinal=index, experiment_seed=1200 + index)
        before, other = dict(old, **fields), dict(bg, **fields)
        revised = dict(bg if index < 60 else old, **fields, policy_kind=POLICY)
        groups['dispatch_inspired'].append(before)
        groups['bg128'].append(other)
        groups[POLICY].append(revised)
        anchors.append(dict(row=before))
    source = tmp_path/'synthetic-historical-report.json'
    source.write_text(json.dumps(dict(shifts=groups['bg128'])))
    previous = dict(anchors=anchors, report=v3_plan.reference(source), completion={'synthetic': True})
    report = v3_runtime._summarize(dict(campaign_id='synthetic-100-pair-test', method={}), groups[POLICY],
                                   [{'synthetic_slot': i} for i in range(100)], previous)
    assert report['physical_shifts'] == 100
    assert report['comparisons'][0]['metrics']['combined_cost']['ties'] == 40
    audit._verify_statistics(report, groups, POLICY)
    altered = deepcopy(report)
    altered['comparisons'][0]['metrics']['combined_cost']['paired_bootstrap_ci95'][0] += .2
    with pytest.raises(ValueError, match='statistic differs'):
        audit._verify_statistics(altered, groups, POLICY)
    altered = deepcopy(report)
    altered['comparisons'][1]['metrics']['complete_task']['mcnemar_two_sided_p'] += .1
    with pytest.raises(ValueError, match='statistic differs'):
        audit._verify_statistics(altered, groups, POLICY)


def test_wrong_slot_and_wrong_terminal_or_claim_path_rejected(tmp_path):
    row = _fifo_row()
    original = _row()
    source = dict(original, pair_ordinal=0)
    bg = dict(source)
    source_ref = dict(path=str(tmp_path/'historical.json'), sha256='0'*64)
    row.update(pair_ordinal=0, schema='test_schema', v2_anchor=source_ref)
    job = {key: row[key] for key in audit.v2.SHIFT_FIELDS}
    job.update(actor_seed=0, actor_sampling_seed_consumed=False, ordinal=0,
               v2_reference_run_index=source['run_index'], v2_recording_id=source['recording_id'],
               v2_phase_config=source['phase_config'], event_bundle=str(tmp_path / row['episode_id']),
               python_output=str(tmp_path), expected_rpc_config=row['actual_rpc_configuration'])
    from python.runtime_scenario import PREFIX, fingerprint
    job['runtime_scene_sha256'] = fingerprint({k: v for k, v in job['expected_rpc_config'].items() if k.startswith(PREFIX)})
    from hashlib import sha256

    def ref(path):
        return dict(path=str(path), sha256=sha256(path.read_bytes()).hexdigest())

    suite = dict(control_directory=str(tmp_path), campaign_id='audit-fixture', master_plan={'sha256': '2'*64},
                 method={'response_target_ms': 1000})
    claim = tmp_path/'java_claims'/f"run_{job['run_index']}.once.claim"
    claim.parent.mkdir()
    fields = dict(protocol='baseline1-one-click-java-start-v1', campaign_id=suite['campaign_id'],
                  master_plan_sha256=suite['master_plan']['sha256'], ordinal=0, run_index=job['run_index'],
                  snapshot_sha256=job['phase_config']['sha256'], config_sha256=job['phase_config']['sha256'])
    claim.write_text(''.join(f'{k}={v}\n' for k, v in fields.items()))
    row['java_start_claim'] = ref(claim)
    path = tmp_path/'evaluation_receipt.json'

    def check():
        path.write_text(json.dumps(row))
        receipt = ref(path)
        slot = dict(status='COMPLETE', receipt=receipt, identity=job)
        audit._verify_receipt(suite, job, row, receipt, slot, source, source_ref, bg,
                              SimpleNamespace(RECEIPT_SCHEMA='test_schema'))

    check()
    bg['experiment_seed'] += 1
    with pytest.raises(ValueError, match='same-slot'):
        check()
    bg['experiment_seed'] -= 1
    row['terminal_request_id'] += '_bad'
    with pytest.raises(ValueError, match='callback, END'):
        check()
    row['java_start_claim']['path'] = str(tmp_path/'wrong_claim')
    with pytest.raises(ValueError, match='unexpected artifact path'):
        check()


def test_unsealed_campaign_cannot_be_reported(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(audit.DEFAULT_PLAN))
    import v3_plan
    monkeypatch.setattr(v3_plan, 'OUTPUT', tmp_path)
    with pytest.raises(ValueError, match='no COMPLETE seal'):
        audit.verify(v3_plan.PLAN)
    assert not list(tmp_path.iterdir())
