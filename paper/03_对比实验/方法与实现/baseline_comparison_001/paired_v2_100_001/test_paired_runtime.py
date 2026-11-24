"""Check the 500-job contract without launching a simulator or listener."""
from copy import deepcopy
from math import comb

import numpy as np
import pytest

import formal_report
import paired_runtime as runtime


def _suite(tmp_path):
    jobs = []
    for slot in range(100):
        for policy in runtime.v2.POLICIES:
            index = len(jobs)
            bg = policy == 'bg128'
            jobs.append(dict(ordinal=index, pair_ordinal=slot, policy_kind=policy, run_index=10_000+index,
                             recording_id=f'test-{index}', scene_id='nominal', experiment_seed=100+slot,
                             actor_seed=8 if bg else 0, actor_sampling_seed=400+slot if bg else 0,
                             actor_sampling_seed_consumed=bg, checkpoint_update=32 if bg else 0,
                             event_bundle=f'event-{index}', python_output=str(tmp_path/f'run_{index}'),
                             phase_config=dict(path=f'phase-{index}', sha256=f'phase-sha-{index}')))
    return dict(schema=runtime.SUITE_SCHEMA, campaign_id='synthetic', evaluation_slots=100, physical_shifts=500,
                jobs=jobs, control_directory=str(tmp_path/'control'), report_directory=str(tmp_path/'report'),
                master_plan={'sha256': 'synthetic-master'}, method={'response_target_ms': 1000},
                pairing={'scope': 'this_campaign_only'})


def _record(job):
    policy, slot = job['policy_kind'], job['pair_ordinal']
    complete = slot >= (20 if policy == 'bg128' else 40)
    cost = 2.5 - .01*slot if policy == 'bg128' else 3.5
    window = dict(tons=4000., metal_tons=1080., grade=.27, empty=False, compliant=complete)
    physical = dict(combined_cost=cost, grade_cost=cost-.1, waiting_cost=.1, production_cost=0.,
                    total_wait_seconds=14_400., ore_tons=16_000., waste_tons=4_000., complete_task=complete,
                    ore_target_met=True, waste_target_met=True, all_four_windows_compliant=complete,
                    wait_breakdown=dict(load_seconds=6000., dump_resource_queue_seconds=6000.,
                                        dump_capacity_admission_seconds=2400.),
                    windows=[dict(window, window=i) for i in range(4)])
    counts = {'planned': 2}
    if policy == 'rolling':
        counts['max_scored_depth'] = 3
    if policy == 'dispatch_inspired':
        counts['max_matched_trucks'] = 2 if slot else 3
    engineering = dict(action_counts=counts, decisions=2, rpc_receive_to_response_ms={'p95': 10., 'max': 20.},
                       rpc_responses_over_target=0, native_fallback_telemetry={'callbacks_with_both_flags': 0,
                                                                             'total_callbacks': 3})
    return dict(job, status='COMPLETE', normal_completion=True, completed_shifts=1, optimizer_updates=0,
                initial_state={'observation_sha256': str(slot)}, actual_rpc_configuration={'scene': 'nominal'},
                physical=physical, engineering=engineering,
                engineering_acceptance={'passed': not (policy == 'rolling' and slot == 20)})


def test_full_report_pairs_by_slot_keeps_engineering_failures_and_reconciles_statistics(tmp_path):
    suite = _suite(tmp_path)
    shifts = [_record(job) for job in suite['jobs']]
    refs = [{'path': str(index)} for index in range(len(shifts))]
    report = formal_report.summarize(shifts, refs, suite=suite)
    assert report['physical_shifts'] == report['completed_shifts'] == 500
    assert report['shifts'] == shifts and report['receipts'] == refs
    assert report['engineering_acceptance']['rolling'] == {'passed': 99, 'failed': 1}
    assert report['groups']['rolling']['action_counts']['max_scored_depth'] == 3
    assert report['groups']['dispatch_inspired']['action_counts']['max_matched_trucks'] == 3
    assert report['groups']['rolling']['native_fallback_telemetry']['total_callbacks'] == 300
    assert [(row['first'], row['second']) for row in report['comparisons']] == list(formal_report.CONTRASTS)
    primary, *secondary = report['comparisons']
    assert primary['primary'] and not any(row['primary'] for row in secondary)
    assert primary['metrics']['combined_cost']['mean'] == pytest.approx(-1.495)
    draws = np.random.default_rng(90221).integers(100, size=(10_000, 100))
    expected_ci = np.percentile((-1 - .01*draws).mean(axis=1), [2.5, 97.5])
    assert primary['metrics']['combined_cost']['paired_bootstrap_ci95'] == pytest.approx(expected_ci)
    task = primary['metrics']['complete_task']
    assert (task['positive'], task['negative'], task['both_complete'], task['both_failed']) == (20, 0, 60, 20)
    assert task['mcnemar_two_sided_p'] == pytest.approx(2*comb(20, 20)/2**20)


@pytest.mark.parametrize('key', ['initial_state', 'actual_rpc_configuration', 'experiment_seed'])
def test_summary_rejects_within_slot_mismatch(tmp_path, key):
    suite = _suite(tmp_path)
    shifts = [_record(job) for job in suite['jobs']]
    shifts[3][key] = 'wrong'
    with pytest.raises(ValueError, match='frozen roster position|differs within the pair'):
        formal_report.summarize(shifts, [None] * 500, suite=suite)


class MemoryLedger:
    def __init__(self, suite):
        self.slots = {str(job['run_index']): dict(status='PLANNED', identity=deepcopy(job))
                      for job in suite['jobs']}

    def snapshot(self):
        return {'slots': deepcopy(self.slots)}


def test_runtime_keeps_slot_anchors_local_and_preserves_every_receipt(monkeypatch, tmp_path):
    suite = _suite(tmp_path)
    ledger = MemoryLedger(suite)
    progress, pins = [], []
    actor = object()
    monkeypatch.setattr(runtime.shared, '_check_startup', pins.append)
    monkeypatch.setattr(runtime.shared, '_load_actor', lambda _: (actor, 'actor-digest'))

    def run_shift(config, _startup, book, job, anchor, active_actor, digest):
        assert config is suite and book is ledger and digest == 'actor-digest'
        assert active_actor is (actor if job['policy_kind'] == 'bg128' else None)
        first = job['policy_kind'] == runtime.v2.POLICIES[0]
        assert (anchor is None) == first
        record = _record(job)
        runtime.v2._verify_anchor(record, anchor)
        ref = {'path': job['python_output'], 'sha256': str(job['ordinal'])}
        slot = ledger.slots[str(job['run_index'])]
        assert slot['status'] == 'PLANNED'
        slot.update(status='COMPLETE', receipt=ref)
        return record, ref

    monkeypatch.setattr(runtime.v2, '_run_shift', run_shift)
    result = runtime.run(suite, ledger, progress=lambda n, job: progress.append((n, job['ordinal'])))
    assert pins == [suite, suite]
    assert progress == [(i+1, i) for i in range(500)]
    report = runtime.shared.load_reference(result)
    assert report['physical_shifts'] == 500 and len(report['shifts']) == 500
    assert report['receipts'][-1] == ledger.slots[str(suite['jobs'][-1]['run_index'])]['receipt']


def test_started_ledger_and_reused_environment_seed_fail_before_listener(tmp_path):
    suite = _suite(tmp_path)
    ledger = MemoryLedger(suite)
    ledger.slots[str(suite['jobs'][0]['run_index'])]['status'] = 'CLAIMED'
    with pytest.raises(ValueError, match='exact fresh ledger'):
        runtime.run(suite, ledger)
    for job in suite['jobs'][5:10]:
        job['experiment_seed'] = suite['jobs'][0]['experiment_seed']
    with pytest.raises(ValueError, match='environment seed repeated outside'):
        runtime.run(suite, MemoryLedger(suite))
