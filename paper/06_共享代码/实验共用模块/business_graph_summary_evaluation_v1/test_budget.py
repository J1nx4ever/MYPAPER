"""Independent slot-state and all-ledger collision regressions (temporary DBs)."""
import copy
import json
import sys
import types

import pytest

from . import budget


@pytest.fixture
def setup(tmp_path, monkeypatch):
    source = dict(maximum_total_shifts=1000, additional_reserved_shifts=1000,
        active_h2_reserved_shifts=0, legacy_reserved_shifts=0,
        campaigns={'prior': {}}, jobs={'1': dict(status='COMPLETE', identity=dict(
            run_index=1, experiment_seed=2, actor_sampling_seed=3, actor_seed=42,
            recording_id='old', source_recording_id='old'))})
    state = dict(legacy_independent_evaluation_budget=dict(path='old.json', sha256='a'*64),
                 sampling_comparison_budget=dict(path='sampling.json', sha256='b'*64))
    monkeypatch.setattr(budget, 'preceding_snapshot', lambda: copy.deepcopy(source))
    monkeypatch.setattr(budget, 'preceding_reference', lambda: copy.deepcopy(state))
    gate = types.ModuleType('business_graph_summary_evaluation_v1.gate')
    gate.check_jobs = budget._scope
    gate.check_sources = lambda pins: budget.require(pins == {'frozen': 'a'*64}, 'sources changed')

    def approval_check(approval, suite, suite_sha):
        budget.require(approval == dict(decision='approved', approved_by='user',
            execution_authorized=True, max_new_shifts=100,
            campaign_id=suite['campaign_id'], campaign_sha256=suite_sha,
            source_user_instruction='训练结束，选择最好的checkpoint推理100轮次'),
            'exact explicit100-shift authorization required')
    gate.check_approval = approval_check
    monkeypatch.setitem(sys.modules, gate.__name__, gate)

    def make(cid='eval', base=10001, expected_total=1100, prior=None, mutate=None):
        directory = tmp_path / cid
        directory.mkdir(exist_ok=True)
        path = directory / 'suite.json'
        jobs = [dict(ordinal=i, run_index=base+i, actor_seed=42,
            experiment_seed=base+100000+i, actor_sampling_seed=base+200000+i,
            recording_id=cid, source_recording_id=cid,
            role='final', partition='validation', checkpoint_update=13,
            target_training_update=0) for i in range(100)]
        suite = dict(contract=budget.SUITE_CONTRACT, kind='independent_evaluation',
            campaign_id=cid, suite_path=str(path), new_shift_cap=100, evaluation_shifts=100,
            training_shifts=0, training_updates=0, optimizer_steps=0,
            checkpoint_update=13, actor_seed=42, jobs=jobs, source_pins={'frozen': 'a'*64},
            budget_state_before=copy.deepcopy(state), evaluation_budget_before=prior,
            maximum_total_if_approved=expected_total)
        if mutate:
            mutate(suite)
        path.write_bytes(budget.encoded(suite))
        suite_sha = budget.reference(path)['sha256']
        approval = dict(decision='approved', approved_by='user', execution_authorized=True,
            max_new_shifts=100, campaign_id=cid, campaign_sha256=suite_sha,
            source_user_instruction='训练结束，选择最好的checkpoint推理100轮次')
        return suite, suite_sha, approval

    ledger_path = tmp_path / 'ledger.sqlite'

    def register(spec=None, **kw):
        suite, sha, approval = make() if spec is None else spec
        return budget.BudgetLedger(ledger_path, suite, campaign_sha256=sha,
            approval=approval, approval_sha256='c'*64, test_only=True, **kw)
    return types.SimpleNamespace(path=ledger_path, make=make, register=register,
                                  state=state, source=source)


def test_read_only_snapshot_never_creates_database(setup):
    assert budget.budget_snapshot(setup.path, test_only=True) == setup.source
    assert not setup.path.exists()


def test_register100_preserves_parent_actor_seed_and_merges_all_history(setup):
    ledger = setup.register()
    result = budget.budget_snapshot(setup.path, test_only=True)
    assert result['maximum_total_shifts'] == 1100
    assert result['independent_bg128_reserved_shifts'] == 100
    assert len(result['jobs']) == 101
    assert result['jobs']['1'] == setup.source['jobs']['1']
    assert all(j['status'] == 'PLANNED' for j in ledger.snapshot()['jobs'].values())
    assert ledger.snapshot()['campaigns']['eval']['commits'] == {}


def test_registration_idempotent_without_double_cap(setup):
    spec = setup.make()
    setup.register(spec)
    setup.register(spec)
    assert budget.budget_snapshot(setup.path, test_only=True)['maximum_total_shifts'] == 1100


def test_second100_uses_same_extension_and_exact_merged_total(setup):
    setup.register()
    ref = budget.reference(setup.path)
    spec = setup.make('eval2', base=20001, expected_total=1200, prior=ref)
    setup.register(spec)
    result = budget.budget_snapshot(setup.path, test_only=True)
    assert result['maximum_total_shifts'] == 1200
    assert len(result['jobs']) == 201


@pytest.mark.parametrize('change', [
    lambda s: s.update(new_shift_cap=99),
    lambda s: s.update(evaluation_shifts=101),
    lambda s: s.update(training_shifts=1),
    lambda s: s.update(training_updates=1),
    lambda s: s.update(optimizer_steps=1),
    lambda s: s['jobs'][0].update(target_training_update=1),
    lambda s: s['jobs'][0].update(checkpoint_update=12),
    lambda s: s['jobs'][0].update(role='train'),
    lambda s: s['jobs'][0].update(partition='train'),
    lambda s: s['jobs'][0].update(actor_seed=43),
    lambda s: s['jobs'][0].update(run_index=s['jobs'][1]['run_index']),
    lambda s: s['jobs'][0].update(experiment_seed=s['jobs'][1]['experiment_seed']),
    lambda s: s['jobs'][0].update(experiment_seed=s['jobs'][1]['actor_sampling_seed']),
])
def test_invalid_scope_rejected_before_database_creation(setup, change):
    with pytest.raises(ValueError):
        setup.register(setup.make(mutate=change))
    assert not setup.path.exists()


@pytest.mark.parametrize('field,value', [
    ('run_index', 1), ('recording_id', 'old'), ('source_recording_id', 'old'),
    ('experiment_seed', 2), ('experiment_seed', 3), ('experiment_seed', 42),
    ('actor_sampling_seed', 2), ('actor_sampling_seed', 3), ('actor_sampling_seed', 42),
])
def test_all_history_cross_type_collisions_rejected(setup, field, value):
    with pytest.raises(ValueError):
        setup.register(setup.make(mutate=lambda s: s['jobs'][0].update({field: value})))
    assert not setup.path.exists()


def test_budget_reference_change_rejected_before_creation(setup):
    spec = setup.make(mutate=lambda s: s.update(budget_state_before={}))
    with pytest.raises(ValueError, match='merged budget'):
        setup.register(spec)
    assert not setup.path.exists()


def test_insufficient_authorization_rejected_before_creation(setup):
    suite, sha, approval = setup.make()
    approval['max_new_shifts'] = 99
    with pytest.raises(ValueError, match='authorization'):
        setup.register((suite, sha, approval))
    assert not setup.path.exists()


def test_claim_complete_exactly_once_and_readback(setup, tmp_path):
    ledger = setup.register()
    job = ledger.suite['jobs'][0]
    artifact = tmp_path / 'dataset.manifest.json'
    artifact.write_text('{}\n')
    ref = budget.reference(artifact)
    receipt = dict(run_index=job['run_index'], experiment_seed=job['experiment_seed'],
                   normal_completion=True)
    with pytest.raises(ValueError, match='exactly one claim'):
        ledger.complete(job, ref, receipt)
    ledger.claim(job)
    with pytest.raises(ValueError, match='twice'):
        ledger.claim(job)
    ledger.complete(job, ref, receipt)
    assert ledger.job(job)['status'] == 'COMPLETE'
    assert ledger.job(job)['dataset_ref'] == ref
    with pytest.raises(ValueError, match='exactly one claim'):
        ledger.complete(job, ref, receipt)
    assert ledger.resume()['jobs'][str(job['run_index'])]['status'] == 'COMPLETE'


@pytest.mark.parametrize('change', [
    lambda r: r.update(run_index=-1), lambda r: r.update(experiment_seed=-1),
    lambda r: r.update(normal_completion=False),
])
def test_completion_requires_actual_matching_normal_receipt(setup, tmp_path, change):
    ledger = setup.register()
    job = ledger.suite['jobs'][0]
    artifact = tmp_path / 'data'
    artifact.write_text('sealed')
    receipt = dict(run_index=job['run_index'], experiment_seed=job['experiment_seed'],
                   normal_completion=True)
    change(receipt)
    ledger.claim(job)
    with pytest.raises(ValueError, match='identity invalid'):
        ledger.complete(job, budget.reference(artifact), receipt)
    assert ledger.job(job)['status'] == 'CLAIMED'


def test_unknown_and_claimed_never_auto_requeued(setup):
    ledger = setup.register()
    job = ledger.suite['jobs'][0]
    ledger.claim(job)
    with pytest.raises(ValueError, match='UNKNOWN/CLAIMED'):
        ledger.resume()
    ledger.unknown(job, 'connection interrupted')
    assert ledger.job(job)['status'] == 'UNKNOWN'
    with pytest.raises(ValueError, match='UNKNOWN/CLAIMED'):
        ledger.resume()
    with pytest.raises(ValueError, match='twice'):
        ledger.claim(job)
    with pytest.raises(ValueError, match='audited recovery'):
        ledger.rebind_zero_action_campaign()


def test_ppo_commit_always_forbidden(setup):
    with pytest.raises(ValueError, match='cannot commit PPO'):
        setup.register().commit(1, [], {}, 'any')


def test_checksum_corruption_detected(setup):
    ledger = setup.register()
    with budget.connection(setup.path) as con:
        con.execute("UPDATE slots SET payload='{}' WHERE run=?", (ledger.suite['jobs'][0]['run_index'],))
    with pytest.raises(ValueError, match='checksum'):
        ledger.snapshot()


def test_noncanonical_job_rejected(setup):
    ledger = setup.register()
    job = dict(ledger.suite['jobs'][0], experiment_seed=999)
    with pytest.raises(ValueError, match='noncanonical'):
        ledger.claim(job)


def test_registered_suite_mutation_detected(setup):
    ledger = setup.register()
    path = ledger.suite['suite_path']
    with open(path, 'a') as stream:
        stream.write(' ')
    with pytest.raises(ValueError, match='suite bytes'):
        budget.budget_snapshot(setup.path, test_only=True)


def test_production_path_rejected_for_temporary_mode(setup):
    suite, sha, approval = setup.make()
    with pytest.raises(ValueError, match='location differs'):
        budget.BudgetLedger(budget.GLOBAL_PATH, suite, campaign_sha256=sha,
            approval=approval, approval_sha256='c'*64, test_only=True)
