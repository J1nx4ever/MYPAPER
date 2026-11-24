"""Offline BG128 evaluation lifecycle tests; no listener or physical shift.

The 100-shift lifecycle uses analytical synthetic tapes and a fake collector.
The historical checkpoint test only loads its sealed bytes and never fits it.
"""
from copy import deepcopy
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from baseline1_v1.evaluation import training_state_fingerprint
from business_graph_summary_recovery_v1.plan_io import ROOT, reference, encoded
from business_graph_summary_v1.learner import JointCostPPO
from business_graph_summary_v1.test_learner import make_model, synthetic_batch

from . import metrics, runtime


def forbidden(*args, **kwargs):
    raise AssertionError('optimizer, physical collector, listener or repeated evaluation is forbidden')


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded(value))
    return reference(path)


@pytest.fixture
def trial(tmp_path, monkeypatch):
    model = make_model()
    model.update_index = 13  # Synthetic counter fixture, never an optimizer step.
    base = synthetic_batch(model)[0]
    episodes = []
    for index in range(100):
        episode = deepcopy(base)
        episode['identity'].update(run_index=9950001+index,
            environment_seed=10001+index, actor_sampling_seed=20001+index,
            split='validation', recording_id='synthetic-eval100')
        episodes.append(episode)
    output = tmp_path/'output'
    parent = {key: write(tmp_path/(key+'.json'), dict(synthetic=key))
              for key in ('checkpoint', 'manifest', 'seal')}
    cfg_path = tmp_path/'phase.json'
    cfg = dict(phase_id='s01_BG128_eval0100', jobs=[])
    phase_ref = write(cfg_path, cfg)
    selection = write(tmp_path/'selection.json', dict(synthetic=True, selected_update=13))
    suite = dict(campaign_id='synthetic-bg128-eval100', checkpoint_update=13,
        python_output=str(output), control_directory=str(output/'control'),
        parent_artifacts=parent, phase_config=phase_ref,
        master_plan=dict(sha256='a'*64), source_pins={}, jobs=[],
        selection=selection)
    suite_path = tmp_path/'suite.json'
    write(suite_path, suite)
    monkeypatch.setattr(runtime, '_load_model', lambda suite: model)
    monkeypatch.setattr(JointCostPPO, 'update', forbidden)
    monkeypatch.setattr(torch.optim.Adam, 'step', forbidden)
    monkeypatch.setattr(runtime, 'MaterialPhaseCollector', forbidden)
    monkeypatch.setattr(runtime, '_collect', lambda *args: (episodes, [], []))
    class Ledger:
        def resume(self):
            return dict(campaigns={suite['campaign_id']: dict(commits={})}, jobs={})
    return SimpleNamespace(model=model, suite=suite, cfg=cfg, path=suite_path,
        ledger=Ledger(), output=output, episodes=episodes)


def test_actual_bg128_numeric_eval100_is_readonly_and_preserves_failures(trial, monkeypatch):
    a = trial
    before_actor = runtime.digest(a.model.actor)
    before_critic = runtime.digest(a.model.critic)
    before_global = torch.get_rng_state().clone()
    result = runtime._trial(a.path, a.suite, a.cfg, a.ledger)
    assert result['physical_shifts'] == 100 and result['checkpoint_update'] == 13
    assert result['optimizer_updates'] == result['gradient_updates'] == result['training_updates'] == 0
    assert result['fingerprints'] == dict(actor=before_actor, critic=before_critic)
    assert torch.equal(before_global, torch.get_rng_state())
    assert result['parent_artifacts_before'] == result['parent_artifacts_after']
    assert result['selection_review'] == a.suite['selection']
    assert set(result['reports']) == {'summary', 'per_shift', 'per_window', 'readme', 'evaluation', 'receipt'}
    summary = json.loads(Path(result['reports']['summary']['path']).read_bytes())
    assert summary['n_shifts'] == 100 and summary['n_states'] == 300
    head = summary['headline_metrics']
    assert set(metrics.HEADLINE) <= set(head)
    assert head['grade_rmse_total_windows'] == 400 and head['empty_windows'] == 200
    assert head['joint_four_window_compliance_rate'] == 0
    assert head['complete_task_rate'] == 0
    assert head['mean_total_wait_seconds'] == 10
    assert summary['failed_shifts_retained'] is True
    with Path(result['reports']['per_shift']['path']).open() as stream:
        assert len(list(csv.DictReader(stream))) == 100
    with Path(result['reports']['per_window']['path']).open() as stream:
        assert len(list(csv.DictReader(stream))) == 400
    assert not list(a.output.glob('**/checkpoint.pt'))
    assert json.loads((a.output/'run_status.json').read_bytes())['status'] == 'COMPLETE'
    monkeypatch.setattr(runtime, '_collect', forbidden)
    monkeypatch.setattr(runtime, '_load_model', forbidden)
    assert runtime._trial(a.path, a.suite, a.cfg, a.ledger) == result


@pytest.mark.parametrize('mutation', ['critic_parameter', 'adam_config', 'private_rng'])
def test_sampling_or_scoring_mutation_is_detected_before_complete(trial, monkeypatch, mutation):
    a = trial
    def mutate(*args):
        if mutation == 'critic_parameter':
            with torch.no_grad():
                next(a.model.critic.parameters()).add_(.001)
        elif mutation == 'adam_config':
            a.model.critic_optimizer.param_groups[0]['lr'] *= 2
        else:
            torch.rand(1, generator=a.model.generator)
        return a.episodes, [], []
    monkeypatch.setattr(runtime, '_collect', mutate)
    with pytest.raises(ValueError, match='changed protected training'):
        runtime._trial(a.path, a.suite, a.cfg, a.ledger)
    assert not (a.output/'suite_complete.json').exists()


def test_selection_evidence_and_checkpoint_bytes_cannot_change(trial):
    a = trial
    Path(a.suite['selection']['path']).write_text('changed selection')
    with pytest.raises(ValueError, match='selection evidence changed'):
        runtime._trial(a.path, a.suite, a.cfg, a.ledger)
    assert not a.output.exists()


def test_ready_binds_fixed_checkpoint_and_exact_claim_without_collection(tmp_path, monkeypatch):
    from business_graph_summary_recovery_v1.plan_io import load_suite
    original = ROOT/'change_plans/token_attention_001/business_graph_summary_009/formal_training_001/prepared_001/suite.json'
    _, phases = load_suite(original)
    job = deepcopy(phases[0]['jobs'][0])
    job.update(run_index=9950001, role='final', partition='validation')
    cfg = write(tmp_path/'phase.json', dict(synthetic=True))
    parent = {key: write(tmp_path/(key+'.json'), dict(synthetic=key))
              for key in ('checkpoint', 'manifest', 'seal')}
    suite = dict(campaign_id='synthetic100', control_directory=str(tmp_path/'control'),
        source_pins={}, parent_artifacts=parent, phase_config=cfg,
        selection=write(tmp_path/'selection.json', dict(selected_update=13)),
        master_plan=dict(sha256='f'*64))
    calls = []
    monkeypatch.setattr(runtime, 'check_sources', lambda pins: calls.append(pins))
    ledger = SimpleNamespace(snapshot=lambda: dict(jobs={str(job['run_index']):
        dict(status='CLAIMED', identity=job)}))
    runtime._publisher(suite, ledger)(job, cfg['path'])
    ready = (Path(suite['control_directory'])/'ready/run_9950001.properties').read_text()
    assert 'run_index=9950001\n' in ready
    assert 'config_sha256='+cfg['sha256']+'\n' in ready
    assert 'all_comparison_checkpoints_frozen=true\n' in ready
    assert calls == [{}]


def test_sealed_real_selected_checkpoint_load_preserves_all_original_artifacts(tmp_path):
    folder = ROOT/'runs/business-graph-summary-formal-seed1-20261118-001/arms/s01_BG128/updates/0013'
    checkpoint = folder/'checkpoint.pt'
    parent = {name: reference(path) for name, path in (
        ('checkpoint', checkpoint), ('manifest', Path(str(checkpoint)+'.manifest.json')),
        ('seal', Path(str(checkpoint)+'.seal.json')))}
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    suite = dict(parent_artifacts=parent, learner_config=saved['config'], checkpoint_update=13,
        selection=write(tmp_path/'selection.json', dict(selected_update=13)))
    global_before = torch.get_rng_state().clone()
    model = runtime._load_model(suite)
    assert model.update_index == 13 and model.config['candidate_id'] == 's01_BG128'
    assert model.config['reward_contract']['wait_weight'] == .2
    assert torch.equal(global_before, torch.get_rng_state())
    assert runtime._parent_artifacts(suite) == parent
    before = training_state_fingerprint(model)
    assert runtime._portable_state_fingerprint(model)
    assert training_state_fingerprint(model) == before
