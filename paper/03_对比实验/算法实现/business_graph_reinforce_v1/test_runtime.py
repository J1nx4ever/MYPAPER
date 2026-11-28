"""Exercise the training transaction and recovery with no AnyLogic process."""

from contextlib import nullcontext
from pathlib import Path

import pytest
import torch

from business_graph_summary_campaign_v1.plan_io import reference, write
from business_graph_summary_v1.learner import default_config
from business_graph_summary_v1.test_learner import synthetic_batch

from .collection import complete_sources
from .learner import FullShiftReinforce
from . import runtime


class _Ledger:
    def __init__(self):
        self.rows = {}
        self.committed = None
        self.interrupted = False

    def resume(self):
        return dict(jobs=self.rows)

    def commit_record(self, update):
        assert update == 1
        return self.committed

    def commit(self, update, jobs, checkpoint, batch_id):
        if not self.interrupted:
            self.interrupted = True
            raise RuntimeError('interrupted before ledger commit')
        assert self.committed is None
        assert update == 1
        self.committed = dict(checkpoint_ref=checkpoint, batch_id=batch_id)
        for job in jobs:
            self.rows[str(job['run_index'])]['committed_update'] = update


def test_sealed_actor_update_recovers_after_interrupted_ledger_commit(tmp_path, monkeypatch):
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        _verify_recovery(tmp_path, monkeypatch)
    finally:
        torch.set_num_threads(threads)


def _verify_recovery(tmp_path, monkeypatch):
    actor_config = default_config(source_kind='synthetic', execution_profile='synthetic_test_v1')
    model = FullShiftReinforce(actor_config=actor_config)
    settings = {key: value for key, value in model.settings.items() if key != 'schema'}
    campaign_file = tmp_path / 'suite.json'
    write(campaign_file, dict(training_updates=1))
    campaign = reference(campaign_file)
    model.runtime_state.update(campaign=campaign, initial_actor=dict(source='synthetic_u0'))
    path = tmp_path / 'initial.pt'
    model.save_checkpoint(path)
    parent = reference(path)
    suite = dict(initial_actor=model.runtime_state['initial_actor'], reinforce=settings,
                 campaign_id='test-transaction', python_output=str(tmp_path),
                 arms=[dict(python_output=str(tmp_path), learner_config=actor_config)])
    jobs = [dict(run_index=run, scene_id='nominal') for run in (1, 2)]
    cfg = dict(arm_update=1, phase_id='u01', jobs=jobs)
    phase_file = tmp_path / 'phase.json'
    write(phase_file, cfg)
    phase = reference(phase_file)
    ledger = _Ledger()
    collection_count = []

    def collect(_suite, phase_cfg, phase_ref, _checkpoint, state, learner):
        collection_count.append(1)
        actor = learner.save_actor(tmp_path / 'rollout_actor.pt')
        actor_ref = reference(actor['path'])
        for job in phase_cfg['jobs']:
            run = job['run_index']
            folder = tmp_path / str(run)
            for name in ('manifest.json', 'episode.h5.manifest.json'):
                write(folder / name, dict(run_index=run))
            dataset = reference(folder / 'episode.h5.manifest.json')
            receipt = dict(phase_config=phase_ref,
                           references=dict(actor=actor_ref, episode_manifest=dataset,
                                           raw_manifest=reference(folder / 'manifest.json')))
            write(folder / 'material_receipt.json', receipt)
            state.rows[str(run)] = dict(status='COMPLETE', identity=job, receipt=receipt, dataset_ref=dataset)
        return synthetic_batch(learner), complete_sources(phase_cfg, phase_ref, state), actor

    monkeypatch.setattr(runtime, 'collect_phase', collect)
    monkeypatch.setattr(runtime.mixed, '_exact_verification', lambda jobs: nullcontext())
    with pytest.raises(RuntimeError, match='interrupted before ledger commit'):
        runtime._train(suite, cfg, phase, campaign, parent, ledger)
    prepared = runtime._train(suite, cfg, phase, campaign, parent, ledger)
    assert runtime._train(suite, cfg, phase, campaign, parent, ledger) == prepared
    assert len(collection_count) == 1
    assert ledger.committed['checkpoint_ref'] == prepared['checkpoint']
    assert runtime._load(suite, campaign, prepared['checkpoint'], 1).update_index == 1

    receipt_path = Path(prepared['intent']['sources'][0]['receipt']['path'])
    receipt_path.write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='physical ledger and durable material receipt differ'):
        runtime._train(suite, cfg, phase, campaign, parent, ledger)
