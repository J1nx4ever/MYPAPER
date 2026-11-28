"""Run one step-matched update on the sealed Full S02 u0 batch, entirely offline."""
import json
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import torch
from torch.optim.optimizer import register_optimizer_step_post_hook

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT))
from change_plans.token_attention_001.business_graph_summary_009.reinforce_comparison_001 import prepare as previous
from change_plans.token_attention_001.business_graph_summary_009.reinforce_comparison_001.ppo_steps_matched_001 import runtime
from business_graph_summary_campaign_v1.plan_io import load_reference, reference, write

REPORT = ROOT / 'reports' / HERE.relative_to(ROOT / 'change_plans') / 'real_first_batch_replay.json'


class ArchivedBatch:
    """The collector can only read the original COMPLETE rows from this fixture."""
    def __init__(self, cfg, phase, intent):
        self.rows = {}
        self.committed = None
        self.attempts = 0
        for job, run, dataset in zip(cfg['jobs'], intent['run_indices'], intent['sources'], strict=True):
            receipt = Path(job['python_output']) / 'episodes' / f'run_{run}' / 'material_receipt.json'
            data = load_reference(reference(receipt))
            assert job['run_index'] == run and data['phase_config'] == phase and data['normal_completion']
            assert data['four_windows'] == 4 and data['references']['episode_manifest'] == dataset
            self.rows[str(run)] = dict(status='COMPLETE', identity=job, receipt=data, dataset_ref=dataset)

    def resume(self):
        return dict(jobs=self.rows)

    def commit_record(self, update):
        assert update == 1
        return self.committed

    def commit(self, update, jobs, checkpoint, batch_id):
        self.attempts += 1
        assert self.committed is None and update == 1
        if self.attempts == 1:
            raise InterruptedError('offline interruption after checkpoint sealing, before ledger commit')
        self.committed = dict(checkpoint_ref=checkpoint, batch_id=batch_id)
        for job in jobs:
            self.rows[str(job['run_index'])]['committed_update'] = update


def main():
    torch.set_num_threads(2)
    source, phases, completion, manifest, seal = previous.historical()
    cfg = phases[0]
    phase = source['phase_configs'][0]
    historical_stats = []
    for checkpoint in completion['training_checkpoints']:
        path = Path(checkpoint['path']).with_name('training_summary.json')
        historical_stats.append(load_reference(reference(path))['update'])
    counts = [stats['actor_optimizer_steps'] for stats in historical_stats]
    assert len(counts) == 32 and counts[0] == 732 and sum(counts) == 23612
    source_folder = Path(completion['training_checkpoints'][0]['path']).parent
    intent = load_reference(reference(source_folder / 'intent.json'))
    archive = ArchivedBatch(cfg, phase, intent)
    counts_seen = []
    def count_step(optimizer, _args, _kwargs):
        assert isinstance(optimizer, torch.optim.Adam)
        counts_seen.append(id(optimizer))

    with TemporaryDirectory(prefix='matched-reinforce-u1-') as scratch:
        folder = Path(scratch)
        arm_cfg = source['arms'][0]['learner_config']
        suite = dict(campaign_id='offline-first-batch', python_output=str(folder),
                     arms=[dict(learner_config=arm_cfg, python_output=str(folder / 'arm'))],
                     initial_actor=previous.INITIAL_ACTOR, initial_actor_digest=previous.INITIAL_DIGEST,
                     initial_actor_manifest=manifest, initial_actor_seal=seal,
                     reinforce=dict(episodes_per_batch=32, actor_lr=3e-4, adam_eps=1e-5,
                                    baseline_decay=.5, actor_steps_by_update=counts))
        suite_file = folder / 'suite.json'
        write(suite_file, suite)
        campaign = reference(suite_file)
        initial = runtime._initial(suite, dict(suite=campaign, actor_digest=previous.INITIAL_DIGEST))
        actor_ref = reference(Path(cfg['python_output']) / 'actor.pt')
        assert len(archive.rows) == 32 and all(
            row['receipt']['identity']['policy_sha256'] == actor_ref['sha256'] for row in archive.rows.values())
        with register_optimizer_step_post_hook(count_step):
            try:
                runtime._train(suite, cfg, phase, campaign, initial, archive)
            except InterruptedError:
                assert archive.attempts == 1 and archive.committed is None and len(counts_seen) == 732
            else:
                raise AssertionError('commit-interruption hook was not reached')
            recovered = runtime._train(suite, cfg, phase, campaign, initial, archive)
            assert runtime._train(suite, cfg, phase, campaign, initial, archive) == recovered
        assert len(counts_seen) == 732 and len(set(counts_seen)) == 1 and archive.attempts == 2
        checkpoint = recovered['checkpoint']
        model = runtime._load(suite, campaign, checkpoint, 1)
        exported = model.save_actor(folder / 'actor_after_732.pt')
        assert exported['actor_digest'] == model.policy_version == recovered['actor_digest'] != previous.INITIAL_DIGEST
        metrics = load_reference(recovered['training_summary'])['reinforce']
        assert metrics['rollout_actor_digest'] == previous.INITIAL_DIGEST
        assert metrics['actor_optimizer_steps'] == metrics['cumulative_actor_optimizer_steps'] == counts[0]
        assert metrics['first_step_max_abs_log_ratio'] < 1e-4 and metrics['post_first_step_max_abs_log_ratio'] > 1e-4
        assert metrics['transitions'] == historical_stats[0]['transitions'] and metrics['episodes'] == 32
        assert math.isclose(metrics['mean_combined_cost'], historical_stats[0]['metrics']['combined_cost_mean'], rel_tol=1e-8)
        assert model.baseline == metrics['baseline_after'] and metrics['baseline_before'] == 0

    report = dict(schema='matched-reinforce-offline-transaction-check-v1', fixture='archived Full PPO S02 first batch',
                  source_suite=completion['suite'], source_update=reference(source_folder / 'training_summary.json'),
                  archive_intent=reference(source_folder / 'intent.json'), rollout_actor=actor_ref,
                  source_shifts=len(archive.rows), historical_ppo_actor_steps=counts[0],
                  observed_adam_step_calls=len(counts_seen), recovery_extra_optimizer_steps=0,
                  replayed_commit_matches=True, restored_actor_exported=True, new_physical_shifts=0,
                  new_training_campaign_registrations=0, metrics=metrics,
                  protected_ppo_sha256=runtime.original.sha(runtime.original.CORE))
    assert report['protected_ppo_sha256'] == runtime.original.PROTECTED_SHA
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'metrics'}, allow_nan=False))


if __name__ == '__main__':
    main()
