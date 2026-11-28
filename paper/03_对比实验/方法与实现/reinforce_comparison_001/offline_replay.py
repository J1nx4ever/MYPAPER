"""Exercise one REINFORCE update on the sealed PPO comparator's first batch.

The original 32 trajectories were sampled on-policy by the common u0 Actor.
They are validation fixtures only. The production controller collects fresh runs.
"""

import json
import math
import sys
from pathlib import Path
from statistics import fmean
from tempfile import TemporaryDirectory

PLAN = Path(__file__).resolve().parent
ROOT = PLAN.parents[3]
MIXED = PLAN.parent / 'generalization_001/grade_assignment_generalization_001/continuous_training_001'
sys.path[:0] = [str(ROOT), str(MIXED)]

import torch
import mixed_runtime as mixed
from recovery_002.storage_scope import bg128_storage
from business_graph_summary_campaign_v1.plan_io import load_reference, reference
from business_graph_summary_v1.targets import composite_contract, grade_cost

SOURCE = ROOT / 'runs/bg128-continuous-mixed-20261120-s02/arms/s01_BG128'
REPORT = ROOT / 'reports/token_attention_001/business_graph_summary_009' / PLAN.name


def archived_batch():
    intent = load_reference(reference(SOURCE / 'updates/0001/intent.json'))
    shifts, jobs = [], []
    for run, source in zip(intent['run_indices'], intent['sources'], strict=True):
        manifest = Path(source['path'])
        receipt = manifest.parents[1] / 'episodes' / f'run_{run}' / 'material_receipt.json'
        data = json.loads(receipt.read_bytes())
        assert source == data['references']['episode_manifest'] == reference(manifest)
        phase = load_reference(data['phase_config'])
        job, = (job for job in phase['jobs'] if job['run_index'] == run)
        jobs.append(job)
        shifts.append(data)
    return jobs, shifts


def main():
    jobs, receipts = archived_batch()
    torch.set_num_threads(2)
    policy_file = SOURCE / 'train_u0001/actor.pt'
    actor, initial_digest = mixed.frozen_actor(policy_file)
    physical = composite_contract()
    assert initial_digest == '2e3685082d4bbf607153062c7da0e820aef94ae168ec5cbf88cadd0165b4ed32'
    with bg128_storage(), mixed._exact_verification(jobs):
        episodes = []
        rewards = []
        for receipt in receipts:
            episode = mixed.load_admitted_episode(receipt)
            evaluation = episode['evaluation']
            ore, waste, wait = (evaluation[key] for key in ('ore_tons', 'waste_tons', 'total_wait_seconds'))
            grades = [grade_cost(s.grade if s.grade_valid else None, empty=not s.grade_valid)
                      for s in episode['memory'].settlements]
            assert len(grades) == 4 and math.isclose(sum(episode['wait_deltas']), wait, abs_tol=1e-7)
            cost = (math.fsum(grades) + physical['ore_shortfall_weight'] * max(0, 1 - ore / physical['ore_target_tons'])
                    + physical['waste_shortfall_weight'] * max(0, 1 - waste / physical['waste_target_tons'])
                    + physical['wait_weight'] * wait / physical['wait_reference_seconds'])
            assert math.isclose(cost, math.fsum(episode['targets'].step_costs), abs_tol=1e-9)
            episodes.append(episode)
            rewards.append(-cost)

        from business_graph_reinforce_v1.learner import FullShiftReinforce
        learner = FullShiftReinforce(actor, actor_config=actor._artifact_config)
        learner.bind_actor_snapshot(policy_file)
        before = learner.policy_version
        result = learner.update(episodes, batch_id='offline-acceptance-u0')
        after = learner.policy_version
        assert learner.update_index == 1 and before == initial_digest and before != after
        with TemporaryDirectory(prefix='bg128-reinforce-replay-') as folder:
            checkpoint = Path(folder) / 'offline.pt'
            learner.save_checkpoint(checkpoint)
            restored = FullShiftReinforce.from_checkpoint(checkpoint, expected_sha256=reference(checkpoint)['sha256'])
            assert restored.policy_version == after and restored.update_index == 1
    expected_baseline = 0.5 * fmean(rewards)
    expected_loss = -fmean(reward * math.fsum(episode['old_log_probs'])
                           for reward, episode in zip(rewards, episodes, strict=True))
    assert result['baseline_before'] == 0 and result['actor_optimizer_steps'] == 1
    assert math.isclose(result['baseline_after'], expected_baseline, abs_tol=1e-10)
    assert math.isclose(result['policy_loss'], expected_loss, rel_tol=1e-6, abs_tol=1e-6)
    report = dict(source=reference(SOURCE / 'updates/0001/intent.json'), actor=reference(policy_file),
                  status='OFFLINE_UPDATE_VERIFIED', comparator='Full mixed S02', historical_shifts=len(episodes),
                  new_physical_shifts=0, optimizer_steps=1, restored_checkpoint=True,
                  decisions=sum(len(ep['actions']) for ep in episodes),
                  actor_digest_before=before, actor_digest_after=after,
                  independently_expected_loss=expected_loss,
                  independently_expected_baseline=expected_baseline,
                  learner_metrics=result)
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / 'offline_replay_s02.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, allow_nan=False))


if __name__ == '__main__':
    main()
