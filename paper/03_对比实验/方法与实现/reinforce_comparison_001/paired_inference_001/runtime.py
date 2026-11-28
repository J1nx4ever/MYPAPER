"""Run seven full-shift evaluations under one frozen REINFORCE S02 Actor."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from baseline1_v1.evaluation import ReadOnlyTrainingGuard
from business_graph_reinforce_v1 import runtime as training
from business_graph_reinforce_v1.collection import PerJobMaterialCollector, bg128_storage, mixed
from business_graph_summary_campaign_v1.gate import check_phase
from business_graph_summary_campaign_v1.plan_io import load_phase, load_reference, reference
from business_graph_summary_campaign_v1.progress import write_progress
from business_graph_summary_campaign_v1.runtime import _same_or_save
from business_graph_summary_v1.contract import require
from business_graph_summary_v1.models import digest
from online_v1_1.operations import _publish


def _frozen_policy(suite):
    completed = load_reference(suite['training_completion'])
    original = load_reference(completed['suite'])
    checkpoint, actor_ref = suite['frozen_checkpoint'], suite['frozen_actor']
    saved_actor = completed['final_actor']
    require(completed['status'] == 'COMPLETE' and completed['schema'] == training.SCHEMA + '-completion'
            and completed['training_updates'] == saved_actor['update_index'] == 32
            and completed['training_shifts'] == 1024 and completed['evaluation_shifts'] == 0
            and original['baseline_seed'] == 's02' and completed['final_checkpoint'] == checkpoint
            and reference(saved_actor['path']) == {k: saved_actor[k] for k in ('path', 'sha256')} == actor_ref,
            'expected the sealed final Actor of the completed S02 REINFORCE training')
    trained = training._load(original, completed['suite'], checkpoint, 32)
    binding = trained.bind_actor_snapshot(actor_ref['path'])
    actor, fingerprint = mixed.frozen_actor(actor_ref['path'])
    require(binding['actor_digest'] == trained.policy_version == saved_actor['actor_digest'] == fingerprint,
            'published Actor weights differ from the trained u32 checkpoint')
    actor.eval().requires_grad_(False)
    return SimpleNamespace(actor=actor, policy_version=fingerprint)


def verify(suite, configs):
    mixed._check_sources(suite)
    policy = _frozen_policy(suite)
    require(len(configs) == len(suite['phase_configs']) == 7 and
            len({cfg['scenario_id'] for cfg in configs}) == 7 and
            suite['training_shifts'] == suite['training_updates'] == 0 and suite['evaluation_shifts'] == 700,
            'expected seven evaluation cases for one frozen Actor')
    for cfg, ref in zip(configs, suite['phase_configs'], strict=True):
        mixed._require_reference(ref)
        require(load_phase(ref['path']) == cfg and cfg['campaign_id'] == suite['campaign_id'],
                'evaluation phase differs from the sealed campaign')
        check_phase(cfg)
        mixed._check_actor_phase(policy.actor, cfg)
        require(cfg['policy_arm'] == 'new' and len(cfg['jobs']) == 100 and
                all(job['role'] == 'final' and job['partition'] == 'validation'
                    and job['checkpoint_update'] == 32 for job in cfg['jobs']),
                'frozen Actor evaluation needs 100 independent complete shifts per case')
    require([job for cfg in configs for job in cfg['jobs']] == suite['jobs'], 'physical roster differs')
    return dict(evaluation_shifts=700, scenarios=7, training_shifts=0, training_updates=0,
                actor_digest=policy.policy_version, checkpoint=suite['frozen_checkpoint'],
                actor_artifact=suite['frozen_actor'])


def _evaluate(suite, cfg, phase_ref, ledger, policy):
    checkpoint, actor_ref = suite['frozen_checkpoint'], suite['frozen_actor']
    with (mixed._exact_verification(cfg['jobs']), bg128_storage(),
          patch.object(mixed, 'MaterialPhaseCollector', PerJobMaterialCollector),
          ReadOnlyTrainingGuard(), torch.no_grad()):
        episodes, sources, actor = mixed._collect(suite, cfg, phase_ref, checkpoint, ledger, policy,
                                                  actor_ref=actor_ref)
        write_progress(suite, status='EVALUATING', phase_id=cfg['phase_id'])
        # Physical denominators and joint costs come from the same evaluator as Full S02.
        # Its optional model argument requests Critic predictions, which REINFORCE does not have.
        report = mixed.evaluate_episodes(episodes, expected_complete_shifts=len(cfg['jobs']),
                                         checkpoint_provenance=checkpoint)
        evidence = mixed._evaluation_sources(episodes, cfg['jobs'])
    require(reference(actor['path']) == actor_ref and digest(policy.actor) == policy.policy_version
            and report['evaluation_optimizer_updates'] == 0 and report['critic_mc'] is None,
            'frozen policy or read-only scoring changed')
    report.update(policy_arm=cfg['policy_arm'], scenario_id=cfg['scenario_id'], phase_config=phase_ref,
                  actor_artifact=actor_ref, source_refs=sources, scenario_evidence=evidence,
                  fingerprints=dict(actor=policy.policy_version, critic=None))
    for record in report['per_episode']:
        record['critic_mc'] = dict(mse=None, explained_variance=None)
    output = Path(cfg['python_output'])
    _same_or_save(output / 'evaluation_complete.json', report)
    summary = mixed.build_report(report, expected_complete_shifts=len(cfg['jobs']))
    summary.update(policy_arm=cfg['policy_arm'], scenario_id=cfg['scenario_id'],
                   scenario_evidence=evidence, critic_primary_weighting=None)
    published = mixed.write_reports(summary, report, output / 'reports')
    return dict(checkpoint=checkpoint, actor_artifact=actor_ref, reports=published,
                report=reference(output / 'evaluation_complete.json'))


def run(suite, configs, ledger):
    from .paired_metrics import write_comparison

    verify(suite, configs)
    torch.set_num_threads(suite['cpu_threads'])
    policy = _frozen_policy(suite)
    stop = Path(suite['control_directory']) / 'STOP.properties'
    require(not stop.exists(), 'preserved STOP requires reconciliation')
    try:
        results = []
        for cfg, phase_ref in zip(configs, suite['phase_configs'], strict=True):
            require(not stop.exists(), 'preserved STOP requires reconciliation')
            results.append(_evaluate(suite, cfg, phase_ref, ledger, policy))
        state = ledger.resume()
        require(not state['campaigns'][suite['campaign_id']]['commits']
                and all(state['jobs'][str(job['run_index'])]['status'] == 'COMPLETE'
                        and 'committed_update' not in state['jobs'][str(job['run_index'])] for job in suite['jobs']),
                'evaluation requires 700 complete runs and zero training commits')
        comparison = write_comparison(suite, configs, results)
        mixed._check_sources(suite)
        require(reference(suite['frozen_actor']['path']) == suite['frozen_actor'] and
                reference(suite['frozen_checkpoint']['path']) == suite['frozen_checkpoint'],
                'training artifacts changed during independent inference')
        result = dict(status='COMPLETE', suite=reference(suite['suite_path']), phase_results=results,
                      checkpoint=suite['frozen_checkpoint'], actor_artifact=suite['frozen_actor'],
                      training_shifts=0, training_updates=0, evaluation_shifts=len(suite['jobs']),
                      evaluation_optimizer_updates=0, comparison=comparison)
        _same_or_save(Path(suite['python_output']) / 'suite_complete.json', result)
        write_progress(suite, status='COMPLETE', completed_shifts=len(suite['jobs']), training_updates=0)
        return result
    except BaseException as error:
        if not stop.exists():
            stop.parent.mkdir(parents=True, exist_ok=True)
            _publish(stop, b'status=STOP\nreason=REINFORCE_INFERENCE_ERROR\n')
        write_progress(suite, status='STOP', error=repr(error))
        raise
