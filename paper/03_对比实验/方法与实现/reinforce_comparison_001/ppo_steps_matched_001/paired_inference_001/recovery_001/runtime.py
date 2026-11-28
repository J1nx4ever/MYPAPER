"""Finish the interrupted S02 u32 evaluation with its 42 sealed prefix shifts."""

from pathlib import Path
from unittest.mock import patch

import torch

from baseline1_v1.evaluation import ReadOnlyTrainingGuard
from business_graph_summary_campaign_v1.gate import check_phase
from business_graph_summary_campaign_v1.plan_io import load_phase, reference
from business_graph_summary_campaign_v1.progress import write_progress
from business_graph_summary_campaign_v1.runtime import _same_or_save
from business_graph_summary_v1.contract import require
from business_graph_summary_v1.models import digest
from online_v1_1.operations import _publish

from .. import budget as previous_budget, prepare as previous_plan, runtime as previous_runtime
from ....paired_inference_001.paired_metrics import write_comparison


mixed = previous_runtime.prior.mixed
bg128_storage = previous_runtime.prior.bg128_storage
PerJobMaterialCollector = previous_runtime.prior.PerJobMaterialCollector


def _prefix(suite, *, require_closed):
    previous, phases = previous_plan.load_package()
    require(reference(previous['suite_path']) == suite['previous_evaluation'] and
            previous['frozen_checkpoint'] == suite['frozen_checkpoint'] and
            previous['frozen_actor'] == suite['frozen_actor'], 'previous evaluation identity differs')
    ledger = previous_budget.InferenceLedger(previous, phases)
    snapshot = ledger.snapshot()
    rows = snapshot['jobs']
    count = suite['previous_completed_count']
    prefix = phases[0]['jobs'][:count]
    stopped = Path(previous['control_directory']) / 'STOP.properties'
    interrupted = rows[str(suite['interrupted_run'])]
    require(count == len(prefix) == 42 and
            suite['interrupted_run'] == phases[0]['jobs'][count]['run_index'] == 10096043 and
            (interrupted['status'] == 'UNKNOWN' or not require_closed and interrupted['status'] == 'CLAIMED') and
            (stopped.is_file() or not require_closed) and
            not snapshot['campaigns'][previous['campaign_id']]['commits'], 'previous evaluation has not closed safely')
    for job in prefix:
        row = rows[str(job['run_index'])]
        require(row['status'] == 'COMPLETE' and row['identity'] == job and
                row['receipt']['phase_config'] == previous['phase_configs'][0] and
                row['receipt']['identity']['policy_sha256'] == suite['frozen_actor']['sha256'],
                'earlier nominal evaluation evidence differs')
        mixed._require_reference(row['dataset_ref'])
    require(all(rows[str(job['run_index'])]['status'] == 'PLANNED'
                for job in previous['jobs'][count + 1:]), 'another old slot has been consumed')
    return phases, prefix, rows


def verify(suite, configs):
    mixed._check_sources(suite)
    policy = previous_runtime._frozen_policy(suite)
    require(len(configs) == len(suite['phase_configs']) == 7 and
            len({cfg['scenario_id'] for cfg in configs}) == 7 and
            suite['training_shifts'] == suite['training_updates'] == 0 and
            suite['evaluation_shifts'] == 658 and suite['comparison_shifts'] == 700,
            'recovery needs seven cases and exactly 658 new physical runs')
    original, prefix, _ = _prefix(suite, require_closed=False)
    old_runs = {job['run_index'] for phase in original for job in phase['jobs']}
    for index, (cfg, ref, prior) in enumerate(zip(configs, suite['phase_configs'], original, strict=True)):
        mixed._require_reference(ref)
        require(load_phase(ref['path']) == cfg and cfg['campaign_id'] == suite['campaign_id'] and
                cfg['scenario_id'] == prior['scenario_id'], 'new frozen phase differs')
        check_phase(cfg)
        mixed._check_actor_phase(policy.actor, cfg)
        wanted = prior['jobs'][len(prefix):] if index == 0 else prior['jobs']
        require(cfg['policy_arm'] == 'new' and len(cfg['jobs']) == len(wanted) and
                all(all(new[key] == old[key] for key in previous_plan.PAIR_KEYS) and
                    new['run_index'] not in old_runs and new['recording_id'] != old['recording_id']
                    for new, old in zip(cfg['jobs'], wanted, strict=True)),
                'new physical identities or matched evaluation slots differ')
    require([job for cfg in configs for job in cfg['jobs']] == suite['jobs'] and
            len({job['run_index'] for job in suite['jobs']}) == 658, 'physical roster differs')
    return dict(evaluation_shifts=658, comparison_shifts=700, reused_completed_shifts=42,
                scenarios=7, training_shifts=0, training_updates=0, actor_digest=policy.policy_version,
                checkpoint=suite['frozen_checkpoint'], actor_artifact=suite['frozen_actor'])


def _evaluate(suite, cfg, phase_ref, ledger, policy, preserved_jobs, preserved_rows):
    checkpoint, actor_ref = suite['frozen_checkpoint'], suite['frozen_actor']
    jobs = [*preserved_jobs, *cfg['jobs']]
    with (mixed._exact_verification(jobs), bg128_storage(),
          patch.object(mixed, 'MaterialPhaseCollector', PerJobMaterialCollector),
          ReadOnlyTrainingGuard(), torch.no_grad()):
        fresh, sources, actor = mixed._collect(suite, cfg, phase_ref, checkpoint, ledger, policy,
                                              actor_ref=actor_ref)
        write_progress(suite, status='EVALUATING', phase_id=cfg['phase_id'])
        archived, archive_sources = [], []
        for job in preserved_jobs:
            row = preserved_rows[str(job['run_index'])]
            episode = mixed.load_admitted_episode(row['receipt'])
            expected = dict(run_index=job['run_index'], scene_id=job['scene_id'], scene_group=job['scene_group'],
                            split=job['partition'], recording_id=job['recording_id'],
                            policy_sha256=actor_ref['sha256'], policy_version=policy.policy_version,
                            environment_seed=job['experiment_seed'], actor_sampling_seed=job['actor_sampling_seed'])
            require(all(episode['identity'][key] == value for key, value in expected.items()),
                    'preserved physical slot does not bind the frozen Actor')
            archived.append(episode)
            archive_sources.append(mixed._require_reference(row['dataset_ref']))
        episodes = [*archived, *fresh]
        require(len(episodes) == 100, 'combined case must contain 100 evaluations')
        report = mixed.evaluate_episodes(episodes, expected_complete_shifts=len(jobs),
                                         checkpoint_provenance=checkpoint)
        evidence = mixed._evaluation_sources(episodes, jobs)
    require(reference(actor['path']) == actor_ref and digest(policy.actor) == policy.policy_version
            and report['evaluation_optimizer_updates'] == 0 and report['critic_mc'] is None,
            'frozen policy or read-only scoring changed')
    report.update(policy_arm=cfg['policy_arm'], scenario_id=cfg['scenario_id'], phase_config=phase_ref,
                  actor_artifact=actor_ref, source_refs=[*archive_sources, *sources], scenario_evidence=evidence,
                  recovered_prefix_runs=[job['run_index'] for job in preserved_jobs],
                  fingerprints=dict(actor=policy.policy_version, critic=None))
    for record in report['per_episode']:
        record['critic_mc'] = dict(mse=None, explained_variance=None)
    output = Path(cfg['python_output'])
    _same_or_save(output / 'evaluation_complete.json', report)
    summary = mixed.build_report(report, expected_complete_shifts=len(jobs))
    summary.update(policy_arm=cfg['policy_arm'], scenario_id=cfg['scenario_id'],
                   scenario_evidence=evidence, critic_primary_weighting=None)
    published = mixed.write_reports(summary, report, output / 'reports')
    return dict(checkpoint=checkpoint, actor_artifact=actor_ref, reports=published,
                report=reference(output / 'evaluation_complete.json'))


def run(suite, configs, ledger):
    verify(suite, configs)
    torch.set_num_threads(suite['cpu_threads'])
    policy = previous_runtime._frozen_policy(suite)
    _, preserved, rows = _prefix(suite, require_closed=True)
    stop = Path(suite['control_directory']) / 'STOP.properties'
    require(not stop.exists(), 'preserved STOP requires reconciliation')
    try:
        results = []
        for index, (cfg, phase_ref) in enumerate(zip(configs, suite['phase_configs'], strict=True)):
            require(not stop.exists(), 'preserved STOP requires reconciliation')
            results.append(_evaluate(suite, cfg, phase_ref, ledger, policy, preserved if index == 0 else [], rows))
        state = ledger.resume()
        require(not state['campaigns'][suite['campaign_id']]['commits'] and
                all(state['jobs'][str(job['run_index'])]['status'] == 'COMPLETE' and
                    'committed_update' not in state['jobs'][str(job['run_index'])] for job in suite['jobs']),
                'recovery needs 658 complete new shifts and zero optimizer updates')
        combined = [dict(configs[0], jobs=[*preserved, *configs[0]['jobs']]), *configs[1:]]
        comparison = write_comparison(suite, combined, results)
        mixed._check_sources(suite)
        require(reference(suite['frozen_actor']['path']) == suite['frozen_actor'] and
                reference(suite['frozen_checkpoint']['path']) == suite['frozen_checkpoint'],
                'training artifacts changed during independent inference')
        result = dict(status='COMPLETE', suite=reference(suite['suite_path']), phase_results=results,
                      checkpoint=suite['frozen_checkpoint'], actor_artifact=suite['frozen_actor'],
                      previous_evaluation=suite['previous_evaluation'], interrupted_run=suite['interrupted_run'],
                      reused_completed_shifts=42, training_shifts=0, training_updates=0, evaluation_shifts=658,
                      comparison_shifts=700, evaluation_optimizer_updates=0, comparison=comparison)
        _same_or_save(Path(suite['python_output']) / 'suite_complete.json', result)
        write_progress(suite, status='COMPLETE', completed_shifts=658, comparison_shifts=700, training_updates=0)
        return result
    except BaseException as error:
        if not stop.exists():
            stop.parent.mkdir(parents=True, exist_ok=True)
            _publish(stop, b'status=STOP\nreason=REINFORCE_INFERENCE_RECOVERY_ERROR\n')
        write_progress(suite, status='STOP', error=repr(error))
        raise
