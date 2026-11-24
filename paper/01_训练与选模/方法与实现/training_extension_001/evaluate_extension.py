"""Run frozen checkpoints on the original BG128 independent validation seeds."""

from itertools import pairwise
from pathlib import Path
from statistics import fmean

from baseline1_v1.evaluation import ReadOnlyTrainingGuard
from business_graph_summary_campaign_v1.plan_io import load_phase, load_reference, reference
from business_graph_summary_evaluation_v1.gate import PLAN as ORIGINAL_EVAL_PLAN
from business_graph_summary_evaluation_v1.gate import check_phase, check_sources, require
from business_graph_summary_evaluation_v1.metrics import build_report, write_reports
from business_graph_summary_evaluation_v1.runtime import (
    _collect, _parent_artifacts, _portable_state_fingerprint, _same_or_save, _status,
)
from business_graph_summary_v1.evaluation import evaluate_episodes
from business_graph_summary_v1.models import digest
from grade_only_ppo_v2.runtime import _actor_snapshot
from online_v1_1.operations import _publish

ORIGINAL_EVAL_SUITE = ORIGINAL_EVAL_PLAN / 'prepared_001/suite.json'
UPDATES = (16, 24, 32)
COUNT = 100
SCOPE = 'paired_seed_fixed_checkpoint_independent_validation'


def _original(suite):
    source = suite['paired_seed_source']
    require(Path(source['path']).resolve() == ORIGINAL_EVAL_SUITE.resolve(),
            'paired evaluation requires the original 100-shift seed source')
    original = load_reference(source)
    require(len(original['jobs']) == COUNT and original['checkpoint_update'] == 13 and
            original['sampling'] == 'stochastic_private_rng',
            'historical BG128 evaluation source differs')
    return original


def _audit_jobs(cfg, original, update):
    require(cfg['evaluation_only'] is True and cfg['optimizer_steps'] == 0 and
            cfg['arm_update'] == update and len(cfg['jobs']) == COUNT,
            'fixed evaluation requires exactly 100 validation slots')
    original_runs = {job['run_index'] for job in original['jobs']}
    require(len({job['run_index'] for job in cfg['jobs']}) == COUNT,
            'duplicate evaluation run index')
    for job, old in zip(cfg['jobs'], original['jobs'], strict=True):
        require(job['run_index'] not in original_runs and
                job['recording_id'] != old['recording_id'] and
                job['source_recording_id'] != old['source_recording_id'] and
                job['python_output'] != old['python_output'] and
                job['material_event_directory'] != old['material_event_directory'],
                'paired seeds still require independent run identities')
        require(all(job[name] == old[name] for name in
                    ('actor_seed', 'experiment_seed', 'actor_sampling_seed',
                     'expected_rpc_config', 'runtime_scene_sha256')),
                'seed pair or RPC contract differs from the corresponding original slot')
        require(job['role'] == 'final' and job['partition'] == 'validation' and
                job['target_training_update'] == 0 and
                job['arm_update'] == job['checkpoint_update'] == update and
                job['phase_id'] == cfg['phase_id'] and
                job['recording_id'] == job['source_recording_id'] == cfg['recording_id'] and
                job['python_output'] == cfg['python_output'],
                'fixed checkpoint, phase or validation identity differs')


def _artifacts(checkpoint, update):
    path = checkpoint['path']
    require(reference(path) == checkpoint, 'evaluation checkpoint SHA differs')
    manifest = reference(path + '.manifest.json')
    seal = reference(path + '.seal.json')
    metadata = load_reference(manifest)
    require(metadata['sha256'] == checkpoint['sha256'] and
            metadata['update_index'] == update and metadata['candidate_id'] == 's01_BG128' and
            load_reference(seal) == dict(sha256=checkpoint['sha256'],
                                         manifest_sha256=manifest['sha256']),
            'evaluation checkpoint manifest, revision or seal differs')
    return dict(checkpoint=checkpoint, manifest=manifest, seal=seal)


def evaluate_phase(suite, phase, ledger, checkpoint_ref):
    """Sample a frozen committed checkpoint on exactly 100 paired slots."""
    from train_extension import load_checkpoint

    update = phase['checkpoint_update']
    require(phase['role'] == 'eval' and update == phase['update_index'] and update in UPDATES,
            'evaluation requires a predeclared checkpoint update')
    original = _original(suite)
    cfg_ref = phase['phase_config']
    config_path = cfg_ref['path']
    require(reference(config_path) == cfg_ref, 'evaluation phase config changed')
    cfg = load_phase(config_path)
    check_phase(cfg)
    _audit_jobs(cfg, original, update)
    require(cfg['campaign_id'] == suite['campaign_id'] and
            cfg['phase_id'] == phase['phase_id'] and
            cfg['python_output'] == phase['python_output'],
            'evaluation phase binding differs')

    output = Path(phase['python_output'])
    artifacts = _artifacts(checkpoint_ref, update)
    commits_before = ledger.resume()['campaigns'][suite['campaign_id']]['commits']
    if update == UPDATES[0]:
        require(artifacts == suite['parent_artifacts'], 'u16 evaluation requires the frozen training parent')
    else:
        require(commits_before[str(update)]['checkpoint_ref'] == checkpoint_ref,
                'evaluation checkpoint differs from the committed training update')
    checkpoint_binding = output / 'evaluation_checkpoint_binding.json'
    result_path = output / 'evaluation_result.json'
    _same_or_save(checkpoint_binding, dict(schema='bg128-extension-evaluation-checkpoint-v1',
        phase_config=cfg_ref, paired_seed_source=suite['paired_seed_source'],
        checkpoint_update=update, parent_artifacts=artifacts, context=SCOPE))
    stage = dict(campaign_id=suite['campaign_id'], control_directory=suite['control_directory'],
        phase_config=cfg_ref, master_plan=suite['master_plan'], python_output=str(output),
        checkpoint_update=update, parent_artifacts=artifacts, selection=reference(checkpoint_binding),
        source_pins=original['source_pins'])
    check_sources(stage['source_pins'])
    if result_path.exists():
        result = load_reference(reference(result_path))
        require(result['binding'] == stage['selection'] and
                result['physical_shifts'] == COUNT and result['optimizer_updates'] == 0 and
                result['checkpoint_update'] == update, 'completed evaluation identity differs')
        state = ledger.resume()
        for job in cfg['jobs']:
            row = state['jobs'][str(job['run_index'])]
            require(row['status'] == 'COMPLETE' and row['identity'] == job and
                    row['receipt']['phase_config'] == cfg_ref and 'committed_update' not in row,
                    'evaluation completion inventory differs')
        for ref in result['reports'].values():
            require(reference(ref['path']) == ref, 'completed evaluation report changed')
        return result

    # The evaluation loader creates a separate model. Protect the live training
    # global streams even while constructing it and during RPC collection.
    with ReadOnlyTrainingGuard():
        model = load_checkpoint(checkpoint_ref, suite['learner_config'], restore=False)
        require(model.update_index == update, 'loaded evaluation checkpoint update differs')
        actor_ref = _actor_snapshot(model, output / 'actor.pt')
        before = _portable_state_fingerprint(model)
        fingerprints = dict(actor=digest(model.actor), critic=digest(model.critic))
        with ReadOnlyTrainingGuard(model):
            episodes, source_refs, _ = _collect(stage, cfg, ledger, model, actor_ref)
            _status(stage, 'EVALUATING', completed_shifts=COUNT, phase_id=cfg['phase_id'])
            report = evaluate_episodes(episodes, model, expected_complete_shifts=COUNT,
                                       checkpoint_provenance=checkpoint_ref)
        after = _portable_state_fingerprint(model)
    require(before == after and report['fingerprints'] == fingerprints and
            _parent_artifacts(stage) == artifacts, 'evaluation altered frozen training state or checkpoint')
    state = ledger.resume()
    require(state['campaigns'][suite['campaign_id']]['commits'] == commits_before and
            all('committed_update' not in state['jobs'][str(job['run_index'])] for job in cfg['jobs']),
            'evaluation created a training commit or consumed a training slot')
    report.update(phase_config=cfg_ref, source_refs=source_refs, evaluation_scope=SCOPE,
                  paired_seed_source=suite['paired_seed_source'], checkpoint_update=update,
                  checkpoint_binding=stage['selection'], model_state_before=before, model_state_after=after)
    summary = build_report(report, expected_complete_shifts=COUNT)
    summary.update(evaluation_scope=SCOPE, checkpoint_update=update)
    report_path = output / 'evaluation_complete.json'
    _same_or_save(report_path, report)
    reports = write_reports(summary, report, output / 'reports')
    result = dict(schema='bg128-extension-evaluation-complete-v1', binding=stage['selection'],
        checkpoint_update=update, checkpoint=checkpoint_ref, actor=actor_ref, fingerprints=fingerprints,
        evaluation_scope=SCOPE, physical_shifts=COUNT, training_updates=0, optimizer_updates=0,
        gradient_updates=0, reports=dict(reports, evaluation=reference(report_path)))
    _same_or_save(result_path, result)
    _status(stage, 'COMPLETE', completed_shifts=COUNT, phase_id=cfg['phase_id'])
    return result


def _audit_rows(report, jobs):
    summary = build_report(report, expected_complete_shifts=COUNT)
    for job, row in zip(jobs, report['per_episode'], strict=True):
        identity = row['identity']
        require(identity['run_index'] == job['run_index'] and
                identity['environment_seed'] == job['experiment_seed'] and
                identity['actor_sampling_seed'] == job['actor_sampling_seed'] and
                identity['recording_id'] == job['recording_id'] and
                identity['source_kind'] == 'fixed_actor' and
                [w['window'] for w in row['physical']['windows']] == list(range(4)),
                'evaluation output identity differs from the frozen pair slot')
    return summary['headline_metrics']


def _physical_row(record):
    physical = record['physical']
    return dict(run_index=record['identity']['run_index'], complete_task=physical['complete_task'],
        all_four_windows_compliant=physical['all_four_windows_compliant'],
        first_window_compliant=physical['windows'][0]['compliant'],
        ore_target_met=physical['ore_target_met'], waste_target_met=physical['waste_target_met'],
        combined_cost=physical['combined_cost'], total_wait_seconds=physical['total_wait_seconds'])


def _paired_delta(rows, before, after):
    pairs = [(row['checkpoints'][str(before)], row['checkpoints'][str(after)]) for row in rows]
    flips = {}
    for key in ('complete_task', 'all_four_windows_compliant', 'first_window_compliant',
                'ore_target_met', 'waste_target_met'):
        improved = sum(not old[key] and new[key] for old, new in pairs)
        worsened = sum(old[key] and not new[key] for old, new in pairs)
        flips[key] = dict(improved=improved, worsened=worsened,
                          both_pass=sum(old[key] and new[key] for old, new in pairs),
                          both_fail=sum(not old[key] and not new[key] for old, new in pairs),
                          difference_in_passes=improved - worsened)
    return dict(before=before, after=after, paired_shifts=len(pairs), outcomes=flips,
        mean_combined_cost_delta=fmean(new['combined_cost'] - old['combined_cost'] for old, new in pairs),
        mean_total_wait_seconds_delta=fmean(new['total_wait_seconds'] - old['total_wait_seconds']
                                            for old, new in pairs))


def write_paired_report(suite, phase_results, report_dir):
    """Audit and compare complete results by both seed streams and physical slot."""
    original = _original(suite)
    old_completion_ref = reference(Path(original['python_output']) / 'suite_complete.json')
    old_completion = load_reference(old_completion_ref)
    require(old_completion['campaign'] == suite['paired_seed_source'] and
            old_completion['physical_shifts'] == COUNT and old_completion['optimizer_updates'] == 0,
            'historical completed fixed evaluation identity differs')
    old_report_ref = old_completion['reports']['evaluation']
    old_report = load_reference(old_report_ref)
    sources = {13: old_report_ref}
    records = {13: old_report['per_episode']}
    headline = {13: _audit_rows(old_report, original['jobs'])}
    results = {result['checkpoint_update']: result for result in phase_results}
    require(len(phase_results) == len(results) == len(UPDATES) and set(results) == set(UPDATES),
            'comparison needs complete evaluations at u16, u24 and u32')
    phases = {phase['checkpoint_update']: phase for phase in suite['phases'] if phase['role'] == 'eval'}
    require(set(phases) == set(UPDATES), 'comparison checkpoints differ from the frozen phase schedule')
    for update, result in sorted(results.items()):
        phase = phases[update]
        cfg_ref = phase['phase_config']
        require(reference(cfg_ref['path']) == cfg_ref and result['physical_shifts'] == COUNT and
                result['optimizer_updates'] == result['training_updates'] == result['gradient_updates'] == 0,
                'paired evaluation phase scope or counters differ')
        cfg = load_phase(cfg_ref['path'])
        _audit_jobs(cfg, original, update)
        sources[update] = result['reports']['evaluation']
        report = load_reference(sources[update])
        require(report['phase_config'] == cfg_ref and report['checkpoint_update'] == update and
                report['checkpoint_provenance'] == result['checkpoint'] and
                report['evaluation_scope'] == SCOPE and
                report['paired_seed_source'] == suite['paired_seed_source'],
                'paired evaluation report provenance differs')
        headline[update] = _audit_rows(report, cfg['jobs'])
        records[update] = report['per_episode']

    rows = [dict(ordinal=ordinal, environment_seed=job['experiment_seed'],
        actor_sampling_seed=job['actor_sampling_seed'],
        checkpoints={str(update): _physical_row(values[ordinal]) for update, values in records.items()})
        for ordinal, job in enumerate(original['jobs'])]
    all_runs = {entry['run_index'] for row in rows for entry in row['checkpoints'].values()}
    require(len(all_runs) == COUNT * len(records), 'physical run repeated across checkpoint evaluations')
    comparisons = [(16, 32), *pairwise(UPDATES), *((13, update) for update in UPDATES)]
    deltas = [_paired_delta(rows, old, new) for old, new in comparisons]
    payload = dict(schema='bg128-extension-paired-evaluation-v1', evaluation_scope=SCOPE,
        seed_source=suite['paired_seed_source'], historical_reference_update=13,
        primary_comparison=dict(before=16, after=32), paired_shifts=COUNT,
        sources={str(update): ref for update, ref in sources.items()},
        metrics={str(update): value for update, value in headline.items()},
        deltas=deltas, rows=rows)
    folder = Path(report_dir)
    json_path = folder / 'paired_evaluation.json'
    _same_or_save(json_path, payload)
    lines = ['# BG128 延训固定策略配对独评', '',
        'u16、u24 和 u32 各实跑 100 班，逐槽复用历史 u13 独评的环境和策略采样种子。',
        '各轮使用独立运行身份。历史 u13 作为同种子参照。主要比较为 u32 对 u16。', '',
        '| 检查点 | 完整任务 | 四窗合规 | 首窗合规 | 矿石达产 | 废石达产 | 品位 RMSE | 平均等待（车辆秒） | Critic MC MSE | Critic EV |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for update, head in headline.items():
        ev = head['critic_mc_explained_variance_pooled']
        lines.append(f"| u{update} | {head['complete_task_shifts']}/100 | "
            f"{head['all_four_compliant_shifts']}/100 | "
            f"{head['per_window_compliance'][0]['compliant_shifts']}/100 | "
            f"{head['ore_target_met_shifts']}/100 | {head['waste_target_met_shifts']}/100 | "
            f"{head['grade_rmse']:.6f} | {head['mean_total_wait_seconds']:.2f} | "
            f"{head['critic_mc_mse_pooled']:.6f} | {f'{ev:.6f}' if ev is not None else 'NA'} |")
    lines.extend(['', '| 前后检查点 | 完整任务改善 / 退化 | 首窗改善 / 退化 | 平均联合成本差 | 平均等待差（车辆秒） |',
                  '| --- | ---: | ---: | ---: | ---: |'])
    for delta in deltas:
        task = delta['outcomes']['complete_task']
        first = delta['outcomes']['first_window_compliant']
        lines.append(f"| u{delta['before']} → u{delta['after']} | "
            f"{task['improved']} / {task['worsened']} | {first['improved']} / {first['worsened']} | "
            f"{delta['mean_combined_cost_delta']:+.6f} | {delta['mean_total_wait_seconds_delta']:+.2f} |")
    lines.extend(['', '差值按同一班次种子对计算，后一个检查点减前一个检查点。',
                  '', '[100 个种子对及配对明细](paired_evaluation.json)', ''])
    markdown_path = folder / 'paired_evaluation.md'
    data = '\n'.join(lines).encode()
    if markdown_path.exists():
        require(markdown_path.read_bytes() == data, 'paired evaluation report changed')
    else:
        _publish(markdown_path, data)
    return dict(json=reference(json_path), markdown=reference(markdown_path))
