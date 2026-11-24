"""Run and compare complete shifts using the sealed BG128 u32 policy."""
from functools import lru_cache
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from statistics import fmean

import numpy as np

from baseline1_v1.evaluation import ReadOnlyTrainingGuard
from business_graph_summary_campaign_v1.plan_io import load_phase, load_reference, reference
from business_graph_summary_evaluation_v1.gate import check_phase, check_sources, require
from business_graph_summary_evaluation_v1.metrics import build_report, write_reports
from business_graph_summary_evaluation_v1.runtime import (
    _collect, _parent_artifacts, _portable_state_fingerprint, _same_or_save, _status,
)
from business_graph_summary_v1.evaluation import evaluate_episodes
from business_graph_summary_v1.models import digest
from grade_only_ppo_v2.runtime import _actor_snapshot
from online_v1_1.operations import _publish
from python.observation_contract import FIELD_NAMES
from python.runtime_scenario import PREFIX, fingerprint

COUNT = 100
UPDATE = 32
SCENARIOS = ('nominal', 'grade_swap', 'speed_18kph', 'speed_15kph',
             'grade_mean_shift', 'grade_variability')
SCOPE = 'one_training_seed_fixed_u32_paired_generalization'
ROAD_INDEX = 6
EXPECTED_OBSERVATIONS = {
    **{f'load.{site}.meanGrade': (f'{PREFIX}load.{site}.grade', 1.)
       for site in range(3)},
    **{f'load.{site}.gradeHalfRange': (f'{PREFIX}load.{site}.gradeHalfRange', 1.)
       for site in range(3)},
    f'road.{ROAD_INDEX}.limit': (f'{PREFIX}road.{ROAD_INDEX}.speedLimit_mps', 10.),
}


@lru_cache(maxsize=1)
def _training_extension():
    path = Path(__file__).resolve().parent.parent / 'training_extension_001' / 'train_extension.py'
    spec = spec_from_file_location('bg128_generalization_checkpoint_loader', path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_fixed_model(suite):
    """Load the selected extended checkpoint without changing live random states."""
    ref = suite['parent_artifacts']['checkpoint']
    model = _training_extension().load_checkpoint(ref, suite['learner_config'], restore=False)
    require(model.update_index == UPDATE and len(model.consumed_episodes) == 1024,
            'generalization requires the frozen completed u32 model')
    return model


def _phase(suite, phase):
    cfg_ref = phase['phase_config']
    require(reference(cfg_ref['path']) == cfg_ref, 'unfrozen generalization phase')
    cfg = load_phase(cfg_ref['path'])
    check_phase(cfg)
    scenario = phase['scenario_id']
    require(scenario in SCENARIOS and phase['checkpoint_update'] == UPDATE and
            cfg['campaign_id'] == suite['campaign_id'] and cfg['phase_id'] == phase['phase_id'] and
            cfg['python_output'] == phase['python_output'] and cfg['arm_update'] == UPDATE and
            cfg['evaluation_only'] is True and cfg['optimizer_steps'] == 0 and
            len(cfg['jobs']) == COUNT, 'generalization phase identity differs')
    rpc = cfg['jobs'][0]['expected_rpc_config']
    scene_hash = fingerprint({key: value for key, value in rpc.items() if key.startswith(PREFIX)})
    require(scene_hash == phase['runtime_scene_sha256'], 'generalization scene fingerprint differs')
    for ordinal, job in enumerate(cfg['jobs']):
        require(job['pair_ordinal'] == ordinal and job['scene_id'] == scenario and
                job['checkpoint_update'] == job['arm_update'] == UPDATE and
                job['expected_rpc_config'] == rpc and job['runtime_scene_sha256'] == scene_hash and
                job['role'] == 'final' and job['partition'] == 'validation' and
                job['target_training_update'] == 0, 'generalization slot identity differs')
    return cfg


def _sampled_speed(episodes):
    # Decision snapshots sample traffic; the resulting quantiles are not
    # full-segment travel times or time-weighted speed statistics.
    columns = [[FIELD_NAMES.index(f'truck.{truck}.{field}') for truck in range(9)]
               for field in (f'segment.{ROAD_INDEX}', 'loaded', 'ore', 'speed')]
    samples = []
    for episode in episodes:
        states = episode['states'][:-1]
        on_road, loaded, ore, speed = (states[:, group] for group in columns)
        observed = (on_road == 1) & (loaded == 1) & (ore == 1)
        samples.append(speed[observed] * 36.)
    values = np.concatenate(samples)
    return dict(road_index=ROAD_INDEX, scope='loaded_ore_trucks_at_dispatch_decisions',
                n_samples=len(values), sampled_shifts=sum(bool(len(row)) for row in samples),
                mean_kph=float(values.mean()) if len(values) else None,
                p50_kph=float(np.quantile(values, .5)) if len(values) else None,
                p90_kph=float(np.quantile(values, .9)) if len(values) else None,
                max_kph=float(values.max()) if len(values) else None)


def scenario_evidence(episodes, cfg, phase):
    """Confirm observed inputs and summarize compact material/traffic evidence."""
    rpc = cfg['jobs'][0]['expected_rpc_config']
    expected = {
        name: float(rpc[key]) / scale for name, (key, scale) in EXPECTED_OBSERVATIONS.items()
    }
    require(rpc[f'{PREFIX}env.deterministicCheck'] == 0.,
            'grade variability evaluation requires stochastic material blocks')
    fields = list(expected)
    indices = [FIELD_NAMES.index(name) for name in fields]
    vector = np.asarray(list(expected.values()), dtype=np.float32)
    for episode in episodes:
        observed = episode['states'][:, indices]
        require(np.allclose(observed, vector, rtol=0, atol=1e-7),
                'actual physical observations differ from frozen grade or speed settings')
    by_source = {site: [] for site in range(3)}
    for episode in episodes:
        for event in episode['memory'].events:
            if event.kind == 'UNLOAD' and event.source_id in by_source:
                by_source[event.source_id].append(event)
    ore_sources = {}
    for site, events in by_source.items():
        grades = [event.grade for event in events]
        total_tons = sum(event.tons for event in events)
        mean_grade = sum(event.metal for event in events) / total_tons if total_tons else None
        grade_std = (sum(event.tons * (event.grade - mean_grade) ** 2 for event in events)
                     / total_tons) ** .5 if total_tons else None
        ore_sources[str(site)] = dict(unload_events=len(events), unloaded_tons=total_tons,
            tonnage_weighted_mean_grade=mean_grade, tonnage_weighted_grade_std=grade_std,
            min_grade=min(grades) if grades else None, max_grade=max(grades) if grades else None)
    return dict(runtime_scene_sha256=phase['runtime_scene_sha256'],
                rpc_verified_shifts=len(episodes), phase_config=phase['phase_config'],
                verified_observations_float32={name: float(value)
                                               for name, value in zip(fields, vector, strict=True)},
                ore_unload_by_source=ore_sources, sampled_road_speed=_sampled_speed(episodes))


def _check_completed_jobs(suite, cfg, ledger, cfg_ref):
    state = ledger.resume()
    require(state['campaigns'][suite['campaign_id']]['commits'] == {},
            'generalization created a training commit')
    for job in cfg['jobs']:
        row = state['jobs'][str(job['run_index'])]
        require(row['status'] == 'COMPLETE' and row['identity'] == job and
                row['receipt']['phase_config'] == cfg_ref and 'committed_update' not in row,
                'generalization completed slot identity differs')


def evaluate_phase(suite, phase, ledger):
    """Run one planned scenario with zero gradient or optimizer updates."""
    cfg = _phase(suite, phase)
    cfg_ref, scenario = phase['phase_config'], phase['scenario_id']
    output = Path(phase['python_output'])
    artifacts = suite['parent_artifacts']
    checkpoint = artifacts['checkpoint']
    metadata = load_reference(artifacts['manifest'])
    require(metadata['sha256'] == checkpoint['sha256'] and metadata['update_index'] == UPDATE and
            metadata['candidate_id'] == 's01_BG128' and load_reference(artifacts['seal']) ==
            dict(sha256=checkpoint['sha256'], manifest_sha256=artifacts['manifest']['sha256']),
            'u32 checkpoint seal does not bind the selected Actor')
    binding_path = output / 'evaluation_checkpoint_binding.json'
    _same_or_save(binding_path, dict(schema='bg128-generalization-checkpoint-v1',
        phase_config=cfg_ref, scenario_id=scenario, runtime_scene_sha256=phase['runtime_scene_sha256'],
        checkpoint_update=UPDATE, parent_artifacts=artifacts, evaluation_scope=SCOPE))
    stage = dict(campaign_id=suite['campaign_id'], control_directory=suite['control_directory'],
        phase_config=cfg_ref, master_plan=suite['master_plan'], python_output=str(output),
        checkpoint_update=UPDATE, parent_artifacts=artifacts, selection=reference(binding_path),
        source_pins=suite['evaluation_source_pins'])
    check_sources(stage['source_pins'])
    require(_parent_artifacts(stage) == artifacts, 'generalization checkpoint artifacts differ')
    result_path = output / 'evaluation_result.json'
    if result_path.exists():
        result = load_reference(reference(result_path))
        require(result['binding'] == stage['selection'] and result['scenario_id'] == scenario and
                result['physical_shifts'] == COUNT and result['checkpoint'] == checkpoint and
                result['optimizer_updates'] == result['training_updates'] == result['gradient_updates'] == 0,
                'existing generalization result differs')
        _check_completed_jobs(suite, cfg, ledger, cfg_ref)
        for ref in result['reports'].values():
            require(reference(ref['path']) == ref, 'completed generalization report changed')
        return result

    with ReadOnlyTrainingGuard():
        model = load_fixed_model(suite)
        actor_ref = _actor_snapshot(model, output / 'actor.pt')
        before = _portable_state_fingerprint(model)
        fingerprints = dict(actor=digest(model.actor), critic=digest(model.critic))
        with ReadOnlyTrainingGuard(model):
            episodes, sources, _ = _collect(stage, cfg, ledger, model, actor_ref)
            _status(stage, 'EVALUATING', completed_shifts=COUNT, phase_id=cfg['phase_id'])
            evidence = scenario_evidence(episodes, cfg, phase)
            report = evaluate_episodes(episodes, model, expected_complete_shifts=COUNT,
                                       checkpoint_provenance=checkpoint)
        after = _portable_state_fingerprint(model)
    require(before == after and report['fingerprints'] == fingerprints and
            metadata['actor_digest'] == fingerprints['actor'] and
            _parent_artifacts(stage) == artifacts, 'generalization changed the selected training state')
    _check_completed_jobs(suite, cfg, ledger, cfg_ref)
    report.update(phase_config=cfg_ref, source_refs=sources, scenario_id=scenario,
                  scenario_evidence=evidence, evaluation_scope=SCOPE, checkpoint_update=UPDATE,
                  checkpoint_binding=stage['selection'], model_state_before=before, model_state_after=after)
    summary = build_report(report, expected_complete_shifts=COUNT)
    summary.update(evaluation_scope=SCOPE, checkpoint_update=UPDATE, scenario_id=scenario,
                   scenario_evidence=evidence)
    report_path = output / 'evaluation_complete.json'
    _same_or_save(report_path, report)
    reports = write_reports(summary, report, output / 'reports')
    result = dict(schema='bg128-generalization-phase-complete-v1', binding=stage['selection'],
        scenario_id=scenario, checkpoint_update=UPDATE, checkpoint=checkpoint, actor=actor_ref,
        fingerprints=fingerprints, evaluation_scope=SCOPE, physical_shifts=COUNT, training_updates=0,
        optimizer_updates=0, gradient_updates=0, reports=dict(reports, evaluation=reference(report_path)))
    _same_or_save(result_path, result)
    _status(stage, 'COMPLETE', completed_shifts=COUNT, phase_id=cfg['phase_id'])
    return result


def _paired_delta(rows, scenario):
    pairs = [(row['scenarios']['nominal'], row['scenarios'][scenario]) for row in rows]
    outcomes = {}
    for key in ('complete_task', 'all_four_windows_compliant', 'first_window_compliant',
                'ore_target_met', 'waste_target_met'):
        outcomes[key] = dict(
            improved=sum(not base[key] and test[key] for base, test in pairs),
            worsened=sum(base[key] and not test[key] for base, test in pairs),
            both_pass=sum(base[key] and test[key] for base, test in pairs),
            both_fail=sum(not base[key] and not test[key] for base, test in pairs),
        )
    continuous = {key: fmean(test[key] - base[key] for base, test in pairs)
                  for key in ('combined_cost', 'total_wait_seconds', 'ore_tons', 'waste_tons',
                              'first_window_tons', 'first_window_metal_tons')}
    return dict(scenario_id=scenario, paired_shifts=len(pairs), outcomes=outcomes,
                mean_test_minus_nominal=continuous)


def _row(record):
    p = record['physical']
    w0 = p['windows'][0]
    return dict(run_index=record['identity']['run_index'],
        first_window_compliant=w0['compliant'], first_window_tons=w0['tons'],
        first_window_metal_tons=w0['metal_tons'], first_window_grade=w0['grade'],
        **{key: p[key] for key in ('complete_task', 'all_four_windows_compliant', 'ore_target_met',
                                  'waste_target_met', 'ore_tons', 'waste_tons', 'combined_cost',
                                  'total_wait_seconds')})


def write_comparison(suite, phase_results, report_dir):
    """Compare each of the five complete perturbations with the nominal arm."""
    results = {result['scenario_id']: result for result in phase_results}
    require(len(results) == len(phase_results) == len(SCENARIOS) and set(results) == set(SCENARIOS),
            'paired report requires all six completed scenarios')
    checkpoint = suite['parent_artifacts']['checkpoint']
    sources, metrics, evidence, all_runs = {}, {}, {}, set()
    rows = [dict(pair_ordinal=i, scenarios={}) for i in range(COUNT)]
    fingerprints = results['nominal']['fingerprints']
    for phase in suite['phases']:
        scenario = phase['scenario_id']
        cfg = _phase(suite, phase)
        result = results[scenario]
        require(result['checkpoint'] == checkpoint and result['fingerprints'] == fingerprints and
                result['physical_shifts'] == COUNT and result['training_updates'] ==
                result['optimizer_updates'] == result['gradient_updates'] == 0,
                'paired result does not belong to the frozen model')
        ref = result['reports']['evaluation']
        report = load_reference(ref)
        require(report['checkpoint_provenance'] == checkpoint and report['scenario_id'] == scenario and
                report['phase_config'] == phase['phase_config'] and report['evaluation_scope'] == SCOPE and
                report['scenario_evidence']['rpc_verified_shifts'] == COUNT and
                report['scenario_evidence']['runtime_scene_sha256'] == phase['runtime_scene_sha256'],
                'paired scenario report is not bound to verified real shifts')
        sources[scenario] = ref
        metrics[scenario] = build_report(report, expected_complete_shifts=COUNT)['headline_metrics']
        evidence[scenario] = report['scenario_evidence']
        for paired, job, record in zip(rows, cfg['jobs'], report['per_episode'], strict=True):
            identity = record['identity']
            seeds = dict(environment_seed=job['experiment_seed'], actor_sampling_seed=job['actor_sampling_seed'])
            require(identity['run_index'] == job['run_index'] and identity['run_index'] not in all_runs and
                    identity['scene_id'] == scenario and identity['recording_id'] == job['recording_id'] and
                    identity['source_kind'] == 'fixed_actor' and
                    all(identity[key] == value for key, value in seeds.items()) and
                    paired.setdefault('seeds', seeds) == seeds and
                    [w['window'] for w in record['physical']['windows']] == list(range(4)),
                    'paired job and physical evidence differ')
            all_runs.add(identity['run_index'])
            paired['scenarios'][scenario] = _row(record)
    require(len(all_runs) == COUNT * len(SCENARIOS), 'generalization physical identities overlap')
    deltas = [_paired_delta(rows, scenario) for scenario in SCENARIOS[1:]]
    folder = Path(report_dir)
    json_path = folder / 'paired_generalization.json'
    payload = dict(schema='bg128-paired-generalization-v1', evaluation_scope=SCOPE,
        checkpoint=checkpoint, training_seeds=1, evaluated_shifts=len(all_runs), paired_slots=COUNT,
        fingerprints=fingerprints, optimizer_updates=0, sources=sources, metrics=metrics,
        scene_evidence=evidence, paired_perturbations=deltas, rows=rows)
    _same_or_save(json_path, payload)

    def fmt(value):
        return format(value, '.6f') if value is not None else 'NA'

    lines = ['# BG128 u32 泛化实测', '',
             '一个训练随机种子的固定 u32 策略。六组各 100 班，组间逐槽匹配环境与动作采样种子。',
             '首次 RPC、每次动作 RPC 与终止 RPC 均按该组冻结工况核对，训练及优化器更新为 0。', '',
             '| 工况 | 完整任务 | 四窗合规 | 首窗合规 | 矿石达产 | 废石达产 | 品位 RMSE | 平均等待（车·秒） | Critic MC MSE | Critic EV |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for scenario in SCENARIOS:
        m = metrics[scenario]
        lines.append(f"| {scenario} | {m['complete_task_shifts']}/100 | {m['all_four_compliant_shifts']}/100 | "
            f"{m['per_window_compliance'][0]['compliant_shifts']}/100 | {m['ore_target_met_shifts']}/100 | "
            f"{m['waste_target_met_shifts']}/100 | {fmt(m['grade_rmse'])} | {m['mean_total_wait_seconds']:.2f} | "
            f"{fmt(m['critic_mc_mse_pooled'])} | {fmt(m['critic_mc_explained_variance_pooled'])} |")
    lines.extend(['', '下表为同槽扰动组减标称组。', '',
        '| 扰动 | 完整任务改善 / 退化 | 首窗改善 / 退化 | 平均成本差 | 平均等待差（车·秒） |',
        '| --- | ---: | ---: | ---: | ---: |'])
    for delta in deltas:
        task = delta['outcomes']['complete_task']
        first = delta['outcomes']['first_window_compliant']
        continuous = delta['mean_test_minus_nominal']
        lines.append(f"| {delta['scenario_id']} | {task['improved']} / {task['worsened']} | "
                     f"{first['improved']} / {first['worsened']} | {continuous['combined_cost']:+.6f} | "
                     f"{continuous['total_wait_seconds']:+.2f} |")
    lines.extend(['', '| 工况 | road.6 载矿车决策快照数 | 速度 P50 km/h | 速度 P90 km/h |',
                  '| --- | ---: | ---: | ---: |'])
    for scenario in SCENARIOS:
        speed = evidence[scenario]['sampled_road_speed']
        lines.append(f"| {scenario} | {speed['n_samples']} | {fmt(speed['p50_kph'])} | "
                     f"{fmt(speed['p90_kph'])} |")
    lines.extend(['', '速度按派车决策快照统计，包含被采到的载矿车。逐班材料结算与各矿源卸载汇总见下方数据。',
                  '', '[同槽明细与工况证据](paired_generalization.json)', ''])
    markdown_path = folder / 'paired_generalization.md'
    data = '\n'.join(lines).encode()
    if markdown_path.exists():
        require(markdown_path.read_bytes() == data, 'paired Markdown report changed')
    else:
        _publish(markdown_path, data)
    return dict(json=reference(json_path), markdown=reference(markdown_path))
