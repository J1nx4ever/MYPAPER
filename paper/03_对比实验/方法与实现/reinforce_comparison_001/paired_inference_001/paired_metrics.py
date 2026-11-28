"""Compare the two frozen S02 policies on the same 700 simulated conditions."""
from pathlib import Path
from statistics import fmean

from business_graph_summary_campaign_v1.plan_io import load_phase, load_reference, reference
from business_graph_summary_evaluation_v1.metrics import _csv, _json, _publish
from business_graph_summary_v1.contract import require
from change_plans.token_attention_001.business_graph_summary_009.reinforce_comparison_001.prepare import COMPLETION


OUTCOMES = ('complete_task', 'all_four_windows_compliant', 'ore_target_met', 'waste_target_met')
QUANTITIES = ('combined_cost', 'total_wait_seconds', 'ore_tons', 'waste_tons')


def _slot(scenario, ppo_job, new_job, ppo_row, new_row):
    pair_keys = ('pair_ordinal', 'scene_id', 'experiment_seed', 'actor_sampling_seed',
                 'runtime_scene_sha256', 'expected_rpc_config')
    require(all(ppo_job[key] == new_job[key] for key in pair_keys)
            and ppo_job['recording_id'] != new_job['recording_id']
            and ppo_job['run_index'] != new_job['run_index'], 'paired physical slot differs')
    row = dict(scenario=scenario, slot=new_job['pair_ordinal'], environment_seed=new_job['experiment_seed'],
               actor_sampling_seed=new_job['actor_sampling_seed'])
    for name, job, record in (('ppo', ppo_job, ppo_row), ('reinforce', new_job, new_row)):
        identity, physical = record['identity'], record['physical']
        require(identity['run_index'] == job['run_index'] and identity['recording_id'] == job['recording_id']
                and identity['environment_seed'] == job['experiment_seed']
                and identity['actor_sampling_seed'] == job['actor_sampling_seed']
                and identity['scene_id'] == job['scene_id'] and identity['split'] == 'validation'
                and len(physical['windows']) == 4, 'evaluation record differs from the paired slot')
        row[name + '_run'] = job['run_index']
        for field in (*OUTCOMES, *QUANTITIES):
            row[f'{name}_{field}'] = physical[field]
    return row


def _summary(rows):
    totals = {}
    for k in OUTCOMES:
        pairs = [(r['ppo_' + k], r['reinforce_' + k]) for r in rows]
        totals[k] = dict(ppo=sum(p for p, _ in pairs), reinforce=sum(n for _, n in pairs),
                         only_ppo=sum(p and not n for p, n in pairs),
                         only_reinforce=sum(n and not p for p, n in pairs))
    means = {k: {arm: fmean(r[f'{arm}_{k}'] for r in rows) for arm in ('ppo', 'reinforce')}
             for k in QUANTITIES}
    deltas = {k: fmean(r['reinforce_' + k] - r['ppo_' + k] for r in rows) for k in QUANTITIES}
    return dict(shifts_per_policy=len(rows), outcomes=totals, means=means, mean_reinforce_minus_ppo=deltas)


def write_comparison(suite, configs, results):
    """Publish after all seven new reports have sealed and passed pair validation."""
    historical = load_reference(COMPLETION)
    require(len(configs) == len(results) == len(historical['phase_results']) == 7, 'seven cases required')
    rows, groups, sources = [], {}, {}
    for cfg, result, old in zip(configs, results, historical['phase_results'], strict=True):
        scenario = cfg['scenario_id']
        ppo = load_reference(old['report'])
        new = load_reference(result['report'])
        ppo_cfg = load_phase(ppo['phase_config']['path'])
        require(ppo['phase_config'] == reference(ppo['phase_config']['path'])
                and new['phase_config'] == suite['phase_configs'][len(groups)]
                and new['scenario_id'] == ppo['scenario_id'] == scenario and scenario not in groups
                and historical['new_checkpoint'] == old['checkpoint'] == ppo['checkpoint_provenance']
                and new['checkpoint_provenance'] == result['checkpoint']
                and new['composite_contract'] == ppo['composite_contract']
                and all(r['evaluation_optimizer_updates'] == 0 for r in (ppo, new)),
                'source policy or evaluation contract differs')
        pairs = [_slot(scenario, a, b, x, y) for a, b, x, y in
                 zip(ppo_cfg['jobs'], cfg['jobs'], ppo['per_episode'], new['per_episode'], strict=True)]
        require(len(pairs) == 100 and len({row['slot'] for row in pairs}) == 100, '100 separate slots required')
        groups[scenario] = _summary(pairs)
        sources[scenario] = dict(ppo=old['report'], reinforce=result['report'])
        rows.extend(pairs)
    folder = Path(suite['report_directory'])
    slots = _publish(folder / 'paired_shifts.csv', _csv(rows))
    summary = dict(campaign=reference(suite['suite_path']), historical_completion=COMPLETION,
                   paired_shifts=slots, sources=sources, by_scenario=groups, overall=_summary(rows))
    return dict(summary=_publish(folder / 'paired_summary.json', _json(summary)), per_slot=slots)
