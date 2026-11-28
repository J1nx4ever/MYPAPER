"""Reconstruct the realized Eq. 19 weights from all sealed S02 training shifts."""

import json
import math
import sys
from pathlib import Path
from statistics import fmean, pstdev

import h5py

FOLDER = Path(__file__).resolve().parent
ROOT = FOLDER.parents[4]
sys.path.insert(0, str(ROOT))
from change_plans.token_attention_001.business_graph_summary_009.reinforce_comparison_001 import compare_training as comparison
from business_graph_summary_campaign_v1.plan_io import load_reference, load_suite, reference
from business_graph_summary_evaluation_v1.metrics import _publish, _json
from business_graph_summary_v1.contract import require

TRAIN = ROOT / 'runs/bg128-reinforce-full-shift-20261127-s02'
COMPLETION = {'path': str(TRAIN / 'suite_complete.json'),
              'sha256': '6fabcaa84bfb9e9efdda903fd4967db9cbeac3c1a3617d6227d5eaddee3b8b8e'}


def aggregate(shifts):
    returns = [s['return'] for s in shifts]
    advantage = [s['advantage'] for s in shifts]
    return dict(shifts=len(shifts), decisions=sum(s['decisions'] for s in shifts),
                return_mean=fmean(returns), return_sd_population=pstdev(returns),
                return_min=min(returns), return_max=max(returns),
                advantage_mean=fmean(advantage), advantage_positive=sum(x > 0 for x in advantage),
                advantage_negative=sum(x < 0 for x in advantage), advantage_zero=sum(x == 0 for x in advantage),
                complete_tasks=sum(s['complete'] for s in shifts),
                positive_advantage_in_incomplete_shifts=sum(s['advantage'] > 0 and not s['complete'] for s in shifts),
                grade_cost_mean=fmean(s['grade_cost'] for s in shifts),
                production_cost_mean=fmean(s['production_cost'] for s in shifts),
                waiting_cost_mean=fmean(s['wait_cost'] for s in shifts),
                decisions_after_first_settled_window=sum(s['after_close'] for s in shifts))


def shift(job, record, phase, phase_ref, source, baseline):
    require(load_reference(source['receipt']) == record['receipt'], 'physical receipt differs from the update')
    cost, complete, wait, ore, waste, decisions = comparison.shift_metrics(job, record, phase, phase_ref, source['dataset'])
    contract = phase['composite_contract']
    wait_cost = wait * contract['wait_weight'] / contract['wait_reference_seconds']
    production = (contract['ore_shortfall_weight'] * max(0., 1 - ore / contract['ore_target_tons'])
                  + contract['waste_shortfall_weight'] * max(0., 1 - waste / contract['waste_target_tons']))
    with h5py.File(record['receipt']['references']['episode']['path'], 'r') as hdf:
        events = hdf['material_events']
        kinds, sequence = events['kind'][:], events['seq'][:]
        decisions_seq = sequence[kinds == b'DECISION']
        close0 = sequence[kinds == b'WINDOW_CLOSE'][0]
        logp = hdf['transitions/old_log_prob'][:]
        require(len(logp) == len(decisions_seq) == decisions, 'decision count differs')
    advantage = -cost - baseline
    return {'return': -cost, 'advantage': advantage, 'complete': complete,
            'grade_cost': cost - wait_cost - production, 'wait_cost': wait_cost, 'production_cost': production,
            'decisions': decisions, 'after_close': int((decisions_seq > close0).sum()),
            'weighted_loss': -advantage * math.fsum(logp)}


def main():
    completed = load_reference(COMPLETION)
    require(completed['status'] == 'COMPLETE' and completed['training_shifts'] == 1024
            and completed['training_updates'] == 32, 'expected full S02 training')
    load_reference(completed['suite'])
    suite, phases = load_suite(completed['suite']['path'])
    commits, records = comparison.ledger_snapshot(suite)
    output = dict(completion=COMPLETION, batches=[], phases={}, physical_shifts_started=0)
    before = 0.
    all_shifts = []
    for u, (phase, ref, summary_ref) in enumerate(zip(phases, suite['phase_configs'], completed['updates'], strict=True), 1):
        summary = load_reference(summary_ref)
        metrics, intent = summary['reinforce'], summary['intent']
        jobs = phase['jobs']
        commit = commits[str(u)]
        require(commit['job_ids'] == [str(job['run_index']) for job in jobs]
                and commit['batch_id'] == intent['batch_id'] == metrics['batch_id']
                and metrics['baseline_before'] == before and metrics['update_index'] == u
                and intent['phase_config'] == ref and metrics['episodes'] == len(jobs) == 32,
                'training transaction differs')
        shifts = [shift(job, records[str(job['run_index'])], phase, ref, source, before)
                  for job, source in zip(jobs, intent['sources'], strict=True)]
        values = aggregate(shifts)
        loss = fmean(x['weighted_loss'] for x in shifts)
        require(math.isclose(loss, metrics['policy_loss'], rel_tol=1e-6, abs_tol=1e-5)
                and math.isclose(values['return_mean'], metrics['mean_return'], abs_tol=1e-10)
                and math.isclose(metrics['baseline_after'], (before + values['return_mean']) / 2, abs_tol=1e-10)
                and values['decisions'] == metrics['transitions'] and metrics['actor_optimizer_steps'] == 1,
                'archived behavior probabilities, costs, baseline or updates disagree')
        output['batches'].append(dict(update=u, baseline_before=before, baseline_after=metrics['baseline_after'],
                                      grad_norm=metrics['actor_grad_norm'], policy_loss=metrics['policy_loss'],
                                      behavior_logp_recomputed_loss=loss, **values))
        all_shifts.extend(shifts)
        before = metrics['baseline_after']
    output['all'] = aggregate(all_shifts)
    for start in (0, 8, 16, 24):
        output['phases'][f'{start+1:02d}-{start+8:02d}'] = aggregate(all_shifts[32*start:32*(start+8)])
    out = ROOT / 'reports' / FOLDER.relative_to(ROOT / 'change_plans') / 'signal_audit.json'
    print(json.dumps(_publish(out, _json(output))))


if __name__ == '__main__':
    main()
