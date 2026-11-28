"""Compare the complete committed training batches on the matched S02 schedule."""

import csv
import json
import math
from pathlib import Path
from statistics import fmean
from tempfile import NamedTemporaryFile

if __package__:
    from . import prepare as plan
else:
    import prepare as plan

import h5py
from business_graph_summary_campaign_v1.budget import _payload, _state, connection
from business_graph_summary_campaign_v1.plan_io import load_reference, reference, sha
from business_graph_summary_v1.contract import require
from business_graph_summary_v1.targets import grade_cost
from single_critic_task_query_v1.evaluation import _grade_compliant


METRICS = ('shifts', 'mean_j', 'complete_task_count', 'complete_task_rate', 'mean_wait_vehicle_seconds',
           'mean_ore_tons', 'mean_waste_tons', 'decisions', 'actor_optimizer_steps')
DELTAS = ('mean_j', 'complete_task_rate', 'mean_wait_vehicle_seconds')
FIELDS = ('update', 'matched_scene_slots',
          *(f'{arm}_{metric}' for arm in ('ppo', 'reinforce') for metric in METRICS),
          *(f'reinforce_minus_ppo_{metric}' for metric in DELTAS))


def ledger_snapshot(suite):
    """Read one SQLite snapshot even if the training controller is committing a batch."""
    with connection(suite['budget_path'], readonly=True) as db:
        db.execute('BEGIN')
        row = db.execute('SELECT payload,checksum FROM campaigns WHERE id=?', (suite['campaign_id'],)).fetchone()
        require(row is not None, 'campaign registration missing from ledger')
        entry = _payload(row)
        require(entry['suite_reference'] == reference(suite['suite_path']), 'registered plan bytes differ')
        state = _state(db, suite, entry)
    return state['campaigns'][suite['campaign_id']]['commits'], state['jobs']


def shift_metrics(job, record, phase, phase_ref, source):
    receipt = record['receipt']
    refs = receipt['references']
    identity = receipt['identity']
    require(receipt['normal_completion'] is True and receipt['four_windows'] == 4
            and receipt['phase_config'] == refs['config'] == phase_ref
            and record['dataset_ref'] == refs['episode_manifest'] == source
            and receipt['composite_contract'] == phase['composite_contract']
            and identity['split'] == job['partition'] == 'train'
            and identity['scene_id'] == job['scene_id'] and identity['scene_group'] == job['scene_group']
            and identity['run_index'] == job['run_index'] and identity['recording_id'] == job['recording_id']
            and identity['environment_seed'] == job['experiment_seed']
            and identity['actor_sampling_seed'] == job['actor_sampling_seed'], 'sealed shift identity differs')
    manifest = load_reference(refs['episode_manifest'])
    seal = load_reference(refs['episode_seal'])
    require(manifest['status'] == seal['status'] == 'complete' and manifest['identity'] == identity
            and seal['manifest_sha256'] == refs['episode_manifest']['sha256']
            and seal['hdf5_sha256'] == manifest['hdf5_sha256'] == refs['episode']['sha256']
            and sha(refs['episode']['path']) == refs['episode']['sha256'], 'physical episode seal differs')
    with h5py.File(refs['episode']['path'], 'r') as hdf:
        require(hdf.attrs['status'] == 'complete' and hdf.attrs['schema'] == manifest['schema'] == seal['schema']
                and json.loads(hdf.attrs['identity_json']) == identity
                and json.loads(hdf.attrs['composite_contract_json']) == phase['composite_contract'],
                'physical episode metadata differs')
        physical = json.loads(hdf.attrs['evaluation_json'])
        events = hdf['material_events']
        close = events['kind'][:] == b'WINDOW_CLOSE'
        require(events['window'][:][close].tolist() == [0, 1, 2, 3], 'expected all four chronological windows')
        grades = [float(value) if valid else None for value, valid in zip(
            events['window_grade'][:][close], events['window_grade_valid'][:][close], strict=True)]
    contract = phase['composite_contract']
    ore, waste, wait = (float(physical[key]) for key in ('ore_tons', 'waste_tons', 'total_wait_seconds'))
    require(all(math.isfinite(value) and value >= 0 for value in (ore, waste, wait)), 'invalid physical metrics')
    ore_target, waste_target = contract['ore_target_tons'], contract['waste_target_tons']
    production = (contract['ore_shortfall_weight'] * max(0., 1 - ore / ore_target)
                  + contract['waste_shortfall_weight'] * max(0., 1 - waste / waste_target))
    combined = (math.fsum(grade_cost(grade, empty=grade is None) for grade in grades)
                + production + contract['wait_weight'] * wait / contract['wait_reference_seconds'])
    complete = ore >= ore_target and waste >= waste_target and all(map(_grade_compliant, grades))
    return combined, complete, wait, ore, waste, receipt['transitions']


def batch_metrics(update, phase, phase_ref, commit, records, *, checkpoint=None):
    jobs = phase['jobs']
    runs = [str(job['run_index']) for job in jobs]
    require(len(runs) == len(set(runs)) == plan.BATCH_SIZE and commit['job_ids'] == runs,
            'committed batch does not contain the planned 32 shifts in order')
    checkpoint_ref = commit['checkpoint_ref']
    require(reference(checkpoint_ref['path']) == checkpoint_ref and (checkpoint is None or checkpoint_ref == checkpoint),
            'committed checkpoint differs')
    folder = Path(checkpoint_ref['path']).parent
    prepared = json.loads((folder / 'prepared.json').read_bytes())
    intent = prepared['intent']
    require(prepared['checkpoint'] == checkpoint_ref and intent['update'] == update
            and intent['batch_id'] == commit['batch_id'] and intent['phase_config'] == phase_ref,
            'sealed optimizer transaction differs')
    summary_ref = reference(folder / 'training_summary.json')
    require(prepared.get('training_summary', summary_ref) == summary_ref, 'committed training summary bytes differ')
    summary = load_reference(summary_ref)
    old = 'update' in summary
    learner_metrics = summary['update'] if old else summary['reinforce']
    require(summary['transaction' if old else 'intent'] == intent
            and learner_metrics['episodes'] == len(jobs) and learner_metrics['update_index'] == update,
            'training summary belongs to a different batch')
    sources = intent['sources']
    if not old:
        require(all(item['run_index'] == int(run) and load_reference(item['receipt']) == records[run]['receipt']
                    for run, item in zip(runs, sources, strict=True)), 'optimizer receipt differs from the ledger')
        sources = [item['dataset'] for item in sources]
    physical = [shift_metrics(job, records[run], phase, phase_ref, source)
                for job, run, source in zip(jobs, runs, sources, strict=True)]
    costs, successes, waits, ore, waste, decisions = zip(*physical, strict=True)
    counts = summary['scenario_statistics'].values()
    archived_cost = learner_metrics['metrics']['combined_cost_mean'] if old else learner_metrics['mean_combined_cost']
    require(math.isclose(fmean(costs), archived_cost, rel_tol=1e-10, abs_tol=1e-10)
            and sum(row['shifts'] for row in counts) == len(jobs)
            and sum(row['complete_task'] for row in counts) == sum(successes)
            and learner_metrics['transitions'] == sum(decisions), 'reconstructed metrics differ from the training summary')
    return dict(shifts=len(jobs), mean_j=fmean(costs), complete_task_count=sum(successes),
                complete_task_rate=fmean(successes), mean_wait_vehicle_seconds=fmean(waits),
                mean_ore_tons=fmean(ore), mean_waste_tons=fmean(waste), decisions=sum(decisions),
                actor_optimizer_steps=learner_metrics['actor_optimizer_steps'])


def comparison_rows():
    suite, phases = plan.load_package()
    ppo_suite, ppo_phases, completion, *_ = plan.historical()
    ppo_commits, ppo_records = ledger_snapshot(ppo_suite)
    require(len(ppo_commits) == len(ppo_phases) == len(phases) == plan.UPDATES, 'PPO training archive is incomplete')
    reinforce_commits, reinforce_records = ledger_snapshot(suite) if Path(suite['budget_path']).is_file() else ({}, {})
    references = zip(ppo_suite['phase_configs'][:plan.UPDATES], suite['phase_configs'], strict=True)
    for update, (ppo_phase, phase, (ppo_ref, phase_ref)) in enumerate(zip(ppo_phases, phases, references, strict=True), 1):
        pairs = list(zip(ppo_phase['jobs'], phase['jobs'], strict=True))
        require(len(pairs) == plan.BATCH_SIZE and all(all(left[key] == right[key] for key in
                ('ordinal', 'scene_id', 'runtime_scene_sha256', 'expected_rpc_config')) for left, right in pairs),
                'PPO and REINFORCE do not use the same ordered training scenes')
        ppo = batch_metrics(update, ppo_phase, ppo_ref, ppo_commits[str(update)], ppo_records,
                            checkpoint=completion['training_checkpoints'][update - 1])
        row = dict(update=update, matched_scene_slots=len(pairs), **{f'ppo_{key}': value for key, value in ppo.items()})
        commit = reinforce_commits.get(str(update))
        if commit is not None:
            reinforce = batch_metrics(update, phase, phase_ref, commit, reinforce_records)
            row.update({f'reinforce_{key}': value for key, value in reinforce.items()})
            row.update({f'reinforce_minus_ppo_{key}': reinforce[key] - ppo[key] for key in DELTAS})
        yield row


def main():
    output = plan.REPORT / 'training_comparison.csv'
    rows = list(comparison_rows())
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode='w', encoding='utf-8', newline='', dir=output.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
    print(json.dumps(dict(report=str(output), ppo_batches=len(rows),
                          reinforce_batches=sum('reinforce_shifts' in row for row in rows))))


if __name__ == '__main__':
    main()
