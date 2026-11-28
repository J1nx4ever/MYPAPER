"""Read-only FIFO-order sensitivity of trained Full/u32 actors on common factual states."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import torch

PROJECT = next(p for p in Path(__file__).resolve().parents if p.name == 'lagrangian_ppo_v1')
sys.path[:0] = [str(PROJECT), str(PROJECT.parents[1])]
from business_graph_summary_v1.admission import frozen_actor
from business_graph_summary_v1.contract import feature_config_for
from business_graph_summary_v1.c_features import FEATURE_NAMES, build_features
from business_graph_summary_v1.features_cache import pack_summary_prefixes
from business_graph_summary_v1.models import digest

REPORT = PROJECT / 'reports/token_attention_001/business_graph_summary_009'
INDEX = REPORT / 'gat_weak_ablation_001/seed_extension_001/paired_evaluation_summary.json'
INDEX_SHA = '3d7483da4ac12919db37108a1d8992f4ec111faed00530ff12c33a024bdc9dcb'
OUT = REPORT / 'delayed_realization_001/frozen_fifo_probe.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(ref):
    assert sha(ref['path']) == ref['sha256'], ref
    return json.loads(Path(ref['path']).read_bytes())


def phase(source, scenario):
    completion = read(source['full_completion'])
    assert completion['training_updates'] == 32 and completion['new_checkpoint'] == source['full_checkpoint']
    match, = [r for r in completion['phase_results'] if Path(r['report']['path']).parts[-3:-1] == (scenario, 'new')]
    data = read(match['report'])
    assert data['checkpoint_provenance'] == source['full_checkpoint'] and data['evaluation_optimizer_updates'] == 0
    return data, Path(match['report']['path']).parent


def transition_column(h5, key):
    segments = []
    for _, group in sorted(h5['transitions'].items()):
        column = json.loads(bytes(group['schema'][:]))['fields'][key]['column']
        assert column['width'] == 1
        segments.append(group[column['dtype']][:, column['offset']])
    return np.concatenate(segments)


def collect(index):
    records, states, alternative = [], [], []
    counts = Counter()
    inputs = []
    config = feature_config_for('s01_BG128')
    for scenario in index['scopes']['held_out']:
        report, directory = phase(index['sources']['s01'], scenario)
        # Equal number of preselected physical slots per held-out condition, independent of outcome.
        for ordinal, episode in enumerate(report['per_episode'][:10]):
            identity = episode['identity']
            run = identity['run_index']
            raw = directory / 'episodes' / f'run_{run}'
            manifest = read(dict(path=str(raw / 'manifest.json'), sha256=identity['raw_manifest_sha256']))
            raw_file = raw / 'samples.h5'
            file_sha = manifest['files_sha256']['samples.h5']
            assert sha(raw_file) == file_sha and manifest['source_kind'] == 'real_anylogic_rpc'
            inputs.append(dict(run=run, scenario=scenario, slot=ordinal, raw_sha256=file_sha))
            with h5py.File(raw_file) as f:
                rows = f['states'][:-1]
                assert np.array_equal(f['state_index'][:], np.arange(len(rows)))
                times = transition_column(f, 'action_time_seconds')
                actions = transition_column(f, 'action')
                logs = transition_column(f, 'old_log_prob')
            assert len(rows) == manifest['transitions']
            assert np.allclose(times, rows[:, 0].astype(np.float64)*28800, rtol=0, atol=.002)
            for boundary in (7200, 14400, 21600):
                counts['shift_boundaries'] += 1
                # Select the latest qualifying decision in the final 600 s, BEFORE computing
                # any alternative summary, action or outcome. Keep invariant projections as controls.
                use = None
                for j in np.flatnonzero((times >= boundary-600) & (times < boundary))[::-1]:
                    state = rows[j]
                    fifo = state[1391:1439].reshape(16, 3)
                    valid = np.flatnonzero((fifo[:, 0] == 1) & (fifo[:, 1] > 0))
                    if len(valid) >= 2 and state[19:23].sum() >= 2:
                        i1, i2 = valid[:2]
                        if fifo[i1, 2] != fifo[i2, 2]:
                            use = j, state, i1, i2
                            break
                if use is None:
                    continue
                j, state, i1, i2 = use
                fifo = state[1391:1439].reshape(16, 3)
                changed = state.copy()
                alt_fifo = changed[1391:1439].reshape(16, 3)
                alt_fifo[[i1, i2]] = fifo[[i2, i1]]
                fact_c = build_features(state, config=config)
                alt_c = build_features(changed, config=config)
                assert fact_c[19] > 0 and fact_c[21] > 0
                assert np.array_equal(fact_c[:44], alt_c[:44]) and np.array_equal(fact_c[88:], alt_c[88:])
                diff = np.flatnonzero(fact_c != alt_c)
                assert all(FEATURE_NAMES[k].startswith(('current.', 'next.')) for k in diff)
                states.append(state)
                alternative.append(alt_c)
                records.append(dict(run=run, scenario=scenario, slot=ordinal, decision=int(j), boundary=boundary,
                    seconds_before=boundary-float(times[j]), parcels_tons=(fifo[[i1,i2],1]*500).tolist(),
                    historical_action=int(actions[j]), historical_action_log_probability=float(logs[j]),
                    factual_actor=manifest['policy_version'],
                    grades=fifo[[i1,i2],2].tolist(), changed_C=[FEATURE_NAMES[k] for k in diff],
                    nominal_grade_current_before_after=[float(fact_c[51]), float(alt_c[51])],
                    nominal_grade_next_before_after=[float(fact_c[73]), float(alt_c[73])]))
    return np.stack(states), np.stack(alternative), records, counts, inputs


def main():
    index = read(dict(path=str(INDEX), sha256=INDEX_SHA))
    torch.set_num_threads(1)
    states, alternatives, records, counts, inputs = collect(index)
    summaries = {}
    for seed, source in index['sources'].items():
        report, _ = phase(source, 'held_order_1')
        actor_ref = report['actor_artifact']
        assert sha(actor_ref['path']) == actor_ref['sha256']
        actor, fingerprint = frozen_actor(actor_ref['path'])
        with torch.inference_mode():
            for i in range(0, len(records), 32):
                chunk = states[i:i+32]
                tensor = torch.from_numpy(chunk)
                prefix = pack_summary_prefixes(chunk, feature_config=actor.feature_config)
                factual = actor(tensor, prefix)
                # Inject ONLY the alternate explicit task summary; the exact same raw observation
                # supplies the entire learned graph, readout, quantities, action C40 and mask.
                alt = torch.tensor(alternatives[i:i+len(chunk)])[:, :88]
                hook = actor.summary_task_encoder.register_forward_pre_hook(lambda mod, args: (alt,))
                try:
                    intervention = actor(tensor, prefix)
                finally:
                    hook.remove()
                assert torch.equal(factual, actor(tensor, prefix))
                probabilities = torch.softmax(factual, -1)
                new_probs = torch.softmax(intervention, -1)
                for record, p, q in zip(records[i:i+len(chunk)], probabilities, new_probs, strict=True):
                    if seed == 's01':
                        assert record['factual_actor'] == fingerprint
                        error = abs(float(torch.log(p[record['historical_action']])) - record['historical_action_log_probability'])
                        assert error < 1e-5, (record['run'], record['decision'], error)
                        record['factual_replay_log_probability_absolute_error'] = error
                    if not record['changed_C']:
                        assert torch.equal(p, q), (record['run'],record['decision'],(p-q).abs().max().item())
                    record.setdefault('outputs', {})[seed] = dict(tv=float((p-q).abs().sum()/2),
                        probabilities=p.tolist(), swapped_order_probabilities=q.tolist(),
                        argmax_changed=bool(p.argmax() != q.argmax()))
        assert digest(actor) == fingerprint
        responsive = [r['outputs'][seed] for r in records if r['changed_C']]
        tv = np.array([x['tv'] for x in responsive])
        summaries[seed] = dict(checkpoint=source['full_checkpoint'], actor=actor_ref, fingerprint=fingerprint,
            states_with_changed_projection=len(tv), nonzero_tv=int((tv > 1e-7).sum()),
            median_tv=float(np.median(tv)), mean_tv=float(tv.mean()), max_tv=float(tv.max()),
            argmax_changes=sum(x['argmax_changed'] for x in responsive))
    counts.update(selected_states=len(records), states_with_changed_projection=sum(bool(x['changed_C']) for x in records))
    result = dict(source_index_sha256=INDEX_SHA, source_script_sha256=sha(__file__), physical_runs_added=0,
        training_updates=0, selection='Full s01, first 10 slots per held-out case; closest eligible decision in last 600s of each internal cutoff. Two positive unequal-grade FIFO packets and >=2 legal actions.',
        intervention='Reverse first two positive packets including remaining mass to recompute certified C128. Feed only changed current/next projections through the task-summary channel. Entire graph, action summary, current facts, pool totals, mask and model weights stay factual. Synthetic feature sensitivity, no physical counterfactual.',
        counts=counts, summaries=summaries, inputs=inputs, observations=records)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)+'\n')
    print(counts)
    for seed, summary in summaries.items():
        print(seed, {k:v for k,v in summary.items() if k not in ('actor','checkpoint','fingerprint')})


if __name__ == '__main__':
    main()
