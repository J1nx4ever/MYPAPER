"""Frozen, untrained BG128 encoder traces for two authenticated historical rows.

Only the official state_encoder is forwarded.  No policy/value head, C128 fusion,
optimizer, learner, RPC or AnyLogic is executed. Internal hybrid interventions
are numerical sensitivity probes, not physically valid simulator states.
"""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT))
import numpy as np
import h5py
import torch
from business_graph_summary_v1.models import Actor, MaterialCostCritic, digest
from business_graph_summary_v1.graph import (NODE_NAMES, GROUP_NAMES, RELATION_NAMES,
    EDGE_ATTRIBUTE_NAMES, build_graph)

INPUT = PROJECT / 'reports/token_attention_001/business_graph_summary_009/historical_snapshot_001'
OUT = PROJECT / 'reports/token_attention_001/business_graph_summary_009/encoder_snapshot_001'
SEED = 2026111801
CORE = PROJECT.parents[1] / 'python/episode_budget_ppo.py'
CORE_SHA = '8e4cfa31c622f5e33756e304eea3f7526547db72ae93f7d9f00808845d61ef96'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def matrix_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def vector_fields(vector, prefix='additive'):
    return {f'{prefix}_{i:02d}': float(x) for i, x in enumerate(vector.reshape(-1))}


def change(before, after):
    delta = after - before
    denom = torch.linalg.vector_norm(before) * torch.linalg.vector_norm(after)
    cosine = float(torch.dot(before, after) / denom) if denom > 0 else None
    return dict(l2=float(torch.linalg.vector_norm(delta)), rms=float(delta.square().mean().sqrt()),
                cosine=cosine, before_norm=float(torch.linalg.vector_norm(before)),
                after_norm=float(torch.linalg.vector_norm(after)))


def authenticate():
    manifest = json.loads((INPUT / 'manifest.json').read_text())
    checked = []
    for entry in manifest['source_references'] + manifest['script_references'] + list(manifest['artifacts'].values()):
        got = sha(entry['path'])
        if got != entry['sha256']:
            raise ValueError('historical manifest mismatch: ' + entry['path'])
        checked.append(dict(path=entry['path'], sha256=got))
    if sha(CORE) != CORE_SHA:
        raise ValueError('protected PPO SHA differs')
    snapshots = json.loads((INPUT / 'snapshots.json').read_text())
    selection = snapshots['selection']
    if selection['T1']['state_index'] != selection['T0']['state_index'] + 1:
        raise ValueError('rows are not adjacent')
    # The original HDF is independently authenticated above and here.
    if sha(selection['source_hdf5']) != selection['source_sha256']:
        raise ValueError('raw HDF SHA differs')
    with h5py.File(selection['source_hdf5'], 'r') as hdf:
        for t in ('T0', 'T1'):
            original = hdf['states'][selection[t]['state_index']]
            serialized = np.asarray(snapshots[t]['raw'], dtype=np.float32)
            if original.dtype != np.float32 or not np.array_equal(original, serialized):
                raise ValueError('serialized snapshot differs from original HDF row ' + t)
    return snapshots, checked


def layer_forward(encoder, h0, graph):
    h1 = encoder.graph_layers[0](h0, graph)
    return encoder.graph_layers[1](h1, graph)


def three_factor(encoder, h0, graphs, target):
    """Exact 3-player Shapley of output H2[target], using eight tensor hybrids."""
    cells = {}
    for own, other, g in itertools.product((0, 1), repeat=3):
        mixed = h0[other].clone()
        mixed[:, target] = h0[own][:, target]
        cells[(own, other, g)] = layer_forward(encoder, mixed, graphs[g])[0, target]
    contributions = []
    for factor in range(3):
        phi = torch.zeros(64)
        others = [i for i in range(3) if i != factor]
        for values in itertools.product((0, 1), repeat=2):
            low = [0, 0, 0]
            for i, value in zip(others, values):
                low[i] = value
            high = low.copy()
            high[factor] = 1
            count = sum(values)
            weight = math.factorial(count) * math.factorial(2 - count) / math.factorial(3)
            phi += weight * (cells[tuple(high)] - cells[tuple(low)])
        contributions.append(phi)
    actual = cells[(1, 1, 1)] - cells[(0, 0, 0)]
    closure = sum(contributions) - actual
    return cells, contributions, actual, closure


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    OUT.mkdir(parents=True, exist_ok=True)
    snapshots, authenticated = authenticate()
    states = {t: torch.tensor([snapshots[t]['raw']], dtype=torch.float32) for t in ('T0', 'T1')}
    graphs = {t: build_graph(states[t]) for t in states}
    representations, attribute_vectors, message_tensors = {}, {}, {}
    node_rows, change_rows, edge_rows, group_rows, evidence_rows, attribution_rows = [], [], [], [], [], []
    output = dict(schema='bg128-encoder-snapshot-trace-v1', pair_seed=SEED,
        parameter_status='fixed named-paired initialization, numerical channel diagnostic; no trained BG128 checkpoint loaded',
        scope='official state_encoder forward only; no Actor/Critic head or C128 neural fusion forward',
        train=False, optimizer_updates=0, rpc_started=False, anylogic_started=False,
        model_roles_independent=True, selection=snapshots['selection'], roles={},
        caveats=[
            'H0 contains deterministic quantities and encoded references/lists/path content; own_H0 is not pure own raw state.',
            'Graph interventions include topology, relation/group and edge attributes; graph quantities are not re-encoded into the fixed hybrid H0.',
            'Signed projection shares are vector projections onto observed delta; norms do not form percentages.',
            'Tensor hybrids need not correspond to physically reachable states.',
            'Edge additive vectors are exact only before LayerNorm; attention is not learned importance.',
            'C128 remains outside state_encoder and is unchanged by these encoder-only forwards.',
        ])
    max_projection_error = 0.
    max_shapley_error = 0.
    for role, cls in (('actor', Actor), ('critic', MaterialCostCritic)):
        model = cls(pair_seed=SEED).eval()
        encoder = model.state_encoder
        before = digest(model)
        role_data = dict(parameter_sha256_before=before, snapshots={}, load_changes=[], attribution=[])
        details_by_time = {}
        with torch.no_grad():
            for t, state in states.items():
                raw_readout, details = encoder(state, diagnostic=True)
                details_by_time[t] = details
                role_data['snapshots'][t] = dict(matrix_sha256={}, node_count=19,
                    edge_count=int(details['graph'].edge_index.shape[1]),
                    candidate_dump_ids=details['candidate_dump_ids'][0].tolist())
                for number, key in ((0, 'z0'), (1, 'h1'), (2, 'h2')):
                    arr = details[key][0].numpy().copy()
                    representations[f'{role}_{t}_H{number}'] = arr
                    role_data['snapshots'][t]['matrix_sha256'][f'H{number}'] = matrix_sha(arr)
                    for nid, name in enumerate(NODE_NAMES):
                        node_rows.append(dict(role=role, snapshot=t, layer=number,
                            node_id=nid, node=name, **vector_fields(arr[nid], 'h')))
                graph = details['graph']
                seen = Counter()
                stable_keys = []
                for ei in range(len(graph.edge_type)):
                    src, dst = graph.edge_index[:, ei].tolist()
                    rel = RELATION_NAMES[int(graph.edge_type[ei])]
                    attr = graph.edge_attr[ei]
                    pos = round(float(attr[12]) * 9) if attr[13] else '-'
                    base = f'{rel}|{src}|{dst}|slot={pos}'
                    instance = seen[base]
                    seen[base] += 1
                    stable_keys.append(base + f'|instance={instance}')
                for ln in (1, 2):
                    ld = details[f'layer{ln}']
                    layer = encoder.graph_layers[ln - 1]
                    edge_sum = torch.zeros((19, 64))
                    edge_projected = []
                    for ei in range(len(graph.edge_type)):
                        src, dst = graph.edge_index[:, ei].tolist()
                        gid = int(graph.edge_group[ei])
                        message = ld['messages'][ei].flatten()
                        weighted = (ld['messages'][ei] * ld['weights'][ei, :, None]).flatten()
                        projected = layer.output.weight[:, gid * 64:(gid + 1) * 64] @ weighted
                        edge_sum[dst] += projected
                        edge_projected.append(projected)
                        row = dict(role=role, snapshot=t, layer=ln, edge_index=ei,
                            stable_key=stable_keys[ei], source=NODE_NAMES[src], source_id=src,
                            destination=NODE_NAMES[dst], destination_id=dst,
                            relation=RELATION_NAMES[int(graph.edge_type[ei])], group=GROUP_NAMES[gid],
                            **{f'alpha{head}': float(ld['weights'][ei, head]) for head in range(4)},
                            message_norm=float(torch.linalg.vector_norm(message)),
                            weighted_message_norm=float(torch.linalg.vector_norm(weighted)),
                            projected_additive_norm=float(torch.linalg.vector_norm(projected)),
                            **{name: float(graph.edge_attr[ei, k]) for k, name in enumerate(EDGE_ATTRIBUTE_NAMES)},
                            **vector_fields(projected))
                        edge_rows.append(row)
                        if (ln == 1 and src < 9 and 13 <= dst < 15) or (ln == 2 and 13 <= src < 15 and 9 <= dst < 13):
                            evidence_rows.append(row.copy())
                    presence = ld['group_masks'][0].float() @ layer.output.weight[:, 256:].T
                    prefix = f'{role}_{t}_layer{ln}'
                    message_tensors[prefix + '_edge_projected'] = torch.stack(edge_projected).numpy()
                    message_tensors[prefix + '_weights'] = ld['weights'].numpy()
                    message_tensors[prefix + '_messages'] = ld['messages'].numpy()
                    message_tensors[prefix + '_presence_projected'] = presence.numpy()
                    message_tensors[prefix + '_projected'] = ld['projected'][0].numpy()
                    message_tensors[prefix + '_pre_norm'] = ld['pre_norm'][0].numpy()
                    error = float((edge_sum + presence - ld['projected'][0]).abs().max())
                    max_projection_error = max(max_projection_error, error)
                    for nid, name in enumerate(NODE_NAMES):
                        for gid, group in enumerate(GROUP_NAMES):
                            raw = ld['group_messages'][0, nid, gid]
                            projected = layer.output.weight[:, gid * 64:(gid + 1) * 64] @ raw
                            bit_projected = layer.output.weight[:, 256 + gid] * ld['group_masks'][0, nid, gid]
                            group_rows.append(dict(role=role, snapshot=t, layer=ln,
                                node=name, node_id=nid, group=group,
                                incoming_count=int(ld['group_counts'][0, nid, gid]),
                                present=bool(ld['group_masks'][0, nid, gid]),
                                aggregate_norm=float(torch.linalg.vector_norm(raw)),
                                projected_norm=float(torch.linalg.vector_norm(projected)),
                                presence_projected_norm=float(torch.linalg.vector_norm(bit_projected)),
                                **vector_fields(projected), **vector_fields(bit_projected, 'presence')))
            for ln, key in ((0, 'z0'), (1, 'h1'), (2, 'h2')):
                for nid, name in enumerate(NODE_NAMES):
                    metrics = change(details_by_time['T0'][key][0, nid], details_by_time['T1'][key][0, nid])
                    row = dict(role=role, layer=ln, node=name, node_id=nid, **metrics)
                    change_rows.append(row)
                    if ln == 2 and 9 <= nid < 13:
                        role_data['load_changes'].append(row)
            h0 = [details_by_time[t]['z0'] for t in ('T0', 'T1')]
            graph_pair = [details_by_time[t]['graph'] for t in ('T0', 'T1')]
            for nid in range(9, 13):
                cells, components, delta, closure = three_factor(encoder, h0, graph_pair, nid)
                name = NODE_NAMES[nid]
                norm2 = float(delta.square().sum())
                actual_endpoint_error = float((cells[(0, 0, 0)] - details_by_time['T0']['h2'][0, nid]).abs().max())
                actual_endpoint_error = max(actual_endpoint_error, float((cells[(1, 1, 1)] - details_by_time['T1']['h2'][0, nid]).abs().max()))
                max_shapley_error = max(max_shapley_error, float(closure.abs().max()), actual_endpoint_error)
                attribute_vectors[f'{role}_{name}_actual_delta'] = delta.numpy()
                attribute_vectors[f'{role}_{name}_closure'] = closure.numpy()
                for factor, vector in zip(('own_H0', 'other_H0', 'graph'), components):
                    attribute_vectors[f'{role}_{name}_{factor}'] = vector.numpy()
                    row = dict(role=role, node=name, node_id=nid, factor=factor,
                        component_l2=float(torch.linalg.vector_norm(vector)),
                        delta_l2=norm2 ** .5,
                        signed_delta_projection_share=float(torch.dot(vector, delta)) / norm2 if norm2 else None,
                        closure_max_abs=float(closure.abs().max()),
                        endpoint_max_abs_error=actual_endpoint_error,
                        **vector_fields(vector, 'contribution'))
                    attribution_rows.append(row)
                    role_data['attribution'].append({k: v for k, v in row.items() if not k.startswith('contribution_')})
                for bits, vector in cells.items():
                    attribute_vectors[f'{role}_{name}_cell_{bits[0]}{bits[1]}{bits[2]}'] = vector.numpy()
        after = digest(model)
        if after != before:
            raise ValueError('parameters changed during frozen forward')
        role_data['parameter_sha256_after'] = after
        role_data['load_changes'].sort(key=lambda r: r['l2'], reverse=True)
        role_data['message_evidence'] = [
            {k: v for k, v in row.items() if not k.startswith('additive_')}
            for row in evidence_rows if row['role'] == role]
        role_data['global_bypass_exact'] = all(
            torch.equal(details_by_time[t]['z0'][:, 18], details_by_time[t]['h1'][:, 18]) and
            torch.equal(details_by_time[t]['z0'][:, 18], details_by_time[t]['h2'][:, 18])
            for t in ('T0', 'T1'))
        output['roles'][role] = role_data
    if max_projection_error > 3e-6 or max_shapley_error > 3e-6:
        raise ValueError(f'closure failed: projection={max_projection_error}, Shapley={max_shapley_error}')
    if sha(CORE) != CORE_SHA:
        raise ValueError('protected PPO changed')
    np.savez_compressed(OUT / 'representations.npz', **representations)
    np.savez_compressed(OUT / 'attribution_vectors.npz', **attribute_vectors)
    np.savez_compressed(OUT / 'message_tensors.npz', **message_tensors)
    for name, rows in (('node_representations', node_rows), ('representation_changes', change_rows),
                       ('edge_messages', edge_rows), ('group_messages', group_rows),
                       ('message_evidence', evidence_rows), ('hybrid_attribution', attribution_rows)):
        write_csv(OUT / (name + '.csv'), rows)
    output['validation'] = dict(authenticated_reference_count=len(authenticated),
        input_manifest_sha256=sha(INPUT / 'manifest.json'),
        raw_snapshot_sha256=sha(INPUT / 'snapshots.json'), protected_ppo_sha256=CORE_SHA,
        additive_projected_max_abs_error=max_projection_error,
        shapley_and_endpoint_max_abs_error=max_shapley_error,
        edge_rows=len(edge_rows), group_rows=len(group_rows), message_evidence_rows=len(evidence_rows),
        no_optimization=True, no_checkpoint_loaded=True, forward_training_flags=False)
    output['validation']['serialized_raw_rows_exact_original_hdf'] = True
    output['source_references'] = [dict(path=str(p), sha256=sha(p))
        for p in sorted((PROJECT / 'business_graph_summary_v1').glob('*.py'))]
    output['source_references'].append(dict(path=str(Path(__file__).resolve()), sha256=sha(__file__)))
    output['npz_schema'] = {
        'representations.npz': {k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k, v in representations.items()},
        'attribution_vectors.npz': {k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k, v in attribute_vectors.items()},
        'message_tensors.npz': {k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k, v in message_tensors.items()},
    }
    write_json(OUT / 'authenticated_inputs.json', authenticated)
    write_json(OUT / 'summary.json', output)
    print(json.dumps(dict(out=str(OUT), validation=output['validation'],
        load_changes={role: data['load_changes'] for role, data in output['roles'].items()}), ensure_ascii=False))


if __name__ == '__main__':
    main()
