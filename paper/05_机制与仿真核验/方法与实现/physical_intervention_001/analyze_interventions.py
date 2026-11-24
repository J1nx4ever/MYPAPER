"""Complete raw -> graph/C128 -> encoder -> readout diagnostics, no training.

All intervention inputs are independently built from a common raw baseline by
build_scenarios.py. No hidden-state replacement or graph editing is used here.
"""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT))
import numpy as np
import torch
from business_graph_summary_v1.models import Actor, MaterialCostCritic, digest, pack_summary_prefixes
from business_graph_summary_v1.c_features import build_features, FEATURE_NAMES, feature_contract
from business_graph_summary_v1.graph import (build_graph, NODE_NAMES, RELATION_NAMES,
    GROUP_NAMES, EDGE_ATTRIBUTE_NAMES, QUANTITY_NAMES, QUANTITY_SCALES, ROAD_ENDPOINTS)
from relation_gat_ego_v1.adapter import validate_state

OUT = PROJECT / 'reports/token_attention_001/business_graph_summary_009/physical_intervention_001'
SEED = 2026111801
CORE = PROJECT.parents[1] / 'python/episode_budget_ppo.py'
CORE_SHA = '8e4cfa31c622f5e33756e304eea3f7526547db72ae93f7d9f00808845d61ef96'
SCENES = ('BASE', 'A', 'B', 'C')
CANDIDATE_BLOCKS = (
    ('ego_H2', 0, 64), ('load_H2', 64, 128), ('dump_H2', 128, 192),
    ('ego_quantities', 192, 213), ('load_quantities', 213, 234),
    ('dump_quantities', 234, 255), ('mask', 255, 256), ('dispatch_tons_scaled', 256, 257),
    ('C_action_embedding', 257, 273),
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def dump_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def changes(x, y):
    delta = y.astype(np.float64) - x.astype(np.float64)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return dict(l2=float(np.linalg.norm(delta)), rms=float(np.sqrt(np.mean(delta ** 2))),
        max_abs=float(np.abs(delta).max()),
        cosine=float(np.dot(x.reshape(-1), y.reshape(-1)) / denom) if denom > 0 else None)


def edge_rows(graph, scene):
    occurrence = Counter()
    rows = []
    for i in range(len(graph.edge_type)):
        src, dst = (int(v) for v in graph.edge_index[:, i])
        relation = RELATION_NAMES[int(graph.edge_type[i])]
        attr = graph.edge_attr[i].numpy()
        position = float(attr[EDGE_ATTRIBUTE_NAMES.index('position_9')])
        key = (src, dst, relation, position)
        instance = occurrence[key]
        occurrence[key] += 1
        rows.append(dict(scene=scene, edge_index=i,
            stable_key=f'{src}|{dst}|{relation}|position={position:g}|instance={instance}',
            source=NODE_NAMES[src], source_id=src, destination=NODE_NAMES[dst], destination_id=dst,
            relation=relation, group=GROUP_NAMES[int(graph.edge_group[i])],
            **{name: float(v) for name, v in zip(EDGE_ATTRIBUTE_NAMES, attr)}))
    return rows


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    assert sha(CORE) == CORE_SHA
    scenarios_path = OUT / 'scenarios.json'
    bundle = json.loads(scenarios_path.read_text())
    inputs_sha = sha(scenarios_path)
    raw = {'BASE': np.asarray(bundle['base']['raw'], dtype=np.float32)}
    raw.update({key: np.asarray(bundle['scenarios'][key]['raw'], dtype=np.float32) for key in SCENES[1:]})
    assert all(value.shape == (1519,) for value in raw.values())
    for value in raw.values():
        validate_state(value)
    # Same clock, decision identity, episode/terminal and material definitions.
    for scene in SCENES[1:]:
        assert raw[scene][0] == raw['BASE'][0]
        assert np.array_equal(raw[scene][27:1107].reshape(9,120)[:,0], raw['BASE'][27:1107].reshape(9,120)[:,0])
        assert raw[scene][17] == raw['BASE'][17] == 0
    tensors = {scene: torch.from_numpy(value[None].copy()) for scene, value in raw.items()}
    graphs = {scene: build_graph(value) for scene, value in tensors.items()}
    arrays = {}
    edges, quantities, edge_diffs = [], [], []
    graph_summary = {}
    by_scene_edges = {}
    for scene, graph in graphs.items():
        rows = edge_rows(graph, scene)
        by_scene_edges[scene] = {r['stable_key']: r for r in rows}
        edges.extend(rows)
        arrays[f'{scene}_raw'] = raw[scene]
        arrays[f'{scene}_quantities'] = graph.quantities[0].numpy()
        arrays[f'{scene}_raw_quantities'] = graph.raw_quantities[0].numpy()
        arrays[f'{scene}_edge_index'] = graph.edge_index.numpy()
        arrays[f'{scene}_edge_type'] = graph.edge_type.numpy()
        arrays[f'{scene}_edge_attr'] = graph.edge_attr.numpy()
        for i, name in enumerate(NODE_NAMES):
            quantities.append(dict(scene=scene, node=name, node_id=i,
                **{qname: float(v) for qname, v in zip(QUANTITY_NAMES, graph.raw_quantities[0,i])}))
        counts = Counter(r['relation'] for r in rows)
        assert counts['road'] == 16
        graph_summary[scene] = dict(node_count=19, edge_count=len(rows),
            relation_counts=dict(counts), candidate_dump_ids=graph.candidate_dump_ids[0].tolist(),
            ego_id=int(graph.ego_ids[0]), raw_sha256=array_sha(raw[scene]))
    for scene in SCENES[1:]:
        before, after = by_scene_edges['BASE'], by_scene_edges[scene]
        counts = Counter()
        for key in sorted(set(before) | set(after)):
            left, right = before.get(key), after.get(key)
            fields = [name for name in EDGE_ATTRIBUTE_NAMES if left is not None and right is not None and left[name] != right[name]]
            kind = 'added' if left is None else 'removed' if right is None else 'attributes_changed' if fields else 'unchanged'
            counts[kind] += 1
            row = right if right is not None else left
            edge_diffs.append(dict(scene=scene, status=kind, stable_key=key,
                source=row['source'], destination=row['destination'], relation=row['relation'],
                group=row['group'], changed_attributes=';'.join(fields),
                before_attributes=json.dumps({f: left[f] for f in EDGE_ATTRIBUTE_NAMES}) if left else '',
                after_attributes=json.dumps({f: right[f] for f in EDGE_ATTRIBUTE_NAMES}) if right else ''))
        graph_summary[scene]['edge_diff_vs_base'] = dict(counts)
    write_csv(OUT / 'graph_edges.csv', edges)
    write_csv(OUT / 'edge_diffs.csv', edge_diffs)
    write_csv(OUT / 'quantities.csv', quantities)

    roles = {}
    node_rows, node_changes, candidate_rows, block_rows, logit_rows, pairwise_rows, feature_rows = [], [], [], [], [], [], []
    for role, cls in (('actor', Actor), ('critic', MaterialCostCritic)):
        model = cls(pair_seed=SEED).eval()
        before = digest(model)
        details = {}
        per_scene = {}
        with torch.no_grad():
            for scene in SCENES:
                # A new certificate binds every newly constructed raw observation.
                prefix = pack_summary_prefixes(raw[scene][None].copy(), feature_config=model.feature_config)
                direct_c = build_features(raw[scene], config=model.feature_config)
                assert np.array_equal(prefix.features[0], direct_c)
                output, detail = model(tensors[scene], prefix, diagnostic=True)
                ordinary = model(tensors[scene], prefix)
                assert torch.equal(output, ordinary)
                details[scene] = detail
                assert not model.training
                current = detail['current_summary'][0].numpy()
                if role == 'actor':
                    arrays[f'{scene}_C128'] = current
                    for i, name in enumerate(FEATURE_NAMES):
                        feature_rows.append(dict(scene=scene, feature_index=i, feature=name,
                            value=float(current[i]), base=float(arrays['BASE_C128'][i]),
                            delta=float(current[i] - arrays['BASE_C128'][i])))
                else:
                    assert np.array_equal(arrays[f'{scene}_C128'], current)
                for k, dkey in enumerate(('z0', 'h1', 'h2')):
                    value = detail[dkey][0].numpy()
                    assert value.shape == (19,64) and np.isfinite(value).all()
                    arrays[f'{role}_{scene}_H{k}'] = value
                    for node_id, node in enumerate(NODE_NAMES):
                        node_rows.append(dict(role=role, scene=scene, layer=k, node=node, node_id=node_id,
                            **{f'h{i:02d}': float(v) for i, v in enumerate(value[node_id])}))
                assert torch.equal(detail['z0'][:,18], detail['h1'][:,18])
                assert torch.equal(detail['z0'][:,18], detail['h2'][:,18])
                assert torch.equal(detail['graph'].quantities, graphs[scene].quantities)
                per_scene[scene] = dict(summary_certificate=prefix.certificate_sha256,
                    c128_sha256=array_sha(current),
                    matrix_sha256={f'H{k}': array_sha(arrays[f'{role}_{scene}_H{k}']) for k in range(3)})
                if role == 'actor':
                    for key in ('candidate_input', 'candidates', 'context_input', 'context',
                                'task_summary', 'action_summary', 'unmasked_logits', 'masked_logits'):
                        arrays[f'actor_{scene}_{key}'] = detail[key][0].numpy()
                    probabilities = torch.softmax(output, dim=-1)[0].numpy()
                    arrays[f'actor_{scene}_probs'] = probabilities
                    assert np.isfinite(probabilities).all() and np.isclose(probabilities.sum(),1)
                    candidate_input = arrays[f'actor_{scene}_candidate_input']
                    candidates = arrays[f'actor_{scene}_candidates']
                    assert candidate_input.shape == (4,273) and candidates.shape == (4,64)
                    for action in range(4):
                        candidate_rows.append(dict(scene=scene, action=action, load=f'Load{action}',
                            **{f'h{i:02d}':float(v) for i,v in enumerate(candidates[action])}))
                        unmasked = float(detail['unmasked_logits'][0,action])
                        masked = float(output[0,action])
                        legal = bool(raw[scene][19+action])
                        assert np.isfinite(unmasked) and (np.isfinite(masked) if legal else np.isneginf(masked))
                        logit_rows.append(dict(scene=scene, action=action, load=f'Load{action}', legal=legal,
                            dispatch_tons=float(raw[scene][23+action]*193), unmasked_logit=unmasked,
                            masked_logit=masked if legal else '-inf', probability=float(probabilities[action]),
                            delta_unmasked_logit=unmasked-float(arrays['actor_BASE_unmasked_logits'][action]),
                            delta_probability=float(probabilities[action]-arrays['actor_BASE_probs'][action])))
                    for a in range(4):
                        for b in range(a+1,4):
                            value = float(detail['unmasked_logits'][0,a]-detail['unmasked_logits'][0,b])
                            base = float(arrays['actor_BASE_unmasked_logits'][a]-arrays['actor_BASE_unmasked_logits'][b])
                            pairwise_rows.append(dict(scene=scene, action_a=a, action_b=b,
                                both_legal=bool(raw[scene][19+a] and raw[scene][19+b]),
                                logit_difference=value, base_logit_difference=base, delta_difference=value-base))
                    per_scene[scene].update(unmasked_logits=detail['unmasked_logits'][0].tolist(),
                        masked_logits=[float(v) if raw[scene][19+i] else None for i,v in enumerate(output[0])],
                        probabilities=probabilities.tolist(), legal_mask=raw[scene][19:23].astype(int).tolist())
                else:
                    arrays[f'critic_{scene}_value'] = output.numpy()
                    per_scene[scene]['value'] = float(output[0])
            for scene in SCENES[1:]:
                for k in range(3):
                    for node_id, node in enumerate(NODE_NAMES):
                        node_changes.append(dict(role=role, scene=scene, layer=k, node=node, node_id=node_id,
                            **changes(arrays[f'{role}_BASE_H{k}'][node_id], arrays[f'{role}_{scene}_H{k}'][node_id])))
                if role == 'actor':
                    for action in range(4):
                        for key in ('candidates', 'candidate_input'):
                            block_rows.append(dict(scene=scene, action=action, load=f'Load{action}', block=key,
                                **changes(arrays[f'actor_BASE_{key}'][action], arrays[f'actor_{scene}_{key}'][action])))
                        for name, lo, hi in CANDIDATE_BLOCKS:
                            block_rows.append(dict(scene=scene, action=action, load=f'Load{action}', block=name,
                                **changes(arrays['actor_BASE_candidate_input'][action,lo:hi], arrays[f'actor_{scene}_candidate_input'][action,lo:hi])))
            after = digest(model)
            assert before == after
        roles[role] = dict(parameter_sha256_before=before, parameter_sha256_after=after,
            parameters_require_no_grad_updates=True, feature_contract=feature_contract(model.feature_config),
            scenes=per_scene)

    np.savez_compressed(OUT / 'analysis_tensors.npz', **arrays)
    for name, rows in (('node_representations',node_rows), ('representation_changes',node_changes),
                       ('actor_candidates',candidate_rows), ('candidate_changes',block_rows),
                       ('actor_logits',logit_rows), ('logit_pairwise_differences',pairwise_rows),
                       ('c128_features',feature_rows)):
        write_csv(OUT / f'{name}.csv',rows)
    assert sha(CORE) == CORE_SHA and sha(scenarios_path) == inputs_sha
    summary = dict(schema='bg128-raw-physical-interventions-v1', pair_seed=SEED,
        parameter_status='fixed random named-paired initialization; untrained; no checkpoint loaded',
        primary_method='independent common-baseline raw edits followed by complete official graph, quantities, token/C128 and Actor/Critic forward; no hidden tensor or graph replacements',
        execution=dict(optimizer_updates=0, rpc_started=False, anylogic_started=False,
            new_physical_shifts=0, budget_registrations=0, final_actor_critic_readout_forward=True),
        base=bundle['base'] | {'raw':'see scenarios.json and analysis_tensors.npz'},
        graph=graph_summary, roles=roles,
        candidate_input_blocks=[dict(name=n,start=lo,stop=hi) for n,lo,hi in CANDIDATE_BLOCKS],
        source_references=[dict(path=str(scenarios_path),sha256=inputs_sha),
            dict(path=str(CORE),sha256=CORE_SHA)] + [dict(path=str(p),sha256=sha(p)) for p in sorted((PROJECT/'business_graph_summary_v1').glob('*.py'))],
        npz_schema={k:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=array_sha(v)) for k,v in arrays.items()},
        interpretation_limits=[
            'Observation-consistent synthetic interventions do not prove reachability from a complete AnyLogic internal state.',
            'Full raw interventions change multiple related encoder and summary paths; H0 can already contain neighbor references.',
            'Untrained logit/probability/value changes are numerical sensitivity, not business correctness or performance.',
            'Common graph pooling and C128 can change all action logits; no zero-response requirement for unrelated Load logits.',
            'Only primary complete raw forwards are compared; no hidden tensor replacement is used.',
        ])
    dump_json(OUT / 'summary.json', summary)
    print(json.dumps(dict(status='PASS',seed=SEED,scenes=list(SCENES),
        graph_counts={s:graph_summary[s]['edge_count'] for s in SCENES},
        actor_logits={s:roles['actor']['scenes'][s]['unmasked_logits'] for s in SCENES}),ensure_ascii=False))


if __name__ == '__main__':
    main()
