#!/usr/bin/env python3
"""Independent deterministic BG009 encoder review; no optimizer or rollout."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[6]
PROJECT = ROOT / 'experiments/lagrangian_ppo_v1'
sys.path.insert(0, str(PROJECT))
from business_graph_summary_v1.models import Actor, MaterialCostCritic, digest
from business_graph_summary_v1.graph import (build_graph, NODE_NAMES, RELATION_NAMES,
                                            GROUP_NAMES)

REPORT = PROJECT / 'reports/token_attention_001/business_graph_summary_009/encoder_snapshot_001'
HISTORY = PROJECT / 'reports/token_attention_001/business_graph_summary_009/historical_snapshot_001'
SEED = 2026111801


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_layers(encoder, z, graph):
    h = encoder.graph_layers[0](z, graph)
    return encoder.graph_layers[1](h, graph)


def direct(graph, dst, source_limit=None):
    return [dict(source=NODE_NAMES[s], destination=NODE_NAMES[d],
                 relation=RELATION_NAMES[t], group=GROUP_NAMES[g])
            for (s, d), t, g in zip(graph.edge_index.T.tolist(),
                                   graph.edge_type.tolist(), graph.edge_group.tolist())
            if d == dst and (source_limit is None or s < source_limit)]


def stats(effect, actual):
    norm2 = float(actual.square().sum())
    return dict(effect_l2=float(effect.norm()), actual_T1_minus_T0_l2=float(actual.norm()),
                projection_fraction_of_actual=(float((effect * actual).sum()) / norm2
                                               if norm2 else None))


def main():
    torch.set_num_threads(1)
    REPORT.mkdir(parents=True, exist_ok=True)
    raw = json.loads((HISTORY / 'snapshots.json').read_text())
    states = {t: torch.tensor([raw[t]['raw']], dtype=torch.float32) for t in ('T0', 'T1')}
    graphs = {t: build_graph(states[t]) for t in states}
    models = {'actor': Actor(pair_seed=SEED).eval(),
              'critic': MaterialCostCritic(pair_seed=SEED).eval()}
    checks, values, path_rows, npz_paths = {}, {}, [], {}
    saved = np.load(REPORT / 'representations.npz')
    saved_attr = np.load(REPORT / 'attribution_vectors.npz')
    saved_messages = np.load(REPORT / 'message_tensors.npz')
    forward = {}
    with torch.inference_mode():
        for role, model in models.items():
            encoder = model.state_encoder
            before = digest(model)
            details = {}
            rolesummary = {}
            for tag, state in states.items():
                h, detail = encoder.scene(state, diagnostic=True)
                details[tag] = detail
                forward[(role, tag)] = detail
                ordinary = encoder.scene(state)
                checks[f'{role}_{tag}_diagnostic_equals_formal_scene'] = torch.equal(h, ordinary)
                stages = [detail['z0'], detail['h1'], detail['h2']]
                for level, stage in enumerate(stages):
                    key = f'{role}_{tag}_H{level}'
                    observed = saved[key]
                    desired = stage.numpy()
                    if observed.shape == (19, 64):
                        desired = desired[0]
                    checks[key + '_saved_exact_formal'] = np.array_equal(observed, desired)
                    if level:
                        checks[key + '_Global_bypass'] = torch.equal(stage[:, 18], stages[0][:, 18])
                layer_stats = []
                for n, layer in enumerate(encoder.graph_layers, 1):
                    d = detail[f'layer{n}']
                    graph = detail['graph']
                    weighted = (d['messages'] * d['weights'][..., None]).flatten(1)
                    pieces = torch.stack([weighted[k] @ layer.output.weight[:, int(group)*64:(int(group)+1)*64].T
                                          for k, group in enumerate(graph.edge_group)])
                    rebuilt = torch.zeros(19, 64).index_add_(0, graph.edge_index[1], pieces)
                    presence = d['group_masks'][0].float() @ layer.output.weight[:, 256:].T
                    for label, vector in (('edge_projected', pieces), ('weights', d['weights']),
                                          ('messages', d['messages']),
                                          ('presence_projected', presence),
                                          ('projected', d['projected'][0]),
                                          ('pre_norm', d['pre_norm'][0])):
                        key = f'{role}_{tag}_layer{n}_{label}'
                        checks[key + '_saved_exact_independent'] = np.array_equal(
                            saved_messages[key], vector.numpy())
                    err = float((rebuilt + presence - d['projected'][0]).abs().max())
                    checks[f'{role}_{tag}_layer{n}_preLN_additive_closure'] = err < 2e-6
                    expected_pre = stages[n-1] + d['projected']
                    checks[f'{role}_{tag}_layer{n}_residual_exact'] = torch.equal(expected_pre, d['pre_norm'])
                    bucket = d['buckets']
                    sums = torch.zeros(19 * 4, 4).index_add_(0, bucket, d['weights'])
                    present = d['group_masks'].flatten()
                    weights_error = float((sums[present] - 1).abs().max())
                    checks[f'{role}_{tag}_layer{n}_within_group_softmax'] = weights_error < 3e-7
                    layer_stats.append(dict(layer=n, projected_closure_max_abs=err,
                                            group_weight_sum_max_abs_error=weights_error))
                rolesummary[tag] = dict(layers=layer_stats)

            z0, z1 = details['T0']['z0'], details['T1']['z0']
            g0, g1 = details['T0']['graph'], details['T1']['graph']
            shapley = []
            for node in range(9, 13):
                cube = {}
                for mask in range(8):
                    z = z0.clone()
                    if mask & 2:
                        z[:] = z1
                        z[:, node] = z0[:, node]
                    if mask & 1:
                        z[:, node] = z1[:, node]
                    cube[mask] = run_layers(encoder, z, g1 if mask & 4 else g0)[0, node]
                phis = []
                for factor in range(3):
                    phi = torch.zeros(64)
                    for mask in range(8):
                        if mask & (1 << factor):
                            continue
                        k = mask.bit_count()
                        weight = math.factorial(k) * math.factorial(2-k) / 6
                        phi += weight * (cube[mask | (1 << factor)] - cube[mask])
                    phis.append(phi)
                delta = cube[7] - cube[0]
                closure = float((sum(phis) - delta).abs().max())
                checks[f'{role}_{NODE_NAMES[node]}_Shapley_closure'] = closure < 2e-6
                for name, vector in zip(('own_H0', 'other_H0', 'graph'), phis):
                    key = f'{role}_{NODE_NAMES[node]}_{name}'
                    observed = saved_attr[key]
                    desired = vector.numpy().reshape(observed.shape)
                    checks[key + '_matches_independent'] = np.allclose(observed, desired, rtol=0, atol=2e-6)
                shapley.append(dict(node=NODE_NAMES[node], closure_max_abs=closure))
            rolesummary['shapley'] = shapley

            baseline_h1 = details['T1']['h1']
            baseline_h2 = details['T1']['h2']
            actual_delta = baseline_h2 - details['T0']['h2']

            def record_patch(name, patched_h1, source_description):
                out = encoder.graph_layers[1](patched_h1, g1)
                for node in range(9, 13):
                    effect = baseline_h2[0, node] - out[0, node]
                    row = dict(role=role, intervention=name, load=NODE_NAMES[node],
                               patched_source=source_description, fixed_graph='G_T1',
                               fixed_destination_H1='T1', fixed_other_H1='T1',
                               **stats(effect, actual_delta[0, node]))
                    path_rows.append(row)
                    npz_paths[f'{role}_{name}_{NODE_NAMES[node]}_baseline_minus_patched'] = effect.numpy()
                return out

            for dump in (13, 14):
                mixed = baseline_h1.clone()
                mixed[:, dump] = details['T0']['h1'][:, dump]
                out = record_patch(f'{NODE_NAMES[dump]}_H1_T0_into_T1', mixed, NODE_NAMES[dump])
                unrelated = [12] if dump == 13 else [9, 10, 11]
                checks[f'{role}_{NODE_NAMES[dump]}_patch_no_direct_edge_no_Load_effect'] = all(
                    torch.equal(out[:, node], baseline_h2[:, node]) for node in unrelated)

            for trucks in ((3,), (5,), (7,), (3, 5, 7)):
                mixed_z = z1.clone()
                mixed_z[:, list(trucks)] = z0[:, list(trucks)]
                changed_h1 = encoder.graph_layers[0](mixed_z, g1)
                isolated_h1 = baseline_h1.clone()
                isolated_h1[:, 13] = changed_h1[:, 13]
                name = 'Truck' + '_'.join(map(str, trucks)) + '_H0_via_Dump0_H1'
                record_patch(name, isolated_h1, ','.join(f'Truck{t}' for t in trucks))
                for row in path_rows[-4:]:
                    row['Dump0_H1_effect_l2'] = float((baseline_h1[:, 13] - changed_h1[:, 13]).norm())
                checks[f'{role}_{name}_Load3_unchanged'] = torch.equal(
                    encoder.graph_layers[1](isolated_h1, g1)[:, 12], baseline_h2[:, 12])
            checks[role + '_all_parameters_and_buffers_unchanged'] = before == digest(model)
            rolesummary['parameter_digest_before_after'] = before
            values[role] = rolesummary

    # Uniform fields for CSV; four H1 snapshot-patch rows lack the Truck H0 field.
    for row in path_rows:
        row.setdefault('Dump0_H1_effect_l2', '')
    write_csv(REPORT / 'path_interventions.csv', path_rows)
    np.savez_compressed(REPORT / 'path_interventions.npz', **npz_paths)
    path_info = dict(scope='initialization-only internal representation patching; not business causal attribution',
                     convention='effect = baseline_T1_H2 - patched_H2; projection fractions signed and noncausal',
                     rows=path_rows,
                     limitations=[
                         'Dump_H1_T0_into_T1 patches the complete encoded row, including own and neighbor content.',
                         'Truck_H0 patch retains T1 edge types, attributes, and all other H0 rows; it does not remove new loaded_dump edges.',
                         'Only the resulting Dump0 H1 row is patched into layer two; other first-layer changes are discarded.',
                         'These hybrid internal tensors need not correspond to a physically realizable observation.',
                         'No action probabilities or trained representation are inferred.'])
    (REPORT / 'path_interventions.json').write_text(json.dumps(path_info, ensure_ascii=False, indent=2) + '\n')
    topology = {}
    for tag, graph in graphs.items():
        dump_edges = direct(graph, 13, 9)
        material = [r for r in graph.records() if r['source'] in (13, 14)
                    and r['destination'] in range(9, 13) and r['group'] == 'material']
        topology[tag] = dict(Truck_into_Dump0=dump_edges, Dump_into_Load=material)
        loaded = {r['source'] for r in dump_edges if r['relation'] == 'loaded_dump'}
        expected = {'Truck0', 'Truck2', 'Truck6'} if tag == 'T0' else {'Truck2', 'Truck3', 'Truck5', 'Truck6', 'Truck7'}
        checks[tag + '_loaded_Truck_Dump0_sources_exact'] = loaded == expected
        checks[tag + '_compatible_Dump_Load_exact'] = {
            (r['source'], r['destination']) for r in material} == {(13, 9), (13, 10), (13, 11), (14, 12)}
    result = dict(status='PASS' if all(checks.values()) else 'FAIL', seed=SEED,
                  initialization='named-paired-v1; independent actor and critic; no trained BG checkpoint',
                  checks=checks, values=values, topology=topology,
                  limitations=[
                      'Attention alpha is only a normalized score, not output contribution.',
                      'Additive projected edge vectors are exact before LayerNorm; norms are not additive and LayerNorm is nonlinear.',
                      'H0 already contains referenced site/truck/route/quantity information; two graph layers add two relation-composition steps.',
                      'Own H0 attribution is not purely raw local-state attribution because H0 embeds referenced neighbor content and deterministic quantities.',
                      'The G factor covers edge topology, relation labels and observed edge attributes jointly.',
                      'Shapley vectors close additively; their norms do not. Signed projections may be negative or exceed one.',
                      'Random initialization diagnostics establish numerical channels and implementation behavior, not learned scheduling use.',
                      'C128 enters final Actor/Critic readouts, not this graph encoder H0/H1/H2.',
                  ],
                  sources=[dict(path=str(p), sha256=sha(p)) for p in
                           (HISTORY / 'snapshots.json', REPORT / 'representations.npz',
                            REPORT / 'attribution_vectors.npz', REPORT / 'message_tensors.npz')])
    (REPORT / 'independent_review.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    failed = [k for k, v in checks.items() if not v]
    md = ['# BG128 编码器快照独立核验', '',
          f'状态：**{result["status"]}**；{len(checks)} 项检查；固定初始化 seed={SEED}。', '',
          '未加载已训练 BG checkpoint。只执行无梯度前向和内部张量替换，无优化器、参数更新或仿真。', '',
          '## 核验内容', '',
          '- 保存 H0/H1/H2 与正式 `GraphStateEncoder.scene` 逐元素一致。',
          '- 正式 diagnostic 与普通 scene 一致；两层 Global 均保持旁路。',
          '- 每条边的加权 value 经对应关系组输出块投影，逐目标求和后加 presence 独立项，恢复 LayerNorm 前 projected。',
          '- 三因素 Shapley 由独立八格混合重算，向量和恢复真实 T1−T0。',
          '- 两种角色全部参数和 buffer 前后摘要一致。', '',
          '## 路径补丁边界', '',
          '`Dump0_H1_T0_into_T1` 只把 Dump0 的第1层表示替换为 T0；第二层图、其余节点 H1 与候选 Load 的目标 H1 固定为 T1。',
          '`Truck*_H0_via_Dump0_H1` 先在 G_T1 上替换所列 Truck 的 H0，再只将所得 Dump0 H1 放入第二层；不改变新 loaded_dump 边或其他节点 H1。',
          '结果保存在 `path_interventions.csv/json/npz`。这是初始化网络内部通道测试，不是可实现原始状态干预，也不是业务因果效应。', '',
          '## 解释边界', '']
    md += ['- ' + line for line in result['limitations']]
    if failed:
        md += ['', '未通过：' + ', '.join(failed)]
    (REPORT / 'independent_review.md').write_text('\n'.join(md) + '\n')
    print(json.dumps(dict(status=result['status'], checks=len(checks), failed=failed,
                          path_rows=len(path_rows)), ensure_ascii=False))
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
