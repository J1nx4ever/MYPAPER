#!/usr/bin/env python3
"""Independent raw-state and numerical-forward audit of BG128 interventions.

No optimizer, RPC, rollout, or simulation is created. The generated scenarios
are explicit counterfactual observation fixtures, not replayed outcomes.
"""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
import h5py

ROOT = Path(__file__).resolve().parents[6]
PROJECT = ROOT / 'experiments/lagrangian_ppo_v1'
sys.path.insert(0, str(PROJECT))
from business_graph_summary_v1.graph import (build_graph, QUANTITY_NAMES,
    PHYSICAL_NODE_TO_TOKEN, ROAD_ENDPOINTS, NODE_NAMES)
from business_graph_summary_v1.models import Actor, MaterialCostCritic, digest
from business_graph_summary_v1.c_features import build_features
from single_critic_task_query_v1.features_cache import pack_summary_prefixes
from relation_gat_ego_v1.adapter import validate_states

OUT = PROJECT / 'reports/token_attention_001/business_graph_summary_009/physical_intervention_001'
HISTORY = PROJECT / 'reports/token_attention_001/business_graph_summary_009/historical_snapshot_001'
SEED = 2026111801


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hot_id(row):
    return int(np.argmax(row)) if np.sum(row) else -1


def wire_ids(row):
    return [int(round(float(x) * 9)) - 1 for x in row]


def quantity_oracle(state):
    """Independent 21-channel physical sums, never attention averages."""
    trucks = state[27:1107].reshape(9, 120)
    loads = state[1107:1295].reshape(4, 47)
    dumps = state[1295:1391].reshape(2, 48)
    quantities = np.zeros((19, 21), dtype=np.float64)
    qi = dict(zip(QUANTITY_NAMES, range(21)))

    def add(node, name, amount=1):
        quantities[node, qi[name]] += amount

    ego = hot_id(trucks[:, 0])
    segments = [hot_id(t[45:61]) for t in trucks]
    occupancy = Counter(s for s in segments if s >= 0)
    for rid, (u, v) in enumerate(ROAD_ENDPOINTS):
        u, v = PHYSICAL_NODE_TO_TOKEN[u], PHYSICAL_NODE_TO_TOKEN[v]
        add(u, 'outgoing_road_count')
        add(v, 'incoming_road_count')
        add(u, 'outgoing_segment_truck_count', occupancy[rid])
        add(v, 'incoming_segment_truck_count', occupancy[rid])
    for tid, t in enumerate(trucks):
        node = hot_id(t[32:41])
        if segments[tid] < 0 and node >= 0:
            add(PHYSICAL_NODE_TO_TOKEN[node], 'at_node_truck_count')
        target_load, target_dump = hot_id(t[26:30]), hot_id(t[30:32])
        if tid != ego and not t[17] and t[20] > 0 and target_load >= 0:
            add(9 + target_load, 'reserved_truck_count')
            add(9 + target_load, 'reserved_tons', float(t[20]) * 193)
        if t[17] and target_dump >= 0:
            add(13 + target_dump, 'inbound_loaded_truck_count')
            add(13 + target_dump, 'inbound_loaded_tons', float(t[19]) * 193)
            if t[21]:
                add(13 + target_dump, 'inbound_known_grade_truck_count')
                add(13 + target_dump, 'inbound_known_grade_tons', float(t[19]) * 193)
                add(13 + target_dump, 'inbound_known_metal_tons', float(t[19]) * 193 * float(t[22]))
    for sites, offset, col, relation in ((loads, 9, 20, 'queue'),
        (loads, 9, 29, 'seize'), (loads, 9, 38, 'service'),
        (dumps, 13, 12, 'queue'), (dumps, 13, 21, 'seize'),
        (dumps, 13, 30, 'capacity_wait'), (dumps, 13, 39, 'service')):
        for j, row in enumerate(sites):
            ids = [x for x in wire_ids(row[col:col+9]) if x >= 0]
            add(offset + j, relation + '_unique_count', len(set(ids)))
            add(offset + j, relation + '_instance_count', len(ids))
    for j, row in enumerate(dumps):
        if wire_ids(row[11:12])[0] >= 0:
            add(13 + j, 'admission_count')
    return quantities


def require(checks, key, condition):
    checks[key] = bool(condition)


def physical_checks(raw):
    t, l, d = raw[27:1107].reshape(9, 120), raw[1107:1295].reshape(4, 47), raw[1295:1391].reshape(2, 48)
    road = raw[1439:].reshape(16, 5)
    checks = {}
    for i, site in enumerate(l):
        reserved = sum(float(v[20]) * 193 for v in t if hot_id(v[26:30]) == i)
        checks[f'Load{i}_reserved_vs_trucks'] = abs(float(site[6]) * 30000 - reserved) < .006
        checks[f'Load{i}_source_conservation'] = abs(float(site[4]) * 30000 - float(site[5] + site[8]) * 30000) < .006
        checks[f'Load{i}_available'] = abs(float(site[7]) * 30000 - max(0., float(site[5] - site[6]) * 30000)) < .006
        service = [x for x in wire_ids(site[38:47]) if x >= 0]
        checks[f'Load{i}_service_counts'] = float(site[15]) == float(site[16]) == len(service)
        checks[f'Load{i}_service_no_duplicate_in_fixture'] = len(service) == len(set(service))
        checks[f'Load{i}_service_target_and_phase'] = all(not t[j, 17] and hot_id(t[j, 26:30]) == i and hot_id(t[j, 4:16]) == 4 for j in service)
        in_progress = sum(float(t[j, 20]) * 193 for j in service)
        checks[f'Load{i}_allocated_cursor_for_fixture'] = abs(float(site[9]) * 30000 - float(site[8]) * 30000 - in_progress) < .006
        incoming = sum(hot_id(v[4:16]) == 2 and hot_id(v[26:30]) == i for v in t)
        checks[f'Load{i}_incoming_count'] = abs(float(site[17]) * 9 - incoming) < 2e-6
    for tid, truck in enumerate(t):
        if truck[17]:
            checks[f'Truck{tid}_loaded_consistency'] = (truck[19] > 0 and truck[20] == 0 and truck[21] == 1 and hot_id(truck[26:30]) < 0 and hot_id(truck[30:32]) == (0 if truck[18] else 1))
        else:
            checks[f'Truck{tid}_unloaded_consistency'] = truck[19] == truck[21] == truck[22] == 0 and hot_id(truck[30:32]) < 0
        seg = hot_id(truck[45:61])
        ranks = np.rint(truck[63:79] * 16).astype(int)
        route = sorted(np.flatnonzero(ranks), key=lambda j: ranks[j])
        if seg >= 0:
            checks[f'Truck{tid}_route_order'] = bool(route and route[0] == seg and [ranks[j] for j in route] == list(range(1, len(route)+1)))
            checks[f'Truck{tid}_route_contiguous'] = all(ROAD_ENDPOINTS[a][1] == ROAD_ENDPOINTS[b][0] for a, b in zip(route, route[1:]))
            distance = float(truck[61]) + sum(float(road[j, 2]) for j in route[1:])
            checks[f'Truck{tid}_route_distance'] = abs(float(truck[62]) - distance) < 3e-7
        else:
            checks[f'Truck{tid}_no_moving_route'] = not route and truck[61] == truck[62] == truck[79] == 0
    produced = sum(float(v[8]) * 30000 for v in l if v[1])
    accounted = float(raw[7]) * 20000 + float(d[0, 5]) * 500 + sum(float(v[19]) * 193 for v in t if v[17] and v[18])
    checks['ore_mass_conservation'] = abs(produced - accounted) < .015
    checks['crusher_buffer_accounting'] = abs(float(d[0, 8] - d[0, 5] - d[0, 6] - d[0, 7])) < 1e-6
    ego = hot_id(t[:, 0]); capacity = float(t[ego, 16]) * 193
    waste = float(raw[9]) * 4000 + sum(float(v[6]) * 30000 for v in l if not v[1]) + sum(float(v[19]) * 193 for v in t if v[17] and not v[18])
    for i, site in enumerate(l):
        q = min(capacity, max(0., float(site[5] - site[6]) * 30000)) if site[0] and not raw[17] else 0.
        if not site[1] and waste >= 6000 - 1e-6:
            q = 0.
        checks[f'candidate{i}_sidecar_consistency'] = raw[19+i] == int(q > 1e-6) and abs(float(raw[23+i]) * 193 - q) < 2e-5
    return checks


def main():
    torch.set_num_threads(1)
    source = OUT / 'scenarios.json'
    source_sha = sha(source)
    package = json.loads(source.read_text())
    summary = json.loads((OUT / 'summary.json').read_text())
    saved = np.load(OUT / 'analysis_tensors.npz')
    raw = {'BASE': np.asarray(package['base']['raw'], dtype=np.float32)}
    raw.update({s: np.asarray(value['raw'], dtype=np.float32) for s, value in package['scenarios'].items()})
    checks, details = {}, {}
    require(checks, 'seed_matches_fixed_initialization', package['seed'] == summary['pair_seed'] == SEED)
    with h5py.File(package['base']['source_hdf5'], 'r') as h:
        require(checks, 'BASE_exact_original_HDF5_T1_row', np.array_equal(raw['BASE'], h['states'][package['base']['state_index']]))
    history = json.loads((HISTORY / 'snapshots.json').read_text())
    require(checks, 'BASE_exact_original_saved_T1', np.array_equal(raw['BASE'], np.asarray(history['T1']['raw'], dtype=np.float32)))
    require(checks, 'source_hdf5_sha_authenticated', sha(package['base']['source_hdf5']) == package['base']['source_sha256'])
    # The field white lists are independent of builder metadata.
    whites = {
        'A': {155, 162, 164, 166, 167, 168, 169, 170, 172, 173, 177, 231, 232, 233, 1112, 1113, 1115, 1122, 1123, 1145},
        'B': {1013, 1014, 1049, 1058, 1062, 1063, 1113, 1114, 1124, 1160, 1161, 1171},
        'C': {1304},
    }
    graphs = {}
    for scenario, x in raw.items():
        validate_states(x[None])
        graphs[scenario] = graph = build_graph(torch.tensor(x[None]))
        for key, value in (('raw', x), ('quantities', graph.quantities[0].numpy()),
            ('raw_quantities', graph.raw_quantities[0].numpy()), ('edge_index', graph.edge_index.numpy()),
            ('edge_type', graph.edge_type.numpy()), ('edge_attr', graph.edge_attr.numpy())):
            require(checks, f'{scenario}_{key}_saved_exact_independent', np.array_equal(saved[f'{scenario}_{key}'], value))
        oracle = quantity_oracle(x)
        err = float(np.max(np.abs(oracle - graph.raw_quantities[0].numpy())))
        require(checks, f'{scenario}_all21_quantities_raw_oracle', err < .0002)
        details[scenario] = dict(quantity_oracle_max_abs_error=err)
        for key, value in physical_checks(x).items():
            require(checks, f'{scenario}_{key}', value)
        if scenario != 'BASE':
            changed = set(np.flatnonzero(x != raw['BASE']).tolist())
            require(checks, scenario + '_raw_changed_indices_exact_whitelist', changed == whites[scenario])
            require(checks, scenario + '_changed_metadata_matches_actual', changed == set(package['scenarios'][scenario]['changed_indices']))
            require(checks, scenario + '_cumulative_history_and_clock_fixed', np.array_equal(x[:19], raw['BASE'][:19]))
            require(checks, scenario + '_FIFO_unchanged', np.array_equal(x[1391:1439], raw['BASE'][1391:1439]))
    base_t = raw['BASE'][27:1107].reshape(9, 120)
    base_l = raw['BASE'][1107:1295].reshape(4, 47)
    at = raw['A'][27:1107].reshape(9, 120)[1]
    al = raw['A'][1107:1295].reshape(4, 47)[0]
    q = float(base_t[1, 20]) * 193
    require(checks, 'A_payload_exact_preexisting_reservation', at[19] == base_t[1, 20])
    require(checks, 'A_grade_only_current_known_inferred_source', base_t[1, 25] == 1 and base_t[1, 24] == 0 and at[22] == base_t[1, 23] == base_l[0, 13])
    require(checks, 'A_phase11_transient_not_TO_DUMP', hot_id(at[4:16]) == 11)
    require(checks, 'A_completed_reserved_source_material', abs(float(base_l[0, 5] - al[5]) * 30000 - q) < .006 and abs(float(base_l[0, 6] - al[6]) * 30000 - q) < .006 and abs(float(al[8] - base_l[0, 8]) * 30000 - q) < .006)
    require(checks, 'A_available_exactly_preserved_no_ULP_artifact', al[7] == base_l[0, 7])
    require(checks, 'A_cursor_and_lastKnown_preserved', np.array_equal(al[9:14], base_l[0, 9:14]))
    bt = raw['B'][27:1107].reshape(9, 120)[8]
    require(checks, 'B_current_segment_position_history_retained', np.array_equal(bt[32:62], base_t[8, 32:62]) and np.array_equal(bt[79:], base_t[8, 79:]))
    require(checks, 'B_route_tail_reaches_Load1', hot_id(bt[26:30]) == 1 and np.flatnonzero(bt[63:79]).tolist() == [7, 12, 13])
    require(checks, 'C_rate_halved_only', raw['C'][1304] == raw['BASE'][1304] / 2)
    require(checks, 'C_graph_and_all_quantities_unchanged', all(torch.equal(getattr(graphs['BASE'], field), getattr(graphs['C'], field)) for field in ('edge_index', 'edge_type', 'edge_group', 'edge_attr', 'quantities')))

    for role, cls in (('actor', Actor), ('critic', MaterialCostCritic)):
        model = cls(pair_seed=SEED).eval()
        before = digest(model)
        role_data = {}
        with torch.inference_mode():
            for scenario, x in raw.items():
                state = torch.tensor(x[None])
                certificate = pack_summary_prefixes(x[None].copy(), feature_config=model.feature_config)
                output, diagnostic = model(state, certificate, diagnostic=True)
                ordinary = model(state, certificate)
                require(checks, f'{role}_{scenario}_ordinary_equals_diagnostic', torch.equal(output, ordinary))
                c128 = build_features(x, config=model.feature_config)
                require(checks, f'{role}_{scenario}_C128_certificate_current_raw', np.array_equal(c128, certificate.features[0]) and np.array_equal(c128, saved[f'{scenario}_C128']))
                require(checks, f'{role}_{scenario}_certificate_sha_matches_analysis', certificate.certificate_sha256 == summary['roles'][role]['scenes'][scenario]['summary_certificate'])
                for level, key in enumerate(('z0', 'h1', 'h2')):
                    require(checks, f'{role}_{scenario}_H{level}_saved_exact_independent', np.array_equal(saved[f'{role}_{scenario}_H{level}'], diagnostic[key][0].numpy()))
                require(checks, f'{role}_{scenario}_Global_bypass', torch.equal(diagnostic['z0'][:, 18], diagnostic['h1'][:, 18]) and torch.equal(diagnostic['z0'][:, 18], diagnostic['h2'][:, 18]))
                if role == 'actor':
                    for key in ('candidate_input', 'candidates', 'context_input', 'context', 'task_summary', 'action_summary', 'unmasked_logits', 'masked_logits'):
                        require(checks, f'{role}_{scenario}_{key}_saved_exact_independent', np.array_equal(saved[f'actor_{scenario}_{key}'], diagnostic[key][0].numpy()))
                    probs = torch.softmax(output, -1)[0].numpy()
                    require(checks, f'{role}_{scenario}_probs_saved_exact_independent', np.array_equal(saved[f'actor_{scenario}_probs'], probs))
                    h2 = diagnostic['h2']; g = graphs[scenario]; ego = int(g.ego_ids[0]); dump = g.candidate_dump_ids[0]
                    manual = torch.cat((h2[:, ego:ego+1].expand(-1,4,-1), h2[:, 9:13], h2[:, dump],
                        g.quantities[:, ego:ego+1].expand(-1,4,-1), g.quantities[:, 9:13], g.quantities[:, dump],
                        state[:, 19:23, None], state[:, 23:27, None], diagnostic['action_summary']), -1)
                    require(checks, f'{scenario}_candidate_direct_Ego_Load_Dump_quantities_C10_exact', torch.equal(manual, diagnostic['candidate_input']))
                    product = diagnostic['candidates'] * diagnostic['context'][:, None]
                    score = model.scorer(torch.cat((diagnostic['candidates'], diagnostic['context'][:, None].expand(-1,4,-1), product), -1)).squeeze(-1) + product.sum(-1) / 8
                    require(checks, f'{scenario}_shared_interaction_score_formula_exact', torch.equal(score, diagnostic['unmasked_logits']))
                    role_data[scenario] = dict(certificate_sha256=certificate.certificate_sha256, probabilities=probs.tolist())
                else:
                    require(checks, f'{role}_{scenario}_value_saved_exact_independent', np.array_equal(saved[f'critic_{scenario}_value'], output.numpy()))
            require(checks, role + '_all_parameters_buffers_unchanged', digest(model) == before == summary['roles'][role]['parameter_sha256_before'] == summary['roles'][role]['parameter_sha256_after'])
        details[role] = dict(parameter_sha256=before, scenes=role_data)

    # Identify content-mediated two-hop channels from saved current-raw forwards.
    paths, changed_sources = [], {}
    records = graphs['C'].records()
    for role in ('actor', 'critic'):
        delta = saved[f'{role}_C_H0'] - saved[f'{role}_BASE_H0']
        sources = set(np.flatnonzero(np.any(delta != 0, axis=1)).tolist())
        changed_sources[role] = [dict(node=NODE_NAMES[node], H0_delta_l2=float(np.linalg.norm(delta[node]))) for node in sorted(sources)]
        require(checks, role + '_C_H0_changed_nodes_exact', sources == {2,3,5,6,7,13})
        require(checks, role + '_C_Load_H0_unchanged', not np.any(delta[9:13]))
        require(checks, role + '_C_Load3_H1_unchanged', np.array_equal(saved[f'{role}_C_H1'][12], saved[f'{role}_BASE_H1'][12]))
        for first in records:
            if first['source'] not in sources:
                continue
            for second in records:
                if first['destination'] == second['source'] and second['destination'] == 12:
                    paths.append(dict(role=role, source=NODE_NAMES[first['source']],
                        relation1=first['relation'], relay=NODE_NAMES[first['destination']],
                        relation2=second['relation'], destination='Load3',
                        first_attributes=first['attributes'], second_attributes=second['attributes']))
    require(checks, 'C_content_change_to_Load3_seven_two_hop_walks_each_role', len(paths) == 14)
    require(checks, 'C128_current_T1_eight_changes', np.flatnonzero(saved['C_C128'] != saved['BASE_C128']).tolist() == [19,45,56,57,58,60,67,78])
    (OUT / 'C_Load3_content_paths.json').write_text(json.dumps(paths, ensure_ascii=False, indent=2) + '\n')
    with (OUT / 'cross_material_paths_C.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=['role', 'source', 'relation1', 'relay',
            'relation2', 'destination', 'first_attributes', 'second_attributes', 'interpretation'])
        writer.writeheader()
        for row in paths:
            writer.writerow(row | dict(first_attributes=json.dumps(row['first_attributes'], sort_keys=True),
                second_attributes=json.dumps(row['second_attributes'], sort_keys=True),
                interpretation='topological content channel only; path effects are not additive; truck_predecessor is initial dispatch ordering, not current moving front truck'))
    require(checks, 'scenarios_source_unchanged_during_review', sha(source) == source_sha)
    core = ROOT / 'python/episode_budget_ppo.py'
    require(checks, 'protected_PPO_SHA_unchanged', sha(core) == '8e4cfa31c622f5e33756e304eea3f7526547db72ae93f7d9f00808845d61ef96')
    result = dict(status='PASS' if all(checks.values()) else 'FAIL', checks=checks, details=details,
        C_changed_H0_sources=changed_sources, C_to_Load3_two_hop_paths=paths,
        scope='independent raw-observation constraints and full random-initialization forward; no training/RPC/simulation',
        limits=[
            'A is source-compatible after-load-completion/before-travel microstep, not proof production RPC can pause there.',
            'A uses a grade already completely inferred at BASE; no T0/T1 future result is inserted.',
            'B is an exogenous reservation/route-tail reassignment, not a normal exposed PPO action.',
            'C is an exogenous same-time positive processing-rate step, outside original runtime configuration.',
            'Raw fixtures satisfy checked observable constraints, not complete hidden AnyLogic-state reachability.',
            'H0 already embeds references and deterministic quantities; no exclusive two-hop mediation claim is made.',
            'The seven C-to-Load3 paths are overlapping information channels; individual path contributions cannot be summed to the total H2 response.',
            'truck_predecessor denotes the initial dispatch predecessor, not the currently leading truck in road motion.',
            'Full Actor/Critic and C128 fusion were run this time; all outputs remain untrained numeric sensitivities.',
        ],
        sources=[dict(path=str(p),sha256=sha(p)) for p in (source, OUT/'analysis_tensors.npz', OUT/'summary.json', core)])
    (OUT / 'independent_review.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    failed = [name for name, value in checks.items() if not value]
    md = ['# BG128 原始观测干预独立核验', '', f'状态：**{result["status"]}**；{len(checks)} 项检查。', '',
        '从当前 scenarios.json 独立读取 BASE/A/B/C，未调用构造脚本；从原始HDF5重新核验共同T1基准。', '',
        '- 原始修改索引严格符合独立白名单：A20项、B12项、C1项；A可用吨位保持原始float32位模式，避免1ULP伪变化。',
        '- 独立重算21维数量、服务/预留/物料守恒、路线连续性与长度、候选mask/tons。',
        '- 从各自原始输入重建图、H0/H1/H2、C128与证书，逐元素核对保存的候选输入、表示、logits、概率和价值。',
        '- Actor普通与diagnostic前向一致；显式重建候选Ego/Load/Dump及乘性交互评分公式；参数/buffer前后摘要不变。', '',
        '## C 的跨物料内容传播', '',
        'C没有修改事实图或21维数量，但Dump0率通过初始dump_ref使Truck2/3/5/6/7的H0改变。Load0..3的H0均未变。',
        '到Load3存在7条两跳消息通道：Truck3→Junction3→Load3；Truck3→Truck4→Load3（三条不同入边）；Truck5→Truck4→Load3（三条不同入边）。',
        '具体方向、关系和边属性见 C_Load3_content_paths.json 与 cross_material_paths_C.csv。它们解释无拓扑变化也可有Load3 H2响应；不意味着矿石变为废石或业务因果正确。',
        '`truck_predecessor` 是首次派车前驱关系，并非当前道路运动前车。7条路径共享节点、非线性与归一化，不能将各条路径范数相加来分摊总响应。',
        '两种角色的H0变化来源及L2：' + json.dumps(changed_sources, ensure_ascii=False), '',
        '## 边界', '']
    md += ['- ' + value for value in result['limits']]
    if failed:
        md += ['', '未通过：' + ', '.join(failed)]
    (OUT / 'independent_review.md').write_text('\n'.join(md) + '\n')
    print(json.dumps(dict(status=result['status'], checks=len(checks), failed=failed), ensure_ascii=False))
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
