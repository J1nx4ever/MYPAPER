"""Frozen-initialization path patches for the two audited BG009 snapshots.

Only internal tensors are patched. They are computation probes, never synthetic
physical observations, training examples, or evidence of a learned policy.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[6]
PROJECT = ROOT / 'experiments/lagrangian_ppo_v1'
sys.path.insert(0, str(PROJECT))
from business_graph_summary_v1.encoder import SITE_NODE_IDS
from business_graph_summary_v1.graph import NODE_NAMES, QUANTITY_NAMES, ROAD_ENDPOINTS, build_graph
from business_graph_summary_v1.models import Actor, MaterialCostCritic, digest

INPUT = PROJECT / 'reports/token_attention_001/business_graph_summary_009/historical_snapshot_001'
OUT = PROJECT / 'reports/token_attention_001/business_graph_summary_009/encoder_snapshot_001'
SEED = 2026111801
PPO = ROOT / 'python/episode_budget_ppo.py'
EXPECTED_PPO = '8e4cfa31c622f5e33756e304eea3f7526547db72ae93f7d9f00808845d61ef96'
BLOCKS = dict(self_sitebase=(0, 16), business_references=(22, 150), geography=(150, 176))
GROUPS = (*BLOCKS, 'quantities')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm(x):
    return float(torch.linalg.vector_norm(x.double()).item())


def cosine(a, b):
    return float(torch.nn.functional.cosine_similarity(a.double(), b.double(), dim=-1).item())


def write_csv(path, rows):
    assert rows
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def forward(encoder, raw, graph):
    h0, d = encoder.token_encoder(raw, graph, diagnostic=True)
    h1 = encoder.graph_layers[0](h0, graph)
    h2 = encoder.graph_layers[1](h1, graph)
    return (h0, h1, h2), d


def decode_slots(raw):
    return (np.rint(np.asarray(raw, dtype=np.float64) * 9).astype(int) - 1).tolist()


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    assert sha(PPO) == EXPECTED_PPO
    source_hashes = {str(p): sha(p) for p in (INPUT / 'snapshots.json', INPUT / 'selection.json', PPO)}
    snapshots = json.loads((INPUT / 'snapshots.json').read_text())
    raw = {t: torch.tensor([snapshots[t]['raw']], dtype=torch.float32) for t in ('T0', 'T1')}
    graph = {t: build_graph(raw[t]) for t in raw}
    schema = json.loads((PROJECT / 'python/observation_schema.json').read_text())
    raw_rows, quantity_rows = [], []
    facts = []
    for a in range(4):
        nid = 9 + a
        start = 1107 + 47 * a
        changes = []
        for j in range(47):
            v0, v1 = (float(raw[t][0, start + j]) for t in ('T0', 'T1'))
            if v0 != v1:
                rec = dict(node=NODE_NAMES[nid], local_index=j, raw_index=start+j,
                           name=schema[start+j]['name'], meaning=schema[start+j].get('meaning', ''),
                           block='site_local_20' if j < 20 else 'truck_reference_slots_27',
                           T0=v0, T1=v1, delta=v1-v0)
                raw_rows.append(rec)
                changes.append(rec)
        qs = []
        for j, name in enumerate(QUANTITY_NAMES):
            v0, v1 = (float(graph[t].raw_quantities[0, nid, j]) for t in ('T0', 'T1'))
            if v0 != v1:
                rec = dict(node=NODE_NAMES[nid], quantity=name, T0=v0, T1=v1, delta=v1-v0,
                           scaled_T0=float(graph['T0'].quantities[0, nid, j]),
                           scaled_T1=float(graph['T1'].quantities[0, nid, j]))
                quantity_rows.append(rec)
                qs.append(rec)
        site_node = SITE_NODE_IDS[a]
        incident = dict(incoming=[j for j, p in enumerate(ROAD_ENDPOINTS) if p[1] == site_node],
                        outgoing=[j for j, p in enumerate(ROAD_ENDPOINTS) if p[0] == site_node])
        lists = {}
        for name, span in dict(queue=(20, 29), seize=(29, 38), service=(38, 47)).items():
            lists[name] = {t: [i for i in decode_slots(raw[t][0, start+span[0]:start+span[1]]) if i >= 0]
                           for t in raw}
        at_node = {}
        for t in raw:
            trucks = raw[t][0, 27:1107].reshape(9, 120)
            at_node[t] = [i for i in range(9) if bool(trucks[i, 32:41].sum())
                          and int(trucks[i, 32:41].argmax()) == site_node
                          and not bool(trucks[i, 45:61].sum())]
        referenced_ids = sorted(set(i for name in lists for t in raw for i in lists[name][t]))
        referenced_changes = []
        for truck_id in referenced_ids:
            truck_start = 27 + 120 * truck_id
            for j in range(120):
                v0, v1 = (float(raw[t][0, truck_start+j]) for t in raw)
                if v0 != v1:
                    referenced_changes.append(dict(truck_id=truck_id, local_index=j,
                                                   name=schema[truck_start+j]['name'], T0=v0, T1=v1))
        facts.append(dict(node=NODE_NAMES[nid], node_id=nid, local20_identical=not any(x['local_index'] < 20 for x in changes),
                          raw47_changes=changes, quantities_changed=qs, observed_lists=lists,
                          referenced_truck_raw_changes=referenced_changes,
                          geographic_node=site_node, incident_roads=incident,
                          physically_at_node_trucks=at_node))
    result = dict(pair_seed=SEED, parameter_status='fixed named-paired initialization; zero optimization',
                  snapshots_source=str(INPUT / 'snapshots.json'), source_hashes=source_hashes,
                  method='single internal path patch and all-path patch from T0 into T1; all other node inputs and G1 fixed',
                  interpretation='sensitivity of this fixed initialization; nonadditive; not learned importance or physical counterfactual',
                  groups=dict(self_sitebase='site_final input 0:16, projected local20 (includes service/commitment summary counts)',
                              business_references='site_final input 22:150: ordered truck refs, service, admission, FIFO and flags; excludes own canonical identity',
                              geography='site_final input 150:176: projected incident road/current-at-node vehicle content, node ID and presence',
                              quantities='quantity_projection pre-ReLU output for the target Load only; deterministic graph quantities'),
                  raw_facts=facts, roles={})
    rows = []
    with torch.no_grad():
        for role, cls in (('actor', Actor), ('critic', MaterialCostCritic)):
            model = cls(pair_seed=SEED).eval()
            before = digest(model)
            encoder = model.state_encoder
            h, d = {}, {}
            for t in raw:
                h[t], d[t] = forward(encoder, raw[t], graph[t])
            quantity_linear0 = encoder.token_encoder.quantity_projection(graph['T0'].quantities).clone()
            rr = []
            for a in range(4):
                nid = 9 + a
                block_deltas = {name: norm(d['T1']['site_input'][0, a, lo:hi] - d['T0']['site_input'][0, a, lo:hi])
                                for name, (lo, hi) in BLOCKS.items()}
                block_deltas['quantities'] = norm(d['T1']['quantity_content'][0, nid] - d['T0']['quantity_content'][0, nid])
                stage = {f'H{k}': dict(T0_T1_l2=norm(h['T1'][k][0, nid]-h['T0'][k][0, nid]),
                                      T0_T1_cosine=cosine(h['T1'][k][0, nid], h['T0'][k][0, nid])) for k in range(3)}
                patches = []
                for chosen in [(g,) for g in GROUPS] + [GROUPS]:
                    def site_hook(_module, args, chosen=chosen):
                        value = args[0].clone()
                        for name in chosen:
                            if name in BLOCKS:
                                lo, hi = BLOCKS[name]
                                value[0, a, lo:hi] = d['T0']['site_input'][0, a, lo:hi]
                        return (value,)
                    def quantity_hook(_module, _args, output, chosen=chosen):
                        if 'quantities' not in chosen:
                            return output
                        output = output.clone()
                        output[0, nid] = quantity_linear0[0, nid]
                        return output
                    handles = [encoder.token_encoder.site_final.register_forward_pre_hook(site_hook),
                               encoder.token_encoder.quantity_projection.register_forward_hook(quantity_hook)]
                    try:
                        patched, _ = forward(encoder, raw['T1'], graph['T1'])
                    finally:
                        for handle in handles:
                            handle.remove()
                    name = '+'.join(chosen)
                    for k in range(3):
                        row = dict(role=role, node=NODE_NAMES[nid], stage=f'H{k}', patch=name,
                                   actual_T0_T1_l2=stage[f'H{k}']['T0_T1_l2'],
                                   baseline_T1_vs_patched_l2=norm(h['T1'][k][0, nid]-patched[k][0, nid]),
                                   T0_vs_patched_l2=norm(h['T0'][k][0, nid]-patched[k][0, nid]),
                                   fixed_graph='G_T1', other_node_initial_states='T1',
                                   interpretation='internal_nonphysical_nonadditive_sensitivity')
                        rows.append(row)
                        patches.append(row)
                    if chosen == GROUPS:
                        assert torch.equal(patched[0][0, nid], h['T0'][0][0, nid]), 'all input blocks must restore exact target H0'
                referenced_ids = sorted(set(i for name in ('queue', 'seize', 'service')
                                            for t in raw for i in facts[a]['observed_lists'][name][t]))
                refs = [dict(truck_id=i, truck_base_l2=norm(d['T1']['truck_base'][0, i]-d['T0']['truck_base'][0, i]))
                        for i in referenced_ids]
                geo_parts = {name: norm(d['T1']['node_input'][0, SITE_NODE_IDS[a], lo:hi]-d['T0']['node_input'][0, SITE_NODE_IDS[a], lo:hi])
                             for name, (lo, hi) in dict(incoming_road_mean=(0, 16), outgoing_road_mean=(16, 32),
                                                       at_node_truck_mean=(32, 48), counts=(48, 51), presence=(51, 54)).items()}
                rr.append(dict(node=NODE_NAMES[nid], stages=stage, preprojection_block_deltas=block_deltas,
                               referenced_truck_base_changes=refs, geography_node_input_deltas=geo_parts, patches=patches))
            assert digest(model) == before
            result['roles'][role] = dict(parameter_digest=before, parameter_unchanged=True, nodes=rr,
                                         H0_ranking=sorted(((x['node'], x['stages']['H0']['T0_T1_l2']) for x in rr), key=lambda z:-z[1]),
                                         H2_ranking=sorted(((x['node'], x['stages']['H2']['T0_T1_l2']) for x in rr), key=lambda z:-z[1]))
    assert all(sha(Path(p)) == v for p, v in source_hashes.items())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'raw_sensitivity.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    write_csv(OUT / 'raw_sensitivity.csv', rows)
    if raw_rows:
        write_csv(OUT / 'raw_field_changes.csv', raw_rows)
    else:
        (OUT / 'raw_field_changes.csv').write_text('node,local_index,raw_index,name,meaning,block,T0,T1,delta\n')
    if quantity_rows:
        write_csv(OUT / 'raw_quantity_changes.csv', quantity_rows)
    print(json.dumps(dict(output=str(OUT), raw_changes=len(raw_rows), quantity_changes=len(quantity_rows),
                          rankings={role:result['roles'][role]['H2_ranking'] for role in result['roles']}), ensure_ascii=False))


if __name__ == '__main__':
    main()
