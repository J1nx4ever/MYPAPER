"""Reproduce Figure 5 from sealed, paired AnyLogic shifts and a frozen Actor.

Run with anylogic-env Python. Vector PDF uses the installed Pycairo and
GIR librsvg, and the 240 dpi PNG uses Poppler. Campaign data is read-only.
"""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from xml.sax.saxutils import escape

import h5py
import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = next(path for path in HERE.parents if (path / 'business_graph_summary_v1').is_dir())
WORKSPACE = ROOT.parents[1]
sys.path[:0] = [str(ROOT), str(WORKSPACE)]

from business_graph_summary_campaign_v1.budget import _payload, connection, digest
from business_graph_summary_v1.admission import frozen_actor
from business_graph_summary_v1.graph import GROUP_NAMES, NODE_NAMES, RELATION_NAMES


RUN = ROOT / 'runs/bg128-continuous-mixed-20261119-001'
SCENES = ('nominal', 'waste_swap')
DECISIONS = (0, 8)
WIDTH, HEIGHT = 1120, 1215
PANEL = ((52, '标称工况', (1, 1, 1, 0)), (580, '矿废互换', (1, 1, 0, 1)))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha(path):
    return sha256(path.read_bytes()).hexdigest()


def reference(path):
    return {'path': str(path.relative_to(WORKSPACE)), 'sha256': file_sha(path)}


def read_json(path):
    return json.loads(path.read_bytes())


def read_snapshot(record, job, actor_sha):
    receipt = record['receipt']
    require(receipt['normal_completion'] and receipt['four_windows'] == 4,
            'ledger receipt must close an eight-hour shift')
    ref = receipt['references']['episode']
    path = Path(ref['path']).resolve()
    expected = Path(job['python_output']) / 'material' / f"run_{job['run_index']}.h5"
    require(path == expected and file_sha(path) == ref['sha256'], 'ledger dataset binding differs')
    manifest = path.with_name(path.name + '.manifest.json')
    manifest_ref = record['dataset_ref']
    require(manifest_ref == receipt['references']['episode_manifest']
            and Path(manifest_ref['path']) == manifest and file_sha(manifest) == manifest_ref['sha256'],
            'ledger material manifest binding differs')
    with h5py.File(path) as source:
        require(source.attrs['status'] == 'complete', 'dataset is not complete')
        identity = json.loads(source.attrs['identity_json'])
        expected_identity = dict(run_index=job['run_index'], environment_seed=job['experiment_seed'],
                                 actor_sampling_seed=job['actor_sampling_seed'], scene_id=job['scene_id'],
                                 policy_sha256=actor_sha)
        require(all(identity[key] == value for key, value in expected_identity.items()),
                'material dataset identity differs')
        require(source['transitions/action'].shape[0] > max(DECISIONS), 'missing decision snapshot')
        states = source['states'][list(DECISIONS)]
    require(states.shape == (2, 1519) and states.dtype == np.float32 and np.all(states[:, 17] == 0),
            'expected two nonterminal decision observations')
    return states, {'dataset': reference(path), 'manifest': reference(manifest), 'run_index': job['run_index']}


def collect(limit):
    arms = {scene: RUN / 'evaluation' / scene / 'new' for scene in SCENES}
    docs = {scene: arm / 'phase_config.json' for scene, arm in arms.items()}
    phases = {scene: read_json(path) for scene, path in docs.items()}
    actors = {scene: arm / 'actor.pt' for scene, arm in arms.items()}
    actor_file = actors['nominal']
    actor_sha = file_sha(actor_file)
    actor, actor_digest = frozen_actor(actor_file)
    jobs = {}
    for scene in SCENES:
        manifest = read_json(arms[scene] / 'actor.pt.manifest.json')
        phase = phases[scene]
        require(manifest['update_index'] == 32 and manifest['actor_digest'] == actor_digest
                and manifest['sha256'] == file_sha(actors[scene]) == actor_sha, 'scenes did not use the same u32')
        require(phase['policy_arm'] == 'new' and phase['scenario_id'] == scene, 'phase identity differs')
        jobs[scene] = {job['pair_ordinal']: job for job in phase['jobs']}
        require(len(jobs[scene]) == 100 and set(jobs[scene]) == set(range(100)), 'planned slots differ')

    with connection(RUN / 'budget.sqlite', readonly=True) as database:
        ledger = {run: (status, _payload((payload, checksum))) for run, status, payload, checksum in
                  database.execute('SELECT run,status,payload,checksum FROM slots')}
    records = {scene: {slot: ledger[job['run_index']][1] for slot, job in roster.items()
                       if ledger[job['run_index']][0] == 'COMPLETE'} for scene, roster in jobs.items()}
    common = sorted(records['nominal'].keys() & records['waste_swap'].keys())
    require(bool(common), 'no completed real same-slot pair is available')
    slots = common if limit is None else common[:limit]
    require(limit is None or len(slots) == limit, 'requested pair count exceeds sealed common slots')
    paired = []
    snapshots = {str(k): {scene: [] for scene in SCENES} for k in DECISIONS}
    for slot in slots:
        seeds = {(jobs[scene][slot]['experiment_seed'], jobs[scene][slot]['actor_sampling_seed']) for scene in SCENES}
        require(len(seeds) == 1, 'planned paired seeds differ')
        environment_seed, sampling_seed = seeds.pop()
        pair = dict(pair_ordinal=slot, environment_seed=environment_seed, actor_sampling_seed=sampling_seed)
        states = {}
        for scene, (_, _, roles) in zip(SCENES, PANEL, strict=True):
            job, record = jobs[scene][slot], records[scene][slot]
            require(record['status'] == 'COMPLETE' and record['identity_sha256'] == digest(job),
                    'ledger COMPLETE job identity differs')
            states[scene], pair[scene] = read_snapshot(record, job, actor_sha)
            load_roles = states[scene][:, 1107:1295].reshape(2, 4, 47)[:, :, 1]
            require(np.all(load_roles == np.asarray(roles)), 'observed ore/waste role differs')
        initial_changes = np.flatnonzero(states['nominal'][0] != states['waste_swap'][0]).tolist()
        source_fields = {1107 + 47 * station + field for station in (2, 3) for field in (1, 2)}
        require(set(initial_changes) == source_fields, 'first snapshot roles and means must be the only changes')
        pair['first_decision_changed_state_indices'] = initial_changes
        pair['paired_times_seconds'] = {}
        for pos, decision in enumerate(DECISIONS):
            a, b = (float(states[scene][pos, 0]) * 28800 for scene in SCENES)
            if not np.isclose(a, b, atol=1e-4, rtol=0):
                continue
            pair['paired_times_seconds'][str(decision)] = [a, b]
            for scene in SCENES:
                snapshots[str(decision)][scene].append(states[scene][pos])
        paired.append(pair)
    require(all(snapshots[str(k)]['nominal'] for k in DECISIONS), 'no equal-time pairs at a requested decision')
    meta = dict(schema='bg128-figure5-paired-relation-attention-v1', data_kind='real_anylogic_rpc',
                captured_utc=datetime.now(timezone.utc).isoformat(),
                actor=dict(update=32, digest=actor_digest, artifacts={s: reference(p) for s, p in actors.items()}),
                phase_configs={scene: reference(path) for scene, path in docs.items()},
                ledger_path=str((RUN / 'budget.sqlite').relative_to(WORKSPACE)),
                planned_pairs=100, complete_shifts_by_scene={scene: len(rows) for scene, rows in records.items()},
                eligible_complete_pairs=len(common), selected_complete_pairs=len(paired), pairs=paired,
                same_time_tolerance_seconds=0.0001,
                attention_rule='softmax over incoming edges per destination, relation group and head',
                displayed_value='layer 2 mean across four heads, then mean across snapshots where the edge exists',
                empty_cell='not observed as a factual edge in any sampled snapshot',
                time_mapping='raw state shiftElapsed times 28800 s',
                post_first_decision_note='Observed trajectories may diverge after different assignments. No edge or weight intervention.')
    return actor, snapshots, meta


def aggregate(actor, snapshots, relation, src_names, dst_names):
    states = torch.from_numpy(np.stack(snapshots))
    with torch.inference_mode():
        _, diagnostics = actor.state_encoder.scene(states, diagnostic=True)
    graph = diagnostics['graph']
    src, dst = graph.edge_index
    relation_id = RELATION_NAMES.index(relation)
    select = (graph.edge_type == relation_id).nonzero(as_tuple=True)[0]
    n_nodes = len(NODE_NAMES)
    weights = {str(layer): diagnostics[f'layer{layer}']['weights'] for layer in (1, 2)}
    seen, by_edge = set(), defaultdict(list)
    for edge in select.tolist():
        sample, source = divmod(int(src[edge]), n_nodes)
        target = int(dst[edge]) % n_nodes
        identity = (sample, source, target)
        require(identity not in seen, 'duplicate factual edge in a decision')
        seen.add(identity)
        by_edge[NODE_NAMES[source], NODE_NAMES[target]].append([weights[k][edge].tolist() for k in weights])
    for layer_weights in weights.values():
        edge_sums = torch.zeros((len(states) * n_nodes, 4)).index_add(0, dst[select], layer_weights[select])
        used = edge_sums[edge_sums.any(dim=1)]
        require(bool(torch.allclose(used, torch.ones_like(used), atol=1e-6, rtol=1e-6)),
                'shown forward group shares a destination with another relation or is not normalized')
    matrix = []
    for source in src_names:
        row = []
        for target in dst_names:
            values = by_edge.get((source, target), [])
            heads = np.asarray(values).mean(0) if values else None
            row.append(dict(source=source, destination=target, support=len(values),
                            mean_alpha_layer2=float(heads[1].mean()) if heads is not None else None,
                            mean_alpha_by_head_by_layer=heads.tolist() if heads is not None else None))
        matrix.append(row)
    all_groups = {}
    for group, name in enumerate(GROUP_NAMES):
        destinations = dst[graph.edge_group == group]
        degree = torch.bincount(destinations, minlength=len(states) * n_nodes).reshape(len(states), n_nodes)
        edges = degree.sum(1)
        all_groups[name] = dict(directed_edges_per_snapshot_min=int(edges.min()),
                                directed_edges_per_snapshot_max=int(edges.max()),
                                destination_buckets_single_edge=int((degree == 1).sum()),
                                destination_buckets_multiple_edges=int((degree > 1).sum()))
    return dict(n=len(states), relation=relation, sources=src_names, destinations=dst_names,
                all_relation_groups=all_groups,
                time_range_seconds=[float(states[:, 0].min()) * 28800, float(states[:, 0].max()) * 28800],
                forward_edges_per_snapshot_range=[min(sum(x[0] == i for x in seen) for i in range(len(states))),
                                                  max(sum(x[0] == i for x in seen) for i in range(len(states)))],
                cells=matrix)


def heat_color(value):
    pale, deep = (233, 245, 248), (9, 83, 108)
    return '#' + ''.join(f'{round(a + (b - a) * value):02x}' for a, b in zip(pale, deep, strict=True))


def make_svg(meta):
    ink, muted, accent = '#112C3D', '#466477', '#B85A26'
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
           '<defs><pattern id="missing" width="8" height="8" patternUnits="userSpaceOnUse">'
           '<rect width="8" height="8" fill="#F4F7F8"/>'
           '<path d="M -2,2 L 2,-2 M 0,8 L 8,0 M 6,10 L 10,6" stroke="#D6E0E5" stroke-width="1"/>'
           '</pattern><linearGradient id="alpha"><stop offset="0%" stop-color="#E9F5F8"/>'
           '<stop offset="100%" stop-color="#09536C"/></linearGradient></defs>']

    def rect(x, y, w, h, color, *, stroke=None, rx=0):
        line = '' if stroke is None else f' stroke="{stroke}" stroke-width="1"'
        svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{color}"{line}/>')

    def text(x, y, message, size, *, color=ink, weight=400, anchor='start'):
        svg.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" '
                   f'text-anchor="{anchor}" fill="{color}">{escape(message)}</text>')

    def line(x1, x2, y):
        svg.append(f'<line x1="{x1}" x2="{x2}" y1="{y}" y2="{y}" stroke="#CDDDE3"/>')

    def matrix(panel, data, top, *, material):
        x, label, roles = panel
        rect(x, top - 67, 488, 49, '#EAF2F5', rx=8)
        text(x + 18, top - 36, label, 21, weight=700)
        text(x + 469, top - 36, f"{data['n']} 班", 15, color=muted, anchor='end')
        cols = len(data['destinations'])
        cell_w = (488 - 121) / cols
        cell_h = 46 if material else 33
        for j, destination in enumerate(data['destinations']):
            label = {'Dump0': 'Dump0 破碎站', 'Dump1': 'Dump1 废石场'}.get(destination, destination)
            text(x + 121 + (j + 0.5) * cell_w, top + 9, label, 14, weight=700, anchor='middle')
        for i, (source, cells) in enumerate(zip(data['sources'], data['cells'], strict=True)):
            y = top + 18 + i * cell_h
            marker = accent if material and i in (2, 3) else muted
            if material and i in (2, 3):
                rect(x + 3, y + 8, 3, cell_h - 16, accent, rx=1)
            label = f"{source} {'矿石' if roles[i] else '废石'}" if material else source
            text(x + 13, y + cell_h / 2 + 4, label, 14, color=marker, weight=700 if material and i in (2, 3) else 400)
            for j, cell in enumerate(cells):
                x0 = x + 121 + j * cell_w
                value, count = cell['mean_alpha_layer2'], cell['support']
                if value is None:
                    rect(x0 + 1, y + 1, cell_w - 3, cell_h - 3, 'url(#missing)', rx=3)
                    continue
                rect(x0 + 1, y + 1, cell_w - 3, cell_h - 3, heat_color(value), rx=3)
                color = '#FFFFFF' if value >= .64 else '#153446'
                center = x0 + cell_w / 2
                text(center, y + (20 if material else 14), f'{value:.3f}', 14 if material else 12,
                     color=color, weight=700, anchor='middle')
                text(center, y + (36 if material else 27), f'{count}/{data["n"]}', 11 if material else 10,
                     color=color, anchor='middle')

    svg.append('<g font-family="Noto Sans CJK SC, sans-serif">')
    rect(0, 0, WIDTH, HEIGHT, '#FFFFFF')
    rect(52, 39, 39, 6, '#0B748D', rx=3)
    text(52, 89, '图 5  矿废互换下的关系注意力', 34, weight=700)
    text(52, 122, f"冻结混合训练 u32  ·  取前 {meta['selected_complete_pairs']} 对已封存同槽班次  ·  第 2 层，四头均值",
         16, color=muted)
    line(52, 1068, 144)
    text(52, 178, 'A   物料兼容  Load → Dump', 23, weight=700)
    text(1068, 178, '第 1 次派车前 · t = 0 s', 15, color=muted, anchor='end')
    for scene, panel in zip(SCENES, PANEL, strict=True):
        matrix(panel, meta['panels']['material'][scene], 260, material=True)
    line(52, 1068, 481)
    text(52, 516, 'B   已预留承诺  Truck → Load', 23, weight=700)
    first = meta['panels']['commitment']['nominal']
    lo, hi = first['time_range_seconds']
    timing = f't = {lo:.1f} s' if abs(lo - hi) < 1e-4 else f't = {lo:.1f}–{hi:.1f} s'
    text(1068, 516, f'第 9 次派车前 · 同槽同时间 · {timing}', 15, color=muted, anchor='end')
    for scene, panel in zip(SCENES, PANEL, strict=True):
        matrix(panel, meta['panels']['commitment'][scene], 597, material=False)
    line(52, 1068, 926)
    text(52, 957, '四组关系的有向边数', 17, weight=700)
    text(1068, 957, '每个快照的最小–最大边数', 13, color=muted, anchor='end')
    centers = (338, 530, 722, 914)
    for center, label in zip(centers, ('运输', '承诺', '服务', '物料兼容'), strict=True):
        text(center, 980, label, 14, color=muted, weight=700, anchor='middle')
    for idx, (key, scene, label) in enumerate(
        (('material', 'nominal', 'A  标称'), ('material', 'waste_swap', 'A  互换'),
         ('commitment', 'nominal', 'B  标称'), ('commitment', 'waste_swap', 'B  互换'))
    ):
        row_y = 1005 + idx * 22
        if idx % 2 == 0:
            rect(52, row_y - 16, 1016, 22, '#F3F7F8', rx=3)
        text(66, row_y, label, 13, color=muted)
        for center, group in zip(centers, GROUP_NAMES, strict=True):
            counts = meta['panels'][key][scene]['all_relation_groups'][group]
            low = counts['directed_edges_per_snapshot_min']
            high = counts['directed_edges_per_snapshot_max']
            text(center, row_y, str(low) if low == high else f'{low}–{high}',
                 13, color=ink, anchor='middle')
    line(52, 1068, 1087)
    text(52, 1116, '均值 α', 14, weight=700)
    rect(119, 1102, 254, 14, 'url(#alpha)', rx=3)
    for value in (0, .25, .5, .75, 1):
        text(119 + 254 * value, 1134, f'{value:g}', 11, color=muted, anchor='middle')
    rect(414, 1100, 27, 18, 'url(#missing)', rx=3)
    text(453, 1115, '未观察到边', 13, color=muted)
    text(609, 1115, '下方数字为边出现班数 / 面板总班数', 13, color=muted)
    text(52, 1162, '按目的节点、关系组和头归一化。颜色仅汇总实际存在的边，单一入边的权重为 1。', 13, color=muted)
    text(52, 1185, 'A 开局仅 Load2 / Load3 的矿废与名义品位不同。B 来自各自实际派车轨迹，含动作反馈。', 13, color=muted)
    svg.append('</g></svg>')
    return ''.join(svg).encode('utf-8')


def render(svg, pdf, png):
    """Render SVG directly into a vector Cairo PDF and rasterize that PDF."""
    sys.path.append('/usr/lib/python3/dist-packages')
    import cairo
    import gi

    gi.require_version('Rsvg', '2.0')
    gi.require_foreign('cairo')
    from gi.repository import Rsvg

    viewport = Rsvg.Rectangle()
    viewport.x, viewport.y, viewport.width, viewport.height = 0, 0, WIDTH, HEIGHT
    with cairo.PDFSurface(str(pdf), WIDTH, HEIGHT) as surface:
        context = cairo.Context(surface)
        Rsvg.Handle.new_from_data(svg).render_document(context, viewport)
        context.show_page()
    subprocess.run(['pdftoppm', '-r', '240', '-png', '-singlefile', str(pdf), str(png.with_suffix(''))], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-pairs', type=int, default=10, help='use this many completed paired slots')
    args = parser.parse_args()
    require(args.max_pairs is None or args.max_pairs > 0, 'pair limit must be positive')
    torch.set_num_threads(1)
    actor, snapshots, meta = collect(args.max_pairs)
    meta['panels'] = {
        'material': {scene: aggregate(actor, snapshots['0'][scene], 'compatible_material',
                                      list(NODE_NAMES[9:13]), list(NODE_NAMES[13:15])) for scene in SCENES},
        'commitment': {scene: aggregate(actor, snapshots['8'][scene], 'reserved_load',
                                        list(NODE_NAMES[:9]), list(NODE_NAMES[9:13])) for scene in SCENES},
    }
    for scene, (_, _, roles) in zip(SCENES, PANEL, strict=True):
        material = meta['panels']['material'][scene]
        require(all([cell['support'] for cell in row] == [material['n'] * role, material['n'] * (1 - role)]
                    for role, row in zip(roles, material['cells'], strict=True)), 'incompatible material edges')
        require(meta['panels']['commitment'][scene]['forward_edges_per_snapshot_range'] == [8, 8],
                'expected eight observed reservations before dispatch nine')
    base = HERE / 'figure5_relation_attention'
    render(make_svg(meta), base.with_suffix('.pdf'), base.with_suffix('.png'))
    base.with_suffix('.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f"Figure 5: {meta['selected_complete_pairs']}/100 real pairs, "
          f"{meta['panels']['commitment']['nominal']['n']} equal-time commitment pairs")


if __name__ == '__main__':
    main()
