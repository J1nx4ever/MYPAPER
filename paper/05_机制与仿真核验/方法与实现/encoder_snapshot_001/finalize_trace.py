"""Seal saved BG128 diagnostics; no model instantiation or forward."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import csv
import hashlib
import importlib.metadata
import json
import platform
import re

import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parents[4]
PLAN = Path(__file__).resolve().parent
OUT = PROJECT / 'reports/token_attention_001/business_graph_summary_009/encoder_snapshot_001'
HISTORY = OUT.parent / 'historical_snapshot_001'
CORE = PROJECT.parents[1] / 'python/episode_budget_ppo.py'
CORE_SHA = '8e4cfa31c622f5e33756e304eea3f7526547db72ae93f7d9f00808845d61ef96'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def ref(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)


def read_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def verify_references(entries):
    for entry in entries:
        assert sha(entry['path']) == entry['sha256'], entry['path']


def check_links(paths):
    checked = []
    for path in paths:
        for target in re.findall(r'\[[^\]\n]*\]\(([^)]+)\)', path.read_text()):
            if target.startswith(('http:', 'https:', '#')):
                continue
            destination = (path.parent / target.split('#')[0]).resolve()
            # This invocation generates the final manifest after all other checks.
            assert destination.exists() or destination == OUT / 'manifest.json', (str(path), target)
            checked.append(dict(document=str(path), target=target))
    return checked


def main():
    summary = read_json(OUT / 'summary.json')
    history_manifest = read_json(HISTORY / 'manifest.json')
    historical_refs = history_manifest['source_references'] + history_manifest['script_references'] + list(history_manifest['artifacts'].values())
    authenticated = read_json(OUT / 'authenticated_inputs.json')
    assert authenticated == [dict(path=r['path'], sha256=r['sha256']) for r in historical_refs]
    verify_references(historical_refs)
    verify_references(summary['source_references'])
    assert sha(HISTORY / 'manifest.json') == summary['validation']['input_manifest_sha256']
    assert sha(HISTORY / 'snapshots.json') == summary['validation']['raw_snapshot_sha256']
    assert sha(CORE) == CORE_SHA
    assert summary['pair_seed'] == 2026111801
    assert summary['optimizer_updates'] == 0 and not summary['train']
    assert not summary['rpc_started'] and not summary['anylogic_started']
    assert summary['validation']['serialized_raw_rows_exact_original_hdf']
    for role in ('actor', 'critic'):
        data = summary['roles'][role]
        assert data['parameter_sha256_before'] == data['parameter_sha256_after']
        assert data['global_bypass_exact']
    assert summary['roles']['actor']['parameter_sha256_after'] != summary['roles']['critic']['parameter_sha256_after']
    review = read_json(OUT / 'independent_review.json')
    assert review['status'] == 'PASS'

    npz_checks = {}
    for name, expected in summary['npz_schema'].items():
        with np.load(OUT / name, allow_pickle=False) as archive:
            assert set(archive.files) == set(expected)
            for key, spec in expected.items():
                value = archive[key]
                assert list(value.shape) == spec['shape'] and str(value.dtype) == spec['dtype'], (name, key)
                assert np.isfinite(value).all(), (name, key)
                if name == 'representations.npz':
                    role, snapshot, layer = key.split('_')
                    assert sha_array(value) == summary['roles'][role]['snapshots'][snapshot]['matrix_sha256'][layer]
            npz_checks[name] = len(archive.files)

    required = {
        'summary.json', 'authenticated_inputs.json', 'representations.npz', 'node_representations.csv',
        'representation_changes.csv', 'edge_messages.csv', 'group_messages.csv', 'message_evidence.csv',
        'message_tensors.npz', 'hybrid_attribution.csv', 'attribution_vectors.npz',
        'raw_sensitivity.json', 'raw_sensitivity.csv', 'raw_field_changes.csv', 'raw_quantity_changes.csv',
        'raw_sensitivity_notes.md', 'independent_review.json', 'independent_review.md',
        'path_interventions.csv', 'path_interventions.json', 'path_interventions.npz',
        'actor_H_layers.png', 'critic_H_layers.png', 'load_changes_and_attribution.png',
        'two_layer_message_flow.png', 'audit_report.md',
    }
    assert required.issubset({p.name for p in OUT.iterdir() if p.is_file()})
    csv_counts = {}
    for path in OUT.glob('*.csv'):
        with path.open(encoding='utf-8-sig', newline='') as handle:
            csv_counts[path.name] = sum(1 for _ in csv.DictReader(handle))
    assert csv_counts['node_representations.csv'] == 228
    assert csv_counts['representation_changes.csv'] == 114
    assert csv_counts['edge_messages.csv'] == summary['validation']['edge_rows'] == 704
    assert csv_counts['group_messages.csv'] == summary['validation']['group_rows'] == 608
    assert csv_counts['message_evidence.csv'] == summary['validation']['message_evidence_rows'] == 56
    png_sizes = {}
    for path in OUT.glob('*.png'):
        with Image.open(path) as im:
            png_sizes[path.name] = list(im.size)
            im.verify()
    links = check_links(list(OUT.glob('*.md')) + [PLAN / 'README.md', PLAN.parent / 'README.md', OUT.parent / 'README.md'])
    validation = dict(status='PASS', historical_reference_checks=len(historical_refs),
        original_history_manifest_unchanged=True, official_source_references_unchanged=True,
        protected_ppo_sha256=CORE_SHA, parameter_digests_unchanged=True,
        role_parameters_independent=True, finite_npz_arrays=npz_checks,
        csv_data_rows=csv_counts, png_dimensions=png_sizes, markdown_links_checked=len(links),
        independent_review_status=review['status'],
        independent_review_checks=len(review['checks']),
        numerical_validation=summary['validation'],
        visual_qa='All four figures inspected: glyphs readable, no clipped labels; values independently reviewed against saved tensors/CSV.',
        limitations='Fixed random initialization, one historical pair and one seed; numerical channels only, no learned semantics or performance evidence.')
    save_json(OUT / 'validation.json', validation)
    artifacts = {p.name: ref(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name != 'manifest.json'}
    scripts = [ref(p) for p in sorted(PLAN.iterdir()) if p.is_file() and p.suffix in ('.py', '.md')]
    manifest = dict(schema='bg128-encoder-snapshot-manifest-v1',
        generated_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        scope=summary['scope'], parameter_status=summary['parameter_status'], pair_seed=summary['pair_seed'],
        selection=summary['selection'],
        execution=dict(optimizer_updates=0, trained_checkpoint_loads=0, rpc_starts=0,
            anylogic_starts=0, new_physical_shifts=0, budget_registrations=0,
            final_actor_critic_head_forwards=0, c128_neural_fusion_forwards=0,
            encoder_and_internal_tensor_forwards=True),
        parameter_digests={role: summary['roles'][role]['parameter_sha256_after'] for role in ('actor', 'critic')},
        input_manifest=ref(HISTORY / 'manifest.json'), raw_snapshots=ref(HISTORY / 'snapshots.json'),
        source_references=summary['source_references'], script_references=scripts,
        documentation_references=[ref(PLAN.parent / 'README.md'), ref(OUT.parent / 'README.md')],
        environment=dict(python=platform.python_version(), executable='/home/jinx/miniconda3/envs/anylogic-env/bin/python',
            packages={name: importlib.metadata.version(name) for name in ('numpy', 'torch', 'h5py', 'Pillow')}),
        validation=validation, artifacts=artifacts)
    save_json(OUT / 'manifest.json', manifest)
    verify_references(artifacts.values())
    verify_references(scripts)
    print(json.dumps(dict(status='PASS', artifacts=len(artifacts), historical_references=len(historical_refs),
        independent_checks=len(review['checks']), manifest_sha256=sha(OUT / 'manifest.json')), ensure_ascii=False))


def sha_array(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


if __name__ == '__main__':
    main()
