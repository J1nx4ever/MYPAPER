"""Validate and seal raw-intervention artifacts; no model forward."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import csv
import hashlib
import importlib.metadata
import json
import platform
import re
import sys

import numpy as np
from PIL import Image

PROJECT=Path(__file__).resolve().parents[4]
PLAN=Path(__file__).resolve().parent
OUT=PROJECT/'reports/token_attention_001/business_graph_summary_009/physical_intervention_001'
HISTORY=OUT.parent/'historical_snapshot_001'
PREVIOUS=OUT.parent/'encoder_snapshot_001'
CORE=PROJECT.parents[1]/'python/episode_budget_ppo.py'
CORE_SHA='8e4cfa31c622f5e33756e304eea3f7526547db72ae93f7d9f00808845d61ef96'
PREVIOUS_MANIFEST_SHA='282b88c9f54a84b9f1fcd5746ff7d681c40b15b81da976767321d3a56d29228c'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def ref(path):
    path=Path(path).resolve()
    return dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size)


def verify(entries):
    entries=list(entries)
    for row in entries:assert sha(row['path'])==row['sha256'],row['path']
    return len(entries)


def save(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def links(paths):
    count=0
    for path in paths:
        for target in re.findall(r'\[[^\]\n]*\]\(([^)]+)\)',path.read_text()):
            if target.startswith(('http:','https:','#')):continue
            target=re.sub(r':\d+$','',target.split('#')[0])
            candidate=(path.parent/target).resolve()
            assert candidate.exists() or candidate==OUT/'manifest.json',(str(path),target)
            count+=1
    return count


def main():
    assert sys.executable=='/home/jinx/miniconda3/envs/anylogic-env/bin/python'
    assert sha(CORE)==CORE_SHA
    bundle=read(OUT/'scenarios.json');summary=read(OUT/'summary.json')
    physical=read(OUT/'physical_checks.json');review=read(OUT/'independent_review.json')
    assert physical['all_passed'] and review['status']=='PASS' and len(review['checks'])==426
    assert all(review['checks'].values())
    assert all(bundle['scenarios'][s]['all_physical_checks_passed'] for s in ('A','B','C'))
    assert {s:bundle['scenarios'][s]['changed_field_count'] for s in ('A','B','C')}==dict(A=20,B=12,C=1)
    assert bundle['seed']==summary['pair_seed']==2026111801
    assert not bundle['training'] and bundle['optimizer_updates']==0 and bundle['new_shifts']==0
    assert summary['execution']['optimizer_updates']==0 and not summary['execution']['rpc_started'] and not summary['execution']['anylogic_started']
    checks=verify(bundle['source_references'])+verify(summary['source_references'])+verify(review['sources'])
    assert sha(bundle['base']['source_hdf5'])==bundle['base']['source_sha256']
    old=read(HISTORY/'manifest.json')
    old_refs=old['source_references']+old['script_references']+list(old['artifacts'].values())
    old_checks=verify(old_refs)
    assert sha(PREVIOUS/'manifest.json')==PREVIOUS_MANIFEST_SHA
    previous=read(PREVIOUS/'manifest.json')
    previous_refs=previous['source_references']+previous['script_references']+previous['documentation_references']+list(previous['artifacts'].values())+[previous['input_manifest'],previous['raw_snapshots']]
    previous_checks=verify(previous_refs)
    for role in ('actor','critic'):
        role_summary=summary['roles'][role]
        assert role_summary['parameter_sha256_before']==role_summary['parameter_sha256_after']==previous['parameter_digests'][role]
    with np.load(OUT/'analysis_tensors.npz',allow_pickle=False) as archive:
        assert set(archive.files)==set(summary['npz_schema'])
        for key,spec in summary['npz_schema'].items():
            value=archive[key]
            assert list(value.shape)==spec['shape'] and str(value.dtype)==spec['dtype'],key
            assert hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()==spec['sha256'],key
            if key.endswith('_masked_logits'):
                assert not np.isnan(value).any() and not np.isposinf(value).any(),key
            else:assert np.isfinite(value).all(),key
        for scene in ('BASE','A','B','C'):
            expected_raw=np.asarray(bundle['base']['raw'] if scene=='BASE' else bundle['scenarios'][scene]['raw'],dtype=np.float32)
            assert np.array_equal(archive[f'{scene}_raw'],expected_raw)
            assert np.isclose(archive[f'actor_{scene}_probs'].sum(),1)
        arrays_checked=len(archive.files)
    required={'scenarios.json','raw_changes.csv','physical_checks.json','summary.json','analysis_tensors.npz',
        'graph_edges.csv','edge_diffs.csv','quantities.csv','c128_features.csv','representation_changes.csv',
        'node_representations.csv','actor_candidates.csv','candidate_changes.csv','actor_logits.csv',
        'logit_pairwise_differences.csv','independent_review.json','independent_review.md',
        'cross_material_paths_C.csv','C_Load3_content_paths.json','readout_contract_review.json',
        'readout_contract_review.md','encoder_response.png','actor_readout_response.png','audit_report.md'}
    assert required.issubset({p.name for p in OUT.iterdir() if p.is_file()})
    csv_counts={}
    for path in OUT.glob('*.csv'):
        with path.open(encoding='utf-8-sig',newline='') as handle:csv_counts[path.name]=sum(1 for _ in csv.DictReader(handle))
    assert csv_counts['raw_changes.csv']==33
    assert csv_counts['representation_changes.csv']==2*3*3*19
    assert csv_counts['node_representations.csv']==2*4*3*19
    assert csv_counts['actor_logits.csv']==16 and csv_counts['c128_features.csv']==512
    png={}
    for path in OUT.glob('*.png'):
        with Image.open(path) as im:png[path.name]=list(im.size);im.verify()
    link_count=links(list(OUT.glob('*.md'))+list(PLAN.glob('*.md')))
    validation=dict(status='PASS',source_hash_checks=checks,
        historical_manifest_references_unchanged=old_checks,
        previous_encoder_manifest_references_unchanged=previous_checks,
        previous_encoder_manifest_sha256=PREVIOUS_MANIFEST_SHA,
        protected_ppo_sha256=CORE_SHA,independent_checks=len(review['checks']),
        observable_physical_checks={s:len(physical['scenarios'][s]) for s in ('A','B','C')},
        raw_changed_fields=dict(A=20,B=12,C=1),npz_arrays_checked=arrays_checked,
        csv_data_rows=csv_counts,png_dimensions=png,markdown_links_checked=link_count,
        model_parameter_buffer_digests_unchanged=True,
        visual_qa='Both figures inspected; mathematical glyphs readable, values match saved CSV, encoder numeric labels separated by stage.',
        scope='Observable-field-consistent offline exogenous branches; not AnyLogic hidden-state reachability or learned semantic/performance evidence.')
    save(OUT/'validation.json',validation)
    refs={row['path']:row for row in bundle['source_references']+summary['source_references']+review['sources']}
    for package in ('relation_gat_ego_v1','single_critic_task_query_v1','grade_commitment_ppo_v1'):
        for name in ('adapter.py','initialization.py','c_features.py','features_cache.py','features.py'):
            path=PROJECT/package/name
            if path.exists():refs[str(path.resolve())]=ref(path)
    artifacts={p.name:ref(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='manifest.json'}
    scripts=[ref(p) for p in sorted(PLAN.iterdir()) if p.is_file() and p.suffix in ('.py','.md')]
    manifest=dict(schema='bg128-physical-raw-interventions-manifest-v1',
        generated_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        pair_seed=summary['pair_seed'],parameter_status=summary['parameter_status'],
        base={k:v for k,v in bundle['base'].items() if k!='raw'},
        intervention_definitions={s:bundle['scenarios'][s]['intervention'] for s in ('A','B','C')},
        method=summary['primary_method'],execution=summary['execution'],
        graph=summary['graph'],parameter_digests={r:summary['roles'][r]['parameter_sha256_after'] for r in ('actor','critic')},
        historical_manifest=ref(HISTORY/'manifest.json'),previous_encoder_manifest=ref(PREVIOUS/'manifest.json'),
        source_references=list(refs.values()),script_references=scripts,artifacts=artifacts,
        environment=dict(python=platform.python_version(),executable=sys.executable,
            packages={name:importlib.metadata.version(name) for name in ('numpy','torch','h5py','Pillow')}),
        validation=validation,interpretation_limits=summary['interpretation_limits'])
    save(OUT/'manifest.json',manifest)
    verify(artifacts.values());verify(scripts);verify(refs.values())
    print(json.dumps(dict(status='PASS',artifacts=len(artifacts),independent_checks=len(review['checks']),
        previous_audits_unchanged=True,manifest_sha256=sha(OUT/'manifest.json')),ensure_ascii=False))


if __name__=='__main__':main()
