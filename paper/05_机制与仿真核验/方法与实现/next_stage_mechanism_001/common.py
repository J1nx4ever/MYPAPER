"""Read-only helpers for the post-LR mechanism study (no runtime admission)."""
from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import sys

PROJECT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT))
PLAN = Path(__file__).resolve().parent
OUT = PROJECT / 'reports/token_attention_001/business_graph_summary_009/next_stage_mechanism_001'
CORE = PROJECT.parents[1] / 'python/episode_budget_ppo.py'
CORE_SHA = '8e4cfa31c622f5e33756e304eea3f7526547db72ae93f7d9f00808845d61ef96'
SUITES = {
    '3e-4': PLAN.parent/'checkpoint_evaluation_001/prepared_001/suite.json',
    '1e-4': PLAN.parent/'optimization_stability_001/paired_evaluation_001/prepared_001/suite.json',
}
AUDITS = {
    '3e-4': OUT.parent/'checkpoint_evaluation_001/completion_independent_audit.json',
    '1e-4': OUT.parent/'optimization_stability_001/paired_evaluation_001/completion_independent_audit.json',
}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def ref(path): return {'path': str(Path(path).resolve()), 'sha256': sha(path)}
def read(path): return json.loads(Path(path).read_bytes())
def save(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/name).open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')

def csv_save(name, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('x', encoding='utf-8-sig', newline='') as f:
        w=csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

def module(name, path):
    spec=importlib.util.spec_from_file_location(name, path)
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def sources():
    assert sha(CORE)==CORE_SHA
    result={}
    for lr,p in SUITES.items():
        s=read(p); a=read(AUDITS[lr]); assert a['status'].startswith('PASS')
        assert a['suite']==ref(p)
        for r in s['parent_artifacts'].values(): assert ref(r['path'])==r
        result[lr]=(s, {r['run_index']:r for r in a['per_run']})
    return result

def cohort():
    """Union of failed seed/window pairs; include BOTH policies at each pair."""
    src=sources(); failed={}; selected=[]
    for lr,(s,a) in src.items():
        with (Path(s['python_output'])/'reports/per_window.csv').open() as f:
            windows=list(csv.DictReader(f))
        for w in windows:
            g=float(w['grade'])
            if not .26-1e-12<=g<=.28+1e-12:
                rid=int(w['run_index']); j=next(j for j in s['jobs'] if j['run_index']==rid)
                failed[(j['ordinal'],int(w['window']))]=True
    for ordinal,window in sorted(failed):
        pair=[]
        for lr,(s,a) in src.items():
            j=next(j for j in s['jobs'] if j['ordinal']==ordinal)
            assert ref(j['material_journal'])==a[j['run_index']]['material_journal']
            raw=Path(s['python_output'])/'evaluation/episodes'/f"run_{j['run_index']}"
            assert ref(raw/'manifest.json')==a[j['run_index']]['raw_manifest']
            pair.append(j)
            selected.append(dict(lr=lr,ordinal=ordinal,window=window,job=j,raw=raw,
                                 suite=s,source_audit=a[j['run_index']]))
        assert pair[0]['experiment_seed']==pair[1]['experiment_seed']
        assert pair[0]['actor_sampling_seed']==pair[1]['actor_sampling_seed']
    assert len(failed)==10 and len(selected)==20
    return selected
