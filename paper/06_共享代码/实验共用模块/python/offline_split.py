"""Versioned, hash-bound complete-shift split identities; no model or RPC code.

V1 callers retain their existing scene_group semantics. V2 can use distinct
realizations of one physical task only after every identity is in a frozen,
source-reviewed manifest. Historical records are mapped without rewriting them.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path

CONTRACT = 'mining-offline-split-manifest-v2'
PARTITIONS = ('train', 'validation', 'test')
POLICIES = (0, 1, 2, 4)
QUOTA = {'train': 35, 'validation': 5, 'test': 10}
JAVA_MASK = (1 << 48) - 1
JAVA_MULTIPLIER = 0x5DEECE66D
GEOLOGY_MULTIPLIER = 0x9E3779B97F4A7C15
DEFAULT_MANIFEST = (Path(__file__).resolve().parents[1] / 'change_plans' /
                    'offline_data_readiness_001' / 'campaign_003' / 'split_manifest.json')
IDENTITY_FIELDS = ('run_index', 'experiment_seed', 'scene_id', 'partition',
                   'physical_scene_family', 'realization_group_id',
                   'runtime_scene_sha256', 'event_bundle')


def _require(ok, message):
    if not ok:
        raise ValueError('offline split: ' + message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def episode_id(run_index, seed):
    return f'dispatch-production-wait-shortfall-v2-run-{run_index}-seed-{seed}'


def java_random_states(seed):
    """Constructor's effective 48-bit state for engine and site RNGs 0..3.

The low 48 bits are unchanged by Java long overflow in the geological xor.
Do not treat distinct states as a proof of statistical independence.
"""
    return {'engine': (seed ^ JAVA_MULTIPLIER) & JAVA_MASK,
            **{f'grade_site_{i}': (seed ^ (GEOLOGY_MULTIPLIER * (i + 1)) ^ JAVA_MULTIPLIER) & JAVA_MASK
               for i in range(4)}}


def _verify_legacy(entry, *, allow_pending):
    path, digest = entry.get('admission_path'), entry.get('admission_sha256')
    if allow_pending and digest is None:
        _require(bool(path), 'pending legacy admission path required')
        return
    _require(path and _hash(digest), 'legacy admission must be hash bound')
    _require(sha256(path) == digest, 'legacy admission hash mismatch')
    admission = json.loads(Path(path).read_text())
    _require(admission.get('status') == 'admitted' and admission.get('formal_training_data') is True,
             'legacy record is not formally admitted')
    _require(admission.get('behavior', {}).get('policy_id') == entry['policy_id'],
             'legacy behavior differs from manifest')
    matches = [j for j in admission.get('jobs', [])
               if j.get('run_index') == entry['run_index'] and j.get('experiment_seed') == entry['experiment_seed']]
    _require(len(matches) == 1 and matches[0].get('partition') == 'train',
             'legacy admission identity/partition differs')
    _require(matches[0].get('runtime_scene_sha256') == entry['runtime_scene_sha256'],
             'legacy runtime scene differs')


def validate_manifest(manifest, *, require_frozen=True):
    """Validate whole 200-shift design, including files and legacy evidence.

require_frozen=False permits only missing *legacy admission hashes* while root
prepares the freeze. It does not weaken quotas, identity or source checks.
"""
    _require(isinstance(manifest, dict) and manifest.get('contract') == CONTRACT, 'manifest contract required')
    _require(manifest.get('status') in ('pending_legacy_admissions', 'frozen'), 'unknown manifest status')
    if require_frozen:
        _require(manifest['status'] == 'frozen', 'manifest is not frozen')
    _require(manifest.get('per_policy_quota') == QUOTA and manifest.get('total_episodes') == 200,
             'approved 200-shift quota required')
    _require(manifest.get('policies') == list(POLICIES), 'approved policy pool required')
    _require(isinstance(manifest.get('physical_scene_family'), str) and manifest['physical_scene_family'],
             'physical family required')
    _require(_hash(manifest.get('runtime_scene_sha256')), 'runtime fingerprint required')
    review = manifest.get('random_source_review', {})
    _require(review.get('java_random_constructor') == 'java.util.Random(long)' and
             review.get('geology_xor_multiplier') == hex(GEOLOGY_MULTIPLIER) and
             review.get('site_count') == 4 and review.get('external_trace_reuse') is False and
             review.get('different_seeds_alone_prove_independence') is False,
             'explicit reviewed random-source semantics required')
    sources = manifest.get('source_proof', [])
    _require(isinstance(sources, list) and {'startup_random_source', 'geology_and_service_random_sources',
                                         'random_source_review'} <= {s.get('role') for s in sources},
             'random-source proof required')
    for source in sources:
        _require(source.get('path') and _hash(source.get('sha256')), 'invalid source proof')
        _require(sha256(source['path']) == source['sha256'], 'source proof changed: ' + source['role'])
    entries = manifest.get('episodes')
    _require(isinstance(entries, list) and len(entries) == 200, '200 complete episode assignments required')
    counts = Counter(); owners = {}; ids = set(); runs = set(); seeds = set(); rng_states = {}; legacy = Counter()
    for e in entries:
        _require(isinstance(e, dict) and e.get('policy_id') in POLICIES and e.get('partition') in PARTITIONS,
                 'invalid manifest policy/partition')
        run, seed = e.get('run_index'), e.get('experiment_seed')
        _require(type(run) is int and run > 0 and type(seed) is int and 0 < seed < 1 << 53,
                 'invalid integer run/seed (RPC exact integer range required)')
        identity = episode_id(run, seed)
        _require(e.get('episode_id') == identity and identity not in ids and run not in runs and seed not in seeds,
                 'duplicate or inconsistent global episode/run/seed')
        ids.add(identity); runs.add(run); seeds.add(seed)
        for source, state in java_random_states(seed).items():
            _require(state not in rng_states, 'Java Random effective 48-bit collision')
            rng_states[state] = (identity, source)
        _require(e.get('physical_scene_family') == manifest['physical_scene_family'] and
                 e.get('runtime_scene_sha256') == manifest['runtime_scene_sha256'], 'physical scene mismatch')
        group = e.get('realization_group_id')
        _require(isinstance(group, str) and group and e.get('input_source') == 'fresh_shift_rng_no_external_trace',
                 'explicit realization/input source required')
        _require(owners.setdefault(group, e['partition']) == e['partition'], 'realization group leakage')
        _require(isinstance(e.get('scene_id'), str) and e['scene_id'], 'scene id required')
        _require(Path(e.get('event_bundle', '')).is_absolute() and
                 Path(e['event_bundle']).name == identity, 'event identity differs')
        _require(isinstance(e.get('recording_id'), str) and e['recording_id'], 'recording identity required')
        counts[e['policy_id'], e['partition']] += 1
        if e.get('kind') == 'legacy_fixed_train':
            _require(e['partition'] == 'train' and e.get('observed_before_freeze') is True,
                     'observed legacy episode must stay train')
            legacy[e['policy_id']] += 1
            _verify_legacy(e, allow_pending=not require_frozen and manifest['status'] != 'frozen')
        else:
            _require(e.get('kind') == 'planned', 'unknown manifest episode kind')
    _require(counts == Counter({(p, part): n for p in POLICIES for part, n in QUOTA.items()}),
             'per-policy train/validation/test quota mismatch')
    _require(legacy == Counter({p: 1 for p in POLICIES}), 'four first episodes must be fixed train')
    return manifest


def load_manifest(split_spec):
    _require(isinstance(split_spec, dict) and set(split_spec) == {'version', 'manifest_path', 'manifest_sha256'}
             and split_spec['version'] == 2, 'explicit supported split_protocol required')
    _require(_hash(split_spec.get('manifest_sha256')), 'manifest byte hash required')
    path = Path(split_spec['manifest_path'])
    _require(path.is_absolute() and sha256(path) == split_spec['manifest_sha256'], 'manifest hash mismatch')
    return validate_manifest(json.loads(path.read_text()))


def _entry_for_job(manifest, job, *, policy_id=None, recording_id=None, legacy=False):
    matches = [e for e in manifest['episodes']
               if e['run_index'] == job.get('run_index') and e['experiment_seed'] == job.get('experiment_seed')]
    _require(len(matches) == 1, 'job not in frozen manifest')
    entry = matches[0]
    if policy_id is not None:
        _require(entry['policy_id'] == policy_id, 'job behavior not in frozen manifest')
    if recording_id is not None:
        _require(entry['recording_id'] == recording_id, 'job recording not in frozen manifest')
    fields = ('run_index', 'experiment_seed', 'scene_id', 'partition', 'runtime_scene_sha256') if legacy else IDENTITY_FIELDS
    _require(all(job.get(k) == entry[k] for k in fields), 'job assignment differs from frozen manifest')
    if not legacy:
        _require(entry['kind'] == 'planned' and job.get('scene_group') == entry['realization_group_id'],
                 'scene_group must equal reviewed realization group')
    return entry


def validate_jobs(cfg):
    """Additional V2 gate; call after existing config validation. V1 is untouched."""
    if 'split_protocol' not in cfg:
        _require(not any({'physical_scene_family', 'realization_group_id', 'split_manifest_sha256'} & set(job)
                         for job in cfg.get('jobs', [])), 'orphan V2 job fields without split_protocol')
        return None
    spec = cfg['split_protocol']; manifest = load_manifest(spec)
    _require(cfg.get('behavior', {}).get('name') == 'anylogic_native', 'V2 requires native behavior provenance')
    jobs = cfg.get('jobs')
    _require(isinstance(jobs, list) and jobs, 'jobs required')
    seen = set()
    for job in jobs:
        _require(job.get('split_manifest_sha256') == spec['manifest_sha256'], 'job manifest differs')
        entry = _entry_for_job(manifest, job, policy_id=cfg['behavior']['policy_id'],
                               recording_id=cfg.get('recording', {}).get('recording_id'))
        _require(entry['episode_id'] not in seen, 'duplicate job')
        seen.add(entry['episode_id'])
    return manifest


def group_assignments(episodes, jobs, split_spec=None, *, admission_path=None):
    """Return one verified group per episode, preserving input order.

New V2 records require split_spec saved in the admission. Legacy records may
be recognized through DEFAULT_MANIFEST only if its frozen mapping binds the
exact admission bytes. Otherwise V1's original scene_group remains in force.
"""
    _require(len(episodes) == len(jobs), 'episode/job count mismatch')
    legacy = split_spec is None
    if legacy:
        _require(not any({'physical_scene_family', 'realization_group_id', 'split_manifest_sha256'} & set(job)
                         for job in jobs), 'orphan V2 job fields without split_protocol')
        if admission_path is None or not DEFAULT_MANIFEST.exists():
            return [e['scene_group'] for e in episodes]
        candidate = json.loads(DEFAULT_MANIFEST.read_text())
        if candidate.get('status') != 'frozen':
            return [e['scene_group'] for e in episodes]
        digest = sha256(admission_path)
        named = [e for e in candidate['episodes']
                 if e.get('admission_path') and Path(e['admission_path']).resolve() == Path(admission_path).resolve()]
        if named:
            _require(all(e.get('admission_sha256') == digest for e in named), 'legacy admission hash mismatch')
        mapped = [e for e in candidate['episodes'] if e.get('admission_sha256') == digest]
        if not mapped:
            return [e['scene_group'] for e in episodes]
        manifest = validate_manifest(candidate)
    else:
        manifest = load_manifest(split_spec)
    result = []
    for episode, job in zip(episodes, jobs):
        entry = _entry_for_job(manifest, job, legacy=legacy)
        _require(episode.get('episode_id') == entry['episode_id'] and
                 episode.get('experiment_seed') == entry['experiment_seed'] and
                 episode.get('experiment_run_index') == entry['run_index'], 'stored episode identity differs')
        if legacy:
            _require(entry['kind'] == 'legacy_fixed_train' and admission_path is not None and
                     sha256(admission_path) == entry['admission_sha256'], 'legacy mapping is not hash bound')
        else:
            _require(job.get('split_manifest_sha256') == split_spec['manifest_sha256'] and
                     episode.get('scene_group') == entry['realization_group_id'], 'stored realization differs')
        result.append(entry['realization_group_id'])
    return result


def group_ids(episodes, jobs, split_spec=None, *, admission_path=None):
    """Verified group set for pretraining train/validation leakage checks."""
    return set(group_assignments(episodes, jobs, split_spec, admission_path=admission_path))
