"""Explicit, hash-bound reconciliation of one event recording label.

The source remains rejected by the default strict path. A fresh capsule copies
its recording bytes unchanged and binds the user's exception to one event
bundle. This never changes the event verifier or authorizes other differences.
"""
import copy
import hashlib
import json
from pathlib import Path
import shutil


CONTRACT = 'mining-offline-recording-identity-reconciliation-v1'
AUTHORIZATION_QUOTE = '虽然上次id是错的，你先归纳进去，本质是对的是policy2'
SUPPLEMENTARY_AUTHORIZATION_QUOTE = '将上次的policy2身份也改正确，可以放入MC进行，总计200班，训练140，验证20，测试40'
SIDECAR = 'identity_reconciliation.json'
CORE_FILES = ('acceptance_run_config.json', 'recording_manifest.json',
              'rpc.jsonl', 'transitions.jsonl', 'candidate/manifest.json',
              'candidate/samples.npz', 'candidate/episodes.json')
OPTIONAL_FILES = ('source_snapshot.json', 'collection_summary.json')
KEYS = {'contract', 'status', 'authorization', 'source_recording',
        'derived_recording', 'source_files_sha256', 'copied_files_sha256',
        'strict_rejection', 'binding', 'strict_identity_passed',
        'raw_data_rewritten'}
BINDING_KEYS = {'event_bundle', 'expected_recording_id', 'observed_recording_id',
                'config_sha256', 'episode_id', 'run_index', 'seed',
                'instrumentation_revision', 'event_manifest_sha256',
                'events_sha256'}


def _require(condition, message):
    if not condition:
        raise ValueError('identity reconciliation: '+message)


def _read(path):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    _require(type(value) is dict, 'JSON object required')
    return value


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def _inventory(directory):
    result = {}
    for item in sorted(directory.rglob('*')):
        _require(not item.is_symlink(), 'symlink in source recording')
        if item.is_file():
            result[str(item.relative_to(directory))] = _sha(item)
    return result


def _authorization(value):
    _require(type(value) is dict and set(value) == {'approved', 'authority', 'quote', 'supplementary_quote', 'date'},
             'explicit user authorization required')
    _require(value['approved'] is True and value['authority'] == 'user' and
             value['quote'] == AUTHORIZATION_QUOTE and value['date'] == '2026-10-30',
             'explicit user authorization required')
    _require(value['supplementary_quote'] == SUPPLEMENTARY_AUTHORIZATION_QUOTE,
             'explicit MC-use authorization required')


def _canonical(path):
    _require(isinstance(path, str) and path == str(Path(path).resolve()),
             'canonical absolute path required')
    return Path(path)


def _validate_source(sidecar):
    _require(type(sidecar) is dict and set(sidecar) == KEYS,
             'unsupported fields or additional alias')
    _require(sidecar['contract'] == CONTRACT and
             sidecar['status'] == 'authorized_single_recording_identity_reconciliation',
             'unsupported contract/status')
    _authorization(sidecar['authorization'])
    _require(sidecar['strict_identity_passed'] is False and
             sidecar['raw_data_rewritten'] is False, 'strict failure must remain explicit')
    source = _canonical(sidecar['source_recording'])
    derived = _canonical(sidecar['derived_recording'])
    _require(source != derived and not derived.is_relative_to(source) and
             not source.is_relative_to(derived), 'source and capsule must be independent')
    _require(not (source/'offline_admission.json').exists() and
             not (source/'ABORTED.json').exists() and
             not (source/SIDECAR).exists(), 'source already admitted/aborted/reconciled')
    inventory = _inventory(source)
    _require(type(sidecar['source_files_sha256']) is dict and
             inventory == sidecar['source_files_sha256'], 'source recording changed')
    copied = {name: inventory[name] for name in CORE_FILES}
    copied.update({name: inventory[name] for name in OPTIONAL_FILES if name in inventory})
    _require(sidecar['copied_files_sha256'] == copied, 'capsule file inventory mismatch')
    cfg = _read(source/'acceptance_run_config.json')
    manifest = _read(source/'recording_manifest.json')
    _require(len(cfg.get('jobs', [])) == 1, 'only one episode alias is permitted')
    job = cfg['jobs'][0]
    binding = sidecar['binding']
    _require(type(binding) is dict and set(binding) == BINDING_KEYS,
             'one exact binding required; additional alias is forbidden')
    bundle = _canonical(binding['event_bundle'])
    _require(bundle == Path(job['event_bundle']).resolve() and
             not bundle.is_relative_to(derived), 'event bundle path mismatch')
    em = _read(bundle/'events_manifest.json')
    _require(_sha(bundle/'events_manifest.json') == binding['event_manifest_sha256'] and
             _sha(bundle/'events.jsonl') == binding['events_sha256'] == em.get('events_sha256'),
             'event evidence changed')
    expected = cfg.get('recording', {}).get('recording_id')
    _require(isinstance(expected, str) and bool(expected) and
             expected == manifest.get('recording_id') == binding['expected_recording_id'],
             'expected recording identity mismatch')
    observed = em.get('recording_id')
    _require(isinstance(observed, str) and bool(observed) and observed != expected and
             observed == binding['observed_recording_id'],
             'observed identity mismatch or no exception is necessary')
    _require(_sha(source/'acceptance_run_config.json') == manifest.get('config_sha256') ==
             em.get('config_sha256') == binding['config_sha256'], 'configuration hash mismatch')
    _require(type(binding['run_index']) is int and type(binding['seed']) is int and
             binding['run_index'] == em.get('run_index') == job.get('run_index') and
             binding['seed'] == em.get('seed') == job.get('experiment_seed'), 'run/seed mismatch')
    episodes = manifest.get('episodes', [])
    candidate = _read(source/'candidate/manifest.json')
    episode_list = json.loads((source/'candidate/episodes.json').read_text())
    _require(len(episodes) == len(episode_list) == 1 and candidate.get('episodes') == 1 and
             binding['episode_id'] == em.get('episode_id') == episodes[0].get('episode_id') ==
             episode_list[0].get('episode_id'), 'episode mismatch')
    _require(binding['instrumentation_revision'] == em.get('instrumentation_revision') and
             isinstance(binding['instrumentation_revision'], str) and
             bool(binding['instrumentation_revision']), 'instrumentation mismatch')
    scope = manifest.get('scope')
    _require(scope in ('actual_anylogic_offline', 'synthetic_unit_test') and
             manifest.get('source_kind') == scope and
             em.get('source_kind') == ('synthetic_unit_test' if scope == 'synthetic_unit_test'
                                       else 'actual_anylogic'), 'recording source mismatch')
    if scope == 'actual_anylogic_offline':
        # This user's exception names one known run, not a reusable waiver for
        # identity failures in future collection campaigns.
        from .experiment_config import ROOT
        actual_source = ROOT/'runs/offline-native-002-p2-first-wait1800-20261030'
        actual_output = ROOT/'runs/offline-native-002-p2-first-wait1800-20261030_reconciled_001'
        exact = dict(expected_recording_id='offline-native-002-p2-first-wait1800-20261030',
                     observed_recording_id='offline-native-002-p1-first-wait1800-20261030',
                     config_sha256='a6dcb523cb8e867694562aaa3f1b1c0bd39f44e5e9e08a4905fe557eaaac2884',
                     episode_id='dispatch-production-wait-shortfall-v2-run-10301-seed-202610300301',
                     run_index=10301, seed=202610300301,
                     instrumentation_revision='boundary-events-v1-native-rpc-runtime-scene-v1',
                     event_manifest_sha256='46cff75d11086cc9212c99deaad6e818973c316c7d70570084dbc5af7e18d1fb',
                     events_sha256='83918f53e503fed9b5593699c4475d97ac2b1c93776709f88644d72e78d07e19')
        _require(source == actual_source.resolve() and derived == actual_output.resolve() and
                 all(binding[key] == value for key, value in exact.items()),
                 'user exception only authorizes the original policy2 first shift')
    rejection = sidecar['strict_rejection']
    _require(type(rejection) is dict and set(rejection) == {'path', 'sha256'},
             'strict rejection evidence required')
    refusal_path = _canonical(rejection['path'])
    _require(_sha(refusal_path) == rejection['sha256'], 'strict rejection changed')
    if scope == 'actual_anylogic_offline':
        actual_refusal = ROOT/'reports/offline_data_readiness_001/native_runtime_004/rejection.json'
        _require(refusal_path == actual_refusal.resolve() and rejection['sha256'] ==
                 '9fd3959761b5a4148eb406ed1f1afc3c4b0f4e0f2986a3f0813efff84873fde5',
                 'original frozen policy2 rejection evidence required')
    refusal = _read(refusal_path)
    _require(refusal.get('status') == 'rejected_not_admitted' and
             refusal.get('formal_training_data') is False and
             refusal.get('admission_exists') is False and
             refusal.get('expected_recording_id') == expected and
             refusal.get('actual_event_recording_id') == observed and
             refusal.get('config_sha256') == binding['config_sha256'] and
             refusal.get('run_index') == binding['run_index'] and
             refusal.get('experiment_seed') == binding['seed'] and
             'recording identity mismatch' in refusal.get('finalize_error', ''),
             'strict rejection does not bind this identity error')
    frozen_files = {str((source/name).resolve()): digest for name, digest in inventory.items()}
    frozen_files.update({str(bundle/'events_manifest.json'): binding['event_manifest_sha256'],
                         str(bundle/'events.jsonl'): binding['events_sha256']})
    _require(refusal.get('evidence_sha256') == frozen_files,
             'source/event bytes differ from the original frozen rejection inventory')
    return source, derived, cfg, binding


def validate(directory, descriptor):
    """Revalidate authorization and source bytes, never infer an exception."""
    directory = Path(directory).resolve()
    descriptor = Path(descriptor).resolve()
    _require(descriptor == directory/SIDECAR, 'sidecar must belong to this capsule')
    sidecar = _read(descriptor)
    _, derived, _, _ = _validate_source(sidecar)
    _require(derived == directory, 'capsule path mismatch')
    for name, digest in sidecar['copied_files_sha256'].items():
        _require(_sha(directory/name) == digest, 'capsule bytes changed: '+name)
    return copy.deepcopy(sidecar)


def event_recording_id(directory, descriptor, bundle):
    """Return the original label for precisely one already-bound event file."""
    sidecar = validate(directory, descriptor)
    binding = sidecar['binding']
    _require(str(Path(bundle).resolve()) == binding['event_bundle'], 'unbound event bundle')
    return binding['observed_recording_id']


def prepare(source, output, *, strict_rejection, authorization):
    """Create a fresh byte-identical capsule; do not admit it or touch source."""
    _authorization(authorization)
    source, output = Path(source).resolve(), Path(output).resolve()
    _require(not output.exists(), 'refusing capsule reuse')
    cfg = _read(source/'acceptance_run_config.json')
    _require(len(cfg.get('jobs', [])) == 1, 'only one episode alias is permitted')
    bundle = Path(cfg['jobs'][0]['event_bundle']).resolve()
    em = _read(bundle/'events_manifest.json')
    inventory = _inventory(source)
    copied = {name: inventory[name] for name in CORE_FILES}
    copied.update({name: inventory[name] for name in OPTIONAL_FILES if name in inventory})
    rejection = Path(strict_rejection).resolve()
    sidecar = dict(contract=CONTRACT, status='authorized_single_recording_identity_reconciliation',
        authorization=copy.deepcopy(authorization), source_recording=str(source),
        derived_recording=str(output), source_files_sha256=inventory,
        copied_files_sha256=copied, strict_rejection=dict(path=str(rejection), sha256=_sha(rejection)),
        binding=dict(event_bundle=str(bundle), expected_recording_id=cfg['recording']['recording_id'],
            observed_recording_id=em['recording_id'], config_sha256=_sha(source/'acceptance_run_config.json'),
            episode_id=em['episode_id'], run_index=em['run_index'], seed=em['seed'],
            instrumentation_revision=em['instrumentation_revision'],
            event_manifest_sha256=_sha(bundle/'events_manifest.json'), events_sha256=_sha(bundle/'events.jsonl')),
        strict_identity_passed=False, raw_data_rewritten=False)
    _validate_source(sidecar)  # all identity/authorization checks before writing
    output.mkdir(parents=True, exist_ok=False)
    for name in copied:
        target = output/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source/name, target)
    descriptor = output/SIDECAR
    descriptor.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2, allow_nan=False)+'\n',
                          encoding='utf-8')
    validate(output, descriptor)
    return descriptor


def reconcile(source, output, *, strict_rejection, authorization, allow_synthetic=False):
    """Prepare and finalize explicitly; no collection, optimizer or simulation."""
    from .collect_offline import finalize
    descriptor = prepare(source, output, strict_rejection=strict_rejection,
                         authorization=authorization)
    return finalize(output, allow_synthetic=allow_synthetic,
                    identity_reconciliation=descriptor)
