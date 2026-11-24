"""Admit the already consumed mixed-training shift without any simulator calls."""
import argparse
from collections import Counter
from contextlib import nullcontext
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from mixed_plan import CAMPAIGN, CORE, LEDGER, MATERIALS, OUTPUT, REPORT, TOTAL, load_package, read, reference, require
from baseline1_v1.collection import _java_start_properties, verify_durable_reference
from business_graph_summary_campaign_v1 import admission, collection
from business_graph_summary_campaign_v1.budget import connection, digest
from business_graph_summary_campaign_v1.health import _process_identity
from business_graph_summary_campaign_v1.plan_io import encoded
from grade_event_memory_v1.journal import read_journal
from grade_only_ppo_v2.runtime import _controller_lock
from mixed_budget import MixedLedger
from mixed_runtime import _exact_verification, _verify_observation_sources
from online_v1_1.operations import _publish
from storage_scope import bg128_storage


RUN_INDEX = 10000001
DECISIONS = 184
ERROR = "ValueError('unknown task-query candidate')"
RECOVERY_OUTPUT = OUTPUT / 'recovery_002'
ADMISSION_REPORT = REPORT / 'recovery_002' / 'admission'
PROOF = ADMISSION_REPORT / 'reconciliation.json'
PREVIOUS = OUTPUT / 'recovery_001'


def _publish_exact(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == data, 'preserved reconciliation artifact differs: ' + str(path))
    else:
        _publish(path, data)
    return reference(path)


def _controller_evidence(suite, cfg, phase_ref, job):
    reason = read(PREVIOUS / 'run_status.json')
    stop = PREVIOUS / 'control/STOP.properties'
    trace = PREVIOUS / 'launch/controller.log'
    launch_receipt = PREVIOUS / 'launch/launch_receipt.json'
    process = read(launch_receipt)
    identity = _process_identity(process['pid'])
    require(reason['campaign_id'] == CAMPAIGN and reason['status'] == 'STOP' and reason['reason'] == ERROR
            and stop.read_text() == 'reason=' + ERROR + '\n'
            and 'JAVA_START_VERIFIED run_index=10000001' in trace.read_text()
            and trace.read_text().rstrip().endswith('ValueError: unknown task-query candidate'),
            'previous controller did not stop at post-collection BG128 admission')
    require(not identity.get('inspection_denied') and
            (identity.get('state') == 'Z' or identity.get('start_ticks') != process['process_start_ticks']),
            'previous controller may still be active')
    control = PREVIOUS / 'control'
    claim = control / 'java_claims' / f'run_{RUN_INDEX}.once.claim'
    ready = control / 'ready' / f'run_{RUN_INDEX}.properties'
    require(set(claim.parent.iterdir()) == {claim} and set(ready.parent.iterdir()) == {ready},
            'expected only the already consumed first Java claim and READY')
    expected = dict(protocol='baseline1-one-click-java-start-v1', campaign_id=CAMPAIGN,
                    master_plan_sha256=suite['master_plan']['sha256'], ordinal='0', run_index=str(RUN_INDEX),
                    snapshot_sha256=phase_ref['sha256'], config_sha256=phase_ref['sha256'])
    require(_java_start_properties(claim) == expected, 'exclusive Java claim differs from the frozen first shift')
    props = _java_start_properties(ready)
    require(props['protocol'] == 'baseline1-one-click-ready-v1' and props['campaign_id'] == CAMPAIGN
            and props['master_plan_sha256'] == suite['master_plan']['sha256']
            and props['config_sha256'] == props['snapshot_sha256'] == phase_ref['sha256']
            and reference(props['snapshot_path']) == phase_ref, 'READY phase binding differs')
    from baseline1_v1.java_plan import COLUMNS
    require(all(props[name] == str(job[name]) for name in COLUMNS), 'READY job identity differs')
    phase_output = Path(cfg['python_output'])
    raw = phase_output / 'episodes' / f'run_{RUN_INDEX}'
    raw_receipt = read(raw / 'receipt.json')
    manifest = verify_durable_reference(reference(raw / 'manifest.json'))
    require(raw_receipt['normal_completion'] is True and raw_receipt['run_index'] == RUN_INDEX
            and raw_receipt['experiment_seed'] == job['experiment_seed']
            and raw_receipt['phase_id'] == cfg['phase_id'] and raw_receipt['role'] == 'train'
            and raw_receipt['phase_config'] == phase_ref and raw_receipt['four_windows'] == 4
            and raw_receipt['transitions'] == manifest['transitions'] == DECISIONS
            and raw_receipt['dataset_files'] == manifest['files_sha256']
            and manifest['status'] == 'complete' and manifest['source_kind'] == 'real_anylogic_rpc',
            'original raw receipt does not bind the sealed complete physical shift')
    require(set((phase_output / 'episodes').iterdir()) == {raw}
            and not (phase_output / 'material').exists(), 'unexpected original collection artifacts')
    journal = Path(job['material_journal'])
    require(set(MATERIALS.rglob('material_events.bin')) == {journal}
            and not list(MATERIALS.rglob('*.partial')), 'unexpected or unsealed physical material evidence')
    material = read_journal(journal)
    header = material.header
    require(header['source_kind'] == 'actual_anylogic' and header['run_index'] == RUN_INDEX
            and header['seed'] == job['experiment_seed'] and header['recording_id'] == job['recording_id']
            and header['config_sha256'] == phase_ref['sha256'] and header['episode_id'] == raw_receipt['episode_id']
            and sum(event.kind == 'DECISION' for event in material.events) == DECISIONS
            and sum(event.kind == 'WINDOW_CLOSE' for event in material.events) == 4,
            'sealed AnyLogic provenance or terminal inventory differs')
    details = dict(java_claim=reference(claim), ready=reference(ready), previous_status=reference(PREVIOUS / 'run_status.json'),
                   previous_stop=reference(stop), previous_trace=reference(trace), previous_launch=reference(launch_receipt),
                   raw_receipt=reference(raw / 'receipt.json'), raw_manifest=reference(raw / 'manifest.json'),
                   material_journal=reference(journal), actor=reference(phase_output / 'actor.pt'))
    return raw, raw_receipt, journal, details


def _read_slot(ledger, suite, job):
    state = ledger.snapshot()
    require(not state['campaigns'][CAMPAIGN]['commits'], 'training started before first admission')
    require(all(row['status'] == 'PLANNED' for run, row in state['jobs'].items() if int(run) != RUN_INDEX),
            'a later physical shift was consumed')
    first = state['jobs'][str(RUN_INDEX)]
    require(first['identity'] == suite['jobs'][0] == job and first['status'] in ('UNKNOWN', 'COMPLETE'),
            'unexpected canonical first-shift status or identity')
    if first['status'] == 'UNKNOWN':
        require(first['error'] == ERROR, 'unexpected original UNKNOWN reason')
    else:
        require(first.get('reconciled_from') == 'UNKNOWN', 'first shift was completed outside this recovery')
    return first


def _admit(path, raw, raw_receipt, journal, phase_ref, job, source):
    path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path = path.with_suffix('.receipt.json')
    actor = source['actor']['path']
    with bg128_storage(), _exact_verification([job]):
        if receipt_path.is_file():
            receipt = read(receipt_path)
        else:
            require(not any(file.exists() for file in (path, Path(str(path) + '.manifest.json'), Path(str(path) + '.seal.json'))),
                    'unbound derived recovery artifacts require review')
            result = admission.admit_episode(raw, journal, config_path=phase_ref['path'], actor_path=actor,
                                             output_path=path, split='train')
            require(result['status'] == 'PASS_DATA_PREPARATION_ONLY' and result['transitions'] == DECISIONS,
                    'physical shift did not pass full offline admission')
            receipt = dict(schema=collection.RECEIPT_SCHEMA, normal_completion=True, identity=result['identity'],
                           phase_config=phase_ref, phase_id=job['phase_id'], role=job['role'], raw_receipt=raw_receipt,
                           run_index=RUN_INDEX, experiment_seed=job['experiment_seed'], four_windows=4,
                           transitions=DECISIONS, training_objective=collection.TRAINING_REWARD,
                           composite_contract=collection._phase(phase_ref['path'])['composite_contract'],
                           material_evidence=True, production_guarantee='pending', production_failures_retained=True,
                           references=collection._references(phase_ref['path'], actor, journal,
                                                             raw / 'manifest.json', path))
        require(receipt['raw_receipt'] == raw_receipt and receipt['phase_config'] == phase_ref
                and receipt['transitions'] == DECISIONS and receipt['four_windows'] == 4
                and receipt['references']['episode'] == reference(path)
                and receipt['references']['actor'] == source['actor']
                and receipt['references']['journal'] == source['material_journal']
                and receipt['references']['raw_manifest'] == source['raw_manifest'],
                'derived receipt cannot rebind the first sealed physical shift')
        episode = collection.load_admitted_episode(receipt)
        _verify_observation_sources(episode, job)
        require(len(episode['actions']) == DECISIONS and len(episode['memory'].settlements) == 4,
                'derived round trip lost decisions or settled windows')
    receipt_ref = _publish_exact(receipt_path, encoded(receipt))
    return receipt, receipt_ref, episode


def _archive(ledger, first, report_root, source, job):
    archive = report_root / 'ledger_before.sqlite'
    if first['status'] == 'UNKNOWN':
        require(not any(Path(str(ledger.path) + suffix).exists() for suffix in ('-wal', '-journal')),
                'live registry sidecars prevent a byte-identical before-image')
        before = _publish_exact(archive, ledger.path.read_bytes())
    else:
        before = reference(archive)
    with connection(archive, readonly=True) as database:
        run, status, text, checksum = database.execute(
            'SELECT run,status,payload,checksum FROM slots WHERE run=? AND campaign=?',
            (RUN_INDEX, CAMPAIGN),
        ).fetchone()
        expected = dict(status='UNKNOWN', error=ERROR, identity_sha256=digest(job))
        require(run == RUN_INDEX and status == 'UNKNOWN' and json.loads(text) == expected and checksum == digest(expected),
                'before-image does not contain the exact UNKNOWN slot')
    unknown = _publish_exact(report_root / 'unknown_slot_before.json',
                             encoded(dict(run=run, status=status, payload=text, checksum=checksum)))
    stop = _publish_exact(report_root / 'STOP.properties.before', Path(source['previous_stop']['path']).read_bytes())
    return dict(ledger_before=before, unknown_slot_before=unknown, stop_before=stop), expected


def _record_completion(ledger, job, original, completed):
    with connection(ledger.path) as database:
        current = ledger._row(database, job)
        require(current == original or current == completed, 'first slot changed during reconciliation')
        require(database.execute('SELECT count(*) FROM commits').fetchone()[0] == 0
                and database.execute('SELECT count(*) FROM slots WHERE run<>? AND status=\'PLANNED\'',
                                     (RUN_INDEX,)).fetchone()[0] == TOTAL - 1, 'later training or physical slots changed')
        if current == original:
            ledger._put(database, job, completed)


def reconcile(*, apply=False, ledger_path=LEDGER, output_root=RECOVERY_OUTPUT, report_root=ADMISSION_REPORT):
    """Dry-run by default. Apply only after the original campaign's recovery is approved."""
    from torch import set_num_threads
    ledger_path, output_root, report_root = (Path(path).resolve() for path in (ledger_path, output_root, report_root))
    production = ledger_path == LEDGER
    require(not apply or production == (output_root == RECOVERY_OUTPUT and report_root == ADMISSION_REPORT),
            'production and test paths cannot be mixed')
    require(not apply or production or (output_root != RECOVERY_OUTPUT and report_root != ADMISSION_REPORT),
            'disposable registry cannot publish production evidence')
    lock = _controller_lock(OUTPUT) if apply and production else nullcontext()
    with lock:
        suite, phases = load_package()
        cfg, phase_ref = phases[0], suite['phase_configs'][0]
        job = cfg['jobs'][0]
        require(job['run_index'] == RUN_INDEX and job['ordinal'] == 0
                and reference(CORE) == suite['protected_core'], 'wrong physical shift or protected PPO core changed')
        ledger = MixedLedger(suite, phases)
        ledger.path = ledger_path
        first = _read_slot(ledger, suite, job)
        require(apply or first['status'] == 'UNKNOWN', 'dry run expects the original UNKNOWN slot')
        raw, raw_receipt, journal, source = _controller_evidence(suite, cfg, phase_ref, job)
        set_num_threads(suite['cpu_threads'])
        with TemporaryDirectory(prefix='mixed-first-shift-admission-') as work:
            root = output_root if apply else Path(work)
            if apply:
                archive, original = _archive(ledger, first, report_root, source, job)
            receipt, receipt_ref, episode = _admit(root / 'material' / f'run_{RUN_INDEX}.h5', raw, raw_receipt,
                                                   journal, phase_ref, job, source)
        proof = dict(schema='bg128-mixed-first-shift-reconciliation-v1',
                     status='APPLIED' if apply else 'PASS_OFFLINE_NO_LEDGER_CHANGE', run_index=RUN_INDEX,
                     original_status='UNKNOWN', new_status='COMPLETE' if apply else 'UNKNOWN',
                     decisions=DECISIONS, four_windows=4, new_physical_shifts=0, new_reserved_shifts=0,
                     completed_after=int(apply), planned_after=TOTAL - 1,
                     environment_seed=job['experiment_seed'], actor_sampling_seed=job['actor_sampling_seed'],
                     recording_id=job['recording_id'], original_suite=reference(suite['suite_path']),
                     phase_config=phase_ref, source=source, summary=episode['evaluation'],
                     window_grades=[window.grade for window in episode['memory'].settlements])
        if apply:
            dataset_ref = receipt['references']['episode_manifest']
            completed = dict(status='COMPLETE', identity_sha256=original['identity_sha256'],
                             dataset_ref=dataset_ref, receipt=receipt, reconciled_from='UNKNOWN',
                             completion_sha256=digest(dict(dataset_ref=dataset_ref, receipt=receipt)))
            _record_completion(ledger, job, original, completed)
            state = ledger.snapshot()
            require(Counter(item['status'] for item in state['jobs'].values()) == {'COMPLETE': 1, 'PLANNED': TOTAL - 1},
                    'unexpected registry state after first-shift admission')
            proof.update(archive, dataset_ref=dataset_ref, material_receipt=receipt_ref)
            _publish_exact(report_root / 'reconciliation.json', encoded(proof))
        return proof


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    print(json.dumps(reconcile(apply=args.apply), indent=2, sort_keys=True, ensure_ascii=False))
