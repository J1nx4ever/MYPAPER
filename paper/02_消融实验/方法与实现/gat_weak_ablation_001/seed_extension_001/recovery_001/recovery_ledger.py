"""Offline final admission of the terminal GAT1/s04 eval shift interrupted by reboot."""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
from shutil import copyfile
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extension_run import ExtensionLedger, load_package, parent, runtime_scope
from baseline1_v1.collection import _java_start_properties, verify_durable_reference
from baseline1_v1.java_plan import COLUMNS as READY_COLUMNS
from business_graph_summary_campaign_v1 import collection
from business_graph_summary_campaign_v1.plan_io import encoded, reference
from efficient_validation import validation_scope
from grade_only_ppo_v2.runtime import _controller_lock
from grade_event_memory_v1.journal import read_journal
from online_v1_1.operations import _publish
from recovery_002.storage_scope import bg128_storage

UNIT = 'gat1-s04'
RUN = 10049254
require = parent.require


def _save(path, data):
    if path.exists():
        require(path.read_bytes() == data, 'existing recovery evidence differs: ' + str(path))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        _publish(path, data)
    return reference(path)


def _sources(suite, job, cfg, phase, raw):
    manifest = verify_durable_reference(reference(raw / 'manifest.json'))
    original = json.loads((raw / 'receipt.json').read_bytes())
    require(manifest['status'] == 'complete' and manifest['source_kind'] == 'real_anylogic_rpc'
            and manifest['provenance'] == dict(configuration=phase, recording_id=job['recording_id'],
                run_index=RUN, seed=job['experiment_seed']), 'RPC raw source provenance differs')
    require(original == dict(normal_completion=True, run_index=RUN, experiment_seed=job['experiment_seed'],
                role=job['role'], phase_id=cfg['phase_id'], episode_id=Path(job['material_event_bundle']).name,
                policy_version=manifest['policy_version'], phase_config=phase, physical_evidence=None,
                storage=manifest['storage'], dataset_files=manifest['files_sha256'],
                transitions=manifest['transitions'], four_windows=4), 'original raw terminal receipt differs')
    journal = collection.material_journal_path(job)
    tape = read_journal(journal)
    require(tape.header['source_kind'] == 'actual_anylogic' and tape.events[-1].kind == 'END'
            and sum(event.kind == 'END' for event in tape.events) == 1
            and sum(event.kind == 'WINDOW_CLOSE' for event in tape.events) == 4
            and sum(event.kind == 'DECISION' for event in tape.events) == original['transitions'],
            'physical material terminal/window inventory differs')

    control = Path(suite['control_directory'])
    claim = control / 'java_claims' / f'run_{RUN}.once.claim'
    ready = control / 'ready' / f'run_{RUN}.properties'
    shared = dict(campaign_id=suite['campaign_id'], master_plan_sha256=suite['master_plan']['sha256'],
                  ordinal=str(job['ordinal']), run_index=str(RUN), config_sha256=phase['sha256'],
                  snapshot_sha256=phase['sha256'])
    require(_java_start_properties(claim) == dict(shared, protocol='baseline1-one-click-java-start-v1'),
            'physical Java claim differs')
    ready_fields = _java_start_properties(ready)
    require(all(ready_fields.get(name) == value for name, value in shared.items())
            and ready_fields.get('protocol') == 'baseline1-one-click-ready-v1'
            and ready_fields.get('all_comparison_checkpoints_frozen') == 'true'
            and ready_fields.get('snapshot_path') == phase['path']
            and all(ready_fields.get(key) == str(job[key]) for key in READY_COLUMNS)
            and all(ready_fields.get(key) == str(job[key]) for key in ('actor_sampling_seed', 'experiment_seed',
                                                                       'actor_seed', 'recording_id')),
            'original READY differs from signed phase or physical identity')
    return original, journal, dict(java_claim=reference(claim), ready=reference(ready),
        raw_receipt=reference(raw / 'receipt.json'), raw_manifest=reference(raw / 'manifest.json'),
        material_journal=reference(journal))


def _receipt(output, identity, raw, original, journal, actor, phase, cfg, job):
    return dict(schema=collection.RECEIPT_SCHEMA, normal_completion=True, identity=identity,
                phase_id=cfg['phase_id'], role=job['role'], raw_receipt=original, run_index=RUN,
                experiment_seed=job['experiment_seed'], phase_config=phase,
                references=collection._references(phase['path'], actor, journal, raw / 'manifest.json', output),
                four_windows=4, transitions=original['transitions'], training_objective=collection.TRAINING_REWARD,
                composite_contract=cfg['composite_contract'], material_evidence=True,
                production_guarantee='pending', production_failures_retained=True)


def _check_episode(runtime, receipt, job, model):
    episode = collection.load_admitted_episode(receipt)
    runtime._verify_observation_sources(episode, job)
    identity = episode['identity']
    expected = dict(run_index=RUN, environment_seed=job['experiment_seed'], actor_sampling_seed=job['actor_sampling_seed'],
                    scene_id=job['scene_id'], scene_group=job['scene_group'], split=job['partition'],
                    recording_id=job['recording_id'], policy_sha256=receipt['references']['actor']['sha256'],
                    policy_version=model.policy_version)
    require(all(identity[key] == value for key, value in expected.items())
            and len(episode['actions']) == receipt['transitions'] and len(episode['memory'].settlements) == 4,
            'sealed derived episode differs from fixed checkpoint, run or terminal inventory')


def _install(stage, live):
    """Publish the same sealed trio on retries. Hard links keep the before-commit evidence."""
    for suffix in ('', '.manifest.json', '.seal.json'):
        source, target = Path(str(stage) + suffix), Path(str(live) + suffix)
        if target.exists():
            require(target.samefile(source), 'unexpected previously published material file: ' + str(target))
        else:
            os.link(source, target)
    fd = os.open(live.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def reconcile_claimed(suite, phases, *, apply=False):
    """Optionally settle the single existing claim, under the campaign controller lock."""
    import torch

    require(suite['unit'] == UNIT and reference(parent.CORE) == suite['protected_core'],
            'reconciliation requires original GAT1/s04 suite and protected PPO')
    ledger = ExtensionLedger(suite, phases)
    output = Path(suite['python_output'])
    report = Path(suite['report_directory']) / 'recovery_001'
    with _controller_lock(output) if apply else nullcontext():
        state = ledger.snapshot()
        cfg, phase = next((cfg, ref) for cfg, ref in zip(phases, suite['phase_configs'], strict=True)
                          if any(job['run_index'] == RUN for job in cfg['jobs']))
        job = next(job for job in cfg['jobs'] if job['run_index'] == RUN)
        row = state['jobs'][str(RUN)]
        require(cfg['policy_arm'] == 'new' and job['checkpoint_update'] == 32 and job['role'] == 'final'
                and reference(phase['path']) == phase and row['identity'] == job
                and row['status'] in ('CLAIMED', 'COMPLETE'), 'interrupted evaluation slot differs')
        require(all(item['status'] == 'COMPLETE' for run, item in state['jobs'].items() if int(run) < RUN)
                and len(state['campaigns'][suite['campaign_id']]['commits']) == 32,
                'earlier physical slots or checkpoint inventory differs')
        if row['status'] == 'CLAIMED':
            require(all(item['status'] == 'PLANNED' for run, item in state['jobs'].items() if int(run) > RUN),
                    'later physical slot advanced past original claim')
        checkpoint = ledger.commit_record(32)['checkpoint_ref']
        require(reference(checkpoint['path']) == checkpoint, 'committed u32 checkpoint bytes differ')
        phase_out = Path(cfg['python_output'])
        raw = phase_out / 'episodes' / f'run_{RUN}'
        actor = phase_out / 'actor.pt'
        original, journal, evidence = _sources(suite, job, cfg, phase, raw)
        evidence.update(actor=reference(actor), checkpoint=checkpoint, phase_config=phase)
        torch.set_num_threads(suite['cpu_threads'])
        live = phase_out / 'material' / f'run_{RUN}.h5'
        stage = output / 'recovery_001' / f'stage_{RUN}' / live.name

        with bg128_storage(), runtime_scope(suite) as runtime, validation_scope([job], runtime) as proof:
            runtime._check_sources(suite)
            model = runtime._load_new(suite, checkpoint, 32)
            model.bind_actor_snapshot(actor)
            require(model.policy_version == original['policy_version'], 'phase Actor differs from committed u32')
            with TemporaryDirectory(prefix='gat1-s04-offline-admission-') as scratch:
                trial = Path(scratch) / live.name
                result = collection.admit_episode(raw, journal, config_path=phase['path'], actor_path=actor,
                                                   output_path=trial, split=job['partition'])
                require(result['status'] == 'PASS_DATA_PREPARATION_ONLY' and result['transitions'] == original['transitions'],
                        'offline admission did not authenticate the complete original shift')
                identity = result['identity']
                trial_receipt = _receipt(trial, identity, raw, original, journal, actor, phase, cfg, job)
                _check_episode(runtime, trial_receipt, job, model)
                require(proof.count['exact_replays'] == 1, 'exact fixed Actor replay absent')

                if apply and row['status'] == 'CLAIMED':
                    _save(report / 'claimed_slot_before.json', encoded(dict(
                        run_index=RUN, status='CLAIMED', identity_sha256=row['identity_sha256'], source=evidence)))
                    stage.parent.parent.mkdir(parents=True, exist_ok=True)
                    if not stage.parent.exists():
                        with TemporaryDirectory(dir=stage.parent.parent, prefix='preparing-') as staging:
                            staged = Path(staging) / 'complete'
                            staged.mkdir()
                            for suffix in ('', '.manifest.json', '.seal.json'):
                                dest = staged / (stage.name + suffix)
                                copyfile(str(trial) + suffix, dest)
                                with dest.open('rb') as stream:
                                    os.fsync(stream.fileno())
                            staged.rename(stage.parent)
                    staged_receipt = _receipt(stage, identity, raw, original, journal, actor, phase, cfg, job)
                    _check_episode(runtime, staged_receipt, job, model)
                    _install(stage, live)
                    material = _receipt(live, identity, raw, original, journal, actor, phase, cfg, job)
                    _check_episode(runtime, material, job, model)
                    _save(raw / 'material_receipt.json',
                          (json.dumps(material, sort_keys=True, allow_nan=False) + '\n').encode())
                    ledger.complete(job, material['references']['episode_manifest'], material)

                if apply or row['status'] == 'COMPLETE':
                    completed = ledger.job(job)
                    material = _receipt(live, identity, raw, original, journal, actor, phase, cfg, job)
                    require(completed['status'] == 'COMPLETE' and completed['receipt'] == material
                            and completed['dataset_ref'] == material['references']['episode_manifest']
                            and json.loads((raw / 'material_receipt.json').read_bytes()) == material,
                            'existing durable receipt or ledger completion differs')
                    _check_episode(runtime, material, job, model)
                    evidence.update(dataset_ref=completed['dataset_ref'], material_receipt=reference(raw / 'material_receipt.json'))
        summary = dict(run_index=RUN, status='COMPLETE' if apply or row['status'] == 'COMPLETE' else 'PASS_OFFLINE',
                       recording_id=job['recording_id'], environment_seed=job['experiment_seed'],
                       actor_sampling_seed=job['actor_sampling_seed'], transitions=original['transitions'],
                       four_windows=4, exact_replays=1, new_physical_shifts=0, new_budget_slots=0, source=evidence)
        if apply:
            _save(report / 'reconciliation.json', encoded(summary))
        return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    print(json.dumps(reconcile_claimed(*load_package(UNIT), apply=parser.parse_args().apply), indent=2))
