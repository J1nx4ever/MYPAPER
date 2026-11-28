"""Collect legal, authenticated on-policy shifts through the existing BG128 RPC."""

from pathlib import Path
from unittest.mock import patch

from business_graph_summary_campaign_v1 import behavior
from business_graph_summary_campaign_v1.collection import load_admitted_episode
from business_graph_summary_campaign_v1.plan_io import load_reference, reference
from business_graph_summary_campaign_v1.progress import write_progress
from business_graph_summary_v1.contract import require
from change_plans.token_attention_001.business_graph_summary_009.generalization_001.grade_assignment_generalization_001.continuous_training_001 import (
    mixed_runtime as mixed,
)
from change_plans.token_attention_001.business_graph_summary_009.generalization_001.grade_assignment_generalization_001.continuous_training_001.recovery_002.storage_scope import (
    bg128_storage,
)
from change_plans.token_attention_001.business_graph_summary_009.generalization_001.grade_assignment_generalization_001.continuous_training_001.recovery_003.transport import (
    PerJobMaterialCollector,
)


def complete_sources(cfg, phase_ref, ledger, *, committed_update=None):
    """Return the ordered durable inputs after checking their physical commitments."""
    rows = ledger.resume()['jobs']
    sources = []
    for job in cfg['jobs']:
        row = rows[str(job['run_index'])]
        receipt = row['receipt']
        require(row['status'] == 'COMPLETE' and row['identity'] == job
                and receipt['phase_config'] == phase_ref,
                'training input is not a complete shift under the frozen phase')
        if committed_update is not None:
            require(row.get('committed_update') == committed_update,
                    'physical shift belongs to a different committed update')
        manifest = row['dataset_ref']
        require(reference(manifest['path']) == manifest
                and receipt['references']['episode_manifest'] == manifest,
                'training input dataset manifest differs from its physical receipt')
        receipt_path = Path(receipt['references']['raw_manifest']['path']).with_name('material_receipt.json')
        # The ledger retains its own checksummed receipt. Seal the matching disk receipt
        # in the optimizer transaction so it can be rechecked after a process restart.
        receipt_ref = reference(receipt_path)
        require(load_reference(receipt_ref) == receipt,
                'physical ledger and durable material receipt differ')
        sources.append(dict(run_index=job['run_index'], dataset=manifest, receipt=receipt_ref))
    return sources


def collect_phase(suite, cfg, phase_ref, checkpoint, ledger, model):
    """Sample the entire phase under one immutable Actor and rebind resumed shifts."""
    actor_path = Path(cfg['python_output']) / 'actor.pt'
    if actor_path.is_file():
        actor_ref = model.bind_actor_snapshot(actor_path)
    else:
        actor_ref = model.save_actor(actor_path)
    rows = ledger.resume()['jobs']
    require(all(rows[str(job['run_index'])]['status'] in ('PLANNED', 'COMPLETE') for job in cfg['jobs']),
            'unresolved physical claims prevent collection')
    remaining = [job for job in cfg['jobs'] if rows[str(job['run_index'])]['status'] == 'PLANNED']
    fresh = {}
    # The existing exact verifier replays each decision with the original one-row
    # shape and RNG draw. Mathematical batch equivalence alone is insufficient here.
    with mixed.VERIFY_LOCK, bg128_storage(), patch.object(behavior, 'verify', mixed.verify_online):
        if remaining:
            sampler = mixed.MixedOnlineActor(model.actor, remaining)
            startup = dict(control_directory=suite['control_directory'], campaign_id=suite['campaign_id'],
                           master_plan_sha256=suite['master_plan']['sha256'])
            publish_ready = mixed._ready_publisher(suite, phase_ref, checkpoint, ledger)
            with PerJobMaterialCollector(dict(cfg, jobs=remaining), phase_ref['path'],
                                         publish_ready=publish_ready, startup=startup) as collector:
                for job in remaining:
                    run = job['run_index']
                    write_progress(suite, status='COLLECTING', phase_id=cfg['phase_id'],
                                   run_index=run, ordinal=job['ordinal'])
                    ledger.claim(job)
                    try:
                        episode, dataset, receipt = collector.collect(job, sampler, actor_path=actor_ref['path'])
                        mixed._verify_observation_sources(episode, job)
                        ledger.complete(job, dataset, receipt)
                        fresh[run] = episode
                    except BaseException as error:
                        if ledger.job(job)['status'] == 'CLAIMED':
                            ledger.unknown(job, repr(error))
                        raise

        sources = complete_sources(cfg, phase_ref, ledger)
        rows = ledger.resume()['jobs']
        episodes = []
        for job in cfg['jobs']:
            run = job['run_index']
            episode = fresh.pop(run, None)
            if episode is None:
                episode = load_admitted_episode(rows[str(run)]['receipt'])
                mixed._verify_observation_sources(episode, job)
            expected = dict(run_index=run, environment_seed=job['experiment_seed'],
                            actor_sampling_seed=job['actor_sampling_seed'], scene_id=job['scene_id'],
                            scene_group=job['scene_group'], split=job['partition'], recording_id=job['recording_id'],
                            policy_sha256=actor_ref['sha256'], policy_version=model.policy_version)
            require(all(episode['identity'][key] == value for key, value in expected.items()),
                    'admitted episode does not belong to the frozen Actor and phase')
            episodes.append(episode)
    return episodes, sources, actor_ref
