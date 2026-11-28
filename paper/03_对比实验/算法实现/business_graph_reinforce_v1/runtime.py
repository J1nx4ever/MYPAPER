"""Run a frozen, registered full-shift BG128 REINFORCE training campaign."""

import hashlib
from pathlib import Path

import torch

from business_graph_summary_campaign_v1.gate import PROTECTED_SHA, check_phase
from business_graph_summary_campaign_v1.plan_io import ROOT, encoded, load_phase, load_reference, reference, sha
from business_graph_summary_campaign_v1.progress import write_progress
from business_graph_summary_campaign_v1.runtime import _same_or_save
from business_graph_summary_v1.contract import require
from business_graph_summary_v1.learner import CHECKPOINT_SCHEMA, _load_checkpoint, _validate_model_state
from business_graph_summary_v1.models import actor_from_config, digest
from online_v1_1.operations import _publish
from python import observation_contract

from .collection import collect_phase, complete_sources, mixed
from .learner import SETTINGS_SCHEMA, FullShiftReinforce


CORE = ROOT.parents[1] / 'python' / 'episode_budget_ppo.py'
SCHEMA = 'bg128-tang-eq19-full-shift-training-v1'


def _matches(ref):
    require(reference(ref['path']) == ref, 'frozen artifact changed: ' + ref['path'])
    return ref


def _source_actor(suite):
    """Read only the matched u0 Actor tensors and authenticate their digest."""
    checkpoint = _matches(suite['initial_actor'])
    for suffix, key in (('.manifest.json', 'initial_actor_manifest'), ('.seal.json', 'initial_actor_seal')):
        require(reference(checkpoint['path'] + suffix) == suite[key],
                'matched historical initialization sidecar differs')
    saved, manifest = _load_checkpoint(checkpoint['path'])
    config = suite['arms'][0]['learner_config']
    require(saved['schema'] == manifest['schema'] == CHECKPOINT_SCHEMA
            and saved['update_index'] == manifest['update_index'] == 0
            and saved['config'] == config and not saved['consumed_batches']
            and not saved['consumed_episodes'] and manifest['sha256'] == checkpoint['sha256'],
            'matched historical BG128 initialization is not an untrained Actor')
    actor = actor_from_config(config)
    _validate_model_state(saved['actor'], actor)
    actor.load_state_dict(saved['actor'], strict=True)
    require(digest(actor) == saved['actor_digest'] == manifest['actor_digest'] == suite['initial_actor_digest'],
            'historical u0 Actor digest differs')
    return actor


def verify(suite, phases):
    """Validate the runtime contract without creating files, ledgers or listeners."""
    require(sha(CORE) == PROTECTED_SHA, 'protected PPO core changed')
    require(Path(observation_contract.__file__).resolve().parent == ROOT / 'python',
            'run with the experiment-local Python package')
    require(suite['source_pins'] and all(sha(path) == value for path, value in suite['source_pins'].items()),
            'implementation pin differs from the frozen campaign')
    require(len(suite['arms']) == 1 and suite['arms'][0]['candidate_id'] == 's01_BG128'
            and len(phases) == len(suite['phase_configs']) == suite['training_updates']
            and suite['evaluation_shifts'] == 0,
            'expected one BG128 training arm without evaluation')
    actor = _source_actor(suite)
    rows = []
    for update, (cfg, phase_ref) in enumerate(zip(phases, suite['phase_configs'], strict=True), 1):
        _matches(phase_ref)
        require(load_phase(phase_ref['path']) == cfg, 'expanded phase differs from frozen bytes')
        check_phase(cfg)
        require(cfg['campaign_id'] == suite['campaign_id'] and cfg['policy_arm'] == 'train'
                and cfg['arm_update'] == update and cfg['execution']['port'] == suite['rpc_port']
                and len(cfg['jobs']) == suite['reinforce']['episodes_per_batch']
                and all(job['role'] == 'train' and job['partition'] == 'train'
                        and job['checkpoint_update'] == update - 1
                        and job['actor_seed'] == suite['arms'][0]['learner_config']['seed']
                        for job in cfg['jobs']), 'training phase identities or batch size differ')
        rows.extend(cfg['jobs'])
    require(rows == suite['jobs'] and len(rows) == suite['new_shift_cap'] == suite['training_shifts'],
            'frozen suite physical roster differs')
    _matches(suite['master_plan'])
    return dict(suite=reference(suite['suite_path']), actor_digest=digest(actor),
                training_shifts=len(rows), phases=len(phases), protected_ppo_sha256=PROTECTED_SHA)


def _load(suite, campaign, checkpoint, update):
    _matches(checkpoint)
    model = FullShiftReinforce.from_checkpoint(checkpoint['path'], expected_sha256=checkpoint['sha256'],
        expected_actor_config=suite['arms'][0]['learner_config'],
        expected_settings=dict(schema=SETTINGS_SCHEMA, **suite['reinforce']))
    require(model.runtime_state['campaign'] == campaign and model.runtime_state['initial_actor'] == suite['initial_actor']
            and model.update_index == update and len(model.consumed_batches) == update
            and len(model.consumed_episodes) == update * suite['reinforce']['episodes_per_batch'],
            'checkpoint lineage or exact shift consumption differs')
    return model


def _initial(suite, verification):
    path = Path(suite['arms'][0]['python_output']) / 'initial.pt'
    campaign = verification['suite']
    if not path.exists():
        model = FullShiftReinforce(_source_actor(suite), actor_config=suite['arms'][0]['learner_config'],
                                   **suite['reinforce'])
        model.runtime_state.update(campaign=campaign, initial_actor=suite['initial_actor'])
        model.save_checkpoint(path)
    checkpoint = reference(path)
    model = _load(suite, campaign, checkpoint, 0)
    require(model.policy_version == verification['actor_digest'] and not model.actor_optimizer.state,
            'initial REINFORCE Actor or fresh optimizer differs')
    return checkpoint


def _verify_prepared(suite, campaign, checkpoint, intent, summary_path):
    model = _load(suite, campaign, checkpoint, intent['update'])
    require(model.runtime_state.get('intent') == intent
            and next(reversed(model.consumed_batches)) == intent['batch_id'],
            'checkpoint optimizer transaction does not match the verified inputs')
    summary = _matches(model.runtime_state['training_summary'])
    require(summary['path'] == str(summary_path.resolve()) and load_reference(summary)['intent'] == intent,
            'committed training summary differs')
    return dict(intent=intent, checkpoint=checkpoint, training_summary=summary, actor_digest=model.policy_version)


def _train(suite, cfg, phase_ref, campaign, parent, ledger):
    update = cfg['arm_update']
    folder = Path(suite['arms'][0]['python_output']) / 'updates' / f'{update:04d}'
    checkpoint_path = folder / 'checkpoint.pt'
    summary_path = folder / 'training_summary.json'
    intent_path = folder / 'intent.json'
    prepared_path = folder / 'prepared.json'
    committed = ledger.commit_record(update)
    model = None
    if committed is None and not intent_path.exists():
        model = _load(suite, campaign, parent, update - 1)
        episodes, sources, actor = collect_phase(suite, cfg, phase_ref, parent, ledger, model)
    else:
        sources = complete_sources(cfg, phase_ref, ledger, committed_update=update if committed is not None else None)
        receipt = load_reference(sources[0]['receipt'])
        actor = receipt['references']['actor']
        _matches(actor)
        require(all(load_reference(source['receipt'])['references']['actor'] == actor for source in sources),
                'a batch combines physical outputs from different Actors')
    transaction = dict(campaign=campaign, phase_config=phase_ref, parent_checkpoint=parent, update=update,
                       sources=sources, actor=dict(path=actor['path'], sha256=actor['sha256']),
                       learner=suite['reinforce'])
    intent = dict(schema=SCHEMA + '-update', **transaction,
                  batch_id=hashlib.sha256(encoded(transaction)).hexdigest())
    if committed is not None or intent_path.exists():
        require(load_reference(reference(intent_path)) == intent,
                'saved optimizer intent does not match the sealed physical batch')
        require(checkpoint_path.is_file(), 'incomplete optimizer transaction requires reconciliation')
        prepared = _verify_prepared(suite, campaign, reference(checkpoint_path), intent, summary_path)
        if committed is not None:
            require(committed['checkpoint_ref'] == prepared['checkpoint'] and committed['batch_id'] == intent['batch_id']
                    and load_reference(reference(prepared_path)) == prepared,
                    'committed ledger and prepared optimizer evidence differ')
            return prepared
    else:
        require(not any(path.exists() for path in (checkpoint_path, prepared_path, summary_path)),
                'optimizer artifacts exist without a frozen transaction')
        _same_or_save(intent_path, intent)
        write_progress(suite, status='UPDATING', phase_id=cfg['phase_id'], update_index=update)
        with mixed._exact_verification(cfg['jobs']):
            result = model.update(episodes, batch_id=intent['batch_id'])
        require(model.update_index == update and next(reversed(model.consumed_batches)) == intent['batch_id'],
                'learner did not consume exactly the current on-policy batch')
        _same_or_save(summary_path, dict(intent=intent, reinforce=result,
                      scenario_statistics=mixed._batch_scene_statistics(episodes, cfg['jobs'])))
        model.runtime_state.update(intent=intent, training_summary=reference(summary_path))
        model.save_checkpoint(checkpoint_path)
        prepared = _verify_prepared(suite, campaign, reference(checkpoint_path), intent, summary_path)
        require(prepared['actor_digest'] == model.policy_version,
                'restored Actor differs from the updated Actor')
    _same_or_save(prepared_path, prepared)
    ledger.commit(update, cfg['jobs'], prepared['checkpoint'], intent['batch_id'])
    write_progress(suite, status='UPDATE_COMMITTED', phase_id=cfg['phase_id'], update_index=update)
    return prepared


def run(suite, phases, ledger):
    """Run after the launcher verifies approval, locks the RPC lane and registers slots."""
    verification = verify(suite, phases)
    require(Path(suite['budget_path']).is_file(), 'training ledger must be registered by the authorized launcher')
    stop = Path(suite['control_directory']) / 'STOP.properties'
    require(not stop.exists(), 'preserved STOP requires reconciliation')
    ledger.resume()
    torch.set_num_threads(suite.get('cpu_threads', 2))
    campaign = verification['suite']
    try:
        initial = _initial(suite, verification)
        checkpoint = initial
        updates = []
        for cfg, phase_ref in zip(phases, suite['phase_configs'], strict=True):
            require(not stop.exists(), 'preserved STOP requires reconciliation')
            prepared = _train(suite, cfg, phase_ref, campaign, checkpoint, ledger)
            checkpoint = prepared['checkpoint']
            updates.append(prepared['training_summary'])
        final_model = _load(suite, campaign, checkpoint, len(updates))
        final_actor_path = Path(suite['arms'][0]['python_output']) / 'final_actor.pt'
        if final_actor_path.is_file():
            final_actor = final_model.bind_actor_snapshot(final_actor_path)
        else:
            final_actor = final_model.save_actor(final_actor_path)
        require(sha(CORE) == PROTECTED_SHA, 'protected PPO core changed during the campaign')
        result = dict(schema=SCHEMA + '-completion', status='COMPLETE', suite=campaign,
                      learner=suite['reinforce'], initialization=suite['initial_actor'], initial_checkpoint=initial,
                      final_checkpoint=checkpoint, final_actor=final_actor,
                      training_shifts=len(suite['jobs']), training_updates=len(updates),
                      evaluation_shifts=0, updates=updates, protected_ppo_sha256=PROTECTED_SHA)
        report = Path(suite['report_directory']) / 'training_complete.json'
        _same_or_save(report, result)
        _same_or_save(Path(suite['python_output']) / 'suite_complete.json', dict(result, report=reference(report)))
        write_progress(suite, status='COMPLETE', completed_shifts=len(suite['jobs']), training_updates=len(updates))
        return result
    except BaseException as error:
        if not stop.exists():
            stop.parent.mkdir(parents=True, exist_ok=True)
            _publish(stop, b'status=STOP\nreason=REINFORCE_CONTROLLER_ERROR\n')
        write_progress(suite, status='STOP', error=repr(error))
        raise
