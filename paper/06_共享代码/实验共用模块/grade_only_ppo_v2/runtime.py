"""Approval-gated, four-shift real smoke for one grade-only advantage candidate.

Import and preflight never start RPC, create a learner, claim budget or run a
simulation. Real execution is an explicit separate command after approval.
"""
import argparse
from contextlib import contextmanager
import copy
import fcntl
import hashlib
import json
from pathlib import Path

import numpy as np

from online_v1_1.operations import _publish
from baseline1_v1.java_plan import COLUMNS
from .gate import encoded, preflight, reference, require, sha, check_sources


def _save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _publish(path, encoded(value))


def _same_or_save(path, value):
    path = Path(path)
    if path.exists():
        require(path.read_bytes() == encoded(value), 'immutable transaction differs: ' + str(path))
    else:
        _save(path, value)


@contextmanager
def _controller_lock(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.controller.lock').open('a+b') as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('another controller owns this campaign') from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _publisher(campaign, ledger):
    control = Path(campaign['control_directory'])
    def publish(job, config_path):
        check_sources(campaign['source_pins'])
        require(reference(config_path) in campaign['phase_configs'], 'unfrozen phase before ready')
        row = ledger.snapshot()['jobs'][str(job['run_index'])]
        require(row['status'] == 'CLAIMED' and row['identity'] == job, 'ready needs exact claimed slot')
        require(not (control / 'STOP.properties').exists(), 'preserved STOP; reconcile without resubmission')
        values = {key: str(job[key]) for key in COLUMNS}
        values.update(protocol='baseline1-one-click-ready-v1', campaign_id=campaign['campaign_id'],
            master_plan_sha256=campaign['master_plan']['sha256'],
            instrumentation_revision='boundary-events-v1-native-rpc-runtime-scene-v1',
            config_sha256=sha(config_path), snapshot_sha256=sha(config_path),
            snapshot_path=str(Path(config_path).resolve()))
        require(all(value.isascii() and '\n' not in value and '\r' not in value for value in values.values()),
                'ambiguous ready Properties')
        path = control / 'ready' / ('run_' + str(job['run_index']) + '.properties')
        path.parent.mkdir(parents=True, exist_ok=True)
        _publish(path, ''.join(key + '=' + value + '\n' for key, value in values.items()).encode('ascii'))
    return publish


def _actor_snapshot(model, path):
    """Bind an unchanged phase Actor artifact, never replace a published file."""
    if Path(path).exists():
        model.bind_actor_snapshot(path)
    else:
        model.save_actor(path)
    return reference(path)


def _phase_data(cfg, config_path, campaign, ledger, model):
    from baseline1_v1.collection import OnlineActor
    from .collection import MaterialPhaseCollector, load_admitted_episode
    check_sources(campaign['source_pins'])
    require(reference(config_path) in campaign['phase_configs'], 'phase config source changed')
    actor_ref = _actor_snapshot(model, Path(cfg['python_output']) / 'actor.pt')
    state = ledger.resume()
    remaining = [job for job in cfg['jobs'] if state['jobs'][str(job['run_index'])]['status'] == 'PLANNED']
    if remaining:
        active = copy.deepcopy(cfg)
        active['jobs'] = remaining
        behavior = OnlineActor(model.actor, remaining)
        with MaterialPhaseCollector(active, config_path, publish_ready=_publisher(campaign, ledger),
                startup=dict(control_directory=campaign['control_directory'], campaign_id=campaign['campaign_id'],
                    master_plan_sha256=campaign['master_plan']['sha256'])) as collector:
            for job in remaining:
                ledger.claim(job)
                try:
                    _, dataset_ref, receipt = collector.collect(job, behavior, actor_path=actor_ref['path'])
                    ledger.complete(job, dataset_ref, receipt)
                except BaseException as error:
                    if ledger.snapshot()['jobs'][str(job['run_index'])]['status'] == 'CLAIMED':
                        ledger.unknown(job, repr(error))
                    raise
                print('COMPLETE phase=' + cfg['phase_id'] + ' run_index=' + str(job['run_index']), flush=True)
    state = ledger.resume()
    episodes, refs = [], []
    for job in cfg['jobs']:
        row = state['jobs'][str(job['run_index'])]
        require(row['status'] == 'COMPLETE' and row['identity'] == job
                and row['receipt']['phase_config'] == reference(config_path), 'durable physical slot differs')
        require(reference(row['dataset_ref']['path']) == row['dataset_ref'], 'derived dataset manifest changed')
        episode = load_admitted_episode(row['receipt'])
        identity = episode['identity']
        require(identity['run_index'] == job['run_index'] and identity['environment_seed'] == job['experiment_seed']
                and identity['actor_sampling_seed'] == job['actor_sampling_seed']
                and identity['policy_sha256'] == actor_ref['sha256']
                and identity['policy_version'] == model.policy_version, 'phase Actor/episode identity changed')
        episodes.append(episode)
        refs.append(row['dataset_ref'])
    return episodes, refs, actor_ref


def _batch_id(campaign, cfg, refs):
    return hashlib.sha256(encoded(dict(campaign_id=campaign['campaign_id'], phase_id=cfg['phase_id'],
                                      sources=refs))).hexdigest()


def _targets(path, arrays, intent):
    from python.hdf5_arrays import write_array_bundle, read_array_bundle
    metadata = dict(schema='grade-only-frozen-targets-v1', transaction=intent,
                    reward_version='grade-only-alp-clip-linear-v1', waiting_in_targets=False,
                    production_in_targets=False)
    if Path(path).exists():
        stored, observed = read_array_bundle(path)
        require(observed == metadata and set(stored) == set(arrays)
                and all(np.array_equal(stored[key], arrays[key]) for key in arrays),
                'previous uncommitted frozen targets differ; stop for reconciliation')
    else:
        write_array_bundle(path, arrays, metadata)
    return reference(path)


def _verify_target_ref(ref, intent):
    from python.hdf5_arrays import read_array_bundle
    require(reference(ref['path']) == ref, 'frozen target bytes changed')
    _, metadata = read_array_bundle(ref['path'])
    require(metadata['transaction'] == intent and metadata['waiting_in_targets'] is False
            and metadata['production_in_targets'] is False, 'frozen target contract or transaction changed')


def _load_model(ref, config, *, restore=True):
    from .learner import GradeOnlyPPO
    return GradeOnlyPPO.from_checkpoint(ref['path'], expected_config=config,
                                        expected_sha256=ref['sha256'], restore_global_rng=restore)


def _learning_contract(campaign):
    cfg = campaign['learner_config']
    return dict(execution_profile=cfg['execution_profile'], candidate_id=cfg['candidate_id'],
                estimator=cfg['actor_advantage']['estimator'], gamma=cfg['ppo']['gamma'],
                gae_lambda=cfg['ppo']['gae_lambda'])


def _equal_tree(left, right):
    import torch
    if torch.is_tensor(left):
        return torch.is_tensor(right) and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(_equal_tree(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(_equal_tree(a,b) for a,b in zip(left,right))
    return type(left) is type(right) and left == right


def _verify_checkpoint_restore(model, ref, receipt_path):
    """Reload without fitting; verify the complete learner transaction state."""
    from python.constrained_ppo import digest
    restored = _load_model(ref, model.config, restore=False)
    checks = dict(actor=restored.policy_version == model.policy_version,
        critic=digest(restored.critic) == digest(model.critic),
        actor_optimizer=_equal_tree(restored.actor_optimizer.state_dict(), model.actor_optimizer.state_dict()),
        critic_optimizer=_equal_tree(restored.critic_optimizer.state_dict(), model.critic_optimizer.state_dict()),
        generator=_equal_tree(restored.generator.get_state(), model.generator.get_state()),
        sampling_generator=_equal_tree(restored.sampling_generator.get_state(), model.sampling_generator.get_state()),
        consumed_batches=_equal_tree(restored.consumed_batches, model.consumed_batches),
        consumed_episodes=_equal_tree(restored.consumed_episodes, model.consumed_episodes),
        runtime_state=_equal_tree(restored.runtime_state, model.runtime_state),
        frozen_targets=_equal_tree(restored.last_targets, model.last_targets),
        target_hash=restored.last_target_hash == model.last_target_hash,
        update_index=restored.update_index == model.update_index)
    require(all(checks.values()), 'smoke checkpoint exact restore differs')
    _same_or_save(receipt_path, dict(schema='grade-adv-smoke-checkpoint-restore-v1',
        checkpoint=ref, candidate_id=model.config['candidate_id'],
        execution_profile=model.config['execution_profile'], update_index=model.update_index,
        optimizer_updates_in_verification=0, checks=checks))


def _train_phase(model, episodes, refs, *, campaign, campaign_path, cfg, ledger, output):
    update = cfg['jobs'][0]['target_training_update']
    folder = Path(output) / 'updates' / ('%04d' % update)
    batch_id = _batch_id(campaign, cfg, refs)
    intent = dict(learning_contract=_learning_contract(campaign), campaign=reference(campaign_path), phase_id=cfg['phase_id'], update=update,
                  batch_id=batch_id, sources=refs, run_indices=[job['run_index'] for job in cfg['jobs']])
    _same_or_save(folder / 'intent.json', intent)
    checkpoint = folder / 'checkpoint.pt'
    if checkpoint.exists():
        model = _load_model(reference(checkpoint), model.config)
        require(model.update_index == update and model.runtime_state.get('transaction') == intent,
                'uncommitted checkpoint belongs to another transaction')
        _verify_target_ref(model.runtime_state['training_targets'], intent)
        require(reference(folder / 'training_summary.json') == model.runtime_state['training_summary'],
                'uncommitted checkpoint summary changed')
    else:
        require(model.update_index == update - 1, 'PPO update is not the next complete batch')
        result = model.update(episodes, batch_id=batch_id)
        targets = _targets(folder / 'training_targets.h5', model.last_targets, intent)
        _same_or_save(folder / 'training_summary.json', dict(transaction=intent, update=result))
        model.runtime_state.update(transaction=intent, training_targets=targets,
                                   training_summary=reference(folder / 'training_summary.json'))
        model.save_checkpoint(checkpoint)
    _verify_checkpoint_restore(model, reference(checkpoint), folder / 'checkpoint_restore_verification.json')
    prepared = dict(checkpoint=reference(checkpoint), intent=intent,
                    training_targets=model.runtime_state['training_targets'])
    _same_or_save(folder / 'prepared.json', prepared)
    ledger.commit(update, cfg['jobs'], prepared['checkpoint'], batch_id)
    print('COMMITTED update=' + str(update) + ' grade-only complete-shift batch', flush=True)
    return model, prepared['checkpoint']


def _checkpoint_for(campaign, ledger, update):
    if update == 0:
        return json.loads((Path(campaign['python_output']) / 'initial_checkpoint.json').read_bytes())
    return ledger.snapshot()['campaigns'][campaign['campaign_id']]['commits'][str(update)]['checkpoint_ref']


def _evaluation(cfg, config_path, campaign, campaign_path, ledger, checkpoint, learner_config):
    from .evaluation import evaluate_episodes
    from .collection import load_admitted_episode
    target = Path(cfg['python_output']) / 'evaluation_complete.json'
    receipt_path = target.with_name('evaluation_receipt.json')
    binding = dict(campaign=reference(campaign_path), phase_config=reference(config_path), checkpoint=checkpoint)
    if target.exists():
        require(receipt_path.is_file(), 'evaluation report lacks complete receipt; never resample')
        receipt = json.loads(receipt_path.read_bytes())
        require(receipt == dict(binding, report=reference(target)), 'saved evaluation provenance changed')
        report = json.loads(target.read_bytes())
        state = ledger.resume()
        refs = []
        for job in cfg['jobs']:
            row = state['jobs'][str(job['run_index'])]
            require(row['status'] == 'COMPLETE' and row['identity'] == job
                    and row['receipt']['phase_config'] == binding['phase_config'], 'evaluation physical receipt changed')
            load_admitted_episode(row['receipt'])
            refs.append(row['dataset_ref'])
        require(report.get('learning_contract') == _learning_contract(campaign)
                and report.get('smoke_only') is True and report.get('learning_effect_claim') is False,
                'saved evaluation smoke/candidate contract differs')
        require(report.get('campaign') == binding['campaign'] and report.get('phase_config') == binding['phase_config']
                and report.get('checkpoint_provenance') == checkpoint and report.get('source_refs') == refs
                and report.get('n_shifts') == len(cfg['jobs']) and report.get('training_state_unchanged') is True
                and report.get('optimizer_updates') == 0 and report.get('gradient_updates') == 0,
                'saved independent evaluation scope differs')
        archived = _load_model(checkpoint, learner_config, restore=False)
        from python.constrained_ppo import digest
        require(report.get('fingerprints') == dict(actor=digest(archived.actor), critic=digest(archived.critic)),
                'evaluation checkpoint fingerprints differ')
        return reference(target)
    model = _load_model(checkpoint, learner_config, restore=False)
    require(model.update_index == cfg['jobs'][0]['checkpoint_update'], 'evaluation wrong checkpoint update')
    episodes, refs, _ = _phase_data(cfg, config_path, campaign, ledger, model)
    report = evaluate_episodes(episodes, model, expected_complete_shifts=len(cfg['jobs']),
                              checkpoint_provenance=checkpoint)
    report.update(learning_contract=_learning_contract(campaign), smoke_only=True, learning_effect_claim=False,
                  campaign=binding['campaign'], phase_config=binding['phase_config'],
                  phase_id=cfg['phase_id'], source_refs=refs)
    _same_or_save(target, report)
    _same_or_save(receipt_path, dict(binding, report=reference(target)))
    return reference(target)


def _verify_committed(cfg, campaign, campaign_path, ledger, learner_config):
    folder = Path(campaign['python_output']) / 'updates' / ('%04d' % cfg['jobs'][0]['target_training_update'])
    prepared = json.loads((folder / 'prepared.json').read_bytes())
    commit = ledger.snapshot()['campaigns'][campaign['campaign_id']]['commits'][str(cfg['jobs'][0]['target_training_update'])]
    require(prepared['checkpoint'] == commit['checkpoint_ref'], 'committed checkpoint differs from prepared receipt')
    archived = _load_model(prepared['checkpoint'], learner_config, restore=False)
    _verify_target_ref(prepared['training_targets'], prepared['intent'])
    require(archived.runtime_state.get('transaction') == prepared['intent']
            and archived.runtime_state.get('training_targets') == prepared['training_targets'],
            'committed learner transaction changed')
    require(reference(folder / 'training_summary.json') == archived.runtime_state['training_summary'],
            'committed training summary changed')
    from .collection import load_admitted_episode
    state = ledger.resume()
    refs = []
    for job in cfg['jobs']:
        row = state['jobs'][str(job['run_index'])]
        load_admitted_episode(row['receipt'])
        refs.append(row['dataset_ref'])
    expected = dict(learning_contract=_learning_contract(campaign), campaign=reference(campaign_path), phase_id=cfg['phase_id'],
        update=cfg['jobs'][0]['target_training_update'], batch_id=_batch_id(campaign, cfg, refs),
        sources=refs, run_indices=[job['run_index'] for job in cfg['jobs']])
    require(prepared['intent'] == expected and commit['batch_id'] == expected['batch_id'],
            'committed durable inputs changed')


def _trial(campaign_path, campaign, phases, ledger):
    import torch
    from .learner import GradeOnlyPPO
    torch.set_num_threads(campaign.get('cpu_threads', 2))
    output = Path(campaign['python_output'])
    config = campaign['learner_config']
    _same_or_save(output / 'authorized_learner_config.json',
                  dict(config=config, campaign=reference(campaign_path)))
    state = ledger.resume()
    commits = state['campaigns'][campaign['campaign_id']]['commits']
    initial = output / 'initial_checkpoint.json'
    if commits:
        checkpoint = commits[str(len(commits))]['checkpoint_ref']
        model = _load_model(checkpoint, config)
    elif initial.exists():
        checkpoint = json.loads(initial.read_bytes())
        model = _load_model(checkpoint, config)
        require(model.update_index == 0 and model.runtime_state.get('campaign') == reference(campaign_path),
                'initial checkpoint campaign differs')
    else:
        require(not (output / 'initial.pt').exists(), 'unpublished initial checkpoint; reconcile without overwrite')
        model = GradeOnlyPPO(config)
        model.runtime_state['campaign'] = reference(campaign_path)
        model.save_checkpoint(output / 'initial.pt')
        checkpoint = reference(output / 'initial.pt')
        _save(initial, checkpoint)
    evaluations = []
    for cfg, spec in zip(phases, campaign['phase_configs']):
        role = cfg['jobs'][0]['role']
        update = cfg['jobs'][0]['target_training_update']
        if role == 'train':
            if update <= model.update_index:
                _verify_committed(cfg, campaign, campaign_path, ledger, config)
                continue
            episodes, refs, _ = _phase_data(cfg, spec['path'], campaign, ledger, model)
            model, checkpoint = _train_phase(model, episodes, refs, campaign=campaign,
                campaign_path=campaign_path, cfg=cfg, ledger=ledger, output=output)
            del episodes
        else:
            evaluations.append(_evaluation(cfg, spec['path'], campaign, campaign_path, ledger,
                                           _checkpoint_for(campaign, ledger, cfg['jobs'][0]['checkpoint_update']), config))
    state = ledger.resume()
    training_updates = sum(cfg['jobs'][0]['role'] == 'train' for cfg in phases)
    require(model.update_index == training_updates and all(state['jobs'][str(job['run_index'])]['status'] == 'COMPLETE'
            for job in campaign['jobs']), 'campaign is incomplete')
    require(training_updates == 1 and len(campaign['jobs']) == 4, 'real smoke must stop after four shifts and one update')
    result = dict(status='GRADE_ADVANTAGE_REAL_SMOKE_COMPLETE_NOT_LEARNING_EVIDENCE',
        learning_contract=_learning_contract(campaign), smoke_only=True, learning_effect_claim=False,
        campaign=reference(campaign_path), checkpoint=checkpoint, evaluation_reports=evaluations,
        training_updates=model.update_index, training_shifts=sum(job['role']=='train' for job in campaign['jobs']),
        physical_shifts=len(campaign['jobs']), production_guarantee=False,
        waiting_in_training_objective=False, production_in_training_objective=False,
        multi_seed_claim=False)
    _same_or_save(output / 'trial_complete.json', result)
    return result


def run(campaign_path, approval_path):
    from .budget import BudgetLedger, GLOBAL_PATH
    campaign, phases, approval = preflight(campaign_path, approval_path=approval_path)
    require(approval is not None, 'separate budget and execution approval required')
    ledger = BudgetLedger(GLOBAL_PATH, campaign, campaign_sha256=sha(campaign_path),
                          approval=approval, approval_sha256=sha(approval_path))
    output = Path(campaign['python_output'])
    with _controller_lock(output):
        ledger.resume()
        stop = Path(campaign['control_directory']) / 'STOP.properties'
        require(not stop.exists(), 'preserved STOP requires recovery review; do not clear automatically')
        try:
            return _trial(campaign_path, campaign, phases, ledger)
        except BaseException as error:
            if not stop.exists():
                stop.parent.mkdir(parents=True, exist_ok=True)
                _publish(stop, ('reason=' + repr(error).replace('\n', ' ') + '\n').encode())
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('preflight', 'run'):
        choice = sub.add_parser(command)
        choice.add_argument('--campaign', required=True)
        choice.add_argument('--approval', required=command=='run')
    args = parser.parse_args(argv)
    if args.command == 'preflight':
        campaign, phases, approval = preflight(args.campaign, approval_path=args.approval)
        print(json.dumps(dict(status='PREFLIGHT_PASS_NO_LISTENER', campaign=reference(args.campaign),
            physical_shifts=campaign['new_shift_cap'], phases=len(phases), execution_authorized=approval is not None,
            learning_contract=_learning_contract(campaign), smoke_only=True,
            reward='negative clipped grade cost only', production_guarantee=False), ensure_ascii=False))
        return 0
    run(args.campaign, args.approval)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
