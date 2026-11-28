"""Collect full S02 shifts and commit exactly the recorded PPO Actor-step schedule."""
from pathlib import Path

import torch

from business_graph_matched_reinforce_v1.learner import MatchedReinforce, SETTINGS_SCHEMA
from business_graph_reinforce_v1 import runtime as original
from business_graph_summary_campaign_v1.plan_io import load_reference, reference
from business_graph_summary_campaign_v1.progress import write_progress
from business_graph_summary_v1.contract import require
from change_plans.token_attention_001.business_graph_summary_009.gat_weak_ablation_001.weak_plan import bind


SCHEMA = 'bg128-ppo-step-matched-full-shift-training-v1'


def verify(suite, phases):
    status = original.verify(suite, phases)
    steps = suite['reinforce']['actor_steps_by_update']
    historical = suite['full_s02_step_evidence']
    require(len(steps) == len(historical) == len(suite['ppo_minibatches_per_epoch']) == len(phases) == 32
            and suite['ppo_epochs'] == 4 and sum(steps) == 23612,
            'S02 checkpoint or total Actor-step budget differs')
    for number, (expected, row, minibatches) in enumerate(zip(steps, historical, suite['ppo_minibatches_per_epoch'],
                                                                strict=True), 1):
        archived = load_reference(row['training_summary'])['update']
        require(archived['update_index'] == row['update'] == number and
                archived['epochs'] == row['epochs'] == suite['ppo_epochs'] and
                row['minibatches_per_epoch'] == minibatches and
                archived['actor_optimizer_steps'] == row['actor_optimizer_steps'] == expected == 4 * minibatches,
                'Actor-step count differs from the sealed Full S02 training summary')
    return dict(status, planned_actor_optimizer_steps=sum(steps), critic_optimizer_steps=0)


# Bind the existing sealed actor, optimizer and physical-ledger transaction to the
# new checkpoint format in a private namespace, preserving the original campaign.
_read_checkpoint = bind(original._load, FullShiftReinforce=MatchedReinforce, SETTINGS_SCHEMA=SETTINGS_SCHEMA)


def _load(suite, campaign, checkpoint, update):
    model = _read_checkpoint(suite, campaign, checkpoint, update)
    expected = suite['reinforce']['actor_steps_by_update'][:update]
    actual = [receipt['actor_optimizer_steps'] for receipt in model.consumed_batches.values()]
    require(actual == expected, 'restored batch history differs from S02 Actor-step schedule')
    if update:
        require(next(reversed(model.consumed_batches.values()))['cumulative_actor_optimizer_steps'] == sum(expected),
                'restored cumulative number of Adam updates differs')
    return model


_initial = bind(original._initial, FullShiftReinforce=MatchedReinforce, _load=_load)
_verify_prepared = bind(original._verify_prepared, _load=_load)
_train = bind(original._train, _load=_load, _verify_prepared=_verify_prepared, SCHEMA=SCHEMA)


def run(suite, phases, ledger):
    status = verify(suite, phases)
    require(Path(suite['budget_path']).is_file(), 'training ledger must be registered by the authorized launcher')
    stop = Path(suite['control_directory']) / 'STOP.properties'
    require(not stop.exists(), 'preserved STOP requires reconciliation')
    ledger.resume()
    torch.set_num_threads(suite.get('cpu_threads', 2))
    campaign = status['suite']
    try:
        initial = _initial(suite, status)
        checkpoint = initial
        updates = []
        for cfg, phase_ref in zip(phases, suite['phase_configs'], strict=True):
            require(not stop.exists(), 'preserved STOP requires reconciliation')
            prepared = _train(suite, cfg, phase_ref, campaign, checkpoint, ledger)
            checkpoint = prepared['checkpoint']
            updates.append(prepared['training_summary'])
        model = _load(suite, campaign, checkpoint, len(updates))
        steps = [receipt['actor_optimizer_steps'] for receipt in model.consumed_batches.values()]
        actor_path = Path(suite['arms'][0]['python_output']) / 'final_actor.pt'
        final_actor = model.bind_actor_snapshot(actor_path) if actor_path.exists() else model.save_actor(actor_path)
        require(original.sha(original.CORE) == original.PROTECTED_SHA, 'protected PPO core changed during training')
        result = dict(schema=SCHEMA + '-completion', status='COMPLETE', suite=campaign,
                      learner=suite['reinforce'], initialization=suite['initial_actor'], initial_checkpoint=initial,
                      final_checkpoint=checkpoint, final_actor=final_actor, training_shifts=len(suite['jobs']),
                      training_updates=len(updates), actor_steps_by_update=steps, actor_optimizer_steps=sum(steps),
                      critic_optimizer_steps=0, evaluation_shifts=0, updates=updates,
                      protected_ppo_sha256=original.PROTECTED_SHA)
        report = Path(suite['report_directory']) / 'training_complete.json'
        original._same_or_save(report, result)
        original._same_or_save(Path(suite['python_output']) / 'suite_complete.json', dict(result, report=reference(report)))
        write_progress(suite, status='COMPLETE', completed_shifts=len(suite['jobs']), training_updates=len(updates),
                       actor_optimizer_steps=sum(steps))
        return result
    except BaseException as error:
        if not stop.exists():
            stop.parent.mkdir(parents=True, exist_ok=True)
            original._publish(stop, b'status=STOP\nreason=MATCHED_REINFORCE_CONTROLLER_ERROR\n')
        write_progress(suite, status='STOP', error=repr(error))
        raise
