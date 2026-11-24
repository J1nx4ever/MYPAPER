"""One AnyLogic Run, 100 new shifts, one frozen BG128 Actor/Critic, no fitting."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path

import torch

from baseline1_v1.evaluation import (ReadOnlyTrainingGuard, training_state_fingerprint,
                                     _module_state, _tree_hash)
from baseline1_v1.java_plan import COLUMNS
from grade_only_ppo_v2.runtime import _controller_lock, _actor_snapshot
from business_graph_summary_recovery_v1.collection import (OnlineActor, MaterialPhaseCollector,
                                                  load_admitted_episode)
from business_graph_summary_v1.evaluation import evaluate_episodes
from business_graph_summary_v1.learner import JointCostPPO
from business_graph_summary_v1.models import digest
from online_v1_1.operations import _publish

from .gate import (encoded, reference, sha, require, preflight,
                   check_sources, RUN_COUNT)


def _save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _publish(path, encoded(value))


def _same_or_save(path, value):
    path = Path(path)
    if path.exists():
        require(path.read_bytes() == encoded(value), 'immutable independent evaluation publication changed')
    else:
        _save(path, value)


def _status(suite, status, *, completed_shifts=0, **values):
    report = dict(schema='business-graph-summary-fixed-checkpoint-evaluation-progress-v1', campaign_id=suite['campaign_id'],
                  updated_at_utc=datetime.now(timezone.utc).isoformat(), status=status,
                  completed_shifts=completed_shifts, total_shifts=RUN_COUNT,
                  checkpoint_update=suite['checkpoint_update'],
                  training_updates=0, optimizer_updates=0, **values)
    path = Path(suite['python_output']) / 'run_status.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    _publish(path, encoded(report), replace=True)
    return report


def _parent_artifacts(suite):
    expected = suite['parent_artifacts']
    actual = {key: reference(ref['path']) for key, ref in expected.items()}
    require(actual == expected and set(actual) == {'checkpoint', 'manifest', 'seal'},
            'fixed BG128 checkpoint/manifest/seal bytes changed')
    selection = suite['selection']
    require(reference(selection['path']) == selection,
            'pre-evaluation checkpoint selection evidence changed')
    return actual


def _load_model(suite):
    parent = _parent_artifacts(suite)['checkpoint']
    model = JointCostPPO.from_checkpoint(parent['path'], expected_config=suite['learner_config'],
        expected_sha256=parent['sha256'], restore_global_rng=False)
    require(model.update_index == suite['checkpoint_update'] and
            model.config['candidate_id'] == 's01_BG128' and
            model.config['reward_contract']['wait_weight'] == .2,
            'independent evaluation requires the selected frozen BG128 checkpoint')
    return model


def _portable_state_fingerprint(model):
    """Persist training state without process-local optimizer parameter addresses.

    The live ReadOnlyTrainingGuard additionally checks those addresses, while
    this receipt digest stays reproducible after a controller-only restart.
    """
    handled = {'actor', 'critic', 'actor_optimizer', 'critic_optimizer'}
    state = dict(actor=_module_state(model.actor), critic=_module_state(model.critic),
                 actor_optimizer=model.actor_optimizer.state_dict(),
                 critic_optimizer=model.critic_optimizer.state_dict())
    for name, value in vars(model).items():
        if name in handled or callable(value):
            continue
        if name == '_summary_cache':
            # Only deterministic input summaries are cached; its object address,
            # LRU entries and hit counters are execution details, not learning.
            value = dict(feature_config=value.feature_config,
                         max_tokens=value.max_tokens, max_bytes=value.max_bytes)
        elif name == '_actor_artifact' and value is not None:
            # save_actor and bind_actor_snapshot return equivalent references
            # with different redundant metadata inventories.
            value = {key: value[key] for key in ('path', 'sha256', 'actor_digest')}
        state[name] = value
    return _tree_hash(state)


def _publisher(suite, ledger):
    control = Path(suite['control_directory'])
    def publish(job, config_path):
        check_sources(suite['source_pins'])
        _parent_artifacts(suite)
        require(reference(config_path) == suite['phase_config'], 'unfrozen independent evaluation config')
        row = ledger.snapshot()['jobs'][str(job['run_index'])]
        require(row['status'] == 'CLAIMED' and row['identity'] == job,
                'evaluation ready requires the exact claimed physical slot')
        require(not (control / 'STOP.properties').exists(), 'preserved STOP; never automatically resubmit')
        values = {key: str(job[key]) for key in COLUMNS}
        values.update(protocol='baseline1-one-click-ready-v1', campaign_id=suite['campaign_id'],
            master_plan_sha256=suite['master_plan']['sha256'],
            instrumentation_revision='boundary-events-v1-native-rpc-runtime-scene-v1',
            config_sha256=sha(config_path), snapshot_sha256=sha(config_path),
            snapshot_path=str(Path(config_path).resolve()),
            all_comparison_checkpoints_frozen='true')
        require(all(value.isascii() and '\n' not in value and '\r' not in value for value in values.values()),
                'ambiguous independent evaluation ready Properties')
        path = control / 'ready' / ('run_' + str(job['run_index']) + '.properties')
        path.parent.mkdir(parents=True, exist_ok=True)
        _publish(path, ''.join(key + '=' + value + '\n' for key, value in values.items()).encode('ascii'))
    return publish


def _load_completed(suite, cfg, ledger, model, actor_ref):
    state = ledger.resume()
    episodes, refs, receipts = [], [], []
    for job in cfg['jobs']:
        row = state['jobs'][str(job['run_index'])]
        require(row['status'] == 'COMPLETE' and row['identity'] == job and
                row['receipt']['phase_config'] == suite['phase_config'],
                'independent evaluation physical receipt differs')
        require(reference(row['dataset_ref']['path']) == row['dataset_ref'],
                'independent evaluation derived manifest changed')
        episode = load_admitted_episode(row['receipt'])
        identity = episode['identity']
        require(identity['run_index'] == job['run_index'] and
                identity['environment_seed'] == job['experiment_seed'] and
                identity['actor_sampling_seed'] == job['actor_sampling_seed'] and
                identity['policy_sha256'] == actor_ref['sha256'] and
                identity['policy_version'] == model.policy_version and identity['split'] == 'validation',
                'fixed evaluation Actor, new seed or validation identity differs')
        episodes.append(episode)
        refs.append(row['dataset_ref'])
        receipts.append(row['receipt'])
    require(len(episodes) == RUN_COUNT, 'all100 completed independent shifts required')
    return episodes, refs, receipts


def _collect(suite, cfg, ledger, model, actor_ref):
    state = ledger.resume()
    remaining = [job for job in cfg['jobs'] if state['jobs'][str(job['run_index'])]['status'] == 'PLANNED']
    complete = len(cfg['jobs']) - len(remaining)
    if remaining:
        active = copy.deepcopy(cfg)
        active['jobs'] = remaining
        behavior = OnlineActor(model.actor, remaining)
        with MaterialPhaseCollector(active, suite['phase_config']['path'],
                publish_ready=_publisher(suite, ledger),
                startup=dict(control_directory=suite['control_directory'], campaign_id=suite['campaign_id'],
                             master_plan_sha256=suite['master_plan']['sha256'])) as collector:
            for job in remaining:
                _status(suite, 'COLLECTING', completed_shifts=complete, run_index=job['run_index'],
                        phase_id=cfg['phase_id'], ordinal=job['ordinal'])
                ledger.claim(job)
                try:
                    _, dataset_ref, receipt = collector.collect(job, behavior, actor_path=actor_ref['path'])
                    ledger.complete(job, dataset_ref, receipt)
                except BaseException as error:
                    if ledger.snapshot()['jobs'][str(job['run_index'])]['status'] == 'CLAIMED':
                        ledger.unknown(job, repr(error))
                    raise
                complete += 1
                _status(suite, 'COLLECTING', completed_shifts=complete, run_index=job['run_index'],
                        phase_id=cfg['phase_id'])
                print('COMPLETE independent_evaluation run_index=' + str(job['run_index']) +
                      ' shifts=' + str(complete) + '/' + str(RUN_COUNT), flush=True)
    return _load_completed(suite, cfg, ledger, model, actor_ref)


def _completed(suite_path, suite, ledger):
    path = Path(suite['python_output']) / 'suite_complete.json'
    if not path.exists():
        return None
    result = json.loads(path.read_bytes())
    require(result.get('campaign') == reference(suite_path) and
            result.get('physical_shifts') == RUN_COUNT and result.get('training_updates') == 0 and
            result.get('optimizer_updates') == 0 and result.get('checkpoint_update') == suite['checkpoint_update'] and
            result.get('parent_artifacts_before') == result.get('parent_artifacts_after') == _parent_artifacts(suite),
            'saved independent evaluation completion identity changed')
    state = ledger.resume()
    require(all(state['jobs'][str(job['run_index'])]['status'] == 'COMPLETE' for job in suite['jobs']) and
            state['campaigns'][suite['campaign_id']]['commits'] == {},
            'saved independent evaluation physical state or zero-update ledger changed')
    for ref in result['reports'].values():
        require(reference(ref['path']) == ref, 'saved independent evaluation report bytes changed')
    return result


def _trial(suite_path, suite, cfg, ledger):
    from .metrics import build_report, write_reports
    completed = _completed(suite_path, suite, ledger)
    if completed is not None:
        return completed
    output = Path(suite['python_output'])
    parent_before = _parent_artifacts(suite)
    model = _load_model(suite)
    # Binding an output Actor artifact changes only its metadata pointer. Finish
    # that publication before guarding the entire sampling/evaluation lifecycle.
    actor_ref = _actor_snapshot(model, output / 'actor.pt')
    before = training_state_fingerprint(model)
    portable_before = _portable_state_fingerprint(model)
    fingerprints = dict(actor=digest(model.actor), critic=digest(model.critic))
    with ReadOnlyTrainingGuard(model):
        episodes, refs, _ = _collect(suite, cfg, ledger, model, actor_ref)
        _status(suite, 'EVALUATING', completed_shifts=RUN_COUNT, phase_id=cfg['phase_id'])
        report = evaluate_episodes(episodes, model, expected_complete_shifts=RUN_COUNT,
                                  checkpoint_provenance=parent_before['checkpoint'])
    require(training_state_fingerprint(model) == before and model.update_index == suite['checkpoint_update'],
            'sampling/evaluation modified frozen model, optimizer, RNG or training ledger')
    require(_portable_state_fingerprint(model) == portable_before, 'portable frozen training state changed')
    require(report.get('optimizer_updates') == report.get('gradient_updates') == 0 and
            report.get('fingerprints') == fingerprints and report.get('training_state_unchanged') is True,
            'evaluation must certify unchanged Actor/Critic and zero optimization')
    parent_after = _parent_artifacts(suite)
    require(parent_before == parent_after, 'fixed historical checkpoint artifacts changed during evaluation')
    binding = dict(campaign=reference(suite_path), phase_config=suite['phase_config'],
                   checkpoint=parent_before['checkpoint'], actor=actor_ref, source_refs=refs)
    report.update(campaign=binding['campaign'], phase_config=binding['phase_config'], source_refs=refs,
                  selection_review=suite['selection'],
                  evaluation_scope='new_seed_fixed_checkpoint_independent_revalidation',
                  smoke_only=False, stable_policy_claim=False, checkpoint_selected_before_sampling=True,
                  model_state_before=portable_before, model_state_after=_portable_state_fingerprint(model))
    report_path = output / 'evaluation_complete.json'
    _same_or_save(report_path, report)
    summary = build_report(report, expected_complete_shifts=RUN_COUNT)
    reports = write_reports(summary, report, output / 'reports')
    require(isinstance(reports, dict) and set(reports) == {'summary', 'per_shift', 'per_window', 'readme'},
            'complete independent evaluation metric publications required')
    reports = dict(reports, evaluation=reference(report_path))
    receipt_path = output / 'evaluation_receipt.json'
    receipt = dict(schema='business-graph-summary-fixed-checkpoint-independent-evaluation-receipt-v1', **binding,
                   reports=reports, parent_artifacts_before=parent_before, parent_artifacts_after=parent_after,
                   model_state_before=portable_before, model_state_after=_portable_state_fingerprint(model),
                   physical_shifts=RUN_COUNT, checkpoint_update=suite['checkpoint_update'], optimizer_updates=0, training_updates=0)
    _same_or_save(receipt_path, receipt)
    reports['receipt'] = reference(receipt_path)
    result = dict(schema='business-graph-summary-fixed-checkpoint-independent-evaluation-complete-v1',
                  campaign=reference(suite_path), campaign_id=suite['campaign_id'],
                  parent_artifacts_before=parent_before, parent_artifacts_after=parent_after,
                  fingerprints=fingerprints, checkpoint_update=suite['checkpoint_update'], physical_shifts=RUN_COUNT,
                  selection_review=suite['selection'],
                  training_shifts=0, training_updates=0, optimizer_updates=0, gradient_updates=0,
                  manual_anylogic_launches=1, source_refs=refs, reports=reports,
                  evaluation_scope='new_seed_fixed_checkpoint_independent_revalidation')
    require(ledger.resume()['campaigns'][suite['campaign_id']]['commits'] == {},
            'evaluation must not create training commits')
    _same_or_save(output / 'suite_complete.json', result)
    _status(suite, 'COMPLETE', completed_shifts=RUN_COUNT, phase_id=cfg['phase_id'])
    print('COMPLETE fixed_checkpoint independent_evaluation100 optimizer_updates=0', flush=True)
    return result


def run(suite_path, approval_path):
    from .budget import BudgetLedger, GLOBAL_PATH
    suite, cfg, approval = preflight(suite_path, approval_path=approval_path)
    require(approval is not None, 'exact100 new independent-evaluation authorization required')
    budget_path = Path(suite.get('evaluation_budget_path', GLOBAL_PATH))
    ledger = BudgetLedger(budget_path, suite, campaign_sha256=sha(suite_path), approval=approval,
                          approval_sha256=sha(approval_path))
    output = Path(suite['python_output'])
    with _controller_lock(output):
        ledger.resume()
        stop = Path(suite['control_directory']) / 'STOP.properties'
        require(not stop.exists(), 'preserved STOP requires explicit reconciliation; never clear automatically')
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(suite.get('cpu_threads', 2))
        try:
            return _trial(suite_path, suite, cfg, ledger)
        except BaseException as error:
            if not stop.exists():
                stop.parent.mkdir(parents=True, exist_ok=True)
                _publish(stop, ('reason=' + repr(error).replace('\n', ' ') + '\n').encode())
            state = ledger.snapshot()
            complete = sum(row['status'] == 'COMPLETE' for row in state['jobs'].values()
                           if row['campaign_id'] == suite['campaign_id'])
            _status(suite, 'STOPPED', completed_shifts=complete, error=repr(error))
            raise
        finally:
            torch.set_num_threads(previous_threads)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('preflight', 'run'):
        command = sub.add_parser(name)
        command.add_argument('--suite', required=True)
        command.add_argument('--approval', required=name == 'run')
    args = parser.parse_args(argv)
    if args.command == 'preflight':
        suite, _, approval = preflight(args.suite, approval_path=args.approval)
        print(json.dumps(dict(status='PREFLIGHT_PASS_NO_LISTENER', suite=reference(args.suite),
            physical_shifts=RUN_COUNT, training_shifts=0, optimizer_updates=0, checkpoint_update=suite['checkpoint_update'],
            execution_authorized=approval is not None, manual_anylogic_launches=1), ensure_ascii=False))
        return 0
    run(args.suite, args.approval)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
