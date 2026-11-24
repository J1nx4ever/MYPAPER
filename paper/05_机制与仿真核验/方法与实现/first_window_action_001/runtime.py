"""Run authenticated original-action controls and then one substitution per case."""
import argparse
from dataclasses import asdict
import json
import math

import numpy as np
import torch

from common import *
from budget import Ledger
from baseline1_v1.collection import PhaseCollector
from baseline1_v1.java_plan import COLUMNS
from business_graph_summary_campaign_v1.admission import (
    _check_actor_phase, _check_query_physics, _validated_wait_deltas,
)
from business_graph_summary_campaign_v1.collection import OnlineActor, wait_for_material_journal
from business_graph_summary_v1.admission import frozen_actor
from business_graph_summary_v1.targets import REWARD_VERSION, build_composite_targets
from grade_event_memory_v1.events import build_memory
from grade_event_memory_v1.journal import read_journal
from grade_only_ppo_v2.runtime import _controller_lock
from python.constrained_batch import load_dataset
from single_critic_task_query_v1.evaluation import _physical
from task_query_cmdp_v1.runtime import validate_snapshot


def physical_info(info):
    # The new diagnostic run deliberately has independent RPC identities.
    identity = {'episode_id', 'request_id', 'experiment_run_index'}
    return {key: value for key, value in info.items() if key not in identity}


class BranchActor(OnlineActor):
    def __init__(self, actor, jobs, cfg, histories):
        super().__init__(actor, jobs)
        self.cfg = cfg
        self.histories = histories

    def sample_with_info(self, observation, mask, info):
        self._job(info)
        job = self.jobs[self.current_job]
        candidate = self.cfg['cases'][str(job['historical_run'])]
        rows, episode, _ = self.histories[job['historical_run']]
        index = info['decision_sequence']
        cut = candidate['decision']
        check = job['branch'] == 'original' or index <= cut
        if check:
            require(index < len(rows), 'original trajectory has already ended')
            expected_row = rows[index]
            expected_info = episode['start_info'] if index == 0 else rows[index-1]['settlement_info']
            require(observation.tobytes() == expected_row['observation'].tobytes()
                    and physical_info(info) == physical_info(expected_info),
                    f'physical prefix differs at decision {index}')
        sampled, log_prob = super().sample_with_info(observation, mask, info)
        if check:
            require(sampled == expected_row['action'] and log_prob == expected_row['old_log_prob'] and
                    all(self.last_metadata[key] == expected_row[key]
                        for key in ('actor_rng_before_sha256', 'actor_rng_after_sha256')),
                    f'original policy draw differs at decision {index}')
        forced = job['branch'] == 'alternative' and index == cut
        action = candidate['alternative'] if forced else sampled
        require(mask[action] == 1, 'substitute action is illegal in the actual simulator')
        if forced:
            require(state_sha(observation) == candidate['state_sha256'] and
                    sampled == candidate['factual'] and action != sampled and
                    observation[23+action] > 0, 'action intervention target differs')
        self.last_metadata.update(behavior_kind=BEHAVIOR_KIND, critic_only=True,
            forced_action=forced, sampled_actor_action=sampled,
            sampled_actor_log_prob=log_prob, observed_common_prefix=check)
        return action, 0. if forced else log_prob


def canonical_events(events):
    return [{k:v for k,v in asdict(e).items() if k != 'request_id'} for e in events]


def through_cut(events, cut):
    stop = next(i for i, e in enumerate(events) if e.kind == 'DECISION' and e.decision_sequence == cut)
    return canonical_events(events[:stop+1])


def bind_journal(cfg, job, dataset, rows, episode, history):
    path = wait_for_material_journal(cfg, job)
    journal = read_journal(path)
    require(journal.header['source_kind'] == 'actual_anylogic' and
            journal.header['episode_id'] == episode['episode_id'] and
            journal.header['recording_id'] == cfg['recording_id'] and
            journal.header['run_index'] == job['run_index'] and
            journal.header['seed'] == job['experiment_seed'] and
            journal.header['config_sha256'] == dataset['provenance']['configuration']['sha256'],
            'material journal identity differs from actual RPC')
    events = journal.events
    memory = build_memory(events)
    infos = [episode['start_info']] + [r['settlement_info'] for r in rows]
    markers = [e for e in events if e.kind in ('DECISION','END')]
    states = np.stack([rows[0]['observation']] + [r['next_observation'] for r in rows])
    require(len(markers) == len(infos), 'RPC/material decision count differs')
    for state, marker, info in zip(states, markers, infos):
        validate_snapshot(state, info['reward_components'], terminal=marker.kind == 'END')
        require((marker.request_id, marker.decision_sequence, marker.time) ==
                (info['request_id'], info['decision_sequence'], info['sim_time_seconds']),
                'RPC/material marker identity differs')
    _check_query_physics(events, states)
    for settle, window in zip(memory.settlements, episode['windows']):
        require(settle.window == window['window_index'] and
                abs(settle.tons-window['ore_tons']) <= 1e-5 and
                abs(settle.metal-window['metal_tons']) <= 1e-5, 'material/RPC window Q/M differs')
    cut = cfg['cases'][str(job['historical_run'])]['decision']
    require(through_cut(events, cut) == through_cut(history[2].events, cut),
            'common journal prefix differs')
    forced = [i for i, row in enumerate(rows) if row['forced_action']]
    require(forced == ([cut] if job['branch'] == 'alternative' else []) and
            all(r['critic_only'] and r['behavior_kind'] == BEHAVIOR_KIND for r in rows),
            'unexpected intervention or training data marker')
    if job['branch'] == 'original':
        require(canonical_events(events) == canonical_events(history[2].events) and
                len(rows) == len(history[0]) and
                states[-1].tobytes() == history[0][-1]['next_observation'].tobytes() and
                physical_info(infos[-1]) == physical_info(history[0][-1]['settlement_info']),
                'original action control does not reproduce the complete historical shift')
    terminal = infos[-1]['reward_components']
    evaluation = dict(ore_tons=terminal['ore_tons_shift'], waste_tons=terminal['waste_tons_shift'],
                      total_wait_seconds=terminal['wait_vehicle_seconds'])
    wait_deltas = _validated_wait_deltas(infos)
    targets = build_composite_targets(memory, wait_deltas=wait_deltas, **evaluation)
    physical = _physical(dict(memory=memory, targets=targets, evaluation=evaluation,
                              states=states, actions=[r["action"] for r in rows], identity=dict(reward_version=REWARD_VERSION),
                              wait_deltas=wait_deltas, composite_contract=cfg['composite_contract']))
    marker = markers[cut]
    truck = rows[cut]['truck_id']
    next_dispatch = next((d.seq for d in markers[cut+1:-1]
                          if rows[d.decision_sequence]['truck_id'] == truck), markers[-1].seq)
    batches = [e for e in events if e.kind == 'UNLOAD' and e.truck_id == truck
               and marker.seq < e.seq < next_dispatch]
    action = rows[cut]['action']
    require(len(batches) <= 1 and all(b.source_id == action for b in batches),
            'action truck unload association differs')
    batch_ids = {e.batch_id for e in batches}
    processed = [e for e in events if e.kind == 'PROCESS' and e.batch_id in batch_ids and e.window == 0]
    from select_actions import summary
    return dict(run_index=job['run_index'], branch=job['branch'], historical_run=job['historical_run'],
        decision=cut, truck=truck, action=action, policy_suggestion=rows[cut]['sampled_actor_action'],
        historical_complete_match=job['branch']=='original', common_prefix_match=True,
        physical=physical, source_windows=summary(journal),
        chosen_trip=dict(unload=[asdict(e) for e in batches],
            first_window_processed_tons=math.fsum(e.tons for e in processed),
            first_window_processed_metal_tons=math.fsum(e.metal for e in processed),
            first_process_seconds=min((e.interval_start for e in processed), default=None)),
        material_journal=reference(path))


def verify(suite_path):
    require(suite_path == PREPARED/'suite.json', 'unexpected experiment suite')
    suite = read(suite_path)
    for ref in suite['pinned'].values():
        checked(ref)
    cfg_path = checked(suite['config'])
    cfg = read(cfg_path)
    require(cfg['campaign_id'] == CID and cfg['recording_id'] == RECORDING and
            tuple(j['run_index'] for j in cfg['jobs']) == RUN_IDS and
            cfg['checkpoint']['sha256'] == CHECKPOINT_SHA and sha(CORE) == CORE_SHA and
            cfg['optimizer_steps'] == 0 and cfg['capture_physical_events'] is False,
            'counterfactual scope changed')
    require(cfg['cases'] == {str(r['historical_run']): r for r in read(checked(suite['candidate_selection']))['candidates']},
            'selected opportunities changed')
    require([(j['historical_run'], j['branch']) for j in cfg['jobs']] ==
            [(run, branch) for run in CASES for branch in ('original','alternative')],
            'experiment pair order differs')
    return suite, cfg


def run(suite_path):
    torch.set_num_threads(2)
    suite, cfg = verify(suite_path)
    with _controller_lock(RUN), torch.no_grad():
        actor, digest = frozen_actor(checked(cfg['actor']))
        _check_actor_phase(actor, cfg)
        histories = {}
        try:
            for case in cfg['cases'].values():
                source = case['historical_failed']
                rows, episodes, *_ = load_dataset(checked(source['raw']).parent, require_real=True)
                histories[source['run_index']] = (rows, episodes[0], read_journal(checked(source['journal'])))
            behavior = BranchActor(actor, cfg['jobs'], cfg, histories)
            ledger = Ledger(cfg, suite['config'])
            control = RUN/'control'
            control.mkdir(exist_ok=True)

            def ready(job, config_path):
                require(not (control/'STOP.properties').exists() and reference(config_path) == suite['config'],
                        'preserved STOP or wrong config')
                values = {key: str(job[key]) for key in COLUMNS}
                values.update(protocol='baseline1-one-click-ready-v1', campaign_id=CID,
                    master_plan_sha256=suite['master_plan']['sha256'],
                    instrumentation_revision='boundary-events-v1-native-rpc-runtime-scene-v1',
                    config_sha256=suite['config']['sha256'], snapshot_sha256=suite['config']['sha256'],
                    snapshot_path=str(config_path), all_comparison_checkpoints_frozen='true')
                path = control/'ready'/f"run_{job['run_index']}.properties"
                path.parent.mkdir(exist_ok=True)
                _publish(path, ''.join(f'{k}={v}\n' for k,v in values.items()).encode('ascii'))

            startup = dict(control_directory=str(control), campaign_id=CID,
                           master_plan_sha256=suite['master_plan']['sha256'])
            results = []
            for job in cfg['jobs']:
                # The existing bridge rejects repeated environment seeds per
                # listener. Each authorized slot gets its own exact manifest
                # and that listener closes before the next ready publication.
                active_phase = dict(cfg, jobs=[job])
                with PhaseCollector(active_phase, checked(suite['config']), publish_ready=ready, startup=startup) as collector:
                    require(not Path(job['material_event_bundle']).exists(), 'material identity already used')
                    MATERIALS.mkdir(parents=True, exist_ok=True)
                    ledger.set(job, 'PLANNED', 'CLAIMED')
                    raw = None
                    try:
                        raw, dataset_ref, _ = collector.collect(job, behavior)
                        rows, episodes, _, _, manifest = load_dataset(Path(dataset_ref["path"]).parent, require_real=True)
                        try:
                            result = bind_journal(cfg, job, manifest, rows, episodes[0], histories[job['historical_run']])
                        finally:
                            rows.close()
                        result['rpc_dataset'] = dataset_ref
                        result['actor_digest'] = digest
                        evidence = save(RUN/f"run_{job['run_index']}_diagnostic.json", result)
                        ledger.set(job,'CLAIMED','COMPLETE',receipt=evidence)
                    except BaseException as error:
                        ledger.set(job, 'CLAIMED', 'UNKNOWN', error=repr(error))
                        raise
                    finally:
                        if raw is not None and hasattr(raw.rows, 'close'):
                            raw.rows.close()
                    results.append(result)
                    print(f"COMPLETE {job['run_index']} {job['branch']}", flush=True)
            behavior.assert_frozen()
            checked(cfg['actor'])
            checked(cfg['checkpoint'])
            require(sha(CORE) == CORE_SHA, 'protected PPO bytes changed')
            from evaluate import compare
            save(REPORT/'experiment_result.json', compare(cfg, results, reference(suite_path)))
            save(RUN/'suite_complete.json', dict(physical_shifts=4, optimizer_updates=0,
                evidence=reference(REPORT/'experiment_result.json')))
        except BaseException as error:
            control = RUN/'control'
            control.mkdir(exist_ok=True)
            if not (control/'STOP.properties').exists():
                _publish(control/'STOP.properties', b'status=STOP\n')
            save(RUN/'error.json', dict(error=repr(error)))
            raise
        finally:
            for rows, _, _ in histories.values():
                rows.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', type=lambda p: Path(p).resolve(), required=True)
    args = parser.parse_args()
    run(args.suite)
