"""Exercise both controls and both substitution gates on sealed inputs."""
import copy
import json
import tempfile

import numpy as np
import torch

from common import *
from budget import budget_snapshot, check_new_slots, LEDGER
from runtime import BranchActor, bind_journal, verify, BEHAVIOR_KIND
from business_graph_summary_campaign_v1.admission import _check_actor_phase
from business_graph_summary_v1.admission import frozen_actor
from grade_event_memory_v1.journal import read_journal
from python.constrained_batch import EpisodeBatchBuffer, collect_batch, load_dataset
from python.collect_offline import validate_runtime


def verify_same_seed_on_fresh_bridges(jobs, reward):
    from python.dispatch_bridge import DispatchCallbackBridge
    from python.dispatch_env import DispatchEnv
    from single_critic_task_query_v1.test_scenario_bridge import (
        InProcessDispatch, FixedBehavior, dispatch_request, rpc_info,
    )
    for job in jobs:
        run, seed = job['run_index'], job['experiment_seed']
        requests = []
        for index, time in enumerate((0., 20., 28800.)):
            info = rpc_info(time, index, terminal=index == 2)
            info.update(experiment_run_index=run, experiment_seed=seed,
                        episode_id=f'probe-{run}', request_id=f'probe-{run}-{index}')
            requests.append(dispatch_request(info, terminal=index == 2))
        bridge = DispatchCallbackBridge(expected_episode_metadata=[(run, seed)])
        producer = InProcessDispatch(bridge, requests)
        env = DispatchEnv(bridge, callback_timeout=2.)
        buffer = EpisodeBatchBuffer(expected_episodes=1, source_kind='synthetic_unit_test')
        producer.start()
        try:
            collect_batch(env, FixedBehavior(), buffer, [job], reward_settings=reward)
        finally:
            env.close()
            producer.join()
        require(not producer.errors and len(producer.responses) == 3 and buffer.complete,
                'a fresh RPC bridge did not accept its precise paired identity')


def main():
    torch.set_num_threads(2)
    suite, cfg = verify(PREPARED/'suite.json')
    before = budget_snapshot()
    check_new_slots(cfg, before)
    verify_same_seed_on_fresh_bridges(cfg['jobs'], cfg['reward'])
    actor, actor_digest = frozen_actor(checked(cfg['actor']))
    _check_actor_phase(actor, cfg)
    expected_action_checks = 0
    prefix_rejection_checks = 0
    control_results = []
    for run, candidate in cfg['cases'].items():
        run = int(run)
        failed = candidate['historical_failed']
        rows, episodes, _, _, manifest = load_dataset(checked(failed['raw']).parent, require_real=True)
        try:
            episode = episodes[0]
            journal = read_journal(checked(failed['journal']))
            histories = {run: (rows, episode, journal)}
            for job in [j for j in cfg['jobs'] if j['historical_run']==run]:
                behavior = BranchActor(actor, [job], cfg, histories)
                original = job['branch']=='original'
                info = episode['start_info']
                cut = candidate['decision']
                limit = len(rows) if original else cut+2
                proof_rows = []
                for i in range(limit):
                    row = rows[i]
                    remapped = dict(info, experiment_run_index=job['run_index'],
                        episode_id='offline-probe-'+str(job['run_index']), request_id='offline-probe-'+str(i))
                    validate_runtime(remapped, job)
                    if i == 0:
                        bad = row['observation'].copy()
                        bad[4] = np.float32(.01)
                        try:
                            behavior.sample_with_info(bad, row['mask'], remapped)
                        except ValueError as error:
                            require('physical prefix differs' in str(error), 'incorrect prefix rejection')
                            prefix_rejection_checks += 1
                        else:
                            raise ValueError('mismatched physical prefix was accepted')
                    a, lp = behavior.sample_with_info(row['observation'], row['mask'], remapped)
                    metadata = behavior.row_metadata(remapped)
                    forced = not original and i==cut
                    require(metadata['forced_action']==forced and metadata['critic_only'] and
                            metadata['behavior_kind']==BEHAVIOR_KIND, 'diagnostic metadata differs')
                    if forced:
                        require(a==candidate['alternative'] and lp==0. and metadata['sampled_actor_action']==candidate['factual'],
                                'forced action contract differs')
                    elif original or i<cut:
                        require(a==row['action'] and lp==row['old_log_prob'], 'control action differs')
                    require(metadata['actor_rng_after_sha256']==row['actor_rng_after_sha256'], 'random stream draw consumption differs')
                    proof_rows.append(dict(row, **metadata))
                    expected_action_checks += 1
                    info = row['settlement_info']
                if original:
                    # Recheck the material acceptance path using the old sealed bytes.
                    # Only the derived row metadata are added in memory.
                    bundle = Path(failed['journal']['path']).parent
                    historical_job = dict(job, run_index=run, material_event_bundle=str(bundle),
                        material_event_directory=str(bundle.parent), material_journal=failed['journal']['path'])
                    historical_cfg = dict(cfg, recording_id=journal.header['recording_id'])
                    with tempfile.TemporaryDirectory(prefix='bg128-action-storage-') as temp:
                        tmp = Path(temp)/'dataset'
                        buffer = EpisodeBatchBuffer(tmp, expected_episodes=1, source_kind='real_anylogic_rpc',
                            storage_format=cfg['storage']['format'], provenance=manifest['provenance'])
                        buffer.rows = proof_rows
                        buffer.episodes = [copy.deepcopy(episode)]
                        buffer.seal()
                        stored, stored_episodes, _, _, stored_manifest = load_dataset(tmp, require_real=True)
                        try:
                            result = bind_journal(historical_cfg, historical_job, stored_manifest, stored, stored_episodes[0], histories[run])
                            require(result['historical_complete_match'] and not result['physical']['windows'][0]['compliant'],
                                    'historical control evidence differs')
                            control_results.append(dict(run=run, window0_grade=result['physical']['windows'][0]['grade'],
                                                        raw_rows=len(stored), trip=result['chosen_trip']))
                        finally:
                            stored.close()
        finally:
            rows.close()
    require(sha(CORE)==CORE_SHA and not LEDGER.exists() and not RUN.exists() and not MATERIALS.exists(),
            'offline test wrote runtime artifacts or changed PPO core')
    path = save(REPORT/'preparation_validation_002.json', dict(status='PASS_OFFLINE', suite=reference(PREPARED/'suite.json'),
        frozen_actor_digest=actor_digest, control_results=control_results, action_checks=expected_action_checks,
        in_process_rpc_identity_checks=len(cfg['jobs']),
        invalid_prefix_rejections=prefix_rejection_checks, budget_before=before['maximum_total_shifts'],
        physical_shifts=0, optimizer_updates=0, protected_ppo=reference(CORE)))
    print(json.dumps(path))


if __name__ == '__main__':
    main()
