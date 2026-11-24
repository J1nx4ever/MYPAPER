"""Regression using archived one-decision logs from the same parent Actor."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from replay_policy import FleetLearner, OriginalLearner
from fleet_policy import PARENT, bg, pack_states, episode_node_order
from test_fleet_policy import episode as synthetic_episode


def archive_episode(path, model):
    pin = json.loads(Path(str(path)+'.seal.json').read_bytes())
    assert pin['hdf5_sha256'] == bg._file_sha(path)
    with h5py.File(path) as f:
        raw = f['states'][:]
        identity = json.loads(f.attrs['identity_json'])
        events = f['material_events']
        kinds = events['kind'][:]
        decision_positions = np.flatnonzero(kinds == b'DECISION')
        closures = [dict(window=int(events['window'][i]), step=int(np.searchsorted(decision_positions,i)-1),
                         tons=float(events['window_q'][i]), metal_tons=float(events['window_m'][i]))
                    for i in np.flatnonzero(kinds == b'WINDOW_CLOSE')]
        return dict(states=raw, dispatchable_masks=np.full(len(raw),511), in_system_masks=np.full(len(raw),511),
                    identity=identity, policy_version=model.policy_version, rpc_config={
                        'config.target_ore_tons_shift':16000., 'config.target_waste_tons_shift':4000.,
                        'config.mass_balance_epsilon':1.e-5}, evaluation=json.loads(f.attrs['evaluation_json']),
                    actions=f['transitions/action'][:], old_log_probs=f['transitions/old_log_prob'][:],
                    wait_deltas=f['transitions/wait_delta_seconds'][:], window_closures=closures)


def test_sealed_single_decision_logs_replay_across_seven_real_conditions():
    torch.set_num_threads(2)
    model = FleetLearner.warm_start(19)
    conditions = sorted((PARENT.parents[4]/'evaluation').glob('*/new/material'))
    count = batch_rejections = 0
    for condition in conditions:
        path = sorted(condition.glob('*.h5'))[0]
        ep = archive_episode(path, model)
        # Uses only archived actions and original probabilities, no reconstructed labels.
        model.verify_collected(ep)
        order = episode_node_order(ep['identity']['actor_sampling_seed'],511)
        packed = pack_states(ep['states'],ep['dispatchable_masks'],ep['in_system_masks'],node_order=order)
        with torch.no_grad():
            batched, _ = OriginalLearner._policy_rows(model,dict(ep,states=packed))
        batch_rejections += np.count_nonzero(~np.isclose(batched.numpy(),ep['old_log_probs'],atol=1.e-6,rtol=1.e-6))
        count += len(ep['actions'])
    assert len(conditions)==7 and count==1296 and batch_rejections>0


def test_rejects_tampered_likelihood_mask_and_policy_lineage():
    torch.set_num_threads(2)
    model = FleetLearner(19)
    ep = synthetic_episode(model,1)
    logs = ep['old_log_probs'].copy()
    logs[0] += 1.e-3
    with pytest.raises(ValueError,match='run 1, decision 0'):
        model.verify_collected(dict(ep,old_log_probs=logs))
    raw = ep['states'].copy()
    raw[0,19+ep['actions'][0]] = 0.
    with pytest.raises(ValueError,match='inconsistent sampled action'):
        model.verify_collected(dict(ep,states=raw))
    with pytest.raises(ValueError,match='stale on-policy episode'):
        model.verify_collected(dict(ep,policy_version='other'))


def test_rpc_worker_sampling_replays_in_training_thread():
    torch.set_num_threads(2)
    model = FleetLearner.warm_start(19)
    path = sorted((PARENT.parents[4]/'evaluation/nominal/new/material').glob('*.h5'))[0]
    ep = archive_episode(path,model)
    generator = torch.Generator().manual_seed(ep['identity']['actor_sampling_seed'])
    with ThreadPoolExecutor(max_workers=3) as rpc_workers:
        sampled = [rpc_workers.submit(model.act,row,511,511,generator).result() for row in ep['states'][:-1]]
    actions, logs, _ = zip(*sampled)
    np.testing.assert_array_equal(actions,ep['actions'])
    model.verify_collected(dict(ep,actions=np.asarray(actions),old_log_probs=np.asarray(logs)))


def test_full_update_is_batched_and_restorable(tmp_path):
    torch.set_num_threads(2)
    model = FleetLearner(19)
    group = [synthetic_episode(model,i) for i in range(32)]
    old = model.policy_version
    assert '_verify_dispatch_likelihood' not in model._prepare_episode(group[0])
    receipt = model.update(group,batch_id='offline-repaired')
    assert receipt['update_index']==1 and receipt['optimizer_steps']>0 and model.policy_version!=old
    checkpoint=tmp_path/'checkpoint.pt'
    model.save(checkpoint)
    restored=FleetLearner.load(checkpoint)
    assert restored.policy_version==model.policy_version and restored.update_index==1
    assert isinstance(restored.fork(42),FleetLearner)
