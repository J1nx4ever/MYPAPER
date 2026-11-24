"""Replay sealed s02/u32 evaluation inputs. Does not collect AnyLogic shifts."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np
import torch

from fleet_policy import (FleetLearner, PARENT, PARENT_SHA, bg, models, pack_states, episode_node_order,
                          require, SummaryFeatureCache)


@torch.no_grad()
def verify_migration(*, permuted=False):
    torch.set_num_threads(2)
    candidate = FleetLearner.warm_start(27231)
    saved, _ = bg._load_checkpoint(PARENT)
    actor = models.Actor()
    critic = models.MaterialCostCritic()
    actor.load_state_dict(saved['actor'])
    critic.load_state_dict(saved['critic'])
    for name in ('actor', 'critic'):
        migrated = getattr(candidate, name).state_dict()
        require(set(migrated) == set(saved[name]) | {'status_projection.weight'} and
                all(torch.equal(migrated[k], v) for k, v in saved[name].items()) and
                torch.count_nonzero(migrated['status_projection.weight']) == 0, 'weight migration differs')
    require(not candidate.actor_optimizer.state and not candidate.critic_optimizer.state and
            candidate.update_index == 0, 'fresh optimizers/u0 required')
    with TemporaryDirectory() as tmp:
        path = Path(tmp)/'u0.pt'
        candidate.save(path)
        restored = FleetLearner.load(path)
        require(candidate.policy_version == restored.policy_version and
                models.digest(candidate.critic) == models.digest(restored.critic), 'u0 restore differs')
    evidence = []
    for archive in sorted((PARENT.parents[4]/'evaluation').glob('*/new/material')):
        path = sorted(archive.glob('*.h5'))[0]
        manifest_path = Path(str(path)+'.manifest.json')
        manifest = json.loads(manifest_path.read_bytes())
        seal = json.loads(Path(str(path)+'.seal.json').read_bytes())
        require(seal['hdf5_sha256'] == manifest['hdf5_sha256'] == bg._file_sha(path) and
                seal['manifest_sha256'] == bg._file_sha(manifest_path), 'archive seal mismatch')
        with h5py.File(path) as f:
            raw = f['states'][:-1]
        seed = manifest['identity']['actor_sampling_seed']
        order = episode_node_order(seed, 511) if permuted else None
        errors = np.zeros(3)
        for offset in range(0, len(raw), 16):
            batch = raw[offset:offset+16]
            states = torch.tensor(batch)
            masks = np.full(len(batch), 511)
            packed = torch.from_numpy(pack_states(batch, masks, masks, node_order=order))
            prefix = SummaryFeatureCache(batch, feature_config=actor.feature_config).prefix()
            old_logits, new_logits = actor(states, prefix), candidate.actor(packed, prefix)
            require(torch.equal(old_logits.isfinite(), new_logits.isfinite()), 'legal actions differ')
            valid = old_logits.isfinite()
            logits_error = (old_logits[valid]-new_logits[valid]).abs().max().item()
            probs_error = (old_logits.softmax(-1)-new_logits.softmax(-1)).abs().max().item()
            values_error = (critic(states, prefix)-candidate.critic(packed, prefix)).abs().max().item()
            errors = np.maximum(errors, (logits_error, probs_error, values_error))
        require(errors.max() < 1.e-6, f'u0 equivalence failed {path}: {errors}')
        evidence.append(dict(scenario=archive.parents[1].name, archive=path.name,
                             sha256=seal['hdf5_sha256'], decisions=len(raw),
                             max_abs_legal_logit_error=errors[0], max_abs_probability_error=errors[1],
                             max_abs_value_error=errors[2]))
        if permuted:
            evidence[-1].update(actor_sampling_seed=seed, graph_slot_to_physical_node=list(order))
    require(len(evidence) == 7, 'expected seven s02/u32 scenarios')
    return dict(parent_sha256=PARENT_SHA, absolute_tolerance=1.e-6, optimizer_updates=0,
                torch_version=str(torch.__version__), device='cpu', dtype='float32', samples=evidence)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--permuted', action='store_true', help='use the sealed shift seed to shuffle graph slots')
    print(json.dumps(verify_migration(permuted=parser.parse_args().permuted), indent=2, allow_nan=False))
