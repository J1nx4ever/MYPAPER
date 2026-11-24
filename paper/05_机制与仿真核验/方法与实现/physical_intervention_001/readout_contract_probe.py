"""Read-only proof of BG128 current-raw/C certificates and full Actor readout.

An in-memory instantaneous positive crusher-rate intervention is a sensitivity
probe, not an evolved simulator trajectory. No source/old observation is edited.
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[6]
PROJECT = ROOT / 'experiments/lagrangian_ppo_v1'
sys.path.insert(0, str(PROJECT))
from business_graph_summary_v1.models import Actor, digest, pack_summary_prefixes
from business_graph_summary_v1.c_features import FEATURE_NAMES
from relation_gat_ego_v1.adapter import validate_state

SOURCE = PROJECT / 'reports/token_attention_001/business_graph_summary_009/historical_snapshot_001/snapshots.json'
OUT = PROJECT / 'reports/token_attention_001/business_graph_summary_009/physical_intervention_001'
SEED = 2026111801


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    paths = [SOURCE, ROOT / 'python/episode_budget_ppo.py',
             PROJECT / 'anylogic/3d/DRL-PPO-baseline1/DRL-PPO-baseline1.alp']
    hashes = {str(p): sha(p) for p in paths}
    snapshot = json.loads(SOURCE.read_text())
    base = np.asarray(snapshot['T0']['raw'], dtype=np.float32)
    reduced = base.copy()
    assert base[1296] == 1 and base[1304] > 0
    reduced[1304] = base[1304] * np.float32(.5)
    states = {'baseline_T0': base, 'crusher_rate_half': reduced}
    model = Actor(pair_seed=SEED).eval()
    parameter_digest = digest(model)
    evidence = {}
    prefixes = {}
    with torch.no_grad():
        for label, raw in states.items():
            validate_state(raw)
            prefix = pack_summary_prefixes(raw[None], feature_config=model.feature_config)
            prefixes[label] = prefix
            logits, d = model(torch.tensor(raw[None].copy(), dtype=torch.float32), prefix, diagnostic=True)
            assert torch.equal(logits, d['masked_logits'])
            assert tuple(d['candidate_input'].shape) == (1, 4, 273)
            assert tuple(d['candidates'].shape) == (1, 4, 64)
            assert tuple(d['context'].shape) == (1, 64)
            evidence[label] = dict(shapes={name:list(d[name].shape) for name in
                ('current_summary', 'task_summary', 'action_summary', 'context_input', 'context',
                 'candidate_input', 'candidates', 'score_input', 'interaction', 'unmasked_logits', 'masked_logits')},
                feature_certificate=prefix.certificate_sha256,
                feature_config_sha256=prefix.feature_config_sha256,
                C128=prefix.features[0].tolist(), logits=logits[0].tolist(),
                probabilities=torch.softmax(logits[0], -1).tolist(),
                candidate_dump_ids=d['candidate_dump_ids'][0].tolist())
        try:
            model(torch.tensor(reduced[None].copy(), dtype=torch.float32),
                  prefixes['baseline_T0'], diagnostic=True)
        except ValueError as exc:
            rejected = str(exc)
        else:
            raise AssertionError('stale C certificate must be rejected')
    c0, c1 = (prefixes[label].features[0] for label in states)
    changes = [dict(index=i, name=FEATURE_NAMES[i], baseline=float(c0[i]), half_rate=float(c1[i]),
                    delta=float(c1[i]-c0[i])) for i in range(128) if c0[i] != c1[i]]
    assert digest(model) == parameter_digest
    assert all(sha(Path(p)) == value for p, value in hashes.items())
    result = dict(parameter_status='named-paired fixed initialization; zero training', pair_seed=SEED,
        parameter_digest=parameter_digest, parameter_unchanged=True,
        source_hashes_before_and_after=hashes, sources_unchanged=True,
        modified_raw_indices=np.flatnonzero(base != reduced).tolist(),
        rate_baseline_tph=float(base[1304])*2500, rate_intervention_tph=float(reduced[1304])*2500,
        intervention_scope='same-instant exogenous positive crusher-rate step; no evolved future or actual causal effect claimed',
        configuration_context='historical runtime scene remains unchanged; this branch is explicitly outside its original rate setting',
        stale_certificate_rejection=rejected, feature_changes=changes,
        action_C10_blocks_equal=bool(np.array_equal(c0[88:], c1[88:])), outputs=evidence,
        optimizer_steps=0, physical_runs=0, rpc_started=False)
    OUT.mkdir(exist_ok=True, parents=True)
    (OUT / 'readout_contract_review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(output=str(OUT / 'readout_contract_review.json'), modified_indices=result['modified_raw_indices'],
        changed_C128_fields=[x['name'] for x in changes], action_C10_equal=result['action_C10_blocks_equal'],
        stale_certificate_rejection=rejected),ensure_ascii=False))


if __name__ == '__main__':
    main()
