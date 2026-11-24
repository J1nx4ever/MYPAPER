"""Run offline engineering checks only; never start RPC or reserve a shift.

Real rows are historical, read-only inputs to freshly initialized models. They
are never used for optimization or reported as independent performance results.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parents[1]
# The project owns the current `python` schema package; the repository root also
# contains a historical package with that name. Resolve the current one first.
sys.path.insert(0, str(PROJECT))
PYTHON = Path('/home/jinx/miniconda3/envs/anylogic-env/bin/python')
CORE = WORKSPACE / 'python/episode_budget_ppo.py'
CORE_SHA = '8e4cfa31c622f5e33756e304eea3f7526547db72ae93f7d9f00808845d61ef96'
OLD_SUITE = PROJECT / 'change_plans/token_attention_001/single_critic_task_query_008/engineering_001/formal_prepared_002/suite.json'
DEFAULT_ARM = PROJECT / 'runs/single-critic-task-query-formal-seed1-20261117-002/arms/s01_TQ128'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')


def verify_old_sources():
    suite = json.loads(OLD_SUITE.read_bytes())
    pins = suite['source_pins']
    references = ([dict(path=p,sha256=s) for p,s in pins.items()]
                  if isinstance(pins,dict) else pins)
    count, bad = 0, []
    for ref in references:
        count += 1
        if sha(ref['path']) != ref['sha256']:
            bad.append(ref['path'])
    if bad:
        raise ValueError('historical pinned sources changed: ' + repr(bad))
    return dict(count=count, mismatches=bad)


def real_forward_audit(arm):
    import h5py
    import numpy as np
    import torch
    from .graph import build_graph, NODE_NAMES, GROUP_NAMES, RELATION_NAMES
    from .models import Actor, MaterialCostCritic, digest
    from .features_cache import pack_summary_prefixes
    from .contract import feature_config_for
    paths = sorted((arm / 'final_u0016/episodes').glob('run_*/samples.h5'))
    if len(paths) != 100:
        raise ValueError('expected all 100 historical independent episodes')
    rows, sources = [], []
    for path in paths:
        before = sha(path)
        with h5py.File(path) as handle:
            x = handle['states'][:]
        decisions = np.flatnonzero(x[:,17] == 0)
        ids = decisions[np.linspace(0, len(decisions)-1, 8, dtype=int)]
        rows.extend(x[ids])
        sources.append(dict(path=str(path), sha256=before, row_indices=ids.tolist()))
    states = np.asarray(rows, dtype=np.float32)
    torch.set_num_threads(2)
    actor, critic = Actor(pair_seed=2026111801), MaterialCostCritic(pair_seed=2026111801)
    actor.eval(); critic.eval()
    before = dict(actor=digest(actor), critic=digest(critic))
    counts = np.zeros(len(RELATION_NAMES), dtype=np.int64)
    values, probabilities = [], []
    checks = dict(finite_outputs=True, exact_mask=True, nineteen_tokens=True,
                  quantity_channels=True, graph_batch_isolation=True,
                  within_group_normalization=True, absent_group_zero=True,
                  no_final_query=True, unchanged_parameters=True)
    start = time.monotonic()
    with torch.no_grad():
        for offset in range(0,len(states),16):
            raw = states[offset:offset+16]
            x = torch.from_numpy(raw.copy())
            prefix = pack_summary_prefixes(raw, feature_config=feature_config_for())
            logits, ad = actor(x,prefix,diagnostic=True)
            value, cd = critic(x,prefix,diagnostic=True)
            p = logits.softmax(-1)
            checks['finite_outputs'] &= bool(torch.isfinite(p).all() and torch.isfinite(value).all())
            checks['exact_mask'] &= bool((p[~x[:,19:23].bool()] == 0).all())
            values.extend(value.tolist()); probabilities.extend(p.tolist())
            graph = ad['graph']
            counts += torch.bincount(graph.edge_type,minlength=len(RELATION_NAMES)).numpy()
            checks['graph_batch_isolation'] &= bool((graph.edge_index[0]//19 == graph.edge_index[1]//19).all())
            for details in (ad,cd):
                checks['nineteen_tokens'] &= details['z2'].shape == (len(raw),19,64)
                checks['quantity_channels'] &= details['quantity_pools'].shape[:2] == (len(raw),4)
                checks['no_final_query'] &= not any(k in details for k in ('queries','query_input','attention'))
                for number in (1,2):
                    layer = details['layer'+str(number)]
                    sums = layer['weights'].new_zeros((len(raw)*19*len(GROUP_NAMES),4))
                    sums.index_add_(0,layer['buckets'],layer['weights'])
                    presence = layer['group_masks'].reshape(-1)
                    checks['within_group_normalization'] &= bool(torch.allclose(
                        sums[presence],torch.ones_like(sums[presence]),atol=2e-6,rtol=0))
                    checks['absent_group_zero'] &= bool((layer['group_messages'][~layer['group_masks']] == 0).all())
            print(f'Historical forward: {offset+len(raw)}/{len(states)}',flush=True)
    after = dict(actor=digest(actor),critic=digest(critic))
    checks['unchanged_parameters'] = before == after
    for ref in sources:
        if sha(ref['path']) != ref['sha256']:
            raise ValueError('historical observation bytes changed')
    if not all(checks.values()):
        raise ValueError('real forward checks failed: ' + repr(checks))
    return dict(scope='engineering forward on historical states; freshly initialized BG009; no performance claim',
        rows=len(states), episodes=len(paths), states_per_episode=8,
        checks=checks, model_digests=before, source_refs=sources,
        relation_record_counts=dict(zip(RELATION_NAMES,map(int,counts))),
        actor_parameters=sum(p.numel() for p in actor.parameters()),
        critic_parameters=sum(p.numel() for p in critic.parameters()),
        actor_encoder_parameters=sum(p.numel() for p in actor.state_encoder.parameters()),
        critic_encoder_parameters=sum(p.numel() for p in critic.state_encoder.parameters()),
        value_range=[min(values),max(values)], forward_seconds=time.monotonic()-start,
        new_physical_shifts=0, real_data_optimizer_updates=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True,
                        help='new evidence directory; must not already exist')
    parser.add_argument('--source-arm',type=Path,default=DEFAULT_ARM)
    args=parser.parse_args()
    if Path(sys.executable).resolve() != PYTHON.resolve():
        raise RuntimeError('use the anylogic-env Python')
    if sha(CORE) != CORE_SHA:
        raise RuntimeError('protected PPO core SHA differs')
    pins_before=verify_old_sources()
    args.output.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ)
    env['PYTHONPATH']=str(PROJECT)+os.pathsep+str(WORKSPACE)
    env['PYTHONDONTWRITEBYTECODE']='1'
    command=[str(PYTHON),'-B','-m','pytest','-q','-p','no:cacheprovider',
             str(PROJECT/'business_graph_summary_v1')]
    result=subprocess.run(command,env=env,cwd=PROJECT,text=True,capture_output=True)
    (args.output/'pytest.txt').write_text(result.stdout+result.stderr)
    print(result.stdout,end=''); print(result.stderr,end='',file=sys.stderr)
    write_json(args.output/'test_command.json',dict(command=command,exit_code=result.returncode))
    if result.returncode:
        raise SystemExit(result.returncode)
    from .contract import ablation_contract,feature_config_for
    from .c_features import feature_contract
    from .learner import default_config,implementation_signature
    from .graph import EDGE_FEATURE_NAMES,QUANTITY_NAMES
    audit=real_forward_audit(args.source_arm)
    write_json(args.output/'real_forward.json',audit)
    write_json(args.output/'architecture.json',dict(graph=ablation_contract(),
        learner_config=default_config(),feature_contract=feature_contract(feature_config_for()),
        implementation=implementation_signature()))
    after=sha(CORE)
    if after!=CORE_SHA:
        raise RuntimeError('protected PPO changed')
    write_json(args.output/'validation.json',dict(date='2026-11-18 Asia/Shanghai',
        tests_passed=True,real_forward_checks=audit['checks'],
        protected_ppo_sha256=dict(before=CORE_SHA,after=after),
        historical_pins_before=pins_before,historical_pins_after=verify_old_sources(),
        new_physical_shifts=0,real_data_optimizer_updates=0,
        synthetic_updates='only ephemeral unit-test models; see pytest.txt',
        launch_prepared=False,alp_modified=False))
    print('Offline acceptance complete: '+str(args.output),flush=True)


if __name__ == '__main__':
    main()
