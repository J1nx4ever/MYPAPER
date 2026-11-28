"""Bind one B fine-tune campaign to independently identified native shifts."""

import argparse
from contextlib import contextmanager
import csv
from functools import lru_cache
from hashlib import sha256
import io
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


HERE = Path(__file__).resolve().parent
STATS = HERE.parent / 'paired_50_001'
sys.path.insert(0, str(STATS))
import prepare_stats as previous
from prepare_stats import PROJECT, canonical, require, witness, write_fresh


CAMPAIGN = 'site-layout-b-finetune-20261127-001'
CAMPAIGN_OUTPUT = PROJECT / 'runs' / CAMPAIGN
MODEL_OUTPUT = CAMPAIGN_OUTPUT / 'models'
OVERVIEW = HERE / 'launch_identity.json'
REPORTS = PROJECT / 'reports' / HERE.relative_to(PROJECT / 'change_plans')
CASES = tuple(f'{prefix}_{n:03d}' for prefix, count in [('T', 128), ('V', 80), ('E', 50)]
              for n in range(1, count + 1))
EXPERIMENTS = {**{f'Topology_B_Train{i+1}':(32, 1001 + i * 32) for i in range(4)},
               'Topology_B_Validate80':(80, 2001), 'Topology_B_Test50':(50, 3001)}
PROTOCOL = dict(parent='dynamic-u9', local_updates=4, training_shifts_per_update=32,
                training_action_mode='sample', evaluation_action_mode='argmax',
                actor_learning_rate=1e-4, critic_learning_rate=3e-4,
                validation_slots=16, validation_candidates=list(range(5)),
                selection_order=['maximize_joint_success_count', 'maximize_compliant_window_count',
                                 'minimize_total_outside_band_percentage_points', 'minimize_local_update'],
                final_slots=50, final_seed_source=previous.CAMPAIGN+'-B',
                final_test_design='matched rerun on the previously evaluated B50 environment seeds')
CORE_SHA = '8e4cfa31c622f5e33756e304eea3f7526547db72ae93f7d9f00808845d61ef96'
ledger_entry = witness.ledger_entry
budget_snapshot = witness.budget_snapshot


@lru_cache(maxsize=50)
def baseline_slot(slot):
    """Verify the exact sealed historical physical run before allowing its seed."""
    label = f'B_{slot:03d}'
    cfg, suite, job = previous.load_case(label)
    status, receipt = ledger_entry(suite, job)
    out = Path(job['python_output'])
    require(status == 'COMPLETE' and receipt == witness.reference(out/'receipt.json'),
            f'{label} paired reference has no sealed physical result')
    payload = json.loads((out/'receipt.json').read_bytes())
    manifest_path, manifest, digest = witness.native_manifest(Path(job['event_directory']), job)
    audit_path = out/'policy_verification.json'
    audit = json.loads(audit_path.read_bytes())
    require(payload['physical_status'] == 'COMPLETE' and payload['manifest'] == dict(path=str(manifest_path), sha256=digest)
            and manifest['complete'] and manifest['terminated'] and not manifest['truncated']
            and payload['config_sha256'] == job['config_sha256'] and payload['material_balance_verified'] is True
            and manifest['config']['actual_runtime_scene'] == cfg['expected_runtime_scene']
            and audit['status'] == 'VERIFIED' and audit['checkpoint_sha256'] == previous.original.CHECKPOINT_SHA256
            and audit['decision_count'] == manifest['native_policy_decisions'],
            f'{label} baseline runtime or controller evidence differs')
    # The original paired-50 source already pinned the policy verification and event seals.
    return dict(run_index=job['run_index'], recording_id=job['recording_id'], environment_seed=job['experiment_seed'],
                config=witness.reference(suite['suite_path']), receipt=receipt, manifest=payload['manifest'],
                policy_verification=witness.reference(audit_path))


def fresh_seed(domain, slot):
    raw = sha256(f'{CAMPAIGN}|{domain}|{slot:03d}'.encode()).digest()
    return 1 + int.from_bytes(raw[:8], 'big') % witness.ledger.JAVA_MASK


def identity(label):
    require(label in CASES, 'unknown fine-tune shift')
    prefix = label[0]
    i = int(label[2:])
    if prefix == 'T':
        run, phase, slot, code = 10055000+i, 'train', i, 1000+i
        update = (i-1)//32
        experiment = f'Topology_B_Train{update+1}'
        env, actor = fresh_seed('train-env', i), fresh_seed('train-actor', i)
    elif prefix == 'V':
        run, phase, slot, code = 10055128+i, 'validation', (i-1)%16+1, 2000+i
        update, experiment = (i-1)//16, 'Topology_B_Validate80'
        env, actor = fresh_seed('validation-env', slot), None
    else:
        run, phase, slot, code = 10055208+i, 'test', i, 3000+i
        update, experiment = -1, 'Topology_B_Test50'
        env, actor = baseline_slot(i)['environment_seed'], None
    output = PROJECT/'runs'/f'{CAMPAIGN}-{label}'
    return dict(case=label, group='B', phase=phase, stage_index=i, slot=slot, policy_update=update,
                native_case_code=code, experiment=experiment, run_index=run, environment_seed=env,
                actor_sampling_seed=actor, slot_permutation_seed=actor or 0, recording_id=f'{CAMPAIGN}-{run}',
                action_mode='sample' if prefix == 'T' else 'argmax',
                event_directory=str(output/'anylogic_events'), python_output_directory=str(output/'python'),
                activation_path=str(output/'activation.json'))


@lru_cache(maxsize=1)
def source_metadata():
    cfg, _, _ = previous.load_case('B_001')
    parent = previous.original.checkpoint_metadata()
    core = PROJECT.parents[1]/'python/episode_budget_ppo.py'
    require(witness.reference(core)['sha256'] == CORE_SHA, 'protected PPO core differs')
    replay = STATS.parents[2]/'fleet_outage_resilience_001/replay_repair_001/replay_policy.py'
    files = [HERE/'train_service.py', HERE/'eval_service.py', HERE/'topology_learner.py', HERE/'campaign.py', replay,
             previous.SCALED/'scaled_service.py', previous.SCALED/'opening_targets.py', core]
    sources = [*parent['inference_sources'], *(witness.reference(p) for p in files)]
    return cfg, dict(parent_checkpoint_sha256=parent['checkpoint_sha256'], checkpoint_path=parent['checkpoint_path'],
                     parent_actor_digest=parent['actor_digest'], parent_checkpoint_manifest=parent['checkpoint_manifest'],
                     parent_checkpoint_seal=parent['checkpoint_seal'], inference_sources=sources,
                     inference_host='127.0.0.1', inference_port=50871)


def make_config(label, model):
    base, sources = source_metadata()
    row = identity(label)
    scene = dict(base['expected_runtime_scene'], model_sha256=model['sha256'])
    return dict(purpose='topology_b_finetune', policy_id='bg128-topology-b-finetune', **row, **sources,
                protocol=PROTOCOL, model_sha256=model['sha256'], model_path=model['path'],
                layout_id='loop_B', layout_path=base['layout_path'], layout_sha256=base['layout_sha256'],
                task=base['task'], task_sha256=base['task_sha256'], expected_runtime_scene=scene,
                expected_runtime_scene_sha256=sha256(canonical(scene)).hexdigest(),
                paired_baseline=baseline_slot(row['slot']) if row['phase']=='test' else None)


def suite_job(config, digest):
    suite, job = previous.suite_job(config, digest)
    suite['suite_path'] = HERE/'configs'/f"{config['case']}.json"
    job['actor_sampling_seed'] = config['actor_sampling_seed'] or 0
    return suite, job


@lru_cache(maxsize=1)
def frozen_index():
    raw = OVERVIEW.read_bytes()
    require(OVERVIEW.with_suffix('.sha256').read_text().strip() == sha256(raw).hexdigest(), 'identity index SHA differs')
    meta = json.loads(raw)
    rows = meta['cases']
    require(meta['campaign_id'] == CAMPAIGN and meta['protocol'] == PROTOCOL
            and tuple(row['case'] for row in rows) == CASES, 'index or protocol differs')
    return {row['case']:row for row in rows}


def load_case(label):
    row = frozen_index()[label]
    ident = identity(label)
    require(all(row[k] == v for k,v in ident.items()), 'case identity differs')
    path = HERE/'configs'/f'{label}.json'
    raw = path.read_bytes()
    digest = sha256(raw).hexdigest()
    require(str(path)==row['config_path'] and digest==row['config_sha256']==path.with_suffix('.sha256').read_text().strip(),
            'case configuration byte SHA differs')
    cfg = json.loads(raw)
    archive = previous.model_pin(REPORTS/Path(cfg['model_path']).name)
    model = dict(path=cfg['model_path'], sha256=archive['sha256'])
    require(cfg == make_config(label,model), 'sealed protocol, parent, layout, or learner source changed')
    suite, job = suite_job(cfg,digest)
    return cfg, suite, job


def read_case(label):
    cfg, _, job = load_case(label)
    require(previous.model_pin(Path(cfg['model_path']))['sha256'] == cfg['model_sha256'], 'native launch model differs')
    return cfg, job['config_sha256'], previous.layout_inputs('B')


def historical_check(rows, history):
    """Allow only the current validation cell or its one specified historical test partner."""
    all_rows = [identity(label) for label in CASES]
    by_run = {r['run_index']:r for r in all_rows}
    record_ids = {r['recording_id'] for r in all_rows}
    require(len(by_run)==len(record_ids)==len(all_rows)==258, 'duplicate physical identity')
    env_owners = {}
    actor_seeds = set()
    for r in all_rows:
        logical = (r['phase'], r['slot'])
        seed = r['environment_seed']
        require(env_owners.setdefault(seed,logical)==logical, 'environment seed reused outside its allowed slot')
        actor = r['actor_sampling_seed']
        if actor is not None:
            require(actor not in actor_seeds, 'training actor seed repeated')
            actor_seeds.add(actor)
    require(len(env_owners)==194 and len(actor_seeds)==128 and actor_seeds.isdisjoint(env_owners),
            'seed design cardinality differs')
    owned = set(env_owners) | actor_seeds
    states = {}
    for seed in owned:
        for state in witness.ledger.java_random_states(seed).values():
            require(state not in states, 'Java state collision within campaign')
            states[state]=seed
    require(len(states)==5*len(owned), 'invalid Java stream cardinality')
    # A previous partial settlement of this campaign may already own other slots.
    # The new physical registrations must be untouched. Existing rows still need exact identity.
    pending = {r['run_index'] for r in rows}
    require(not any(f"{CAMPAIGN}-{by_run[n]['case']}" in history['campaigns'] for n in pending),
            'attempt to register an already registered physical case')
    eligible = {baseline_slot(i)['environment_seed']:baseline_slot(i) for i in range(1,51)}
    source_runs = {b['run_index'] for b in eligible.values()}
    require(len(source_runs)==50 and len(eligible)==50,'test pairs overlap')
    prior = witness.bind(witness.ledger.previous.paired10._prior_rows, SUITE=OVERVIEW)(history)
    prior.extend(j['identity'] for j in history['jobs'].values() if isinstance(j.get('identity'),dict))
    for path in witness.GRAPH.rglob('launch_identity.json'):
        if path.resolve()==OVERVIEW: continue
        cases=json.loads(path.read_bytes()).get('cases', [])
        prior.extend(cases.values() if isinstance(cases,dict) else cases)
    old_values = set()
    for prior_row in prior:
        run = prior_row.get('run_index')
        rec = prior_row.get('recording_id')
        if run in by_run:
            expected=by_run[run]
            actual=history['jobs'].get(str(run),{})
            require(run not in pending and actual.get('campaign_id') == f"{CAMPAIGN}-{expected['case']}"
                    and rec==expected['recording_id'] and prior_row.get('experiment_seed',prior_row.get('environment_seed'))
                    == expected['environment_seed']
                    and (prior_row.get('actor_sampling_seed') or 0)==(expected['actor_sampling_seed'] or 0),
                    'overlapping run or mismatched existing in-campaign receipt')
            continue
        require(rec not in record_ids and prior_row.get('source_recording_id') not in record_ids,
                'historical recording collision')
        for key in witness.SEED_FIELDS:
            v=prior_row.get(key)
            if type(v) is not int or v<=0: continue
            seed=v & witness.ledger.JAVA_MASK
            if seed in owned:
                base=eligible.get(seed)
                require(key in ('experiment_seed','environment_seed') and base is not None
                        and base['run_index']==run and rec==base['recording_id'],
                        f'historical {key} collision at run {run}')
            old_values.add(seed)
    for old in old_values:
        for state in witness.ledger.java_random_states(old).values():
            if state in states:
                require(old in eligible and old==states[state], 'Java state overlaps an unrelated historical seed')


# The per-case SQLite format is the format already merged by the project's budget_snapshot().
register_started = witness.bind(previous.register_started, CASES=CASES, identity=identity, load_case=load_case,
                                historical_check=historical_check)


def _receipt(job, result):
    return dict(result, **witness.ledger.identity(job), actor_sampling_seed_consumed=bool(job['actor_sampling_seed']))


@contextmanager
def settlement_session():
    register_started()
    persist = witness.bind(witness.settle, receipt_payload=_receipt)
    def settle(suite, job, physical, *, collect=True):
        if collect and physical['physical_status'] in ('COMPLETE','UNKNOWN') and not witness.ledger.ledger_path(suite).exists():
            register_started()
            require(witness.ledger.ledger_path(suite).is_file(), 'no identity-bound native evidence to register')
        with witness.ledger._launch_lock(Path(suite['output'])/'collector'):
            return persist(suite,job,physical,collect=collect)
    yield settle


def prepare(path):
    model = previous.model_pin(path.resolve(strict=True))
    model_bytes=Path(model['path']).read_bytes()
    experiments = {e.findtext('Name'):e for e in ET.fromstring(model_bytes).findall('./Model/Experiments/ParamVariationExperiment')}
    require(path.suffix=='.alp' and sha256(model_bytes).hexdigest()==model['sha256'], 'invalid model')
    for name,(count,code) in EXPERIMENTS.items():
        e=experiments.get(name)
        require(e is not None and e.findtext('NumberOfRuns')==str(count) and e.findtext('AllowParallelEvaluations')=='false'
                and e.findtext('FreeformParamValue/Expression/Code')==f'{code} + index', f'wrong native experiment {name}')
    cases=[]
    configs=[]
    for label in CASES:
        cfg=make_config(label,model)
        raw=canonical(cfg)
        digest=sha256(raw).hexdigest()
        suite, job=suite_job(cfg,digest)
        require(not Path(job['python_output']).exists() and not Path(job['event_directory']).exists()
                and not witness.ledger.ledger_path(suite).exists() and not Path(cfg['activation_path']).exists(),
                'shift has already been started or activated')
        row=dict(identity(label), config_path=str(suite['suite_path']), config_sha256=digest,
                 task_sha256=cfg['task_sha256'], parent_checkpoint_sha256=cfg['parent_checkpoint_sha256'],
                 matched_reference_run=(cfg['paired_baseline'] or {}).get('run_index'))
        cases.append(row)
        configs.append((suite['suite_path'],raw,digest))
    with witness.ledger._launch_lock(witness.ledger.previous.LOCKS/'budget'):
        history=witness.budget_snapshot()
        historical_check(cases,history)
        (HERE/'configs').mkdir(parents=True,exist_ok=True)
        REPORTS.mkdir(parents=True,exist_ok=True)
        CAMPAIGN_OUTPUT.mkdir(parents=True,exist_ok=True)
        archive=REPORTS/Path(model['path']).name
        write_fresh(archive,model_bytes)
        for row in cases:
            Path(row['event_directory']).parent.mkdir(parents=True,exist_ok=True)
        for target,raw,digest in configs:
            write_fresh(target,raw)
            write_fresh(target.with_suffix('.sha256'),(digest+'\n').encode())
        stream=io.StringIO(newline='')
        writer=csv.DictWriter(stream,fieldnames=cases[0],lineterminator='\n')
        writer.writeheader()
        writer.writerows(cases)
        write_fresh(HERE/'run_index_seed_manifest.csv',stream.getvalue().encode())
        overview=dict(campaign_id=CAMPAIGN, model=model, archived_model=witness.reference(archive),
                      protocol=PROTOCOL, cases=cases, registered_at_preparation=history['maximum_total_shifts'],
                      registered_by_preparation=0)
        data=canonical(overview)
        write_fresh(OVERVIEW,data)
        write_fresh(OVERVIEW.with_suffix('.sha256'),(sha256(data).hexdigest()+'\n').encode())
    return overview


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True,type=Path)
    result=prepare(parser.parse_args().model)
    print(json.dumps(dict(campaign_id=CAMPAIGN, cases=len(result['cases']), index=witness.reference(OVERVIEW),
                          registered_at_preparation=result['registered_at_preparation'])))
