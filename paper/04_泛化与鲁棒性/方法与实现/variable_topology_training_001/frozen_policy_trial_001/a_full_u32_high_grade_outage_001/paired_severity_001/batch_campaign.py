"""Freeze and account for the three-policy, three-duration matched A-map experiment."""
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
sys.path[:0] = [str(HERE.parent), str(HERE.parent/'same_seed_control_001')]
import a_outage_campaign as pilot
import control_campaign as control
from a_outage_campaign import PROJECT, canonical, require, stats, witness, write_fresh

CAMPAIGN = 'site-layout-a-high-grade-paired-severity-20261127-001'
CAMPAIGN_OUTPUT = PROJECT/'runs'/CAMPAIGN
REPORTS = PROJECT/'reports'/HERE.relative_to(PROJECT/'change_plans')
OVERVIEW = HERE/'launch_identity.json'
EXPERIMENT = 'Topology_A_HighGradePairedSeverity'
PURPOSE = 'topology_a_high_grade_paired_severity'
STRATEGIES = ('full','flow','flat')
DURATIONS = (0,30,60)
SLOTS = 30
PER_SLOT = len(STRATEGIES)*len(DURATIONS)
CASES = tuple(f'P_{i:03d}' for i in range(1, SLOTS*PER_SLOT+1))
FIRST_RUN = 10056101
FIRST_CODE = 5001
START = 14400
PROTOCOL = HERE/'protocol.json'
PROTOCOL_SHA = '943a40bc281ea57e01cad966f0863b4cfdcc7944389b83976e3482692f92eb3e'
ledger_entry = control.ledger_entry
budget_snapshot = control.budget_snapshot
_receipt = witness.receipt_payload


def identity(label):
    require(label in CASES, 'unknown paired severity case')
    ordinal = int(label[2:])-1
    slot, within = divmod(ordinal, PER_SLOT)
    strategy_i, duration_i = divmod(within, len(DURATIONS))
    slot += 1
    run = FIRST_RUN+ordinal
    seed_data = sha256(f'{CAMPAIGN}|environment|slot={slot}'.encode('ascii')).digest()
    seed = 1 + int.from_bytes(seed_data[:8],'big') % witness.ledger.JAVA_MASK
    out = PROJECT/'runs'/f'{CAMPAIGN}-{label}'
    return dict(case=label,slot=slot,strategy=STRATEGIES[strategy_i],duration_minutes=DURATIONS[duration_i],
                group='A',phase='test',experiment=EXPERIMENT,native_case_code=FIRST_CODE+ordinal,
                run_index=run,recording_id=f'{CAMPAIGN}-{run}',environment_seed=seed,
                actor_sampling_seed=None,slot_permutation_seed=0,event_directory=str(out/'anylogic_events'),
                python_output_directory=str(out/'python'))


@lru_cache(maxsize=len(STRATEGIES))
def policy(strategy):
    from batch_policy_pins import policy_metadata
    data = dict(policy_metadata(strategy))
    require(strategy in STRATEGIES and data['policy_id'] and type(data['policy_update']) is int
            and data['policy_update'] >= 0 and data['action_mode'] == ('deterministic' if strategy=='flow' else 'argmax')
            and len(data['checkpoint_sha256'])==64
            and all(c in '0123456789abcdef' for c in data['checkpoint_sha256'])
            and data['inference_host']=='127.0.0.1' and data['inference_port']==50871,
            f'invalid or incomplete {strategy} policy metadata')
    sources=data['inference_sources']
    require(all(witness.reference(Path(ref['path']))==ref for ref in sources)
            and witness.reference(pilot.CORE)['sha256']==pilot.finetune.CORE_SHA,
            f'changed {strategy} execution source or protected PPO core')
    required={str(HERE/'batch_policy_pins.py'),str(HERE/'batch_service.py'),str(pilot.CORE)}
    require(required.issubset(ref['path'] for ref in sources), f'{strategy} lacks runtime or PPO code pins')
    data['campaign_source']=witness.reference(Path(__file__))
    return data


@lru_cache(maxsize=1)
def physical_base():
    cfg,suite,job=control.load_case('C_001')
    status,receipt=ledger_entry(suite,job)
    manifest_path,manifest,digest=witness.native_manifest(Path(job['event_directory']),job)
    require(status=='COMPLETE' and receipt==witness.reference(Path(job['python_output'])/'receipt.json')
            and (cfg['run_index'],cfg['environment_seed'])==(control.RUN,control.SEED)
            and manifest['complete'] and manifest['terminated'] and not manifest['truncated']
            and manifest['config']['actual_runtime_scene']==cfg['expected_runtime_scene']
            and manifest['control_available_decisions']==manifest['native_policy_decisions'],
            'original control no longer binds its A road/nominal workload')
    source=cfg['expected_runtime_scene']
    require(source['exogenous_config']['load_site_outage'] is False and cfg['outage'] is None
            and len(source['opening_fleet'])==sum(r[1] for r in source['opening_fleet'])==9
            and source['load_sites'][0][:5]==[0,5,True,320000,10000]
            and (cfg['layout_id'],cfg['task']['ore_target_tons'],cfg['task']['waste_target_tons'])==('tree_A',16000,4000),
            'standard full nine-truck A scenario differs')
    return cfg, dict(role='physical configuration reference only; excluded from the 30 matched slots',
                     run_index=cfg['run_index'], config=witness.reference(suite['suite_path']),
                     manifest=dict(path=str(manifest_path),sha256=digest),receipt=receipt)


def make_config(label,model):
    idn=identity(label)
    minutes=idn['duration_minutes']
    fault=dict(site_id=0,start_seconds=START,end_seconds=START+minutes*60) if minutes else None
    base,ref=physical_base()
    scene=base['expected_runtime_scene']
    expected=dict(scene, model_sha256=model['sha256'],
                  exogenous_config=dict(scene['exogenous_config'], load_site_outage=fault or False))
    return dict(**idn, **policy(idn['strategy']), purpose=PURPOSE,layout_id=base['layout_id'],
                layout_path=base['layout_path'],layout_sha256=base['layout_sha256'],outage=fault,
                physical_reference=ref,task=base['task'],task_sha256=base['task_sha256'],
                model_path=model['path'],model_sha256=model['sha256'],expected_runtime_scene=expected,
                expected_runtime_scene_sha256=sha256(canonical(expected)).hexdigest())


suite_job = witness.bind(pilot.suite_job, HERE=HERE)


@lru_cache(maxsize=1)
def index_doc():
    raw=OVERVIEW.read_bytes()
    doc=json.loads(raw)
    require(sha256(raw).hexdigest()==OVERVIEW.with_suffix('.sha256').read_text().strip()
            and doc['campaign_id']==CAMPAIGN and doc['slots']==SLOTS
            and doc['strategies']==list(STRATEGIES) and doc['durations_minutes']==list(DURATIONS)
            and tuple(row['case'] for row in doc['cases'])==CASES, 'frozen paired severity index differs')
    require(doc['protocol']==witness.reference(PROTOCOL) and doc['protocol']['sha256']==PROTOCOL_SHA,
            'frozen protocol changed')
    return doc


case_labels=witness.bind(pilot.case_labels,index_doc=index_doc)
frozen_index=lru_cache(maxsize=1)(witness.bind(pilot.frozen_index,index_doc=index_doc))
load_case=witness.bind(pilot.load_case,HERE=HERE,index_doc=index_doc,frozen_index=frozen_index,
                        identity=identity,make_config=make_config,suite_job=suite_job)
read_case=witness.bind(pilot.read_case,load_case=load_case)


def historical_check(pending,history):
    """Admit only within-slot seed sharing, including previous registrations from the same batch."""
    rows=[identity(k) for k in CASES]
    by_run={r['run_index']:r for r in rows}
    recordings={r['recording_id'] for r in rows}
    pending_runs={r['run_index'] for r in pending}
    seeds={r['environment_seed'] for r in rows}
    require(len(by_run)==len(recordings)==len(rows) and len(seeds)==SLOTS
            and control.SEED not in seeds and len(pending_runs)==len(pending),
            'physical IDs, environment seeds or paired cohort size differ')
    cells={(r['slot'],r['strategy'],r['duration_minutes']) for r in rows}
    require(len(cells)==len(rows) and all(
        sum(r['environment_seed']==identity(CASES[i*PER_SLOT])['environment_seed'] and r['slot']==i+1
            for r in rows)==PER_SLOT for i in range(SLOTS)), 'slot pairing differs')
    state_values={state for s in seeds for state in witness.ledger.java_random_states(s).values()}
    require(len(state_values)==5*SLOTS, 'Java stream collision within new roster')
    for row in pending:
        run=row['run_index']
        require(run in by_run and all(row[k]==v for k,v in by_run[run].items())
                and str(run) not in history['jobs']
                and f"{CAMPAIGN}-{row['case']}" not in history['campaigns'], 'unknown or used physical ID')

    prior=witness.bind(witness.ledger.previous.paired10._prior_rows,SUITE=OVERVIEW)(history)
    prior.extend(job['identity'] for job in history['jobs'].values() if isinstance(job.get('identity'),dict))
    for path in witness.GRAPH.rglob('launch_identity.json'):
        if path.resolve()==OVERVIEW:
            continue
        records=json.loads(path.read_bytes()).get('cases',[])
        prior.extend(records.values() if isinstance(records,dict) else records)
    old_seeds=set()
    for item in prior:
        run=item.get('run_index')
        if run in by_run:
            r=by_run[run]
            merged=history['jobs'].get(str(run),{})
            require(run not in pending_runs and merged.get('campaign_id')==f"{CAMPAIGN}-{r['case']}"
                    and item.get('recording_id')==r['recording_id']
                    and item.get('experiment_seed',item.get('environment_seed'))==r['environment_seed']
                    and not item.get('actor_sampling_seed'), 'another activity owns a batch run')
            continue
        require(item.get('recording_id') not in recordings and item.get('source_recording_id') not in recordings,
                'recording already registered or reserved')
        for key in witness.SEED_FIELDS:
            value=item.get(key)
            if type(value) is int and value>0:
                old_seeds.add(value & witness.ledger.JAVA_MASK)
    require(seeds.isdisjoint(old_seeds) and all(state_values.isdisjoint(witness.ledger.java_random_states(s).values())
                                               for s in old_seeds), 'environment seeds/derived streams intersect history')
    return dict(historical_rows=len(prior),new_runs=len(pending),unique_environment_seeds=len(seeds),
                new_actor_seeds=0,distinct_java_states=len(state_values))


register_started=witness.bind(stats.register_started,CASES=CASES,identity=identity,load_case=load_case,
                              historical_check=historical_check)
settlement_session=contextmanager(witness.bind(pilot.settlement_session.__wrapped__, register_started=register_started))
verify_budget=witness.bind(pilot.verify_budget,CAMPAIGN=CAMPAIGN,OVERVIEW=OVERVIEW,CASES=CASES,
                           load_case=load_case,REPORTS=REPORTS)


def prepare(path):
    model=stats.model_pin(path.resolve(strict=True))
    contents=Path(model['path']).read_bytes()
    require(path.suffix=='.alp' and sha256(contents).hexdigest()==model['sha256'], 'native source changed')
    e,=[e for e in ET.fromstring(contents).findall('./Model/Experiments/ParamVariationExperiment')
        if e.findtext('Name')==EXPERIMENT]
    require(e.findtext('NumberOfRuns')==str(len(CASES)) and e.findtext('AllowParallelEvaluations')=='false'
            and e.findtext('FreeformParamValue/Expression/Code')==f'{FIRST_CODE} + index',
            'native iteration schedule differs')
    protocol=witness.reference(PROTOCOL)
    require(protocol['sha256']==PROTOCOL_SHA, 'predeclared statistical protocol changed')
    if OVERVIEW.exists() and OVERVIEW.with_suffix('.sha256').exists():
        sealed=index_doc()
        require(sealed['model']==model, 'requested model differs from the sealed batch')
        for label in CASES:
            load_case(label)
        return sealed
    rows=[]
    configs=[]
    suites=[]
    for label in CASES:
        cfg=make_config(label,model)
        raw=canonical(cfg)
        digest=sha256(raw).hexdigest()
        suite,job=suite_job(cfg,digest)
        require(not Path(job['event_directory']).exists() and not Path(job['python_output']).exists()
                and not witness.ledger.ledger_path(suite).exists(),'physical output or budget ledger already exists')
        row=dict(identity(label),config_path=str(suite['suite_path']),config_sha256=digest,
                 policy_id=cfg['policy_id'],policy_update=cfg['policy_update'],
                 checkpoint_sha256=cfg['checkpoint_sha256'],action_mode=cfg['action_mode'],task_sha256=cfg['task_sha256'])
        rows.append(row)
        suites.append(suite)
        configs.append((suite['suite_path'],raw,digest))
    with witness.ledger._launch_lock(witness.ledger.previous.LOCKS/'budget'):
        # A partially written, unlaunched packet is safe to resume byte for byte.
        # The index seal is published last. A native or registered shift is never overwritten.
        require(not any(witness.ledger.ledger_path(suite).exists() for suite in suites),
                'a shift was already registered during preparation')
        require(all(not Path(row['event_directory']).exists() and not Path(row['python_output_directory']).exists()
                    for row in rows), 'physical control or inference output appeared during preparation')
        history=budget_snapshot()
        audit=historical_check(rows,history)
        archive=REPORTS/Path(model['path']).name
        REPORTS.mkdir(parents=True,exist_ok=True)
        write_fresh(archive,contents)
        (HERE/'configs').mkdir(exist_ok=True)
        CAMPAIGN_OUTPUT.mkdir(exist_ok=True)
        for row in rows:
            Path(row['event_directory']).parent.mkdir(exist_ok=True)
        for path,raw,digest in configs:
            write_fresh(path,raw)
            write_fresh(path.with_suffix('.sha256'),(digest+'\n').encode())
        stream=io.StringIO(newline='')
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
        write_fresh(HERE/'run_index_seed_manifest.csv',stream.getvalue().encode())
        snapshot=dict(campaign_id=CAMPAIGN,slots=SLOTS,strategies=list(STRATEGIES),durations_minutes=list(DURATIONS),
                      model=model,archived_model=witness.reference(archive),protocol=protocol,
                      excluded_previous_runs=[control.ORIGINAL_RUN,control.RUN],case_identity_audit=audit,
                      registered_at_preparation=history['maximum_total_shifts'],registered_by_preparation=0,cases=rows)
        data=canonical(snapshot)
        write_fresh(OVERVIEW,data)
        write_fresh(OVERVIEW.with_suffix('.sha256'),(sha256(data).hexdigest()+'\n').encode())
    return snapshot


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    doc=prepare(parser.parse_args().model)
    print(json.dumps(dict(cases=len(doc['cases']),index=witness.reference(OVERVIEW),
                          first=doc['cases'][0],last=doc['cases'][-1],before=doc['registered_at_preparation'])))
