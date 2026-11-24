"""Admit 32 replacement shifts and settle the 960 unused identities in their original ledger."""
import json
from pathlib import Path
import sqlite3
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import resilience_budget as old
from resilience_campaign import PLAN, ROOT, digest, encoded, load_suite, reference

SOURCE = PLAN/'prepared_001/suite.json'
SOURCE_SHA = '529408ae4f263852ea27379b09488aa445829acc8a7c7653bfcecb0526d029aa'
CAMPAIGN = 'bg128-fleet-resilience-20261124-repair01'
NEW_RUNS = range(10044001, 10044033)
PREFIX_SHA = '4c3d0db07bc4cf8087b3c13a84319ee39ff625514f4269474fe970108aa9ad84'
FAILURE_PINS = {
    'models/warmup/u0000/checkpoint.pt':'7bb809db30e5539c34abb22415af5137500ec22ad0652cdeea2cba1859ab681b',
    'models/warmup/u0000/checkpoint.pt.manifest.json':'9940afecc39411170c793f1bb0e13115ecbd2253f4e9b6b672f9b796900510bd',
    'models/warmup/u0000/checkpoint.pt.seal.json':'9c16f9bb154ee8f91309e1881d1a6d00272674e5b14f8739d3f08f7b1661994a',
    'models/warmup/u0001/intent.json':'41070b3d971ec0189512c3f106376efa6a39d93bd674a3b30bde994525314b9a',
    'run_status.json':'753b33e2cae9334bc3b99b638cf68fbaf36d4bb2ea1e38107557cab7e94a7636',
    'control/STOP.properties':'2ff8f56767ee0f92092deb8842112fe689f4ec0a7a73147caa2f5a8a13173093'
}


def require(test, message):
    if not test:
        raise ValueError(message)


def source_ref():
    return dict(path=str(SOURCE), sha256=SOURCE_SHA)


def source_output():
    return ROOT/'runs'/'bg128-fleet-resilience-20261124-001'


def ledger_rows(path):
    with sqlite3.connect(Path(path).as_uri()+'?mode=ro',uri=True) as db:
        name,pin = db.execute('SELECT name,suite FROM campaign').fetchone()
        rows = db.execute('SELECT run,id,status,evidence FROM slots ORDER BY run').fetchall()
    return name,json.loads(pin),{n:(json.loads(identity),status,evidence) for n,identity,status,evidence in rows}


def inspect_source(source, *, unused):
    root = Path(source['output'])
    name,pin,rows = ledger_rows(old.ledger_path(source))
    jobs = source['jobs']
    require(name==source['campaign_id'] and root==source_output() and pin==source_ref() and len(rows)==len(jobs)==992,
            'original registered source differs')
    for key,value in FAILURE_PINS.items():
        require(reference(root/key)['sha256']==value,'historical failure evidence differs: '+key)
    require({p.name for p in (root/'models/warmup/u0001').iterdir()}=={'intent.json'},'unexpected original u1 artifact')
    ctl = root/'control'
    prefix_ids = [j['run_index'] for j in jobs[:32]]
    for folder,pattern in [('ready','run_{}.properties'),('java_claims','run_{}.once.claim')]:
        require({p.name for p in (ctl/folder).iterdir()}=={pattern.format(r) for r in prefix_ids},
                'original Java handshake inventory differs')
    prefix_pins=[]
    for i,j in enumerate(jobs):
        run = j['run_index']
        identity,status,evidence = rows[run]
        require(old.identity(j)==identity,'original ledger identity differs')
        receipt = Path(j['python_output'])/'receipt.json'
        if i<32:
            require(status=='COMPLETE' and json.loads(evidence)==reference(receipt),
                    'original completed receipt differs')
            prefix_pins.append([run,reference(ctl/'ready'/f'run_{run}.properties')['sha256'],
                               reference(ctl/'java_claims'/f'run_{run}.once.claim')['sha256'],
                               json.loads(evidence)['sha256']])
        elif unused:
            require(status=='PLANNED' and evidence is None and not Path(j['python_output']).exists()
                    and not Path(j['event_directory']).exists(),'old physical slot is no longer unused')
    require(digest(encoded(prefix_pins))==PREFIX_SHA,'historical prefix receipts/claims/ready differ')
    return dict(complete_count=32, remaining_count=960, original_prefix_sha256=PREFIX_SHA)


def audit_source(source_suite):
    """Read-only check performed before a replacement is prepared or first registered."""
    ref = source_suite if 'path' in source_suite else reference(source_suite['suite_path'])
    require(ref==source_ref() and reference(SOURCE)==ref,'wrong source package')
    return inspect_source(load_suite(SOURCE),unused=True)


def budget_snapshot():
    # The original merger discovers the standard side ledger of 32 extra slots. The
    # 960 retained slots continue to be counted exactly once in the old ledger.
    return old.budget_snapshot()


def validate_roster(suite,source):
    jobs=suite['jobs']
    require(suite['campaign_id']==CAMPAIGN and Path(suite['output'])==ROOT/'runs'/CAMPAIGN and
            suite['source_suite']==source_ref() and suite['extra_count']==32 and suite['protocol']==source['protocol']
            and len(jobs)==992 and suite['learner_seed']==source['learner_seed']
            and suite['branch_seed']==source['branch_seed'],'unexpected recovery campaign or protocol')
    for left,right in zip(jobs[32:],source['jobs'][32:],strict=True):
        # Runtime may attach validated RPC fields; optional recovery-only metadata
        # cannot override physical identity, config path/bytes, scenario or update.
        require(all(left.get(k)==v for k,v in right.items() if k!='expected_rpc_config') and
                ('expected_rpc_config' not in left or left['expected_rpc_config']==right['expected_rpc_config']),
                'retained job differs from its unused source row')
    scene_keys=('assignment','waste_site','starting_count','initial_mask','grades','half_ranges','ore_target',
                'waste_target','waste_stop','fault_truck_id','fault_trigger_seconds','phase','arm','role','condition',
                'split','pair_ordinal','ordinal','update','checkpoint_update','target_training_update','actor_seed')
    for run,j,prior in zip(NEW_RUNS,jobs[:32],source['jobs'][:32],strict=True):
        rec=f'{CAMPAIGN}-{run}'
        require(j['run_index']==run and all(j[k]==prior[k] for k in scene_keys)
                and j['source_recording_id']==j['recording_id']==rec and
                Path(j['python_output'])==Path(suite['output'])/'shifts'/f'run_{run}' and
                Path(j['event_directory'])==Path(suite['output'])/'anylogic_events'/f'run_{run}',
                'replacement must use the original scene and fresh physical identity')


def check_fresh(suite):
    require(suite['source_suite']==source_ref() and reference(SOURCE)==source_ref(),'wrong original suite pin')
    source=load_suite(SOURCE)
    inspect_source(source,unused=True)
    validate_roster(suite,source)
    history=budget_snapshot()
    require(CAMPAIGN not in history['campaigns'],'replacement campaign already registered')
    from weak_plan import bind
    prior=bind(old.previous.paired10._prior_rows,SUITE=Path(suite['suite_path']))(history)
    prior.extend(row['identity'] for row in history['jobs'].values() if isinstance(row.get('identity'),dict))
    used_runs={j.get('run_index') for j in prior}
    recordings={j.get(k) for j in prior for k in ('recording_id','source_recording_id')}
    used_seeds={j[k]&old.JAVA_MASK for j in prior for k in ('actor_seed','experiment_seed','actor_sampling_seed')
                if type(j.get(k)) is int and j[k]>0}
    environments={j['experiment_seed'] for j in prior if type(j.get('experiment_seed')) is int and j['experiment_seed']>0}
    states={v for s in environments for v in old.java_random_states(s).values()}
    new=suite['jobs'][:32]
    for job in new:
        require(job['run_index'] not in used_runs and job['recording_id'] not in recordings
                and not Path(job['python_output']).exists() and not Path(job['event_directory']).exists(),
                'replacement run, recording or output already exists')
    seeds=[j[k] for j in new for k in ('experiment_seed','actor_sampling_seed')]
    require(len(set(seeds))==64 and all(type(s) is int and 0<s<=old.JAVA_MASK for s in seeds)
            and set(seeds).isdisjoint(used_seeds),'replacement environment/Actor seed collision')
    streams={v for j in new for v in old.java_random_states(j['experiment_seed']).values()}
    require(len(streams)==160 and streams.isdisjoint(states),'replacement Java state collision')
    before=history['maximum_total_shifts']
    return dict(registered_before=before,additional_if_started=32,total_if_started=before+32,
                reused_unclaimed=960,new_environment_seeds=32,new_actor_sampling_seeds=32)


def register(suite):
    """Run only on authorized activation. Install the complete side ledger exclusively."""
    path=old.ledger_path(suite)
    with old._launch_lock(old.previous.LOCKS/'budget'):
        if path.exists():
            name,pin,rows=ledger_rows(path)
            require(name==suite['campaign_id'] and pin==reference(suite['suite_path'])
                    and list(rows)==list(NEW_RUNS),'registered repair suite differs')
            original=load_suite(SOURCE)
            validate_roster(suite,original)
            inspect_source(original,unused=False)
            state_view(suite)
            return
        check_fresh(suite)
        path.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
            staging=Path(tmp)/path.name
            with sqlite3.connect(staging) as db:
                db.execute('CREATE TABLE campaign(name TEXT PRIMARY KEY,suite TEXT NOT NULL)')
                db.execute('CREATE TABLE slots(run INTEGER PRIMARY KEY,id TEXT NOT NULL,status TEXT NOT NULL,evidence TEXT)')
                db.execute('INSERT INTO campaign VALUES (?,?)',(CAMPAIGN,encoded(reference(suite['suite_path'])).decode()))
                db.executemany("INSERT INTO slots VALUES (?,?,'PLANNED',NULL)",
                               [(j['run_index'],encoded(old.identity(j)).decode()) for j in suite['jobs'][:32]])
            staging.rename(path)


def state_view(suite):
    """Expose only the 992 slots actually executed by the repair program."""
    name,pin,new=ledger_rows(old.ledger_path(suite))
    require(name==CAMPAIGN and pin==reference(suite['suite_path']) and list(new)==list(NEW_RUNS),
            'repair ledger or suite pin differs')
    _,original_pin,original=ledger_rows(old.ledger_path({'output':source_output()}))
    require(original_pin==suite['source_suite']==source_ref(),'original ledger pin differs')
    view={}
    for i,j in enumerate(suite['jobs']):
        run=j['run_index']
        require(run==(NEW_RUNS.start+i if i<32 else 10043001+i),'unexpected repair run order')
        saved,status,evidence=(new if i<32 else original)[run]
        require(saved==old.identity(j),'physical identity differs')
        view[run]=(status,evidence)
    require(len(view)==992,'recovery roster length differs')
    return view


def owner(suite,job, *, claiming=False):
    run=job['run_index']
    if run in NEW_RUNS:
        index=run-NEW_RUNS.start
        owner_suite=suite
    else:
        require(10043033<=run<=10043992,'attempted use of consumed or foreign run')
        index=run-10043001
        owner_suite={'output':source_output()}
    require(suite['campaign_id']==CAMPAIGN and suite['source_suite']==source_ref()
            and old.identity(job)==old.identity(suite['jobs'][index]),'unauthorized physical job')
    with sqlite3.connect(old.ledger_path(suite).as_uri()+'?mode=ro',uri=True) as db:
        name,pin=db.execute('SELECT name,suite FROM campaign').fetchone()
        require(name==CAMPAIGN and json.loads(pin)==reference(suite['suite_path']),'unregistered repair')
        if claiming and run not in NEW_RUNS:
            require(db.execute("SELECT COUNT(*) FROM slots WHERE status='COMPLETE'").fetchone()[0]==32,
                    'replacement batch not complete')
    return owner_suite


def claim(suite,job):
    old.transition(owner(suite,job,claiming=True),job,'PLANNED','CLAIMED')


def complete(suite,job,receipt_path):
    old.complete(owner(suite,job),job,receipt_path)


def unknown(suite,job,reason):
    old.unknown(owner(suite,job),job,reason)
