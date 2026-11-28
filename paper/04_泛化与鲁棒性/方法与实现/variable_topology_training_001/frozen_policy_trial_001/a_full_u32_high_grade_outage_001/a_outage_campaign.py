"""Freeze and settle one stochastic tree_A outage shift using the unmigrated Full s02/u32."""

import argparse
from contextlib import contextmanager
from functools import lru_cache
from hashlib import sha256
import csv
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
TRIAL = HERE.parent
sys.path[:0] = [str(TRIAL/'b_finetune_001'), str(TRIAL/'paired_50_001')]
import campaign as finetune
import prepare_stats as stats
from campaign import PROJECT, canonical, require, witness, write_fresh

CAMPAIGN = 'site-layout-a-full-u32-high-grade-outage-20261127-001'
CAMPAIGN_OUTPUT = PROJECT/'runs'/CAMPAIGN
REPORTS = PROJECT/'reports'/HERE.relative_to(PROJECT/'change_plans')
OVERVIEW = HERE/'launch_identity.json'
EXPERIMENT = 'Topology_A_Full_HighGradeOutage'
PURPOSE = 'topology_a_full_high_grade_outage'
CASES = ('O_001',)
CHECKPOINT = PROJECT/'runs/bg128-continuous-mixed-20261120-s02/arms/s01_BG128/updates/0032/checkpoint.pt'
CHECKPOINT_SHA = '67b7b2c7104079836ccef8f5747e7aebb4e9e5a92dcaac67c94e42412f104f41'
ACTOR_DIGEST = '189a2c571c4bcf0fbe978330d92dddb84f3b196b753f7ed93e533988d4c59abd'
CORE = PROJECT.parents[1]/'python/episode_budget_ppo.py'
OUTAGE = dict(site_id=0, start_seconds=14400, end_seconds=16200)
budget_snapshot = witness.budget_snapshot
ledger_entry = witness.ledger_entry
_receipt = witness.receipt_payload


def identity(label):
    require(label in CASES, 'unknown physical outage slot')
    run = 10056051
    seed = 1 + int.from_bytes(sha256(f'{CAMPAIGN}|environment|{run}'.encode()).digest()[:8],'big') % witness.ledger.JAVA_MASK
    folder = PROJECT/'runs'/f'{CAMPAIGN}-{label}'
    return dict(case=label, slot=1, group='A', phase='test', experiment=EXPERIMENT, native_case_code=4001,
                run_index=run, recording_id=f'{CAMPAIGN}-{run}', environment_seed=seed, action_mode='argmax',
                actor_sampling_seed=None, slot_permutation_seed=0, event_directory=str(folder/'anylogic_events'),
                python_output_directory=str(folder/'python'))


@lru_cache(maxsize=1)
def policy():
    cp = witness.reference(CHECKPOINT)
    manifest_path = CHECKPOINT.with_suffix('.pt.manifest.json')
    seal_path = CHECKPOINT.with_suffix('.pt.seal.json')
    manifest = json.loads(manifest_path.read_bytes())
    seal = json.loads(seal_path.read_bytes())
    manifest_pin = witness.reference(manifest_path)
    require(cp['sha256']==manifest['sha256']==seal['sha256']==CHECKPOINT_SHA
            and manifest['schema']=='business-graph-summary-ppo-checkpoint-v1' and manifest['update_index']==32
            and manifest['actor_digest']==ACTOR_DIGEST and manifest['bytes']==CHECKPOINT.stat().st_size
            and seal['manifest_sha256']==manifest_pin['sha256']
            and witness.reference(CORE)['sha256']==finetune.CORE_SHA, 'Full parent checkpoint, seal or PPO core differs')
    files = [HERE/'outage_service.py', HERE/'a_outage_campaign.py', TRIAL/'trial_service.py',
             TRIAL/'capacity_scaled_001/scaled_service.py', TRIAL.parent/'topology_policy.py', CORE]
    for pkg, names in (
        ('business_graph_summary_v1', ('models','graph','graph_layers','encoder','contract')),
        ('single_critic_task_query_v1', ('features_cache','c_features')),
        ('relation_gat_ego_v1', ('adapter','initialization'))):
        files.extend(PROJECT/pkg/f'{name}.py' for name in names)
    return dict(policy_id='bg128-full-s02-u32', policy_update=32, checkpoint_path=cp['path'],
                checkpoint_sha256=CHECKPOINT_SHA, checkpoint_manifest=manifest_pin, checkpoint_seal=witness.reference(seal_path),
                actor_digest=ACTOR_DIGEST, inference_sources=[witness.reference(p) for p in files],
                inference_host='127.0.0.1', inference_port=50871)


@lru_cache(maxsize=1)
def physical_base():
    base,suite,job = stats.load_case('A_001')
    status, receipt = ledger_entry(suite,job)
    out = Path(job['python_output'])/'receipt.json'
    bound_path,manifest,digest = witness.native_manifest(Path(job['event_directory']),job)
    require(status=='COMPLETE' and receipt==witness.reference(out) and manifest['complete'] and manifest['terminated']
            and not manifest['truncated'] and manifest['config']['actual_runtime_scene']==base['expected_runtime_scene'],
            'reference A road, equipment, opening target or native bound did not verify')
    row=json.loads(out.read_bytes())
    require(row['manifest']==dict(path=str(bound_path),sha256=digest) and row['material_balance_verified']
            and row['physical_status']=='COMPLETE','A reference material audit differs')
    pin=dict(role='A road, opening fleet and task reference',run_index=job['run_index'],config=witness.reference(suite['suite_path']),
             receipt=receipt,manifest=row['manifest'])
    return base,pin


def make_config(label, model):
    base, ref=physical_base()
    scene=dict(base['expected_runtime_scene'],model_sha256=model['sha256'])
    native=dict(scene['exogenous_config'],load_site_outage=OUTAGE)
    scene['exogenous_config']=native
    site=max((s for s in scene['load_sites'] if s[2]),key=lambda s:s[3])
    require(scene['layout_id']=='tree_A' and site[:5]==[0,5,True,320000,10000]
            and len(scene['opening_fleet'])==sum(v[1] for v in scene['opening_fleet'])==9
            and (base['task']['ore_target_tons'],base['task']['waste_target_tons'])==(16000,4000)
            and not native['deterministic'] and native['fault_truck_id']==-1,
            'A grade ranking, fleet, targets or random conditions differ')
    return dict(**identity(label),**policy(),purpose=PURPOSE,model_path=model['path'],model_sha256=model['sha256'],
                layout_id=scene['layout_id'],layout_path=base['layout_path'],layout_sha256=base['layout_sha256'],
                expected_runtime_scene=scene,expected_runtime_scene_sha256=sha256(canonical(scene)).hexdigest(),
                task=base['task'],task_sha256=base['task_sha256'],outage=OUTAGE,paired_baseline=None,
                physical_reference=ref)


def suite_job(cfg,digest):
    suite,job=stats.suite_job(cfg,digest)
    suite['suite_path']=HERE/'configs'/f"{cfg['case']}.json"
    return suite,job


@lru_cache(maxsize=1)
def index_doc():
    raw=OVERVIEW.read_bytes()
    doc=json.loads(raw)
    require(sha256(raw).hexdigest()==OVERVIEW.with_suffix('.sha256').read_text().strip()
            and doc['campaign_id']==CAMPAIGN and doc['outage']==OUTAGE
            and tuple(row['case'] for row in doc['cases'])==CASES,'launch identity differs')
    return doc


def case_labels():
    return tuple(row['case'] for row in index_doc()['cases'])


def frozen_index():
    return {row['case']:row for row in index_doc()['cases']}


def load_case(label):
    row=frozen_index()[label]
    doc=index_doc()
    path=HERE/'configs'/f'{label}.json'
    raw=path.read_bytes()
    sha=sha256(raw).hexdigest()
    cfg=json.loads(raw)
    require(row['config_path']==str(path) and row['config_sha256']==path.with_suffix('.sha256').read_text().strip()==sha
            and all(row[k]==v for k,v in identity(label).items())
            and stats.model_pin(Path(doc['archived_model']['path']))['sha256']==doc['model']['sha256']
            and cfg==make_config(label,doc['model']), 'frozen configuration, source, checkpoint or physical inputs differ')
    return cfg,*suite_job(cfg,sha)


def read_case(label):
    cfg,_,job=load_case(label)
    require(stats.model_pin(Path(cfg['model_path']))['sha256']==cfg['model_sha256'],'installed model SHA differs')
    return cfg,job['config_sha256'],stats.layout_inputs('A')


def historical_check(rows, history):
    witness.bind(stats.historical_check,CAMPAIGN=CAMPAIGN,OVERVIEW=OVERVIEW)(rows,history)
    # Compare effective Java stream states versus historical Actor seeds as well.
    states=set(witness.ledger.java_random_states(rows[0]['environment_seed']).values())
    prior=witness.bind(witness.ledger.previous.paired10._prior_rows,SUITE=OVERVIEW)(history)
    prior.extend(j['identity'] for j in history['jobs'].values() if isinstance(j.get('identity'),dict))
    for p in witness.GRAPH.rglob('launch_identity.json'):
        if p!=OVERVIEW:
            rs=json.loads(p.read_bytes()).get('cases',[])
            prior.extend(rs.values() if isinstance(rs,dict) else rs)
    actors={r[key] for r in prior for key in ('actor_seed','actor_sampling_seed')
            if type(r.get(key)) is int and r[key]>0}
    require(all(states.isdisjoint(witness.ledger.java_random_states(v).values()) for v in actors),
            'effective A outage stream collides with a historical Actor seed')


register_started = witness.bind(stats.register_started, CASES=CASES, identity=identity, load_case=load_case,
                                historical_check=historical_check)


@contextmanager
def settlement_session():
    register_started()
    def settle(suite, job, physical, *, collect=True):
        if collect and physical['physical_status'] in ('COMPLETE','UNKNOWN') and not witness.ledger.ledger_path(suite).exists():
            register_started()
            require(witness.ledger.ledger_path(suite).is_file(), 'no native manifest for registration')
        with witness.ledger._launch_lock(Path(suite['output'])/'collector'):
            return witness.settle(suite,job,physical,collect=collect)
    yield settle


def verify_budget(rows):
    import results as prior
    require(witness.reference(CORE)['sha256']==finetune.CORE_SHA,'protected PPO core differs after evaluation')
    facade=SimpleNamespace(CAMPAIGN=CAMPAIGN,OVERVIEW=OVERVIEW,CASES=CASES,budget_snapshot=budget_snapshot,
                           ledger_entry=ledger_entry,load_case=load_case,canonical=canonical,_receipt=_receipt)
    return witness.bind(prior.verify_budget,campaign=facade,REPORTS=REPORTS)(rows)


def prepare(path):
    label,=CASES
    model=stats.model_pin(path.resolve(strict=True))
    model_bytes=Path(model['path']).read_bytes()
    require(path.suffix=='.alp' and model['sha256']==sha256(model_bytes).hexdigest(),'invalid native model')
    ex,=[e for e in ET.fromstring(model_bytes).findall('./Model/Experiments/ParamVariationExperiment')
         if e.findtext('Name')==EXPERIMENT]
    require(ex.findtext('NumberOfRuns')=='1' and ex.findtext('AllowParallelEvaluations')=='false'
            and ex.findtext('FreeformParamValue/Expression/Code')=='4001 + index','expected a single native slot')
    cfg=make_config(label,model)
    raw=canonical(cfg)
    sha=sha256(raw).hexdigest()
    suite,job=suite_job(cfg,sha)
    row=dict(identity(label),config_path=str(suite['suite_path']),config_sha256=sha,checkpoint_sha256=CHECKPOINT_SHA,
             task_sha256=cfg['task_sha256'],site_id=OUTAGE['site_id'],start_seconds=OUTAGE['start_seconds'],
             end_seconds=OUTAGE['end_seconds'])
    with witness.ledger._launch_lock(witness.ledger.previous.LOCKS/'budget'):
        require(not any(p.exists() for p in (OVERVIEW,OVERVIEW.with_suffix('.sha256'), HERE/'configs',
            HERE/'run_index_seed_manifest.csv',CAMPAIGN_OUTPUT,Path(job['event_directory']).parent)),
            'physical output or prepared identities already exist')
        history=budget_snapshot()
        historical_check([row],history)
        archive=REPORTS/Path(model['path']).name
        require(not archive.exists() or archive.read_bytes()==model_bytes,'archived model differs')
        REPORTS.mkdir(parents=True,exist_ok=True)
        (HERE/'configs').mkdir()
        CAMPAIGN_OUTPUT.mkdir()
        Path(job['event_directory']).parent.mkdir()
        write_fresh(archive,model_bytes)
        write_fresh(suite['suite_path'],raw)
        write_fresh(suite['suite_path'].with_suffix('.sha256'),(sha+'\n').encode())
        stream=io.StringIO(newline='')
        writer=csv.DictWriter(stream,fieldnames=list(row),lineterminator='\n')
        writer.writeheader()
        writer.writerow(row)
        write_fresh(HERE/'run_index_seed_manifest.csv',stream.getvalue().encode())
        doc=dict(campaign_id=CAMPAIGN,cases=[row],outage=OUTAGE,model=model,archived_model=witness.reference(archive),
                 registered_at_preparation=history['maximum_total_shifts'],registered_by_preparation=0)
        raw_index=canonical(doc)
        write_fresh(OVERVIEW,raw_index)
        write_fresh(OVERVIEW.with_suffix('.sha256'),(sha256(raw_index).hexdigest()+'\n').encode())
    return doc


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True,type=Path)
    doc=prepare(parser.parse_args().model)
    print(json.dumps(dict(index=witness.reference(OVERVIEW), case=doc['cases'][0], before=doc['registered_at_preparation'])))
