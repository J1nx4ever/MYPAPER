"""Prepare the 32-shift replacement and resume the 960 never-claimed shifts."""
import argparse
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import resilience_campaign as base
from resilience_java import MASTER_FIELDS, SCENE_FIELDS, render
from schedule import seed

SOURCE_SUITE = HERE.parent/'prepared_001/suite.json'
SOURCE_SHA = '529408ae4f263852ea27379b09488aa445829acc8a7c7653bfcecb0526d029aa'
CAMPAIGN = 'bg128-fleet-resilience-20261124-repair01'
FIRST_REPLACEMENT = 10044001
REPLACEMENTS = 32
IDENTITY_FIELDS = ('ordinal','arm','role','update','split','pair_ordinal','condition','run_index','recording_id',
                   'experiment_seed','actor_seed','actor_sampling_seed','config_sha256','config_path','python_output','event_directory')


def source_suite():
    if base.reference(SOURCE_SUITE)['sha256'] != SOURCE_SHA:
        raise ValueError('original suite seal differs')
    return base.load_suite(SOURCE_SUITE)


def planned_job(job):
    return {k:v for k,v in job.items() if k!='expected_rpc_config'}


def replacement(original, output):
    """Use the same exogenous design cell and two new independent random streams."""
    j = dict(original)
    for key in ('expected_rpc_config','config_path','config_sha256'):
        j.pop(key)
    i = j['ordinal']
    run = FIRST_REPLACEMENT + i
    env = seed(CAMPAIGN,'env',i)
    rec = f'{CAMPAIGN}-{run}'
    events = output/'anylogic_events'/f'run_{run}'
    return dict(j,run_index=run,recording_id=rec,source_recording_id=rec,experiment_seed=env,
                actor_sampling_seed=seed(CAMPAIGN,'sample',i),event_directory=str(events),
                event_bundle=str(events/f'dispatch-production-wait-shortfall-v2-run-{run}-seed-{env}'),
                python_output=str(output/'shifts'/f'run_{run}'))


def java_source(suite):
    launcher = HERE/'repair_launch.py'
    src = render(suite,suite['scenario_table'],base.reference(launcher)['sha256']).decode()
    old_bridge = str(HERE.parent/'launch.py')
    old_phase = 'recordingId(index).substring(CAMPAIGN_ID.length() + 1)'
    if src.count(old_bridge)!=1 or src.count(old_phase)!=1:
        raise ValueError('unexpected Java bridge template')
    return src.replace(old_bridge,str(launcher)).replace(old_phase,'Long.toString(runIndex(index))').encode()


def prepare(directory):
    dest = Path(directory).resolve()
    output = base.ROOT/'runs'/CAMPAIGN
    if dest.exists() or output.exists():
        raise FileExistsError('recovery package or run already exists')
    src = source_suite()
    u0 = Path(src['output'])/'models/warmup/u0000/checkpoint.pt'
    suite = dict(schema='fleet-replay-recovery-v1',campaign_id=CAMPAIGN,output=str(output),
                 control_directory=str(output/'control'),suite_path=str(dest/'suite.json'),
                 source_suite=base.reference(SOURCE_SUITE),source_u0=base.reference(u0),extra_count=REPLACEMENTS,
                 sources=[base.reference(p) for p in sorted(HERE.glob('*.py'))],
                 jobs=[],**{k:src[k] for k in ('protocol','learner_seed','branch_seed','grade_margin','rpc_source',
                                                 'protected_core','parent')})
    if len(src['jobs'])!=992 or src['protocol']!=dict(warmups=4,cycles=1,validation_slots=36,test_slots=36):
        raise ValueError('unexpected original protocol')
    with TemporaryDirectory(dir=dest.parent) as tmp:
        stage=Path(tmp)/'package'
        (stage/'configs').mkdir(parents=True)
        def publish(name,data):
            (stage/name).write_bytes(data)
            return dict(path=str(dest/name),sha256=base.digest(data))
        for i, original in enumerate(src['jobs']):
            if i<REPLACEMENTS:
                j=replacement(original,output)
                cfg=dict(schema=suite['schema'],campaign_id=CAMPAIGN,job=j,replaces_run=original['run_index'],
                         rpc_source=src['rpc_source'],rpc_overrides=base.rpc_overrides(j),
                         parent=src['parent'],protected_core=src['protected_core'])
                pin=publish(f"configs/run_{j['run_index']}.json",base.encoded(cfg))
                j=dict(j,config_path=pin['path'],config_sha256=pin['sha256'])
            else:
                j=planned_job(original)
            suite['jobs'].append(j)
        rows=[]
        scenes=[]
        for j, original in zip(suite['jobs'],src['jobs'],strict=True):
            row=dict(j,role='train' if j['role']=='train' else 'final')
            rows.append([row[k] for k in MASTER_FIELDS])
            speed=original['expected_rpc_config']['config.runtime_scene.road.6.speedLimit_mps']
            scenes.append([j['run_index'],'scene-'+str(j['run_index']),*[int(r!='W') for r in j['assignment']],
                           *j['grades'],*j['half_ranges'],speed])
        suite['master_plan']=publish('master_plan.csv',base.csv_bytes(MASTER_FIELDS,rows))
        suite['scenario_table']=publish('scenario_table.csv',base.csv_bytes(SCENE_FIELDS,scenes))
        suite['identities']=publish('run_seed.csv',base.csv_bytes(IDENTITY_FIELDS,
                                    ([j[k] for k in IDENTITY_FIELDS] for j in suite['jobs'])))
        suite['java_helper']=publish('Baseline1CampaignPlan.java',java_source(suite))
        import recovery_budget
        suite['budget_at_preparation']=recovery_budget.check_fresh(suite)
        seal=publish('suite.json',base.encoded(suite))
        publish('suite.sha256',(seal['sha256']+'\n').encode())
        os.rename(stage,dest)
    return seal


def load_suite(path):
    path=Path(path).resolve(strict=True)
    data=path.read_bytes()
    if base.digest(data)!=path.with_name('suite.sha256').read_text().strip():
        raise ValueError('recovery suite seal mismatch')
    suite=json.loads(data)
    src=source_suite()
    expected=dict(schema='fleet-replay-recovery-v1',campaign_id=CAMPAIGN,extra_count=REPLACEMENTS,
                  suite_path=str(path),source_suite=base.reference(SOURCE_SUITE),output=str(base.ROOT/'runs'/CAMPAIGN),
                  source_u0=base.reference(Path(src['output'])/'models/warmup/u0000/checkpoint.pt'),
                  **{k:src[k] for k in ('protocol','learner_seed','branch_seed','grade_margin','rpc_source','protected_core','parent')})
    if any(suite[k]!=v for k,v in expected.items()) or suite['control_directory']!=str(Path(suite['output'])/'control'):
        raise ValueError('recovery lineage differs')
    for ref in [*suite['sources'],*[suite[k] for k in ('master_plan','identities','scenario_table','java_helper')]]:
        base.load_ref(ref)
    for i,(j,original) in enumerate(zip(suite['jobs'],src['jobs'],strict=True)):
        if i<REPLACEMENTS:
            want=replacement(original,Path(suite['output']))
            if {k:v for k,v in j.items() if k not in ('config_path','config_sha256')}!=want:
                raise ValueError('replacement shift diverges from authorized schedule')
            expect_path=path.parent/'configs'/f"run_{j['run_index']}.json"
            cfg=json.loads(base.load_ref(dict(path=j['config_path'],sha256=j['config_sha256'])))
            if (Path(j['config_path'])!=expect_path or cfg!=dict(schema=suite['schema'],campaign_id=CAMPAIGN,
                    job=want,replaces_run=original['run_index'],rpc_source=src['rpc_source'],
                    rpc_overrides=base.rpc_overrides(j),parent=src['parent'],protected_core=src['protected_core'])):
                raise ValueError('replacement config differs')
        elif j!=planned_job(original):
            raise ValueError('never-claimed shift changed')
        j['expected_rpc_config']=dict(original['expected_rpc_config'])
    return suite


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    print(json.dumps(prepare(parser.parse_args().directory)))
