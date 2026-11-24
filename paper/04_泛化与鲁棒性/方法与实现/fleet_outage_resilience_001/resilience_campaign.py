"""Prepare reproducible baseline2 campaign identities. Preparation consumes zero shifts."""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from resilience_java import MASTER_FIELDS, SCENE_FIELDS, TEMPLATE, render
from schedule import schedule, seed

PLAN = Path(__file__).resolve().parent
ROOT = PLAN.parents[3]
CORE = ROOT.parents[1]/'python/episode_budget_ppo.py'
PARENT = ROOT/'runs/bg128-continuous-mixed-20261120-s02/arms/s01_BG128/updates/0032/checkpoint.pt'
CORE_SHA = '8e4cfa31c622f5e33756e304eea3f7526547db72ae93f7d9f00808845d61ef96'
PARENT_SHA = '67b7b2c7104079836ccef8f5747e7aebb4e9e5a92dcaac67c94e42412f104f41'
BASE = ROOT/'runs/bg128-full-s02-u32-fleet6-zero-shot10-20261124-001/shifts/run_10042001/evaluation_receipt.json'


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def reference(path):
    return dict(path=str(path), sha256=digest(Path(path).read_bytes()))


def load_ref(ref):
    data = Path(ref['path']).read_bytes()
    if digest(data) != ref['sha256']:
        raise ValueError('frozen bytes differ: '+str(ref['path']))
    return data


def csv_bytes(columns, rows):
    output = io.StringIO(newline='')
    writer = csv.writer(output, lineterminator='\n')
    writer.writerow(columns)
    writer.writerows(rows)
    return output.getvalue().encode()


def rpc_overrides(scene):
    result = {'config.fleet.outage_version':2., 'config.fleet.active_count':float(scene['starting_count']),
              'config.fleet.disabled_mask':float(511 ^ scene['initial_mask']),
              'config.target_ore_tons_shift':scene['ore_target'], 'config.target_ore_tons_2h':scene['ore_target']/4,
              'config.target_waste_tons_shift':scene['waste_target'], 'config.waste_task_dispatch_stop_tons':scene['waste_stop'],
              'config.waste_task_upper_tons':scene['waste_stop']+193., 'config.material_event_memory_enabled':0.}
    for i, role in enumerate(scene['assignment']):
        prefix=f'config.runtime_scene.load.{i}.'
        result.update({prefix+'isOre':float(role!='W'),prefix+'grade':scene['grades'][i],
                       prefix+'gradeHalfRange':scene['half_ranges'][i]})
    return result


def prepare(directory, campaign_id, first_run, warmups, cycles, validation_slots, test_slots, grade_margin=0.):
    directory = Path(directory).resolve()
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{2,80}', campaign_id) or first_run <= 0:
        raise ValueError('invalid campaign or first run')
    if not 0 <= grade_margin <= 1:
        raise ValueError('grade tolerance must be in [0,1]')
    output = ROOT/'runs'/campaign_id
    if directory.exists() or output.exists():
        raise FileExistsError('choose a fresh preparation directory and campaign')
    if reference(CORE)['sha256']!=CORE_SHA or reference(PARENT)['sha256']!=PARENT_SHA:
        raise ValueError('parent checkpoint or PPO core differs')
    directory.parent.mkdir(parents=True,exist_ok=True)
    base_rpc=json.loads(BASE.read_bytes())['actual_rpc_configuration']
    sources = sorted(PLAN.glob('*.py')) + sorted(PLAN.glob('*.java.txt')) + [
        TEMPLATE, PLAN/'manual_anylogic.md', PARENT.with_suffix('.pt.manifest.json'), PARENT.with_suffix('.pt.seal.json')]
    model_dependencies = ('business_graph_summary_v1', 'single_critic_task_query_v1', 'relation_gat_ego_v1')
    for module in model_dependencies:
        sources.extend(sorted(p for p in (ROOT/module).glob('*.py') if not p.name.startswith('test_')))
    sources.extend(ROOT/'python'/name for name in ('dispatch_bridge.py','reward_config.py','reward_contract.py',
                   'observation_contract.py','observation_schema.json','scenario_contract.py','runtime_scenario.py',
                   'generated/dispatch_pb2.py','generated/dispatch_pb2_grpc.py'))
    suite = dict(schema='bg128-fleet-resilience-v1',campaign_id=campaign_id,output=str(output),
                 suite_path=str(directory/'suite.json'),control_directory=str(output/'control'),
                 protected_core=reference(CORE), parent=reference(PARENT),rpc_source=reference(BASE),
                 baseline2_audit_sha256='eb8c46b7f7410daeb9997bf18ac0aa07e53ca5e5a652e0d79e15ceb425baae26',
                 sources=[reference(p) for p in sources],learner_seed=seed(campaign_id,'learner')%(2**31-1)+1,
                 branch_seed=seed(campaign_id,'branch')%(2**31-1)+1,grade_margin=grade_margin,
                 protocol=dict(warmups=warmups,cycles=cycles,validation_slots=validation_slots,test_slots=test_slots),
                 jobs=[])
    master, scenarios, identities=[],[],[]
    with TemporaryDirectory(prefix='.prepare-',dir=directory.parent) as temporary:
        staging=Path(temporary)/'package'
        (staging/'configs').mkdir(parents=True)

        def publish(name,data):
            (staging/name).write_bytes(data)
            return dict(path=str(directory/name),sha256=digest(data))

        for i, scene in enumerate(schedule(campaign_id,warmups,cycles,validation_slots,test_slots)):
            run=first_run+i
            if run > 2147483647:
                raise ValueError('physical run index exceeds baseline2 Java int')
            group=(scene['split'],scene['pair_ordinal']) if scene['role']=='eval' else ('train',i)
            env=seed(campaign_id,'env',*group)
            actor=seed(campaign_id,'sample',*group)
            rec=f'{campaign_id}-{run}'
            events=str(output/'anylogic_events'/f'run_{run}')
            initializer = suite['learner_seed'] if scene['arm']=='warmup' else suite['branch_seed']
            job=dict(scene,ordinal=i,run_index=run,recording_id=rec,source_recording_id=rec,actor_seed=initializer,
                     experiment_seed=env,actor_sampling_seed=actor,event_directory=events,
                     event_bundle=f'{events}/dispatch-production-wait-shortfall-v2-run-{run}-seed-{env}',
                     python_output=str(output/'shifts'/f'run_{run}'),group='BASELINE',
                     target_training_update=scene['update'] if scene['role']=='train' else 0,
                     checkpoint_update=scene['update']-(scene['role']=='train'))
            cfg=dict(schema=suite['schema'],campaign_id=campaign_id,job=job,rpc_source=suite['rpc_source'],
                     rpc_overrides=rpc_overrides(job),parent=suite['parent'],protected_core=suite['protected_core'])
            pin=publish(f'configs/run_{run}.json',encoded(cfg))
            job=dict(job,config_path=pin['path'], config_sha256=pin['sha256'])
            suite['jobs'].append(job)
            master_row=dict(job,role='train' if scene['role']=='train' else 'final')
            master.append([master_row[k] for k in MASTER_FIELDS])
            scenarios.append([run,'scene-'+str(run),*[int(r!='W') for r in scene['assignment']],*scene['grades'],
                              *scene['half_ranges'],base_rpc['config.runtime_scene.road.6.speedLimit_mps']])
            identities.append([job[k] for k in ('ordinal','phase','arm','split','pair_ordinal','condition','run_index',
                               'experiment_seed','actor_seed','actor_sampling_seed','recording_id','config_sha256','config_path','python_output','event_directory')])
        suite['master_plan']=publish('master_plan.csv',csv_bytes(MASTER_FIELDS,master))
        suite['scenario_table']=publish('scenario_table.csv',csv_bytes(SCENE_FIELDS,scenarios))
        cols='ordinal phase arm split pair_ordinal condition run_index experiment_seed actor_seed actor_sampling_seed recording_id config_sha256 config_path python_output event_directory'.split()
        suite['identities']=publish('run_seed.csv',csv_bytes(cols,identities))
        suite['java_helper']=publish('Baseline1CampaignPlan.java',render(suite,suite['scenario_table'],reference(PLAN/'launch.py')['sha256']))
        from resilience_budget import check_fresh
        suite['budget_at_preparation']=check_fresh(suite)
        seal=publish('suite.json',encoded(suite))
        publish('suite.sha256',(seal['sha256']+'\n').encode())
        os.rename(staging,directory)
    return seal


def load_suite(path, *, check_sources=True):
    path=Path(path).resolve()
    data=path.read_bytes()
    if digest(data)!=path.with_name('suite.sha256').read_text().strip():
        raise ValueError('suite seal mismatch')
    suite=json.loads(data)
    if str(path)!=suite['suite_path'] or suite['schema']!='bg128-fleet-resilience-v1':
        raise ValueError('wrong suite path or schema')
    for name in ('master_plan','scenario_table','identities','java_helper','protected_core','parent'):
        load_ref(suite[name])
    if check_sources:
        for ref in suite['sources']:
            load_ref(ref)
    base=json.loads(load_ref(suite['rpc_source']))['actual_rpc_configuration']
    for j in suite['jobs']:
        cfg=json.loads(load_ref(dict(path=j['config_path'],sha256=j['config_sha256'])))
        if (cfg['job']!={k:v for k,v in j.items() if k not in ('config_path','config_sha256')}
                or cfg['rpc_overrides']!=rpc_overrides(j)):
            raise ValueError('job and frozen configuration differ')
        j['expected_rpc_config']={**base,**cfg['rpc_overrides']}
    return suite


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('directory','campaign-id'):
        parser.add_argument('--'+key,required=True)
    for key in ('first-run','warmups','cycles','validation-slots','test-slots'):
        parser.add_argument('--'+key,required=True,type=int)
    parser.add_argument('--grade-margin',type=float,default=0.)
    print(json.dumps(prepare(**vars(parser.parse_args())),ensure_ascii=False))
