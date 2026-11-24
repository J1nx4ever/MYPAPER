"""Pure offline collection: no network, optimizer, PPO, GAE or multiplier update.

run seals a candidate; finalize independently joins physical evidence before
admission. Neither command starts AnyLogic. Full audit journaling is opt-in.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
from .constrained_adapter import require
from .constrained_batch import EpisodeBatchBuffer, collect_batch, load_dataset, sha, write_json
from .acceptance_recording import BoundaryRecorder, config_bytes, validate_recording_options
from .experiment_config import ROOT, SECTIONS, ConfigurationError, source_pins
from .protocol_artifacts import verify_protocol_artifacts

CONTRACT = 'mining-offline-collection-v1'
ENV_CHECKS = {'reward_ledger','window_material_fifo','termination_masks_mc'}
ENV_SOURCES = ('constrained_adapter.py','constrained_batch.py','reward_contract.py','reward_config.py','dispatch_bridge.py',
               'dispatch_env.py','observation_contract.py','scenario_contract.py','observation_schema.json','native_behavior.py',
               'runtime_scenario.py')
NORMAL_FIXTURE = {'config.acceptance.scenario_version':1., 'config.acceptance.scenario_code':0.,
                  'config.acceptance.load_not_before_seconds':14500.}


def environment_pins():
    return {'python/'+n:sha(ROOT/'python'/n) for n in ENV_SOURCES}


def verify_environment(spec, reward, *, storage_migration=False):
    require(isinstance(spec,dict) and spec.get('path') and spec.get('sha256'),'missing environment acceptance')
    p=Path(spec['path']);require(sha(p)==spec['sha256'],'environment certificate hash mismatch')
    a=json.loads(p.read_text())
    require(a.get('scope')=='actual_anylogic_environment_three_criteria' and a.get('status')=='passed',
            'actual three-criteria environment evidence required; synthetic/unit-test evidence cannot authorize collection or training')
    require(set(a.get('passed_checks',[]))==ENV_CHECKS,'incomplete environment acceptance')
    from .storage_compatibility import source_pins_match
    require(source_pins_match(a.get('source_sha256'),environment_pins(),storage_migration=storage_migration),
            'environment implementation changed; review compatibility')
    require(a.get('reward_settings')==reward,'reward semantics differ from reviewed evidence')
    require(bool(a.get('evidence')),'missing environment evidence')
    for item in a['evidence']:
        require(sha(item['path'])==item['sha256'],'environment evidence changed')
    return a


def validate_collection_config(cfg):
    require(isinstance(cfg,dict) and cfg.get('contract')==CONTRACT,'offline collection contract required')
    require(cfg.get('collection_approved') is True,'collection_approved: explicit collection settings required')
    r=cfg.get('reward',{})
    for k in SECTIONS['reward']:require(r.get(k) is not None,'missing reward.'+k)
    for k in SECTIONS['reward'][:8]:
        require(type(r[k]) in (float,int) and math.isfinite(r[k]) and r[k]>=0,'invalid reward.'+k)
    require(r['kappa']>0 and 0<=r['grade_lower']<=r['grade_upper']<=1,'invalid reward bounds')
    require(r['wait_definition']=='load_dump_resource_including_crusher_capacity' and
            r['cutoff_rule']=='exclude_loading_unloading_at_cutoff_integrate_crushing_to_cutoff' and
            r['simultaneous_events']=='LIFO','unsupported reward semantics')
    b=cfg.get('behavior',{})
    if b.get('name')=='anylogic_native':
        from .native_behavior import validate_spec
        validate_spec(b)
    else:
        require(set(b)=={'name','version','epsilon','seed'} and b['name']=='first_legal_epsilon_uniform'
                and b['version']=='1','explicit supported behavior/version required')
        require(type(b['epsilon']) in (float,int) and math.isfinite(b['epsilon']) and 0<=b['epsilon']<=1,'epsilon must be in [0,1]')
        require(type(b['seed']) is int and b['seed']>=0,'explicit behavior RNG seed required')
    jobs=cfg.get('jobs');require(isinstance(jobs,list) and bool(jobs),'explicit jobs required')
    groups={};seeds=set();indices=set()
    for j in jobs:
        require(type(j.get('run_index')) is int and j['run_index']>0 and
                type(j.get('experiment_seed')) is int and j['experiment_seed']>0,'invalid run/seed')
        require(j['run_index'] not in indices and j['experiment_seed'] not in seeds,'duplicate run/seed')
        indices.add(j['run_index']);seeds.add(j['experiment_seed'])
        require(all(isinstance(j.get(k),str) and j[k].strip() for k in ('scene_id','scene_group','event_bundle')),'missing scene/event path')
        require(Path(j['event_bundle']).is_absolute(),'event_bundle must be an absolute path')
        require(j.get('partition') in ('train','validation','test'),'predefined partition required')
        require(groups.setdefault(j['scene_group'],j['partition'])==j['partition'],'scene group split leakage')
        expected=j.get('expected_rpc_config',{})
        required={'config.production_reward_version','config.shift_seconds','config.window_seconds',
                  'config.target_ore_tons_shift','config.target_waste_tons_shift','config.waste_task_limit_enabled',
                  'config.waste_task_limit_version','config.waste_task_upper_tons','config.waste_task_overrun_tons',
                  'config.num_trucks','config.target_grade',*NORMAL_FIXTURE}
        require(required<=set(expected),'missing explicit runtime task/scenario/fixture settings')
        require(all(k.startswith('config.') and type(v) in (int,float) and math.isfinite(v) for k,v in expected.items()),'invalid expected RPC configuration')
        require(all(expected[k]==v for k,v in NORMAL_FIXTURE.items()),'normal collection must disable boundary fixtures explicitly')
        if b.get('name')=='anylogic_native':
            from .native_behavior import runtime_fields, PREFIX
            from .runtime_scenario import PREFIX as SCENE_PREFIX, fingerprint, validate_runtime_fields
            require({k:v for k,v in expected.items() if k.startswith(PREFIX)}==runtime_fields(b),
                    'job must declare native runtime identity')
            scene={k:v for k,v in expected.items() if k.startswith(SCENE_PREFIX)}
            validate_runtime_fields(scene,scene)
            require(j.get('runtime_scene_sha256')==fingerprint(scene),
                    'job must bind actual runtime scene fields; design file is not runtime evidence')
        require(expected['config.shift_seconds']==28800 and expected['config.window_seconds']==7200 and
                expected['config.target_ore_tons_shift']==16000 and expected['config.target_waste_tons_shift']==4000 and
                expected['config.num_trucks']==9,'task differs from frozen baseline')
    validate_recording_options(cfg,enabled=True)
    require(cfg.get('full_evidence_recording') is True,'explicit full_evidence_recording required by current event-backed admission')
    require(type(cfg.get('callback_timeout_seconds')) in (int,float) and
            math.isfinite(cfg['callback_timeout_seconds']) and cfg['callback_timeout_seconds']>0,'explicit finite callback timeout required')
    from .offline_split import validate_jobs
    validate_jobs(cfg)
    return copy.deepcopy(cfg)


def preflight(cfg):
    cfg=validate_collection_config(cfg)
    if cfg['behavior']['name']=='anylogic_native':
        from .native_behavior import validate_spec
        validate_spec(cfg['behavior'],check_files=True)
    verify_protocol_artifacts();verify_environment(cfg.get('environment_acceptance'),cfg['reward'])
    return cfg


def make_behavior(spec):
    if spec['name']=='anylogic_native':
        from .native_behavior import NativeBehavior
        return NativeBehavior(spec)
    return RuleBehavior(spec)


class RuleBehavior:
    """Fixed rule distribution with a private RNG; no learned parameters."""
    def __init__(self,spec):
        self.spec=copy.deepcopy(spec)
        self.policy_version=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()+Path(__file__).read_bytes()).hexdigest()
        self.rng=np.random.default_rng(spec['seed'])
    def assert_frozen(self):
        require(self.policy_version==hashlib.sha256(json.dumps(self.spec,sort_keys=True).encode()+Path(__file__).read_bytes()).hexdigest(),'behavior changed')
    def sample(self,obs,mask):
        self.assert_frozen()
        require(np.array_equal(np.asarray(obs)[19:23],mask),'behavior mask differs from stored network input')
        legal=np.flatnonzero(mask);require(len(legal)>0,'no legal action')
        probs=np.full(len(legal),self.spec['epsilon']/len(legal));probs[0]+=1-self.spec['epsilon']
        i=int(self.rng.choice(len(legal),p=probs))
        return int(legal[i]),float(np.log(probs[i]))


def validate_runtime(info,job):
    c=info['reward_components']
    if 'runtime_scene_sha256' in job:
        from .runtime_scenario import validate_runtime_fields, PREFIX
        validate_runtime_fields(c,{k:v for k,v in job['expected_rpc_config'].items() if k.startswith(PREFIX)})
    require(all(k in c and c[k]==v for k,v in job['expected_rpc_config'].items()),'RPC differs from explicit job configuration')
    require({k:v for k,v in c.items() if k.startswith('config.acceptance.')}==NORMAL_FIXTURE,
            'unknown/missing/enabled acceptance fixture; reject normal dataset')


def run(cfg,output,*,host='127.0.0.1',port=50051):
    cfg=preflight(cfg)  # before any listener/output/model creation
    from .dispatch_bridge import DispatchCallbackBridge
    from .dispatch_env import DispatchEnv
    from .dispatch_service import create_server
    from .episode_runner import ParameterVariationEpisodeRunner
    output=Path(output);require(not output.exists(),'refusing output reuse')
    for old in (ROOT/'runs').glob('**/recording_manifest.json'):
        require(json.loads(old.read_text()).get('recording_id')!=cfg['recording']['recording_id'],
                'recording_id already used in project runs')
    for j in cfg['jobs']:require(not Path(j['event_bundle']).exists(),'refusing existing event identity')
    # Java creates only the fresh episode leaf with CREATE_NEW. Prepare its
    # parent here; never pre-create the leaf or replace an existing recording.
    for j in cfg['jobs']:Path(j['event_bundle']).parent.mkdir(parents=True,exist_ok=True)
    output.mkdir(parents=True)
    write_json(output/'acceptance_run_config.json',cfg)  # existing journal format, exact config bytes
    write_json(output/'source_snapshot.json',source_pins())
    recorder=BoundaryRecorder(output,cfg,cfg['jobs'],source_kind='actual_anylogic_offline',source_sha256=source_pins())
    buffer=EpisodeBatchBuffer(output/'candidate',expected_episodes=len(cfg['jobs']),source_kind='real_anylogic_rpc')
    bridge=DispatchCallbackBridge(runtime_configuration=True,expected_episode_metadata=[(j['run_index'],j['experiment_seed']) for j in cfg['jobs']],evidence_sink=recorder)
    env=DispatchEnv(bridge,ParameterVariationEpisodeRunner(len(cfg['jobs'])),cfg['callback_timeout_seconds'])
    server=None
    try:
        server=create_server(bridge,host=host,port=port);server.start()
        print(f'LISTENING {host}:{port}; offline collection only; start experimental AnyLogic manually',flush=True)
        behavior=make_behavior(cfg['behavior'])
        collect_batch(env,behavior,buffer,cfg['jobs'],reward_settings=cfg['reward'],recorder=recorder,runtime_validator=validate_runtime)
        server.stop(grace=2).wait();server=None
        recorder.complete()
        write_json(output/'collection_summary.json',dict(status='candidate_requires_event_finalization',
                   episodes=len(buffer.episodes),transitions=len(buffer.rows),policy_version=behavior.policy_version,
                   behavior=cfg['behavior'],rng_final=behavior.rng.bit_generator.state if hasattr(behavior,'rng') else None,
                   model_updates=0,multiplier_updates=0))
    except BaseException as exc:
        env.close()
        if server is not None:server.stop(grace=2).wait();server=None
        buffer.abort(exc);recorder.abort(exc)
        write_json(output/'ABORTED.json',dict(error=str(exc),trainable=False))
        raise
    finally:
        env.close()
        if server is not None:server.stop(grace=2).wait()


def coverage(rows,episodes,jobs):
    stats={}
    for e,j in zip(episodes,jobs):
        key=j['partition'];s=stats.setdefault(key,dict(episodes=0,behavior_counts={},heads=[dict(compliant_episodes=0,violation_episodes=0,valid_samples=0,compliant_samples=0,violation_samples=0) for _ in range(4)]))
        s['episodes']+=1;s['behavior_counts'][e['policy_version']]=s['behavior_counts'].get(e['policy_version'],0)+1
        for h,cost in enumerate(e['window_costs']):
            n=sum(bool(r['grade_valid'][h]) for r in rows[e['start']:e['stop']]);d=s['heads'][h]
            kind='violation' if cost else 'compliant';d[kind+'_episodes']+=1;d[kind+'_samples']+=n;d['valid_samples']+=n
    return stats


def finalize(directory,*,allow_synthetic=False,identity_reconciliation=None):
    from .acceptance_batch_audit import audit_recording
    p=Path(directory);target=p/'offline_admission.json'
    require(not target.exists(),'refusing admission overwrite')
    require(not (p/'ABORTED.json').exists(),'aborted collection')
    reconciliation=None
    if identity_reconciliation is not None:
        from .offline_identity_reconciliation import validate
        reconciliation=validate(p,identity_reconciliation)
    cfg=json.loads((p/'acceptance_run_config.json').read_text());validate_collection_config(cfg)
    manifest=json.loads((p/'recording_manifest.json').read_text())
    synthetic=manifest.get('scope')=='synthetic_unit_test'
    require((synthetic and allow_synthetic) or manifest.get('scope')=='actual_anylogic_offline','acceptance/legacy recording cannot become formal data')
    if not synthetic:preflight(cfg)
    rows,episodes,yr,yg,m=load_dataset(p/'candidate',require_real=not synthetic)
    require(m['source_kind']==('synthetic_unit_test' if synthetic else 'real_anylogic_rpc'),'candidate source differs')
    jobs=cfg['jobs'];require(len(episodes)==len(jobs),'job count mismatch')
    raw=[json.loads(s) for s in (p/'transitions.jsonl').read_text().splitlines()]
    transitions=[r for r in raw if r['event']=='transition']
    require(len(transitions)==len(rows),'candidate/recording length differs')
    behavior=make_behavior(cfg['behavior'])
    for r,t in zip(rows,transitions):
        require(np.array_equal(r['observation'],np.asarray(t['observation'],dtype=np.float32)) and
                np.array_equal(r['next_observation'],np.asarray(t['next_observation'],dtype=np.float32)), 'candidate/wire observations differ')
        require(r['settlement_info']==t['next_info'] and r['action']==t['action'] and r['reward']==t['formal_reward'],'candidate/journal mismatch')
        if hasattr(behavior,'sample_with_info'):
            from .native_behavior import validate_row
            require(r.get('behavior_spec')==cfg['behavior'],'candidate native specification differs')
            validate_row(r,t['action_info'])
            a,lp=behavior.sample_with_info(r['observation'],r['mask'],t['action_info'])
        else:
            a,lp=behavior.sample(r['observation'],r['mask'])
        require(a==r['action'] and lp==r['old_log_prob'] and r['policy_version']==behavior.policy_version,'behavior reproduction failed')
    for e,j in zip(episodes,jobs):
        require(e['experiment_seed']==j['experiment_seed'] and e['experiment_run_index']==j['run_index'] and
                e['scene_group']==j['scene_group'] and e['scene_id']==j['scene_id'],'candidate job differs')
        starts=[t for t in raw if t['event']=='episode_start' and t['episode_id']==e['episode_id']]
        require(len(starts)==1 and starts[0]['job']==j and starts[0]['info']==e['start_info'] and
                np.array_equal(np.asarray(starts[0]['observation'],dtype=np.float32),e['initial_observation']),
                'candidate initial snapshot differs from journal')
        validate_runtime(e['start_info'],j)
        for r in rows[e['start']:e['stop']]:validate_runtime(r['settlement_info'],j)
    audit=audit_recording(p,[Path(j['event_bundle']) for j in jobs],allow_synthetic=synthetic,
                         identity_reconciliation=identity_reconciliation)
    # Independent physical arithmetic and journal replay both precede admission.
    require(audit['event_evidence_verified'] and audit['inputs_unchanged'],'event audit incomplete')
    files=[p/n for n in ('acceptance_run_config.json','recording_manifest.json','rpc.jsonl','transitions.jsonl')]
    files += [p/'candidate'/n for n in ('manifest.json','samples.npz','episodes.json')]
    if reconciliation is not None:files.append(Path(identity_reconciliation))
    if 'split_protocol' in cfg:files.append(Path(cfg['split_protocol']['manifest_path']))
    for j in jobs:files += [Path(j['event_bundle'])/n for n in ('events_manifest.json','events.jsonl')]
    if cfg['behavior']['name']=='anylogic_native':
        from .native_behavior import validate_spec
        validate_spec(cfg['behavior'],check_files=True)
        files += [Path(a['path']) for a in cfg['behavior']['artifacts']]
    write_json(p/'offline_audit.json',audit)
    files.append(p/'offline_audit.json')
    report=dict(contract=CONTRACT,status='admitted',source_kind='synthetic_unit_test' if synthetic else 'real_anylogic_rpc',
                formal_training_data=not synthetic,dataset=str((p/'candidate').resolve()),config_sha256=sha(p/'acceptance_run_config.json'),
                evidence_sha256={str(f.resolve()):sha(f) for f in files},jobs=jobs,behavior=cfg['behavior'],
                coverage=coverage(rows,episodes,jobs),model_updates=0,multiplier_updates=0)
    if 'split_protocol' in cfg:report['split_protocol']=copy.deepcopy(cfg['split_protocol'])
    if reconciliation is not None:
        # Recheck the frozen source after the physical audit as well as before
        # it. A concurrent source/capsule change cannot be sealed by this path.
        from .offline_identity_reconciliation import validate
        require(validate(p,identity_reconciliation)==reconciliation,
                'identity reconciliation changed during finalization')
        # The old source still fails strict identity verification. Only this
        # independent, unchanged-byte capsule has an authorized association.
        report.update(admission_mode='recording_identity_reconciled',strict_identity_passed=False,
                      alias_review_passed=True,
                      identity_reconciliation=dict(path=str(Path(identity_reconciliation).resolve()),
                                                   sha256=sha(identity_reconciliation)),
                      identity_reconciliation_auditor_sha256=sha(Path(__file__).with_name('offline_identity_reconciliation.py')))
    if cfg['behavior']['name']=='anylogic_native':
        report['native_verification']='wire proposal/response/applied action agreement; not a rerun of simulator rules'
    write_json(target,report)
    return report


def load_admitted(directory,*,require_real=True,partition=None):
    p=Path(directory)
    if p.name=='candidate':p=p.parent
    a=json.loads((p/'offline_admission.json').read_text())
    require(a.get('contract')==CONTRACT and a.get('status')=='admitted','offline data not admitted')
    require(a.get('dataset')==str((p/'candidate').resolve()),'admission dataset location mismatch')
    require(not (p/'ABORTED.json').exists(),'aborted collection')
    if require_real:require(a.get('source_kind')=='real_anylogic_rpc' and a.get('formal_training_data') is True,'synthetic/acceptance data cannot train')
    mode=a.get('admission_mode','strict')
    require(mode in ('strict','recording_identity_reconciled'),'unknown offline admission mode')
    descriptor=None
    if mode=='recording_identity_reconciled':
        from .offline_identity_reconciliation import validate, SIDECAR
        descriptor=p/SIDECAR
        require(a.get('strict_identity_passed') is False and a.get('alias_review_passed') is True,
                'reconciliation must preserve the strict identity failure')
        require(a.get('identity_reconciliation')==dict(path=str(descriptor.resolve()),sha256=sha(descriptor)),
                'identity reconciliation evidence changed')
        require(a.get('identity_reconciliation_auditor_sha256')==sha(Path(__file__).with_name('offline_identity_reconciliation.py')),
                'identity reconciliation auditor changed; explicit compatibility review required')
        validate(p,descriptor)
        review=json.loads((p/'offline_audit.json').read_text())
        require(review.get('identity_reconciliation_used') is True and
                review.get('strict_identity_passed') is False and
                review.get('event_evidence_verified') is True and review.get('inputs_unchanged') is True,
                'identity reconciliation physical audit incomplete')
    else:
        require(not (p/'identity_reconciliation.json').exists() and
                'identity_reconciliation' not in a,'reconciliation requires an explicit admission mode')
    required=[p/n for n in ('acceptance_run_config.json','recording_manifest.json','rpc.jsonl','transitions.jsonl','offline_audit.json')]
    required += [p/'candidate'/n for n in ('manifest.json','samples.npz','episodes.json')]
    if descriptor is not None:required.append(descriptor)
    if 'split_protocol' in a:
        from .offline_split import load_manifest
        load_manifest(a['split_protocol'])
        required.append(Path(a['split_protocol']['manifest_path']))
    for j in a['jobs']:required += [Path(j['event_bundle'])/n for n in ('events_manifest.json','events.jsonl')]
    if a['behavior']['name']=='anylogic_native':
        required += [Path(item['path']) for item in a['behavior']['artifacts']]
    require(set(a['evidence_sha256'])=={str(f.resolve()) for f in required},'admission evidence inventory incomplete')
    for f,h in a['evidence_sha256'].items():require(sha(f)==h,'offline evidence changed: '+f)
    rows,episodes,yr,yg,m=load_dataset(p/'candidate',require_real=require_real)
    cfg=json.loads((p/'acceptance_run_config.json').read_text())
    validate_collection_config(cfg)
    require(a['jobs']==cfg['jobs'] and a['behavior']==cfg['behavior'],'admission/config mismatch')
    require(a.get('split_protocol')==cfg.get('split_protocol'),'admission split/config mismatch')
    require(len(episodes)==len(cfg['jobs']),'admission job count mismatch')
    from .offline_split import group_assignments
    group_assignments(episodes,a['jobs'],a.get('split_protocol'),admission_path=p/'offline_admission.json')
    for e,j in zip(episodes,cfg['jobs']):
        validate_runtime(e['start_info'],j)
        for r in rows[e['start']:e['stop']]:validate_runtime(r['settlement_info'],j)
    if partition is None:return rows,episodes,yr,yg,m
    require(partition in ('train','validation','test'),'invalid partition')
    selected=[];ep=[];rr=[];gg=[]
    for e,j in zip(episodes,a['jobs']):
        if j['partition']!=partition:continue
        item=copy.deepcopy(e);start=len(selected);selected.extend(rows[e['start']:e['stop']])
        rr.extend(yr[e['start']:e['stop']]);gg.extend(yg[e['start']:e['stop']])
        item.update(start=start,stop=len(selected));ep.append(item)
    require(bool(ep),'empty requested partition')
    return selected,ep,np.asarray(rr),np.asarray(gg),m


def audit_catalog(paths,*,require_real=True,include_test_labels=False):
    """Check every partition; keep test labels sealed unless explicitly released.

    This output gate prevents accidental inspection during model selection. It
    is not an access-control boundary and never weakens admission or split checks.
    """
    owners={k:{} for k in ('episode_id','experiment_seed','scene_group')}
    counts={};identities=set();breakdown=[];protocols={}
    for path in paths:
        p=Path(path);p=p.parent if p.name=='candidate' else p
        rows,episodes,_,_,_=load_admitted(p,require_real=require_real)
        a=json.loads((p/'offline_admission.json').read_text())
        from .offline_split import group_assignments
        groups=group_assignments(episodes,a['jobs'],a.get('split_protocol'),admission_path=p/'offline_admission.json')
        if 'split_protocol' in a:
            spec=a['split_protocol'];protocols[spec['manifest_sha256']]=spec
        for scene in sorted({j['scene_id'] for j in a['jobs']}):
            paired=[(e,j) for e,j in zip(episodes,a['jobs']) if j['scene_id']==scene]
            statistics=coverage(rows,[e for e,j in paired],[j for e,j in paired])
            if not include_test_labels and 'test' in statistics:
                statistics['test'].pop('heads')
                statistics['test']['labels_sealed']=True
            breakdown.append(dict(dataset=str(p.resolve()),behavior=a['behavior'],scene_id=scene,
                                  admission_mode=a.get('admission_mode','strict'),
                                  partitions=statistics))
        for e,j,group in zip(episodes,a['jobs'],groups):
            require(e['episode_id'] not in identities,'duplicate episode across catalog')
            identities.add(e['episode_id']);part=j['partition']
            for key in owners:
                value=group if key=='scene_group' else e[key]
                require(value not in owners[key] or (key=='scene_group' and owners[key][value]==part),
                        'catalog leakage/duplicate: '+key)
                owners[key][value]=part
            counts[part]=counts.get(part,0)+1
    return dict(partition_episode_counts=counts,owners=owners,coverage_by_behavior_scene=breakdown,
                split_protocols=list(protocols.values()),
                test_labels_included=bool(include_test_labels),
                sealed_test_episodes=0 if include_test_labels else counts.get('test',0),
                complete_three_way_split=set(counts)=={'train','validation','test'})


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=('preflight','run','finalize','describe-native','catalog'));ap.add_argument('--config',type=Path)
    ap.add_argument('--output',type=Path);ap.add_argument('--host',default='127.0.0.1');ap.add_argument('--port',type=int,default=50051)
    ap.add_argument('--dataset',type=Path,action='append')
    ap.add_argument('--include-test',action='store_true',help='release catalog test labels after model selection is frozen')
    a=ap.parse_args(argv)
    try:
        if a.command=='catalog':
            require(bool(a.dataset),'catalog requires --dataset')
            print(json.dumps(audit_catalog(a.dataset,include_test_labels=a.include_test),ensure_ascii=False,indent=2));return 0
        if a.command=='finalize':
            require(a.output is not None,'finalize requires --output');finalize(a.output);print('Offline data admitted; no training');return 0
        require(a.config is not None,'config required');cfg=json.loads(a.config.read_text())
        if a.command=='describe-native':
            from .native_behavior import validate_spec,identity,runtime_fields
            validate_spec(cfg['behavior'],check_files=True)
            print(json.dumps(dict(offlineBehaviorSha256=identity(cfg['behavior']),
                                  expected_rpc_config=runtime_fields(cfg['behavior']),
                                  collection_started=False),ensure_ascii=False,indent=2));return 0
        preflight(cfg)
        require(a.config.read_bytes()==config_bytes(cfg),'config must use canonical exact saved bytes')
        if a.command=='preflight':print('Offline collection preflight passed; no listener or updates');return 0
        require(a.output is not None,'run requires --output');run(cfg,a.output,host=a.host,port=a.port)
    except (ValueError,RuntimeError,OSError,KeyError,TypeError) as exc:
        print('OFFLINE REFUSED: '+str(exc),file=sys.stderr);return 2
    return 0

if __name__=='__main__':raise SystemExit(main())
