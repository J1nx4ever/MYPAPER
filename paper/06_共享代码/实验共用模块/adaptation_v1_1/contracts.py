"""Explicit, versioned gates for the user-approved fixed-Actor stage."""
import copy
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
COLLECTION='mining-fixed-actor-collection-v1.1'
TRAINING='mining-fixed-actor-critic-adaptation-v1.1'
FORMAL_REWARD=dict(kappa=4e-5,w_o=4.,w_w=4.,beta=.25,center=.27,scale=.01,gamma_reward=1,gamma_grade=1)
NETWORK=dict(hidden_sizes=[128,128],activation='relu',initialization='orthogonal_sqrt2_hidden_unit_output',risk_output='sigmoid')
ADAPTATION=dict(epochs=20,minibatch_size=256,learning_rate=1e-4,adam_eps=1e-8,reward_loss='mse',grade_loss='bce',huber_delta=1.,
                reward_value_coef=1.,grade_value_coef=1.,max_grad_norm=.5,grade_loss_aggregation='mean_head_means')


def require(ok,message):
    if not ok:raise ValueError(message)
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def path_of(value):
    p=(ROOT/value).resolve();require(p.is_relative_to(ROOT/'runs') and p!=ROOT/'runs','dedicated runs path required');return p
def pin(spec):
    p=Path(spec['path']);require(p.is_absolute() and sha(p)==spec['sha256'],'source reference changed: '+str(p));return p


def validate_campaign(campaign):
    require(campaign['contract']=='mining-fixed-actor-campaign-v1.1','wrong campaign')
    entries=campaign['collections'];require(len(entries)==6,'six collection partitions required')
    indices=set();identities=set();rng=set();groups={};actor_parts={};all_rows=[]
    for e in entries:
        require(e['partition'] in ['train','validation'],'test sealed')
        key=(e['actor_seed'],e['partition']);require(key not in actor_parts,'duplicate actor partition')
        jobs=read(pin(e['jobs']));require(len(jobs)==(32 if e['partition']=='train' else 16),'approved shift budget differs')
        actor_parts[key]=jobs
        require(e['recording_id'] not in identities,'duplicate recording');identities.add(e['recording_id'])
        for j in jobs:
            require(j['actor_seed']==e['actor_seed'] and j['partition']==e['partition'] and j['recording_id']==e['recording_id'],'allocation identity differs')
            require(type(j['run_index']) is int and j['run_index']>0 and j['run_index'] not in indices,'duplicate/invalid run index');indices.add(j['run_index'])
            require(type(j['experiment_seed']) is int and j['experiment_seed']>0,'invalid realization seed')
            require(type(j['actor_sampling_seed']) is int and 0<=j['actor_sampling_seed']<2**63 and j['actor_sampling_seed'] not in rng,'duplicate/invalid Actor RNG');rng.add(j['actor_sampling_seed'])
            groups.setdefault(j['realization_group_id'],[]).append(j);all_rows.append(j)
    require({s for s,p in actor_parts}=={2026103101,2026103102,2026103103},'all three Actor seeds required')
    require(len(groups)==48 and len(all_rows)==144,'campaign group/count mismatch')
    for g,rows in groups.items():
        require(len(rows)==3 and len({r['actor_seed'] for r in rows})==3,'shared realization allocation differs')
        require(len({r['partition'] for r in rows})==len({r['experiment_seed'] for r in rows})==1,'realization group split leakage/seed mismatch')
    for seed in {s for s,p in actor_parts}:
        fit={j['experiment_seed'] for j in actor_parts[seed,'train']};val={j['experiment_seed'] for j in actor_parts[seed,'validation']}
        require(len(fit)==32 and len(val)==16 and not fit&val,'Actor train/validation overlap')
    return entries


def check_config(config_path,kind):
    require(kind in ['collection','training'],'explicit stage required')
    p=Path(config_path).resolve();cfg=read(p)
    from python.acceptance_recording import config_bytes
    require(p.read_bytes()==config_bytes(cfg),'canonical configuration bytes required')
    require(cfg.get('contract')==({'collection':COLLECTION,'training':TRAINING}[kind]) and cfg.get('approved') is True,'explicit new-stage approval required')
    decision=read(pin(cfg['decision']))
    require(decision.get('status')=='user_confirmed' and decision.get('user_statement')=='确定这份预算','confirmed new budget required')
    require(decision['actor_seeds']==[2026103101,2026103102,2026103103] and decision['new_unique_actor_trajectories']==144 and
            decision['fit_epochs_per_arm']==20 and decision['validation_model_queries_total']==12,'budget decision mismatch')
    require(not any(decision[k] for k in ['actor_updates_authorized','ppo_updates_authorized','multiplier_updates_authorized','test_business_access_authorized']),'wrong stage permissions')
    require(cfg['network']==NETWORK and cfg['formal_reward']==FORMAL_REWARD and cfg['cpu_threads']==2,'frozen model/reward/runtime mismatch')
    for f,h in cfg['source_pins'].items():require(sha(f)==h,'source changed: '+f)
    init=read(pin(cfg['initialization_manifest']));require(init['status']=='prepared_only' and not init['training_result'],'initialization provenance required')
    pairs={e['seed']:e for e in init['pairs']}
    campaign=read(pin(cfg['campaign']));entries=validate_campaign(campaign)
    require(campaign['decision']==cfg['decision'],'campaign decision differs')
    def actor_bundle(seed,bundle,actor_digest):
        require(seed in pairs,'unapproved Actor seed');expected=pairs[seed]
        require(str(pin(bundle))==expected['path'] and bundle['sha256']==expected['sha256'] and actor_digest==expected['actor_digest'],'Actor initialization differs')
    if kind=='collection':
        require(decision['collection_authorized'],'collection not authorized')
        require(cfg['partition'] in ['train','validation'],'test sealed')
        matching=[e for e in entries if e['recording_id']==cfg['recording']['recording_id']]
        require(len(matching)==1,'unknown collection identity');entry=matching[0]
        require(cfg['actor_seed']==entry['actor_seed'] and cfg['partition']==entry['partition'] and cfg['jobs']==read(pin(entry['jobs'])),'jobs differ from approved campaign')
        require(cfg['output']==entry['output'] and cfg['admission_output']==entry['admission_output'],'output identity differs')
        require(path_of(cfg['output'])!=path_of(cfg['admission_output']),'raw/derived outputs must be independent')
        require(cfg['full_evidence_recording'] is True and cfg['record_bc_dataset'] is False and cfg['capture_rollout_snapshots'] is False,'explicit first-Actor-chain audit recording required')
        require(cfg['callback_timeout_seconds']==1800 and cfg['host']=='127.0.0.1' and cfg['port']==50051,'unexpected communication settings')
        require(cfg['behavior']==dict(name='frozen_actor_private_rng',version='1',actor_digest=cfg['behavior']['actor_digest']),'wrong behavior contract')
        actor_bundle(cfg['actor_seed'],cfg['actor_bundle'],cfg['behavior']['actor_digest'])
        from python.collect_offline import verify_environment
        from python.protocol_artifacts import verify_protocol_artifacts
        verify_protocol_artifacts();verify_environment(cfg['environment_acceptance'],cfg['reward'])
        reference=read(pin(cfg['runtime_reference']))
        for job in cfg['jobs']:
            expected=job['expected_rpc_config'];physical={k:v for k,v in expected.items() if k.startswith('config.runtime_scene.')}
            require(physical==reference['expected_physical_rpc_fields'] and job['runtime_scene_sha256']==reference['runtime_scene_sha256'],'physical scenario changed')
            require(not any(k.startswith('config.offline_native.') for k in expected),'native echo not Actor path')
            require(job['event_bundle']==str(path_of(entry['event_root'])/f"dispatch-production-wait-shortfall-v2-run-{job['run_index']}-seed-{job['experiment_seed']}"),'event identity differs')
    else:
        require(decision['critic_adaptation_authorized_after_new_data_admission'],'Critic adaptation not authorized')
        require(cfg['adaptation']==ADAPTATION and len(cfg['runs'])==3 and [e['seed'] for e in cfg['runs']]==decision['actor_seeds'],'training budget differs')
        require(cfg['multipliers']==dict(limits=[.05]*4,state_not_instantiated_this_stage=True,updates=0),'multiplier metadata differs')
        require(cfg['evaluation']==dict(risk_neighborhood=[.025,.075],calibration_edges=[i/10 for i in range(11)],bootstrap_replicates=2000,bootstrap_seed=2026110102),'evaluation protocol differs')
        for e in cfg['runs']:
            actor_bundle(e['seed'],e['bundle'],e['actor_digest']);path_of(e['output'])
            for partition in ['train','validation']:
                ref=e['train_config' if partition=='train' else 'validation_config'];source=read(pin(ref))
                require(source['contract']==COLLECTION and source['actor_seed']==e['seed'] and source['partition']==partition,'wrong adaptation data source')
                require(path_of(e[partition+'_directory'])==path_of(source['admission_output']),'derived data directory mismatch')
        path_of(cfg['review_output'])
    return copy.deepcopy(cfg)
