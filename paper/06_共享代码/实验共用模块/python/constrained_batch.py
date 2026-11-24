"""Complete-shift storage, independent window labels, and immutable actor sampling."""
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .constrained_adapter import RewardCostAdapter, TruncatedShift, require
from .reward_config import configuration_from_components
from .observation_contract import validate_observation, OBSERVATION_SCHEMA_ID
from .storage_contract import HDF5, LEGACY, storage_options

DATA_CONTRACT='mining-complete-shift-mc-v1'


def write_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def dataset_source_pins():
    """Pin executable data readers/schema/protocol without requiring local ALP."""
    from .experiment_config import ROOT
    paths=[ROOT/'proto-src/src/main/proto/dispatch.proto',ROOT/'dist/protocol_manifest.json',
           *sorted((ROOT/'python').glob('*.py')),ROOT/'python/observation_schema.json']
    return {str(path.relative_to(ROOT)):sha(path) for path in paths}


class FrozenActor:
    def __init__(self,actor):
        from .constrained_ppo import digest
        self.actor=copy.deepcopy(actor).cpu().eval()
        for p in self.actor.parameters():p.requires_grad_(False)
        self.policy_version=digest(self.actor)

    def assert_frozen(self):
        from .constrained_ppo import digest
        require(digest(self.actor)==self.policy_version and not self.actor.training and
                not any(p.requires_grad for p in self.actor.parameters()),'sampling actor changed')

    def sample(self,obs,mask):
        self.assert_frozen()
        require(np.array_equal(np.asarray(obs)[19:23],mask),'sampling mask differs from network input')
        with torch.no_grad():
            dist=self.actor.distribution(torch.as_tensor(np.asarray(obs).copy(),dtype=torch.float32)[None,:])
            action=dist.sample()
        return int(action.item()),float(dist.log_prob(action).item())


def mc_labels(rows,episodes):
    yr=np.zeros(len(rows),dtype=np.float64);yg=np.zeros((len(rows),4),dtype=np.float64)
    for e in episodes:
        carry=0.
        for t in range(e['stop']-1,e['start']-1,-1):
            carry+=rows[t]['reward'];yr[t]=carry
            yg[t]=np.where(rows[t]['grade_valid'],e['window_costs'],0.)
    return yr,yg


def validate_rows(rows,episodes):
    require(bool(rows) and bool(episodes),'empty complete-shift dataset')
    cursor=0;ids=set();seeds=set();versions=set()
    for e in episodes:
        require(e['start']==cursor and e['stop']>cursor and e['stop']<=len(rows),'episode boundary gap/overlap')
        require(e['normal_completion'] is True and len(e['window_costs'])==4,'incomplete shift')
        require(e['episode_id'] not in ids and e['experiment_seed'] not in seeds,'duplicate episode/seed')
        ids.add(e['episode_id']);seeds.add(e['experiment_seed'])
        require(e.get('scene_id') and e.get('scene_group') and e.get('scenario_fingerprint'),'missing scene/scenario provenance')
        start_info=e['start_info'];cfg=configuration_from_components(start_info['reward_components'])
        from .scenario_contract import from_components
        scenario=from_components(start_info['reward_components'])
        require(scenario is not None and scenario.fingerprint==e['scenario_fingerprint'],'runtime scenario mismatch')
        adapter=RewardCostAdapter(cfg);adapter.start(start_info)
        previous=np.asarray(e['initial_observation'],dtype=np.float32)
        action_info=start_info
        for t in range(cursor,e['stop']):
            r=rows[t];versions.add(r['policy_version'])
            require(r['episode_id']==e['episode_id'] and r['experiment_seed']==e['experiment_seed'] and
                    r['scene_id']==e['scene_id'] and r['scene_group']==e['scene_group'], 'row provenance mismatch')
            require(r['policy_version']==e['policy_version'],'mixed policy within shift')
            require(r['action_time_seconds']==adapter.time,'action time differs from decision state')
            require(r['scenario_fingerprint']==e['scenario_fingerprint'],'row scenario differs from shift')
            require(not r['abnormal_interruption'] and not r['sampler_truncated'] and not r['truncated'],
                    'abnormal/truncated shift is not trainable')
            require(r['terminated']==(t==e['stop']-1),'terminal at wrong episode position')
            require(np.array_equal(previous,r['observation']),'discontinuous states')
            validate_observation(OBSERVATION_SCHEMA_ID,r['observation'],terminal=False,truck_id=r['truck_id'])
            validate_observation(OBSERVATION_SCHEMA_ID,r['next_observation'],terminal=r['terminated'],truck_id=r['next_truck_id'])
            require(np.array_equal(r['mask'],np.asarray(r['observation'])[19:23]),'stored mask mismatch')
            require(type(r['action']) is int and 0<=r['action']<4 and r['mask'][r['action']]==1,'illegal stored action')
            if r.get('behavior_kind') == 'anylogic_native':
                from .native_behavior import validate_row
                validate_row(r,action_info)
            else:
                require(r['old_log_prob'] is not None and np.isfinite(r['old_log_prob']) and r['old_log_prob']<=0,
                        'invalid old log probability')
            require(r['settlement_info']['previous_action']==r['action'],'settlement action mismatch')
            expected=adapter.transition(r['settlement_info'],terminated=r['terminated'],truncated=r['truncated'])
            for key in expected:
                require(np.array_equal(expected[key],r[key]),'stored transition/target boundary mismatch: '+key)
            previous=np.asarray(r['next_observation'],dtype=np.float32)
            action_info=r['settlement_info']
        costs=[float(w['violation_upper'] or w['violation_lower']) for w in adapter.windows]
        require(costs==e['window_costs'],'episode final window costs disagree')
        require(e['windows']==adapter.windows,'episode window material summaries disagree')
        cursor=e['stop']
    require(cursor==len(rows),'unassigned rows')
    return versions


class EpisodeBatchBuffer:
    def __init__(self,output_dir=None,*,expected_episodes,source_kind,
                 storage_format=LEGACY,chunk_rows=64,provenance=None):
        require(type(expected_episodes) is int and expected_episodes>0,'explicit episode count required')
        require(source_kind in ('real_anylogic_rpc','synthetic_unit_test'),'explicit source_kind required')
        self.output_dir=Path(output_dir) if output_dir is not None else None
        if self.output_dir is not None:self.output_dir.mkdir(parents=True,exist_ok=False)
        self.expected_episodes,self.source_kind=expected_episodes,source_kind
        self.rows=[];self.episodes=[];self.complete=False
        self.storage = storage_options(dict(format=storage_format, chunk_rows=chunk_rows))
        require(provenance is None or isinstance(provenance,dict),'invalid dataset provenance')
        self.provenance = copy.deepcopy(provenance)
        self._journal = None
        if self.output_dir is not None and storage_format == HDF5:
            from .hdf5_storage import HDF5RowJournal
            self._journal = HDF5RowJournal(self.output_dir / '.samples.partial.h5', chunk_rows=chunk_rows)
            self.rows = self._journal

    def append(self,row):
        require(not self.complete,'sealed batch')
        r=copy.deepcopy(row)
        for key in ('observation','next_observation'):r[key]=np.asarray(r[key],dtype=np.float32).copy()
        self.rows.append(r)

    def finish_episode(self,start,*,start_info,initial_observation,job,scenario_fingerprint):
        require(len(self.rows)>start,'incomplete episode')
        last=self.rows[-1]
        require(last['terminated'] and not last['truncated'],'incomplete episode')
        windows=[w for index in range(start,len(self.rows))
                 for w in self.rows[index]['settlement_info']['closed_windows']]
        require([w['window_index'] for w in windows]==list(range(4)),'missing independent windows')
        e=dict(start=start,stop=len(self.rows),episode_id=start_info['episode_id'],experiment_seed=start_info['experiment_seed'],
               experiment_run_index=start_info['experiment_run_index'],policy_version=self.rows[start]['policy_version'],
               normal_completion=True,window_costs=[float(w['violation_upper'] or w['violation_lower']) for w in windows],
               windows=windows,start_info=copy.deepcopy(start_info),initial_observation=np.asarray(initial_observation).tolist(),
               scene_id=job['scene_id'],scene_group=job['scene_group'],scenario_fingerprint=scenario_fingerprint)
        self.episodes.append(e)

    def seal(self):
        require(not self.complete and len(self.episodes)==self.expected_episodes,'incomplete batch')
        if self._journal is not None:
            self._journal.flush()
        versions=validate_rows(self.rows,self.episodes)
        require(len(versions)==1,'online batch mixes policy versions')
        self.mc_reward,self.mc_grade=mc_labels(self.rows,self.episodes)
        self.complete=True
        if self.output_dir is not None:self.save()

    def save(self):
        require(self.complete,'cannot save incomplete data as trainable')
        p=self.output_dir
        require(not (p/'samples.npz').exists() and not (p/'samples.h5').exists()
                and not (p/'manifest.json').exists(),'refusing dataset overwrite')
        from .experiment_config import source_pins
        sample_name = 'samples.h5' if self.storage['format'] == HDF5 else 'samples.npz'
        if self.storage['format'] == HDF5:
            from .hdf5_storage import write_dataset
            write_dataset(p/sample_name,self.rows,self.episodes,self.mc_reward,self.mc_grade,
                          chunk_rows=self.storage['chunk_rows'])
        else:
            metadata=[{k:v for k,v in r.items() if k not in ('observation','next_observation')} for r in self.rows]
            np.savez_compressed(p/sample_name,observations=np.stack([r['observation'] for r in self.rows]),
                                next_observations=np.stack([r['next_observation'] for r in self.rows]),
                                metadata_json=np.array([json.dumps(r,allow_nan=False) for r in metadata]),
                                mc_reward=self.mc_reward,mc_grade=self.mc_grade)
        write_json(p/'episodes.json',self.episodes)
        write_json(p/'manifest.json',dict(contract=DATA_CONTRACT,status='complete',source_kind=self.source_kind,
                   episodes=len(self.episodes),transitions=len(self.rows),policy_version=self.rows[0]['policy_version'],
                   storage=self.storage,provenance=self.provenance,
                   source_sha256=dataset_source_pins() if self.storage['format']==HDF5 else source_pins(),
                   files_sha256={name:sha(p/name) for name in (sample_name,'episodes.json')}))
        if self.storage['format'] == HDF5:
            from .hdf5_storage import read_dataset
            if self._journal is not None:
                journal_path = self._journal.path
                self._journal.close()
                # Only this operation's temporary spool is removed after a complete seal.
                journal_path.unlink()
                self._journal = None
            self.rows, _, _, _ = read_dataset(p/sample_name)

    def abort(self,error):
        self.complete=False
        if self.output_dir is not None:
            write_json(self.output_dir/'ABORTED.json',dict(status='aborted',trainable=False,source_kind=self.source_kind,
                       completed_episodes=len(self.episodes),transitions=len(self.rows),error=str(error),
                       abnormal_interruption=True,
                       sampler_truncated=getattr(error,'sampler_truncated',isinstance(error,TruncatedShift)),
                       failure_context=getattr(error,'failure_context',None)))
        if self._journal is not None:
            self._journal.flush()
            self._journal.close()


@dataclass
class TrainingBatch:
    rows:list
    episodes:list
    policy_version:str
    batch_id:str

    @classmethod
    def from_buffer(cls,b):
        require(b.complete,'aborted/incomplete batch is not trainable')
        require(all(not r.get('critic_only',False) and r.get('behavior_kind') != 'anylogic_native'
                    and r['old_log_prob'] is not None for r in b.rows),
                'native Critic-only data cannot become an online PPO batch')
        versions=validate_rows(b.rows,b.episodes);require(len(versions)==1,'online data mixes policy versions')
        identity=json.dumps([e['episode_id'] for e in b.episodes])+next(iter(versions))
        return cls(copy.deepcopy(b.rows),copy.deepcopy(b.episodes),next(iter(versions)),hashlib.sha256(identity.encode()).hexdigest())

    @property
    def observations(self):return np.stack([r['observation'] for r in self.rows])
    @property
    def next_observations(self):return np.stack([r['next_observation'] for r in self.rows])
    @property
    def bounds(self):return [(e['start'],e['stop']) for e in self.episodes]
    def __getattr__(self,name):
        key={'rewards':'reward','actions':'action','old_log_probs':'old_log_prob'}.get(name,name)
        if key in ('reward','action','old_log_prob','grade_costs','grade_valid','grade_bootstrap','grade_trace','reward_bootstrap','reward_trace'):
            return np.asarray([r[key] for r in self.rows])
        raise AttributeError(name)


def load_dataset(path,*,require_real):
    p=Path(path);m=json.loads((p/'manifest.json').read_text())
    require(m['contract']==DATA_CONTRACT and m['status']=='complete' and not (p/'ABORTED.json').exists(),'incomplete/legacy dataset')
    if require_real:
        require(m['source_kind']=='real_anylogic_rpc','synthetic data cannot enter formal training')
        from .experiment_config import ROOT
        # Saved ALP parameter values and unrelated training entry points are
        # provenance, not a runtime configuration oracle. Pin the data path;
        # task/schema identity continues to be checked from RPC and events.
        data_paths=['python/'+n for n in ('constrained_batch.py','constrained_adapter.py',
            'reward_config.py','reward_contract.py','scenario_contract.py',
            'dispatch_bridge.py','dispatch_env.py','observation_contract.py','observation_schema.json')]
        data_paths += ['proto-src/src/main/proto/dispatch.proto','dist/protocol_manifest.json']
        if 'python/native_behavior.py' in m['source_sha256']:
            data_paths.append('python/native_behavior.py')
        if 'python/runtime_scenario.py' in m['source_sha256']:
            data_paths.append('python/runtime_scenario.py')
        if m.get('storage',{}).get('format') == HDF5:
            data_paths += ['python/'+n for n in ('hdf5_storage.py','storage_contract.py','storage_compatibility.py')]
        current={key:sha(ROOT/key) for key in data_paths}
        from .storage_compatibility import source_pins_match
        require(source_pins_match({k:m['source_sha256'].get(k) for k in data_paths},
                                 {k:current[k] for k in data_paths},
                                 storage_migration=m.get('storage',{}).get('format',LEGACY)==LEGACY),
                'dataset data-path revision differs; explicit compatibility review required')
    storage=storage_options(m.get('storage'))
    sample_name='samples.h5' if storage['format']==HDF5 else 'samples.npz'
    require(set(m['files_sha256'])=={sample_name,'episodes.json'},'dataset hash inventory incomplete')
    for name,d in m['files_sha256'].items():require(sha(p/name)==d,'dataset integrity failure: '+name)
    episodes=json.loads((p/'episodes.json').read_text());rows=[]
    if storage['format']==HDF5:
        from .hdf5_storage import read_dataset
        rows,embedded_episodes,saved_yr,saved_yg=read_dataset(p/sample_name)
        require(embedded_episodes==episodes,'HDF5 episode ledger differs from sidecar')
        require(len(rows)==m['transitions'] and len(episodes)==m['episodes'],'episode/transition count mismatch')
        try:
            validate_rows(rows,episodes)
            yr,yg=mc_labels(rows,episodes)
            require(np.array_equal(yr,saved_yr) and np.array_equal(yg,saved_yg),
                    'stored MC labels disagree with complete shifts')
        except BaseException:
            rows.close()
            raise
        return rows,episodes,yr,yg,m
    with np.load(p/sample_name,allow_pickle=False) as block:
        require(len(block['observations'])==len(block['next_observations'])==len(block['metadata_json'])==m['transitions'],
                'sample arrays differ in length')
        for obs,nxt,meta in zip(block['observations'],block['next_observations'],block['metadata_json']):
            rows.append(dict(json.loads(str(meta)),observation=obs.copy(),next_observation=nxt.copy()))
        validate_rows(rows,episodes)
        require(len(episodes)==m['episodes'],'episode manifest mismatch')
        yr,yg=mc_labels(rows,episodes)
        require(np.array_equal(yr,block['mc_reward']) and np.array_equal(yg,block['mc_grade']),'stored MC labels disagree with complete shifts')
    return rows,episodes,yr,yg,m


def collect_batch(env,frozen,buffer,jobs,*,reward_settings,recorder=None,runtime_validator=None):
    from .scenario_contract import from_components
    adapter=None
    try:
        for job in jobs:
            obs,info=env.reset();start_info=copy.deepcopy(info);initial=obs.copy();start=len(buffer.rows)
            if runtime_validator is not None:runtime_validator(info,job)
            if recorder is not None:recorder.start_episode(job,obs,info)
            require((info['experiment_run_index'],info['experiment_seed'])==(job['run_index'],job['experiment_seed']), 'seed manifest mismatch')
            cfg=configuration_from_components(info['reward_components'])
            checks={'wait_kappa':'kappa','w_production':'ore_weight','waste_shortfall_weight':'waste_weight',
                    'grade_lower':'grade_lower','grade_upper':'grade_upper','grade_epsilon':'grade_epsilon',
                    'tonnage_epsilon':'tonnage_epsilon','mass_balance_epsilon':'mass_balance_epsilon'}
            require(all(cfg.get(k)==reward_settings[v] for k,v in checks.items()),'RPC differs from approved reward configuration')
            if adapter is None:adapter=RewardCostAdapter(cfg)
            adapter.start(info)
            scenario=from_components(info['reward_components'])
            require(scenario is not None,'explicit runtime scenario required')
            while True:
                if hasattr(frozen,'sample_with_info'):
                    action,old=frozen.sample_with_info(obs,info['action_mask'],info)
                else:
                    action,old=frozen.sample(obs,info['action_mask'])
                if recorder is not None:recorder.action(obs,info,action)
                nxt,reward,terminated,truncated,next_info=env.step(action)
                if runtime_validator is not None:runtime_validator(next_info,job)
                adapted=adapter.transition(next_info,terminated=terminated,truncated=truncated)
                require(reward==adapted['reward'],'environment recomputed production reward')
                if recorder is not None:recorder.transition(obs,info,action,nxt,reward,next_info,adapted)
                buffer.append(dict(**adapted,episode_id=info['episode_id'],experiment_seed=info['experiment_seed'],
                    scene_id=job['scene_id'],scene_group=job['scene_group'],scenario_fingerprint=scenario.fingerprint,
                    action_time_seconds=info['sim_time_seconds'],truck_id=info['truck_id'],next_truck_id=next_info['truck_id'],
                    observation=obs.copy(),next_observation=nxt.copy(),action=action,mask=info['action_mask'],
                    old_log_prob=old,policy_version=frozen.policy_version,settlement_info=copy.deepcopy(next_info),
                    **(frozen.row_metadata(info) if hasattr(frozen,'row_metadata') else {})))
                frozen.assert_frozen();obs,info=nxt,next_info
                if terminated:break
            buffer.finish_episode(start,start_info=start_info,initial_observation=initial,job=job,scenario_fingerprint=scenario.fingerprint)
            if recorder is not None:recorder.finish_episode(job,info)
        buffer.seal()
    except BaseException as exc:
        buffer.abort(exc);raise
    return buffer
