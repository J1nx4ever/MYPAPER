"""Waste-task scenario RPC acceptance with seeded legal dispatch, no PPO or training."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import time

import gymnasium as gym
import numpy as np
from .dispatch_bridge import DispatchCallbackBridge
from .dispatch_env import DispatchEnv
from .dispatch_service import create_server
from .episode_runner import ParameterVariationEpisodeRunner
from .evaluate import _evaluate_episode,_load_seed_manifest
from .scenario_contract import ScenarioConfiguration,metadata
from .observation_contract import FIELD_NAMES
from .p0_live_acceptance import CORE_HASH,sha,write_json
from .protocol_artifacts import verify_protocol_artifacts

ROOT=Path(__file__).resolve().parents[1]


def quantization_error(value,scale):
    x=np.float32(value)
    return max(float(np.nextafter(x,np.float32(np.inf)))-float(x),
               float(x)-float(np.nextafter(x,np.float32(-np.inf))))*scale/2


class ScenarioObserver(gym.Wrapper):
    def __init__(self,env,scenario):
        super().__init__(env);self.scenario=scenario;self.stats=None
        for i in range(4):
            if FIELD_NAMES[1108+47*i]!=f'load.{i}.isOre':raise ValueError('source-role schema changed')
        for index,name in [(9,'global.wasteDelivered'),(1254,'load.3.reservedTons'),(1256,'load.3.producedTons')]:
            if FIELD_NAMES[index]!=name:raise ValueError('scenario observer raw schema changed')

    def reset(self,**kwargs):
        self.stats=dict(callbacks=0,open_points=0,closed_points=0,ambiguous_points=0,
                        closed_waste_mask_violations=0,high_precision_label_points=0,first_observed_closed_time=None)
        raw,info=self.env.reset(**kwargs)
        if not info.get('experiment_complete'):
            self.observe(raw,info,False)
        return raw,info

    def step(self,action):
        raw,reward,end,truncated,info=self.env.step(action)
        self.observe(raw,info,end or truncated);return raw,reward,end,truncated,info

    def observe(self,raw,info,terminal):
        if np.asarray(raw).shape!=(1519,):raise ValueError('raw RPC dimension must remain 1519')
        cfg=self.scenario
        if [float(raw[1108+47*i]) for i in range(4)] != [1,1,1,0]:
            raise ValueError('scenario observer requires three ore sources and waste source at id3')
        committed=math.fsum([float(raw[1254])*30000,float(raw[1256])*30000])
        error=quantization_error(raw[1254],30000)+quantization_error(raw[1256],30000)
        delivered=float(raw[9])*4000;de=quantization_error(raw[9],4000)
        label_keys=('audit.waste_task.committed_tons','audit.waste_task.delivered_tons','audit.waste_task.assignment_closed')
        components=info.get('reward_components',{})
        label_present=[key in components for key in label_keys]
        exact_commit=None
        if any(label_present):
            if not all(label_present):raise ValueError('partial high-precision waste audit labels')
            exact_commit=float(components[label_keys[0]]);exact_delivery=float(components[label_keys[1]])
            closed=components[label_keys[2]]
            if not (math.isfinite(exact_commit) and math.isfinite(exact_delivery) and closed in (0,1)):
                raise ValueError('invalid high-precision waste audit labels')
            if abs(exact_commit-committed)>error+1e-6 or abs(exact_delivery-delivered)>de+1e-6:
                raise ValueError('waste audit labels disagree with public observations')
            if bool(closed)!=(exact_commit>=cfg.waste_target_tons-1e-6):
                raise ValueError('waste closure label disagrees with committed task')
            if not (-1e-6<=exact_delivery<=exact_commit+1e-6 and exact_commit<=cfg.waste_task_upper_tons+1e-6):
                raise ValueError('high-precision waste budget violation')
            self.stats['high_precision_label_points']+=1
        if cfg.waste_task_limit_enabled:
            if (committed < -error-1e-6 or delivered < -de-1e-6 or
                committed>cfg.waste_task_upper_tons+error+1e-6 or delivered>committed+error+de+1e-6):
                raise ValueError('waste observation exceeds commitment/budget uncertainty bound')
            if not terminal:
                threshold=cfg.waste_target_tons-1e-6
                low=committed-error if exact_commit is None else exact_commit
                high=committed+error if exact_commit is None else exact_commit
                if low>=threshold:
                    self.stats['closed_points']+=1
                    if self.stats['first_observed_closed_time'] is None:self.stats['first_observed_closed_time']=info['sim_time_seconds']
                    if raw[22]!=0:
                        self.stats['closed_waste_mask_violations']+=1
                        raise ValueError('waste action remains available after confirmed commitment coverage')
                elif high<threshold:self.stats['open_points']+=1
                else:self.stats['ambiguous_points']+=1
        self.stats['callbacks']+=1
        if terminal:
            self.stats.update(terminal_commit_estimate_tons=committed,commit_quantization_bound_tons=error,
                              terminal_delivered_estimate_tons=delivered,delivered_quantization_bound_tons=de)


class LegalRule:
    def __init__(self,seed,policy):self.rng=np.random.Generator(np.random.PCG64(seed));self.policy=policy
    def predict(self,raw,deterministic=True):
        legal=np.flatnonzero(np.asarray(raw)[19:23])
        if not len(legal):raise ValueError('empty legal action mask')
        return int(legal[0] if self.policy=='first-legal' else self.rng.choice(legal)),None


def collect(args):
    cfg=ScenarioConfiguration(**json.loads(args.scenario_config.read_text()))
    if not cfg.waste_task_limit_enabled:raise ValueError('this acceptance entry expects the enabled waste-task scenario')
    if sha(ROOT/'python/episode_budget_ppo.py')!=CORE_HASH:raise ValueError('protected PPO core changed')
    if 'p0-ablation-formal' in args.output_dir.parts or 'p0-ablation-integration' in args.output_dir.parts:
        raise ValueError('new scenario must use a separate output namespace')
    expected=_load_seed_manifest(args.anylogic_seed_manifest,args.episodes)
    protocol=verify_protocol_artifacts()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    manifest=dict(status='waiting',purpose='scenario_rpc_acceptance_not_training',**metadata(cfg),
        ppo_training=False,model_created=False,policy=args.policy,policy_seed=args.policy_seed,
        expected_episode_metadata=[dict(run_index=i,experiment_seed=s) for i,s in expected],
        scenario_config_sha256=sha(args.scenario_config),seed_manifest_sha256=sha(args.anylogic_seed_manifest),
        raw_dimension=1519,configuration_source='anylogic_rpc',alp_runtime_verified=False,protocol=protocol,
        source_hashes={f:sha(ROOT/f) for f in ('python/scenario_contract.py','python/scenario_smoke.py',
            'python/dispatch_bridge.py','python/dispatch_env.py','python/observation_schema.json',
            'python/episode_budget_ppo.py','python/reward_config.py','python/reward_contract.py')})
    write_json(args.output_dir/'manifest.json',manifest)
    bridge=DispatchCallbackBridge(runtime_configuration=True,expected_episode_metadata=expected,expected_scenario=cfg)
    env=ScenarioObserver(DispatchEnv(bridge,ParameterVariationEpisodeRunner(args.episodes),args.callback_timeout),cfg)
    server=None;records=[]
    try:
        server=create_server(bridge,args.host,args.port);server.start()
        print('Waste scenario RPC acceptance listening; start RL_Training. Legal rule only, no PPO.',flush=True)
        reward=bridge.wait_for_configuration(args.callback_timeout)
        manifest.update(status='running',reward_configuration=reward.to_dict(),reward_configuration_fingerprint=reward.fingerprint)
        write_json(args.output_dir/'manifest.json',manifest)
        rule=LegalRule(args.policy_seed,args.policy);start=time.perf_counter()
        with (args.output_dir/'episodes.jsonl').open('x') as stream:
            for i in range(args.episodes):
                row=_evaluate_episode(env,rule,deterministic=True)
                if (row['run_index'],row['experiment_seed'])!=expected[i]:raise ValueError('episode identity differs')
                waste=row['terminal_reward_components']['waste_tons_shift']
                row.update(scenario_observation_audit=dict(env.stats),**metadata(cfg))
                row['waste_band_met']=cfg.waste_target_tons-1e-6<=waste<=cfg.waste_task_upper_tons+1e-6
                row['scenario_true_joint_met']=row['ore_plan_met'] and row['waste_band_met'] and row['all_windows_compliant']
                records.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
        covered=all(r['scenario_observation_audit']['closed_points']>0 and r['scenario_observation_audit']['open_points']>0 for r in records)
        result=dict(status='passed' if covered else 'incomplete_coverage',business_joint_passed=sum(r['scenario_true_joint_met'] for r in records),
            episodes=len(records),elapsed_seconds=time.perf_counter()-start,**metadata(cfg),
            coverage='open and definitely closed commitment states in every episode',
            caveat='commitment is reconstructed at float32 uncertainty; no new engine events or full observation logs')
        if sha(ROOT/'python/episode_budget_ppo.py')!=CORE_HASH:raise ValueError('protected PPO core changed')
        manifest.update(status='complete',acceptance_status=result['status'],episodes_sha256=sha(args.output_dir/'episodes.jsonl'))
        write_json(args.output_dir/'manifest.json',manifest);write_json(args.output_dir/'acceptance.json',result)
        return 0 if covered else 2
    except BaseException as exc:
        manifest.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        write_json(args.output_dir/'manifest.json',manifest)
        write_json(args.output_dir/'acceptance.json',dict(status='failed',error=manifest['error']))
        raise
    finally:
        env.close()
        if server is not None:server.stop(2).wait()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario-config',type=Path,default=ROOT/'configs/waste_task_scenario_v1.json')
    parser.add_argument('--anylogic-seed-manifest',type=Path,default=ROOT/'configs/seeds/waste_task_smoke_seed_manifest.json')
    parser.add_argument('--episodes',type=int,default=2)
    parser.add_argument('--policy',choices=('uniform-legal','first-legal'),default='uniform-legal')
    parser.add_argument('--policy-seed',type=int,default=7)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--host',default='127.0.0.1');parser.add_argument('--port',type=int,default=50051)
    parser.add_argument('--callback-timeout',type=float,default=3600)
    args=parser.parse_args()
    if args.episodes<1 or args.policy_seed<0 or args.callback_timeout<=0:parser.error('positive budget/timeout and nonnegative seed required')
    return collect(args)


if __name__=='__main__':raise SystemExit(main())
