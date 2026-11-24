"""Random-initialized 1519D clipped PPO: one Actor, reward and Grade Critics.

AnyLogic RL_Training supplies 50 serial training shifts then five validation
shifts. A persistent RPC server performs on-policy updates every 128 decision
transitions, including a final shorter rollout at the 50th terminal. No BC
weights or BC preprocessing are used. Raw ALP reward and observations unchanged.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
import threading
import time

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from gymnasium import spaces

from .constrained_adapter import AdapterConfiguration, RewardCostAdapter, SamplingError
from .constrained_ppo import CriticNetwork
from .dispatch_bridge import DispatchCallbackBridge, ProtocolError
from .dispatch_env import DispatchEnv
from .dispatch_service import create_server
from .generated import dispatch_pb2 as pb
from .masked_policy import DispatchMaskedPolicy
from .observation_contract import OBSERVATION_SIZE,EGO_INDICES,FIELD_NAMES
from .scenario_contract import from_components

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'runs2/ppo-1519-random-two-critic-50-20261028-v1'
CORE=ROOT/'python/episode_budget_ppo.py'
ALP=ROOT/'anylogic/DRL-PPO.alp'
ORDER_SHA=hashlib.sha256(('\n'.join(FIELD_NAMES)+'\n').encode()).hexdigest()


def require(ok,message):
    if not ok: raise ValueError(message)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(module):
    h=hashlib.sha256()
    for key,v in sorted(module.state_dict().items()):
        h.update(key.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def write(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def append(path,obj):
    with Path(path).open('a',encoding='utf-8') as f:
        f.write(json.dumps(obj,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n');f.flush()


def stable_info(info):
    result=dict(info)
    result['reward_components']={k:v for k,v in info['reward_components'].items() if k!='config.eval.shovel_idle_seconds'}
    return result


def schedule():
    return [dict(run_index=i,experiment_seed=202610280400+i,sampling_seed=202610289400+i,
                 split='train' if i<=50 else 'validation') for i in range(1,56)]


class Actor(nn.Module):
    """Untrained SB3 dispatch actor modules, with no value head or BC weights."""
    def __init__(self,seed):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            template=DispatchMaskedPolicy(spaces.Box(low=-np.inf,high=np.inf,shape=(OBSERVATION_SIZE,),dtype=np.float32),
                spaces.Discrete(4),lambda _:3e-4,net_arch=dict(pi=[128,128],vf=[128,128]))
        self.policy_net=template.mlp_extractor.policy_net
        self.action_net=template.action_net

    def forward(self,obs):
        mask=DispatchMaskedPolicy._decision_mask(obs)
        logits=self.action_net(self.policy_net(obs))
        require(bool(torch.isfinite(logits).all()),'nonfinite Actor logits')
        return logits.masked_fill(~mask,-torch.inf)


class TwoCritics(nn.Module):
    def __init__(self,seed):
        super().__init__()
        self.reward=CriticNetwork(1519,(128,128),seed+1)
        self.grade=CriticNetwork(1520,(128,128),seed+2)

    def forward(self,obs,history):
        require(obs.shape[1]==1519 and history.shape==obs.shape[:1],'Critic input mismatch')
        grade=self.grade(torch.cat((obs,history[:,None]),dim=1))*(1-history)
        return self.reward(obs),grade


class RuntimeBridge(DispatchCallbackBridge):
    """Runtime config/scenario must be explicit, consistent and RPC-sourced."""
    def __init__(self,root,jobs):
        super().__init__(runtime_configuration=True,expected_episode_metadata=[(j['run_index'],j['experiment_seed']) for j in jobs])
        self.root=root;self.adapter_configuration=None;self.fatal_error=None
        self._failure_lock=threading.Lock();self.terminal_returned=threading.Event()

    def _register(self,request):
        try:
            c={k:float(v) for k,v in request.reward_components.items() if k!='config.eval.shovel_idle_seconds'}
            cfg=AdapterConfiguration.from_rpc(c)
            if self.adapter_configuration is None:
                data=from_components(c)
                require(data is not None and data.waste_task_limit_enabled,'explicit enabled waste-task rule required')
                self.expected_scenario=data
            else:
                require(cfg==self.adapter_configuration,'runtime RPC configuration changed')
            # When experiment-1 diagnostic telemetry is present, validate it;
            # legacy RPC without this extension is not forced to supply it.
            expected={'config.eval.version':1.,'config.eval.policy':3.,'config.eval.bc_collect':0.,
                      'config.eval.fallback_enabled':0.,'config.eval.fallback_used':0.,
                      'config.eval.physical':1.,'config.eval.ai_enabled':1.,
                      'config.eval.deterministic_check':0.}
            present={k:v for k,v in c.items() if k.startswith('config.eval.')}
            if present:
                for key,val in expected.items(): require(c.get(key)==val,'runtime execution flag mismatch: '+key)
                for i in range(8):
                    key=f'config.eval.schema_word_{i}'
                    require(c.get(key)==int(ORDER_SHA[8*i:8*i+8],16),'runtime field order mismatch: '+key)
            ticket=super()._register(request)
            self.adapter_configuration=cfg
            return ticket
        except (ValueError,SamplingError) as exc:
            raise ProtocolError(str(exc)) from exc

    def submit_callback(self,request,timeout=None):
        try:
            if self.fatal_error is not None: raise self.fatal_error
            response=super().submit_callback(request,timeout)
            if request.kind==pb.EPISODE_END:
                append(self.root/'terminal_acks.jsonl',dict(episode_id=request.episode_id,run_index=request.experiment_run_index,
                    seed=request.experiment_seed,sequence=request.decision_sequence,has_action=response.has_action,
                    scope='server returned ACK; client receipt needs AnyLogic console'))
                self.terminal_returned.set()
            return response
        except Exception as exc:
            with self._failure_lock:
                if self.fatal_error is None: self.fatal_error=exc
                append(self.root/'rpc_errors.jsonl',dict(error=f'{type(exc).__name__}: {exc}',episode_id=request.episode_id,
                    run_index=request.experiment_run_index,seed=request.experiment_seed,sequence=request.decision_sequence,
                    config={k:v for k,v in request.reward_components.items() if k.startswith('config.')}))
            raise


def gae_two(rollout,critics,gamma,lam):
    obs=torch.as_tensor(np.stack([r['obs'] for r in rollout]),dtype=torch.float32)
    nxt=torch.as_tensor(np.stack([r['next_obs'] for r in rollout]),dtype=torch.float32)
    before=torch.as_tensor([r['history_before'] for r in rollout],dtype=torch.float32)
    after=torch.as_tensor([r['history_after'] for r in rollout],dtype=torch.float32)
    with torch.no_grad():
        vr,vg=critics(obs,before);nr,ng=critics(nxt,after)
    vr=vr.numpy().astype(np.float64);vg=vg.numpy().astype(np.float64)
    nr=nr.numpy().astype(np.float64);ng=ng.numpy().astype(np.float64)
    signals=np.array([[r['reward'],r['grade_cost']] for r in rollout],dtype=np.float64)
    done=np.array([r['terminated'] for r in rollout],dtype=bool)
    adv=np.zeros((len(rollout),2),dtype=np.float64);carry=np.zeros(2)
    for i in range(len(rollout)-1,-1,-1):
        nonterminal=0. if done[i] else 1.
        delta=signals[i]+gamma*nonterminal*np.array([nr[i],ng[i]])-np.array([vr[i],vg[i]])
        carry=delta+gamma*lam*nonterminal*carry
        adv[i]=carry
    targets=adv+np.column_stack((vr,vg))
    require(np.isfinite(adv).all() and np.isfinite(targets).all(),'nonfinite GAE/targets')
    require(np.all(vg[before.numpy()==1]==0),'Grade known-zero violated')
    return adv,targets,dict(last_done=bool(done[-1]),last_bootstrap_reward=0. if done[-1] else float(nr[-1]),
                            last_bootstrap_grade=0. if done[-1] else float(ng[-1]))


def update(rollout,actor,critics,actor_opt,critic_opt,shuffle,cfg,index,at):
    require(0<len(rollout)<=cfg['n_steps'],'invalid rollout length')
    adv,targets,bootstrap=gae_two(rollout,critics,cfg['gamma'],cfg['gae_lambda'])
    raw=torch.as_tensor(np.stack([r['obs'] for r in rollout]),dtype=torch.float32)
    history=torch.as_tensor([r['history_before'] for r in rollout],dtype=torch.float32)
    actions=torch.as_tensor([r['action'] for r in rollout],dtype=torch.int64)
    old=torch.as_tensor([r['old_log_prob'] for r in rollout],dtype=torch.float32)
    combined=torch.as_tensor(adv[:,0]-cfg['grade_coefficient']*adv[:,1],dtype=torch.float32)
    target=torch.as_tensor(targets,dtype=torch.float32)
    actor_before=digest(actor);critic_before=digest(critics)
    with torch.no_grad():
        initial_log=Categorical(logits=actor(raw)).log_prob(actions)
        ratio_error=float((torch.exp(initial_log-old)-1).abs().max())
    require(ratio_error<1e-5,'rollout is not on-policy or old log-prob differs')
    metrics=[];used=set();actor.train();critics.train()
    for _ in range(cfg['epochs']):
        order=torch.randperm(len(rollout),generator=shuffle)
        for start in range(0,len(rollout),cfg['batch_size']):
            ids=order[start:start+cfg['batch_size']]
            advantages=combined[ids]
            if len(ids)>1: advantages=(advantages-advantages.mean())/(advantages.std()+1e-8)
            dist=Categorical(logits=actor(raw[ids]));logp=dist.log_prob(actions[ids]);ratio=torch.exp(logp-old[ids])
            surrogate=torch.minimum(ratio*advantages,torch.clamp(ratio,1-cfg['clip'],1+cfg['clip'])*advantages)
            policy_loss=-surrogate.mean()-cfg['entropy_coefficient']*dist.entropy().mean()
            require(bool(torch.isfinite(policy_loss)),'nonfinite policy loss')
            actor_opt.zero_grad(set_to_none=True);policy_loss.backward()
            actor_grad=float(torch.nn.utils.clip_grad_norm_(actor.parameters(),cfg['max_grad_norm']))
            actor_opt.step()
            reward_value,grade_value=critics(raw[ids],history[ids])
            reward_loss=((reward_value-target[ids,0])**2).mean()
            grade_loss=((grade_value-target[ids,1])**2).mean()
            critic_loss=cfg['value_coefficient']*(reward_loss+grade_loss)
            require(bool(torch.isfinite(critic_loss)),'nonfinite Critic loss')
            critic_opt.zero_grad(set_to_none=True);critic_loss.backward()
            critic_grad=float(torch.nn.utils.clip_grad_norm_(critics.parameters(),cfg['max_grad_norm']))
            critic_opt.step();used.update(ids.tolist())
            with torch.no_grad():
                logratio=logp-old[ids]
                metrics.append(dict(policy_loss=float(policy_loss),reward_value_mse=float(reward_loss),grade_value_mse=float(grade_loss),
                    approx_kl=float(((ratio-1)-logratio).mean()),clip_fraction=float(((ratio-1).abs()>cfg['clip']).float().mean()),
                    entropy=float(dist.entropy().mean()),actor_grad_norm=actor_grad,critic_grad_norm=critic_grad))
    require(len(used)==len(rollout),'rollout samples omitted by optimizer')
    require(all(torch.isfinite(p).all() for p in list(actor.parameters())+list(critics.parameters())),'nonfinite trained parameters')
    return dict(update_index=index,transitions=len(rollout),unique_transitions_updated=len(used),epochs=cfg['epochs'],
        actor_optimizer_steps=len(metrics),critic_optimizer_steps=len(metrics),first_ratio_max_error=ratio_error,
        actor_digest_before=actor_before,actor_digest_after=digest(actor),critics_digest_before=critic_before,critics_digest_after=digest(critics),
        trigger=at,**bootstrap,mean={k:statistics.mean(m[k] for m in metrics) for k in metrics[0]})


def prepare(root):
    root=Path(root).resolve();require(not root.exists(),'result directory exists; choose a new one')
    root.mkdir(parents=True)
    cfg=dict(contract='dispatch-1519-random-two-critic-ppo-v1',status='READY',
        training_seed=901,actor_initialization='random SB3 architecture seed901; no BC model or normalizer',
        critics='independent reward 1519 and Grade 1520, 128-Tanh-128-Tanh-1; known-zero Grade gate',
        preprocessing='raw 1519D RPC observation and ALP native field scaling; no online or BC normalization',
        reward='-delta_wait_vehicle_seconds/25000 + terminal RPC production + terminal RPC waste; no grade reward double counting',
        grade='first violating/empty window per shift costs 1; subsequent violations zero; Grade known-zero if history_before=1',
        grade_coefficient=1.,coefficient_policy='fixed coefficient; no multiplier updates',
        field_order_sha256=ORDER_SHA,
        gamma=.995,gae_lambda=.95,n_steps=128,batch_size=32,epochs=10,actor_lr=.0003,critic_lr=.0003,
        clip=.2,entropy_coefficient=.01,value_coefficient=.5,max_grad_norm=.5,
        advantage_normalization='SB3-style within each minibatch, not across episodes',
        episodes_train=50,episodes_validation=5,validation='fixed final checkpoint; stochastic masked action; no reselection',
        checkpoint_interval_updates=10,record_transition_trace=False,schedule=schedule(),
        config_authority='explicit AnyLogic RPC reward and scenario; first configuration locks all subsequent calls',
        protected_core_sha256=sha(CORE),local_alp_sha256_at_prepare=sha(ALP),local_alp_hash_is_not_runtime_build_verification=True)
    src=['two_critic_ppo.py','episode_budget_ppo.py','constrained_adapter.py','constrained_ppo.py',
         'dispatch_bridge.py','dispatch_env.py','dispatch_service.py','masked_policy.py','observation_contract.py','observation_schema.json',
         'reward_contract.py','reward_config.py','reward_v1_configuration.json','scenario_contract.py']
    cfg['pins']={str((ROOT/'python'/name).resolve()):sha(ROOT/'python'/name) for name in src}
    write(root/'plan.json',cfg)
    with (root/'seeds.csv').open('w',encoding='utf-8',newline='') as f:
        import csv
        writer=csv.DictWriter(f,fieldnames=list(cfg['schedule'][0]));writer.writeheader();writer.writerows(cfg['schedule'])
    write(root/'status.json',dict(status='PREPARED_WAITING_ANYLOGIC',completed_training_episodes=0,completed_validation_episodes=0,updates=0))
    instructions(root)
    print('PREPARED',root,flush=True)


def verify_plan(root):
    p=json.loads((root/'plan.json').read_text())
    for path,d in p['pins'].items(): require(sha(path)==d,'protected source changed: '+path)
    return p


def instructions(root):
    (root/'README.md').write_text(f'''# 1519D 随机初始化双Critic PPO：50班训练 + 5班验证

本次Actor与Critic使用独立随机初始化，训练随机seed=901；不读取BC模型、BC预处理或实验3A Critic。正式1519D、mask及ALP业务reward保持原样。
Reward训练信号：`-delta_wait_vehicle_seconds/25000 + RPC production + RPC waste`，矿废项仅在班末归最后transition。Grade：四窗口首次违规/无供矿计1，后续不重复，Grade Critic历史已违规时输出恒0。Actor优势 `A_reward - 1.0*A_grade`，固定权重、不更新乘子；不是SB3单Critic原生PPO，而是两个价值通道的clipped PPO。

## RL_Training设置一次

在 **RL_Training → Properties → Parameters → Freeform参数表** 修改：

| 项 | 之前磁盘源码值 | 本轮值 |
|---|---|---|
| Number of runs | 30 | **55**（前50训练，后5验证） |
| currentDispatchPolicy | 4 | **3** |
| experimentRunIndex | `(long) index + 1L` | 保持 |
| experimentSeed | `202610270101L + (long) index` | **`202610280401L + (long) index`** |
| bcCollectEnabled | true | **false** |
| deterministicCheck | false | 保持false |
| usePhysicalObservation | true | 保持true |
| enableWasteTaskLimit | 默认true | **true** |
| diagEvalEnabled | 默认false | **true**（如使用实验1已接好的诊断遥测） |

保持串行、关闭replication、关闭snapshot、Stop time=28800秒；Main普通变量enableOnnxDispatch=true、enableDispatchFallback=false、dispatchInferencePort=50051。三组开关只控制运行路径，不修改业务规则、reward/cost或观测。保存并Build，关闭旧运行实例。上述磁盘源码值为准备时记录，实际GUI若已有改动，以当前值为修改前核对；无需重新上传或审计ALP。运行配置以RPC实际值校验，首条锁定后不得变化。

## 开始

```bash
cd /home/jinx/New_DispatchPPO
/home/jinx/miniconda3/envs/anylogic-env/bin/python -u -m python.two_critic_ppo run --root {root}
```

看到 `LISTENING` 后点击RL_Training的Run一次。第1—50班训练，51—55班验证；同一个RPC服务不中断。仿真seed分别为202610280401—455，每班独立动作采样seed见seeds.csv；训练网络/洗牌仅一个seed901。后5班用最终固定Actor采样验证，不用于选模型。

## rollout何时更新

- **每累计128条有效决策transition后立即更新一次**，通常发生在班中；当前下一次派车请求会等待Python完成更新。
- 一个rollout可以跨班；真正terminal截断GAE，班中截断使用该下一状态的Critic价值bootstrap，Grade历史跨rollout保持正确。
- 第50班真正终止后，若不足128条仍有尾段，立即以有效长度更新；没有padding和丢弃。若刚好整除则无尾段。
- 每次更新10个epoch、小批32条，clip=0.2，gamma=0.995、GAE lambda=0.95；保持和普通PPO相同的逐minibatch优势标准化。随机种子901只用于本次一个训练运行。
- 每次更新时机、run_index、仿真时间、决策数、尾段、样本覆盖和损失/KL等汇总在 `rollouts.jsonl`；每10次更新保存checkpoint，末次另存final。默认不保存逐transition或观测快照。
- 验证5班期间不更新Actor、Critic、乘子或预处理。Python自动输出逐班、训练统计和验证报告；如技术异常，记录状态并保留原seed，不静默替换。

完整结果输出在 `{root}`。本次只有一个PPO训练seed和5个验证班，不作统计显著性或可靠合规声明。
''',encoding='utf-8')


def save_checkpoint(root,actor,critics,actor_opt,critic_opt,index,plan,shuffle,final=False):
    folder=root/'checkpoints';folder.mkdir(exist_ok=True)
    path=folder/('final.pt' if final else f'update_{index:04d}.pt')
    require(not path.exists(),'refusing to overwrite checkpoint')
    torch.save(dict(contract=plan['contract'],update_index=index,actor=actor.state_dict(),critics=critics.state_dict(),
        actor_optimizer=actor_opt.state_dict(),critic_optimizer=critic_opt.state_dict(),
        training_seed=plan['training_seed'],shuffle_generator=shuffle.get_state(),plan_sha256=sha(root/'plan.json'),
        preprocessing=plan['preprocessing'],reward=plan['reward'],grade=plan['grade']),path)
    return dict(path=str(path.resolve()),sha256=sha(path),update_index=index)


def mean_dict(rows,key):
    return statistics.mean(float(row[key]) for row in rows)


def validation_report(root,records,validation_samples,training_samples,cfg,initial_actor,final_actor,steps):
    val=[r for r in records if r['split']=='validation']
    train=[r for r in records if r['split']=='train']
    require(len(train)==50 and len(val)==5,'expected fifty training and five validation shifts')
    def make_baseline(samples,channel):
        use=[r for r in samples if channel!='grade' or not r['history']]
        require(use,'baseline needs training samples')
        y=np.array([r['return_'+channel] for r in use]);t=np.array([r['remaining']/28800 for r in use])
        phi=np.column_stack((np.ones(len(t)),t,t*t,t*t*t))
        return dict(constant=float(y.mean()),time_coefficients=np.linalg.lstsq(phi,y,rcond=None)[0].tolist(),
                    training_samples=len(use),training_source='50 on-policy training episodes from evolving Actor; not final-policy holdout')
    baselines={c:make_baseline(training_samples,c) for c in ('reward','grade')}
    value_metrics={}
    for c in ('reward','grade'):
        subset=[r for r in validation_samples if c!='grade' or not r['history']]
        y=np.array([r['return_'+c] for r in subset]);p=np.array([r['value_'+c] for r in subset])
        t=np.array([r['remaining']/28800 for r in subset]);phi=np.column_stack((np.ones(len(t)),t,t*t,t*t*t))
        constant=baselines[c]['constant'];timed=phi@np.asarray(baselines[c]['time_coefficients'])
        value_metrics[c]=dict(samples=len(y),critic_mse=float(np.mean((p-y)**2)),
            train_constant_mse=float(np.mean((constant-y)**2)),train_time_mse=float(np.mean((timed-y)**2)),
            label_variance=float(np.var(y)),validation_episode_failures=sum(r['grade_failed'] for r in val) if c=='grade' else None)
    data=dict(status='TRAINING_AND_VALIDATION_COMPLETE',training_seed=cfg['training_seed'],
        training_episodes=50,validation_episodes=5,rollout_updates=steps['updates'],
        training_transitions=steps['training_transitions'],unique_updated_transitions=steps['updated_transitions'],
        discarded_transitions=steps['training_transitions']-steps['updated_transitions'],
        initial_actor_sha256=initial_actor,final_actor_sha256=final_actor,
        validation_actor_digest_before=steps['validation_actor_before'],validation_actor_digest_after=steps['validation_actor_after'],
        validation_critics_digest_before=steps['validation_critics_before'],validation_critics_digest_after=steps['validation_critics_after'],
        training_actor_optimizer_steps=steps['actor_optimizer_steps'],training_critic_optimizer_steps=steps['critic_optimizer_steps'],
        validation_optimizer_steps=0,grade_coefficient=cfg['grade_coefficient'],
        validation_means={k:mean_dict(val,k) for k in ('ore_tons','waste_tons','wait_vehicle_seconds','reward_total')},
        validation_all_windows_compliant=sum(r['all_windows_compliant'] for r in val),
        validation_joint_business_success=sum(r['joint_business_success'] for r in val),
        validation_grade_failure=sum(r['grade_failed'] for r in val),validation_value_metrics=value_metrics,
        limitations=['one training seed, five development-validation shifts; no significance or reliable compliance claim',
            'first-failure Grade cost is discounted with gamma=0.995 per decision; value is a discounted cost, not an undiscounted failure probability',
            'terminal ACK evidence confirms server-side return; AnyLogic client receipt needs console',
            'schema ID and observation contract are validated; runtime field-order hash and execution flags require optional config.eval telemetry',
            'local ALP file hash does not prove running build identity'])
    write(root/'validation_baselines.json',baselines);write(root/'summary.json',data)
    import csv
    columns=['split','run_index','experiment_seed','sampling_seed','transitions','action_counts','ore_tons','waste_tons',
             'ore_gap_tons','waste_gap_tons','wait_vehicle_seconds','grade_failed','all_windows_compliant','joint_business_success',
             'reward_total','runtime_seconds','model_version_start','model_version_end','terminal_ack_returned','windows']
    with (root/'episodes.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');w.writeheader()
        for row in records: w.writerow({k:(json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v) for k,v in row.items() if k in columns})
    text=f'''# 1519D 随机初始化双Critic clipped PPO：50班与验证

训练完成：一个训练seed={cfg['training_seed']}，50个完整训练班、5个独立仿真seed验证班；{steps['training_transitions']}条训练transition，{steps['updates']}次rollout更新，覆盖{steps['updated_transitions']}条，丢弃{data['discarded_transitions']}条。训练reward为等待诊断分量及RPC班末矿废reward，不包含品位reward；Grade用首次窗口违规成本、固定系数{cfg['grade_coefficient']}。

每128条有效transition更新；最后不足128条的尾段在第50班终止后更新。更新时机与损失/KL等见rollouts.jsonl；验证期间Actor/Critic/optimizer冻结，不按验证挑选checkpoint。

| 验证指标 | 五班结果 |
|---|---:|
| 平均矿石破碎量 t | {data['validation_means']['ore_tons']:.3f} |
| 平均废石交付量 t | {data['validation_means']['waste_tons']:.3f} |
| 平均等待 vehicle-seconds | {data['validation_means']['wait_vehicle_seconds']:.3f} |
| 四窗口全合规班次 | {data['validation_all_windows_compliant']}/5 |
| 品位、矿废任务联合合规班次 | {data['validation_joint_business_success']}/5 |

| 价值通道 | Critic验证MSE | 训练集常数baseline MSE | 训练集时间baseline MSE |
|---|---:|---:|---:|
| Reward（等待+矿废） | {value_metrics['reward']['critic_mse']:.6g} | {value_metrics['reward']['train_constant_mse']:.6g} | {value_metrics['reward']['train_time_mse']:.6g} |
| Grade（此前未违规） | {value_metrics['grade']['critic_mse']:.6g} | {value_metrics['grade']['train_constant_mse']:.6g} | {value_metrics['grade']['train_time_mse']:.6g} |

baseline只拟合50个训练班，来自逐步变化的历史Actor，因此不是最终策略分布的无偏估计；数字仅作开发诊断。gamma=0.995按决策步折扣，Grade值是折扣cost，不应直接表述为不折扣的班次失败概率。窗口品位、吨位、达标及每班耗时见episodes.csv。

初始Actor digest `{initial_actor}`；最终Actor digest `{final_actor}`；验证前后Actor和两个Critic digest一致。训练只使用seed901的一次随机初始化，未使用BC权重、BC统计量、在线标准化或乘子更新。Actor与两个独立Critic参数分离。

RPC实际传入配置已锁定，schema ID、mask、窗口结算与终止检查通过；仅当RPC提供 `config.eval` 遥测时才能独立校验字段顺序哈希及AI路径开关，实际遥测见runtime_configuration.json。服务端终止ACK已记录，客户端是否接收需AnyLogic控制台。小样本验证不能证明稳定收敛或可靠合规，也未包含与其他策略的同种子对照。受保护PPO核心的SHA前后相同。
'''
    (root/'report.md').write_text(text,encoding='utf-8')
    return data


def run(root,timeout,port):
    root=Path(root).resolve();cfg=verify_plan(root)
    require(not (root/'run_started.json').exists(),'run already attempted; preserve results and use a new directory')
    require(timeout>0,'timeout must be positive')
    write(root/'run_started.json',dict(status='RUNNING',start_unix_seconds=time.time(),training_seed=cfg['training_seed']))
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.manual_seed(cfg['training_seed']);np.random.seed(cfg['training_seed'])
    actor=Actor(cfg['training_seed']);critics=TwoCritics(cfg['training_seed'])
    require(not set(map(id,actor.parameters())).intersection(map(id,critics.parameters())),'Actor/Critic parameter sharing forbidden')
    actor_opt=torch.optim.Adam(actor.parameters(),lr=cfg['actor_lr'],eps=1e-5)
    critic_opt=torch.optim.Adam(critics.parameters(),lr=cfg['critic_lr'],eps=1e-5)
    shuffle=torch.Generator().manual_seed(cfg['training_seed'])
    initial_actor=digest(actor);initial_critics=digest(critics)
    steps=dict(updates=0,training_transitions=0,updated_transitions=0,actor_optimizer_steps=0,critic_optimizer_steps=0)
    records=[];buffer=[];training_samples=[];validation_samples=[];bridge=None;env=None;server=None;current=None
    core_before=sha(CORE);alp_before=sha(ALP);started=time.monotonic();error=None
    try:
        bridge=RuntimeBridge(root,cfg['schedule']);env=DispatchEnv(bridge,callback_timeout=1)
        server=create_server(bridge,port=port);server.start()
        print(f'LISTENING 127.0.0.1:{port}; RL_Training: 50 train + 5 validation; PPO seed={cfg["training_seed"]}',flush=True)
        for job in cfg['schedule']:
            episode_start=time.monotonic();bridge.terminal_returned.clear()
            current=dict(job,transitions=0,action_counts=[0]*4,model_version_start=steps['updates'],status='RUNNING')
            def poll(fun):
                while True:
                    require(bridge.fatal_error is None,'RPC error: '+str(bridge.fatal_error))
                    require(time.monotonic()-episode_start<timeout,'AnyLogic callback timeout')
                    try: return fun()
                    except TimeoutError: pass
            obs,info=poll(env.reset)
            received=time.monotonic()
            if job['run_index']==1:
                config=stable_info(info)['reward_components']
                write(root/'runtime_configuration.json',dict(source='AnyLogic EPISODE_START RPC',config=config,
                    adapter_configuration_fingerprint=bridge.adapter_configuration.fingerprint,
                    schema_id=info['observation_schema_id'],reward_contract_id=info['reward_contract_id']))
            if job['split']=='validation' and job['run_index']==51:
                require(not buffer,'training rollout must be flushed before validation')
                steps['validation_actor_before']=digest(actor);steps['validation_critics_before']=digest(critics)
                steps['validation_actor_steps_before']=steps['actor_optimizer_steps'];steps['validation_critic_steps_before']=steps['critic_optimizer_steps']
            adapter=RewardCostAdapter(bridge.adapter_configuration);adapter.start(stable_info(info))
            gen=torch.Generator().manual_seed(job['sampling_seed']);episode_signals=[];episode_start_observations=[]
            actor.eval();critics.eval()
            while True:
                history_before=float(adapter.grade_failed_so_far)
                ob=torch.from_numpy(np.asarray(obs,dtype=np.float32).copy()).unsqueeze(0)
                with torch.inference_mode():
                    dist=Categorical(logits=actor(ob));probs=dist.probs[0]
                    action=int(torch.multinomial(probs,1,generator=gen));logprob=float(dist.log_prob(torch.tensor([action]))[0])
                    if job['split']=='validation':
                        v_reward,v_grade=critics(ob,torch.tensor([history_before]))
                        values=(float(v_reward[0]),float(v_grade[0]))
                    else: values=(None,None)
                require(obs[19+action]==1,'Actor sampled a masked action')
                env.bridge.respond_action(env._pending,action);env._pending=None
                ticket=poll(lambda:bridge.next_callback(timeout=1));request=ticket.request
                next_info=env._info(request);next_obs=np.asarray(request.obs.values,dtype=np.float32)
                require(request.episode_id==info['episode_id'],'episode identity changed')
                terminal=request.kind==pb.EPISODE_END
                adapted=adapter.transition(stable_info(next_info),terminated=bool(request.terminated),truncated=bool(request.truncated))
                components=next_info['reward_components']
                for key in ('production','waste'):
                    require(key in components and math.isfinite(components[key]),'RPC missing terminal reward component: '+key)
                    if not terminal: require(components[key]==0,'production/waste appeared before terminal')
                reward=adapted['reward']+float(components['production'])+float(components['waste'])
                entry=dict(obs=np.asarray(obs,dtype=np.float32).copy(),next_obs=next_obs.copy(),action=action,old_log_prob=logprob,
                    reward=reward,grade_cost=adapted['grade_cost'],history_before=history_before,
                    history_after=float(adapted['history_after']),terminated=bool(request.terminated))
                episode_signals.append(dict(reward=reward,grade=adapted['grade_cost'],history=history_before,
                    remaining=28800-info['sim_time_seconds'],value_reward=values[0],value_grade=values[1]))
                current['transitions']+=1;current['action_counts'][action]+=1
                if job['split']=='train':
                    buffer.append(entry);steps['training_transitions']+=1
                    if len(buffer)==cfg['n_steps']:
                        at=dict(kind='rollout_full',run_index=job['run_index'],simulation_seed=job['experiment_seed'],
                                simulation_time=next_info['sim_time_seconds'],settlement_sequence=next_info['decision_sequence'],
                                episode_terminated=bool(request.terminated))
                        metrics=update(buffer,actor,critics,actor_opt,critic_opt,shuffle,cfg,steps['updates']+1,at)
                        steps['updates']+=1;steps['updated_transitions']+=metrics['unique_transitions_updated']
                        steps['actor_optimizer_steps']+=metrics['actor_optimizer_steps'];steps['critic_optimizer_steps']+=metrics['critic_optimizer_steps']
                        append(root/'rollouts.jsonl',metrics);buffer.clear()
                        print(f'ROLLOUT {steps["updates"]} transitions={metrics["transitions"]} run={job["run_index"]} sim_time={at["simulation_time"]:.1f}',flush=True)
                        if steps['updates']%cfg['checkpoint_interval_updates']==0:
                            save_checkpoint(root,actor,critics,actor_opt,critic_opt,steps['updates'],cfg,shuffle)
                if terminal:
                    require(request.terminated and not request.truncated,'only complete true terminal is allowed')
                    require(bridge.terminal_returned.wait(5),'terminal ACK was not returned')
                    if job['split']=='train' and job['run_index']==50 and buffer:
                        at=dict(kind='final_training_tail',run_index=50,simulation_seed=job['experiment_seed'],
                                simulation_time=next_info['sim_time_seconds'],settlement_sequence=next_info['decision_sequence'],episode_terminated=True)
                        metrics=update(buffer,actor,critics,actor_opt,critic_opt,shuffle,cfg,steps['updates']+1,at)
                        steps['updates']+=1;steps['updated_transitions']+=metrics['unique_transitions_updated']
                        steps['actor_optimizer_steps']+=metrics['actor_optimizer_steps'];steps['critic_optimizer_steps']+=metrics['critic_optimizer_steps']
                        append(root/'rollouts.jsonl',metrics);buffer.clear()
                        print(f'ROLLOUT {steps["updates"]} FINAL_TAIL transitions={metrics["transitions"]} run=50',flush=True)
                    c=next_info['reward_components'];target_ore=adapter.config.get('target_ore_tons_shift');target_waste=adapter.config.get('target_waste_tons_shift')
                    upper=adapter.config.get('waste_task_upper_tons');eps=adapter.config.get('tonnage_epsilon')
                    ore=float(c['ore_tons_shift']);waste=float(c['waste_tons_shift'])
                    compliant=not adapter.grade_failed_so_far
                    current.update(ore_tons=ore,waste_tons=waste,ore_gap_tons=max(0.,target_ore-ore),waste_gap_tons=max(0.,target_waste-waste),
                        wait_vehicle_seconds=float(c['wait_vehicle_seconds']),grade_failed=adapter.grade_failed_so_far,
                        all_windows_compliant=compliant,joint_business_success=bool(compliant and ore>=target_ore-eps and target_waste-eps<=waste<=upper+eps),
                        reward_total=math.fsum(x['reward'] for x in episode_signals),windows=adapter.windows,
                        runtime_seconds=time.monotonic()-received,model_version_end=steps['updates'],terminal_ack_returned=True,status='COMPLETE')
                    # Training baseline uses only past training episodes; keep
                    # scalar times/returns in memory, never save observation snapshots.
                    rg=gg=0.
                    for item in reversed(episode_signals):
                        rg=item['reward']+cfg['gamma']*rg;gg=item['grade']+cfg['gamma']*gg
                        item.update(return_reward=rg,return_grade=gg)
                    (training_samples if job['split']=='train' else validation_samples).extend(episode_signals)
                    records.append(current);append(root/'episodes.jsonl',current);current=None
                    if job['run_index']==50:
                        require(not buffer and steps['training_transitions']==steps['updated_transitions'],'training transition coverage mismatch')
                        require(digest(actor)!=initial_actor,'Actor weights did not change during training')
                        final_checkpoint=save_checkpoint(root,actor,critics,actor_opt,critic_opt,steps['updates'],cfg,shuffle,final=True)
                        write(root/'training_finish.json',dict(status='TRAINING_COMPLETE',completed_episodes=50,
                            steps=steps,checkpoint=final_checkpoint,actor_digest_initial=initial_actor,actor_digest_final=digest(actor),
                            reward_critic_and_grade_critic_digest_initial=initial_critics,critics_digest_final=digest(critics)))
                        print('TRAINING_COMPLETE: final checkpoint frozen; next 5 episodes are validation',flush=True)
                    write(root/'status.json',dict(status='RUNNING',completed_training_episodes=sum(r['split']=='train' for r in records),
                        completed_validation_episodes=sum(r['split']=='validation' for r in records),updates=steps['updates']))
                    print(f'EPISODE {job["run_index"]}/55 split={job["split"]} steps={current if current else records[-1]["transitions"]}',flush=True)
                    break
                env._pending=ticket;obs,info=next_obs,next_info
        steps['validation_actor_after']=digest(actor);steps['validation_critics_after']=digest(critics)
        require(steps['validation_actor_before']==steps['validation_actor_after'] and
                steps['validation_critics_before']==steps['validation_critics_after'],'models changed during validation')
        require(steps['validation_actor_steps_before']==steps['actor_optimizer_steps'] and
                steps['validation_critic_steps_before']==steps['critic_optimizer_steps'],'optimizer used during validation')
        require(sha(CORE)==core_before,'protected PPO core changed');verify_plan(root)
        summary=validation_report(root,records,validation_samples,training_samples,cfg,initial_actor,digest(actor),steps)
        write(root/'status.json',dict(status=summary['status'],completed_training_episodes=50,completed_validation_episodes=5,updates=steps['updates']))
        print('TRAINING_AND_VALIDATION_COMPLETE; report:',root/'report.md',flush=True)
    except BaseException as exc:
        error=exc
        write(root/'ABORTED.json',dict(status='TECHNICAL_FAILURE',error=f'{type(exc).__name__}: {exc}',
            completed_training_episodes=sum(r['split']=='train' for r in records),
            completed_validation_episodes=sum(r['split']=='validation' for r in records),
            current_episode=current,steps=steps,buffered_transitions=len(buffer),
            protected_core_sha256_before=core_before,protected_core_sha256_after=sha(CORE)))
        write(root/'status.json',dict(status='TECHNICAL_FAILURE',error=str(exc),
            completed_training_episodes=sum(r['split']=='train' for r in records),
            completed_validation_episodes=sum(r['split']=='validation' for r in records),updates=steps['updates']))
    finally:
        if env: env.close()
        if server: server.stop(2).wait(3)
    if error: raise error
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','run'])
    parser.add_argument('--root',type=Path,default=DEFAULT)
    parser.add_argument('--port',type=int,default=50051)
    parser.add_argument('--timeout-seconds',type=float,default=36000)
    args=parser.parse_args(argv)
    if args.command=='prepare': prepare(args.root)
    else: run(args.root,args.timeout_seconds,args.port)


if __name__=='__main__': main()
