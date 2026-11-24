"""1519D Actor + scalar Reward Critic + one four-head Grade Critic."""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from .constrained_adapter import require
from .experiment_config import validate_config

CHECKPOINT_FORMAT='mining-two-critic-four-window-v1'
COST_NAMES=tuple('window_'+str(j+1) for j in range(4))


def digest(module):
    h=hashlib.sha256()
    for key,value in sorted(module.state_dict().items()):
        h.update(key.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def network(spec, output):
    activation={'tanh':nn.Tanh,'relu':nn.ReLU}[spec['activation']]
    layers=[];last=1519
    for width in spec['hidden_sizes']:
        layers.extend((nn.Linear(last,width),activation()));last=width
    layers.append(nn.Linear(last,output))
    result=nn.Sequential(*layers)
    if spec['initialization']=='orthogonal_sqrt2_hidden_unit_output':
        for m in result:
            if isinstance(m,nn.Linear):nn.init.orthogonal_(m.weight,gain=np.sqrt(2));nn.init.zeros_(m.bias)
        nn.init.orthogonal_(result[-1].weight,gain=1.)
    elif spec['initialization']!='torch_default':raise ValueError('explicit initialization required')
    return result


def input_check(x):
    require(x.ndim==2 and x.shape[1]==1519 and bool(torch.isfinite(x).all()),'network requires finite [batch,1519]')


class Actor(nn.Module):
    def __init__(self,spec):
        super().__init__();self.mlp=network(spec,4)

    def forward(self,x):
        input_check(x)
        mask=x[:,19:23]
        require(bool(((mask==0)|(mask==1)).all()) and bool(mask.bool().any(dim=1).all()),'invalid action mask')
        require(bool((x[:,17]==0).all()),'terminal observations cannot request actions')
        return self.mlp(x).masked_fill(~mask.bool(),-torch.inf)

    def distribution(self,x):return Categorical(logits=self(x))


class RewardCritic(nn.Module):
    def __init__(self,spec):super().__init__();self.mlp=network(spec,1)
    def forward(self,x):input_check(x);return self.mlp(x).squeeze(-1)


class GradeCostCritic(nn.Module):
    def __init__(self,spec):
        super().__init__();self.mlp=network(spec,4);self.output=spec['risk_output']
        require(self.output in ('linear','sigmoid'),'explicit risk output required')
    def forward(self,x):
        input_check(x);value=self.mlp(x)
        return value.sigmoid() if self.output=='sigmoid' else value


class TwoCritics(nn.Module):
    def __init__(self,spec):
        super().__init__();self.reward=RewardCritic(spec);self.grade=GradeCostCritic(spec)
    def forward(self,x):return self.reward(x),self.grade(x)


def build_models(spec,seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed);return Actor(spec),TwoCritics(spec)


def gae(values,next_values,signals,bootstrap,trace,valid,episode_bounds,lam):
    """Gamma is frozen at one. Bootstrap and trace stop independently per head."""
    v,n,s,b,c,m=[np.asarray(x) for x in (values,next_values,signals,bootstrap,trace,valid)]
    require(v.shape==n.shape==s.shape==b.shape==c.shape==m.shape,'GAE shape mismatch')
    adv=np.zeros_like(v,dtype=np.float64)
    for start,stop in episode_bounds:
        carry=np.zeros(v.shape[1:] or (),dtype=np.float64)
        for t in range(stop-1,start-1,-1):
            delta=s[t]+b[t]*n[t]-v[t]
            carry=np.where(m[t],delta+lam*c[t]*carry,0.)
            adv[t]=carry
    return adv,np.where(m,adv+v,0.)


def joint_advantage(ar,ag,valid,lambdas,processing):
    a=np.asarray(ar)-np.sum(np.asarray(valid)*np.asarray(ag)*np.asarray(lambdas),axis=1)
    if processing in ('center','standardize'):a=a-a.mean()
    if processing=='standardize':a=a/(a.std()+1e-8)
    require(processing in ('none','center','standardize'),'explicit joint advantage processing required')
    return a


def element_loss(predicted,target,kind,delta):
    if kind=='mse':return (predicted-target).square()
    if kind=='huber':return nn.functional.huber_loss(predicted,target,reduction='none',delta=delta)
    if kind=='bce':return nn.functional.binary_cross_entropy(predicted,target,reduction='none')
    raise ValueError('explicit supported loss required')


def critic_loss(critics,obs,yr,yg,valid,cfg):
    vr,vg=critics(obs)
    reward=element_loss(vr,yr,cfg['reward_loss'],cfg['huber_delta']).mean()
    # Select only valid elements before evaluating a loss (invalid labels may be NaN).
    heads=[];counts=[]
    for j in range(4):
        chosen=valid[:,j].bool();counts.append(int(chosen.sum()))
        heads.append(element_loss(vg[chosen,j],yg[chosen,j],cfg['grade_loss'],cfg['huber_delta']).mean()
                     if chosen.any() else vg[:,j].sum()*0.)
    grade=torch.stack(heads).sum()
    if cfg['grade_loss_aggregation']=='mean_head_means':grade=grade/4
    elif cfg['grade_loss_aggregation']!='sum_head_means':raise ValueError('explicit head loss aggregation required')
    total=cfg['reward_value_coef']*reward+cfg['grade_value_coef']*grade
    return total,dict(reward=float(reward.detach()),grade_heads=[float(h.detach()) for h in heads],valid_counts=counts)


class ConstrainedPPO:
    def __init__(self,cfg):
        self.config=validate_config(cfg,mode='online')
        self.actor,self.critics=build_models(cfg['network'],cfg['collection']['seed'])
        p=cfg['ppo'];self.lambdas=np.asarray(cfg['multipliers']['initial'],dtype=np.float64)
        self.actor_optimizer=torch.optim.Adam(self.actor.parameters(),lr=p['actor_lr'],eps=p['adam_eps'])
        self.critic_optimizer=torch.optim.Adam(self.critics.parameters(),lr=p['critic_lr'],eps=p['adam_eps'])
        self.generator=torch.Generator().manual_seed(cfg['collection']['seed']);self.update_index=0
        self._pending_multiplier=None

    def compute_targets(self,batch):
        from .constrained_batch import validate_rows
        validate_rows(batch.rows,batch.episodes)
        x=torch.as_tensor(batch.observations,dtype=torch.float32)
        nx=torch.as_tensor(batch.next_observations,dtype=torch.float32)
        with torch.no_grad():vr,vg=self.critics(x);nr,ng=self.critics(nx)
        p=self.config['ppo']
        ar,yr=gae(vr.numpy(),nr.numpy(),batch.rewards,batch.reward_bootstrap,batch.reward_trace,
                  np.ones(len(x),dtype=bool),batch.bounds,p['gae_reward'])
        ag,yg=gae(vg.numpy(),ng.numpy(),batch.grade_costs,batch.grade_bootstrap,batch.grade_trace,
                  batch.grade_valid,batch.bounds,p['gae_grade'])
        joint=joint_advantage(ar,ag,batch.grade_valid,self.lambdas,p['joint_advantage_processing'])
        return dict(reward_advantage=ar,grade_advantage=ag,reward_target=yr,grade_target=yg,joint=joint,
                    old_reward_values=vr.numpy().copy(),old_grade_values=vg.numpy().copy())

    def update(self,batch):
        require(self._pending_multiplier is None,'previous batch multiplier update is pending')
        require(batch.policy_version==digest(self.actor),'batch policy version differs from current actor')
        targets=self.compute_targets(batch) # evaluated exactly once, outside all epochs
        self.last_targets={k:np.array(v,copy=True) for k,v in targets.items()}
        for value in self.last_targets.values():value.setflags(write=False)
        p=self.config['ppo'];x=torch.tensor(batch.observations,dtype=torch.float32)
        actions=torch.tensor(batch.actions);old=torch.tensor(batch.old_log_probs,dtype=torch.float32)
        joint=torch.tensor(targets['joint'],dtype=torch.float32)
        yr=torch.tensor(targets['reward_target'],dtype=torch.float32);yg=torch.tensor(targets['grade_target'],dtype=torch.float32)
        valid=torch.tensor(batch.grade_valid,dtype=torch.bool);coefficients=self.lambdas.copy();metrics=[]
        # Epoch-global per-head denominators: no dilution by settled states or longer late-window traces.
        n=len(x);counts=valid.sum(dim=0).clamp_min(1)
        for _ in range(p['epochs']):
            for ids in torch.randperm(n,generator=self.generator).split(p['minibatch_size']):
                dist=self.actor.distribution(x[ids]);logp=dist.log_prob(actions[ids]);ratio=(logp-old[ids]).exp()
                terms=-torch.minimum(ratio*joint[ids],ratio.clamp(1-p['clip_range'],1+p['clip_range'])*joint[ids])
                terms=terms-p['entropy_coef']*dist.entropy()
                actor_loss=terms.sum()/len(batch.episodes) if p['actor_aggregation']=='episode_sum' else terms.mean()
                self.actor_optimizer.zero_grad();actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(),p['max_grad_norm']);self.actor_optimizer.step()
                vr,vg=self.critics(x[ids])
                rloss=element_loss(vr,yr[ids],p['reward_loss'],p['huber_delta']).sum()/n
                losses=[]
                for j in range(4):
                    m=valid[ids,j]
                    losses.append(element_loss(vg[m,j],yg[ids[m],j],p['grade_loss'],p['huber_delta']).sum()/counts[j]
                                  if m.any() else vg[:,j].sum()*0.)
                gloss=torch.stack(losses).sum()/(4 if p['grade_loss_aggregation']=='mean_head_means' else 1)
                loss=p['reward_value_coef']*rloss+p['grade_value_coef']*gloss
                self.critic_optimizer.zero_grad();loss.backward()
                nn.utils.clip_grad_norm_(self.critics.parameters(),p['max_grad_norm']);self.critic_optimizer.step()
                # Observation only: preserve every loss, backward and optimizer operation.
                metrics.append(dict(actor_loss=float(actor_loss.detach()),critic_loss=float(loss.detach()),
                    reward_critic_loss=float(rloss.detach()),grade_critic_loss=float(gloss.detach()),
                    **{'grade_head_'+str(j+1)+'_loss':float(value.detach()) for j,value in enumerate(losses)}))
        require(np.array_equal(coefficients,self.lambdas),'multipliers changed during PPO')
        require(all(bool(torch.isfinite(t).all()) for t in list(self.actor.parameters())+list(self.critics.parameters())),
                'nonfinite trained parameters')
        self.update_index+=1;self._pending_multiplier=batch.batch_id
        return dict(update_index=self.update_index,policy_version=batch.policy_version,coefficients_used=coefficients.tolist(),
                    valid_counts=batch.grade_valid.sum(axis=0).tolist(),optimizer_steps=len(metrics),
                    mean={key:float(np.mean([m[key] for m in metrics])) for key in metrics[0]})

    def apply_multiplier_update(self,batch):
        require(self._pending_multiplier==batch.batch_id,'multipliers must update once, after PPO for this batch')
        # One independent result per shift/window. Never average repeated state labels.
        rates=np.array([e['window_costs'] for e in batch.episodes]).mean(axis=0)
        m=self.config['multipliers'];before=self.lambdas.copy()
        self.lambdas=np.clip(before+m['learning_rate']*(rates-np.asarray(m['limits'])),0,m['maximum'])
        self._pending_multiplier=None
        return dict(rates=rates.tolist(),before=before.tolist(),after=self.lambdas.tolist())

    def save_checkpoint(self,path,extra=None):
        require(self._pending_multiplier is None,'checkpoint cannot publish an unfinished update')
        path=Path(path);require(not path.exists(),'refusing checkpoint overwrite');path.parent.mkdir(parents=True,exist_ok=True)
        payload=dict(format=CHECKPOINT_FORMAT,config=self.config,actor=self.actor.state_dict(),critics=self.critics.state_dict(),
                     actor_optimizer=self.actor_optimizer.state_dict(),critic_optimizer=self.critic_optimizer.state_dict(),
                     multipliers=self.lambdas.tolist(),update_index=self.update_index,actor_digest=digest(self.actor),
                     rng_torch=torch.get_rng_state(),rng_numpy=np.random.get_state(),rng_shuffle=self.generator.get_state(),
                     rng_python=__import__("random").getstate(),
                     metadata=extra or {})
        torch.save(payload,path)
