"""Explicit settings and evidence gates. Templates deliberately cannot train."""
import copy
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SECTIONS = {
    'network': ('hidden_sizes','activation','initialization','risk_output'),
    'reward': ('kappa','ore_weight','waste_weight','grade_lower','grade_upper','grade_epsilon',
               'tonnage_epsilon','mass_balance_epsilon','wait_definition','cutoff_rule','simultaneous_events'),
    'ppo': ('gamma_reward','gamma_grade','gae_reward','gae_grade','epochs','minibatch_size','clip_range',
            'actor_lr','critic_lr','adam_eps','entropy_coef','reward_value_coef','grade_value_coef',
            'max_grad_norm','reward_loss','grade_loss','huber_delta','grade_loss_aggregation',
            'actor_aggregation','joint_advantage_processing'),
    'multipliers': ('initial','limits','rule','learning_rate','maximum'),
    'collection': ('episodes_per_batch','batches','seed'),
    'pretrain': ('epochs','minibatch_size','learning_rate','adam_eps','reward_loss','grade_loss',
                 'huber_delta','grade_loss_aggregation','reward_value_coef','grade_value_coef','max_grad_norm','checkpoint_selection'),
}
ACCEPTANCE_CHECKS = ('reward_ledger','waiting_tail','production_once','windows_material','empty_window',
                     'second_window_masks','multiple_windows','terminal_window','fifo_conservation',
                     'observation_runtime','mc_labels','abnormal_boundaries','diagnostic_lifecycle','event_cutoff_order')


class ConfigurationError(ValueError):
    pass


def template():
    return dict(contract='mining-lagrangian-ppo-v1', approved=False,
                **{k:{name:None for name in names} for k,names in SECTIONS.items()}, acceptance=None)


def validate_config(cfg, *, mode):
    errors=[]
    if not isinstance(cfg,dict):raise ConfigurationError('configuration must be an object')
    if mode not in ('online','offline'):
        errors.append('mode must be online or offline')
    if cfg.get('contract') != 'mining-lagrangian-ppo-v1': errors.append('contract')
    if cfg.get('approved') is not True: errors.append('approved: explicit design approval required')
    groups=('network','reward','ppo','multipliers','collection') if mode=='online' else ('network','reward','pretrain','collection')
    for group in groups:
        if not isinstance(cfg.get(group),dict):
            errors.append('missing/invalid section '+group)
            continue
        for key in (('seed',) if group=='collection' and mode=='offline' else SECTIONS[group]):
            if cfg.get(group,{}).get(key) is None: errors.append('missing '+group+'.'+key)
    if errors:
        raise ConfigurationError('\n'.join(errors))
    n,p,m,r,c = (cfg.get(k,{}) for k in ('network','ppo','multipliers','reward','collection'))
    def check(ok,msg):
        if not ok: errors.append(msg)
    def numeric(v):return type(v) in (int,float) and math.isfinite(v)
    check(isinstance(n['hidden_sizes'],list) and bool(n['hidden_sizes']) and
          all(type(v) is int and v>0 for v in n['hidden_sizes']), 'network.hidden_sizes')
    check(n['activation'] in ('tanh','relu'), 'network.activation')
    check(n['initialization'] in ('torch_default','orthogonal_sqrt2_hidden_unit_output'), 'network.initialization')
    check(n['risk_output'] in ('linear','sigmoid'), 'network.risk_output')
    for key in ('kappa','ore_weight','waste_weight','grade_lower','grade_upper','grade_epsilon','tonnage_epsilon','mass_balance_epsilon'):
        check(numeric(r[key]) and r[key]>=0, 'reward.'+key)
    if all(numeric(r[k]) for k in ('grade_lower','grade_upper','kappa')):
        check(0<=r['grade_lower']<=r['grade_upper']<=1 and r['kappa']>0,'reward bounds/kappa')
    check(r['wait_definition']=='load_dump_resource_including_crusher_capacity', 'unconfirmed wait_definition')
    check(r['cutoff_rule']=='exclude_loading_unloading_at_cutoff_integrate_crushing_to_cutoff', 'unconfirmed cutoff_rule')
    check(r['simultaneous_events']=='LIFO', 'event rule differs from current ALP')
    check(type(c['seed']) is int and c['seed']>=0, 'collection.seed')
    for key in ('episodes_per_batch','batches'):
        if mode=='online':check(type(c[key]) is int and c[key]>0,'collection.'+key)
    loss = p if mode=='online' else cfg['pretrain']
    for key in ('epochs','minibatch_size'):
        check(type(loss[key]) is int and loss[key]>0, key+' must be a positive integer')
    for key in ('adam_eps','huber_delta','max_grad_norm'):
        check(numeric(loss[key]) and loss[key]>0,'invalid '+key)
    for key in ('reward_value_coef','grade_value_coef'):
        check(numeric(loss[key]) and loss[key]>=0,'invalid '+key)
    check(loss['reward_loss'] in ('mse','huber'), 'reward_loss')
    check(loss['grade_loss'] in ('mse','huber','bce'), 'grade_loss')
    check(loss['grade_loss_aggregation'] in ('sum_head_means','mean_head_means'), 'grade_loss_aggregation')
    if loss['grade_loss']=='bce':check(n['risk_output']=='sigmoid','BCE requires explicit sigmoid risk output')
    if mode=='online':
        for k in ('gamma_reward','gamma_grade'):check(numeric(p[k]) and p[k]==1,'frozen '+k+' must be 1')
        for k in ('gae_reward','gae_grade'):check(numeric(p[k]) and 0<p[k]<=1,'invalid '+k)
        for k in ('actor_lr','critic_lr','clip_range'):check(numeric(p[k]) and p[k]>0,'invalid '+k)
        check(numeric(p['entropy_coef']) and p['entropy_coef']>=0,'entropy_coef')
        check(p['actor_aggregation'] in ('episode_sum','transition_mean'),'actor_aggregation')
        check(p['joint_advantage_processing'] in ('none','center','standardize'),'joint_advantage_processing')
        for k in ('initial','limits'):
            check(isinstance(m[k],list) and len(m[k])==4 and all(numeric(x) and x>=0 for x in m[k]),'multipliers.'+k)
        if isinstance(m['limits'],list):
            check(all(numeric(x) and 0<=x<=1 for x in m['limits']),'limits must be probabilities')
        check(m['rule']=='projected_additive_after_ppo','unsupported/unconfirmed multiplier rule')
        for k in ('learning_rate','maximum'):check(numeric(m[k]) and m[k]>=0,'multipliers.'+k)
        if numeric(m['maximum']) and isinstance(m['initial'],list) and all(numeric(x) for x in m['initial']):
            check(all(x<=m['maximum'] for x in m['initial']),'initial multipliers exceed maximum')
    else:
        check(numeric(loss['learning_rate']) and loss['learning_rate']>0,'pretrain.learning_rate')
        check(loss['checkpoint_selection'] in ('last_epoch','best_validation_objective'),'pretrain.checkpoint_selection')
    if errors:raise ConfigurationError('\n'.join(errors))
    return copy.deepcopy(cfg)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_pins():
    paths=[ROOT/'anylogic/DRL-PPO.alp',ROOT/'proto-src/src/main/proto/dispatch.proto',ROOT/'dist/protocol_manifest.json',
           *sorted((ROOT/'python').glob('*.py')),ROOT/'python/observation_schema.json']
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}


def verify_acceptance(cfg):
    spec=cfg.get('acceptance')
    if not isinstance(spec,dict) or not all(spec.get(k) for k in ('path','sha256')):
        raise ConfigurationError('missing acceptance.path / acceptance.sha256; actual AnyLogic acceptance is required')
    path=Path(spec['path'])
    if sha(path)!=spec['sha256']:raise ConfigurationError('acceptance file hash mismatch')
    a=json.loads(path.read_text())
    if a.get('scope')!='actual_anylogic' or a.get('status')!='passed':
        raise ConfigurationError('synthetic/unit-test evidence cannot authorize formal training')
    if set(a.get('passed_checks',[]))!=set(ACCEPTANCE_CHECKS):raise ConfigurationError('incomplete acceptance checks')
    if a.get('source_sha256')!=source_pins():raise ConfigurationError('acceptance source revision differs')
    if a.get('reward_settings')!=cfg['reward']:raise ConfigurationError('acceptance reward settings differ')
    if not a.get('evidence'):raise ConfigurationError('missing actual event/RPC evidence')
    for evidence in a['evidence']:
        if sha(evidence['path'])!=evidence['sha256']:raise ConfigurationError('acceptance evidence hash mismatch')
    return a


def preflight(cfg, *, mode):
    errors=[]
    try:result=validate_config(cfg,mode=mode)
    except (ValueError,TypeError,KeyError) as exc:errors.append(str(exc))
    from .protocol_artifacts import verify_protocol_artifacts
    try:verify_protocol_artifacts()
    except (ValueError,OSError,KeyError) as exc:errors.append('protocol: '+str(exc))
    if isinstance(cfg,dict):
        try:
            if mode=='offline':
                from .collect_offline import verify_environment
                verify_environment(cfg.get('acceptance'),cfg.get('reward'))
            else:
                verify_acceptance(cfg)
        except (ValueError,RuntimeError,OSError,KeyError,TypeError) as exc:errors.append('acceptance: '+str(exc))
    if errors:raise ConfigurationError('\n'.join(errors))
    return result
