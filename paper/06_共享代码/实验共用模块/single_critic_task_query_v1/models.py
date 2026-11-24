"""Independent task-query Actor and scalar cost Critic, with optional C128.

All four actor queries share their generator, attention and scalar scorer.
Relation information enters through two GATv2-style layers; cross-attention
always reads the complete 32-entity scene without an additional bias or mask.
"""
import copy
import hashlib
import json
import torch
from torch import nn
from torch.distributions import Categorical
from relation_gat_ego_v1.adapter import validate_states
from relation_gat_ego_v1.token_encoder import TokenEncoder
from relation_gat_ego_v1.relation_edges import GraphEdgeCache
from relation_gat_ego_v1.graph_layers import GraphLayer
from relation_gat_ego_v1.state_encoder import finite, checked_norm
from relation_gat_ego_v1.initialization import initialize
from .contract import ablation_contract, require, feature_config_for
from .c_features import feature_contract, normalize_feature_config
from .features_cache import validate_sequence, pack_summary_prefixes

ACTOR_SPEC = dict(query_input=130,query_hidden=128,query_width=64,attention_heads=4,
    state_width=128,action_embedding=32,scorer_hidden=128,summary_embedding=32,
    initialization='named-paired-v1')
CRITIC_SPEC = dict(query_input=128,query_hidden=128,query_width=64,attention_heads=4,
    state_width=128,action_embedding=32,fusion_hidden=128,summary_embedding=32,
    initialization='named-paired-v1')


def _stamp(value):
    raw = (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
    return torch.tensor(list(hashlib.sha256(raw).digest()),dtype=torch.uint8)


def digest(module):
    h = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def feature_config_sha256(config=None):
    return feature_contract(normalize_feature_config(config))['config_sha256']


def _states(states):
    require(torch.is_tensor(states) and states.dtype == torch.float32 and states.device.type == 'cpu'
        and states.ndim == 2 and states.shape[1] == 1519 and len(states) > 0
        and bool(torch.isfinite(states).all()), 'finite CPU float32 states[N,1519] required')
    require(not torch.is_autocast_enabled('cpu'), 'CPU autocast is outside the float32 contract')


class GlobalRetrieval(nn.Module):
    """Four heads of width 16, all entities visible; explicit ego residual."""
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(64,64)
        self.k = nn.Linear(64,64)
        self.v = nn.Linear(64,64)
        self.o = nn.Linear(64,64)
        self.ln = nn.LayerNorm(64,eps=1e-5)

    def forward(self, queries, scene, residual, *, diagnostic=False):
        require(queries.ndim == 3 and queries.shape[2] == 64 and
            scene.shape == (len(queries),32,64) and residual.shape == queries.shape,
            'retrieval query, scene or residual shape differs')
        batch, count, _ = queries.shape
        q = finite(self.q(queries),'retrieval Q').reshape(batch,count,4,16).transpose(1,2)
        k = finite(self.k(scene),'retrieval K').reshape(batch,32,4,16).transpose(1,2)
        v = finite(self.v(scene),'retrieval V').reshape(batch,32,4,16).transpose(1,2)
        scores = finite(q @ k.transpose(-1,-2) / 4.,'retrieval scores')
        weights = finite(scores.softmax(-1),'retrieval weights')
        context = finite((weights @ v).transpose(1,2).reshape(batch,count,64),'retrieval context')
        result = checked_norm(self.ln,finite(residual+self.o(context),'retrieval residual'), 'retrieval LN')
        details = dict(q=q,k=k,v=v,scores=scores,weights=weights,residual=residual,
                       context=context,query_input=queries)
        return (result,details) if diagnostic else result


class TaskStateEncoder(nn.Module):
    def __init__(self, *, role):
        super().__init__()
        require(role in ('actor','critic'), 'unknown task readout role')
        self.role = role
        self.mode = 'G2'
        self.token_encoder = TokenEncoder()
        self.graph_layers = nn.ModuleList([GraphLayer(mode='G2') for _ in range(2)])
        self.edge_cache = GraphEdgeCache(max_entries=4096,max_bytes=64*1024*1024)
        self.query_encoder = nn.Sequential(nn.Linear(130 if role == 'actor' else 128,128),
            nn.ReLU(),nn.Linear(128,64))
        if role == 'critic':
            # A learned task vector represented as a Linear weight so the frozen
            # named initializer classifies every parameter without modification.
            self.value_task = nn.Linear(1,64,bias=False)
        self.retrieval = GlobalRetrieval()
        self.action = nn.Linear(66,32)
        self.readout = nn.Linear(192,128)

    def scene(self, states, *, diagnostic=False):
        z0 = self.token_encoder(states)
        edges = self.edge_cache.get(states)
        h = z0[:,:31,:]
        details = dict(z0=z0,edges=edges)
        for number, layer in enumerate(self.graph_layers,1):
            h = layer(h,edges)
            details['h'+str(number)] = h
        z2 = torch.cat((h,z0[:,31:32,:]),dim=1)
        details['z2'] = z2
        return (z2,details) if diagnostic else z2

    def forward(self, states, *, diagnostic=False):
        require(bool((states[:,17] == 0).all()), 'task readout requires decision rows')
        z2, details = self.scene(states,diagnostic=True)
        ids = states[:,27:1107].reshape(-1,9,120)[:,:,0].argmax(dim=1)
        ego = z2[torch.arange(len(states)),ids]
        action_input = torch.cat((z2[:,9:13,:],states[:,19:23,None],states[:,23:27,None]),dim=-1)
        actions = torch.relu(finite(self.action(action_input),'action projection'))
        if self.role == 'actor':
            query_input = torch.cat((ego[:,None,:].expand(-1,4,-1),action_input),dim=-1)
            queries = finite(self.query_encoder(query_input),'action queries')
            context, attention = self.retrieval(queries,z2,ego[:,None,:].expand(-1,4,-1),diagnostic=True)
            all_actions = actions.flatten(1)[:,None,:].expand(-1,4,-1)
            raw = torch.cat((context,all_actions),dim=-1)
        else:
            query_input = torch.cat((z2[:,31,:],ego),dim=-1)
            queries = (finite(self.query_encoder(query_input),'value query') +
                       self.value_task(states.new_ones((len(states),1))))[:,None,:]
            context, attention = self.retrieval(queries,z2,ego[:,None,:],diagnostic=True)
            raw = torch.cat((context[:,0],actions.flatten(1)),dim=-1)
        result = torch.relu(finite(self.readout(raw),'task S readout'))
        if diagnostic:
            details.update(ego_ids=ids,ego_token=ego,query_input=query_input,queries=queries,
                attention=attention,context=context,action_input=action_input,actions=actions)
            return result,details
        return result


def _common(module, spec, candidate, feature_config, role):
    module.candidate_id = candidate
    module.spec = copy.deepcopy(spec)
    module.feature_config = normalize_feature_config(feature_config_for(candidate) if feature_config is None else feature_config)
    require(module.feature_config['enabled'] == ablation_contract(candidate)['summary_enabled'],
        'C128 enabled flag differs from candidate identity')
    module.state_encoder = TaskStateEncoder(role=role)
    if module.feature_config['enabled']:
        module.summary_encoder = nn.Sequential(nn.Linear(128,32),nn.ReLU())
    module.register_buffer('_feature_contract',_stamp(feature_contract(module.feature_config)))
    module.register_buffer('_architecture_contract',_stamp(dict(graph=ablation_contract(candidate),spec=spec,role=role)))


class Actor(nn.Module):
    def __init__(self,spec,*,candidate_id='s01_TQ0',feature_config=None,pair_seed=1):
        super().__init__()
        require(spec == ACTOR_SPEC, 'frozen task-query Actor specification required')
        with torch.random.fork_rng(devices=[]):
            _common(self,spec,candidate_id,feature_config,'actor')
            width = 160 + (32 if self.feature_config['enabled'] else 0)
            self.scorer = nn.Sequential(nn.Linear(width,128),nn.ReLU(),nn.Linear(128,1))
            initialize(self,pair_seed=pair_seed,role='actor')

    def forward(self,states,prefixes=None,*,diagnostic=False):
        _states(states)
        mask = states[:,19:23]
        require(bool((states[:,17] == 0).all()) and bool(((mask == 0)|(mask == 1)).all()) and
            bool(mask.bool().any(dim=1).all()), 'invalid action mask or terminal Actor request')
        current = validate_sequence(prefixes,states,self.feature_config)
        raw,details = self.state_encoder(states,diagnostic=True)
        pieces = [raw,details['actions']]
        if self.feature_config['enabled']:
            pieces.append(self.summary_encoder(current)[:,None,:].expand(-1,4,-1))
        logits = finite(self.scorer(torch.cat(pieces,dim=-1)).squeeze(-1),'Actor logits')
        result = logits.masked_fill(~mask.bool(),-torch.inf)
        if diagnostic:
            details.update(unmasked_logits=logits,masked_logits=result)
            return result,details
        return result

    def distribution(self,states,prefixes=None):
        return Categorical(logits=self(states,prefixes))


class MaterialCostCritic(nn.Module):
    def __init__(self,spec,*,candidate_id='s01_TQ0',feature_config=None,pair_seed=1):
        super().__init__()
        require(spec == CRITIC_SPEC,
            'frozen task-query scalar Critic specification required')
        with torch.random.fork_rng(devices=[]):
            _common(self,spec,candidate_id,feature_config,'critic')
            self.fusion = nn.Sequential(nn.Linear(128+(32 if self.feature_config['enabled'] else 0),128),
                nn.ReLU(),nn.Linear(128,1))
            initialize(self,pair_seed=pair_seed,role='critic')

    def forward(self,states,prefixes=None,*,diagnostic=False):
        _states(states)
        validate_states(states)
        decisions = states[:,17] == 0
        result = states.new_zeros(len(states))
        details = dict(terminal_bootstrap=0.)
        if bool(decisions.any()):
            current = validate_sequence(prefixes,states[decisions],self.feature_config)
            raw,details = self.state_encoder(states[decisions],diagnostic=True)
            pieces = [raw]
            if self.feature_config['enabled']:
                pieces.append(self.summary_encoder(current))
            values = finite(self.fusion(torch.cat(pieces,dim=1)),'combined-cost Critic').squeeze(-1)
            result = result.index_copy(0,torch.where(decisions)[0],values)
        return (result,details) if diagnostic else result


CommitmentCostCritic = MaterialCostCritic


def actor_from_config(config):
    require(config['graph_contract'] == ablation_contract(config['candidate_id']), 'Actor graph contract differs')
    return Actor(config['network'],candidate_id=config['candidate_id'],feature_config=config['feature_config'],
        pair_seed=config['seed'])


def common_actor_digest(actor):
    """Digest only exactly paired parameters; C fusion input shape differs."""
    h = hashlib.sha256()
    for name,value in actor.named_parameters():
        if name.startswith('summary_encoder.') or name.startswith('scorer.0.'):
            continue
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
