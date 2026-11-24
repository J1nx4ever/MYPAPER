"""Independent Actor and scalar combined-cost Critic for the 003 encoders."""
import copy
import hashlib
import json
import torch
from torch import nn
from torch.distributions import Categorical
from .contract import group, ablation_contract, require, parameter_digest
from .state_encoder import StateEncoder, finite
from .initialization import initialize
from .c_features import FEATURE_CONFIG, feature_contract, normalize_feature_config
from .features_cache import validate_sequence, pack_summary_prefixes


def _canonical(value):
    return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()


def _stamp(value):
    return torch.tensor(list(hashlib.sha256(_canonical(value)).digest()),dtype=torch.uint8)


def digest(module):
    h = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def feature_config_sha256(config=None):
    return feature_contract(normalize_feature_config(config))['config_sha256']


def _states(states):
    require(torch.is_tensor(states) and states.dtype==torch.float32 and states.device.type=='cpu' and
            states.ndim==2 and states.shape[1]==1519 and len(states)>0 and
            bool(torch.isfinite(states).all()),'finite CPU float32 states[N,1519] required')


def _common(module, spec, candidate, feature_config):
    module.candidate_id = candidate
    module.spec = copy.deepcopy(spec)
    module.feature_config = normalize_feature_config(feature_config)
    module.state_encoder = StateEncoder(mode=group(candidate))
    module.summary_encoder = nn.Sequential(nn.Linear(128,32),nn.ReLU())
    module.register_buffer('_feature_contract',_stamp(feature_contract(module.feature_config)))
    module.register_buffer('_architecture_contract',_stamp(dict(graph=ablation_contract(candidate),spec=spec)))


class Actor(nn.Module):
    def __init__(self,spec,*,candidate_id='s01_G0',feature_config=None,pair_seed=1):
        super().__init__()
        require(spec==dict(hidden_sizes=[128,128],activation='relu',initialization='named-paired-v1',
                          summary_embedding=32,fusion_hidden=128),'frozen 003 Actor specification required')
        with torch.random.fork_rng(devices=[]):
            _common(self,spec,candidate_id,feature_config)
            self.base_head = nn.Sequential(nn.Linear(160,128),nn.ReLU(),nn.Linear(128,4))
            initialize(self,pair_seed=pair_seed,role='actor')

    def forward(self,states,prefixes=None):
        _states(states)
        mask = states[:,19:23]
        require(bool((states[:,17]==0).all()) and bool(((mask==0)|(mask==1)).all()) and
                bool(mask.bool().any(dim=1).all()),'invalid action mask or terminal Actor request')
        current = validate_sequence(prefixes,states,self.feature_config)
        logits = finite(self.base_head(torch.cat((self.state_encoder(states),
                              self.summary_encoder(current)),dim=-1)),'Actor logits')
        return logits.masked_fill(~mask.bool(),-torch.inf)

    def distribution(self,states,prefixes=None):
        return Categorical(logits=self(states,prefixes))


class MaterialCostCritic(nn.Module):
    def __init__(self,spec,*,candidate_id='s01_G0',feature_config=None,pair_seed=1):
        super().__init__()
        require(spec==dict(state_hidden_sizes=[128,128],summary_embedding=32,gru_hidden=64,fusion_hidden=128),
                'frozen 003 scalar Critic specification required')
        with torch.random.fork_rng(devices=[]):
            _common(self,spec,candidate_id,feature_config)
            self.fusion = nn.Sequential(nn.Linear(224,128),nn.ReLU(),nn.Linear(128,1))
            initialize(self,pair_seed=pair_seed,role='critic')

    def forward(self,states,prefixes=None):
        _states(states)
        require(bool(((states[:,17]==0)|(states[:,17]==1)).all()),'invalid terminal flag')
        # Terminal rows are validated but never sent through learned encoders.
        from .adapter import validate_states
        validate_states(states)
        decisions = states[:,17]==0
        result = states.new_zeros(len(states))
        if bool(decisions.any()):
            current = validate_sequence(prefixes,states[decisions],self.feature_config)
            raw = self.state_encoder(states[decisions])
            summary = self.summary_encoder(current)
            values = finite(self.fusion(torch.cat((raw,summary,states.new_zeros((len(raw),64))),dim=1)),
                            'combined-cost Critic').squeeze(-1)
            result = result.index_copy(0,torch.where(decisions)[0],values)
        return result


CommitmentCostCritic = MaterialCostCritic


def actor_from_config(config):
    require(config['graph_contract']==ablation_contract(config['candidate_id']),'Actor graph contract differs')
    return Actor(config['network'],candidate_id=config['candidate_id'],feature_config=config['feature_config'],
                 pair_seed=config['seed'])


def common_actor_digest(actor):
    h = hashlib.sha256()
    for name, value in actor.named_parameters():
        if not name.startswith('state_encoder.graph_layers.'):
            h.update(name.encode())
            h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()
