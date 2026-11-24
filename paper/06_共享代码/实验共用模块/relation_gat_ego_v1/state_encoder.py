"""Complete 003 S path; graph input/output boundaries remain inspectable."""
import torch
from torch import nn
from .contract import require, GROUPS, S_PARAMETERS
from .token_encoder import TokenEncoder
from .relation_edges import GraphEdgeCache
from .graph_layers import GraphLayer


def finite(value, label):
    require(bool(torch.isfinite(value).all()), 'nonfinite ' + label)
    return value


def checked_norm(module, value, label):
    centered=finite(value-finite(value.mean(dim=-1,keepdim=True),label+' mean'),label+' centered')
    finite(centered.square().mean(dim=-1),label+' variance')
    return finite(module(value),label)


class EgoReadout(nn.Module):
    def __init__(self, *, uniform=False):
        super().__init__()
        self.uniform = bool(uniform)
        if not self.uniform:
            self.q = nn.Linear(64,64)
            self.k = nn.Linear(64,64)
        self.v = nn.Linear(64,64)
        self.o = nn.Linear(64,64)
        self.ln = nn.LayerNorm(64,eps=1e-5)

    def forward(self, scene, ego, *, diagnostic=False):
        require(not torch.is_autocast_enabled('cpu'),'Ego CPU autocast is outside the float32 contract')
        require(scene.ndim == 3 and scene.shape[1:] == (32,64) and
                scene.dtype == torch.float32 and scene.device.type == 'cpu', 'scene must be CPU float32[B,32,64]')
        require(ego.dtype == torch.int64 and ego.shape == (len(scene),) and
                bool(((ego >= 0) & (ego < 32)).all()), 'invalid Ego row index')
        finite(scene, 'Ego scene')
        residual = scene[torch.arange(len(scene)),ego]
        v = finite(self.v(scene), 'Ego V').reshape(-1,32,4,16).transpose(1,2)
        if self.uniform:
            weights = scene.new_full((len(scene),4,32),1/32)
            q = k = scores = None
        else:
            q = finite(self.q(residual), 'Ego Q').reshape(-1,4,16)
            k = finite(self.k(scene), 'Ego K').reshape(-1,32,4,16).transpose(1,2)
            scores = finite(torch.einsum('bhd,bhnd->bhn',q,k)/4, 'Ego score')
            weights = finite(torch.softmax(scores,dim=-1), 'Ego softmax')
        context = finite(torch.einsum('bhn,bhnd->bhd',weights,v).reshape(-1,64), 'Ego context')
        projected = finite(self.o(context), 'Ego O')
        result = checked_norm(self.ln,finite(residual+projected,'Ego residual'), 'Ego LN')
        return (result,dict(q=q,k=k,v=v,scores=scores,weights=weights,residual=residual)) if diagnostic else result


class StateEncoder(nn.Module):
    def __init__(self, mode='G2'):
        super().__init__()
        require(mode in GROUPS, 'unknown S representation')
        self.mode = mode
        self.token_encoder = TokenEncoder()
        self.graph_layers = nn.ModuleList([] if mode=='G0' else [GraphLayer(mode=mode) for _ in range(2)])
        if mode != 'G0':
            # Numerical current-snapshot topology only. Each S owns a bounded
            # cache; Token, H1/H2 and every learned readout are recomputed.
            self.edge_cache = GraphEdgeCache(max_entries=4096, max_bytes=64*1024*1024)
        self.ego = EgoReadout()
        self.action = nn.Linear(66,32)
        self.readout = nn.Linear(192,128)
        require(sum(p.numel() for p in self.parameters()) == S_PARAMETERS[mode], 'S parameter inventory differs')

    def scene(self, states, *, diagnostic=False):
        z0 = self.token_encoder(states)
        details = dict(z0=z0)
        if self.mode == 'G0':
            z2 = z0
        else:
            edges = self.edge_cache.get(states)
            h = z0[:,:31,:]
            details['edges'] = edges
            for number, layer in enumerate(self.graph_layers,1):
                h = layer(h,edges)
                details['h'+str(number)] = h
            z2 = torch.cat((h,z0[:,31:32,:]),dim=1)
        details['z2'] = z2
        return (z2,details) if diagnostic else z2

    def forward(self, states, *, diagnostic=False):
        require(states.ndim == 2 and states.shape[1] == 1519 and len(states)>0 and
                bool((states[:,17] == 0).all()), 'normal S requires decision rows; terminal must be split first')
        z2, details = self.scene(states,diagnostic=True)
        ego_flags = states[:,27:1107].reshape(-1,9,120)[:,:,0]
        ego = ego_flags.argmax(dim=1)
        h_ego, ego_details = self.ego(z2,ego,diagnostic=True)
        action_input = torch.cat((z2[:,9:13,:],states[:,19:23,None],states[:,23:27,None]),dim=-1)
        actions = torch.relu(finite(self.action(action_input),'action projection'))
        result = torch.relu(finite(self.readout(torch.cat((h_ego,actions.flatten(1)),dim=1)),'S readout'))
        if diagnostic:
            details.update(ego=ego_details,h_ego=h_ego,action_input=action_input,actions=actions)
            return result,details
        return result

    def terminal_diagnostic(self, states):
        require(bool((states[:,17]==1).all()), 'explicit diagnostic requires true terminal rows')
        scene = self.scene(states)
        return dict(terminal_scene64=checked_norm(self.ego.ln,finite(scene.mean(dim=1),'terminal mean'), 'terminal diagnostic'),
                    ego_attention_applied=False, graph_message_passing_applied=self.mode!='G0')
